//! Backend-neutral GPU kernel tests for the `inkling2` kernel pack.
//!
//! Each test compares a device kernel launch through [`shiso_core::Backend`]
//! against a plain Rust reference oracle over the same inputs.

#![expect(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    reason = "kernel oracle tests compute synthetic reference fixtures over small shapes"
)]

use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs, alloc};

use crate::harness::{Gpu, drain, f32_bytes, run};

/// Module name in both Metal and CUDA registries.
pub const MODULE: &str = "inkling2";

/// Every entry point the `inkling2` module exports.
pub const ENTRIES: &[&str] = &[
    "kernel_inkling2_embed_rmsnorm_bf16",
    "kernel_inkling2_rmsnorm_bf16",
    "kernel_inkling2_linear_bf16",
    "kernel_inkling2_qkvr_gate_sink_bf16",
    "kernel_inkling2_lm_head_softcap",
    "kernel_inkling2_attn_res",
    "kernel_inkling2_loop_control",
    "kernel_inkling2_loop_ctrl_down",
    "kernel_inkling2_loop_ctrl_apply",
    "kernel_inkling2_short_conv_static",
    "kernel_inkling2_short_conv_dynamic",
    "kernel_inkling2_qkvr_prep_store",
    "kernel_inkling2_rel_bias_bf16",
    "kernel_inkling2_rel_attn",
    "kernel_inkling2_rel_attn_splitk",
    "kernel_inkling2_rel_attn_reduce",
    "kernel_inkling2_route_topk",
    "kernel_inkling2_moe_nvfp4_w13_swiglu",
    "kernel_inkling2_moe_nvfp4_w2_accum",
    "kernel_inkling2_moe_mxfp8_w13_swiglu",
    "kernel_inkling2_moe_mxfp8_w2_accum",
    "kernel_inkling2_moe_bucket",
    "kernel_inkling2_moe_nvfp4_w13_grouped",
    "kernel_inkling2_moe_nvfp4_w2_grouped",
    "kernel_inkling2_moe_mxfp8_w13_grouped",
    "kernel_inkling2_moe_mxfp8_w2_grouped",
    "kernel_inkling2_moe_reduce_topk",
    "kernel_inkling2_add2_f32",
    "kernel_inkling2_add3_f32",
    "kernel_inkling2_vit_layernorm_bf16",
    "kernel_inkling2_vit_linear_bias_bf16",
    "kernel_inkling2_vit_axial_rope",
    "kernel_inkling2_vit_attn",
    "kernel_inkling2_vit_gelu_erf",
    "kernel_inkling2_vit_layerscale_add",
    "kernel_inkling2_mtp_mix_rmsnorm",
    "kernel_inkling2_mtp_qk_norm_rope",
    "kernel_inkling2_mtp_block_attn",
    "kernel_inkling2_mtp_swiglu_mul",
    "kernel_inkling2_mtp_rnn_step",
];

fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn e4m3_to_f32(b: u8) -> f32 {
    let e = (b >> 3) & 0xF;
    let m = b & 7;
    let v = if e == 0 {
        f32::from(m) * 0.001_953_125
    } else if e == 15 && m == 7 {
        0.0
    } else {
        (1.0 + f32::from(m) * 0.125) * 2.0_f32.powi(i32::from(e) - 7)
    };
    if (b & 0x80) != 0 { -v } else { v }
}

fn ue8m0_to_f32(b: u8) -> f32 {
    if b == 255 {
        0.0
    } else {
        f32::from_bits(u32::from(b) << 23)
    }
}

const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softcap(x: f32, cap: f32) -> f32 {
    if cap <= 0.0 {
        x
    } else {
        cap * (x / cap).tanh()
    }
}

fn upload_bytes(gpu: &Gpu, bytes: &[u8]) -> DeviceBuffer {
    let buf = alloc(gpu.device(), bytes.len().max(4) as u64).expect("alloc");
    if !bytes.is_empty() {
        gpu.device().copy_h2d(buf.view(), bytes).expect("upload");
    }
    buf
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let mut max_ratio = 0.0_f32;
    let mut worst = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let allowed = tol + 1e-5 * w.abs();
        let ratio = (g - w).abs() / allowed;
        if ratio > max_ratio {
            max_ratio = ratio;
            worst = i;
        }
    }
    assert!(
        max_ratio <= 1.0,
        "{label}: err={} > allowed={} at index {worst} (got={}, want={})",
        (got[worst] - want[worst]).abs(),
        tol + 1e-5 * want[worst].abs(),
        got[worst],
        want[worst]
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NormArgs {
    rows: u32,
    hidden: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearArgs {
    rows: u32,
    in_dim: u32,
    out_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttnResArgs {
    rows: u32,
    hidden: u32,
    n_blocks: u32,
    n_queries: u32,
    eps: f32,
    has_addend: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LoopCtrlArgs {
    rows: u32,
    hidden: u32,
    rank: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SconvStaticArgs {
    rows: u32,
    channels: u32,
    kernel_size: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SconvDynamicArgs {
    rows: u32,
    kv_heads: u32,
    head_dim: u32,
    kernel_size: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QkvrPrepArgs {
    rows: u32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    d_rel: u32,
    qk_clip: f32,
    rel_clip: f32,
    eps: f32,
    is_full: u32,
    log_n_floor: f32,
    log_alpha: f32,
    pos0: u32,
    kv_cap: u32,
    is_swa: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RelAttnArgs {
    rows: u32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    rel_extent: u32,
    pos0: u32,
    kv_cap: u32,
    window_size: u32,
    is_swa: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RouteArgs {
    rows: u32,
    n_experts: u32,
    top_k: u32,
    route_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeW13Args {
    rows: u32,
    in_dim: u32,
    inter_dim: u32,
    top_k: u32,
    gate_cap: f32,
    up_cap: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MoeW2Args {
    rows: u32,
    inter_dim: u32,
    out_dim: u32,
    top_k: u32,
}

/// Verifies `kernel_inkling2_attn_res` against an exact Rust oracle.
pub fn attn_res_matches_oracle(gpu: &Gpu) {
    let rows = 2usize;
    let hidden = 64usize;
    let n_blocks = 2usize;
    let n_queries = 3usize;
    let eps = 1e-6_f32;

    let blocks: Vec<f32> = (0..n_blocks * rows * hidden)
        .map(|i| ((i as f32) * 0.13).sin())
        .collect();
    let partial: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.27).cos())
        .collect();
    let addend: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.07).sin() * 0.5)
        .collect();
    let queries: Vec<f32> = (0..n_queries * hidden)
        .map(|i| ((i as f32) * 0.19).cos() * 0.3)
        .collect();

    let mut want = vec![0.0_f32; n_queries * rows * hidden];
    let scale = 1.0 / (hidden as f32).sqrt();
    for r in 0..rows {
        let mut srcs = Vec::new();
        for b in 0..n_blocks {
            let off = b * rows * hidden + r * hidden;
            srcs.push(blocks[off..off + hidden].to_vec());
        }
        let mut last = vec![0.0_f32; hidden];
        for h in 0..hidden {
            last[h] = partial[r * hidden + h] + addend[r * hidden + h];
        }
        srcs.push(last);
        let normed: Vec<Vec<f32>> = srcs
            .iter()
            .map(|s| {
                let sq: f32 = s.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
                let inv = 1.0 / (sq + eps).sqrt();
                s.iter().map(|v| v * inv).collect()
            })
            .collect();
        for q in 0..n_queries {
            let qr = &queries[q * hidden..(q + 1) * hidden];
            let scores: Vec<f32> = normed
                .iter()
                .map(|s| qr.iter().zip(s).map(|(a, b)| a * b).sum::<f32>() * scale)
                .collect();
            let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for h in 0..hidden {
                let mut acc = 0.0_f32;
                for (s, nv) in normed.iter().enumerate() {
                    acc += (exps[s] / sum) * nv[h];
                }
                want[q * rows * hidden + r * hidden + h] = acc;
            }
        }
    }

    let d_blocks = upload_bytes(gpu, &f32_bytes(&blocks));
    let d_partial = upload_bytes(gpu, &f32_bytes(&partial));
    let d_addend = upload_bytes(gpu, &f32_bytes(&addend));
    let d_queries = upload_bytes(gpu, &f32_bytes(&queries));
    let d_out = alloc(gpu.device(), (want.len() * 4) as u64).expect("alloc out");

    let args = AttnResArgs {
        rows: rows as u32,
        hidden: hidden as u32,
        n_blocks: n_blocks as u32,
        n_queries: n_queries as u32,
        eps,
        has_addend: 1,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_attn_res"),
        &LaunchArgs::new(
            Extent3::linear(rows as u32),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(d_blocks.view()),
                KernelArg::Buffer(d_partial.view()),
                KernelArg::Buffer(d_addend.view()),
                KernelArg::Buffer(d_queries.view()),
                KernelArg::Buffer(d_out.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_out), &want, 1e-4, "attn_res");
}

/// Verifies `kernel_inkling2_loop_control` against an exact Rust oracle.
pub fn loop_control_matches_oracle(gpu: &Gpu) {
    let rows = 2usize;
    let hidden = 64usize;
    let rank = 16usize;
    let eps = 1e-6_f32;

    let cur: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.11).sin())
        .collect();
    let msg: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32) * 0.17).cos())
        .collect();
    let down_w: Vec<u16> = (0..rank * 2 * hidden)
        .map(|i| f32_to_bf16(((i as f32) * 0.05).sin() * 0.2))
        .collect();
    let gain_w: Vec<u16> = (0..hidden * rank)
        .map(|i| f32_to_bf16(((i as f32) * 0.09).cos() * 0.2))
        .collect();
    let forget_w: Vec<u16> = (0..hidden * rank)
        .map(|i| f32_to_bf16(((i as f32) * 0.07).sin() * 0.2))
        .collect();

    let mut want = vec![0.0_f32; rows * hidden];
    for r in 0..rows {
        let cr = &cur[r * hidden..(r + 1) * hidden];
        let mr = &msg[r * hidden..(r + 1) * hidden];
        let inv_c = 1.0 / (cr.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
        let inv_m = 1.0 / (mr.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
        let mut hv = vec![0.0_f32; rank];
        for k in 0..rank {
            let mut acc = 0.0_f32;
            for i in 0..hidden {
                acc += bf16_to_f32(down_w[k * 2 * hidden + i]) * (cr[i] * inv_c);
                acc += bf16_to_f32(down_w[k * 2 * hidden + hidden + i]) * (mr[i] * inv_m);
            }
            hv[k] = acc;
        }
        for i in 0..hidden {
            let mut g = 0.0_f32;
            let mut f = 0.0_f32;
            for k in 0..rank {
                g += bf16_to_f32(gain_w[i * rank + k]) * hv[k];
                f += bf16_to_f32(forget_w[i * rank + k]) * hv[k];
            }
            want[r * hidden + i] = (1.0 - f.tanh()) * cr[i] + g.tanh() * mr[i];
        }
    }

    let d_cur = upload_bytes(gpu, &f32_bytes(&cur));
    let d_msg = upload_bytes(gpu, &f32_bytes(&msg));
    let d_down = upload_bytes(gpu, cast_slice(&down_w));
    let d_gain = upload_bytes(gpu, cast_slice(&gain_w));
    let d_forget = upload_bytes(gpu, cast_slice(&forget_w));
    let d_out = alloc(gpu.device(), (want.len() * 4) as u64).expect("alloc out");

    let args = LoopCtrlArgs {
        rows: rows as u32,
        hidden: hidden as u32,
        rank: rank as u32,
        eps,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_loop_control"),
        &LaunchArgs::new(
            Extent3::linear(rows as u32),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(d_cur.view()),
                KernelArg::Buffer(d_msg.view()),
                KernelArg::Buffer(d_down.view()),
                KernelArg::Buffer(d_gain.view()),
                KernelArg::Buffer(d_forget.view()),
                KernelArg::Buffer(d_out.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_out), &want, 1e-4, "loop_control");
}

/// Verifies both static and dynamic short convolutions against Rust oracles.
pub fn short_convs_match_oracle(gpu: &Gpu) {
    let rows = 5usize;
    let channels = 32usize;
    let k = 4usize;
    let scale = 1.0 / ((k + 1) as f32).sqrt();
    let x: Vec<f32> = (0..rows * channels)
        .map(|i| ((i as f32) * 0.21).sin())
        .collect();
    let w: Vec<u16> = (0..channels * k)
        .map(|i| f32_to_bf16(((i as f32) * 0.31).cos() * 0.5))
        .collect();
    let init_state = vec![0.0_f32; (k - 1) * channels];

    let mut want = vec![0.0_f32; rows * channels];
    let mut want_state = init_state.clone();
    for c in 0..channels {
        let mut hist = [0.0_f32; 3];
        for t in 0..rows {
            let cur = x[t * channels + c];
            let acc = cur * bf16_to_f32(w[c * k + 3])
                + hist[0] * bf16_to_f32(w[c * k])
                + hist[1] * bf16_to_f32(w[c * k + 1])
                + hist[2] * bf16_to_f32(w[c * k + 2]);
            want[t * channels + c] = (acc + cur) * scale;
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = cur;
        }
        for l in 0..3 {
            want_state[l * channels + c] = hist[l];
        }
    }

    let d_x = upload_bytes(gpu, &f32_bytes(&x));
    let d_w = upload_bytes(gpu, cast_slice(&w));
    let d_state = upload_bytes(gpu, &f32_bytes(&init_state));
    let d_out = alloc(gpu.device(), (want.len() * 4) as u64).expect("alloc");

    let args = SconvStaticArgs {
        rows: rows as u32,
        channels: channels as u32,
        kernel_size: k as u32,
        scale,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_short_conv_static"),
        &LaunchArgs::new(
            Extent3::linear(1),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(d_x.view()),
                KernelArg::Buffer(d_w.view()),
                KernelArg::Buffer(d_state.view()),
                KernelArg::Buffer(d_out.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_out), &want, 1e-5, "sconv_static");
    assert_close(
        &gpu.read_f32(&d_state),
        &want_state,
        1e-5,
        "sconv_static_state",
    );

    let kv_heads = 2usize;
    let head_dim = 16usize;
    let dyn_scale = 1.0 / ((2 * k + 1) as f32).sqrt();
    let dyn_w: Vec<f32> = (0..rows * kv_heads * k)
        .map(|i| ((i as f32) * 0.13).cos() * 0.25)
        .collect();
    let static_w: Vec<u16> = (0..k * channels)
        .map(|i| f32_to_bf16(((i as f32) * 0.19).sin() * 0.25))
        .collect();
    let mut want_dyn = vec![0.0_f32; rows * channels];
    for c in 0..channels {
        let h = c / head_dim;
        let mut hist = [0.0_f32; 3];
        for t in 0..rows {
            let cur = x[t * channels + c];
            let dw = &dyn_w[(t * kv_heads + h) * k..][..k];
            let mut acc = cur * (bf16_to_f32(static_w[c]) + dw[0]);
            for lag in 1..k {
                acc += hist[3 - lag] * (bf16_to_f32(static_w[lag * channels + c]) + dw[lag]);
            }
            want_dyn[t * channels + c] = (acc + cur) * dyn_scale;
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = cur;
        }
    }
    let d_dyn_w = upload_bytes(gpu, &f32_bytes(&dyn_w));
    let d_sw = upload_bytes(gpu, cast_slice(&static_w));
    let d_state2 = upload_bytes(gpu, &f32_bytes(&init_state));
    let d_out2 = alloc(gpu.device(), (want_dyn.len() * 4) as u64).expect("alloc");
    let dargs = SconvDynamicArgs {
        rows: rows as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        kernel_size: k as u32,
        scale: dyn_scale,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_short_conv_dynamic"),
        &LaunchArgs::new(
            Extent3::linear(1),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&dargs)),
                KernelArg::Buffer(d_x.view()),
                KernelArg::Buffer(d_dyn_w.view()),
                KernelArg::Buffer(d_sw.view()),
                KernelArg::Buffer(d_state2.view()),
                KernelArg::Buffer(d_out2.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_out2), &want_dyn, 1e-5, "sconv_dynamic");
}

/// Verifies NVFP4 and MXFP8 routed MoE W13 (interleaved softcapped SwiGLU) + W2 accum.
pub fn moe_nvfp4_and_mxfp8_match_oracle(gpu: &Gpu) {
    let rows = 2usize;
    let in_dim = 64usize;
    let inter_dim = 32usize;
    let out_dim = 64usize;
    let n_experts = 4usize;
    let top_k = 2usize;
    let gate_cap = 4.0_f32;
    let up_cap = 25.0_f32;

    let x: Vec<f32> = (0..rows * in_dim)
        .map(|i| ((i as f32) * 0.17).sin() * 0.5)
        .collect();
    let expert_ids: Vec<u32> = vec![1, 3, 0, 2];
    let expert_weights: Vec<f32> = vec![4.5, 3.5, 5.0, 3.0];

    let w13_packed: Vec<u8> = (0..n_experts * 2 * inter_dim * (in_dim / 2))
        .map(|i| (((i * 7 + 3) & 0xF) | (((i * 11 + 5) & 0xF) << 4)) as u8)
        .collect();
    let w13_scale: Vec<u8> = (0..n_experts * 2 * inter_dim * (in_dim / 16))
        .map(|i| 0x30 + (i % 16) as u8)
        .collect();
    let w13_amax: Vec<f32> = vec![1.2, 0.8, 1.5, 0.9];
    let w13_denom: Vec<f32> = vec![2.5];

    let w2_packed: Vec<u8> = (0..n_experts * out_dim * (inter_dim / 2))
        .map(|i| (((i * 5 + 1) & 0xF) | (((i * 9 + 2) & 0xF) << 4)) as u8)
        .collect();
    let w2_scale: Vec<u8> = (0..n_experts * out_dim * (inter_dim / 16))
        .map(|i| 0x30 + (i % 12) as u8)
        .collect();
    let w2_amax: Vec<f32> = vec![0.7, 1.1, 0.9, 1.3];
    let w2_denom: Vec<f32> = vec![2.0];

    let nvfp4_dot = |w: &[u8], s: &[u8], xv: &[f32]| -> f32 {
        let mut acc = 0.0_f32;
        for (g, &sb) in s.iter().enumerate() {
            let gs = e4m3_to_f32(sb);
            let mut part = 0.0_f32;
            for b in 0..8 {
                let byte = w[g * 8 + b];
                part += E2M1_LUT[(byte & 0xF) as usize] * xv[g * 16 + b * 2];
                part += E2M1_LUT[(byte >> 4) as usize] * xv[g * 16 + b * 2 + 1];
            }
            acc += gs * part;
        }
        acc
    };

    let mut want_act = vec![0.0_f32; rows * top_k * inter_dim];
    for col in 0..rows * top_k {
        let r = col / top_k;
        let e = expert_ids[col] as usize;
        let outer = w13_amax[e] / w13_denom[0];
        let rw = expert_weights[col];
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        for j in 0..inter_dim {
            let gr = e * 2 * inter_dim + 2 * j;
            let ur = gr + 1;
            let g = nvfp4_dot(
                &w13_packed[gr * (in_dim / 2)..(gr + 1) * (in_dim / 2)],
                &w13_scale[gr * (in_dim / 16)..(gr + 1) * (in_dim / 16)],
                xr,
            ) * outer;
            let u = nvfp4_dot(
                &w13_packed[ur * (in_dim / 2)..(ur + 1) * (in_dim / 2)],
                &w13_scale[ur * (in_dim / 16)..(ur + 1) * (in_dim / 16)],
                xr,
            ) * outer;
            want_act[col * inter_dim + j] =
                softcap(g, gate_cap) * sigmoid(g) * softcap(u, up_cap) * rw;
        }
    }

    let mut want_out = vec![0.0_f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut sum = 0.0_f32;
            for slot in 0..top_k {
                let col = r * top_k + slot;
                let e = expert_ids[col] as usize;
                let outer = w2_amax[e] / w2_denom[0];
                let row_idx = e * out_dim + o;
                sum += nvfp4_dot(
                    &w2_packed[row_idx * (inter_dim / 2)..(row_idx + 1) * (inter_dim / 2)],
                    &w2_scale[row_idx * (inter_dim / 16)..(row_idx + 1) * (inter_dim / 16)],
                    &want_act[col * inter_dim..(col + 1) * inter_dim],
                ) * outer;
            }
            want_out[r * out_dim + o] = sum;
        }
    }

    let d_x = upload_bytes(gpu, &f32_bytes(&x));
    let d_ids = upload_bytes(gpu, cast_slice(&expert_ids));
    let d_wts = upload_bytes(gpu, &f32_bytes(&expert_weights));
    let d_w13 = upload_bytes(gpu, &w13_packed);
    let d_s13 = upload_bytes(gpu, &w13_scale);
    let d_a13 = upload_bytes(gpu, &f32_bytes(&w13_amax));
    let d_d13 = upload_bytes(gpu, &f32_bytes(&w13_denom));
    let d_act = alloc(gpu.device(), (want_act.len() * 4) as u64).expect("alloc act");

    let args13 = MoeW13Args {
        rows: rows as u32,
        in_dim: in_dim as u32,
        inter_dim: inter_dim as u32,
        top_k: top_k as u32,
        gate_cap,
        up_cap,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_moe_nvfp4_w13_swiglu"),
        &LaunchArgs::new(
            Extent3::new((inter_dim as u32).div_ceil(4), (rows * top_k) as u32, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args13)),
                KernelArg::Buffer(d_x.view()),
                KernelArg::Buffer(d_w13.view()),
                KernelArg::Buffer(d_s13.view()),
                KernelArg::Buffer(d_a13.view()),
                KernelArg::Buffer(d_d13.view()),
                KernelArg::Buffer(d_ids.view()),
                KernelArg::Buffer(d_wts.view()),
                KernelArg::Buffer(d_act.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_act), &want_act, 1e-3, "moe_nvfp4_w13");

    let d_w2 = upload_bytes(gpu, &w2_packed);
    let d_s2 = upload_bytes(gpu, &w2_scale);
    let d_a2 = upload_bytes(gpu, &f32_bytes(&w2_amax));
    let d_d2 = upload_bytes(gpu, &f32_bytes(&w2_denom));
    let d_out = alloc(gpu.device(), (want_out.len() * 4) as u64).expect("alloc out");
    let args2 = MoeW2Args {
        rows: rows as u32,
        inter_dim: inter_dim as u32,
        out_dim: out_dim as u32,
        top_k: top_k as u32,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_moe_nvfp4_w2_accum"),
        &LaunchArgs::new(
            Extent3::new((out_dim as u32).div_ceil(4), rows as u32, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args2)),
                KernelArg::Buffer(d_act.view()),
                KernelArg::Buffer(d_w2.view()),
                KernelArg::Buffer(d_s2.view()),
                KernelArg::Buffer(d_a2.view()),
                KernelArg::Buffer(d_d2.view()),
                KernelArg::Buffer(d_ids.view()),
                KernelArg::Buffer(d_out.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_out), &want_out, 1e-2, "moe_nvfp4_w2");

    // --- MXFP8 ---
    let mx_w13: Vec<u8> = (0..n_experts * 2 * inter_dim * in_dim)
        .map(|i| ((i * 13 + 7) % 120) as u8)
        .collect();
    let mx_s13: Vec<u8> = (0..n_experts * 2 * inter_dim * (in_dim / 32))
        .map(|i| 125 + (i % 4) as u8)
        .collect();
    let mx_w2: Vec<u8> = (0..n_experts * out_dim * inter_dim)
        .map(|i| ((i * 17 + 3) % 120) as u8)
        .collect();
    let mx_s2: Vec<u8> = (0..n_experts * out_dim * (inter_dim / 32))
        .map(|i| 125 + (i % 4) as u8)
        .collect();

    let mxfp8_dot = |w: &[u8], s: &[u8], xv: &[f32]| -> f32 {
        let mut acc = 0.0_f32;
        for (g, &sb) in s.iter().enumerate() {
            let gs = ue8m0_to_f32(sb);
            let mut part = 0.0_f32;
            for i in 0..32 {
                part += e4m3_to_f32(w[g * 32 + i]) * xv[g * 32 + i];
            }
            acc += gs * part;
        }
        acc
    };

    let mut want_mx_act = vec![0.0_f32; rows * top_k * inter_dim];
    for col in 0..rows * top_k {
        let r = col / top_k;
        let e = expert_ids[col] as usize;
        let rw = expert_weights[col];
        let xr = &x[r * in_dim..(r + 1) * in_dim];
        for j in 0..inter_dim {
            let gr = e * 2 * inter_dim + 2 * j;
            let ur = gr + 1;
            let g = mxfp8_dot(
                &mx_w13[gr * in_dim..(gr + 1) * in_dim],
                &mx_s13[gr * (in_dim / 32)..(gr + 1) * (in_dim / 32)],
                xr,
            );
            let u = mxfp8_dot(
                &mx_w13[ur * in_dim..(ur + 1) * in_dim],
                &mx_s13[ur * (in_dim / 32)..(ur + 1) * (in_dim / 32)],
                xr,
            );
            want_mx_act[col * inter_dim + j] =
                softcap(g, gate_cap) * sigmoid(g) * softcap(u, up_cap) * rw;
        }
    }
    let mut want_mx_out = vec![0.0_f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut sum = 0.0_f32;
            for slot in 0..top_k {
                let col = r * top_k + slot;
                let e = expert_ids[col] as usize;
                let row_idx = e * out_dim + o;
                sum += mxfp8_dot(
                    &mx_w2[row_idx * inter_dim..(row_idx + 1) * inter_dim],
                    &mx_s2[row_idx * (inter_dim / 32)..(row_idx + 1) * (inter_dim / 32)],
                    &want_mx_act[col * inter_dim..(col + 1) * inter_dim],
                );
            }
            want_mx_out[r * out_dim + o] = sum;
        }
    }

    let d_mx_w13 = upload_bytes(gpu, &mx_w13);
    let d_mx_s13 = upload_bytes(gpu, &mx_s13);
    let d_mx_w2 = upload_bytes(gpu, &mx_w2);
    let d_mx_s2 = upload_bytes(gpu, &mx_s2);
    let d_mx_act = alloc(gpu.device(), (want_mx_act.len() * 4) as u64).expect("alloc");
    let d_mx_out = alloc(gpu.device(), (want_mx_out.len() * 4) as u64).expect("alloc");

    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_moe_mxfp8_w13_swiglu"),
        &LaunchArgs::new(
            Extent3::new((inter_dim as u32).div_ceil(4), (rows * top_k) as u32, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args13)),
                KernelArg::Buffer(d_x.view()),
                KernelArg::Buffer(d_mx_w13.view()),
                KernelArg::Buffer(d_mx_s13.view()),
                KernelArg::Buffer(d_ids.view()),
                KernelArg::Buffer(d_wts.view()),
                KernelArg::Buffer(d_mx_act.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(
        &gpu.read_f32(&d_mx_act),
        &want_mx_act,
        1e-2,
        "moe_mxfp8_w13",
    );

    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_moe_mxfp8_w2_accum"),
        &LaunchArgs::new(
            Extent3::new((out_dim as u32).div_ceil(4), rows as u32, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args2)),
                KernelArg::Buffer(d_mx_act.view()),
                KernelArg::Buffer(d_mx_w2.view()),
                KernelArg::Buffer(d_mx_s2.view()),
                KernelArg::Buffer(d_ids.view()),
                KernelArg::Buffer(d_mx_out.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_mx_out), &want_mx_out, 5e-2, "moe_mxfp8_w2");
}

/// Verifies QKVR prep + KV cache store + relative-bias sink gated attention.
pub fn rel_attn_with_sink_matches_oracle(gpu: &Gpu) {
    let rows = 4usize;
    let n_head = 4usize;
    let n_head_kv = 2usize;
    let head_dim = 128usize;
    let d_rel = 16usize;
    let rel_extent = 8usize;
    let kv_cap = 16usize;
    let eps = 1e-6_f32;
    let qk_clip = 22.627_417_f32;
    let rel_clip = 8.0_f32;

    let q_in: Vec<f32> = (0..rows * n_head * head_dim)
        .map(|i| ((i as f32) * 0.11).sin() * 0.5)
        .collect();
    let k_in: Vec<f32> = (0..rows * n_head_kv * head_dim)
        .map(|i| ((i as f32) * 0.13).cos() * 0.5)
        .collect();
    let v_in: Vec<f32> = (0..rows * n_head_kv * head_dim)
        .map(|i| ((i as f32) * 0.07).sin() * 0.5)
        .collect();
    let r_in: Vec<f32> = (0..rows * n_head * d_rel)
        .map(|i| ((i as f32) * 0.19).cos() * 0.4)
        .collect();
    let q_scale: Vec<u16> = vec![f32_to_bf16(1.0); n_head * head_dim];
    let r_scale: Vec<u16> = vec![f32_to_bf16(1.0); n_head * d_rel];
    let rel_bias: Vec<f32> = (0..rows * n_head * rel_extent)
        .map(|i| ((i as f32) * 0.23).sin() * 0.3)
        .collect();
    let sinks: Vec<f32> = (0..rows * n_head).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let gate: Vec<f32> = (0..rows * n_head * head_dim)
        .map(|i| ((i as f32) * 0.05).cos())
        .collect();

    let mut q_prepped = q_in.clone();
    let mut k_prepped = k_in.clone();
    let mut v_prepped = v_in.clone();
    for r in 0..rows {
        for h in 0..n_head {
            let sl = &mut q_prepped[(r * n_head + h) * head_dim..][..head_dim];
            let norm = sl.iter().map(|v| v * v).sum::<f32>().sqrt();
            let f = (qk_clip / (norm + 1e-6)).min(1.0);
            for v in sl {
                *v *= f;
            }
        }
        for kvh in 0..n_head_kv {
            let ks = &mut k_prepped[(r * n_head_kv + kvh) * head_dim..][..head_dim];
            let knorm = ks.iter().map(|v| v * v).sum::<f32>().sqrt();
            let kf = (qk_clip / (knorm + 1e-6)).min(1.0);
            for v in ks {
                *v *= kf;
            }
            let vs = &mut v_prepped[(r * n_head_kv + kvh) * head_dim..][..head_dim];
            let vinv = 1.0 / (vs.iter().map(|v| v * v).sum::<f32>() / head_dim as f32 + eps).sqrt();
            for v in vs {
                *v *= vinv;
            }
        }
    }

    let attn_scale = 1.0 / (head_dim as f32).sqrt();
    let mut want = vec![0.0_f32; rows * n_head * head_dim];
    for r in 0..rows {
        for h in 0..n_head {
            let kvh = h / (n_head / n_head_kv);
            let qv = &q_prepped[(r * n_head + h) * head_dim..][..head_dim];
            let rb = &rel_bias[(r * n_head + h) * rel_extent..][..rel_extent];
            let sink = sinks[r * n_head + h];
            let mut scores = Vec::new();
            for kp in 0..=r {
                let kv = &k_prepped[(kp * n_head_kv + kvh) * head_dim..][..head_dim];
                let dot: f32 = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * attn_scale;
                let dist = r - kp;
                let b = if dist < rel_extent { rb[dist] } else { 0.0 };
                scores.push(dot + b);
            }
            let mx = scores.iter().copied().fold(sink, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
            let denom = exps.iter().sum::<f32>() + (sink - mx).exp();
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for (kp, &e) in exps.iter().enumerate() {
                    acc += (e / denom) * v_prepped[(kp * n_head_kv + kvh) * head_dim + d];
                }
                let idx = (r * n_head + h) * head_dim + d;
                want[idx] = acc * sigmoid(gate[idx]);
            }
        }
    }

    let d_q = upload_bytes(gpu, &f32_bytes(&q_in));
    let d_k = upload_bytes(gpu, &f32_bytes(&k_in));
    let d_v = upload_bytes(gpu, &f32_bytes(&v_in));
    let d_r = upload_bytes(gpu, &f32_bytes(&r_in));
    let d_qs = upload_bytes(gpu, cast_slice(&q_scale));
    let d_rs = upload_bytes(gpu, cast_slice(&r_scale));
    let d_kcache = alloc(gpu.device(), (kv_cap * n_head_kv * head_dim * 2) as u64).expect("alloc");
    let d_vcache = alloc(gpu.device(), (kv_cap * n_head_kv * head_dim * 2) as u64).expect("alloc");

    let prep_args = QkvrPrepArgs {
        rows: rows as u32,
        n_head: n_head as u32,
        n_head_kv: n_head_kv as u32,
        head_dim: head_dim as u32,
        d_rel: d_rel as u32,
        qk_clip,
        rel_clip,
        eps,
        is_full: 0,
        log_n_floor: 8192.0,
        log_alpha: 0.1,
        pos0: 0,
        kv_cap: kv_cap as u32,
        is_swa: 1,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_qkvr_prep_store"),
        &LaunchArgs::new(
            Extent3::new((n_head + n_head_kv) as u32, rows as u32, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&prep_args)),
                KernelArg::Buffer(d_q.view()),
                KernelArg::Buffer(d_k.view()),
                KernelArg::Buffer(d_v.view()),
                KernelArg::Buffer(d_r.view()),
                KernelArg::Buffer(d_qs.view()),
                KernelArg::Buffer(d_rs.view()),
                KernelArg::Buffer(d_kcache.view()),
                KernelArg::Buffer(d_vcache.view()),
            ],
        ),
    );

    let d_rb = upload_bytes(gpu, &f32_bytes(&rel_bias));
    let d_sinks = upload_bytes(gpu, &f32_bytes(&sinks));
    let d_gate = upload_bytes(gpu, &f32_bytes(&gate));
    let d_out = alloc(gpu.device(), (want.len() * 4) as u64).expect("alloc");

    let attn_args = RelAttnArgs {
        rows: rows as u32,
        n_head: n_head as u32,
        n_head_kv: n_head_kv as u32,
        head_dim: head_dim as u32,
        rel_extent: rel_extent as u32,
        pos0: 0,
        kv_cap: kv_cap as u32,
        window_size: 1024,
        is_swa: 1,
        scale: attn_scale,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_rel_attn"),
        &LaunchArgs::new(
            Extent3::new(n_head as u32, (rows as u32).div_ceil(4), 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&attn_args)),
                KernelArg::Buffer(d_q.view()),
                KernelArg::Buffer(d_rb.view()),
                KernelArg::Buffer(d_sinks.view()),
                KernelArg::Buffer(d_gate.view()),
                KernelArg::Buffer(d_kcache.view()),
                KernelArg::Buffer(d_vcache.view()),
                KernelArg::Buffer(d_out.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_out), &want, 5e-3, "rel_attn_sink");
}

/// Verifies top-k sigmoid router with correction bias and route_scale.
pub fn route_topk_matches_oracle(gpu: &Gpu) {
    let rows = 2usize;
    let n_experts = 16usize;
    let top_k = 4usize;
    let route_scale = 8.0_f32;
    let logits: Vec<f32> = (0..rows * n_experts)
        .map(|i| ((i as f32) * 0.37).sin() * 2.0)
        .collect();
    let bias: Vec<f32> = (0..n_experts)
        .map(|i| ((i as f32) * 0.29).cos() * 0.1)
        .collect();

    let d_logits = upload_bytes(gpu, &f32_bytes(&logits));
    let d_bias = upload_bytes(gpu, &f32_bytes(&bias));
    let d_ids = alloc(gpu.device(), (rows * top_k * 4) as u64).expect("alloc");
    let d_wts = alloc(gpu.device(), (rows * top_k * 4) as u64).expect("alloc");

    let args = RouteArgs {
        rows: rows as u32,
        n_experts: n_experts as u32,
        top_k: top_k as u32,
        route_scale,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_route_topk"),
        &LaunchArgs::new(
            Extent3::linear(rows as u32),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(d_logits.view()),
                KernelArg::Buffer(d_bias.view()),
                KernelArg::Buffer(d_ids.view()),
                KernelArg::Buffer(d_wts.view()),
            ],
        ),
    );
    drain(gpu);
    let got_wts = gpu.read_f32(&d_wts);
    for r in 0..rows {
        let sum: f32 = got_wts[r * top_k..(r + 1) * top_k].iter().sum();
        assert!(
            (sum - route_scale).abs() < 1e-4,
            "weights sum {sum} != {route_scale}"
        );
    }
}

/// Verifies BF16 linear + RMSNorm.
pub fn bf16_linear_and_norms_match_oracle(gpu: &Gpu) {
    let rows = 2usize;
    let in_dim = 64usize;
    let out_dim = 32usize;
    let x: Vec<f32> = (0..rows * in_dim)
        .map(|i| ((i as f32) * 0.11).sin())
        .collect();
    let w: Vec<u16> = (0..out_dim * in_dim)
        .map(|i| f32_to_bf16(((i as f32) * 0.07).cos()))
        .collect();
    let mut want = vec![0.0_f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0.0_f32;
            for k in 0..in_dim {
                acc += x[r * in_dim + k] * bf16_to_f32(w[o * in_dim + k]);
            }
            want[r * out_dim + o] = acc;
        }
    }
    let d_x = upload_bytes(gpu, &f32_bytes(&x));
    let d_w = upload_bytes(gpu, cast_slice(&w));
    let d_out = alloc(gpu.device(), (want.len() * 4) as u64).expect("alloc");
    let args = LinearArgs {
        rows: rows as u32,
        in_dim: in_dim as u32,
        out_dim: out_dim as u32,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_linear_bf16"),
        &LaunchArgs::new(
            Extent3::new((out_dim as u32).div_ceil(4), (rows as u32).div_ceil(8), 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(d_x.view()),
                KernelArg::Buffer(d_w.view()),
                KernelArg::Buffer(d_out.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_out), &want, 1e-4, "linear_bf16");

    let nw: Vec<u16> = (0..in_dim)
        .map(|i| f32_to_bf16(1.0 + (i as f32) * 0.01))
        .collect();
    let mut want_norm = vec![0.0_f32; rows * in_dim];
    for r in 0..rows {
        let sl = &x[r * in_dim..(r + 1) * in_dim];
        let inv = 1.0 / (sl.iter().map(|v| v * v).sum::<f32>() / in_dim as f32 + 1e-6).sqrt();
        for i in 0..in_dim {
            want_norm[r * in_dim + i] = sl[i] * inv * bf16_to_f32(nw[i]);
        }
    }
    let d_nw = upload_bytes(gpu, cast_slice(&nw));
    let d_nout = alloc(gpu.device(), (want_norm.len() * 4) as u64).expect("alloc");
    let nargs = NormArgs {
        rows: rows as u32,
        hidden: in_dim as u32,
        eps: 1e-6,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_rmsnorm_bf16"),
        &LaunchArgs::new(
            Extent3::linear(rows as u32),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&nargs)),
                KernelArg::Buffer(d_x.view()),
                KernelArg::Buffer(d_nw.view()),
                KernelArg::Buffer(d_nout.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_nout), &want_norm, 1e-4, "rmsnorm_bf16");
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QkvrGateSinkArgs {
    rows: u32,
    in_dim: u32,
    q_dim: u32,
    kv_dim: u32,
    r_dim: u32,
    n_head: u32,
}

/// Tests the fused Q+K+V+R+Gate+Sink BF16 mega-projection against CPU matmul oracles.
pub fn qkvr_gate_sink_matches_oracle(gpu: &Gpu) {
    let rows = 10usize; // full 8-row tile + 2-row tail tile
    let in_dim = 68usize; // intentionally non-multiple of 128 to exercise tail path
    let q_dim = 16usize;
    let kv_dim = 8usize;
    let r_dim = 6usize;
    let n_head = 4usize;

    let mk_x = |seed: f32| -> Vec<f32> {
        (0..rows * in_dim)
            .map(|i| (((i * 11 + 3) % 23) as f32 - 11.0) * 0.05 + seed)
            .collect()
    };
    let mk_w = |out_d: usize, seed: usize| -> Vec<u16> {
        (0..out_d * in_dim)
            .map(|i| f32_to_bf16((((i * 7 + seed) % 19) as f32 - 9.0) * 0.04))
            .collect()
    };
    let matmul = |x: &[f32], w: &[u16], out_d: usize| -> Vec<f32> {
        let mut y = vec![0.0_f32; rows * out_d];
        for r in 0..rows {
            for o in 0..out_d {
                let mut acc = 0.0_f32;
                for k in 0..in_dim {
                    acc += x[r * in_dim + k] * bf16_to_f32(w[o * in_dim + k]);
                }
                y[r * out_d + o] = acc;
            }
        }
        y
    };

    let x_q = mk_x(0.1);
    let x_k = mk_x(-0.2);
    let x_v = mk_x(0.3);
    let wq = mk_w(q_dim, 1);
    let wk = mk_w(kv_dim, 2);
    let wv = mk_w(kv_dim, 3);
    let wr = mk_w(r_dim, 4);
    let wgate = mk_w(q_dim, 5);
    let wsink = mk_w(n_head, 6);

    let want_q = matmul(&x_q, &wq, q_dim);
    let want_k = matmul(&x_k, &wk, kv_dim);
    let want_v = matmul(&x_v, &wv, kv_dim);
    let want_r = matmul(&x_q, &wr, r_dim);
    let want_gate = matmul(&x_q, &wgate, q_dim);
    let want_sink = matmul(&x_q, &wsink, n_head);

    let d_xq = upload_bytes(gpu, &f32_bytes(&x_q));
    let d_xk = upload_bytes(gpu, &f32_bytes(&x_k));
    let d_xv = upload_bytes(gpu, &f32_bytes(&x_v));
    let d_wq = upload_bytes(gpu, cast_slice(&wq));
    let d_wk = upload_bytes(gpu, cast_slice(&wk));
    let d_wv = upload_bytes(gpu, cast_slice(&wv));
    let d_wr = upload_bytes(gpu, cast_slice(&wr));
    let d_wg = upload_bytes(gpu, cast_slice(&wgate));
    let d_ws = upload_bytes(gpu, cast_slice(&wsink));

    let d_oq = alloc(gpu.device(), (want_q.len() * 4) as u64).expect("alloc");
    let d_ok = alloc(gpu.device(), (want_k.len() * 4) as u64).expect("alloc");
    let d_ov = alloc(gpu.device(), (want_v.len() * 4) as u64).expect("alloc");
    let d_or = alloc(gpu.device(), (want_r.len() * 4) as u64).expect("alloc");
    let d_og = alloc(gpu.device(), (want_gate.len() * 4) as u64).expect("alloc");
    let d_os = alloc(gpu.device(), (want_sink.len() * 4) as u64).expect("alloc");

    let args = QkvrGateSinkArgs {
        rows: rows as u32,
        in_dim: in_dim as u32,
        q_dim: q_dim as u32,
        kv_dim: kv_dim as u32,
        r_dim: r_dim as u32,
        n_head: n_head as u32,
    };
    let total_out = (q_dim + kv_dim + kv_dim + r_dim + q_dim + n_head) as u32;
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_qkvr_gate_sink_bf16"),
        &LaunchArgs::new(
            Extent3::new(total_out.div_ceil(4), (rows as u32).div_ceil(8), 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&args)),
                KernelArg::Buffer(d_xq.view()),
                KernelArg::Buffer(d_xk.view()),
                KernelArg::Buffer(d_xv.view()),
                KernelArg::Buffer(d_wq.view()),
                KernelArg::Buffer(d_wk.view()),
                KernelArg::Buffer(d_wv.view()),
                KernelArg::Buffer(d_wr.view()),
                KernelArg::Buffer(d_wg.view()),
                KernelArg::Buffer(d_ws.view()),
                KernelArg::Buffer(d_oq.view()),
                KernelArg::Buffer(d_ok.view()),
                KernelArg::Buffer(d_ov.view()),
                KernelArg::Buffer(d_or.view()),
                KernelArg::Buffer(d_og.view()),
                KernelArg::Buffer(d_os.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_oq), &want_q, 1e-4, "fused_q");
    assert_close(&gpu.read_f32(&d_ok), &want_k, 1e-4, "fused_k");
    assert_close(&gpu.read_f32(&d_ov), &want_v, 1e-4, "fused_v");
    assert_close(&gpu.read_f32(&d_or), &want_r, 1e-4, "fused_r");
    assert_close(&gpu.read_f32(&d_og), &want_gate, 1e-4, "fused_gate");
    assert_close(&gpu.read_f32(&d_os), &want_sink, 1e-4, "fused_sink");
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VitLayerscaleArgs {
    rows: u32,
    hidden: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AddArgs {
    count: u32,
}

/// Tests Vision Transformer kernels (`vit_layernorm_bf16`, `vit_linear_bias_bf16`, `vit_layerscale_add`).
pub fn vit_kernels_match_oracle(gpu: &Gpu) {
    let rows = 3usize;
    let hidden = 32usize;
    let out_dim = 16usize;

    let x: Vec<f32> = (0..rows * hidden)
        .map(|i| (((i * 13 + 7) % 29) as f32 - 14.0) * 0.1)
        .collect();
    let ln_w: Vec<u16> = (0..hidden)
        .map(|i| f32_to_bf16(0.9 + (i as f32) * 0.01))
        .collect();
    let ln_b: Vec<u16> = (0..hidden)
        .map(|i| f32_to_bf16(-0.1 + (i as f32) * 0.005))
        .collect();

    let mut want_ln = vec![0.0_f32; rows * hidden];
    for r in 0..rows {
        let sl = &x[r * hidden..(r + 1) * hidden];
        let mean = sl.iter().sum::<f32>() / hidden as f32;
        let var = sl.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for i in 0..hidden {
            want_ln[r * hidden + i] =
                (sl[i] - mean) * inv * bf16_to_f32(ln_w[i]) + bf16_to_f32(ln_b[i]);
        }
    }

    let d_x = upload_bytes(gpu, &f32_bytes(&x));
    let d_lnw = upload_bytes(gpu, cast_slice(&ln_w));
    let d_lnb = upload_bytes(gpu, cast_slice(&ln_b));
    let d_ln_out = alloc(gpu.device(), (want_ln.len() * 4) as u64).expect("alloc");
    let ln_args = NormArgs {
        rows: rows as u32,
        hidden: hidden as u32,
        eps: 1e-6,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_vit_layernorm_bf16"),
        &LaunchArgs::new(
            Extent3::linear(rows as u32),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&ln_args)),
                KernelArg::Buffer(d_x.view()),
                KernelArg::Buffer(d_lnw.view()),
                KernelArg::Buffer(d_lnb.view()),
                KernelArg::Buffer(d_ln_out.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_ln_out), &want_ln, 1e-4, "vit_layernorm");

    // vit_linear_bias_bf16
    let w: Vec<u16> = (0..out_dim * hidden)
        .map(|i| f32_to_bf16((((i * 5 + 3) % 17) as f32 - 8.0) * 0.05))
        .collect();
    let b: Vec<u16> = (0..out_dim)
        .map(|i| f32_to_bf16((i as f32 - 7.0) * 0.02))
        .collect();
    let mut want_lin = vec![0.0_f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = bf16_to_f32(b[o]);
            for k in 0..hidden {
                acc += want_ln[r * hidden + k] * bf16_to_f32(w[o * hidden + k]);
            }
            want_lin[r * out_dim + o] = acc;
        }
    }
    let d_w = upload_bytes(gpu, cast_slice(&w));
    let d_b = upload_bytes(gpu, cast_slice(&b));
    let d_lin = alloc(gpu.device(), (want_lin.len() * 4) as u64).expect("alloc");
    let lin_args = LinearArgs {
        rows: rows as u32,
        in_dim: hidden as u32,
        out_dim: out_dim as u32,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_vit_linear_bias_bf16"),
        &LaunchArgs::new(
            Extent3::new((out_dim as u32).div_ceil(4), rows as u32, 1),
            Extent3::linear(128),
            &[
                KernelArg::Bytes(bytes_of(&lin_args)),
                KernelArg::Buffer(d_ln_out.view()),
                KernelArg::Buffer(d_w.view()),
                KernelArg::Buffer(d_b.view()),
                KernelArg::Buffer(d_lin.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_lin), &want_lin, 1e-4, "vit_linear_bias");

    // vit_layerscale_add
    let gamma: Vec<u16> = (0..hidden)
        .map(|i| f32_to_bf16(0.2 + (i as f32) * 0.01))
        .collect();
    let mut want_ls = vec![0.0_f32; rows * hidden];
    for r in 0..rows {
        for i in 0..hidden {
            want_ls[r * hidden + i] =
                x[r * hidden + i] + want_ln[r * hidden + i] * bf16_to_f32(gamma[i]);
        }
    }
    let d_gamma = upload_bytes(gpu, cast_slice(&gamma));
    let d_res = upload_bytes(gpu, &f32_bytes(&x));
    let d_ls_out = alloc(gpu.device(), (want_ls.len() * 4) as u64).expect("alloc");
    let ls_args = VitLayerscaleArgs {
        rows: rows as u32,
        hidden: hidden as u32,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_vit_layerscale_add"),
        &LaunchArgs::new(
            Extent3::linear(((rows * hidden) as u32).div_ceil(256)),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&ls_args)),
                KernelArg::Buffer(d_res.view()),
                KernelArg::Buffer(d_ln_out.view()),
                KernelArg::Buffer(d_gamma.view()),
                KernelArg::Buffer(d_ls_out.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(
        &gpu.read_f32(&d_ls_out),
        &want_ls,
        1e-4,
        "vit_layerscale_add",
    );
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MtpMixArgs {
    rows: u32,
    stride_rows: u32,
    hidden: u32,
    n_taps: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MtpRnnArgs {
    rows: u32,
    rank: u32,
}

/// Tests MTP / DSpark kernels (`mtp_mix_rmsnorm`, `mtp_swiglu_mul`, `mtp_rnn_step`).
pub fn mtp_kernels_match_oracle(gpu: &Gpu) {
    let rows = 2usize;
    let hidden = 32usize;
    let alpha = [0.1_f32, 0.2, 0.3, 0.4];
    let taps: Vec<f32> = (0..4 * rows * hidden)
        .map(|i| (((i * 9 + 5) % 23) as f32 - 11.0) * 0.08)
        .collect();
    let nw: Vec<u16> = (0..hidden)
        .map(|i| f32_to_bf16(0.8 + (i as f32) * 0.02))
        .collect();

    let mut want_mix = vec![0.0_f32; rows * hidden];
    for r in 0..rows {
        let mut mixed = vec![0.0_f32; hidden];
        for t in 0..4 {
            for h in 0..hidden {
                mixed[h] += alpha[t] * taps[(t * rows + r) * hidden + h];
            }
        }
        let inv = 1.0 / (mixed.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + 1e-6).sqrt();
        for h in 0..hidden {
            want_mix[r * hidden + h] = mixed[h] * inv * bf16_to_f32(nw[h]);
        }
    }

    let d_taps = upload_bytes(gpu, &f32_bytes(&taps));
    let d_alpha = upload_bytes(gpu, &f32_bytes(&alpha));
    let d_nw = upload_bytes(gpu, cast_slice(&nw));
    let d_mix = alloc(gpu.device(), (want_mix.len() * 4) as u64).expect("alloc");
    let mix_args = MtpMixArgs {
        rows: rows as u32,
        stride_rows: rows as u32,
        hidden: hidden as u32,
        n_taps: 4,
        eps: 1e-6,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_mtp_mix_rmsnorm"),
        &LaunchArgs::new(
            Extent3::linear(rows as u32),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&mix_args)),
                KernelArg::Buffer(d_taps.view()),
                KernelArg::Buffer(d_alpha.view()),
                KernelArg::Buffer(d_nw.view()),
                KernelArg::Buffer(d_mix.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_mix), &want_mix, 1e-4, "mtp_mix_rmsnorm");

    // mtp_swiglu_mul
    let count = rows * hidden;
    let gate: Vec<f32> = (0..count).map(|i| (i as f32 - 30.0) * 0.05).collect();
    let up: Vec<f32> = (0..count).map(|i| (i as f32 - 20.0) * 0.03).collect();
    let want_swi: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| (g * sigmoid(g)) * u)
        .collect();
    let d_gate = upload_bytes(gpu, &f32_bytes(&gate));
    let d_up = upload_bytes(gpu, &f32_bytes(&up));
    let d_swi = alloc(gpu.device(), (count * 4) as u64).expect("alloc");
    let swi_args = AddArgs {
        count: count as u32,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_mtp_swiglu_mul"),
        &LaunchArgs::new(
            Extent3::linear((count as u32).div_ceil(256)),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&swi_args)),
                KernelArg::Buffer(d_gate.view()),
                KernelArg::Buffer(d_up.view()),
                KernelArg::Buffer(d_swi.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_swi), &want_swi, 1e-5, "mtp_swiglu_mul");

    // mtp_rnn_step
    let rank = 16usize;
    let state: Vec<f32> = (0..rank).map(|i| (i as f32 - 7.0) * 0.1).collect();
    let pre: Vec<f32> = (0..3 * rank).map(|i| (i as f32 - 20.0) * 0.07).collect();
    let mut want_state = vec![0.0_f32; rank];
    let mut want_latent = vec![0.0_f32; rank];
    for i in 0..rank {
        let g = sigmoid(pre[i]);
        let cand = pre[rank + i].tanh();
        let out_v = pre[2 * rank + i].tanh();
        want_state[i] = g * state[i] + (1.0 - g) * cand;
        want_latent[i] = out_v;
    }
    let d_pre = upload_bytes(gpu, &f32_bytes(&pre));
    let d_state = upload_bytes(gpu, &f32_bytes(&state));
    let d_latent = alloc(gpu.device(), (rank * 4) as u64).expect("alloc");
    let rnn_args = MtpRnnArgs {
        rows: 1,
        rank: rank as u32,
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_inkling2_mtp_rnn_step"),
        &LaunchArgs::new(
            Extent3::new((rank as u32).div_ceil(256), 1, 1),
            Extent3::linear(256),
            &[
                KernelArg::Bytes(bytes_of(&rnn_args)),
                KernelArg::Buffer(d_pre.view()),
                KernelArg::Buffer(d_state.view()),
                KernelArg::Buffer(d_latent.view()),
            ],
        ),
    );
    drain(gpu);
    assert_close(&gpu.read_f32(&d_state), &want_state, 1e-5, "mtp_rnn_state");
    assert_close(
        &gpu.read_f32(&d_latent),
        &want_latent,
        1e-5,
        "mtp_rnn_latent",
    );
}
