//! The entries a prefix-cached diffusion transformer adds: causal fused
//! attention for the text prefix, `QK-norm` fused with a per-token `RoPE`
//! table over separate q and k buffers, and the `tanh` / tanh-`GELU`
//! elementwise passes its embedders and gates use.
//!
//! The attention oracle is a direct `f64` softmax over exactly the keys a
//! row may see; the case also asserts the bidirectional answer differs, so
//! a kernel that ignored the mask could not pass. The rotation oracle turns
//! adjacent pairs `(x[2i], x[2i+1])` and the case asserts the rotate-half
//! pairing `(x[i], x[i + d/2])` gives a different answer on the same data.
#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture dimensions are small literals marshalled as the kernels' u32 argument fields"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "fixture sizes converted to floats are small, exact in f32"
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// Every entry point this file's cases launch.
pub const ENTRIES: [&str; 9] = [
    "kernel_dit_gqa_f32",
    "kernel_dit_rope_half",
    "kernel_dit_flash_sdpa_f32",
    "kernel_dit_flash_sdpa_causal_f32",
    "kernel_dit_flash_sdpa_d128_f32",
    "kernel_dit_flash_sdpa_causal_d128_f32",
    "kernel_dit_qk_norm_rope_f32",
    "kernel_dit_tanh_f32",
    "kernel_dit_gelu_tanh_f32",
];

const MODULE: &str = "diffusion";
/// Threads per elementwise / per-(token, head) threadgroup.
const BLOCK: u32 = 64;
/// Query rows per fused-attention threadgroup.
const FA_QROWS: u32 = 64;
/// Threads per fused-attention threadgroup.
const FA_THREADS: u32 = 256;

// Argument blocks: byte for byte `diffusion_common.h` and the driver's Pods.

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SdpaArgs {
    txt_tokens: u32,
    img_tokens: u32,
    cache_tokens: u32,
    heads: u32,
    head_dim: u32,
    txt_stride: u32,
    img_stride: u32,
    cache_stride: u32,
    q_offset: u32,
    k_offset: u32,
    v_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct QkNormRopeArgs {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    q_stride: u32,
    k_stride: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct UnaryArgs {
    count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GqaArgs {
    tokens: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    valid_keys: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RopeHalfArgs {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    pinned_from: u32,
    pinned_pos: u32,
}

const _: () = assert!(size_of::<GqaArgs>() == 20);
const _: () = assert!(size_of::<RopeHalfArgs>() == 20);
const _: () = assert!(size_of::<SdpaArgs>() == 44);
const _: () = assert!(size_of::<QkNormRopeArgs>() == 24);
const _: () = assert!(size_of::<UnaryArgs>() == 4);

fn launch(gpu: &Gpu, entry: &'static str, grid: Extent3, block: Extent3, args: &[KernelArg<'_>]) {
    run(
        gpu,
        KernelRef::new(MODULE, entry),
        &LaunchArgs {
            grid,
            block,
            smem_bytes: 0,
            args,
        },
    );
}

fn upload(gpu: &Gpu, values: &[f32]) -> DeviceBuffer {
    gpu.upload(&f32_bytes(values))
}

/// Asserts `|got - want| <= tol * max(|want|, floor)` element by element.
fn close(got: &[f32], want: &[f32], tol: f32, floor: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = tol * w.abs().max(floor);
        assert!(
            (g - w).abs() <= bound,
            "{what}: element {i} is {g}, expected {w} (tolerance {bound})"
        );
    }
}

/// Asserts two candidate answers differ somewhere by more than `tol`, so a
/// case that tells them apart proves something.
fn differs(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert!(
        a.iter().zip(b).any(|(&x, &y)| (x - y).abs() > tol),
        "{what}: the data does not tell the two readings apart"
    );
}

/// Packed `(rows, heads * hd)` attention in `f64`: row `t` attends keys
/// `0..=t` when `causal`, else every key.
fn attention_host(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: (usize, usize, usize),
    causal: bool,
) -> Vec<f32> {
    let (rows, heads, hd) = shape;
    let width = heads * hd;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut out = vec![0f32; rows * width];
    for t in 0..rows {
        for h in 0..heads {
            let seen = if causal { t + 1 } else { rows };
            let scores: Vec<f64> = (0..seen)
                .map(|s| {
                    (0..hd)
                        .map(|d| {
                            f64::from(q[t * width + h * hd + d])
                                * f64::from(k[s * width + h * hd + d])
                        })
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|&x| (x - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for d in 0..hd {
                let acc: f64 = exps
                    .iter()
                    .enumerate()
                    .map(|(s, &p)| p * f64::from(v[s * width + h * hd + d]))
                    .sum();
                out[t * width + h * hd + d] = (acc / sum) as f32;
            }
        }
    }
    out
}

/// One fused-attention launch over packed `txt` rows (no image rows, no
/// cache), the prefix's binding.
fn flash_txt_only(
    gpu: &Gpu,
    entry: &'static str,
    qkv: (&[f32], &[f32], &[f32]),
    shape: (usize, usize, usize),
) -> Vec<f32> {
    let (rows, heads, hd) = shape;
    let width = heads * hd;
    let (q, k, v) = (upload(gpu, qkv.0), upload(gpu, qkv.1), upload(gpu, qkv.2));
    let out = upload(gpu, &vec![0f32; rows * width]);
    let dummy = upload(gpu, &[0.0]);
    let args = SdpaArgs {
        txt_tokens: rows as u32,
        img_tokens: 0,
        cache_tokens: 0,
        heads: heads as u32,
        head_dim: hd as u32,
        txt_stride: width as u32,
        img_stride: width as u32,
        cache_stride: width as u32,
        q_offset: 0,
        k_offset: 0,
        v_offset: 0,
    };
    launch(
        gpu,
        entry,
        Extent3::new((rows as u32).div_ceil(FA_QROWS), heads as u32, 1),
        Extent3::linear(FA_THREADS),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(dummy.view()),
            KernelArg::Buffer(q.view()),
            KernelArg::Buffer(k.view()),
            KernelArg::Buffer(v.view()),
            KernelArg::Buffer(dummy.view()),
            KernelArg::Buffer(dummy.view()),
            KernelArg::Buffer(dummy.view()),
            KernelArg::Buffer(dummy.view()),
            KernelArg::Buffer(dummy.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// The causal entry attends each row to itself and the rows before it and
/// to nothing after: 70 rows straddle three query tiles (the last ragged),
/// so the diagonal block, the early stop and the ragged tail all run, on
/// the transformer's 128-wide heads and a narrower 64-wide one.
///
/// # Panics
///
/// When the device answer leaves the half-staging tolerance of the `f64`
/// oracle, or when the bidirectional entry's answer does not differ.
pub fn flash_sdpa_causal_masks_every_later_key(gpu: &Gpu) {
    for (heads, hd, seed) in [(2usize, 128usize, 11u32), (3, 64, 21)] {
        let rows = 70usize;
        let width = heads * hd;
        let scaled = |seed: u32| -> Vec<f32> {
            signal(rows * width, seed).iter().map(|v| v * 0.4).collect()
        };
        let (q, k, v) = (scaled(seed), scaled(seed + 1), scaled(seed + 2));
        let shape = (rows, heads, hd);
        let want = attention_host(&q, &k, &v, shape, true);
        // The 128-wide entries are the same kernel with the head width
        // fixed at compile time; they take only that width.
        let mut pairs = vec![(
            "kernel_dit_flash_sdpa_causal_f32",
            "kernel_dit_flash_sdpa_f32",
        )];
        if hd == 128 {
            pairs.push((
                "kernel_dit_flash_sdpa_causal_d128_f32",
                "kernel_dit_flash_sdpa_d128_f32",
            ));
        }
        for (causal, bidirectional) in pairs {
            let got = flash_txt_only(gpu, causal, (&q, &k, &v), shape);
            close(&got, &want, 3e-3, 0.05, causal);
            let full = flash_txt_only(gpu, bidirectional, (&q, &k, &v), shape);
            close(
                &full,
                &attention_host(&q, &k, &v, shape, false),
                3e-3,
                0.05,
                bidirectional,
            );
            differs(&got, &full, 1e-2, "causal vs bidirectional");
        }
    }
}

/// `x * rsqrt(mean(x^2) + eps) * scale`, then adjacent pairs turned by the
/// table, in `f64`; `half_split` pairs `(i, i + d/2)` instead (the reading
/// the kernel must not take).
fn norm_rope_host(
    row: &[f32],
    scale: &[f32],
    angles: (&[f32], &[f32]),
    eps: f32,
    half_split: bool,
) -> Vec<f32> {
    let d = row.len();
    let ms = row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / d as f64;
    let inv = 1.0 / (ms + f64::from(eps)).sqrt();
    let n: Vec<f64> = row
        .iter()
        .zip(scale)
        .map(|(&x, &s)| f64::from(x) * inv * f64::from(s))
        .collect();
    let mut out = vec![0f32; d];
    for p in 0..d / 2 {
        let (i0, i1) = if half_split {
            (p, p + d / 2)
        } else {
            (2 * p, 2 * p + 1)
        };
        let (c, s) = (f64::from(angles.0[p]), f64::from(angles.1[p]));
        out[i0] = (n[i0] * c - n[i1] * s) as f32;
        out[i1] = (n[i0] * s + n[i1] * c) as f32;
    }
    out
}

/// The fused `QK-norm` + table `RoPE` normalizes each head of q and of k
/// by its own scale vector, then turns adjacent pairs by the token's row
/// of the table. q is packed and k sits in wider rows (its own stride), so
/// a kernel reading k at q's stride fails; the bytes past each k head row
/// are left alone.
///
/// # Panics
///
/// When an element leaves 1e-5 of the `f64` oracle, the padding changed, or
/// the rotate-half reading would also have passed.
pub fn qk_norm_rope_normalizes_then_turns_adjacent_pairs(gpu: &Gpu) {
    let (tokens, heads, hd) = (5usize, 3usize, 128usize);
    let width = heads * hd;
    let k_stride = width + 40;
    let eps = 1e-6f32;
    let q: Vec<f32> = signal(tokens * width, 31).iter().map(|v| v * 3.0).collect();
    let k: Vec<f32> = signal(tokens * k_stride, 32);
    let q_scale: Vec<f32> = signal(hd, 33).iter().map(|v| 1.0 + 0.5 * v).collect();
    let k_scale: Vec<f32> = signal(hd, 34).iter().map(|v| 1.0 - 0.5 * v).collect();
    // Angles as a model's table would hold them: position times a falling
    // frequency, positions including negative ones.
    let half = hd / 2;
    let mut cos = vec![0f32; tokens * half];
    let mut sin = vec![0f32; tokens * half];
    for t in 0..tokens {
        for p in 0..half {
            let angle = (t as f64 - 2.0) * 10000f64.powf(-(2.0 * p as f64) / hd as f64) * 1.7;
            cos[t * half + p] = angle.cos() as f32;
            sin[t * half + p] = angle.sin() as f32;
        }
    }
    let (dq, dk) = (upload(gpu, &q), upload(gpu, &k));
    let (dqs, dks) = (upload(gpu, &q_scale), upload(gpu, &k_scale));
    let (dc, ds) = (upload(gpu, &cos), upload(gpu, &sin));
    let args = QkNormRopeArgs {
        tokens: tokens as u32,
        heads: heads as u32,
        head_dim: hd as u32,
        q_stride: width as u32,
        k_stride: k_stride as u32,
        eps,
    };
    launch(
        gpu,
        "kernel_dit_qk_norm_rope_f32",
        Extent3::new(tokens as u32, heads as u32, 1),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(dq.view()),
            KernelArg::Buffer(dk.view()),
            KernelArg::Buffer(dqs.view()),
            KernelArg::Buffer(dks.view()),
            KernelArg::Buffer(dc.view()),
            KernelArg::Buffer(ds.view()),
        ],
    );
    drain(gpu);
    let (got_q, got_k) = (gpu.read_f32(&dq), gpu.read_f32(&dk));
    let mut want_q = Vec::new();
    let mut half_q = Vec::new();
    for t in 0..tokens {
        let table = (
            &cos[t * half..(t + 1) * half],
            &sin[t * half..(t + 1) * half],
        );
        for h in 0..heads {
            let row = &q[t * width + h * hd..t * width + (h + 1) * hd];
            want_q.extend(norm_rope_host(row, &q_scale, table, eps, false));
            half_q.extend(norm_rope_host(row, &q_scale, table, eps, true));
            let at = t * k_stride + h * hd;
            let want_k = norm_rope_host(&k[at..at + hd], &k_scale, table, eps, false);
            close(&got_k[at..at + hd], &want_k, 1e-5, 1.0, "k head");
        }
        let pad = t * k_stride + width..(t + 1) * k_stride;
        assert_eq!(&got_k[pad.clone()], &k[pad], "k row padding is untouched");
    }
    close(&got_q, &want_q, 1e-5, 1.0, "q heads");
    differs(&want_q, &half_q, 1e-2, "adjacent vs rotate-half pairs");
}

/// `tanh` and the tanh form of `GELU` against their formulas in `f64`,
/// over a range that reaches both saturated tails.
///
/// # Panics
///
/// When an element leaves 2e-6 (relative, floor 1) of the formula.
pub fn tanh_and_gelu_tanh_match_their_formulas(gpu: &Gpu) {
    let count = 1000usize;
    let x: Vec<f32> = signal(count, 41).iter().map(|v| v * 12.0).collect();
    let input = upload(gpu, &x);
    let run_unary = |entry: &'static str| -> Vec<f32> {
        let out = upload(gpu, &vec![0f32; count]);
        let args = UnaryArgs {
            count: count as u32,
        };
        launch(
            gpu,
            entry,
            Extent3::linear((count as u32).div_ceil(BLOCK)),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(input.view()),
            ],
        );
        drain(gpu);
        gpu.read_f32(&out)
    };
    let tanh: Vec<f32> = x.iter().map(|&v| f64::from(v).tanh() as f32).collect();
    close(&run_unary("kernel_dit_tanh_f32"), &tanh, 2e-6, 1.0, "tanh");
    let gelu: Vec<f32> = x
        .iter()
        .map(|&v| {
            let v = f64::from(v);
            let inner = (2.0 / std::f64::consts::PI).sqrt() * (v + 0.044_715 * v.powi(3));
            (0.5 * v * (1.0 + inner.tanh())) as f32
        })
        .collect();
    close(
        &run_unary("kernel_dit_gelu_tanh_f32"),
        &gelu,
        2e-6,
        1.0,
        "gelu",
    );
}

/// Grouped-query attention in `f64` over packed `(rows, heads * hd)` q and
/// `(rows, kv_heads * hd)` k/v: row `t` attends key `s` when `s <= t` and
/// `s < valid`.
fn gqa_host(
    qkv: (&[f32], &[f32], &[f32]),
    shape: (usize, usize, usize, usize),
    valid: usize,
) -> Vec<f32> {
    let (q, k, v) = qkv;
    let (rows, q_heads, kv_heads, hd) = shape;
    let group = q_heads / kv_heads;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut out = vec![0f32; rows * q_heads * hd];
    for t in 0..rows {
        for h in 0..q_heads {
            let kvh = h / group;
            let keys: Vec<usize> = (0..=t).filter(|&s| s < valid).collect();
            let scores: Vec<f64> = keys
                .iter()
                .map(|&s| {
                    (0..hd)
                        .map(|d| {
                            f64::from(q[(t * q_heads + h) * hd + d])
                                * f64::from(k[(s * kv_heads + kvh) * hd + d])
                        })
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|&x| (x - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for d in 0..hd {
                let acc: f64 = keys
                    .iter()
                    .zip(&exps)
                    .map(|(&s, &p)| p * f64::from(v[(s * kv_heads + kvh) * hd + d]))
                    .sum();
                out[(t * q_heads + h) * hd + d] = (acc / sum) as f32;
            }
        }
    }
    out
}

fn gqa_device(
    gpu: &Gpu,
    qkv: (&[f32], &[f32], &[f32]),
    shape: (usize, usize, usize, usize),
    valid: usize,
) -> Vec<f32> {
    let (rows, q_heads, kv_heads, hd) = shape;
    let (q, k, v) = (upload(gpu, qkv.0), upload(gpu, qkv.1), upload(gpu, qkv.2));
    let out = upload(gpu, &vec![0f32; rows * q_heads * hd]);
    let args = GqaArgs {
        tokens: rows as u32,
        q_heads: q_heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: hd as u32,
        valid_keys: valid as u32,
    };
    launch(
        gpu,
        "kernel_dit_gqa_f32",
        Extent3::new(rows as u32, q_heads as u32, 1),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(q.view()),
            KernelArg::Buffer(k.view()),
            KernelArg::Buffer(v.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// The text encoder's padded prefill: rows at or past `valid_keys` are pad
/// rows that attend only the real keys (`s < valid`, still causal), rows
/// before it are untouched, and `valid_keys == tokens` is the plain causal
/// answer. The pad rows' answer is asserted to differ from the unmasked
/// causal one, so a kernel that ignored the limit could not pass.
///
/// # Panics
///
/// When the device answer leaves 1e-4 of the `f64` oracle, or the data does
/// not tell the masked and unmasked readings apart.
pub fn gqa_valid_keys_masks_pad_keys(gpu: &Gpu) {
    let (rows, q_heads, kv_heads, hd) = (9usize, 4usize, 2usize, 16usize);
    let valid = 5usize;
    let scaled = |n: usize, seed: u32| -> Vec<f32> {
        signal(rows * n * hd, seed)
            .iter()
            .map(|v| v * 0.8)
            .collect()
    };
    let (q, k, v) = (
        scaled(q_heads, 61),
        scaled(kv_heads, 62),
        scaled(kv_heads, 63),
    );
    let shape = (rows, q_heads, kv_heads, hd);
    let masked = gqa_device(gpu, (&q, &k, &v), shape, valid);
    close(
        &masked,
        &gqa_host((&q, &k, &v), shape, valid),
        1e-4,
        1e-3,
        "gqa valid_keys",
    );
    // The real rows are exactly the plain causal answer.
    let unmasked = gqa_device(gpu, (&q, &k, &v), shape, rows);
    let real = valid * q_heads * hd;
    close(
        &masked[..real],
        &unmasked[..real],
        1e-5,
        1e-3,
        "gqa real rows",
    );
    close(
        &unmasked,
        &gqa_host((&q, &k, &v), shape, rows),
        1e-4,
        1e-3,
        "gqa valid_keys == tokens",
    );
    differs(
        &masked[real..],
        &unmasked[real..],
        1e-3,
        "gqa pad rows masked vs unmasked",
    );
}

/// Rotate-half `RoPE` with the trailing rows pinned to one position: rows
/// before `pinned_from` turn by their own index, rows from it on by
/// `pinned_pos` (the padded text prefill rotates pad rows at position 1).
///
/// # Panics
///
/// When a row leaves 1e-5 of the `f64` oracle, or pinning changed nothing.
pub fn rope_half_pins_trailing_rows_to_one_position(gpu: &Gpu) {
    let (tokens, heads, hd) = (6usize, 3usize, 8usize);
    let pinned_from = 4usize;
    let x = signal(tokens * heads * hd, 71);
    let freqs: Vec<f32> = (0..hd / 2)
        .map(|i| 10000f32.powf(-(2.0 * i as f32) / hd as f32))
        .collect();
    let run_with = |pinned_from: usize| -> Vec<f32> {
        let dx = upload(gpu, &x);
        let df = upload(gpu, &freqs);
        let args = RopeHalfArgs {
            tokens: tokens as u32,
            heads: heads as u32,
            head_dim: hd as u32,
            pinned_from: pinned_from as u32,
            pinned_pos: 1,
        };
        launch(
            gpu,
            "kernel_dit_rope_half",
            Extent3::new(tokens as u32, heads as u32, 1),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(dx.view()),
                KernelArg::Buffer(df.view()),
            ],
        );
        drain(gpu);
        gpu.read_f32(&dx)
    };
    let host = |pinned_from: usize| -> Vec<f32> {
        let mut out = x.clone();
        for t in 0..tokens {
            let position = if t >= pinned_from { 1.0 } else { t as f64 };
            for h in 0..heads {
                let row = (t * heads + h) * hd;
                for i in 0..hd / 2 {
                    let angle = position * f64::from(freqs[i]);
                    let (c, s) = (angle.cos(), angle.sin());
                    let (x0, x1) = (f64::from(x[row + i]), f64::from(x[row + i + hd / 2]));
                    out[row + i] = (x0 * c - x1 * s) as f32;
                    out[row + i + hd / 2] = (x1 * c + x0 * s) as f32;
                }
            }
        }
        out
    };
    let pinned = run_with(pinned_from);
    close(&pinned, &host(pinned_from), 1e-5, 1e-3, "rope_half pinned");
    let plain = run_with(tokens);
    close(&plain, &host(tokens), 1e-5, 1e-3, "rope_half unpinned");
    differs(&pinned, &plain, 1e-3, "rope_half pinned vs unpinned");
}
