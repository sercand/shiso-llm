//! Cases for the routed feed-forward over row-major affine stacks and the
//! plain attention store (`dense_moe.metal`).
//!
//! The router is graded exactly on its choice and within a float tolerance on
//! its weights; the expert stage against the host product of the dequantized
//! rows, in both of its input conventions and with and without the squared
//! rectifier; the store exactly, since it only narrows to f16.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "fixture dimensions and ids are small non-negative integers"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

/// Every entry point this module's cases launch.
pub const ENTRIES: [&str; 6] = [
    "kernel_dense_moe_route_sigmoid_f32",
    "kernel_dense_moe_route_sort_sigmoid_f32",
    "kernel_dense_moe_combine_add_f32",
    "kernel_dense_experts_affine4_f32",
    "kernel_dense_relu2_f32",
    "kernel_dense_kv_store_f32",
];

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RouteArgs {
    experts: u32,
    used: u32,
    scale: f32,
    norm: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ExpertArgs {
    in_dim: u32,
    out_dim: u32,
    n_expert: u32,
    n_used: u32,
    by_token: u32,
    act: u32,
    pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FlatArgs {
    n: u32,
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct KvArgs {
    head_dim: u32,
    n_head_kv: u32,
    n_rot: u32,
    freq_dims: u32,
    n_tok: u32,
    rope_base: f32,
    eps: f32,
    page_tokens: u32,
    v_norm: u32,
    page_stride: u32,
}

fn signal(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f32 / 500.0 - 1.0)
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().copied().flat_map(i32::to_le_bytes).collect()
}

fn i32_values(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn launch(
    gpu: &Gpu,
    entry: &'static str,
    grid: Extent3,
    block: u32,
    smem: u32,
    args: &[KernelArg<'_>],
) {
    run(
        gpu,
        KernelRef::new("dense", entry),
        &LaunchArgs {
            grid,
            block: Extent3::linear(block),
            smem_bytes: smem,
            args,
        },
    );
    drain(gpu);
}

/// The choice follows the biased score, the weight the unbiased one.
pub fn route_sigmoid_chooses_by_biased_score(gpu: &Gpu) {
    let (n_tok, experts, used, scale) = (3usize, 40usize, 4usize, 2.5f32);
    let logits = signal(n_tok * experts, 51);
    let bias: Vec<f32> = signal(experts, 52).iter().map(|v| v * 0.3).collect();
    let ids_buf = gpu.upload(&i32_bytes(&vec![0; n_tok * used]));
    let w_buf = gpu.upload(&f32_bytes(&vec![0.0; n_tok * used]));
    let l_buf = gpu.upload(&f32_bytes(&logits));
    let b_buf = gpu.upload(&f32_bytes(&bias));
    let args = RouteArgs {
        experts: experts as u32,
        used: used as u32,
        scale,
        norm: 1,
    };
    launch(
        gpu,
        "kernel_dense_moe_route_sigmoid_f32",
        Extent3::linear(n_tok as u32),
        128,
        (2 * experts * 4) as u32,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(ids_buf.view()),
            KernelArg::Buffer(w_buf.view()),
            KernelArg::Buffer(l_buf.view()),
            KernelArg::Buffer(b_buf.view()),
        ],
    );
    let ids = i32_values(&gpu.read(&ids_buf));
    let weights = gpu.read_f32(&w_buf);
    for t in 0..n_tok {
        let score: Vec<f32> = (0..experts)
            .map(|e| 1.0 / (1.0 + (-logits[t * experts + e]).exp()))
            .collect();
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| {
            (score[b] + bias[b])
                .total_cmp(&(score[a] + bias[a]))
                .then(a.cmp(&b))
        });
        let chosen = &order[..used];
        let total: f32 = chosen.iter().map(|&e| score[e]).sum();
        for (k, &e) in chosen.iter().enumerate() {
            assert_eq!(ids[t * used + k], e as i32, "token {t} slot {k}");
            let want = score[e] / (total + 1e-20) * scale;
            let have = weights[t * used + k];
            assert!(
                (have - want).abs() <= 1e-5 * (1.0 + want.abs()),
                "token {t} slot {k}: {have} against {want}"
            );
        }
    }
}

/// The fused router and sort: the same choice as the router alone, and a
/// column map grouped by expert in column order.
pub fn route_sort_groups_columns_by_expert(gpu: &Gpu) {
    let (n_tok, experts, used, scale) = (5usize, 40usize, 4usize, 2.5f32);
    let columns = n_tok * used;
    let logits = signal(n_tok * experts, 55);
    let bias: Vec<f32> = signal(experts, 56).iter().map(|v| v * 0.3).collect();
    let args = RouteArgs {
        experts: experts as u32,
        used: used as u32,
        scale,
        norm: 1,
    };
    let l_buf = gpu.upload(&f32_bytes(&logits));
    let b_buf = gpu.upload(&f32_bytes(&bias));
    // The plain router, as the oracle for the choice and the weights.
    let want_ids = gpu.upload(&i32_bytes(&vec![0; columns]));
    let want_w = gpu.upload(&f32_bytes(&vec![0.0; columns]));
    launch(
        gpu,
        "kernel_dense_moe_route_sigmoid_f32",
        Extent3::linear(n_tok as u32),
        128,
        (2 * experts * 4) as u32,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(want_ids.view()),
            KernelArg::Buffer(want_w.view()),
            KernelArg::Buffer(l_buf.view()),
            KernelArg::Buffer(b_buf.view()),
        ],
    );
    let ids_buf = gpu.upload(&i32_bytes(&vec![0; columns]));
    let w_buf = gpu.upload(&f32_bytes(&vec![0.0; columns]));
    let sorted_buf = gpu.upload(&i32_bytes(&vec![-1; columns]));
    let bounds_buf = gpu.upload(&i32_bytes(&vec![-1; experts + 1]));
    let active_buf = gpu.upload(&i32_bytes(&vec![-1; experts + 1]));
    launch(
        gpu,
        "kernel_dense_moe_route_sort_sigmoid_f32",
        Extent3::linear(1),
        128,
        (3 * experts * 4) as u32,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(ids_buf.view()),
            KernelArg::Buffer(w_buf.view()),
            KernelArg::Buffer(l_buf.view()),
            KernelArg::Buffer(b_buf.view()),
            KernelArg::Buffer(sorted_buf.view()),
            KernelArg::Buffer(bounds_buf.view()),
            KernelArg::Buffer(active_buf.view()),
            KernelArg::U32(n_tok as u32),
        ],
    );
    let ids = i32_values(&gpu.read(&ids_buf));
    assert_eq!(ids, i32_values(&gpu.read(&want_ids)), "the fused choice");
    let (w, want) = (gpu.read_f32(&w_buf), gpu.read_f32(&want_w));
    for (c, (a, b)) in w.iter().zip(&want).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "column {c} weight");
    }
    let mut sorted = Vec::new();
    let mut bounds = vec![0i32; experts + 1];
    let mut active = Vec::new();
    for e in 0..experts {
        let cols: Vec<i32> = (0..columns as i32)
            .filter(|&c| ids[c as usize] == e as i32)
            .collect();
        if !cols.is_empty() {
            active.push(e as i32);
        }
        sorted.extend_from_slice(&cols);
        bounds[e + 1] = sorted.len() as i32;
    }
    assert_eq!(i32_values(&gpu.read(&sorted_buf)), sorted, "sorted columns");
    assert_eq!(i32_values(&gpu.read(&bounds_buf)), bounds, "expert bounds");
    let got_active = i32_values(&gpu.read(&active_buf));
    assert_eq!(got_active[experts], active.len() as i32, "active count");
    assert_eq!(&got_active[..active.len()], &active[..], "active experts");
}

/// `out = shared + sum_k w_k * down_k`, per token.
pub fn combine_add_folds_the_shared_row_in(gpu: &Gpu) {
    let (n_tok, used, hidden) = (3usize, 4usize, 300usize);
    let columns = n_tok * used;
    let shared = signal(n_tok * hidden, 57);
    let down = signal(columns * hidden, 58);
    let w = signal(columns, 59);
    let out = gpu.upload(&f32_bytes(&vec![0.0; n_tok * hidden]));
    let s_buf = gpu.upload(&f32_bytes(&shared));
    let d_buf = gpu.upload(&f32_bytes(&down));
    let w_buf = gpu.upload(&f32_bytes(&w));
    let args = RouteArgs {
        experts: 0,
        used: used as u32,
        scale: 1.0,
        norm: 0,
    };
    launch(
        gpu,
        "kernel_dense_moe_combine_add_f32",
        Extent3::new((hidden as u32).div_ceil(256), n_tok as u32, 1),
        256,
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(s_buf.view()),
            KernelArg::Buffer(d_buf.view()),
            KernelArg::Buffer(w_buf.view()),
            KernelArg::U32(hidden as u32),
        ],
    );
    let got = gpu.read_f32(&out);
    for t in 0..n_tok {
        for e in 0..hidden {
            let mut acc = 0.0f32;
            for k in 0..used {
                acc += w[t * used + k] * down[(t * used + k) * hidden + e];
            }
            let want = shared[t * hidden + e] + acc;
            let have = got[t * hidden + e];
            assert!(
                (have - want).abs() <= 1e-5 * (1.0 + want.abs()),
                "token {t} value {e}: {have} against {want}"
            );
        }
    }
}

/// A bf16-exact value near `v`.
fn bf16_round(v: f32) -> (f32, u16) {
    let bits = (v.to_bits() >> 16) as u16;
    (f32::from_bits(u32::from(bits) << 16), bits)
}

/// Both stage conventions and both activations against the host product.
#[expect(clippy::too_many_lines, reason = "one fixture, two stage conventions")]
pub fn experts_affine4_matches_the_host_product(gpu: &Gpu) {
    let (experts, in_dim, out_dim, n_tok, used) = (4usize, 192usize, 10usize, 3usize, 2usize);
    let groups = in_dim / 64;
    let codes: Vec<u8> = (0..experts * out_dim * in_dim)
        .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 7) as u8 & 0xf)
        .collect();
    let words: Vec<u32> = codes
        .chunks_exact(8)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u32, |acc, (i, &q)| acc | (u32::from(q) << (4 * i)))
        })
        .collect();
    let (scales, sbits): (Vec<f32>, Vec<u16>) = signal(experts * out_dim * groups, 61)
        .iter()
        .map(|v| bf16_round(v * 0.05))
        .unzip();
    let (biases, bbits): (Vec<f32>, Vec<u16>) = signal(experts * out_dim * groups, 62)
        .iter()
        .map(|v| bf16_round(v * 0.1))
        .unzip();
    // Tokens 0..3, two slots each; expert 2 receives nothing.
    let ids = [0i32, 3, 3, 1, 0, 3];
    let columns = n_tok * used;
    let mut sorted = Vec::new();
    let mut bounds = vec![0i32; experts + 1];
    let mut active = Vec::new();
    for e in 0..experts {
        let cols: Vec<i32> = (0..columns as i32)
            .filter(|&c| ids[c as usize] == e as i32)
            .collect();
        if !cols.is_empty() {
            active.push(e as i32);
        }
        sorted.extend_from_slice(&cols);
        bounds[e + 1] = sorted.len() as i32;
    }
    let n_active = active.len() as i32;
    active.resize(experts, 0);
    active.push(n_active);

    let w_buf = gpu.upload(bytemuck::cast_slice(&words));
    let s_buf = gpu.upload(bytemuck::cast_slice(&sbits));
    let b_buf = gpu.upload(bytemuck::cast_slice(&bbits));
    let sorted_buf = gpu.upload(&i32_bytes(&sorted));
    let bounds_buf = gpu.upload(&i32_bytes(&bounds));
    let active_buf = gpu.upload(&i32_bytes(&active));
    for (by_token, act) in [(1u32, 1u32), (0, 0)] {
        let rows = if by_token == 1 { n_tok } else { columns };
        let x = signal(rows * in_dim, 63 + by_token);
        let x_buf = gpu.upload(&f32_bytes(&x));
        let out_buf = gpu.upload(&f32_bytes(&vec![0.0; columns * out_dim]));
        let args = ExpertArgs {
            in_dim: in_dim as u32,
            out_dim: out_dim as u32,
            n_expert: experts as u32,
            n_used: used as u32,
            by_token,
            act,
            pad: [0; 2],
        };
        launch(
            gpu,
            "kernel_dense_experts_affine4_f32",
            Extent3::new(
                (out_dim as u32).div_ceil(16),
                columns.min(experts) as u32,
                1,
            ),
            128,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w_buf.view()),
                KernelArg::Buffer(s_buf.view()),
                KernelArg::Buffer(b_buf.view()),
                KernelArg::Buffer(x_buf.view()),
                KernelArg::Buffer(sorted_buf.view()),
                KernelArg::Buffer(bounds_buf.view()),
                KernelArg::Buffer(active_buf.view()),
                KernelArg::Buffer(out_buf.view()),
            ],
        );
        let got = gpu.read_f32(&out_buf);
        for col in 0..columns {
            let e = ids[col] as usize;
            let xr = if by_token == 1 { col / used } else { col };
            for r in 0..out_dim {
                let at = e * out_dim + r;
                let mut acc = 0.0f32;
                for k in 0..in_dim {
                    let g = k / 64;
                    let q = f32::from(codes[at * in_dim + k]);
                    acc += (q * scales[at * groups + g] + biases[at * groups + g])
                        * x[xr * in_dim + k];
                }
                let want = if act == 1 { acc.max(0.0).powi(2) } else { acc };
                let have = got[col * out_dim + r];
                assert!(
                    (have - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "by_token {by_token} column {col} row {r}: {have} against {want}"
                );
            }
        }
    }
}

/// `max(x, 0)^2`, elementwise.
pub fn relu2_squares_the_rectified_value(gpu: &Gpu) {
    let n = 1000usize;
    let x = signal(n, 71);
    let src = gpu.upload(&f32_bytes(&x));
    let dst = gpu.upload(&f32_bytes(&vec![0.0; n]));
    let args = FlatArgs {
        n: n as u32,
        pad: [0; 3],
    };
    launch(
        gpu,
        "kernel_dense_relu2_f32",
        Extent3::linear((n as u32).div_ceil(256)),
        256,
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(src.view()),
            KernelArg::Buffer(dst.view()),
        ],
    );
    let got = gpu.read_f32(&dst);
    for (i, (have, v)) in got.iter().zip(&x).enumerate() {
        let r = v.max(0.0);
        assert_eq!(have.to_bits(), (r * r).to_bits(), "element {i}");
    }
}

/// Keys and values land, narrowed to f16, in the rows the page table names.
pub fn kv_store_writes_the_paged_rows(gpu: &Gpu) {
    let (head_dim, heads, n_tok, page_tokens) = (128usize, 2usize, 3usize, 4usize);
    let pos = [5i32, 6, 9];
    let pages = [2i32, 0, 1];
    let rows = pages.len() * page_tokens;
    let k = signal(n_tok * heads * head_dim, 81);
    let v = signal(n_tok * heads * head_dim, 82);
    let zeros = vec![0u8; rows * heads * head_dim * 2];
    let kc = gpu.upload(&zeros);
    let vc = gpu.upload(&zeros);
    let k_buf = gpu.upload(&f32_bytes(&k));
    let v_buf = gpu.upload(&f32_bytes(&v));
    let pos_buf = gpu.upload(&i32_bytes(&pos));
    let pages_buf = gpu.upload(&i32_bytes(&pages));
    let args = KvArgs {
        head_dim: head_dim as u32,
        n_head_kv: heads as u32,
        n_rot: 0,
        freq_dims: 0,
        n_tok: n_tok as u32,
        rope_base: 0.0,
        eps: 0.0,
        page_tokens: page_tokens as u32,
        v_norm: 0,
        page_stride: 0,
    };
    launch(
        gpu,
        "kernel_dense_kv_store_f32",
        Extent3::new(heads as u32, n_tok as u32, 1),
        128,
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(kc.view()),
            KernelArg::Buffer(vc.view()),
            KernelArg::Buffer(k_buf.view()),
            KernelArg::Buffer(v_buf.view()),
            KernelArg::Buffer(pos_buf.view()),
            KernelArg::Buffer(pages_buf.view()),
        ],
    );
    let halves = |bytes: Vec<u8>| -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let kh = halves(gpu.read(&kc));
    let vh = halves(gpu.read(&vc));
    for (t, &at) in pos.iter().enumerate() {
        let p = at as usize;
        let row = pages[p / page_tokens] as usize * page_tokens + p % page_tokens;
        for h in 0..heads {
            for i in 0..head_dim {
                let src = (t * heads + h) * head_dim + i;
                let dst = (row * heads + h) * head_dim + i;
                assert_eq!(
                    kh[dst],
                    half::f16::from_f32(k[src]).to_bits(),
                    "key token {t} head {h} dim {i}"
                );
                assert_eq!(
                    vh[dst],
                    half::f16::from_f32(v[src]).to_bits(),
                    "value token {t} head {h} dim {i}"
                );
            }
        }
    }
}
