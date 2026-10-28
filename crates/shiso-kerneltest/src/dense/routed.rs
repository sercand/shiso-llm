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
use shiso_modelfile::quant::BlockIq4Nl;
use shiso_modelfile::quant_tables::IQ4NL_VALUES;

/// Every entry point this module's cases launch.
pub const ENTRIES: [&str; 8] = [
    "kernel_dense_moe_route_sigmoid_f32",
    "kernel_dense_moe_route_sort_sigmoid_f32",
    "kernel_dense_moe_combine_add_f32",
    "kernel_dense_experts_affine4_f32",
    "kernel_dense_experts_iq4nl_f32",
    "kernel_dense_experts_iq4nl_gemm_f32",
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

/// An `IQ4_NL` routed-stage fixture at the decoder's real input width: 2688
/// inputs is 84 blocks a row, which no lane count divides, so a lane walk that
/// assumed it would drop or repeat a block.
struct Iq4NlFixture {
    experts: usize,
    in_dim: usize,
    out_dim: usize,
    n_tok: usize,
    used: usize,
    /// The stack's blocks, `[experts][out_dim][in_dim / 32]`.
    blocks: Vec<BlockIq4Nl>,
    /// Each column's expert.
    ids: Vec<i32>,
    sorted: Vec<i32>,
    bounds: Vec<i32>,
    active: Vec<i32>,
}

impl Iq4NlFixture {
    fn new() -> Self {
        Self::with_tokens(5, 20)
    }

    /// `n_tok` tokens over `out_dim` rows a projection: the routing of the
    /// first five tokens is the same whatever `n_tok` is, so their columns can
    /// be compared across fixtures.
    fn with_tokens(n_tok: usize, out_dim: usize) -> Self {
        let (experts, in_dim, used) = (8usize, 2688usize, 6usize);
        let per_row = in_dim / 32;
        let scales = signal(experts * out_dim * per_row, 71);
        let blocks: Vec<BlockIq4Nl> = (0..experts * out_dim * per_row)
            .map(|b| {
                let mut qs = [0u8; 16];
                for (j, q) in qs.iter_mut().enumerate() {
                    *q = (((b * 16 + j) as u32).wrapping_mul(2_654_435_761) >> 11) as u8;
                }
                BlockIq4Nl {
                    d: half::f16::from_f32(scales[b] * 0.01).to_bits(),
                    qs,
                }
            })
            .collect();
        // Five tokens, six distinct experts each out of eight; expert 7 is
        // never chosen, so the active list is shorter than the bank.
        let ids: Vec<i32> = (0..n_tok * used)
            .map(|c| ((c / used + c % used) % (experts - 1)) as i32)
            .collect();
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
        Self {
            experts,
            in_dim,
            out_dim,
            n_tok,
            used,
            blocks,
            ids,
            sorted,
            bounds,
            active,
        }
    }

    /// Row `row` of expert `expert`, dequantized on the host in f64.
    fn row(&self, expert: usize, row: usize) -> Vec<f64> {
        let per_row = self.in_dim / 32;
        let first = (expert * self.out_dim + row) * per_row;
        self.blocks[first..first + per_row]
            .iter()
            .flat_map(|b| {
                let d = f64::from(half::f16::from_bits(b.d).to_f32());
                let mut v = [0.0f64; 32];
                for j in 0..16 {
                    v[j] = d * f64::from(IQ4NL_VALUES[usize::from(b.qs[j] & 0xf)]);
                    v[j + 16] = d * f64::from(IQ4NL_VALUES[usize::from(b.qs[j] >> 4)]);
                }
                v
            })
            .collect()
    }

    /// Runs the stage over `stack` (whole, or a slot pool when `map` is
    /// given) and returns the `[columns, out_dim]` output.
    fn run(&self, gpu: &Gpu, stack: &[u8], map: Option<&[i32]>, act: u32, x: &[f32]) -> Vec<f32> {
        self.run_entry(
            gpu,
            "kernel_dense_experts_iq4nl_f32",
            16,
            stack,
            map,
            act,
            x,
        )
    }

    /// As [`Iq4NlFixture::run`], through `entry`, whose threadgroup covers
    /// `rows_per_group` output rows.
    #[expect(
        clippy::too_many_arguments,
        reason = "a test fixture's one launch helper"
    )]
    fn run_entry(
        &self,
        gpu: &Gpu,
        entry: &'static str,
        rows_per_group: u32,
        stack: &[u8],
        map: Option<&[i32]>,
        act: u32,
        x: &[f32],
    ) -> Vec<f32> {
        let columns = self.n_tok * self.used;
        let w_buf = gpu.upload(stack);
        let map_buf = gpu.upload(&i32_bytes(map.unwrap_or(&[0])));
        let x_buf = gpu.upload(&f32_bytes(x));
        let sorted_buf = gpu.upload(&i32_bytes(&self.sorted));
        let bounds_buf = gpu.upload(&i32_bytes(&self.bounds));
        let active_buf = gpu.upload(&i32_bytes(&self.active));
        let out_buf = gpu.upload(&f32_bytes(&vec![0.0; columns * self.out_dim]));
        let args = ExpertArgs {
            in_dim: self.in_dim as u32,
            out_dim: self.out_dim as u32,
            n_expert: self.experts as u32,
            n_used: self.used as u32,
            by_token: 1,
            act,
            // `mapped`, then `layer_base`: the map below is one layer's, so
            // its base is zero.
            pad: [u32::from(map.is_some()), 0],
        };
        launch(
            gpu,
            entry,
            Extent3::new(
                (self.out_dim as u32).div_ceil(rows_per_group),
                columns.min(self.experts) as u32,
                1,
            ),
            128,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(w_buf.view()),
                KernelArg::Buffer(map_buf.view()),
                KernelArg::Buffer(x_buf.view()),
                KernelArg::Buffer(sorted_buf.view()),
                KernelArg::Buffer(bounds_buf.view()),
                KernelArg::Buffer(active_buf.view()),
                KernelArg::Buffer(out_buf.view()),
            ],
        );
        gpu.read_f32(&out_buf)
    }

    fn input(&self) -> Vec<f32> {
        signal(self.n_tok * self.in_dim, 73)
    }
}

/// The `IQ4_NL` stage against the host product of the dequantized rows.
pub fn iq4nl_expert_stage_matches_host_dequant(gpu: &Gpu) {
    let f = Iq4NlFixture::new();
    let x = f.input();
    let got = f.run(gpu, bytemuck::cast_slice(&f.blocks), None, 0, &x);
    let mut worst = 0.0f64;
    let mut peak = 0.0f64;
    let mut wanted = vec![0.0f64; got.len()];
    for (col, &e) in f.ids.iter().enumerate() {
        let token = col / f.used;
        for r in 0..f.out_dim {
            let w = f.row(e as usize, r);
            let want: f64 = w
                .iter()
                .zip(&x[token * f.in_dim..(token + 1) * f.in_dim])
                .map(|(a, b)| a * f64::from(*b))
                .sum();
            wanted[col * f.out_dim + r] = want;
            peak = peak.max(want.abs());
        }
    }
    for (have, want) in got.iter().zip(&wanted) {
        worst = worst.max((f64::from(*have) - want).abs());
    }
    assert!(
        worst <= 1e-3 * peak,
        "worst error {worst} against a peak of {peak}"
    );
}

/// A stage through a slot map reads expert `e` from slot `map[e]`, and is
/// bit-identical to the same stage over the whole stack.
pub fn iq4nl_expert_stage_reads_through_the_slot_map(gpu: &Gpu) {
    let f = Iq4NlFixture::new();
    let x = f.input();
    let whole = f.run(gpu, bytemuck::cast_slice(&f.blocks), None, 0, &x);
    let expert_blocks = f.out_dim * f.in_dim / 32;
    let slots = 12usize;
    // Expert e lives in slot (5 * e + 3) % 12; the other slots hold noise.
    let map: Vec<i32> = (0..f.experts)
        .map(|e| ((5 * e + 3) % slots) as i32)
        .collect();
    let noise = BlockIq4Nl {
        d: half::f16::from_f32(7.0).to_bits(),
        qs: [0x5a; 16],
    };
    let mut pool = vec![noise; slots * expert_blocks];
    for (e, &slot) in map.iter().enumerate() {
        let slot = slot as usize;
        pool[slot * expert_blocks..(slot + 1) * expert_blocks]
            .copy_from_slice(&f.blocks[e * expert_blocks..(e + 1) * expert_blocks]);
    }
    let mapped = f.run(gpu, bytemuck::cast_slice(&pool), Some(&map), 0, &x);
    assert!(
        whole
            .iter()
            .zip(&mapped)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the mapped stage differs from the whole-stack stage"
    );
}

/// `act` 1 stores `max(v, 0)^2` of exactly the value `act` 0 stores.
pub fn iq4nl_expert_stage_relu2_is_fused(gpu: &Gpu) {
    let f = Iq4NlFixture::new();
    let x = f.input();
    let stack: &[u8] = bytemuck::cast_slice(&f.blocks);
    let plain = f.run(gpu, stack, None, 0, &x);
    let squared = f.run(gpu, stack, None, 1, &x);
    for (i, (p, s)) in plain.iter().zip(&squared).enumerate() {
        let want = p.max(0.0) * p.max(0.0);
        assert_eq!(
            s.to_bits(),
            want.to_bits(),
            "element {i}: {s} against {want}"
        );
    }
}

/// The tiled `IQ4_NL` stage a prompt pass takes, run over `f`.
fn gemm(
    f: &Iq4NlFixture,
    gpu: &Gpu,
    stack: &[u8],
    map: Option<&[i32]>,
    act: u32,
    x: &[f32],
) -> Vec<f32> {
    f.run_entry(
        gpu,
        "kernel_dense_experts_iq4nl_gemm_f32",
        64,
        stack,
        map,
        act,
        x,
    )
}

/// The tiled stage against the host product of the dequantized rows, with a
/// row count that leaves the last 64-row tile ragged and an expert with more
/// columns than one 16-column tile.
pub fn iq4nl_expert_gemm_matches_host_dequant(gpu: &Gpu) {
    let f = Iq4NlFixture::with_tokens(40, 100);
    let x = signal(f.n_tok * f.in_dim, 73);
    for act in [0u32, 1] {
        let got = gemm(&f, gpu, bytemuck::cast_slice(&f.blocks), None, act, &x);
        let mut peak = 0.0f64;
        let mut worst = 0.0f64;
        for (col, &e) in f.ids.iter().enumerate() {
            let token = col / f.used;
            for r in 0..f.out_dim {
                let w = f.row(e as usize, r);
                let dot: f64 = w
                    .iter()
                    .zip(&x[token * f.in_dim..(token + 1) * f.in_dim])
                    .map(|(a, b)| a * f64::from(*b))
                    .sum();
                let want = if act == 1 { dot.max(0.0).powi(2) } else { dot };
                peak = peak.max(want.abs());
                worst = worst.max((f64::from(got[col * f.out_dim + r]) - want).abs());
            }
        }
        assert!(
            worst <= 1e-3 * peak,
            "act {act}: worst error {worst} against a peak of {peak}"
        );
    }
}

/// The tiled stage through a slot map is bit-identical to it over the whole
/// stack.
pub fn iq4nl_expert_gemm_reads_through_the_slot_map(gpu: &Gpu) {
    let f = Iq4NlFixture::with_tokens(40, 100);
    let x = signal(f.n_tok * f.in_dim, 73);
    let whole = gemm(&f, gpu, bytemuck::cast_slice(&f.blocks), None, 0, &x);
    let expert_blocks = f.out_dim * f.in_dim / 32;
    let slots = 12usize;
    let map: Vec<i32> = (0..f.experts)
        .map(|e| ((5 * e + 3) % slots) as i32)
        .collect();
    let noise = BlockIq4Nl {
        d: half::f16::from_f32(7.0).to_bits(),
        qs: [0x5a; 16],
    };
    let mut pool = vec![noise; slots * expert_blocks];
    for (e, &slot) in map.iter().enumerate() {
        let slot = slot as usize;
        pool[slot * expert_blocks..(slot + 1) * expert_blocks]
            .copy_from_slice(&f.blocks[e * expert_blocks..(e + 1) * expert_blocks]);
    }
    let mapped = gemm(&f, gpu, bytemuck::cast_slice(&pool), Some(&map), 0, &x);
    assert!(
        whole
            .iter()
            .zip(&mapped)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the mapped tile differs from the whole-stack tile"
    );
}

/// A column's result does not depend on which columns share its tile: the
/// first five tokens' columns are bit-identical with 35 more tokens beside
/// them.
pub fn iq4nl_expert_gemm_column_is_independent_of_its_neighbours(gpu: &Gpu) {
    let small = Iq4NlFixture::with_tokens(5, 100);
    let large = Iq4NlFixture::with_tokens(40, 100);
    let x = signal(large.n_tok * large.in_dim, 73);
    let a = gemm(
        &small,
        gpu,
        bytemuck::cast_slice(&small.blocks),
        None,
        0,
        &x[..5 * small.in_dim],
    );
    let b = gemm(
        &large,
        gpu,
        bytemuck::cast_slice(&large.blocks),
        None,
        0,
        &x,
    );
    let n = small.n_tok * small.used * small.out_dim;
    assert!(
        a[..n]
            .iter()
            .zip(&b[..n])
            .all(|(p, q)| p.to_bits() == q.to_bits()),
        "a column's result changed with its tile's other columns"
    );
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
