//! The sparse family's hc kernels: cases graded against a host restatement of the CUDA bodies.
//!
//! Every case runs at the shipped shape - a 2560-wide residual in four streams,
//! a 320-wide low-rank vector - and at the grid, block and shared-memory size
//! the engine launches with (`shiso_engine::sparse::kernels::hc_init`,
//! `hc_norm`, `hc_collapse`, `hc_combine`, `elementwise`; all 256 threads). The
//! combine runs at both of its second grid extents: one block per (token,
//! stream) below 256 tokens, one per token from 256 up, since the stream walk is
//! a grid stride whose step is that extent.
//!
//! The oracles sum in `f64`, so the tolerance covers the kernel's `f32`
//! summation order and its fast `exp`/`rsqrt`, and a half-precision output is
//! graded against the oracle within one half-precision rounding.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "kernel arguments are u32 while the fixtures index in usize; every size here is a \
              small literal, and the f64 oracles narrow to compare against f32 outputs"
)]

use half::f16;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The module every entry here lives in, mirroring `kernels::MOD_HC`.
const MODULE: &str = "q4e_hc";
/// Threads per block of every hc launch, mirroring `kernels::BLOCK_LINEAR`.
const BLOCK: u32 = 256;
/// The residual width of one stream.
const HIDDEN: u32 = 2_560;
/// Residual streams.
const STREAMS: u32 = 4;
/// The mix's low-rank width.
const STREAM_RANK: u32 = 320;
/// The stack's `RMSNorm` epsilon.
const EPS: f32 = 1.0e-6;
/// Tokens in the decode-shaped cases: fewer than a block's width, so the
/// combine takes one block per (token, stream).
const FEW: u32 = 3;
/// Tokens in the prefill-shaped combine: enough that the combine takes one
/// block per token and walks all four streams in it.
const MANY: u32 = 256;

/// `f16` values as little-endian bytes.
fn f16_bytes(values: &[f16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A device buffer read back as `f16` values widened to `f32`.
fn read_f16(gpu: &Gpu, buffer: &DeviceBuffer) -> Vec<f32> {
    gpu.read(buffer)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| f16::from_le_bytes(*b).to_f32())
        .collect()
}

/// Fixture values rounded to `f16`, for the half-precision inputs.
fn halves(len: usize, seed: u32) -> Vec<f16> {
    signal(len, seed).into_iter().map(f16::from_f32).collect()
}

/// A norm weight: near one and never zero, as a trained `RMSNorm` gain is.
fn gain(len: usize, seed: u32) -> Vec<f32> {
    signal(len, seed)
        .into_iter()
        .map(|v| 1.0 + 0.5 * v)
        .collect()
}

/// Launches one entry and waits for it.
fn launch(gpu: &Gpu, entry: &'static str, grid: Extent3, args: &[KernelArg<'_>]) {
    run(
        gpu,
        KernelRef::new(MODULE, entry),
        &LaunchArgs {
            grid,
            block: Extent3::new(BLOCK, 1, 1),
            smem_bytes: 0,
            args,
        },
    );
    drain(gpu);
}

/// The engine's `elementwise(n)` grid.
fn flat(n: u64) -> Extent3 {
    Extent3::new(n.div_ceil(u64::from(BLOCK)) as u32, 1, 1)
}

/// Asserts `got` matches `want` within `rel * |want| + abs`, naming the first
/// element that does not.
fn close(got: &[f32], want: &[f64], rel: f64, abs: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs();
        assert!(
            err <= rel * w.abs() + abs,
            "{what}: element {i} is {g}, expected {w} (error {err:e})"
        );
    }
}

/// Tolerance for an `f32` output against the `f64` oracle.
const F32_REL: f64 = 2.0e-5;
/// Absolute floor for an `f32` output.
const F32_ABS: f64 = 2.0e-6;
/// Tolerance for an `f16` output: one rounding (2^-11 relative) plus the `f32`
/// error that can move a value across a rounding boundary.
const F16_REL: f64 = 1.0e-3;
/// Absolute floor for an `f16` output, around the smallest normal half.
const F16_ABS: f64 = 1.0e-4;

/// `q4e_hc_init_kernel` tiles each token's embedding across every stream.
pub fn the_residual_broadcast_tiles_the_embedding_across_streams(gpu: &Gpu) {
    let (h, s, n) = (HIDDEN as usize, STREAMS as usize, FEW as usize);
    let embed = signal(n * h, 3);
    let embed_dev = gpu.upload(&f32_bytes(&embed));
    let res = gpu.upload(&f32_bytes(&vec![f32::NAN; n * s * h]));
    launch(
        gpu,
        "q4e_hc_init_kernel",
        flat((n * s * h) as u64),
        &[
            KernelArg::Buffer(res.view()),
            KernelArg::Buffer(embed_dev.view()),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(STREAMS),
            KernelArg::U32(FEW),
        ],
    );
    let want: Vec<f64> = (0..n * s * h)
        .map(|i| f64::from(embed[(i / (s * h)) * h + i % h]))
        .collect();
    close(&gpu.read_f32(&res), &want, 0.0, 0.0, "hc init");
}

/// `q4e_hc_init_add_kernel` adds the per-stream term to the broadcast.
pub fn the_draft_broadcast_adds_the_per_stream_term(gpu: &Gpu) {
    let (h, s, n) = (HIDDEN as usize, STREAMS as usize, FEW as usize);
    let embed = signal(n * h, 5);
    let term = signal(n * s * h, 7);
    let embed_dev = gpu.upload(&f32_bytes(&embed));
    let term_dev = gpu.upload(&f32_bytes(&term));
    let res = gpu.upload(&f32_bytes(&vec![f32::NAN; n * s * h]));
    launch(
        gpu,
        "q4e_hc_init_add_kernel",
        flat((n * s * h) as u64),
        &[
            KernelArg::Buffer(res.view()),
            KernelArg::Buffer(embed_dev.view()),
            KernelArg::Buffer(term_dev.view()),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(STREAMS),
            KernelArg::U32(FEW),
        ],
    );
    // One f32 add, so the oracle computes it in f32 and the match is exact.
    let want: Vec<f64> = (0..n * s * h)
        .map(|i| f64::from(embed[(i / (s * h)) * h + i % h] + term[i]))
        .collect();
    close(&gpu.read_f32(&res), &want, 0.0, 0.0, "hc init add");
}

/// The group `RMSNorm` oracle: each `width` run normed on its own, against the
/// slice of `w` for its group.
fn norm_oracle(src: &[f32], w: &[f32], width: usize, groups: usize) -> Vec<f64> {
    src.chunks(width)
        .enumerate()
        .flat_map(|(g, row)| {
            let wh = &w[(g % groups) * width..(g % groups + 1) * width];
            let ms = row
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                / width as f64;
            let scale = 1.0 / (ms + f64::from(EPS)).sqrt();
            row.iter()
                .zip(wh)
                .map(move |(&v, &wv)| f64::from(v) * scale * f64::from(wv))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Runs `q4e_hc_norm_kernel` at one width and group count.
fn norm_case(gpu: &Gpu, half_out: bool, width: u32, groups: u32, n_tok: u32) {
    let (wd, g, n) = (width as usize, groups as usize, n_tok as usize);
    let src = signal(n * g * wd, 11);
    let w = gain(g * wd, 13);
    let src_dev = gpu.upload(&f32_bytes(&src));
    let w_dev = gpu.upload(&f32_bytes(&w));
    let (entry, out) = if half_out {
        (
            "_Z18q4e_hc_norm_kernelI6__halfEv",
            gpu.upload(&f16_bytes(&vec![f16::NAN; n * g * wd])),
        )
    } else {
        (
            "_Z18q4e_hc_norm_kernelIfEv",
            gpu.upload(&f32_bytes(&vec![f32::NAN; n * g * wd])),
        )
    };
    launch(
        gpu,
        entry,
        // `hc_norm`: one block per (token, group).
        Extent3::new(n_tok * groups, 1, 1),
        &[
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(src_dev.view()),
            KernelArg::Buffer(w_dev.view()),
            KernelArg::U32(width),
            KernelArg::U32(groups),
            KernelArg::F32(EPS),
        ],
    );
    let want = norm_oracle(&src, &w, wd, g);
    if half_out {
        close(&read_f16(gpu, &out), &want, F16_REL, F16_ABS, entry);
    } else {
        close(&gpu.read_f32(&out), &want, F32_REL, F32_ABS, entry);
    }
}

/// The stream norm into `f32`, one group per stream.
pub fn the_stream_norm_normalizes_each_stream_on_its_own(gpu: &Gpu) {
    norm_case(gpu, false, HIDDEN, STREAMS, FEW);
}

/// The stream norm into `f16`, one group per stream.
pub fn the_half_stream_norm_normalizes_each_stream_on_its_own(gpu: &Gpu) {
    norm_case(gpu, true, HIDDEN, STREAMS, FEW);
}

/// The draft head's norm: the whole multi-stream vector as one group.
pub fn the_draft_norm_normalizes_all_streams_as_one_group(gpu: &Gpu) {
    norm_case(gpu, false, HIDDEN * STREAMS, 1, FEW);
}

/// `q4e_scale_silu_kernel`: `silu(x / streams)` in place over the low-rank
/// vector.
pub fn the_mix_activation_scales_inside_the_silu(gpu: &Gpu) {
    let n = (5 * STREAM_RANK) as usize;
    let x: Vec<f32> = signal(n, 17).into_iter().map(|v| 6.0 * v).collect();
    let x_dev = gpu.upload(&f32_bytes(&x));
    let inv = 1.0 / STREAMS as f32;
    launch(
        gpu,
        "q4e_scale_silu_kernel",
        flat(n as u64),
        &[
            KernelArg::Buffer(x_dev.view()),
            KernelArg::F32(inv),
            KernelArg::U64(n as u64),
        ],
    );
    let want: Vec<f64> = x
        .iter()
        .map(|&v| {
            let v = f64::from(v * inv);
            v / (1.0 + (-v).exp())
        })
        .collect();
    close(&gpu.read_f32(&x_dev), &want, F32_REL, F32_ABS, "scale silu");
}

/// The collapse oracle: the mean over streams of `xn * sigmoid(up)`.
fn collapse_oracle(xn: &[f32], up: &[f32], n_tok: usize) -> Vec<f64> {
    let (width, streams) = (HIDDEN as usize, STREAMS as usize);
    (0..n_tok * width)
        .map(|i| {
            let (tok, e) = (i / width, i % width);
            let acc: f64 = (0..streams)
                .map(|st| {
                    let at = (tok * streams + st) * width + e;
                    f64::from(xn[at]) / (1.0 + (-f64::from(up[at])).exp())
                })
                .sum();
            acc / streams as f64
        })
        .collect()
}

/// `q4e_hc_collapse_kernel<float>`.
pub fn the_collapse_takes_the_mean_of_the_gated_streams(gpu: &Gpu) {
    let len = (FEW * STREAMS * HIDDEN) as usize;
    let xn = signal(len, 19);
    let up: Vec<f32> = signal(len, 23).into_iter().map(|v| 4.0 * v).collect();
    let xn_dev = gpu.upload(&f32_bytes(&xn));
    let up_dev = gpu.upload(&f32_bytes(&up));
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; (FEW * HIDDEN) as usize]));
    launch(
        gpu,
        "_Z22q4e_hc_collapse_kernelIfEv",
        Extent3::new(FEW, 1, 1),
        &[
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(xn_dev.view()),
            KernelArg::Buffer(up_dev.view()),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(STREAMS),
        ],
    );
    let want = collapse_oracle(&xn, &up, FEW as usize);
    close(&gpu.read_f32(&out), &want, F32_REL, F32_ABS, "collapse f32");
}

/// `q4e_hc_collapse_kernel<__half>` and `q4e_hc_collapse_f16_dual_kernel`
/// over the same half-precision streams: the dual entry's `f32` output is the
/// other's exactly, and its half copy is that rounded.
///
/// # Panics
///
/// When any output disagrees with the oracle or the two entries disagree.
pub fn the_half_collapses_agree_and_round_the_block_input_once(gpu: &Gpu) {
    let len = (FEW * STREAMS * HIDDEN) as usize;
    let out_len = (FEW * HIDDEN) as usize;
    let xn = halves(len, 29);
    let up: Vec<f16> = signal(len, 31)
        .into_iter()
        .map(|v| f16::from_f32(4.0 * v))
        .collect();
    let xn_dev = gpu.upload(&f16_bytes(&xn));
    let up_dev = gpu.upload(&f16_bytes(&up));
    let single = gpu.upload(&f32_bytes(&vec![f32::NAN; out_len]));
    launch(
        gpu,
        "_Z22q4e_hc_collapse_kernelI6__halfEv",
        Extent3::new(FEW, 1, 1),
        &[
            KernelArg::Buffer(single.view()),
            KernelArg::Buffer(xn_dev.view()),
            KernelArg::Buffer(up_dev.view()),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(STREAMS),
        ],
    );
    let dual = gpu.upload(&f32_bytes(&vec![f32::NAN; out_len]));
    let dual_h = gpu.upload(&f16_bytes(&vec![f16::NAN; out_len]));
    launch(
        gpu,
        "q4e_hc_collapse_f16_dual_kernel",
        Extent3::new(FEW, 1, 1),
        &[
            KernelArg::Buffer(dual.view()),
            KernelArg::Buffer(dual_h.view()),
            KernelArg::Buffer(xn_dev.view()),
            KernelArg::Buffer(up_dev.view()),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(STREAMS),
        ],
    );
    let xn32: Vec<f32> = xn.iter().map(|v| v.to_f32()).collect();
    let up32: Vec<f32> = up.iter().map(|v| v.to_f32()).collect();
    let want = collapse_oracle(&xn32, &up32, FEW as usize);
    let single = gpu.read_f32(&single);
    let dual = gpu.read_f32(&dual);
    close(&single, &want, F32_REL, F32_ABS, "collapse f16");
    close(&dual, &want, F32_REL, F32_ABS, "collapse f16 dual");
    let dual_h = read_f16(gpu, &dual_h);
    for (i, (&d, &dh)) in dual.iter().zip(&dual_h).enumerate() {
        assert!(
            d.to_bits() == single[i].to_bits(),
            "collapse f16 dual: element {i} is {d}, the single-output collapse wrote {}",
            single[i]
        );
        assert_eq!(
            f16::from_f32(d).to_f32().to_bits(),
            dh.to_bits(),
            "collapse f16 dual: element {i}'s half copy {dh} is not {d} rounded"
        );
    }
}

/// What one combine launch reads, on the device and on the host.
struct Combine {
    n_tok: u32,
    res: Vec<f32>,
    block_out: Vec<f32>,
    inject: Vec<f32>,
    w: Vec<f32>,
}

impl Combine {
    fn new(n_tok: u32) -> Self {
        let (h, s, n) = (HIDDEN as usize, STREAMS as usize, n_tok as usize);
        Self {
            n_tok,
            res: signal(n * s * h, 37),
            block_out: signal(n * h, 41),
            inject: signal(n * s, 43).into_iter().map(|v| 3.0 * v).collect(),
            w: gain(s * h, 47),
        }
    }

    /// The updated residual and, when `normed`, the next mix's normed input.
    fn oracle(&self) -> (Vec<f64>, Vec<f64>) {
        let (h, s, n) = (HIDDEN as usize, STREAMS as usize, self.n_tok as usize);
        let mut res = Vec::with_capacity(n * s * h);
        for t in 0..n {
            for st in 0..s {
                // The residual is stored in f32 and the norm reads the stored
                // value, so the update is restated in f32 exactly as the
                // kernel computes it, with the weight's `exp` in f64.
                let logit = f64::from(self.inject[t * s + st]) / s as f64;
                let wgt = (2.0 / (1.0 + (-logit).exp())) as f32;
                for e in 0..h {
                    let o = (t * s + st) * h + e;
                    res.push(self.res[o] + self.block_out[t * h + e] * wgt);
                }
            }
        }
        let normed = norm_oracle(&res, &self.w, h, s);
        (res.into_iter().map(f64::from).collect(), normed)
    }
}

/// Runs `q4e_hc_combine_norm_kernel` once, at the engine's grid for the token
/// count, with or without the next norm's weight.
fn combine_case(gpu: &Gpu, half_out: bool, n_tok: u32, chained: bool) {
    let fx = Combine::new(n_tok);
    let xn_len = fx.res.len();
    let res_dev = gpu.upload(&f32_bytes(&fx.res));
    let block_dev = gpu.upload(&f32_bytes(&fx.block_out));
    let inject_dev = gpu.upload(&f32_bytes(&fx.inject));
    let w_dev = gpu.upload(&f32_bytes(&fx.w));
    let (entry, xn) = if half_out {
        (
            "_Z26q4e_hc_combine_norm_kernelI6__halfEv",
            gpu.upload(&f16_bytes(&vec![f16::NAN; xn_len])),
        )
    } else {
        (
            "_Z26q4e_hc_combine_norm_kernelIfEv",
            gpu.upload(&f32_bytes(&vec![f32::NAN; xn_len])),
        )
    };
    // `Operand::Null` is eight zero bytes in the pointer's slot.
    let w_arg = if chained {
        KernelArg::Buffer(w_dev.view())
    } else {
        KernelArg::U64(0)
    };
    // `hc_combine`: a second extent of one from a block's width of tokens up.
    let y = if n_tok >= BLOCK { 1 } else { STREAMS };
    launch(
        gpu,
        entry,
        Extent3::new(n_tok, y, 1),
        &[
            KernelArg::Buffer(res_dev.view()),
            KernelArg::Buffer(xn.view()),
            KernelArg::Buffer(block_dev.view()),
            KernelArg::Buffer(inject_dev.view()),
            w_arg,
            KernelArg::U32(HIDDEN),
            KernelArg::U32(STREAMS),
            KernelArg::F32(EPS),
        ],
    );
    let (want_res, want_xn) = fx.oracle();
    let what = format!("{entry} at {n_tok} tokens, grid y {y}");
    // The weight's `exp` is the only step that is not an exact f32 restatement.
    close(&gpu.read_f32(&res_dev), &want_res, F32_REL, F32_ABS, &what);
    let got_xn = if half_out {
        read_f16(gpu, &xn)
    } else {
        gpu.read_f32(&xn)
    };
    if chained {
        let (rel, abs) = if half_out {
            (F16_REL, F16_ABS)
        } else {
            (F32_REL, F32_ABS)
        };
        close(&got_xn, &want_xn, rel, abs, &what);
    } else {
        // Without a next norm the launch is the residual update alone and
        // must not write the normed buffer at all.
        assert!(
            got_xn.iter().all(|v| v.is_nan()),
            "{what}: an unchained combine wrote the normed buffer"
        );
    }
}

/// The fused combine into `f32`, one block per (token, stream).
pub fn the_combine_updates_and_norms_per_stream_block(gpu: &Gpu) {
    combine_case(gpu, false, FEW, true);
}

/// The fused combine into `f32`, one block per token walking all streams.
pub fn the_combine_updates_and_norms_per_token_block(gpu: &Gpu) {
    combine_case(gpu, false, MANY, true);
}

/// The fused combine into `f16`, at both grids.
pub fn the_half_combine_updates_and_norms(gpu: &Gpu) {
    combine_case(gpu, true, FEW, true);
    combine_case(gpu, true, MANY, true);
}

/// With the engine's null weight the combine is the residual update alone.
pub fn the_unchained_combine_only_updates_the_residual(gpu: &Gpu) {
    combine_case(gpu, false, FEW, false);
    combine_case(gpu, true, MANY, false);
}
