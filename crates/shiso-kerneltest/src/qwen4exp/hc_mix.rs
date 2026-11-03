//! The stream mix of a prompt pass, fused: the contraction and the injection
//! logits as one launch over the normed streams (`q4e_bf16_gemm_pair_kernel`),
//! and the expansion with the collapse of the gated streams in its epilogue
//! (`q4e_bf16_gemm_collapse_kernel`), graded bit for bit against the launches
//! they replace - `q4e_bf16_gemm_kernel` per weight, and that kernel's
//! expansion followed by `q4e_hc_collapse_kernel<float>`.
//!
//! The bench fixtures time each fused launch beside the launches it replaces
//! at a 4096-row prompt chunk, where the streams (`[rows, 4, 2560]` f32, 168
//! MB) are what every one of them is bound by.
#![expect(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "the stream count and the fixture widths are small literals; a case \
              reads best as the launches it compares, in order"
)]

use half::bf16;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::bench::Fixture;
use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The brain-float module, mirroring `kernels::MOD_BF16`.
const BF16: &str = "q4e_bf16";
/// The hyper-connection module, mirroring `kernels::MOD_HC`.
const HC: &str = "q4e_hc";
/// The residual width.
const HIDDEN: u32 = 2_560;
/// The stream count.
const STREAMS: u32 = 4;
/// The low-rank width the mix contracts to.
const RANK: u32 = 320;
/// The injection logits a mix projects, one a stream.
const INJECT: u32 = 4;
/// One GEMM tile's side, mirroring `matmul::BF16_GEMM_TILE`.
const TILE: u32 = 128;
/// Threads a GEMM tile runs with, mirroring `matmul::BF16_GEMM_THREADS`.
const THREADS: u32 = 512;
/// The plain GEMM's dynamic shared memory, mirroring `matmul::BF16_GEMM_SMEM`;
/// the pair and the fused collapse take the same.
const SMEM: u32 = 90_112;
/// Block-input elements one fused-collapse tile covers, mirroring
/// `Q4E_BF_COLLAPSE_E`.
const COLLAPSE_E: u32 = 32;

/// A brain-float matrix `[out, in]` of small values, as bytes.
fn matrix(out_dim: u32, in_dim: u32, seed: u32) -> Vec<u8> {
    signal(out_dim as usize * in_dim as usize, seed)
        .iter()
        .flat_map(|v| bf16::from_f32(v * 0.05).to_bits().to_le_bytes())
        .collect()
}

/// A device buffer of `len` NaNs, so an output a kernel never writes fails.
fn sentinel(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![f32::NAN; len]))
}

/// One launch of `entry` in `module`.
fn launch(
    gpu: &Gpu,
    module: &'static str,
    entry: &'static str,
    geometry: (Extent3, u32, u32),
    args: &[KernelArg<'_>],
) {
    let (grid, threads, smem) = geometry;
    let mut l = LaunchArgs::new(grid, Extent3::new(threads, 1, 1), args);
    l.smem_bytes = smem;
    run(gpu, KernelRef::new(module, entry), &l);
}

/// `q4e_bf16_gemm_kernel`: `out [rows, out_dim] = x [rows, in_dim] w^T`.
fn gemm(gpu: &Gpu, out: &DeviceBuffer, w: &DeviceBuffer, x: &DeviceBuffer, dims: (u32, u32, u32)) {
    let (in_dim, out_dim, rows) = dims;
    launch(
        gpu,
        BF16,
        "q4e_bf16_gemm_kernel",
        (
            Extent3::new(out_dim.div_ceil(TILE), rows.div_ceil(TILE), 1),
            THREADS,
            SMEM,
        ),
        &[
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(w.view()),
            KernelArg::Buffer(x.view()),
            KernelArg::U32(in_dim),
            KernelArg::U32(out_dim),
            KernelArg::U32(rows),
        ],
    );
}

/// The two outputs and weights of one pair launch.
struct Pair<'a> {
    out_a: &'a DeviceBuffer,
    w_a: &'a DeviceBuffer,
    out_b: &'a DeviceBuffer,
    w_b: &'a DeviceBuffer,
}

/// `q4e_bf16_gemm_pair_kernel` over the contraction and the injection.
fn pair(gpu: &Gpu, p: &Pair<'_>, x: &DeviceBuffer, rows: u32) {
    let in_dim = STREAMS * HIDDEN;
    launch(
        gpu,
        BF16,
        "q4e_bf16_gemm_pair_kernel",
        (
            Extent3::new((RANK + INJECT).div_ceil(TILE), rows.div_ceil(TILE), 1),
            THREADS,
            SMEM,
        ),
        &[
            KernelArg::Buffer(p.out_a.view()),
            KernelArg::Buffer(p.w_a.view()),
            KernelArg::U32(RANK),
            KernelArg::Buffer(p.out_b.view()),
            KernelArg::Buffer(p.w_b.view()),
            KernelArg::U32(INJECT),
            KernelArg::Buffer(x.view()),
            KernelArg::U32(in_dim),
            KernelArg::U32(rows),
        ],
    );
}

/// `q4e_hc_collapse_kernel<float>`, as this device names it.
fn collapse_kernel(gpu: &Gpu) -> KernelRef {
    let logical = "_Z22q4e_hc_collapse_kernelIfEv";
    gpu.device()
        .resolve_entry(HC, logical)
        .unwrap_or_else(|| KernelRef::new(HC, logical))
}

/// The expansion and the collapse as today's two launches: `up` is the
/// expanded streams' scratch.
fn split_collapse(gpu: &Gpu, bufs: &CollapseBufs<'_>, up: &DeviceBuffer, rows: u32) {
    gemm(gpu, up, bufs.w_up, bufs.x, (RANK, STREAMS * HIDDEN, rows));
    run(
        gpu,
        collapse_kernel(gpu),
        &LaunchArgs::new(
            Extent3::new(rows, 1, 1),
            Extent3::new(256, 1, 1),
            &[
                KernelArg::Buffer(bufs.out.view()),
                KernelArg::Buffer(bufs.xn.view()),
                KernelArg::Buffer(up.view()),
                KernelArg::U32(HIDDEN),
                KernelArg::U32(STREAMS),
            ],
        ),
    );
}

/// One collapse's operands.
struct CollapseBufs<'a> {
    out: &'a DeviceBuffer,
    w_up: &'a DeviceBuffer,
    x: &'a DeviceBuffer,
    xn: &'a DeviceBuffer,
}

/// `q4e_bf16_gemm_collapse_kernel`.
fn fused_collapse(gpu: &Gpu, bufs: &CollapseBufs<'_>, rows: u32) {
    launch(
        gpu,
        BF16,
        "q4e_bf16_gemm_collapse_kernel",
        (
            Extent3::new(HIDDEN / COLLAPSE_E, rows.div_ceil(TILE), 1),
            THREADS,
            SMEM,
        ),
        &[
            KernelArg::Buffer(bufs.out.view()),
            KernelArg::Buffer(bufs.w_up.view()),
            KernelArg::Buffer(bufs.x.view()),
            KernelArg::Buffer(bufs.xn.view()),
            KernelArg::U32(RANK),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(STREAMS),
            KernelArg::U32(rows),
        ],
    );
}

/// Asserts two outputs are the same bits, naming the first that is not.
fn same_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{what}: element {i} is {g:e}, expected {w:e}"
        );
    }
}

/// The pair launch writes, weight for weight, the plain GEMM's bits - at a
/// row count inside one tile, one past it and a ragged wider one.
///
/// # Panics
///
/// When an output differs, or the device refuses a launch.
pub fn the_pair_gemm_is_its_two_gemms_bit_for_bit(gpu: &Gpu) {
    let in_dim = STREAMS * HIDDEN;
    let w_a = gpu.upload(&matrix(RANK, in_dim, 0xD0));
    let w_b = gpu.upload(&matrix(INJECT, in_dim, 0x1E));
    for rows in [1_u32, 129, 333] {
        let x = gpu.upload(&f32_bytes(&signal(
            rows as usize * in_dim as usize,
            0x77 ^ rows,
        )));
        let (want_a, want_b) = (
            sentinel(gpu, (rows * RANK) as usize),
            sentinel(gpu, (rows * INJECT) as usize),
        );
        gemm(gpu, &want_a, &w_a, &x, (in_dim, RANK, rows));
        gemm(gpu, &want_b, &w_b, &x, (in_dim, INJECT, rows));
        let (got_a, got_b) = (
            sentinel(gpu, (rows * RANK) as usize),
            sentinel(gpu, (rows * INJECT) as usize),
        );
        let p = Pair {
            out_a: &got_a,
            w_a: &w_a,
            out_b: &got_b,
            w_b: &w_b,
        };
        pair(gpu, &p, &x, rows);
        drain(gpu);
        same_bits(
            &format!("contraction at {rows} rows"),
            &gpu.read_f32(&got_a),
            &gpu.read_f32(&want_a),
        );
        same_bits(
            &format!("injection at {rows} rows"),
            &gpu.read_f32(&got_b),
            &gpu.read_f32(&want_b),
        );
    }
}

/// The fused expansion and collapse writes the bits the two launches write,
/// at the same row counts.
///
/// # Panics
///
/// When an output differs, or the device refuses a launch.
pub fn the_fused_collapse_is_the_expansion_and_the_collapse_bit_for_bit(gpu: &Gpu) {
    let wide = STREAMS * HIDDEN;
    let w_up = gpu.upload(&matrix(wide, RANK, 0x0B));
    for rows in [1_u32, 129, 333] {
        // The activated low-rank rows sit in silu's range; the expanded
        // values then cover both tails of the gate.
        let x = gpu.upload(&f32_bytes(
            &signal(rows as usize * RANK as usize, 0x55 ^ rows)
                .iter()
                .map(|v| 8.0 * v)
                .collect::<Vec<_>>(),
        ));
        let xn = gpu.upload(&f32_bytes(&signal(
            rows as usize * wide as usize,
            0x99 ^ rows,
        )));
        let up = sentinel(gpu, (rows * wide) as usize);
        let want = sentinel(gpu, (rows * HIDDEN) as usize);
        let got = sentinel(gpu, (rows * HIDDEN) as usize);
        let bufs = CollapseBufs {
            out: &want,
            w_up: &w_up,
            x: &x,
            xn: &xn,
        };
        split_collapse(gpu, &bufs, &up, rows);
        fused_collapse(gpu, &CollapseBufs { out: &got, ..bufs }, rows);
        drain(gpu);
        same_bits(
            &format!("collapse at {rows} rows"),
            &gpu.read_f32(&got),
            &gpu.read_f32(&want),
        );
    }
}

/// The combine's launch geometry, mirroring `kernels::hc_combine`.
fn combine_grid(rows: u32) -> Extent3 {
    Extent3::new(rows, if rows >= 256 { 1 } else { STREAMS }, 1)
}

/// `q4e_hc_combine_norm_kernel<float>`, as this device names it.
fn combine_norm_kernel(gpu: &Gpu) -> KernelRef {
    let logical = "_Z26q4e_hc_combine_norm_kernelIfEv";
    gpu.device()
        .resolve_entry(HC, logical)
        .unwrap_or_else(|| KernelRef::new(HC, logical))
}

/// The residual a combine updates and what it reads, for one row count.
struct CombineInputs {
    res: Vec<f32>,
    block_out: DeviceBuffer,
    inject: DeviceBuffer,
    norm: DeviceBuffer,
}

impl CombineInputs {
    fn new(gpu: &Gpu, rows: u32) -> Self {
        let wide = (rows * STREAMS * HIDDEN) as usize;
        Self {
            res: signal(wide, 0x31 ^ rows),
            block_out: gpu.upload(&f32_bytes(&signal((rows * HIDDEN) as usize, 0x32 ^ rows))),
            inject: gpu.upload(&f32_bytes(&signal((rows * STREAMS) as usize, 0x33 ^ rows))),
            norm: gpu.upload(&f32_bytes(
                &signal((STREAMS * HIDDEN) as usize, 0x34)
                    .iter()
                    .map(|v| 1.0 + 0.5 * v)
                    .collect::<Vec<_>>(),
            )),
        }
    }
}

/// A prompt pass's chained combine and the next mix, both ways: today's
/// combine storing the normed streams and the fused launches reading them,
/// and the combine leaving only the norm scale with the launches norming the
/// residual as they read it. The residual, the low-rank rows, the injection
/// logits and the block input all come out the same bits.
///
/// # Panics
///
/// When an output differs, or the device refuses a launch.
pub fn the_scaled_combine_and_norming_reads_are_the_stored_normed_streams(gpu: &Gpu) {
    let wide = STREAMS * HIDDEN;
    let w_a = gpu.upload(&matrix(RANK, wide, 0xD0));
    let w_b = gpu.upload(&matrix(INJECT, wide, 0x1E));
    let w_up = gpu.upload(&matrix(wide, RANK, 0x0B));
    for rows in [1_u32, 129, 333] {
        let inputs = CombineInputs::new(gpu, rows);
        let x = gpu.upload(&f32_bytes(
            &signal((rows * RANK) as usize, 0x55 ^ rows)
                .iter()
                .map(|v| 8.0 * v)
                .collect::<Vec<_>>(),
        ));
        let mut outs = Vec::new();
        for scaled in [false, true] {
            let res = gpu.upload(&f32_bytes(&inputs.res));
            let xn = sentinel(gpu, (rows * wide) as usize);
            let scale = sentinel(gpu, (rows * STREAMS) as usize);
            let low = sentinel(gpu, (rows * RANK) as usize);
            let inj = sentinel(gpu, (rows * INJECT) as usize);
            let mixed = sentinel(gpu, (rows * HIDDEN) as usize);
            let eps = 1.0e-6_f32;
            if scaled {
                launch(
                    gpu,
                    HC,
                    "q4e_hc_combine_scale_kernel",
                    (combine_grid(rows), 256, 0),
                    &[
                        KernelArg::Buffer(res.view()),
                        KernelArg::Buffer(scale.view()),
                        KernelArg::Buffer(inputs.block_out.view()),
                        KernelArg::Buffer(inputs.inject.view()),
                        KernelArg::U32(HIDDEN),
                        KernelArg::U32(STREAMS),
                        KernelArg::F32(eps),
                    ],
                );
                launch(
                    gpu,
                    BF16,
                    "q4e_bf16_gemm_pair_norm_kernel",
                    (
                        Extent3::new((RANK + INJECT).div_ceil(TILE), rows.div_ceil(TILE), 1),
                        THREADS,
                        SMEM,
                    ),
                    &[
                        KernelArg::Buffer(low.view()),
                        KernelArg::Buffer(w_a.view()),
                        KernelArg::U32(RANK),
                        KernelArg::Buffer(inj.view()),
                        KernelArg::Buffer(w_b.view()),
                        KernelArg::U32(INJECT),
                        KernelArg::Buffer(res.view()),
                        KernelArg::Buffer(scale.view()),
                        KernelArg::Buffer(inputs.norm.view()),
                        KernelArg::U32(HIDDEN),
                        KernelArg::U32(wide),
                        KernelArg::U32(rows),
                    ],
                );
                launch(
                    gpu,
                    BF16,
                    "q4e_bf16_gemm_collapse_norm_kernel",
                    (
                        Extent3::new(HIDDEN / COLLAPSE_E, rows.div_ceil(TILE), 1),
                        THREADS,
                        SMEM,
                    ),
                    &[
                        KernelArg::Buffer(mixed.view()),
                        KernelArg::Buffer(w_up.view()),
                        KernelArg::Buffer(x.view()),
                        KernelArg::Buffer(res.view()),
                        KernelArg::Buffer(scale.view()),
                        KernelArg::Buffer(inputs.norm.view()),
                        KernelArg::U32(RANK),
                        KernelArg::U32(HIDDEN),
                        KernelArg::U32(STREAMS),
                        KernelArg::U32(rows),
                    ],
                );
            } else {
                run(
                    gpu,
                    combine_norm_kernel(gpu),
                    &LaunchArgs::new(
                        combine_grid(rows),
                        Extent3::new(256, 1, 1),
                        &[
                            KernelArg::Buffer(res.view()),
                            KernelArg::Buffer(xn.view()),
                            KernelArg::Buffer(inputs.block_out.view()),
                            KernelArg::Buffer(inputs.inject.view()),
                            KernelArg::Buffer(inputs.norm.view()),
                            KernelArg::U32(HIDDEN),
                            KernelArg::U32(STREAMS),
                            KernelArg::F32(eps),
                        ],
                    ),
                );
                let p = Pair {
                    out_a: &low,
                    w_a: &w_a,
                    out_b: &inj,
                    w_b: &w_b,
                };
                pair(gpu, &p, &xn, rows);
                fused_collapse(
                    gpu,
                    &CollapseBufs {
                        out: &mixed,
                        w_up: &w_up,
                        x: &x,
                        xn: &xn,
                    },
                    rows,
                );
            }
            drain(gpu);
            outs.push([
                gpu.read_f32(&res),
                gpu.read_f32(&low),
                gpu.read_f32(&inj),
                gpu.read_f32(&mixed),
            ]);
        }
        for (i, what) in ["residual", "low-rank rows", "injection", "block input"]
            .iter()
            .enumerate()
        {
            same_bits(&format!("{what} at {rows} rows"), &outs[1][i], &outs[0][i]);
        }
    }
}

/// A prompt pass's chained combine over a 4096-row chunk: storing the
/// normed streams, or (`scaled`) leaving their scale.
fn bench_combine(gpu: &Gpu, scaled: bool) -> Fixture {
    let inputs = CombineInputs::new(gpu, BENCH_ROWS);
    let res = gpu.upload(&f32_bytes(&inputs.res));
    let xn = gpu.upload(&vec![0u8; (BENCH_ROWS * STREAMS * HIDDEN * 4) as usize]);
    let scale = gpu.upload(&vec![0u8; (BENCH_ROWS * STREAMS * 4) as usize]);
    let wide = u64::from(BENCH_ROWS) * u64::from(STREAMS * HIDDEN) * 4;
    // The floor of each: residual read and written, block output read, and
    // the normed copy written where it is.
    let bytes =
        2 * wide + u64::from(BENCH_ROWS) * u64::from(HIDDEN) * 4 + if scaled { 0 } else { wide };
    let kernel = combine_norm_kernel(gpu);
    Fixture::new(bytes, 0, move |gpu| {
        if scaled {
            launch(
                gpu,
                HC,
                "q4e_hc_combine_scale_kernel",
                (combine_grid(BENCH_ROWS), 256, 0),
                &[
                    KernelArg::Buffer(res.view()),
                    KernelArg::Buffer(scale.view()),
                    KernelArg::Buffer(inputs.block_out.view()),
                    KernelArg::Buffer(inputs.inject.view()),
                    KernelArg::U32(HIDDEN),
                    KernelArg::U32(STREAMS),
                    KernelArg::F32(1.0e-6),
                ],
            );
        } else {
            run(
                gpu,
                kernel,
                &LaunchArgs::new(
                    combine_grid(BENCH_ROWS),
                    Extent3::new(256, 1, 1),
                    &[
                        KernelArg::Buffer(res.view()),
                        KernelArg::Buffer(xn.view()),
                        KernelArg::Buffer(inputs.block_out.view()),
                        KernelArg::Buffer(inputs.inject.view()),
                        KernelArg::Buffer(inputs.norm.view()),
                        KernelArg::U32(HIDDEN),
                        KernelArg::U32(STREAMS),
                        KernelArg::F32(1.0e-6),
                    ],
                ),
            );
        }
    })
}

/// The combine storing the normed streams at a 4096-row chunk.
pub(crate) fn bench_combine_norm_4096(gpu: &Gpu) -> Fixture {
    bench_combine(gpu, false)
}
/// The combine leaving the norm scale at a 4096-row chunk.
pub(crate) fn bench_combine_scale_4096(gpu: &Gpu) -> Fixture {
    bench_combine(gpu, true)
}

/* ---------------------------------------------------------------------------
 * The step (decode and verify) kernels.
 * ------------------------------------------------------------------------ */

/// Slices the stream mix's contraction (and its injection, and the two
/// stacked) is cut into, mirroring `matmul::bf16_split(10240, ..)`.
const MIX_SPLITS: u32 = 8;
/// Threads a step block runs, mirroring `matmul::BF16_STEP_THREADS`.
const STEP_THREADS: u32 = 256;
/// Block-input elements one fused step block covers (`Q4E_BF_MIX_E`).
const MIX_E: u32 = 8;

/// One step launch over `rows` rows of the plain step kernels: `out, w, x,
/// in_dim, out_dim, splits, rows`.
fn step(gpu: &Gpu, out: &DeviceBuffer, w: &DeviceBuffer, x: &DeviceBuffer, dims: (u32, u32, u32)) {
    let (in_dim, out_dim, rows) = dims;
    let splits = if in_dim == STREAMS * HIDDEN {
        MIX_SPLITS
    } else {
        1
    };
    let per_warp = if rows == 1 { 1 } else { 2 };
    let entry = if rows == 1 {
        "q4e_bf16_gemv_kernel"
    } else {
        "q4e_bf16_rows_kernel"
    };
    launch(
        gpu,
        BF16,
        entry,
        (
            Extent3::new(out_dim.div_ceil(8 / splits * per_warp), 1, 1),
            STEP_THREADS,
            0,
        ),
        &[
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(w.view()),
            KernelArg::Buffer(x.view()),
            KernelArg::U32(in_dim),
            KernelArg::U32(out_dim),
            KernelArg::U32(splits),
            KernelArg::U32(rows),
        ],
    );
}

/// The step pair over the contraction and the injection.
fn step_pair(gpu: &Gpu, p: &Pair<'_>, x: &DeviceBuffer, rows: u32) {
    let per_warp = if rows == 1 { 1 } else { 2 };
    let entry = if rows == 1 {
        "q4e_bf16_gemv_pair_kernel"
    } else {
        "q4e_bf16_rows_pair_kernel"
    };
    launch(
        gpu,
        BF16,
        entry,
        (
            Extent3::new((RANK + INJECT).div_ceil(8 / MIX_SPLITS * per_warp), 1, 1),
            STEP_THREADS,
            0,
        ),
        &[
            KernelArg::Buffer(p.out_a.view()),
            KernelArg::Buffer(p.w_a.view()),
            KernelArg::U32(RANK),
            KernelArg::Buffer(p.out_b.view()),
            KernelArg::Buffer(p.w_b.view()),
            KernelArg::U32(INJECT),
            KernelArg::Buffer(x.view()),
            KernelArg::U32(STREAMS * HIDDEN),
            KernelArg::U32(MIX_SPLITS),
            KernelArg::U32(rows),
        ],
    );
}

/// The fused activation, expansion and collapse step.
fn mix_step(gpu: &Gpu, bufs: &CollapseBufs<'_>, rows: u32) {
    launch(
        gpu,
        BF16,
        "q4e_bf16_mix_step_kernel",
        (Extent3::new(HIDDEN / MIX_E, 1, 1), STEP_THREADS, 0),
        &[
            KernelArg::Buffer(bufs.out.view()),
            KernelArg::Buffer(bufs.w_up.view()),
            KernelArg::Buffer(bufs.x.view()),
            KernelArg::Buffer(bufs.xn.view()),
            KernelArg::F32(1.0 / STREAMS as f32),
            KernelArg::U32(RANK),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(STREAMS),
            KernelArg::U32(rows),
        ],
    );
}

/// The step pair writes, weight for weight, the plain step kernels' bits at
/// one row and every verify width.
///
/// # Panics
///
/// When an output differs, or the device refuses a launch.
pub fn the_step_pair_is_its_two_steps_bit_for_bit(gpu: &Gpu) {
    let in_dim = STREAMS * HIDDEN;
    let w_a = gpu.upload(&matrix(RANK, in_dim, 0xD0));
    let w_b = gpu.upload(&matrix(INJECT, in_dim, 0x1E));
    for rows in 1..=8_u32 {
        let x = gpu.upload(&f32_bytes(&signal((rows * in_dim) as usize, 0x17 ^ rows)));
        let (want_a, want_b) = (
            sentinel(gpu, (rows * RANK) as usize),
            sentinel(gpu, (rows * INJECT) as usize),
        );
        step(gpu, &want_a, &w_a, &x, (in_dim, RANK, rows));
        step(gpu, &want_b, &w_b, &x, (in_dim, INJECT, rows));
        let (got_a, got_b) = (
            sentinel(gpu, (rows * RANK) as usize),
            sentinel(gpu, (rows * INJECT) as usize),
        );
        let p = Pair {
            out_a: &got_a,
            w_a: &w_a,
            out_b: &got_b,
            w_b: &w_b,
        };
        step_pair(gpu, &p, &x, rows);
        drain(gpu);
        same_bits(
            &format!("step contraction at {rows} rows"),
            &gpu.read_f32(&got_a),
            &gpu.read_f32(&want_a),
        );
        same_bits(
            &format!("step injection at {rows} rows"),
            &gpu.read_f32(&got_b),
            &gpu.read_f32(&want_b),
        );
    }
}

/// The host restatement of the fused step, in `f64`.
fn mix_oracle(w_up: &[u8], x: &[f32], xn: &[f32], rows: usize) -> Vec<f64> {
    let (hidden, streams, rank) = (HIDDEN as usize, STREAMS as usize, RANK as usize);
    let w = |o: usize, k: usize| {
        let at = 2 * (o * rank + k);
        f64::from(bf16::from_bits(u16::from_le_bytes([w_up[at], w_up[at + 1]])).to_f32())
    };
    let mut out = vec![0f64; rows * hidden];
    for t in 0..rows {
        let act: Vec<f64> = (0..rank)
            .map(|k| {
                let v = f64::from(x[t * rank + k]) / streams as f64;
                v / (1.0 + (-v).exp())
            })
            .collect();
        for e in 0..hidden {
            let mut sum = 0.0;
            for h in 0..streams {
                let o = h * hidden + e;
                let up: f64 = (0..rank).map(|k| w(o, k) * act[k]).sum();
                sum += f64::from(xn[t * streams * hidden + o]) / (1.0 + (-up).exp());
            }
            out[t * hidden + e] = sum / streams as f64;
        }
    }
    out
}

/// The fused step agrees with the host restatement (the gates and the
/// activation take the device's approximate exponential, so to a few parts
/// in a million of the streams' magnitude), and each verify row is the single
/// step's row bit for bit.
///
/// # Panics
///
/// When an output differs, or the device refuses a launch.
pub fn the_fused_mix_step_is_the_mix_and_a_verify_row_is_the_step(gpu: &Gpu) {
    let wide = STREAMS * HIDDEN;
    let w_bytes = matrix(wide, RANK, 0x0B);
    let w_up = gpu.upload(&w_bytes);
    let x: Vec<f32> = signal((ROWS * RANK) as usize, 0x55)
        .iter()
        .map(|v| 8.0 * v)
        .collect();
    let xn = signal((ROWS * wide) as usize, 0x99);
    let xs = gpu.upload(&f32_bytes(&x));
    let xns = gpu.upload(&f32_bytes(&xn));
    let mut alone = Vec::new();
    for t in 0..ROWS as usize {
        let xr = gpu.upload(&f32_bytes(&x[t * RANK as usize..(t + 1) * RANK as usize]));
        let xnr = gpu.upload(&f32_bytes(&xn[t * wide as usize..(t + 1) * wide as usize]));
        let out = sentinel(gpu, HIDDEN as usize);
        mix_step(
            gpu,
            &CollapseBufs {
                out: &out,
                w_up: &w_up,
                x: &xr,
                xn: &xnr,
            },
            1,
        );
        drain(gpu);
        alone.push(gpu.read_f32(&out));
    }
    let want = mix_oracle(&w_bytes, &x, &xn, ROWS as usize);
    for (t, row) in alone.iter().enumerate() {
        for (e, &g) in row.iter().enumerate() {
            let w = want[t * HIDDEN as usize + e];
            assert!(
                (f64::from(g) - w).abs() <= 1e-5 * (1.0 + w.abs()),
                "row {t} element {e}: {g} vs {w}"
            );
        }
    }
    for rows in 2..=ROWS {
        let out = sentinel(gpu, (rows * HIDDEN) as usize);
        mix_step(
            gpu,
            &CollapseBufs {
                out: &out,
                w_up: &w_up,
                x: &xs,
                xn: &xns,
            },
            rows,
        );
        drain(gpu);
        let got = gpu.read_f32(&out);
        for (t, row) in alone.iter().enumerate().take(rows as usize) {
            same_bits(
                &format!("row {t} of {rows}"),
                &got[t * HIDDEN as usize..(t + 1) * HIDDEN as usize],
                row,
            );
        }
    }
}

/// Rows the fused step case checks, the widest step launch.
const ROWS: u32 = 8;

/// The working set the step fixtures rotate over (see the brain-float
/// fixtures): weights never found in a cache.
const STEP_ROTATION_BYTES: usize = 2 << 30;

/// Copies of a `bytes`-byte weight that fill [`STEP_ROTATION_BYTES`].
fn step_copies(bytes: usize) -> usize {
    STEP_ROTATION_BYTES.div_ceil(bytes.max(1)).max(2)
}

/// The mix's contraction and injection at `rows` step rows: one pair launch,
/// or today's two.
fn bench_step_down(gpu: &Gpu, rows: u32, fused: bool) -> Fixture {
    let in_dim = STREAMS * HIDDEN;
    let (a, b) = (matrix(RANK, in_dim, 0xD0), matrix(INJECT, in_dim, 0x1E));
    let sets: Vec<(DeviceBuffer, DeviceBuffer)> = (0..step_copies(a.len() + b.len()))
        .map(|_| (gpu.upload(&a), gpu.upload(&b)))
        .collect();
    let x = gpu.upload(&f32_bytes(&signal((rows * in_dim) as usize, 0x77)));
    let out_a = gpu.upload(&vec![0u8; (rows * RANK * 4) as usize]);
    let out_b = gpu.upload(&vec![0u8; (rows * INJECT * 4) as usize]);
    let bytes = (a.len() + b.len()) as u64;
    let next = std::cell::Cell::new(0usize);
    Fixture::new(bytes, 0, move |gpu| {
        let (w_a, w_b) = &sets[next.get() % sets.len()];
        next.set(next.get().wrapping_add(1));
        if fused {
            let p = Pair {
                out_a: &out_a,
                w_a,
                out_b: &out_b,
                w_b,
            };
            step_pair(gpu, &p, &x, rows);
        } else {
            step(gpu, &out_a, w_a, &x, (in_dim, RANK, rows));
            step(gpu, &out_b, w_b, &x, (in_dim, INJECT, rows));
        }
    })
}

/// `q4e_scale_silu_kernel` over `n` values in place.
fn scale_silu(gpu: &Gpu, x: &DeviceBuffer, n: u32) {
    launch(
        gpu,
        HC,
        "q4e_scale_silu_kernel",
        (Extent3::new(n.div_ceil(256), 1, 1), 256, 0),
        &[
            KernelArg::Buffer(x.view()),
            KernelArg::F32(1.0 / STREAMS as f32),
            KernelArg::U64(u64::from(n)),
        ],
    );
}

/// The mix's activation, expansion and collapse at `rows` step rows: the
/// fused step, or today's three launches.
fn bench_step_mix(gpu: &Gpu, rows: u32, fused: bool) -> Fixture {
    let wide = STREAMS * HIDDEN;
    let w = matrix(wide, RANK, 0x0B);
    let sets: Vec<DeviceBuffer> = (0..step_copies(w.len())).map(|_| gpu.upload(&w)).collect();
    let x = gpu.upload(&f32_bytes(&signal((rows * RANK) as usize, 0x55)));
    let xn = gpu.upload(&f32_bytes(&signal((rows * wide) as usize, 0x99)));
    let up = gpu.upload(&vec![0u8; (rows * wide * 4) as usize]);
    let out = gpu.upload(&vec![0u8; (rows * HIDDEN * 4) as usize]);
    let bytes = w.len() as u64;
    let next = std::cell::Cell::new(0usize);
    Fixture::new(bytes, 0, move |gpu| {
        let w_up = &sets[next.get() % sets.len()];
        next.set(next.get().wrapping_add(1));
        let bufs = CollapseBufs {
            out: &out,
            w_up,
            x: &x,
            xn: &xn,
        };
        if fused {
            mix_step(gpu, &bufs, rows);
        } else {
            scale_silu(gpu, &x, rows * RANK);
            step(gpu, &up, w_up, &x, (RANK, wide, rows));
            run(
                gpu,
                collapse_kernel(gpu),
                &LaunchArgs::new(
                    Extent3::new(rows, 1, 1),
                    Extent3::new(256, 1, 1),
                    &[
                        KernelArg::Buffer(out.view()),
                        KernelArg::Buffer(xn.view()),
                        KernelArg::Buffer(up.view()),
                        KernelArg::U32(HIDDEN),
                        KernelArg::U32(STREAMS),
                    ],
                ),
            );
        }
    })
}

/// The step pair at one row.
pub(crate) fn bench_step_pair_t1(gpu: &Gpu) -> Fixture {
    bench_step_down(gpu, 1, true)
}
/// The contraction and injection as two steps at one row.
pub(crate) fn bench_step_down_inject_t1(gpu: &Gpu) -> Fixture {
    bench_step_down(gpu, 1, false)
}
/// The step pair over a four-row verify.
pub(crate) fn bench_step_pair_t4(gpu: &Gpu) -> Fixture {
    bench_step_down(gpu, 4, true)
}
/// The same as two launches.
pub(crate) fn bench_step_down_inject_t4(gpu: &Gpu) -> Fixture {
    bench_step_down(gpu, 4, false)
}
/// The fused mix step at one row.
pub(crate) fn bench_step_mix_t1(gpu: &Gpu) -> Fixture {
    bench_step_mix(gpu, 1, true)
}
/// The activation, expansion and collapse as three launches at one row.
pub(crate) fn bench_step_up_collapse_t1(gpu: &Gpu) -> Fixture {
    bench_step_mix(gpu, 1, false)
}
/// The fused mix step over a four-row verify.
pub(crate) fn bench_step_mix_t4(gpu: &Gpu) -> Fixture {
    bench_step_mix(gpu, 4, true)
}
/// The same as three launches.
pub(crate) fn bench_step_up_collapse_t4(gpu: &Gpu) -> Fixture {
    bench_step_mix(gpu, 4, false)
}

/// Rows of the bench's prompt chunk.
const BENCH_ROWS: u32 = 4_096;

/// The mix's contraction and injection over a 4096-row chunk: as one pair
/// launch, or (`fused == false`) as today's two launches.
fn bench_down(gpu: &Gpu, fused: bool) -> Fixture {
    let in_dim = STREAMS * HIDDEN;
    let w_a = gpu.upload(&matrix(RANK, in_dim, 0xD0));
    let w_b = gpu.upload(&matrix(INJECT, in_dim, 0x1E));
    let x = gpu.upload(&f32_bytes(&signal((BENCH_ROWS * in_dim) as usize, 0x77)));
    let out_a = gpu.upload(&vec![0u8; (BENCH_ROWS * RANK * 4) as usize]);
    let out_b = gpu.upload(&vec![0u8; (BENCH_ROWS * INJECT * 4) as usize]);
    let bytes = 4 * u64::from(BENCH_ROWS) * u64::from(in_dim + RANK + INJECT);
    let flops = 2 * u64::from(BENCH_ROWS) * u64::from(in_dim) * u64::from(RANK + INJECT);
    Fixture::new(bytes, flops, move |gpu| {
        if fused {
            let p = Pair {
                out_a: &out_a,
                w_a: &w_a,
                out_b: &out_b,
                w_b: &w_b,
            };
            pair(gpu, &p, &x, BENCH_ROWS);
        } else {
            gemm(gpu, &out_a, &w_a, &x, (in_dim, RANK, BENCH_ROWS));
            gemm(gpu, &out_b, &w_b, &x, (in_dim, INJECT, BENCH_ROWS));
        }
    })
}

/// The mix's expansion and collapse over a 4096-row chunk: fused, or as
/// today's two launches through the expanded streams.
fn bench_collapse(gpu: &Gpu, fused: bool) -> Fixture {
    let wide = STREAMS * HIDDEN;
    let w_up = gpu.upload(&matrix(wide, RANK, 0x0B));
    let x = gpu.upload(&f32_bytes(&signal((BENCH_ROWS * RANK) as usize, 0x55)));
    let xn = gpu.upload(&f32_bytes(&signal((BENCH_ROWS * wide) as usize, 0x99)));
    let up = gpu.upload(&vec![0u8; (BENCH_ROWS * wide * 4) as usize]);
    let out = gpu.upload(&vec![0u8; (BENCH_ROWS * HIDDEN * 4) as usize]);
    // The floor either way: the normed streams read once, the block input
    // written once.
    let bytes = 4 * u64::from(BENCH_ROWS) * u64::from(wide + HIDDEN + RANK);
    let flops = 2 * u64::from(BENCH_ROWS) * u64::from(RANK) * u64::from(wide);
    Fixture::new(bytes, flops, move |gpu| {
        let bufs = CollapseBufs {
            out: &out,
            w_up: &w_up,
            x: &x,
            xn: &xn,
        };
        if fused {
            fused_collapse(gpu, &bufs, BENCH_ROWS);
        } else {
            split_collapse(gpu, &bufs, &up, BENCH_ROWS);
        }
    })
}

/// The norming pair and collapse over a 4096-row chunk: the residual, the
/// scale and the norm weight in place of the normed streams.
fn bench_norming(gpu: &Gpu, collapse: bool) -> Fixture {
    let wide = STREAMS * HIDDEN;
    let w_a = gpu.upload(&matrix(RANK, wide, 0xD0));
    let w_b = gpu.upload(&matrix(INJECT, wide, 0x1E));
    let w_up = gpu.upload(&matrix(wide, RANK, 0x0B));
    let res = gpu.upload(&f32_bytes(&signal((BENCH_ROWS * wide) as usize, 0x99)));
    let scale = gpu.upload(&f32_bytes(&vec![0.5f32; (BENCH_ROWS * STREAMS) as usize]));
    let nw = gpu.upload(&f32_bytes(&vec![1.0f32; wide as usize]));
    let x = gpu.upload(&f32_bytes(&signal((BENCH_ROWS * RANK) as usize, 0x55)));
    let low = gpu.upload(&vec![0u8; (BENCH_ROWS * RANK * 4) as usize]);
    let inj = gpu.upload(&vec![0u8; (BENCH_ROWS * INJECT * 4) as usize]);
    let out = gpu.upload(&vec![0u8; (BENCH_ROWS * HIDDEN * 4) as usize]);
    let bytes = 4 * u64::from(BENCH_ROWS) * u64::from(wide + if collapse { HIDDEN } else { RANK });
    Fixture::new(bytes, 0, move |gpu| {
        if collapse {
            launch(
                gpu,
                BF16,
                "q4e_bf16_gemm_collapse_norm_kernel",
                (
                    Extent3::new(HIDDEN / COLLAPSE_E, BENCH_ROWS.div_ceil(TILE), 1),
                    THREADS,
                    SMEM,
                ),
                &[
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(w_up.view()),
                    KernelArg::Buffer(x.view()),
                    KernelArg::Buffer(res.view()),
                    KernelArg::Buffer(scale.view()),
                    KernelArg::Buffer(nw.view()),
                    KernelArg::U32(RANK),
                    KernelArg::U32(HIDDEN),
                    KernelArg::U32(STREAMS),
                    KernelArg::U32(BENCH_ROWS),
                ],
            );
        } else {
            launch(
                gpu,
                BF16,
                "q4e_bf16_gemm_pair_norm_kernel",
                (
                    Extent3::new((RANK + INJECT).div_ceil(TILE), BENCH_ROWS.div_ceil(TILE), 1),
                    THREADS,
                    SMEM,
                ),
                &[
                    KernelArg::Buffer(low.view()),
                    KernelArg::Buffer(w_a.view()),
                    KernelArg::U32(RANK),
                    KernelArg::Buffer(inj.view()),
                    KernelArg::Buffer(w_b.view()),
                    KernelArg::U32(INJECT),
                    KernelArg::Buffer(res.view()),
                    KernelArg::Buffer(scale.view()),
                    KernelArg::Buffer(nw.view()),
                    KernelArg::U32(HIDDEN),
                    KernelArg::U32(wide),
                    KernelArg::U32(BENCH_ROWS),
                ],
            );
        }
    })
}

/// The norming pair at a 4096-row chunk.
pub(crate) fn bench_pair_norm_4096(gpu: &Gpu) -> Fixture {
    bench_norming(gpu, false)
}
/// The norming fused collapse at a 4096-row chunk.
pub(crate) fn bench_collapse_norm_4096(gpu: &Gpu) -> Fixture {
    bench_norming(gpu, true)
}

/// The pair launch at a 4096-row chunk.
pub(crate) fn bench_pair_4096(gpu: &Gpu) -> Fixture {
    bench_down(gpu, true)
}
/// The contraction and the injection as two launches at a 4096-row chunk.
pub(crate) fn bench_down_inject_4096(gpu: &Gpu) -> Fixture {
    bench_down(gpu, false)
}
/// The fused expansion and collapse at a 4096-row chunk.
pub(crate) fn bench_fused_collapse_4096(gpu: &Gpu) -> Fixture {
    bench_collapse(gpu, true)
}
/// The expansion and the collapse as two launches at a 4096-row chunk.
pub(crate) fn bench_up_collapse_4096(gpu: &Gpu) -> Fixture {
    bench_collapse(gpu, false)
}
