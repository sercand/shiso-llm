//! The sparse family's brain-float dense projections on a device with tensor
//! cores (`q4e_bf16`), graded against a host `f64` product and against
//! themselves across pass widths.
//!
//! Two promises are checked, one per kind of pass:
//!
//! - **A verify row is the single-row step's row.** The step kernels
//!   (`q4e_bf16_gemv_kernel`, `q4e_bf16_rows_kernel`,
//!   `q4e_bf16_rows_wide_kernel`) run with the engine's slicing, and every
//!   row of a 2..32-row pass must be the bits the one-row kernel returns
//!   for that activation row alone. They are also held to the host product
//!   within [`STEP_TOLERANCE`] of each row's absolute sum, which is
//!   single-precision accumulation and nothing else.
//! - **A prompt row does not depend on its chunk.** The GEMMs
//!   (`q4e_bf16_gemm_kernel`, `q4e_bf16_gemm_split_kernel`) at widths 1, 7,
//!   64, 128, 129 and 333 must give the 333-wide pass's rows bit for bit. The
//!   plain GEMM rounds its activations to brain floats, so it is graded
//!   against the host product over the rounded activations
//!   ([`GEMM_TOLERANCE`]); the split one carries each activation as two brain
//!   floats and is graded against the unrounded activations
//!   ([`SPLIT_TOLERANCE`]).
//!
//! The geometry is transcribed from `shiso_engine::sparse::matmul`, which this
//! crate may not depend on; each constant names the one it mirrors.
#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture dimensions are model widths far below u32::MAX"
)]

use half::bf16;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::bench::Fixture;
use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The module the entries live in, mirroring `kernels::MOD_BF16`.
const MODULE: &str = "q4e_bf16";
/// The module of the fold and of the ported kernels the benches compare
/// against, mirroring `kernels::MOD_MATVEC`.
const MATVEC: &str = "q4e_matvec";
/// The one-row step kernel.
const GEMV: &str = "q4e_bf16_gemv_kernel";
/// The two-to-eight-row step kernel.
const ROWS: &str = "q4e_bf16_rows_kernel";
/// The nine-to-thirty-two-row step kernel.
const ROWS_WIDE: &str = "q4e_bf16_rows_wide_kernel";
/// The prompt GEMM.
const GEMM: &str = "q4e_bf16_gemm_kernel";
/// The split-activation prompt GEMM.
const GEMM_SPLIT: &str = "q4e_bf16_gemm_split_kernel";
/// Warps a step block runs, mirroring `matmul::BF16_STEP_WARPS`.
const STEP_WARPS: u32 = 8;
/// Threads a step block runs, mirroring `matmul::BF16_STEP_THREADS`.
const STEP_THREADS: u32 = STEP_WARPS * 32;
/// The widest step launch, mirroring `matmul::BF16_STEP_ROWS`.
const STEP_ROWS: u32 = 32;
/// One GEMM tile's side, mirroring `matmul::BF16_GEMM_TILE`.
const GEMM_TILE: u32 = 128;
/// Threads a GEMM tile runs with, mirroring `matmul::BF16_GEMM_THREADS`.
const GEMM_THREADS: u32 = 512;
/// The plain GEMM's dynamic shared memory, mirroring `matmul::BF16_GEMM_SMEM`.
const GEMM_SMEM: u32 = 90_112;
/// The split GEMM's, mirroring `matmul::BF16_GEMM_SPLIT_SMEM`.
const GEMM_SPLIT_SMEM: u32 = 81_920;
/// Threads of the half tier's fold, mirroring `kernels::BLOCK_NARROW`.
const FOLD_THREADS: u32 = 128;
/// Contraction a half-tier split chunk rounds to, mirroring
/// `kernels::BLOCK_LINEAR`.
const CHUNK_QUANTUM: u32 = 256;

/// The checkpoint's brain-float shapes, `(out, in)`: the stream mix down and
/// up, the block injection, the router, the shared router, the n-gram key and
/// value projections, and the draft head's input projections and query.
pub const SHAPES: [(u32, u32); 8] = [
    (320, 10_240),
    (10_240, 320),
    (4, 10_240),
    (512, 2_560),
    (1, 2_560),
    (10_240, 2_560),
    (2_560, 2_560),
    (12_288, 2_560),
];

/// The step kernels' budget against the host product, as a fraction of the
/// row's absolute sum: single-precision fused multiply-adds over a slice, a
/// shuffle fold, the slices' fold. The worst ratio measured over every shape
/// below on a GB10 was 2.9e-8; 1e-6 still fails a kernel that drops or
/// doubles one sixteen-byte load by orders of magnitude.
pub const STEP_TOLERANCE: f64 = 1e-6;

/// The plain GEMM's budget against the host product over brain-float-rounded
/// activations, as a fraction of the row's absolute sum. Only the tensor
/// core's single-precision accumulation is left; the worst ratio measured on
/// a GB10 over every shape and width below was 6.7e-7.
pub const GEMM_TOLERANCE: f64 = 1e-6;

/// The split GEMM's budget against the host product over the *unrounded*
/// activations: what the two brain floats leave of an activation is below
/// 2^-16 of it, and the accumulation adds the rest. Measured worst on a GB10:
/// 1.3e-6 - where rounding the activations to one brain float alone would
/// leave up to 2^-9 of a term.
pub const SPLIT_TOLERANCE: f64 = 5e-6;

/// A brain-float matrix `[out, in]`: its bits and their values.
struct Matrix {
    bits: Vec<u16>,
    values: Vec<f32>,
    in_dim: u32,
    out_dim: u32,
}

impl Matrix {
    fn new(out_dim: u32, in_dim: u32, seed: u32) -> Self {
        let bits: Vec<u16> = signal(out_dim as usize * in_dim as usize, seed)
            .iter()
            .map(|v| bf16::from_f32(v * 0.05).to_bits())
            .collect();
        let values = bits.iter().map(|b| bf16::from_bits(*b).to_f32()).collect();
        Self {
            bits,
            values,
            in_dim,
            out_dim,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        self.bits.iter().flat_map(|b| b.to_le_bytes()).collect()
    }
}

/// A device buffer of NaNs, so an output the kernel never writes fails.
fn sentinel(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![f32::NAN; len]))
}

/// `matmul::bf16_split`, restated.
fn split(in_dim: u32, out_dim: u32) -> u32 {
    let mut split = 1u32;
    while split < STEP_WARPS && out_dim * split < 2_048 && in_dim / (split * 2) >= 256 {
        split *= 2;
    }
    split
}

/// `kernels::f16_split`, restated: the half tier's split, for its baseline.
fn half_split(in_dim: u32, out_dim: u32) -> u32 {
    let mut split = 1u32;
    while out_dim * split < 1_024 && in_dim / (split * 2) >= 256 {
        split *= 2;
    }
    split
}

/// `matmul::half_chunk`, restated.
fn chunk(in_dim: u32, splits: u32) -> u32 {
    in_dim.div_ceil(splits).next_multiple_of(CHUNK_QUANTUM)
}

/// The step entry and its output rows a warp for `rows` activation rows,
/// mirroring `matmul::brain_tensor` and `matmul::bf16_warp_rows`.
fn step_entry(rows: u32) -> (&'static str, u32) {
    match rows {
        1 => (GEMV, 1),
        2..=8 => (ROWS, 2),
        _ => (ROWS_WIDE, 4),
    }
}

/// `q4e_matvec_combine_t_kernel<float>`, as this device names it.
fn fold_kernel(gpu: &Gpu) -> KernelRef {
    let logical = "_Z27q4e_matvec_combine_t_kernelIfEv";
    gpu.device()
        .resolve_entry(MATVEC, logical)
        .unwrap_or_else(|| KernelRef::new(MATVEC, logical))
}

/// One step pass's device buffers.
struct StepPass<'a> {
    out: &'a DeviceBuffer,
    w: &'a DeviceBuffer,
    x: &'a DeviceBuffer,
}

/// The shape of one pass.
#[derive(Debug, Clone, Copy)]
struct PassDesc {
    in_dim: u32,
    out_dim: u32,
    rows: u32,
}

/// Launches one step pass as `matmul::project_one` plans it.
fn launch_step(gpu: &Gpu, bufs: &StepPass<'_>, pass: PassDesc) {
    let splits = split(pass.in_dim, pass.out_dim);
    let (entry, per_warp) = step_entry(pass.rows);
    run(
        gpu,
        KernelRef::new(MODULE, entry),
        &LaunchArgs::new(
            Extent3::new(pass.out_dim.div_ceil(STEP_WARPS / splits * per_warp), 1, 1),
            Extent3::new(STEP_THREADS, 1, 1),
            &[
                KernelArg::Buffer(bufs.out.view()),
                KernelArg::Buffer(bufs.w.view()),
                KernelArg::Buffer(bufs.x.view()),
                KernelArg::U32(pass.in_dim),
                KernelArg::U32(pass.out_dim),
                KernelArg::U32(splits),
                KernelArg::U32(pass.rows),
            ],
        ),
    );
}

/// One step pass over `rows` activation rows; `[rows, out_dim]`.
fn step(gpu: &Gpu, w: &DeviceBuffer, m: &Matrix, x: &[f32], rows: u32) -> Vec<f32> {
    let xs = gpu.upload(&f32_bytes(x));
    let out = sentinel(gpu, rows as usize * m.out_dim as usize);
    let pass = PassDesc {
        in_dim: m.in_dim,
        out_dim: m.out_dim,
        rows,
    };
    launch_step(
        gpu,
        &StepPass {
            out: &out,
            w,
            x: &xs,
        },
        pass,
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// Launches one GEMM over `pass.rows` activation rows into `out`.
fn launch_gemm(
    gpu: &Gpu,
    entry: &'static str,
    out: &DeviceBuffer,
    w: &DeviceBuffer,
    x: &DeviceBuffer,
    pass: PassDesc,
) {
    let smem = if entry == GEMM_SPLIT {
        GEMM_SPLIT_SMEM
    } else {
        GEMM_SMEM
    };
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(w.view()),
        KernelArg::Buffer(x.view()),
        KernelArg::U32(pass.in_dim),
        KernelArg::U32(pass.out_dim),
        KernelArg::U32(pass.rows),
    ];
    let mut launch = LaunchArgs::new(
        Extent3::new(
            pass.out_dim.div_ceil(GEMM_TILE),
            pass.rows.div_ceil(GEMM_TILE),
            1,
        ),
        Extent3::new(GEMM_THREADS, 1, 1),
        &args,
    );
    launch.smem_bytes = smem;
    run(gpu, KernelRef::new(MODULE, entry), &launch);
}

/// One GEMM over `rows` activation rows; `[rows, out_dim]`.
fn gemm(
    gpu: &Gpu,
    entry: &'static str,
    w: &DeviceBuffer,
    m: &Matrix,
    x: &[f32],
    rows: u32,
) -> Vec<f32> {
    let xs = gpu.upload(&f32_bytes(x));
    let out = sentinel(gpu, rows as usize * m.out_dim as usize);
    let pass = PassDesc {
        in_dim: m.in_dim,
        out_dim: m.out_dim,
        rows,
    };
    launch_gemm(gpu, entry, &out, w, &xs, pass);
    drain(gpu);
    gpu.read_f32(&out)
}

/// Asserts two outputs are the same bits, naming the first that is not.
fn same_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{what}: element {i} is {g:e} ({:#010x}), expected {w:e} ({:#010x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

/// Holds `got` (`[rows, out]`) to the host product of `m` with `x` (already
/// rounded as the kernel rounds it) on every `stride`-th output of every
/// `row_step`-th row, within `tolerance` of each row's absolute sum. Returns
/// the worst ratio.
fn against_host(
    what: &str,
    got: &[f32],
    m: &Matrix,
    x: &[f64],
    (row_step, stride): (usize, usize),
    tolerance: f64,
) -> f64 {
    let width = m.in_dim as usize;
    let od = m.out_dim as usize;
    let rows = x.len() / width;
    let mut worst = 0.0f64;
    for t in (0..rows).step_by(row_step) {
        let xr = &x[t * width..(t + 1) * width];
        for o in (0..od).step_by(stride) {
            let wr = &m.values[o * width..(o + 1) * width];
            let (mut sum, mut abs) = (0.0f64, 0.0f64);
            for (w, xv) in wr.iter().zip(xr) {
                let p = f64::from(*w) * xv;
                sum += p;
                abs += p.abs();
            }
            let g = f64::from(got[t * od + o]);
            let err = (g - sum).abs() / abs.max(f64::MIN_POSITIVE);
            worst = worst.max(err);
            assert!(
                err <= tolerance,
                "{what}: row {t} output {o} is {g:e} against {sum:e}, {err:e} of the absolute sum"
            );
        }
    }
    worst
}

/// Every row of a 2..32-row step pass is the single-row step's row bit for
/// bit, at every checkpoint shape, through the engine's slicing; and
/// the one-row step is the host product within [`STEP_TOLERANCE`].
///
/// Returns the worst error against the host, as a fraction of the absolute
/// sum, for the driver to report.
#[must_use]
pub fn a_verify_row_reproduces_the_single_row_step(gpu: &Gpu) -> f64 {
    let mut worst = 0.0f64;
    for (i, &(out_dim, in_dim)) in SHAPES.iter().enumerate() {
        let m = Matrix::new(out_dim, in_dim, 0xB16 + i as u32);
        let w = gpu.upload(&m.bytes());
        let width = in_dim as usize;
        let od = out_dim as usize;
        let x = signal(STEP_ROWS as usize * width, 0x5E9 ^ in_dim ^ out_dim);
        let alone: Vec<Vec<f32>> = (0..STEP_ROWS as usize)
            .map(|t| step(gpu, &w, &m, &x[t * width..(t + 1) * width], 1))
            .collect();
        let flat: Vec<f32> = alone.iter().flatten().copied().collect();
        let xh: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let stride = if out_dim > 1_024 { 13 } else { 1 };
        worst = worst.max(against_host(
            &format!("gemv [{out_dim} x {in_dim}]"),
            &flat,
            &m,
            &xh,
            (3, stride),
            STEP_TOLERANCE,
        ));
        for rows in [2u32, 3, 5, 8, 9, 16, 20, STEP_ROWS] {
            let got = step(gpu, &w, &m, &x[..rows as usize * width], rows);
            for (t, row) in alone.iter().enumerate().take(rows as usize) {
                same_bits(
                    &format!("[{out_dim} x {in_dim}] row {t} of {rows}"),
                    &got[t * od..(t + 1) * od],
                    row,
                );
            }
        }
    }
    worst
}

/// Every row count the step kernels instantiate (1 to 32) resolves and
/// computes its own rows, at a narrow output sliced eight ways.
pub fn every_step_row_count_matches(gpu: &Gpu) {
    let m = Matrix::new(48, 2_560, 0x48);
    let w = gpu.upload(&m.bytes());
    let width = m.in_dim as usize;
    let x = signal(STEP_ROWS as usize * width, 0x1D);
    let one: Vec<f32> = (0..STEP_ROWS as usize)
        .flat_map(|t| step(gpu, &w, &m, &x[t * width..(t + 1) * width], 1))
        .collect();
    for rows in 2..=STEP_ROWS {
        let got = step(gpu, &w, &m, &x[..rows as usize * width], rows);
        same_bits(
            &format!("[48 x 2560] at {rows} rows"),
            &got,
            &one[..rows as usize * 48],
        );
    }
}

/// Both prompt GEMMs at every checkpoint shape and widths 1, 7, 64, 128, 129
/// and 333: each width's rows are bit for bit the 333-wide pass's rows, and
/// the 333-wide pass is the host product within its tolerance.
///
/// Returns the worst errors measured (plain, split), as fractions of the
/// absolute sum, for the driver to report.
#[must_use]
pub fn the_prompt_gemms_hold_their_rows_to_every_width(gpu: &Gpu) -> (f64, f64) {
    const WIDEST: u32 = 333;
    let mut worst = (0.0f64, 0.0f64);
    for (i, &(out_dim, in_dim)) in SHAPES.iter().enumerate() {
        let m = Matrix::new(out_dim, in_dim, 0x6E1 + i as u32);
        let w = gpu.upload(&m.bytes());
        let width = in_dim as usize;
        let od = out_dim as usize;
        let x = signal(WIDEST as usize * width, 0xC0DE ^ in_dim ^ out_dim);
        let stride = if out_dim > 1_024 { 11 } else { 1 };
        for entry in [GEMM, GEMM_SPLIT] {
            let full = gemm(gpu, entry, &w, &m, &x, WIDEST);
            for rows in [1u32, 7, 64, 128, 129] {
                let got = gemm(gpu, entry, &w, &m, &x[..rows as usize * width], rows);
                same_bits(
                    &format!("{entry} [{out_dim} x {in_dim}] width {rows} against 333"),
                    &got,
                    &full[..rows as usize * od],
                );
            }
            let (xr, tolerance): (Vec<f64>, f64) = if entry == GEMM {
                (
                    x.iter()
                        .map(|&v| f64::from(bf16::from_f32(v).to_f32()))
                        .collect(),
                    GEMM_TOLERANCE,
                )
            } else {
                (x.iter().map(|&v| f64::from(v)).collect(), SPLIT_TOLERANCE)
            };
            let err = against_host(
                &format!("{entry} [{out_dim} x {in_dim}]"),
                &full,
                &m,
                &xr,
                (5, stride),
                tolerance,
            );
            if entry == GEMM {
                worst.0 = worst.0.max(err);
            } else {
                worst.1 = worst.1.max(err);
            }
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// benches
// ---------------------------------------------------------------------------

/// Bytes of weight copies a step fixture cycles through, so that no launch
/// finds its weight still in a cache (see the MXFP8 fixtures).
const BENCH_ROTATION_BYTES: usize = 512 << 20;

/// How many copies of a `bytes`-byte weight fill [`BENCH_ROTATION_BYTES`];
/// one for a weight that is itself larger than every cache.
fn copies(bytes: usize) -> usize {
    if bytes >= BENCH_ROTATION_BYTES {
        1
    } else {
        BENCH_ROTATION_BYTES.div_ceil(bytes.max(1)).max(2)
    }
}

/// The step kernels the engine plans for `rows` rows at one shape: the
/// weight read once, the activations read and the output written once.
fn bench_step(gpu: &Gpu, out_dim: u32, in_dim: u32, rows: u32) -> Fixture {
    let m = Matrix::new(out_dim, in_dim, 0xBE7C);
    let bytes_w = m.bytes();
    let weights: Vec<DeviceBuffer> = (0..copies(bytes_w.len()))
        .map(|_| gpu.upload(&bytes_w))
        .collect();
    let x = gpu.upload(&f32_bytes(&signal(rows as usize * in_dim as usize, 0x0B0B)));
    let out = gpu.upload(&vec![0u8; rows as usize * out_dim as usize * 4]);
    let bytes = bytes_w.len() as u64 + 4 * u64::from(rows) * u64::from(in_dim + out_dim);
    let flops = 2 * u64::from(rows) * u64::from(in_dim) * u64::from(out_dim);
    let pass = PassDesc {
        in_dim,
        out_dim,
        rows,
    };
    let next = std::cell::Cell::new(0usize);
    Fixture::new(bytes, flops, move |gpu| {
        let w = &weights[next.get() % weights.len()];
        next.set(next.get().wrapping_add(1));
        launch_step(
            gpu,
            &StepPass {
                out: &out,
                w,
                x: &x,
            },
            pass,
        );
    })
}

/// The prompt GEMM the engine plans at one shape over `rows` rows.
fn bench_gemm(gpu: &Gpu, entry: &'static str, out_dim: u32, in_dim: u32, rows: u32) -> Fixture {
    let m = Matrix::new(out_dim, in_dim, 0xBE7C);
    let w = gpu.upload(&m.bytes());
    let x = gpu.upload(&f32_bytes(&signal(rows as usize * in_dim as usize, 0x0B0B)));
    let out = gpu.upload(&vec![0u8; rows as usize * out_dim as usize * 4]);
    let bytes = 2 * u64::from(in_dim) * u64::from(out_dim)
        + 4 * u64::from(rows) * u64::from(in_dim + out_dim);
    let flops = 2 * u64::from(rows) * u64::from(in_dim) * u64::from(out_dim);
    let pass = PassDesc {
        in_dim,
        out_dim,
        rows,
    };
    Fixture::new(bytes, flops, move |gpu| {
        launch_gemm(gpu, entry, &out, &w, &x, pass);
    })
}

/// The ported warp-per-row kernel (`q4e_matmul_bf16_rows_kernel<rows>`, 16
/// rows at most), the tier this module replaces: over a prompt chunk it is
/// one launch a sixteen-row pass, each re-reading the weight.
fn bench_ported(gpu: &Gpu, out_dim: u32, in_dim: u32, rows: u32) -> Fixture {
    let m = Matrix::new(out_dim, in_dim, 0xBE7C);
    let bytes_w = m.bytes();
    let n_copies = if rows > 16 { 1 } else { copies(bytes_w.len()) };
    let weights: Vec<DeviceBuffer> = (0..n_copies).map(|_| gpu.upload(&bytes_w)).collect();
    let x = gpu.upload(&f32_bytes(&signal(rows as usize * in_dim as usize, 0x0B0B)));
    let out = gpu.upload(&vec![0u8; rows as usize * out_dim as usize * 4]);
    let bytes = bytes_w.len() as u64 + 4 * u64::from(rows) * u64::from(in_dim + out_dim);
    let flops = 2 * u64::from(rows) * u64::from(in_dim) * u64::from(out_dim);
    let pass = rows.min(16);
    let resolve = |nt: u32| {
        let logical = format!("_Z27q4e_matmul_bf16_rows_kernelILi{nt}EEv");
        gpu.device()
            .resolve_entry(MATVEC, &logical)
            .unwrap_or_else(|| KernelRef::new(MATVEC, Box::leak(logical.into_boxed_str())))
    };
    let full = resolve(pass);
    let tail = resolve(if rows.is_multiple_of(pass) {
        pass
    } else {
        rows % pass
    });
    let next = std::cell::Cell::new(0usize);
    Fixture::new(bytes, flops, move |gpu| {
        let w = &weights[next.get() % weights.len()];
        next.set(next.get().wrapping_add(1));
        let mut done = 0u32;
        while done < rows {
            let take = (rows - done).min(pass);
            let kernel = if take == pass { full } else { tail };
            let out_at = u64::from(done) * u64::from(out_dim) * 4;
            let x_at = u64::from(done) * u64::from(in_dim) * 4;
            let out_view = out
                .slice(out_at, out.len() - out_at)
                .expect("the pass is inside the output");
            let x_view = x
                .slice(x_at, x.len() - x_at)
                .expect("the pass is inside the input");
            run(
                gpu,
                kernel,
                &LaunchArgs::new(
                    Extent3::new(out_dim.div_ceil(8), 1, 1),
                    Extent3::new(256, 1, 1),
                    &[
                        KernelArg::Buffer(out_view),
                        KernelArg::Buffer(w.view()),
                        KernelArg::Buffer(x_view),
                        KernelArg::U32(in_dim),
                        KernelArg::U32(out_dim),
                    ],
                ),
            );
            done += take;
        }
    })
}

/// A twelve-bit packed half-precision matrix (`DType::F16Pack12`'s layout)
/// with no exceptions: random sign and mantissa bytes, exponent codes 1 to
/// 15 against a row base of -4. The packed tier's decode baseline.
fn p12_bytes(out_dim: u32, in_dim: u32) -> Vec<u8> {
    let cells = out_dim as usize * in_dim as usize;
    let rows = out_dim as usize;
    let noise = signal(cells, 0x12);
    let mut out: Vec<u8> = noise.iter().map(|v| (v.to_bits() >> 7) as u8).collect();
    out.extend(noise.chunks(2).map(|p| {
        (((p[0].to_bits() >> 3) % 15 + 1) | (((p[1].to_bits() >> 3) % 15 + 1) << 4)) as u8
    }));
    out.extend(std::iter::repeat_n((-4i8).cast_unsigned(), rows));
    out.resize(out.len().next_multiple_of(16), 0);
    out.extend(std::iter::repeat_n(0u8, 4 * (rows + 1)));
    out.resize(out.len().next_multiple_of(16), 0);
    out
}

/// The packed half-precision rows kernel (`q4e_matmul_p12_rows_kernel<rows>`)
/// at one shape, with the half tier's split and fold: what the checkpoint's
/// brain-float matrices ran on while they were widened and repacked at load.
fn bench_p12(gpu: &Gpu, out_dim: u32, in_dim: u32, rows: u32) -> Fixture {
    let packed = p12_bytes(out_dim, in_dim);
    let weights: Vec<DeviceBuffer> = (0..copies(packed.len()))
        .map(|_| gpu.upload(&packed))
        .collect();
    let x = gpu.upload(&f32_bytes(&signal(rows as usize * in_dim as usize, 0x0B0B)));
    let width = rows as usize * out_dim as usize;
    let out = gpu.upload(&vec![0u8; width * 4]);
    let splits = half_split(in_dim, out_dim);
    let partial = gpu.upload(&vec![0u8; splits as usize * width * 4]);
    let bytes = packed.len() as u64 + 4 * u64::from(rows) * u64::from(in_dim + out_dim);
    let flops = 2 * u64::from(rows) * u64::from(in_dim) * u64::from(out_dim);
    let logical = format!("_Z26q4e_matmul_p12_rows_kernelILi{rows}EEv");
    let kernel = gpu
        .device()
        .resolve_entry(MATVEC, &logical)
        .unwrap_or_else(|| KernelRef::new(MATVEC, Box::leak(logical.into_boxed_str())));
    // `kernels::matmul_p12_rows`, restated.
    let per_warp = if rows <= 4 {
        1
    } else if rows <= 16 {
        2
    } else {
        4
    };
    let threads = if per_warp == 1 { 128 } else { 256 };
    let fold = fold_kernel(gpu);
    let chunk_len = chunk(in_dim, splits);
    let next = std::cell::Cell::new(0usize);
    Fixture::new(bytes, flops, move |gpu| {
        let w = &weights[next.get() % weights.len()];
        next.set(next.get().wrapping_add(1));
        let first = if splits > 1 { &partial } else { &out };
        run(
            gpu,
            kernel,
            &LaunchArgs::new(
                Extent3::new(out_dim.div_ceil(threads / 32 * per_warp), splits, 1),
                Extent3::new(threads, 1, 1),
                &[
                    KernelArg::Buffer(first.view()),
                    KernelArg::Buffer(w.view()),
                    KernelArg::Buffer(x.view()),
                    KernelArg::U32(in_dim),
                    KernelArg::U32(out_dim),
                    KernelArg::U32(chunk_len),
                ],
            ),
        );
        if splits > 1 {
            let n = rows * out_dim;
            run(
                gpu,
                fold,
                &LaunchArgs::new(
                    Extent3::new(n.div_ceil(FOLD_THREADS), 1, 1),
                    Extent3::new(FOLD_THREADS, 1, 1),
                    &[
                        KernelArg::Buffer(out.view()),
                        KernelArg::Buffer(partial.view()),
                        KernelArg::U32(n),
                        KernelArg::U32(splits),
                    ],
                ),
            );
        }
    })
}

/// Declares one bench fixture function per `(name, builder call)`.
macro_rules! benches {
    ($($(#[$doc:meta])* $name:ident => $body:expr;)*) => {
        $($(#[$doc])* pub(crate) fn $name(gpu: &Gpu) -> Fixture {
            let build: fn(&Gpu) -> Fixture = $body;
            build(gpu)
        })*
    };
}

benches! {
    /// One-row step at the stream mix down's shape.
    bench_gemv_320x10240 => |g| bench_step(g, 320, 10_240, 1);
    /// One-row step at the stream mix up's shape.
    bench_gemv_10240x320 => |g| bench_step(g, 10_240, 320, 1);
    /// One-row step at the block injection's shape.
    bench_gemv_4x10240 => |g| bench_step(g, 4, 10_240, 1);
    /// One-row step at the router's shape.
    bench_gemv_512x2560 => |g| bench_step(g, 512, 2_560, 1);
    /// One-row step at the n-gram key projection's shape.
    bench_gemv_10240x2560 => |g| bench_step(g, 10_240, 2_560, 1);
    /// One-row step at the n-gram value projection's shape.
    bench_gemv_2560x2560 => |g| bench_step(g, 2_560, 2_560, 1);
    /// One-row step at the output head's shape.
    bench_gemv_248320x2560 => |g| bench_step(g, 248_320, 2_560, 1);
    /// Five-row verify at the stream mix down's shape.
    bench_rows5_320x10240 => |g| bench_step(g, 320, 10_240, 5);
    /// Five-row verify at the stream mix up's shape.
    bench_rows5_10240x320 => |g| bench_step(g, 10_240, 320, 5);
    /// Five-row verify at the router's shape.
    bench_rows5_512x2560 => |g| bench_step(g, 512, 2_560, 5);
    /// Five-row verify at the n-gram key projection's shape.
    bench_rows5_10240x2560 => |g| bench_step(g, 10_240, 2_560, 5);
    /// Eight-row verify at the stream mix down's shape.
    bench_rows8_320x10240 => |g| bench_step(g, 320, 10_240, 8);
    /// Eight-row verify at the n-gram key projection's shape.
    bench_rows8_10240x2560 => |g| bench_step(g, 10_240, 2_560, 8);
    /// Twenty-row batched verify at the stream mix down's shape.
    bench_rows20_320x10240 => |g| bench_step(g, 320, 10_240, 20);
    /// 2048-row prompt chunk at the stream mix down's shape.
    bench_gemm2048_320x10240 => |g| bench_gemm(g, GEMM, 320, 10_240, 2048);
    /// 2048-row prompt chunk at the stream mix up's shape.
    bench_gemm2048_10240x320 => |g| bench_gemm(g, GEMM, 10_240, 320, 2048);
    /// 2048-row prompt chunk at the block injection's shape.
    bench_gemm2048_4x10240 => |g| bench_gemm(g, GEMM, 4, 10_240, 2048);
    /// 2048-row prompt chunk at the router's shape, split activations.
    bench_gemm_split2048_512x2560 => |g| bench_gemm(g, GEMM_SPLIT, 512, 2_560, 2048);
    /// 2048-row prompt chunk at the n-gram key projection's shape.
    bench_gemm2048_10240x2560 => |g| bench_gemm(g, GEMM, 10_240, 2_560, 2048);
    /// 2048-row prompt chunk at the n-gram value projection's shape.
    bench_gemm2048_2560x2560 => |g| bench_gemm(g, GEMM, 2_560, 2_560, 2048);
    /// 2048-row prompt chunk at the draft head's query shape.
    bench_gemm2048_12288x2560 => |g| bench_gemm(g, GEMM, 12_288, 2_560, 2048);
    /// The ported kernel's sixteen-row passes over a 2048-row chunk at the
    /// stream mix down's shape: the prefill this module replaces.
    bench_ported2048_320x10240 => |g| bench_ported(g, 320, 10_240, 2048);
    /// The same at the router's shape.
    bench_ported2048_512x2560 => |g| bench_ported(g, 512, 2_560, 2048);
    /// The ported one-row kernel at the stream mix down's shape.
    bench_ported1_320x10240 => |g| bench_ported(g, 320, 10_240, 1);
    /// The ported one-row kernel at the n-gram key projection's shape.
    bench_ported1_10240x2560 => |g| bench_ported(g, 10_240, 2_560, 1);
    /// The ported one-row kernel at the output head's shape.
    bench_ported1_248320x2560 => |g| bench_ported(g, 248_320, 2_560, 1);
    /// The packed half tier, one row, at the stream mix down's shape.
    bench_p12_1_320x10240 => |g| bench_p12(g, 320, 10_240, 1);
    /// The packed half tier, one row, at the stream mix up's shape.
    bench_p12_1_10240x320 => |g| bench_p12(g, 10_240, 320, 1);
    /// The packed half tier, one row, at the router's shape.
    bench_p12_1_512x2560 => |g| bench_p12(g, 512, 2_560, 1);
    /// The packed half tier, one row, at the n-gram key projection's shape.
    bench_p12_1_10240x2560 => |g| bench_p12(g, 10_240, 2_560, 1);
    /// The packed half tier, five rows, at the stream mix down's shape.
    bench_p12_5_320x10240 => |g| bench_p12(g, 320, 10_240, 5);
    /// The packed half tier, five rows, at the n-gram key projection's shape.
    bench_p12_5_10240x2560 => |g| bench_p12(g, 10_240, 2_560, 5);
}
