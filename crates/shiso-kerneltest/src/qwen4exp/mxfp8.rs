//! The sparse family's MXFP8 dense projections (`q4e_mxfp8`), graded against
//! the brain-float tier and the host decode.
//!
//! An MXFP8 weight is `e4m3` codes under one `ue8m0` scale per 32 weights, and
//! every such weight is exactly a brain float while it is a normal number. So
//! the step kernels are not graded against a tolerance at all: the same
//! matrix, widened on the host by [`dequant_mxfp8`] and rounded to brain
//! floats (a rounding the case asserts is exact), goes through
//! `q4e_matmul_bf16_rows_kernel`, and the MXFP8 kernels must return its bits.
//! That is the design: the decode and verify kernels sum a row in the
//! brain-float kernel's order, so the tier change moves no output.
//!
//! The prompt kernel rounds its activations to halves on tensor cores, as the
//! dense half-precision prompt tier does, so no existing tier shares its sums.
//! It is graded against a host `f64` product over the same half-rounded
//! activations - which leaves only its single-precision accumulation as slack,
//! a budget stated and measured at [`GEMM_TOLERANCE`] - and, bit for bit,
//! against itself across prompt widths: a column comes out the same whether
//! its chunk is 1, 7, 64, 128, 129 or 333 rows wide.
//!
//! The fixtures' scales sit between 2^-15 and 2^-3, the range a checkpoint's
//! weights occupy, so every widened weight and every product is a normal
//! number; the decode case covers the rest of the scale range, subnormal
//! results included.
#![expect(
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc,
    clippy::float_cmp,
    reason = "fixture dimensions are model widths far below u32::MAX; every case \
              asserts by panicking; the decode case compares values that must be \
              exactly equal"
)]

use half::{bf16, f16};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::fp8::e4m3;
use shiso_modelfile::mxfp8::{GROUP, Mxfp8Layout, dequant_mxfp8, ue8m0};

use crate::bench::Fixture;
use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The module the entries live in, mirroring `kernels::MOD_MXFP8`.
const MODULE: &str = "q4e_mxfp8";
/// The brain-float tier's module, mirroring `kernels::MOD_MATVEC`.
const BF16_MODULE: &str = "q4e_matvec";
/// Output rows one block of the narrow one-row and the narrow multi-row
/// kernels covers, mirroring `matmul::MXFP8_ROWS_PER_BLOCK`.
const ROWS_PER_BLOCK: u32 = 4;
/// Output rows one block of the wide multi-row kernel covers, mirroring
/// `matmul::MXFP8_VERIFY_ROWS_PER_BLOCK`.
const VERIFY_ROWS_PER_BLOCK: u32 = 16;
/// Output rows one block of the wide one-row kernel covers, mirroring
/// `matmul::MXFP8_GEMV_ROWS_PER_BLOCK`.
const GEMV_ROWS_PER_BLOCK: u32 = 8;
/// Threads a step-kernel block runs, mirroring `matmul::MXFP8_THREADS`.
const STEP_THREADS: u32 = 128;
/// Output width at or below which a decode row takes the narrow kernel,
/// mirroring `matmul::NARROW_OUTPUT`.
const NARROW_OUTPUT: u32 = 1_024;
/// The wide one-row kernel.
const GEMV: &str = "q4e_mxfp8_gemv_kernel";
/// The narrow one-row kernel.
const GEMV_NARROW: &str = "q4e_mxfp8_gemv_narrow_kernel";
/// The wide multi-row step kernel.
const ROWS: &str = "q4e_mxfp8_rows_kernel";
/// The narrow multi-row step kernel.
const ROWS_NARROW: &str = "q4e_mxfp8_rows_narrow_kernel";
/// The prompt kernel.
const GEMM: &str = "q4e_mxfp8_gemm_kernel";
/// The widest multi-row step launch, mirroring `matmul::MXFP8_STEP_ROWS`.
const STEP_ROWS: u32 = 8;
/// One prompt-kernel tile's side, mirroring `matmul::MXFP8_GEMM_TILE`.
const GEMM_TILE: u32 = 128;
/// Threads the prompt kernel runs a tile with, mirroring
/// `matmul::MXFP8_GEMM_THREADS`.
const GEMM_THREADS: u32 = 512;
/// Dynamic shared memory the prompt kernel takes, mirroring
/// `matmul::MXFP8_GEMM_SMEM`.
const GEMM_SMEM: u32 = 95_232;
/// The brain-float rows kernel's block: eight warps, a row each, mirroring
/// `kernels::BLOCK_LINEAR` and `kernels::ROWS_PER_WARP_BLOCK`.
const BF16_BLOCK: u32 = 256;

/// The shapes the model projects at, `(out, in)`: the attention query and
/// gate, the recurrent input, the recurrent output and expert down, the
/// stream mix, the shared expert and the key and value heads.
pub const SHAPES: [(u32, u32); 7] = [
    (10_240, 2_560),
    (6_144, 2_560),
    (2_560, 6_144),
    (48, 2_560),
    (640, 2_560),
    (12_288, 2_560),
    (512, 2_560),
];

/// The prompt kernel's budget against the host product over half-rounded
/// activations, as a fraction of the row's absolute sum `sum |w x|`.
///
/// The only slack left is single-precision accumulation: 32-term group
/// products inside the matrix unit, then one add per group. The worst ratio
/// measured over every shape and width below on a GB10 was 2.2e-7; 1e-6
/// leaves a factor of four and still fails a kernel that drops or doubles one
/// group, or applies a neighbouring row's scale, by orders of magnitude.
pub const GEMM_TOLERANCE: f64 = 1e-6;

/// A deterministic byte stream, fast enough for a 31-million-weight fixture
/// in a debug build.
fn bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

/// One MXFP8 matrix, `[out, in]`, and its exact widening.
struct Matrix {
    codes: Vec<u8>,
    scales: Vec<u8>,
    /// The decoded weights, row-major.
    values: Vec<f32>,
    in_dim: u32,
    out_dim: u32,
}

impl Matrix {
    /// Random codes (never the two NaN codes) under scales of 2^-15 to 2^-3.
    fn new(out_dim: u32, in_dim: u32, seed: u64) -> Self {
        let n = out_dim as usize * in_dim as usize;
        let codes = bytes(n, seed)
            .into_iter()
            .map(|c| if c & 0x7F == 0x7F { c ^ 0x01 } else { c })
            .collect();
        let scales = bytes(n / GROUP, seed ^ 0x5CA1E)
            .into_iter()
            .map(|b| 112 + b % 13)
            .collect();
        Self::from_parts(out_dim, in_dim, codes, scales)
    }

    fn from_parts(out_dim: u32, in_dim: u32, codes: Vec<u8>, scales: Vec<u8>) -> Self {
        let mut values = Vec::new();
        dequant_mxfp8(
            "fixture",
            Mxfp8Layout {
                rows: out_dim as usize,
                cols: in_dim as usize,
            },
            &codes,
            &scales,
            &mut values,
        )
        .expect("the fixture is a well-formed MXFP8 matrix");
        Self {
            codes,
            scales,
            values,
            in_dim,
            out_dim,
        }
    }

    /// The widened weights as brain-float bit patterns, which must be exact.
    fn bf16_bytes(&self) -> Vec<u8> {
        self.values
            .iter()
            .flat_map(|&v| {
                let b = bf16::from_f32(v);
                assert_eq!(
                    b.to_f32().to_bits(),
                    v.to_bits(),
                    "{v} is not a brain float"
                );
                b.to_bits().to_le_bytes()
            })
            .collect()
    }
}

/// The matrix on the device, codes and scales.
struct Device {
    codes: DeviceBuffer,
    scales: DeviceBuffer,
}

impl Device {
    fn upload(gpu: &Gpu, m: &Matrix) -> Self {
        Self {
            codes: gpu.upload(&m.codes),
            scales: gpu.upload(&m.scales),
        }
    }
}

/// A device buffer of NaNs, so an output the kernel never writes fails the
/// comparison rather than reading as zero.
fn sentinel(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![f32::NAN; len]))
}

/// Grid, threads and dynamic shared memory of one MXFP8 entry, as the engine
/// plans them.
fn geometry(entry: &str, out_dim: u32, rows: u32) -> (Extent3, u32, u32) {
    match entry {
        GEMM => (
            Extent3::new(out_dim.div_ceil(GEMM_TILE), rows.div_ceil(GEMM_TILE), 1),
            GEMM_THREADS,
            GEMM_SMEM,
        ),
        GEMV => (
            Extent3::new(out_dim.div_ceil(GEMV_ROWS_PER_BLOCK), 1, 1),
            STEP_THREADS,
            0,
        ),
        ROWS => (
            Extent3::new(out_dim.div_ceil(VERIFY_ROWS_PER_BLOCK), 1, 1),
            STEP_THREADS,
            0,
        ),
        _ => (
            Extent3::new(out_dim.div_ceil(ROWS_PER_BLOCK), 1, 1),
            STEP_THREADS,
            0,
        ),
    }
}

/// The shape of one pass: the matrix and the activation rows it projects.
#[derive(Debug, Clone, Copy)]
struct PassDesc {
    in_dim: u32,
    out_dim: u32,
    rows: u32,
}

/// Launches one MXFP8 entry over `pass.rows` activation rows into `out`.
fn launch(
    gpu: &Gpu,
    entry: &'static str,
    out: &DeviceBuffer,
    w: &Device,
    x: &DeviceBuffer,
    pass: PassDesc,
) {
    let (grid, threads, smem) = geometry(entry, pass.out_dim, pass.rows);
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(w.codes.view()),
        KernelArg::Buffer(w.scales.view()),
        KernelArg::Buffer(x.view()),
        KernelArg::U32(pass.in_dim),
        KernelArg::U32(pass.out_dim),
        KernelArg::U32(pass.rows),
    ];
    let mut launch = LaunchArgs::new(grid, Extent3::new(threads, 1, 1), &args);
    launch.smem_bytes = smem;
    run(gpu, KernelRef::new(MODULE, entry), &launch);
}

/// Runs one MXFP8 entry over `rows` activation rows; `[rows, out_dim]`.
fn mxfp8(gpu: &Gpu, entry: &'static str, w: &Device, m: &Matrix, x: &[f32], rows: u32) -> Vec<f32> {
    let xs = gpu.upload(&f32_bytes(x));
    let out = sentinel(gpu, rows as usize * m.out_dim as usize);
    let pass = PassDesc {
        in_dim: m.in_dim,
        out_dim: m.out_dim,
        rows,
    };
    launch(gpu, entry, &out, w, &xs, pass);
    drain(gpu);
    gpu.read_f32(&out)
}

/// The step entry the engine plans for `rows` activation rows into
/// `out_dim` outputs.
fn step_entry(rows: u32, out_dim: u32) -> &'static str {
    match rows {
        1 if out_dim > NARROW_OUTPUT => GEMV,
        1 => GEMV_NARROW,
        _ if out_dim > NARROW_OUTPUT => ROWS,
        _ => ROWS_NARROW,
    }
}

/// `q4e_matmul_bf16_rows_kernel<rows>`, as this device names it.
///
/// The mangled name without its parameter list is how the Metal set exports
/// the instantiation; a CUDA module's entry carries the parameter types after
/// it, so the backend is asked for the one entry that contains it.
fn bf16_rows(gpu: &Gpu, rows: u32) -> KernelRef {
    let logical = format!("_Z27q4e_matmul_bf16_rows_kernelILi{rows}EEv");
    gpu.device()
        .resolve_entry(BF16_MODULE, &logical)
        .unwrap_or_else(|| KernelRef::new(BF16_MODULE, Box::leak(logical.into_boxed_str())))
}

/// The brain-float tier over the widened matrix; `[rows, out_dim]`.
fn bf16_tier(gpu: &Gpu, wide: &DeviceBuffer, m: &Matrix, x: &[f32], rows: u32) -> Vec<f32> {
    let xs = gpu.upload(&f32_bytes(x));
    let out = sentinel(gpu, rows as usize * m.out_dim as usize);
    run(
        gpu,
        bf16_rows(gpu, rows),
        &LaunchArgs::new(
            Extent3::new(m.out_dim.div_ceil(BF16_BLOCK / 32), 1, 1),
            Extent3::new(BF16_BLOCK, 1, 1),
            &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(wide.view()),
                KernelArg::Buffer(xs.view()),
                KernelArg::U32(m.in_dim),
                KernelArg::U32(m.out_dim),
            ],
        ),
    );
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

/// Every one of the 256 codes at the scale bytes 1, 100, 127, 140 and 254
/// comes back from the kernels exactly as the host decode makes it: the
/// subnormal results at 2^-126 kept, the overflow at 2^127 an infinity, the
/// two NaN codes NaN.
///
/// Row `r` holds code `r % 256` in its first column and zeros after it, under
/// scale `r / 256`; the activation is one in the first column, so each output
/// is one weight times one. The prompt kernel gets the same rows: one times a
/// half is exact, and its group sum is that one weight.
pub fn every_code_decodes_exactly_at_every_scale(gpu: &Gpu) {
    const SCALES: [u8; 5] = [1, 100, 127, 140, 254];
    let in_dim = 64u32;
    let out_dim = 256 * SCALES.len() as u32;
    let mut codes = vec![0u8; out_dim as usize * in_dim as usize];
    let mut scales = vec![0u8; out_dim as usize * (in_dim as usize / GROUP)];
    for r in 0..out_dim as usize {
        codes[r * in_dim as usize] = (r % 256) as u8;
        // The second group's scale is its own, so a kernel that read the
        // wrong group's byte would scale the first column by it.
        scales[r * 2] = SCALES[r / 256];
        scales[r * 2 + 1] = 127;
    }
    let m = Matrix::from_parts(out_dim, in_dim, codes, scales);
    let want: Vec<f32> = (0..out_dim as usize)
        .map(|r| {
            let byte = SCALES[r / 256];
            e4m3((r % 256) as u8) * ue8m0(byte).expect("no NaN scale")
        })
        .collect();
    let dev = Device::upload(gpu, &m);
    let mut x = vec![0.0f32; in_dim as usize];
    x[0] = 1.0;
    let check = |what: &str, got: &[f32]| {
        for (r, (g, w)) in got.iter().zip(&want).enumerate() {
            let ok = if w.is_nan() { g.is_nan() } else { g == w };
            assert!(
                ok,
                "{what}: code {:#04x} at scale byte {} is {g:e}, expected {w:e}",
                r % 256,
                SCALES[r / 256]
            );
        }
    };
    check("gemv", &mxfp8(gpu, GEMV, &dev, &m, &x, 1));
    check("gemv narrow", &mxfp8(gpu, GEMV_NARROW, &dev, &m, &x, 1));
    let x3: Vec<f32> = x
        .iter()
        .cycle()
        .take(3 * in_dim as usize)
        .copied()
        .collect();
    for entry in [ROWS, ROWS_NARROW] {
        let rows = mxfp8(gpu, entry, &dev, &m, &x3, 3);
        for t in 0..3 {
            check(
                &format!("{entry} row {t} of 3"),
                &rows[t * out_dim as usize..(t + 1) * out_dim as usize],
            );
        }
    }
    check("gemm", &mxfp8(gpu, GEMM, &dev, &m, &x, 1));
}

/// The decode and verify kernels return the brain-float tier's bits on the
/// widened matrix, at every model shape and at one to five rows and the
/// widest verify launch.
pub fn the_step_kernels_are_the_brain_float_tier_bit_for_bit(gpu: &Gpu) {
    for (i, &(out_dim, in_dim)) in SHAPES.iter().enumerate() {
        let m = Matrix::new(out_dim, in_dim, 0x3F8 + i as u64);
        let dev = Device::upload(gpu, &m);
        let wide = gpu.upload(&m.bf16_bytes());
        for rows in [1, 2, 3, 4, 5, STEP_ROWS] {
            let x = signal(rows as usize * in_dim as usize, 0xA11 ^ out_dim ^ rows);
            let want = bf16_tier(gpu, &wide, &m, &x, rows);
            // Both blockings at every shape, not only the one the engine
            // plans there: neither may move a bit.
            let entries: &[&'static str] = if rows == 1 {
                &[GEMV, GEMV_NARROW]
            } else {
                &[ROWS, ROWS_NARROW]
            };
            for &entry in entries {
                let got = mxfp8(gpu, entry, &dev, &m, &x, rows);
                same_bits(
                    &format!("{entry} [{out_dim} x {in_dim}] at {rows} rows"),
                    &got,
                    &want,
                );
            }
        }
    }
}

/// A verify row is the single-row step's row: each of the multi-row kernel's
/// rows, and each pass of a wider batch, returns the bits the one-row kernel
/// returns for that activation row alone.
pub fn a_verify_row_reproduces_the_single_row_step(gpu: &Gpu) {
    for &(out_dim, in_dim) in &[
        (10_240u32, 2_560u32),
        (2_560, 6_144),
        (48, 2_560),
        (2_560, 640),
    ] {
        let m = Matrix::new(out_dim, in_dim, 0x51E9 ^ u64::from(out_dim));
        let dev = Device::upload(gpu, &m);
        let x = signal(STEP_ROWS as usize * in_dim as usize, 0xBA7C ^ in_dim);
        let width = in_dim as usize;
        let alone: Vec<Vec<f32>> = (0..STEP_ROWS as usize)
            .map(|t| {
                let row = &x[t * width..(t + 1) * width];
                mxfp8(gpu, step_entry(1, out_dim), &dev, &m, row, 1)
            })
            .collect();
        for rows in 2..=STEP_ROWS {
            let batch = &x[..rows as usize * width];
            let got = mxfp8(gpu, step_entry(rows, out_dim), &dev, &m, batch, rows);
            for (t, row) in alone.iter().enumerate().take(rows as usize) {
                same_bits(
                    &format!("[{out_dim} x {in_dim}] row {t} of {rows}"),
                    &got[t * out_dim as usize..(t + 1) * out_dim as usize],
                    row,
                );
            }
        }
    }
}

/// The prompt kernel at every model shape and prompt widths 1, 7, 64, 128,
/// 129 and 333:
/// each width's rows are bit for bit the 333-wide pass's rows, and the
/// 333-wide pass is the host product over half-rounded activations within
/// [`GEMM_TOLERANCE`] of each row's absolute sum, checked on every seventh
/// output (all of them at the narrow shapes) so the host side stays a few
/// seconds.
///
/// Returns the worst error measured, as a fraction of the absolute sum, for
/// the driver to report.
#[must_use]
pub fn the_prompt_kernel_holds_its_rows_to_every_width(gpu: &Gpu) -> f64 {
    const WIDEST: u32 = 333;
    let mut worst = 0.0f64;
    for (i, &(out_dim, in_dim)) in SHAPES.iter().enumerate() {
        let m = Matrix::new(out_dim, in_dim, 0x6E44 + i as u64);
        let dev = Device::upload(gpu, &m);
        let width = in_dim as usize;
        let od = out_dim as usize;
        let x = signal(WIDEST as usize * width, 0xC0DE ^ in_dim ^ out_dim);
        let full = mxfp8(gpu, GEMM, &dev, &m, &x, WIDEST);
        for rows in [1u32, 7, 64, 128, 129] {
            let got = mxfp8(gpu, GEMM, &dev, &m, &x[..rows as usize * width], rows);
            same_bits(
                &format!("[{out_dim} x {in_dim}] prompt width {rows} against 333"),
                &got,
                &full[..rows as usize * od],
            );
        }
        let stride = if out_dim > 1_024 { 7 } else { 1 };
        let xh: Vec<f64> = x
            .iter()
            .map(|&v| f64::from(f16::from_f32(v).to_f32()))
            .collect();
        for t in (0..WIDEST as usize).step_by(5) {
            let xr = &xh[t * width..(t + 1) * width];
            for o in (0..od).step_by(stride) {
                let wr = &m.values[o * width..(o + 1) * width];
                let (mut sum, mut abs) = (0.0f64, 0.0f64);
                for (w, xv) in wr.iter().zip(xr) {
                    let p = f64::from(*w) * xv;
                    sum += p;
                    abs += p.abs();
                }
                let got = f64::from(full[t * od + o]);
                let err = (got - sum).abs() / abs.max(f64::MIN_POSITIVE);
                worst = worst.max(err);
                assert!(
                    err <= GEMM_TOLERANCE,
                    "[{out_dim} x {in_dim}] column {t} row {o}: {got:e} against {sum:e}, \
                     {err:e} of the absolute sum"
                );
            }
        }
    }
    worst
}

/// Bytes of weight copies a bench fixture cycles through, so that no launch
/// finds its weight still in a cache: a decode step reads every layer's
/// matrix once, and a fixture that re-read one 26 MB matrix measured 135% of
/// a GB10's DRAM bandwidth from the last-level caches.
const BENCH_ROTATION_BYTES: usize = 512 << 20;

/// How many copies of a `bytes`-byte weight fill [`BENCH_ROTATION_BYTES`].
fn copies(bytes: usize) -> usize {
    BENCH_ROTATION_BYTES.div_ceil(bytes.max(1)).max(2)
}

/// The kernel the engine plans for `rows` activation rows at one shape, for
/// the kernel bench: codes, scales and the activations read once, the output
/// written once.
///
/// A step pass (at most [`STEP_ROWS`] rows) takes its step entry and a
/// different copy of the weight each launch, since it is bound by the weight
/// read; a wider pass takes the prompt kernel, which is bound by its
/// arithmetic and reuses each weight tile across many columns, so two copies
/// keep the cache honest without spending memory on a fixture of 2048 rows.
fn bench_entry(gpu: &Gpu, out_dim: u32, in_dim: u32, rows: u32) -> Fixture {
    let m = Matrix::new(out_dim, in_dim, 0xBE7C);
    let (entry, n_copies) = if rows > STEP_ROWS {
        (GEMM, 2)
    } else {
        (step_entry(rows, out_dim), copies(m.codes.len()))
    };
    let weights: Vec<Device> = (0..n_copies).map(|_| Device::upload(gpu, &m)).collect();
    let x = gpu.upload(&f32_bytes(&signal(rows as usize * in_dim as usize, 0x0B0B)));
    let out = gpu.upload(&vec![0u8; rows as usize * out_dim as usize * 4]);
    let bytes =
        (m.codes.len() + m.scales.len()) as u64 + 4 * u64::from(rows) * u64::from(in_dim + out_dim);
    let flops = 2 * u64::from(rows) * u64::from(in_dim) * u64::from(out_dim);
    let pass = PassDesc {
        in_dim,
        out_dim,
        rows,
    };
    let next = std::cell::Cell::new(0usize);
    Fixture::new(bytes, flops, move |gpu| {
        let dev = &weights[next.get() % weights.len()];
        next.set(next.get().wrapping_add(1));
        launch(gpu, entry, &out, dev, &x, pass);
    })
}

/// The brain-float tier's decode at one shape, the MXFP8 decode's baseline:
/// twice the weight bytes through the kernel whose sums the MXFP8 one keeps,
/// rotated the same way.
fn bench_bf16(gpu: &Gpu, out_dim: u32, in_dim: u32) -> Fixture {
    let m = Matrix::new(out_dim, in_dim, 0xBE7C);
    let wide_bytes = m.bf16_bytes();
    let weights: Vec<DeviceBuffer> = (0..copies(wide_bytes.len()))
        .map(|_| gpu.upload(&wide_bytes))
        .collect();
    let x = gpu.upload(&f32_bytes(&signal(in_dim as usize, 0x0B0B)));
    let out = gpu.upload(&vec![0u8; out_dim as usize * 4]);
    let bytes = 2 * m.codes.len() as u64 + 4 * u64::from(in_dim + out_dim);
    let flops = 2 * u64::from(in_dim) * u64::from(out_dim);
    // Resolved once: a lookup by name per launch is host time the wall clock
    // would charge to the kernel.
    let kernel = bf16_rows(gpu, 1);
    let next = std::cell::Cell::new(0usize);
    Fixture::new(bytes, flops, move |gpu| {
        let wide = &weights[next.get() % weights.len()];
        next.set(next.get().wrapping_add(1));
        run(
            gpu,
            kernel,
            &LaunchArgs::new(
                Extent3::new(out_dim.div_ceil(BF16_BLOCK / 32), 1, 1),
                Extent3::new(BF16_BLOCK, 1, 1),
                &[
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(wide.view()),
                    KernelArg::Buffer(x.view()),
                    KernelArg::U32(in_dim),
                    KernelArg::U32(out_dim),
                ],
            ),
        );
    })
}

/// `q4e_mxfp8_gemv_kernel` at the recurrent input's shape.
pub(crate) fn bench_gemv_10240x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 10_240, 2_560, 1)
}

/// `q4e_mxfp8_gemv_kernel` at the recurrent output's shape.
pub(crate) fn bench_gemv_2560x6144(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 2_560, 6_144, 1)
}

/// `q4e_mxfp8_gemv_kernel` at the recurrent gate's shape.
pub(crate) fn bench_gemv_6144x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 6_144, 2_560, 1)
}

/// `q4e_mxfp8_rows_kernel` over 2 rows at the recurrent input's shape.
pub(crate) fn bench_rows2_10240x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 10_240, 2_560, 2)
}

/// `q4e_mxfp8_rows_kernel` over 2 rows at the recurrent output's shape.
pub(crate) fn bench_rows2_2560x6144(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 2_560, 6_144, 2)
}

/// `q4e_mxfp8_rows_kernel` over 5 rows at the recurrent input's shape.
pub(crate) fn bench_rows5_10240x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 10_240, 2_560, 5)
}

/// `q4e_mxfp8_rows_kernel` over 5 rows at the recurrent output's shape.
pub(crate) fn bench_rows5_2560x6144(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 2_560, 6_144, 5)
}

/// `q4e_mxfp8_rows_kernel` over 8 rows at the recurrent input's shape.
pub(crate) fn bench_rows8_10240x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 10_240, 2_560, 8)
}

/// `q4e_mxfp8_rows_kernel` over 8 rows at the recurrent output's shape.
pub(crate) fn bench_rows8_2560x6144(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 2_560, 6_144, 8)
}

/// `q4e_mxfp8_rows_kernel` over 5 rows at the recurrent gate's shape.
pub(crate) fn bench_rows5_6144x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 6_144, 2_560, 5)
}

/// `q4e_mxfp8_rows_kernel` over 5 rows at the attention query's shape.
pub(crate) fn bench_rows5_12288x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 12_288, 2_560, 5)
}

/// `q4e_mxfp8_rows_narrow_kernel` over 5 rows at a key head's shape.
pub(crate) fn bench_rows5_512x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 512, 2_560, 5)
}

/// `q4e_mxfp8_rows_narrow_kernel` over 5 rows at the shared expert's shape.
pub(crate) fn bench_rows5_640x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 640, 2_560, 5)
}

/// `q4e_mxfp8_rows_narrow_kernel` over 5 rows at the stream mix's shape.
pub(crate) fn bench_rows5_48x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 48, 2_560, 5)
}

/// `q4e_mxfp8_gemm_kernel` over a 2048-row prompt chunk at 10240 x 2560.
pub(crate) fn bench_gemm2048_10240x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 10_240, 2_560, 2048)
}

/// `q4e_mxfp8_gemm_kernel` over a 2048-row prompt chunk at 12288 x 2560.
pub(crate) fn bench_gemm2048_12288x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 12_288, 2_560, 2048)
}

/// `q4e_mxfp8_gemm_kernel` over a 2048-row prompt chunk at 6144 x 2560.
pub(crate) fn bench_gemm2048_6144x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 6_144, 2_560, 2048)
}

/// `q4e_mxfp8_gemm_kernel` over a 2048-row prompt chunk at 2560 x 6144.
pub(crate) fn bench_gemm2048_2560x6144(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 2_560, 6_144, 2048)
}

/// `q4e_mxfp8_gemm_kernel` over a 512-row prompt chunk at 10240 x 2560.
pub(crate) fn bench_gemm512_10240x2560(gpu: &Gpu) -> Fixture {
    bench_entry(gpu, 10_240, 2_560, 512)
}

/// The brain-float rows kernel at the attention query's shape.
pub(crate) fn bench_bf16_10240x2560(gpu: &Gpu) -> Fixture {
    bench_bf16(gpu, 10_240, 2_560)
}

/// The brain-float rows kernel at the expert down projection's shape.
pub(crate) fn bench_bf16_2560x6144(gpu: &Gpu) -> Fixture {
    bench_bf16(gpu, 2_560, 6_144)
}
