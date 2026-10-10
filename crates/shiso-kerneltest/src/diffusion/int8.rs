//! The per-row INT8 linear and its `ConvRot` activation rotation.
//!
//! A `ConvRot` checkpoint stores `W_rot = W . H` per 256-column group as signed
//! byte codes `q` with one `f32` scale per weight row, `W_rot[o, i] = q[o, i] *
//! s[o]`, where `H = kron(H4, H4, H4, H4) / 16` is symmetric and orthogonal. At
//! run time each input row is rotated the same way and multiplied against the
//! codes, the row's scale riding the epilogue. These cases grade the three
//! entries that does it:
//!
//! - `kernel_dit_hadamard256_f32` against the explicit 256x256 matrix built
//!   here by Kronecker products (not by butterflies, which is what the kernel
//!   does), against itself (`H . H = I`), and reading the fused linear's
//!   `[a | silu(g) * u]` row;
//! - `kernel_dit_gemm64_i8` and the scalar `kernel_dit_gemm_i8` against a
//!   plain `f64` product over the dequantized weight, with and without a bias,
//!   on shapes that leave ragged tiles and cross many k slices - and against
//!   the wrong reading of the scale (by input column), which must fail;
//! - `kernel_dit_gemm64_fused_i8` against the separate passes it replaces.
#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture dimensions are small literals marshalled as the kernels' u32 argument fields"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "the oracles are written in the formula's own letters (m, n, k, x, q, s), which is what makes them checkable against it"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "fixture indices and codes converted to floats are small, exact in f32"
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, DeviceView, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// Every entry point this file's cases launch.
pub const ENTRIES: [&str; 7] = [
    "kernel_dit_hadamard256_f32",
    "kernel_dit_gemm_i8",
    "kernel_dit_gemm64_i8",
    "kernel_dit_gemm64_fused_i8",
    "kernel_dit_swiglu_f32",
    "kernel_dit_concat_rows_f32",
    "kernel_dit_add_gated_f32",
];

/// The module every entry lives under.
const MODULE: &str = "diffusion";
/// The `ConvRot` group: the width `H` acts on.
const GROUP: usize = 256;
/// Threads per rotation threadgroup (one (row, group) each).
const ROT_THREADS: u32 = 64;
/// Tile edge of the scalar GEMM.
const GEMM_TILE: u32 = 16;
/// Output tile edge of the simdgroup GEMM.
const G64_TILE: u32 = 64;
/// Threads per simdgroup-tile threadgroup.
const G64_THREADS: u32 = 128;
/// Threads per elementwise threadgroup.
const BLOCK: u32 = 64;

// Argument blocks: byte for byte `diffusion_common.h` and the driver's Pods.

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GemmArgs {
    m: u32,
    n: u32,
    k: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GemmFusedArgs {
    m: u32,
    n: u32,
    cols_a: u32,
    inner: u32,
    gu_stride: u32,
    gu_offset: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct HadamardArgs {
    rows: u32,
    cols_a: u32,
    inner: u32,
    gu_stride: u32,
    gu_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SwigluArgs {
    rows: u32,
    inner: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ConcatArgs {
    rows: u32,
    cols_a: u32,
    cols_b: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AddGatedArgs {
    count: u32,
    period: u32,
}

const _: () = assert!(size_of::<GemmArgs>() == 16);
const _: () = assert!(size_of::<GemmFusedArgs>() == 28);
const _: () = assert!(size_of::<HadamardArgs>() == 20);
const _: () = assert!(size_of::<SwigluArgs>() == 12);
const _: () = assert!(size_of::<ConcatArgs>() == 12);
const _: () = assert!(size_of::<AddGatedArgs>() == 8);

// --- fixtures and oracles --------------------------------------------------

/// The explicit group matrix: `H[i][j] = prod_d H4[i_d][j_d] / 16` over the
/// four base-4 digits, i.e. `kron(H4, H4, H4, H4) / 16`.
fn hadamard_matrix() -> Vec<f64> {
    const H4: [[f64; 4]; 4] = [
        [1.0, 1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0, 1.0],
    ];
    let mut h = vec![1.0f64; GROUP * GROUP];
    for i in 0..GROUP {
        for j in 0..GROUP {
            let mut v = 1.0 / 16.0;
            for d in 0..4 {
                v *= H4[(i >> (2 * d)) & 3][(j >> (2 * d)) & 3];
            }
            h[i * GROUP + j] = v;
        }
    }
    h
}

/// `x . H` per 256-group of each `width`-wide row, by the explicit matrix.
fn rotate_host(x: &[f32], width: usize) -> Vec<f32> {
    let h = hadamard_matrix();
    let mut out = vec![0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
        for g in (0..width).step_by(GROUP) {
            for j in 0..GROUP {
                let mut acc = 0f64;
                for i in 0..GROUP {
                    acc += f64::from(row_in[g + i]) * h[i * GROUP + j];
                }
                row_out[g + j] = acc as f32;
            }
        }
    }
    out
}

/// Deterministic signed codes covering the whole `[-127, 127]` range.
fn codes(len: usize, seed: u32) -> Vec<i8> {
    let mut state = seed | 0x2545_F491;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = i32::try_from((state >> 8) % 255).unwrap_or(0) - 127;
            i8::try_from(v).unwrap_or(0)
        })
        .collect()
}

/// Per-row scales spread over a decade, so a scale read from the wrong row
/// changes the answer.
fn row_scales(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.002 + 0.0007 * ((i * 13) % 29) as f32)
        .collect()
}

/// `y[r][o] = sum_i x[r][i] * q[o][i] * s[o] + bias[o]`, in `f64`.
fn product_host(
    x: &[f32],
    q: &[i8],
    s: &[f32],
    bias: Option<&[f32]>,
    shape: (usize, usize, usize),
) -> Vec<f32> {
    let (m, n, k) = shape;
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for o in 0..n {
            let mut acc = 0f64;
            for i in 0..k {
                acc += f64::from(x[r * k + i]) * f64::from(q[o * k + i]);
            }
            let b = bias.map_or(0.0, |b| f64::from(b[o]));
            out[r * n + o] = (acc * f64::from(s[o]) + b) as f32;
        }
    }
    out
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn rms(values: &[f32]) -> f32 {
    let sum: f64 = values.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    (sum / values.len().max(1) as f64).sqrt() as f32
}

/// Asserts agreement within `tol` of each element's magnitude, floored at the
/// set's RMS so an element that cancels to near zero is graded against the
/// size of the terms that produced it.
fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let floor = rms(want).max(f32::MIN_POSITIVE);
    for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = tol * w.abs().max(floor);
        assert!(
            (g - w).abs() <= bound,
            "{what}: element {index} is {g}, expected {w} (tolerance {bound})",
        );
    }
}

/// The two readings must differ well beyond the tolerance, or a case that
/// tells them apart proves nothing.
fn differs(a: &[f32], b: &[f32], what: &str) {
    let floor = rms(b).max(f32::MIN_POSITIVE);
    let apart = a
        .iter()
        .zip(b)
        .filter(|&(&x, &y)| (x - y).abs() > 1e-2 * y.abs().max(floor))
        .count();
    assert!(
        apart * 4 > a.len(),
        "{what}: only {apart} of {} elements tell the readings apart",
        a.len()
    );
}

fn i8_bytes(codes: &[i8]) -> Vec<u8> {
    codes.iter().map(|&c| c.to_le_bytes()[0]).collect()
}

fn zeros(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![0.0f32; len.max(1)]))
}

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

/// One rotation over `[a | swiglu(gu)]` rows into a packed `out`.
struct Rotation<'a> {
    out: DeviceView<'a>,
    a: DeviceView<'a>,
    gu: DeviceView<'a>,
    rows: usize,
    cols_a: usize,
    inner: usize,
    gu_stride: usize,
    gu_offset: usize,
}

fn launch_rotation(gpu: &Gpu, r: &Rotation<'_>) {
    let args = HadamardArgs {
        rows: r.rows as u32,
        cols_a: r.cols_a as u32,
        inner: r.inner as u32,
        gu_stride: r.gu_stride as u32,
        gu_offset: r.gu_offset as u32,
    };
    launch(
        gpu,
        "kernel_dit_hadamard256_f32",
        Extent3::new(((r.cols_a + r.inner) / GROUP) as u32, r.rows as u32, 1),
        Extent3::linear(ROT_THREADS),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(r.out),
            KernelArg::Buffer(r.a),
            KernelArg::Buffer(r.gu),
        ],
    );
}

/// Rotates plain `width`-wide rows on the device and reads them back.
fn rotate_device(gpu: &Gpu, x: &[f32], width: usize) -> Vec<f32> {
    let d_x = gpu.upload(&f32_bytes(x));
    let d_out = zeros(gpu, x.len());
    let stand_in = zeros(gpu, 1);
    launch_rotation(
        gpu,
        &Rotation {
            out: d_out.view(),
            a: d_x.view(),
            gu: stand_in.view(),
            rows: x.len() / width,
            cols_a: width,
            inner: 0,
            gu_stride: 0,
            gu_offset: 0,
        },
    );
    drain(gpu);
    gpu.read_f32(&d_out)
}

/// One per-row INT8 GEMM launch: `out = x . (w * s)^T + bias`.
#[derive(Clone, Copy)]
struct GemmLaunch<'a> {
    entry: &'static str,
    out: DeviceView<'a>,
    x: DeviceView<'a>,
    w: DeviceView<'a>,
    s: DeviceView<'a>,
    bias: Option<DeviceView<'a>>,
    stand_in: DeviceView<'a>,
    /// `(m, n, k)`.
    shape: (usize, usize, usize),
}

/// Launches one per-row INT8 GEMM with the driver's geometry.
fn launch_gemm(gpu: &Gpu, g: &GemmLaunch<'_>) {
    let GemmLaunch {
        entry,
        out,
        x,
        w,
        s,
        bias,
        stand_in,
        shape,
    } = *g;
    let (m, n, k) = shape;
    let args = GemmArgs {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        has_bias: u32::from(bias.is_some()),
    };
    let (grid, block) = if entry == "kernel_dit_gemm_i8" {
        (
            Extent3::new(
                (n as u32).div_ceil(GEMM_TILE),
                (m as u32).div_ceil(GEMM_TILE),
                1,
            ),
            Extent3::new(GEMM_TILE, GEMM_TILE, 1),
        )
    } else {
        (
            Extent3::new(
                (n as u32).div_ceil(G64_TILE),
                (m as u32).div_ceil(G64_TILE),
                1,
            ),
            Extent3::linear(G64_THREADS),
        )
    };
    launch(
        gpu,
        entry,
        grid,
        block,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out),
            KernelArg::Buffer(x),
            KernelArg::Buffer(w),
            KernelArg::Buffer(s),
            KernelArg::Buffer(bias.unwrap_or(stand_in)),
        ],
    );
}

/// Runs one GEMM entry over host fixtures and reads the product back.
fn gemm_device(
    gpu: &Gpu,
    entry: &'static str,
    x: &[f32],
    q: &[i8],
    s: &[f32],
    bias: Option<&[f32]>,
    shape: (usize, usize, usize),
) -> Vec<f32> {
    let (m, n, _) = shape;
    let d_x = gpu.upload(&f32_bytes(x));
    let d_q = gpu.upload(&i8_bytes(q));
    let d_s = gpu.upload(&f32_bytes(s));
    let d_b = bias.map(|b| gpu.upload(&f32_bytes(b)));
    let stand_in = zeros(gpu, 1);
    let d_out = zeros(gpu, m * n);
    launch_gemm(
        gpu,
        &GemmLaunch {
            entry,
            out: d_out.view(),
            x: d_x.view(),
            w: d_q.view(),
            s: d_s.view(),
            bias: d_b.as_ref().map(DeviceBuffer::view),
            stand_in: stand_in.view(),
            shape,
        },
    );
    drain(gpu);
    gpu.read_f32(&d_out)
}

// --- cases -----------------------------------------------------------------

/// The butterfly rotation equals the explicit Kronecker matrix product on
/// three 768-wide rows (three groups each), to f32 rounding.
pub fn hadamard256_matches_the_explicit_matrix(gpu: &Gpu) {
    let width = 768;
    let x = signal(3 * width, 0x4ad1);
    let got = rotate_device(gpu, &x, width);
    let want = rotate_host(&x, width);
    close(&got, &want, 1e-6, "hadamard256 vs explicit H");
    // A wrong digit order or sign pattern is a different orthogonal matrix,
    // so the input itself must not pass for its rotation.
    differs(&x, &want, "hadamard256 fixture");
}

/// `H` is symmetric and orthogonal, so rotating twice is the identity.
pub fn hadamard256_is_its_own_inverse(gpu: &Gpu) {
    let width = 768;
    let x = signal(3 * width, 0x77e1);
    let once = rotate_device(gpu, &x, width);
    let twice = rotate_device(gpu, &once, width);
    close(&twice, &x, 1e-6, "H . (H . x)");
}

/// The rotation reads the fused linear's `[a | silu(g) * u]` row, with the
/// gate/up pair starting mid-row in a wider buffer, and a group straddles
/// the two parts.
pub fn hadamard256_rotates_the_swiglu_row_it_forms(gpu: &Gpu) {
    let (rows, cols_a, inner) = (5, 128, 384);
    let (gu_stride, gu_offset) = (2 * inner + 12, 8);
    let width = cols_a + inner;
    let a = signal(rows * cols_a, 0x1001);
    let gu = signal(rows * gu_stride, 0x1002);
    let mut joined = vec![0f32; rows * width];
    for r in 0..rows {
        joined[r * width..r * width + cols_a].copy_from_slice(&a[r * cols_a..(r + 1) * cols_a]);
        for i in 0..inner {
            let g = gu[r * gu_stride + gu_offset + i];
            let u = gu[r * gu_stride + gu_offset + inner + i];
            joined[r * width + cols_a + i] = silu(g) * u;
        }
    }
    let want = rotate_host(&joined, width);
    let d_a = gpu.upload(&f32_bytes(&a));
    let d_gu = gpu.upload(&f32_bytes(&gu));
    let d_out = zeros(gpu, rows * width);
    launch_rotation(
        gpu,
        &Rotation {
            out: d_out.view(),
            a: d_a.view(),
            gu: d_gu.view(),
            rows,
            cols_a,
            inner,
            gu_stride,
            gu_offset,
        },
    );
    drain(gpu);
    close(
        &gpu.read_f32(&d_out),
        &want,
        1e-5,
        "hadamard256 over [a | swiglu]",
    );
}

/// Both INT8 GEMM entries equal the dequantized `f64` product on shapes that
/// leave ragged 64x64 tiles and run 16 to 256 k slices, with and without a
/// bias; the scalar entry also takes a k that is not a multiple of 16.
pub fn gemm_i8_matches_the_dequantized_product(gpu: &Gpu) {
    let shapes: [(usize, usize, usize); 3] = [(5, 70, 256), (64, 64, 512), (130, 200, 4096)];
    for (index, &(m, n, k)) in shapes.iter().enumerate() {
        let seed = u32::try_from(index).unwrap_or(0);
        let x = signal(m * k, 0x2000 + seed);
        let q = codes(n * k, 0x3000 + seed);
        let s = row_scales(n);
        let bias = signal(n, 0x4000 + seed);
        for with_bias in [false, true] {
            let b = with_bias.then_some(bias.as_slice());
            let want = product_host(&x, &q, &s, b, (m, n, k));
            for entry in ["kernel_dit_gemm64_i8", "kernel_dit_gemm_i8"] {
                let got = gemm_device(gpu, entry, &x, &q, &s, b, (m, n, k));
                close(
                    &got,
                    &want,
                    1e-5,
                    &format!("{entry} ({m},{n},{k}) bias={with_bias}"),
                );
            }
        }
    }
    let (m, n, k) = (7, 9, 40);
    let x = signal(m * k, 0x2100);
    let q = codes(n * k, 0x3100);
    let s = row_scales(n);
    let want = product_host(&x, &q, &s, None, (m, n, k));
    let got = gemm_device(gpu, "kernel_dit_gemm_i8", &x, &q, &s, None, (m, n, k));
    close(&got, &want, 1e-5, "kernel_dit_gemm_i8 with k % 16 != 0");
}

/// The scale belongs to the weight row, i.e. the output column. Reading it by
/// input column instead (a transposed `[1, in]` scale) is a different
/// answer, and the case would prove nothing if the fixture could not tell
/// the two apart.
pub fn gemm_i8_scales_each_output_row_not_each_input_column(gpu: &Gpu) {
    let (m, n, k) = (9, 256, 256);
    let x = signal(m * k, 0x5001);
    let q = codes(n * k, 0x5002);
    let s = row_scales(n);
    let right = product_host(&x, &q, &s, None, (m, n, k));
    let mut wrong = vec![0f32; m * n];
    for r in 0..m {
        for o in 0..n {
            let mut acc = 0f64;
            for i in 0..k {
                acc += f64::from(x[r * k + i]) * f64::from(q[o * k + i]) * f64::from(s[i]);
            }
            wrong[r * n + o] = acc as f32;
        }
    }
    differs(&wrong, &right, "row-scale vs column-scale readings");
    for entry in ["kernel_dit_gemm64_i8", "kernel_dit_gemm_i8"] {
        let got = gemm_device(gpu, entry, &x, &q, &s, None, (m, n, k));
        close(&got, &right, 1e-5, entry);
        differs(
            &got,
            &wrong,
            &format!("{entry} against the column-scale reading"),
        );
    }
}

/// One fused-linear fixture: `residual += gate * linear([a | swiglu(gu)])`.
struct FusedCase {
    m: usize,
    n: usize,
    cols_a: usize,
    inner: usize,
    gu_stride: usize,
    gu_offset: usize,
    with_bias: bool,
}

/// The fused INT8 linear equals the separate swiglu, concat, GEMM and gated
/// add it replaces, run on the same device, and the `f64` oracle of the
/// whole chain - for a dense-only, a swiglu-only and a joined input.
pub fn fused_i8_is_the_unfused_composition(gpu: &Gpu) {
    let cases = [
        FusedCase {
            m: 70,
            n: 67,
            cols_a: 256,
            inner: 0,
            gu_stride: 0,
            gu_offset: 0,
            with_bias: true,
        },
        FusedCase {
            m: 33,
            n: 130,
            cols_a: 0,
            inner: 512,
            gu_stride: 1040,
            gu_offset: 12,
            with_bias: false,
        },
        FusedCase {
            m: 65,
            n: 64,
            cols_a: 128,
            inner: 384,
            gu_stride: 768,
            gu_offset: 0,
            with_bias: true,
        },
    ];
    for (index, c) in cases.iter().enumerate() {
        fused_case(gpu, c, u32::try_from(index).unwrap_or(0));
    }
}

fn fused_case(gpu: &Gpu, c: &FusedCase, seed: u32) {
    let k = c.cols_a + c.inner;
    let (m, n) = (c.m, c.n);
    let a = signal(m * c.cols_a, 0x6000 + seed);
    let gu = signal(m * c.gu_stride, 0x6100 + seed);
    let q = codes(n * k, 0x6200 + seed);
    let s = row_scales(n);
    let bias = signal(n, 0x6300 + seed);
    let gate = signal(n, 0x6400 + seed);
    let residual = signal(m * n, 0x6500 + seed);

    let mut joined = vec![0f32; m * k];
    for r in 0..m {
        joined[r * k..r * k + c.cols_a].copy_from_slice(&a[r * c.cols_a..(r + 1) * c.cols_a]);
        for i in 0..c.inner {
            let g = gu[r * c.gu_stride + c.gu_offset + i];
            let u = gu[r * c.gu_stride + c.gu_offset + c.inner + i];
            joined[r * k + c.cols_a + i] = silu(g) * u;
        }
    }
    let b = c.with_bias.then_some(bias.as_slice());
    let linear = product_host(&joined, &q, &s, b, (m, n, k));
    let want: Vec<f32> = residual
        .iter()
        .enumerate()
        .map(|(i, &x)| x + gate[i % n] * linear[i])
        .collect();

    let d_a = gpu.upload(&f32_bytes(&if a.is_empty() {
        vec![0.0]
    } else {
        a.clone()
    }));
    let d_gu = gpu.upload(&f32_bytes(&if gu.is_empty() {
        vec![0.0]
    } else {
        gu.clone()
    }));
    let d_q = gpu.upload(&i8_bytes(&q));
    let d_s = gpu.upload(&f32_bytes(&s));
    let d_bias = gpu.upload(&f32_bytes(&bias));
    let d_gate = gpu.upload(&f32_bytes(&gate));
    let stand_in = zeros(gpu, 1);
    let bias_view = if c.with_bias {
        d_bias.view()
    } else {
        stand_in.view()
    };

    // Fused.
    let d_fused = gpu.upload(&f32_bytes(&residual));
    let args = GemmFusedArgs {
        m: m as u32,
        n: n as u32,
        cols_a: c.cols_a as u32,
        inner: c.inner as u32,
        gu_stride: c.gu_stride as u32,
        gu_offset: c.gu_offset as u32,
        has_bias: u32::from(c.with_bias),
    };
    launch(
        gpu,
        "kernel_dit_gemm64_fused_i8",
        Extent3::new(
            (n as u32).div_ceil(G64_TILE),
            (m as u32).div_ceil(G64_TILE),
            1,
        ),
        Extent3::linear(G64_THREADS),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_fused.view()),
            KernelArg::Buffer(d_a.view()),
            KernelArg::Buffer(d_gu.view()),
            KernelArg::Buffer(d_q.view()),
            KernelArg::Buffer(d_s.view()),
            KernelArg::Buffer(bias_view),
            KernelArg::Buffer(d_gate.view()),
        ],
    );

    let unfused = unfused_device(
        gpu,
        c,
        &Operands {
            a: d_a.view(),
            gu: d_gu.view(),
            q: d_q.view(),
            s: d_s.view(),
            bias: c.with_bias.then(|| d_bias.view()),
            gate: d_gate.view(),
            stand_in: stand_in.view(),
        },
        &residual,
    );
    drain(gpu);
    let fused = gpu.read_f32(&d_fused);
    let what = format!("fused_i8 m={m} n={n} cols_a={} inner={}", c.cols_a, c.inner);
    close(
        &fused,
        &unfused,
        1e-6,
        &format!("{what} vs the separate passes"),
    );
    close(&fused, &want, 1e-5, &format!("{what} vs the f64 chain"));
}

/// The fused linear's device operands, shared by both runs of a case.
struct Operands<'a> {
    a: DeviceView<'a>,
    gu: DeviceView<'a>,
    q: DeviceView<'a>,
    s: DeviceView<'a>,
    bias: Option<DeviceView<'a>>,
    gate: DeviceView<'a>,
    stand_in: DeviceView<'a>,
}

/// The chain the fused linear replaces, as separate launches: swiglu into a
/// packed buffer, concat, the INT8 GEMM, the gated add.
fn unfused_device(gpu: &Gpu, c: &FusedCase, o: &Operands<'_>, residual: &[f32]) -> Vec<f32> {
    let k = c.cols_a + c.inner;
    let (m, n) = (c.m, c.n);
    // Unfused: swiglu into a packed buffer, concat, GEMM, gated add.
    let d_act = zeros(gpu, m * c.inner);
    if c.inner > 0 {
        let sw = SwigluArgs {
            rows: m as u32,
            inner: c.inner as u32,
            stride: c.gu_stride as u32,
        };
        let start = (c.gu_offset * 4) as u64;
        let gu_view =
            o.gu.slice(start, o.gu.len() - start)
                .expect("the gate/up view is in range");
        launch(
            gpu,
            "kernel_dit_swiglu_f32",
            Extent3::new((c.inner as u32).div_ceil(BLOCK), m as u32, 1),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&sw)),
                KernelArg::Buffer(d_act.view()),
                KernelArg::Buffer(gu_view),
            ],
        );
    }
    let d_joined = zeros(gpu, m * k);
    let input = match (c.cols_a > 0, c.inner > 0) {
        (true, true) => {
            let ca = ConcatArgs {
                rows: m as u32,
                cols_a: c.cols_a as u32,
                cols_b: c.inner as u32,
            };
            launch(
                gpu,
                "kernel_dit_concat_rows_f32",
                Extent3::linear(m as u32),
                Extent3::linear(BLOCK),
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&ca)),
                    KernelArg::Buffer(d_joined.view()),
                    KernelArg::Buffer(o.a),
                    KernelArg::Buffer(d_act.view()),
                ],
            );
            d_joined.view()
        }
        (true, false) => o.a,
        _ => d_act.view(),
    };
    let d_tmp = zeros(gpu, m * n);
    launch_gemm(
        gpu,
        &GemmLaunch {
            entry: "kernel_dit_gemm64_i8",
            out: d_tmp.view(),
            x: input,
            w: o.q,
            s: o.s,
            bias: o.bias,
            stand_in: o.stand_in,
            shape: (m, n, k),
        },
    );
    let d_unfused = gpu.upload(&f32_bytes(residual));
    let ag = AddGatedArgs {
        count: (m * n) as u32,
        period: n as u32,
    };
    launch(
        gpu,
        "kernel_dit_add_gated_f32",
        Extent3::linear(((m * n) as u32).div_ceil(BLOCK)),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&ag)),
            KernelArg::Buffer(d_unfused.view()),
            KernelArg::Buffer(d_unfused.view()),
            KernelArg::Buffer(o.gate),
            KernelArg::Buffer(d_tmp.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&d_unfused)
}
