//! Device cases for the vision-tower kernel set.
//!
//! There is no upstream shading-language source for this tower, and no upstream
//! CUDA source either that the shared encoder is sequenced against — the
//! `qwen3vl_*` set in `kernels/cuda/vision` is a different decomposition, and
//! `kernels/cuda/vision/KERNEL_NAMES.md` records entry by entry why. The usual
//! defence, diff the port against what it was ported from, is therefore not
//! available. What stands in for it is the oracle below: every case compares the
//! device against the simplest possible reading of the formula, written as a
//! plain Rust `f32` loop, so a disagreement points at the kernel or at its
//! launch geometry rather than at a second, equally subtle implementation.
//!
//! An oracle written from the same misreading as the kernel proves nothing, so
//! the cases that can be pinned by a structural property are pinned by one as
//! well as by a number: a rotation at the grid origin must be the identity, a
//! query must not see a patch in another frame, the two GELUs must not agree.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "kernel argument structs take u32 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]
#![expect(
    clippy::excessive_precision,
    reason = "the activation constants are written with the digits the tower's own description gives them, which round to the f32 the kernels hold"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "the oracles use the tower's own names - q, k, v for the attention planes - and renaming them to satisfy a lint would make them worse transcriptions"
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::oracle;
use crate::{Gpu, drain, f32_bytes, run};

/// Relative tolerance for a path whose only difference from the oracle is the
/// order a reduction associates in.
///
/// Calibrated rather than chosen: the widest disagreement any case below
/// actually shows is about 1e-6, from the one kernel whose staging changes where
/// a partial sum rounds. The bound is an order of magnitude above that and no
/// further, because headroom in a tolerance is headroom for a mistake.
const TOL: f32 = 2e-5;

/// Threads the elementwise and gather kernels are launched with, per
/// `KERNELS.md`. The elementwise kernels hard-code this stride, so it is the one
/// block size they may be launched with.
const LINEAR_BLOCK: u32 = 256;
/// Threads the normalization is launched with.
const NORM_BLOCK: u32 = 256;
/// Threadgroup bytes the normalization's block reduction needs:
/// `VIT_BLOCK_SUM_SLOTS` floats.
const NORM_SMEM: u32 = 132;
/// Threads the attention and the rotation are launched with.
const ATTN_BLOCK: u32 = 128;
/// Square tile the blocked GEMM walks, in both output axes.
const GEMM_TILE: u32 = 16;

// -- argument blocks, mirroring `vit_common.h` and `vit_common.cuh` ----------

/// `vit_args_gemm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GemmArgs {
    m: u32,
    n: u32,
    k: u32,
}

/// `vit_args_layer_norm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LayerNormArgs {
    rows: u32,
    dim: u32,
    eps: f32,
}

/// `vit_args_unary`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct UnaryArgs {
    count: u32,
}

/// `vit_args_add`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AddArgs {
    count: u32,
    period: u32,
}

/// `vit_args_qkv_rope`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RopeArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    grid_w: u32,
    plane: u32,
    rope_base: f32,
}

/// `vit_args_attention`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AttentionArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    plane: u32,
    scale: f32,
}

/// `vit_args_merge`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MergeArgs {
    merged_rows: u32,
    hidden: u32,
    grid_h: u32,
    grid_w: u32,
    merge: u32,
}

/// `vit_args_splice`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SpliceArgs {
    rows: u32,
    hidden: u32,
    src_rows: u32,
}

// -- fixture helpers --------------------------------------------------------

/// The 16 bits a brain-float keeps of an `f32`, by truncation.
///
/// The fixtures upload these bits and the oracle reads the same bits back
/// through [`widen`], so the pair never depends on which rounding the host
/// chose: kernel and oracle see one value, not two roundings of one value.
fn narrow(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// One brain-float widened, exactly as `vit_bf16` widens it.
fn widen(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// A bf16 fixture: the little-endian bytes to upload, and the values the oracle
/// must use.
fn bf16_fixture(len: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let raw = oracle::signal(len, seed);
    let bits: Vec<u16> = raw.into_iter().map(narrow).collect();
    let bytes = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let values = bits.into_iter().map(widen).collect();
    (bytes, values)
}

/// Launches one vision kernel with the geometry `KERNELS.md` records for it.
fn launch(
    gpu: &Gpu,
    entry: &'static str,
    grid: Extent3,
    block: Extent3,
    smem_bytes: u32,
    args: &[KernelArg<'_>],
) {
    run(
        gpu,
        KernelRef::new("vision", entry),
        &LaunchArgs {
            grid,
            block,
            smem_bytes,
            args,
        },
    );
}

/// `ceil(a / b)` over the launch dimensions.
fn blocks(count: usize, per_block: u32) -> u32 {
    count.div_ceil(per_block as usize) as u32
}

/// An f32 device buffer of `len` zeros.
fn zeros(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![0.0f32; len]))
}

/// Asserts every value agrees within [`TOL`], scaled by the expected magnitude.
fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = TOL * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= bound,
            "{what}: element {index} is {g}, expected {w} (tolerance {bound})",
        );
    }
}

// -- the entry points exist -------------------------------------------------

/// Every entry point the encoder dispatches, in the order the tower runs them,
/// plus the splice that hands its output to the language model.
///
/// A backend driver walks this list and asks its loader to resolve each one.
/// That check is not device-gated on either backend: a kernel file that no
/// longer compiles breaks every consumer of the module, not just the vision
/// path, and a typo in an entry-point name is a runtime failure with no
/// compile-time warning anywhere.
pub const ENTRIES: [&str; 9] = [
    "kernel_vit_gemm_bias",
    "kernel_vit_layer_norm",
    "kernel_vit_gelu_tanh",
    "kernel_vit_gelu_erf",
    "kernel_vit_add_inplace",
    "kernel_vit_qkv_rope",
    "kernel_vit_attention",
    "kernel_vit_spatial_merge",
    "kernel_vit_splice_rows",
];

// -- the matrix multiply ----------------------------------------------------

/// Rows, output width and reduction width of the GEMM fixture.
///
/// None of the three is a multiple of the tile, which is the point: the tail
/// guards are the part of a blocked GEMM that a square fixture cannot reach, and
/// a tower whose widths happen to be multiples would hide a broken one until a
/// checkpoint with an odd feed-forward width arrived.
const GEMM_M: usize = 37;
const GEMM_N: usize = 53;
const GEMM_K: usize = 71;

/// The naive reading of the GEMM: one row times one column, summed front to
/// back, plus the bias.
fn gemm_oracle(input: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; GEMM_M * GEMM_N];
    for m in 0..GEMM_M {
        for n in 0..GEMM_N {
            let mut acc = 0.0f32;
            for k in 0..GEMM_K {
                acc += input[m * GEMM_K + k] * weight[n * GEMM_K + k];
            }
            out[m * GEMM_N + n] = acc + bias[n];
        }
    }
    out
}

/// The blocked GEMM agrees with a row-times-column loop.
pub fn the_blocked_gemm_agrees_with_a_row_times_column_loop(gpu: &Gpu) {
    let input = oracle::signal(GEMM_M * GEMM_K, 11);
    let (weight_bytes, weight) = bf16_fixture(GEMM_N * GEMM_K, 12);
    let (bias_bytes, bias) = bf16_fixture(GEMM_N, 13);

    let out = zeros(gpu, GEMM_M * GEMM_N);
    let d_in = gpu.upload(&f32_bytes(&input));
    let d_w = gpu.upload(&weight_bytes);
    let d_b = gpu.upload(&bias_bytes);
    let args = GemmArgs {
        m: GEMM_M as u32,
        n: GEMM_N as u32,
        k: GEMM_K as u32,
    };
    launch(
        gpu,
        "kernel_vit_gemm_bias",
        Extent3::new(blocks(GEMM_N, GEMM_TILE), blocks(GEMM_M, GEMM_TILE), 1),
        Extent3::new(GEMM_TILE, GEMM_TILE, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_in.view()),
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(d_b.view()),
        ],
    );
    drain(gpu);
    close(
        &gpu.read_f32(&out),
        &gemm_oracle(&input, &weight, &bias),
        "gemm",
    );
}

// -- the normalization ------------------------------------------------------

/// The naive reading of the normalization: the mean, then the variance about
/// it, then the affine map.
fn layer_norm_oracle(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &input[r * dim..][..dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        for i in 0..dim {
            out[r * dim + i] = (row[i] - mean) * rstd * weight[i] + bias[i];
        }
    }
    out
}

/// Runs the normalization at one shape, optionally in place.
fn run_layer_norm(gpu: &Gpu, rows: usize, dim: usize, in_place: bool) {
    const EPS: f32 = 1e-6;
    let input = oracle::signal(rows * dim, 21);
    let (weight_bytes, weight) = bf16_fixture(dim, 22);
    let (bias_bytes, bias) = bf16_fixture(dim, 23);

    let d_in = gpu.upload(&f32_bytes(&input));
    let out = if in_place {
        None
    } else {
        Some(zeros(gpu, rows * dim))
    };
    let d_w = gpu.upload(&weight_bytes);
    let d_b = gpu.upload(&bias_bytes);
    let args = LayerNormArgs {
        rows: rows as u32,
        dim: dim as u32,
        eps: EPS,
    };
    let dst = out.as_ref().unwrap_or(&d_in);
    launch(
        gpu,
        "kernel_vit_layer_norm",
        Extent3::linear(rows as u32),
        Extent3::linear(NORM_BLOCK),
        NORM_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(dst.view()),
            KernelArg::Buffer(d_in.view()),
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(d_b.view()),
        ],
    );
    drain(gpu);
    close(
        &gpu.read_f32(dst),
        &layer_norm_oracle(&input, &weight, &bias, rows, dim, EPS),
        if in_place {
            "layer_norm in place"
        } else {
            "layer_norm"
        },
    );
}

/// The normalization subtracts the mean as well as scaling by the deviation.
pub fn the_normalization_subtracts_the_mean_as_well_as_scaling_by_the_deviation(gpu: &Gpu) {
    // A width the block divides, and one it does not: the second leaves whole
    // warps with nothing to add, which is the case a block reduction gets wrong
    // by contributing stale scratch.
    run_layer_norm(gpu, 6, 1024, false);
    run_layer_norm(gpu, 5, 100, false);
}

/// The normalization is safe to run over its own input.
pub fn the_normalization_is_safe_to_run_over_its_own_input(gpu: &Gpu) {
    run_layer_norm(gpu, 6, 1024, true);
}

// -- the activations --------------------------------------------------------

/// The tanh approximation, as the reference writes it.
fn gelu_tanh(v: f32) -> f32 {
    let c = 0.797_884_560_802_865_4_f32;
    0.5 * v * (1.0 + (c * (v + 0.044_715 * v * v * v)).tanh())
}

/// The exact form, through a series for the error function.
///
/// The shading language has no error function, so the Metal kernel carries this
/// same series; the CUDA twin carries it too rather than calling `erff`, so that
/// one case grades one function on both. This case cannot therefore be a check
/// on the series itself — only on the plumbing around it. What bounds the series
/// is its own published error, 1.5e-7 (Abramowitz and Stegun 7.1.26): about one
/// unit in the last place of an f32 near 1, and three orders of magnitude
/// tighter than the 1e-4 that separates the two GELUs, which is the difference
/// this case exists to catch.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// The exact GELU.
fn gelu_erf(v: f32) -> f32 {
    0.5 * v * (1.0 + erf(v * std::f32::consts::FRAC_1_SQRT_2))
}

/// Applies one activation on the device and returns what it produced.
fn run_gelu(gpu: &Gpu, entry: &'static str, input: &[f32]) -> Vec<f32> {
    let x = gpu.upload(&f32_bytes(input));
    let args = UnaryArgs {
        count: input.len() as u32,
    };
    launch(
        gpu,
        entry,
        Extent3::linear(blocks(input.len(), LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(x.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&x)
}

/// Each activation matches its own formula and not the other one.
pub fn each_activation_matches_its_own_formula_and_not_the_other_one(gpu: &Gpu) {
    // Scaled past the unit interval so the two forms are far enough apart to
    // separate: near zero they agree to more digits than either is measured to.
    let input: Vec<f32> = oracle::signal(777, 31).iter().map(|v| v * 3.0).collect();

    let tanh_got = run_gelu(gpu, "kernel_vit_gelu_tanh", &input);
    let erf_got = run_gelu(gpu, "kernel_vit_gelu_erf", &input);
    let tanh_want: Vec<f32> = input.iter().copied().map(gelu_tanh).collect();
    let erf_want: Vec<f32> = input.iter().copied().map(gelu_erf).collect();
    close(&tanh_got, &tanh_want, "gelu tanh");
    close(&erf_got, &erf_want, "gelu erf");

    // The two are not the same function. A kernel set that used one where the
    // other belongs would pass both cases above if this assertion were absent,
    // because each would still match the formula it was written from.
    let spread = tanh_got
        .iter()
        .zip(&erf_got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        spread > 1e-4,
        "the two activations agreed to {spread}, so this fixture cannot tell them apart",
    );
}

/// The activation over the range a real feed-forward row actually spans.
///
/// A measured range, not a guessed one: the shipped tower's first block reaches
/// about +/-12 on a 224-pixel image, and the tanh form is evaluated as
/// `(e^2x - 1) / (e^2x + 1)`, whose numerator and denominator both overflow
/// single precision once the cubic inside it passes about 44 - which happens at
/// an input of about 10.3. Before the argument was guarded this produced a few
/// dozen NaNs per image in block zero; the layer normalization that follows
/// spread each one over its whole row, and by the last block every row was NaN.
/// Nothing in the kernel set caught it, because every case here ran on values
/// inside the unit interval scaled by three.
///
/// So the assertion is the one that was missing: finite in, finite out, across
/// the whole range the tower produces and well past it.
pub fn the_activations_stay_finite_over_the_range_a_feed_forward_row_reaches(gpu: &Gpu) {
    // A sweep rather than a sample: the failure had a threshold, and a random
    // fixture either straddles it or does not.
    let input: Vec<f32> = (-400..=400).map(|step| step as f32 * 0.1).collect();

    for entry in ["kernel_vit_gelu_tanh", "kernel_vit_gelu_erf"] {
        let got = run_gelu(gpu, entry, &input);
        for (value, out) in input.iter().zip(&got) {
            assert!(
                out.is_finite(),
                "{entry} produced {out} for an input of {value}",
            );
        }
        // The limits, which is what a guard could get wrong in the other
        // direction: this activation is the identity for large positive inputs
        // and zero for large negative ones.
        let last = got.last().copied().unwrap_or_default();
        let first = got.first().copied().unwrap_or_default();
        assert!(
            (last - 40.0).abs() < 1e-3,
            "{entry} is not the identity at +40: {last}",
        );
        assert!(first.abs() < 1e-3, "{entry} is not zero at -40: {first}");
    }
}

// -- the residual and position adds -----------------------------------------

/// The add repeats a shorter period and leaves an equal one elementwise.
pub fn the_add_repeats_a_shorter_period_and_leaves_an_equal_one_elementwise(gpu: &Gpu) {
    let base = oracle::signal(600, 41);
    let short = oracle::signal(200, 42);
    let full = oracle::signal(600, 43);

    // The position embedding's case, where one grid's worth of vectors is added
    // to each temporal group, and the residual's case, where the period is the
    // whole buffer and the add is plain and elementwise.
    for source in [&short, &full] {
        let period = source.len();
        let x = gpu.upload(&f32_bytes(&base));
        let a = gpu.upload(&f32_bytes(source));
        let args = AddArgs {
            count: base.len() as u32,
            period: period as u32,
        };
        launch(
            gpu,
            "kernel_vit_add_inplace",
            Extent3::linear(blocks(base.len(), LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(x.view()),
                KernelArg::Buffer(a.view()),
            ],
        );
        drain(gpu);
        let want: Vec<f32> = base
            .iter()
            .enumerate()
            .map(|(i, v)| v + source[i % period])
            .collect();
        close(&gpu.read_f32(&x), &want, "add");
    }
}

// -- the rotation -----------------------------------------------------------

/// Heads, head width and grid of the rotation and attention fixtures.
const HEADS: usize = 3;
const HEAD_DIM: usize = 64;
const GRID_H: usize = 3;
const GRID_W: usize = 4;
const PLANE: usize = GRID_H * GRID_W;
const ROPE_BASE: f32 = 10_000.0;

/// The naive reading of the split and the rotation: three planes out of one
/// fused row, with the query and the key rotated in two spatial axes.
///
/// Returns `(q, k, v)`.
fn rope_oracle(
    qkv: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    grid_w: usize,
    plane: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = heads * head_dim;
    let half = head_dim / 2;
    let per_axis = half / 2;
    let mut q = vec![0.0f32; rows * dim];
    let mut k = vec![0.0f32; rows * dim];
    let mut v = vec![0.0f32; rows * dim];
    for n in 0..rows {
        let local = n % plane;
        let row = (local / grid_w) as f32;
        let col = (local % grid_w) as f32;
        let src = &qkv[n * 3 * dim..][..3 * dim];
        for h in 0..heads {
            let off = h * head_dim;
            for i in 0..head_dim {
                v[n * dim + off + i] = src[2 * dim + off + i];
            }
            for i in 0..half {
                let (freq, pos) = if i < per_axis {
                    (i, row)
                } else {
                    (i - per_axis, col)
                };
                let inv = ROPE_BASE.powf(-2.0 * freq as f32 / half as f32);
                let theta = pos * inv;
                let (s, c) = (theta.sin(), theta.cos());
                let q0 = src[off + i];
                let q1 = src[off + i + half];
                q[n * dim + off + i] = q0 * c - q1 * s;
                q[n * dim + off + i + half] = q1 * c + q0 * s;
                let k0 = src[dim + off + i];
                let k1 = src[dim + off + i + half];
                k[n * dim + off + i] = k0 * c - k1 * s;
                k[n * dim + off + i + half] = k1 * c + k0 * s;
            }
        }
    }
    (q, k, v)
}

/// Runs the split-and-rotate on the device and reads the three planes back.
fn run_rope(
    gpu: &Gpu,
    qkv: &[f32],
    rows: usize,
    grid_w: usize,
    plane: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = HEADS * HEAD_DIM;
    let q = zeros(gpu, rows * dim);
    let k = zeros(gpu, rows * dim);
    let v = zeros(gpu, rows * dim);
    let d_qkv = gpu.upload(&f32_bytes(qkv));
    let args = RopeArgs {
        rows: rows as u32,
        heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        grid_w: grid_w as u32,
        plane: plane as u32,
        rope_base: ROPE_BASE,
    };
    launch(
        gpu,
        "kernel_vit_qkv_rope",
        Extent3::new(HEADS as u32, rows as u32, 1),
        Extent3::linear(ATTN_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(q.view()),
            KernelArg::Buffer(k.view()),
            KernelArg::Buffer(v.view()),
            KernelArg::Buffer(d_qkv.view()),
        ],
    );
    drain(gpu);
    (gpu.read_f32(&q), gpu.read_f32(&k), gpu.read_f32(&v))
}

/// The rotation splits the head between the row axis and the column axis.
pub fn the_rotation_splits_the_head_between_the_row_axis_and_the_column_axis(gpu: &Gpu) {
    // Two temporal groups, so the case also covers a row whose grid position is
    // its offset inside its own group rather than its index in the sequence.
    let rows = 2 * PLANE;
    let dim = HEADS * HEAD_DIM;
    let qkv = oracle::signal(rows * 3 * dim, 51);

    let (q, k, v) = run_rope(gpu, &qkv, rows, GRID_W, PLANE);
    let (wq, wk, wv) = rope_oracle(&qkv, rows, HEADS, HEAD_DIM, GRID_W, PLANE);
    close(&q, &wq, "rope q");
    close(&k, &wk, "rope k");
    close(&v, &wv, "rope v");
}

/// The rotation is the identity at the grid origin and leaves the values alone.
pub fn the_rotation_is_the_identity_at_the_grid_origin_and_leaves_the_values_alone(gpu: &Gpu) {
    // A structural pin the oracle cannot supply: at row 0, column 0 every angle
    // is zero, so the query and key planes must come out of the fused row
    // untouched. A rotation that read the wrong position, paired the wrong
    // channels or swapped the two axes would still match an oracle written from
    // the same mistake -- it cannot survive this.
    let dim = HEADS * HEAD_DIM;
    let qkv = oracle::signal(3 * dim, 52);
    let (q, k, v) = run_rope(gpu, &qkv, 1, GRID_W, PLANE);
    close(&q, &qkv[..dim], "rope identity q");
    close(&k, &qkv[dim..2 * dim], "rope identity k");
    close(&v, &qkv[2 * dim..], "rope identity v");
}

// -- the attention ----------------------------------------------------------

/// The naive reading of the attention: every score, one maximum subtraction, one
/// division. Bidirectional, and bounded by the temporal group the query is in.
fn attention_oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    plane: usize,
) -> Vec<f32> {
    let dim = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; rows * dim];
    for n in 0..rows {
        let base = (n / plane) * plane;
        for h in 0..heads {
            let off = h * head_dim;
            let scores: Vec<f32> = (base..base + plane)
                .map(|p| {
                    let dot: f32 = (0..head_dim)
                        .map(|d| q[n * dim + off + d] * k[p * dim + off + d])
                        .sum();
                    dot * scale
                })
                .collect();
            let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let den: f32 = weights.iter().sum();
            for d in 0..head_dim {
                let num: f32 = weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| w * v[(base + i) * dim + off + d])
                    .sum();
                out[n * dim + off + d] = num / den;
            }
        }
    }
    out
}

/// Runs the attention on the device over `rows` patch rows in groups of `plane`.
fn run_attention(
    gpu: &Gpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    plane: usize,
) -> Vec<f32> {
    let dim = HEADS * HEAD_DIM;
    let out = zeros(gpu, rows * dim);
    let d_q = gpu.upload(&f32_bytes(q));
    let d_k = gpu.upload(&f32_bytes(k));
    let d_v = gpu.upload(&f32_bytes(v));
    let warps = ATTN_BLOCK / 32;
    let smem = 4 * (HEAD_DIM as u32 * (warps + 1) + 2 * warps);
    let args = AttentionArgs {
        rows: rows as u32,
        heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        plane: plane as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };
    launch(
        gpu,
        "kernel_vit_attention",
        Extent3::new(HEADS as u32, rows as u32, 1),
        Extent3::linear(ATTN_BLOCK),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_q.view()),
            KernelArg::Buffer(d_k.view()),
            KernelArg::Buffer(d_v.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// The attention agrees with a plain softmax over every patch.
pub fn the_attention_agrees_with_a_plain_softmax_over_every_patch(gpu: &Gpu) {
    let rows = 2 * PLANE;
    let dim = HEADS * HEAD_DIM;
    let q = oracle::signal(rows * dim, 61);
    let k = oracle::signal(rows * dim, 62);
    let v = oracle::signal(rows * dim, 63);
    let got = run_attention(gpu, &q, &k, &v, rows, PLANE);
    let want = attention_oracle(&q, &k, &v, rows, HEADS, HEAD_DIM, PLANE);
    close(&got, &want, "attention");
}

/// A query never attends across a temporal group boundary.
pub fn a_query_never_attends_across_a_temporal_group_boundary(gpu: &Gpu) {
    // The first group's rows must produce the same vectors whether or not a
    // second group is present. An attention that took its key range from the
    // row count instead of from the group would pass the oracle case above only
    // if the oracle made the same mistake; here the device is compared against
    // itself, so it cannot.
    let dim = HEADS * HEAD_DIM;
    let rows = 2 * PLANE;
    let q = oracle::signal(rows * dim, 71);
    let k = oracle::signal(rows * dim, 72);
    let v = oracle::signal(rows * dim, 73);

    let both = run_attention(gpu, &q, &k, &v, rows, PLANE);
    let alone = run_attention(
        gpu,
        &q[..PLANE * dim],
        &k[..PLANE * dim],
        &v[..PLANE * dim],
        PLANE,
        PLANE,
    );
    close(&both[..PLANE * dim], &alone, "attention first group");
}

// -- the spatial merge ------------------------------------------------------

/// Grid height for the merge fixture, which needs an even one.
const GRID_H_EVEN: usize = 4;
/// Width of the merge fixture's rows; small, because the merge moves values
/// rather than combining them and a wide row would only make the case slower.
const HIDDEN: usize = 5;
/// Side of the block the merge folds.
const MERGE: usize = 2;
/// Temporal groups in the merge fixture: a second group is what shows the output
/// rows of one frame do not run into the next frame's.
const GROUPS: usize = 2;

/// The spatial merge gathers the block the grid names.
pub fn the_spatial_merge_gathers_the_block_the_grid_names(gpu: &Gpu) {
    let plane = GRID_H_EVEN * GRID_W;
    let rows = GROUPS * plane;
    let merged_rows = GROUPS * (GRID_H_EVEN / MERGE) * (GRID_W / MERGE);
    let width = MERGE * MERGE * HIDDEN;

    let input = oracle::signal(rows * HIDDEN, 81);
    let out = zeros(gpu, merged_rows * width);
    let d_in = gpu.upload(&f32_bytes(&input));
    let args = MergeArgs {
        merged_rows: merged_rows as u32,
        hidden: HIDDEN as u32,
        grid_h: GRID_H_EVEN as u32,
        grid_w: GRID_W as u32,
        merge: MERGE as u32,
    };
    launch(
        gpu,
        "kernel_vit_spatial_merge",
        Extent3::linear(merged_rows as u32),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_in.view()),
        ],
    );
    drain(gpu);

    let out_gw = GRID_W / MERGE;
    let per_group = (GRID_H_EVEN / MERGE) * out_gw;
    let mut want = vec![0.0f32; merged_rows * width];
    for o in 0..merged_rows {
        let group = o / per_group;
        let local = o % per_group;
        let (oh, ow) = (local / out_gw, local % out_gw);
        for slot in 0..MERGE * MERGE {
            let ph = oh * MERGE + slot / MERGE;
            let pw = ow * MERGE + slot % MERGE;
            let src = (group * plane + ph * GRID_W + pw) * HIDDEN;
            for d in 0..HIDDEN {
                want[o * width + slot * HIDDEN + d] = input[src + d];
            }
        }
    }
    close(&gpu.read_f32(&out), &want, "spatial merge");
}

// -- the splice into the residual stream ------------------------------------
//
// The last kernel of the set is the only one that does not belong to the tower:
// it is where the tower's rows displace the placeholder tokens' embeddings. Two
// things have to be true of it and neither is checkable downstream. Text rows
// must survive exactly - not approximately, exactly, because the value there is
// an embedding the model was going to use - and a row index the block does not
// contain must be refused rather than read.

/// Width of the residual stream in the splice fixture. Not the tower's hidden
/// width: the rows the merger emits are already at the language model's.
const SPLICE_HIDDEN: usize = 24;

/// Runs one splice and returns the destination rows.
fn run_splice(
    gpu: &Gpu,
    embedded: &[f32],
    rows_in: &[f32],
    map: &[i32],
    src_rows: u32,
) -> Vec<f32> {
    let n_tok = embedded.len() / SPLICE_HIDDEN;
    let hidden = gpu.upload(&f32_bytes(embedded));
    let src = gpu.upload(&f32_bytes(rows_in));
    let map_bytes: Vec<u8> = map.iter().copied().flat_map(i32::to_le_bytes).collect();
    let map = gpu.upload(&map_bytes);
    let args = SpliceArgs {
        rows: n_tok as u32,
        hidden: SPLICE_HIDDEN as u32,
        src_rows,
    };
    launch(
        gpu,
        "kernel_vit_splice_rows",
        Extent3::linear(n_tok as u32),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(hidden.view()),
            KernelArg::Buffer(src.view()),
            KernelArg::Buffer(map.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&hidden)
}

/// The rows the map names are replaced bit-for-bit and the rest are untouched
/// bit-for-bit. Both halves matter: a splice that rounded a text row would
/// perturb a token the tower never saw.
pub fn the_splice_replaces_exactly_the_rows_the_map_names(gpu: &Gpu) {
    let n_tok = 7usize;
    let src_rows = 3usize;
    let embedded = oracle::signal(n_tok * SPLICE_HIDDEN, 11);
    let rows = oracle::signal(src_rows * SPLICE_HIDDEN, 29);
    // Two text tokens, a run of three placeholders, two more text tokens.
    let map = [-1i32, -1, 0, 1, 2, -1, -1];

    let got = run_splice(gpu, &embedded, &rows, &map, src_rows as u32);

    for (token, &row) in map.iter().enumerate() {
        for d in 0..SPLICE_HIDDEN {
            let at = token * SPLICE_HIDDEN + d;
            let want = match usize::try_from(row) {
                Ok(row) => rows[row * SPLICE_HIDDEN + d],
                Err(_) => embedded[at],
            };
            assert_eq!(
                got[at].to_bits(),
                want.to_bits(),
                "token {token} dim {d}: map entry {row}",
            );
        }
    }
}

/// A row index past the block is skipped, not read. The host checks the row
/// count before any of this runs, so reaching here means an invariant already
/// broke; what must not happen then is the model being handed whatever memory
/// followed the tower's output and answering from it.
pub fn a_source_row_past_the_block_leaves_the_embedding_alone(gpu: &Gpu) {
    let n_tok = 3usize;
    let src_rows = 2usize;
    let embedded = oracle::signal(n_tok * SPLICE_HIDDEN, 13);
    // The source buffer is deliberately larger than the block the args declare,
    // so an out-of-range read would find plausible values rather than faulting.
    let rows = oracle::signal(8 * SPLICE_HIDDEN, 31);
    let map = [0i32, 5, -1];

    let got = run_splice(gpu, &embedded, &rows, &map, src_rows as u32);

    for d in 0..SPLICE_HIDDEN {
        assert_eq!(
            got[d].to_bits(),
            rows[d].to_bits(),
            "the in-range row is still spliced: dim {d}",
        );
        let at = SPLICE_HIDDEN + d;
        assert_eq!(
            got[at].to_bits(),
            embedded[at].to_bits(),
            "the out-of-range row was read: dim {d}",
        );
    }
}

// -- the tensor-core path (CUDA only) ----------------------------------------
//
// These entries exist only where a backend compiled them; the host finds them
// in the module's entry list and falls back to the scalar set above when they
// are absent. The oracle is the same one, and the budget is wider for one
// reason only: the operands are carried as two brain-floats each, about
// sixteen bits, where the scalar set keeps twenty-four.

/// Relative budget for the tensor-core attention against the plain softmax.
///
/// Calibrated: the widest disagreement the case below shows is between 3e-5
/// and 4e-5 of the larger of a value and one (measured on the GB10). The bound
/// is about three times that.
const MMA_TOL: f32 = 1e-4;
/// Head width of the tensor-core fixture: the served tower's, which is not a
/// whole number of sixteen-wide steps and so exercises the zero padding.
const MMA_HEAD_DIM: usize = 72;
/// A group longer than one 128-query block and not a multiple of the 64-key
/// tile, so both the partial query block and the masked key tile run.
const MMA_PLANE: usize = 200;
/// Queries one block owns and keys one tile stages.
const MMA_QUERIES: usize = 128;
const MMA_KEYS: usize = 64;
/// `VIT_FA_HD`: the padded head width the split planes are laid out in.
const MMA_PADDED_DIM: usize = 80;

/// `vit_args_unary`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CountArgs {
    count: u32,
}

/// `vit_args_attention_mma`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AttentionMmaArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    plane: u32,
    plane_pad: u32,
    scale: f32,
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// The operand split writes each value as its nearest brain-float and the
/// nearest brain-float to what that left over, and the two sum back to the
/// value to about sixteen bits.
pub fn the_operand_split_sums_back_to_its_input(gpu: &Gpu) {
    let count = 1000;
    let x: Vec<f32> = oracle::signal(count, 81)
        .iter()
        .enumerate()
        .map(|(i, v)| v * [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3][i % 7])
        .collect();
    let d_x = gpu.upload(&f32_bytes(&x));
    let out = gpu.upload(&vec![0u8; count * 4]);
    launch(
        gpu,
        "kernel_vit_to_bf16",
        Extent3::linear((count as u32).div_ceil(LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&CountArgs {
                count: count as u32,
            })),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_x.view()),
        ],
    );
    drain(gpu);
    let bytes = gpu.read(&out);
    let halves: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    for (i, want) in x.iter().enumerate() {
        let hi = bf16_to_f32(halves[i]);
        let lo = bf16_to_f32(halves[count + i]);
        // The big half is the nearest brain-float: within half a unit of the
        // last of its eight significant bits.
        assert!(
            (hi - want).abs() <= want.abs() * 2f32.powi(-8),
            "value {i}: {hi} is not the nearest brain-float to {want}"
        );
        assert!(
            (hi + lo - want).abs() <= want.abs() * 2f32.powi(-16),
            "value {i}: {hi} + {lo} does not sum back to {want}"
        );
    }
}

/// The brain-float bias repeats with its period, as the projection's own
/// epilogue would add it.
pub fn the_brain_float_bias_repeats_with_its_period(gpu: &Gpu) {
    let (rows, width) = (5usize, 13usize);
    let x = oracle::signal(rows * width, 82);
    // Values that are exact in brain-float, so the oracle needs no rounding.
    let bias: Vec<u16> = (0..width).map(|i| (0x3f80 + i * 3) as u16).collect();
    let d_x = gpu.upload(&f32_bytes(&x));
    let d_bias = gpu.upload(
        &bias
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    launch(
        gpu,
        "kernel_vit_add_bias_bf16",
        Extent3::linear(((rows * width) as u32).div_ceil(LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&AddArgs {
                count: (rows * width) as u32,
                period: width as u32,
            })),
            KernelArg::Buffer(d_x.view()),
            KernelArg::Buffer(d_bias.view()),
        ],
    );
    drain(gpu);
    let want: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, v)| v + bf16_to_f32(bias[i % width]))
        .collect();
    close(&gpu.read_f32(&d_x), &want, "brain-float bias");
}

/// Runs the split and the tensor-core attention over `rows` rows in groups of
/// `plane`, `heads` heads of `MMA_HEAD_DIM`.
fn run_attention_mma(
    gpu: &Gpu,
    (q, k, v): (&[f32], &[f32], &[f32]),
    rows: usize,
    heads: usize,
    plane: usize,
) -> Vec<f32> {
    let dim = heads * MMA_HEAD_DIM;
    let plane_pad = plane.div_ceil(MMA_KEYS) * MMA_KEYS;
    let groups = rows / plane;
    let args = AttentionMmaArgs {
        rows: rows as u32,
        heads: heads as u32,
        head_dim: MMA_HEAD_DIM as u32,
        plane: plane as u32,
        plane_pad: plane_pad as u32,
        scale: 1.0 / (MMA_HEAD_DIM as f32).sqrt(),
    };
    let split_bytes = 2 * heads * groups * plane_pad * MMA_PADDED_DIM * 2;
    let keys = gpu.upload(&vec![0u8; split_bytes]);
    let values = gpu.upload(&vec![0u8; split_bytes]);
    let (d_q, d_k, d_v) = (
        gpu.upload(&f32_bytes(q)),
        gpu.upload(&f32_bytes(k)),
        gpu.upload(&f32_bytes(v)),
    );
    let out = zeros(gpu, rows * dim);
    launch(
        gpu,
        "kernel_vit_split_kv",
        Extent3::linear((groups * plane_pad) as u32),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(keys.view()),
            KernelArg::Buffer(values.view()),
            KernelArg::Buffer(d_k.view()),
            KernelArg::Buffer(d_v.view()),
        ],
    );
    launch(
        gpu,
        "kernel_vit_attention_mma",
        Extent3::new(
            heads as u32,
            (groups * plane.div_ceil(MMA_QUERIES)) as u32,
            1,
        ),
        Extent3::linear(256),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_q.view()),
            KernelArg::Buffer(keys.view()),
            KernelArg::Buffer(values.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// The tensor-core attention agrees with the plain softmax over every patch,
/// across two temporal groups, a partial query block and a masked key tile.
pub fn the_tensor_core_attention_agrees_with_a_plain_softmax(gpu: &Gpu) {
    let heads = 2;
    let rows = 2 * MMA_PLANE;
    let dim = heads * MMA_HEAD_DIM;
    // Scaled up so the softmax is peaked rather than nearly uniform: a flat
    // distribution would hide an error in the maximum's bookkeeping.
    let scaled = |seed| -> Vec<f32> {
        oracle::signal(rows * dim, seed)
            .iter()
            .map(|v| v * 4.0)
            .collect()
    };
    let (q, k, v) = (scaled(91), scaled(92), oracle::signal(rows * dim, 93));
    let got = run_attention_mma(gpu, (&q, &k, &v), rows, heads, MMA_PLANE);
    let want = attention_oracle(&q, &k, &v, rows, heads, MMA_HEAD_DIM, MMA_PLANE);
    for (index, (&g, &w)) in got.iter().zip(&want).enumerate() {
        let bound = MMA_TOL * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= bound,
            "tensor-core attention [{index}]: got {g}, want {w}, bound {bound}"
        );
    }
}

/// The first group's rows come out the same whether or not a second group
/// follows it: no query reads a key across the boundary, padding included.
pub fn a_tensor_core_query_never_attends_across_a_group(gpu: &Gpu) {
    let heads = 2;
    let dim = heads * MMA_HEAD_DIM;
    let rows = 2 * MMA_PLANE;
    let (q, k, v) = (
        oracle::signal(rows * dim, 94),
        oracle::signal(rows * dim, 95),
        oracle::signal(rows * dim, 96),
    );
    let both = run_attention_mma(gpu, (&q, &k, &v), rows, heads, MMA_PLANE);
    let first = MMA_PLANE * dim;
    let alone = run_attention_mma(
        gpu,
        (&q[..first], &k[..first], &v[..first]),
        MMA_PLANE,
        heads,
        MMA_PLANE,
    );
    assert_eq!(
        both[..first],
        alone[..],
        "the first group moved when a second joined it"
    );
}
