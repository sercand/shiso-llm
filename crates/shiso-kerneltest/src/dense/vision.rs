//! Device cases for the Gemma 4 vision tower's three dense kernels, plus the
//! attention entry the tower shares with the Qwen one.
//!
//! The three here are the ones a second family added: a rotation over two
//! spatial axes instead of the patch's grid position, a pooling fold instead of
//! a spatial merge, and the elementwise clamp every clamped linear projection
//! is built from. The fourth case belongs to the `vision` module, whose score
//! scale stopped being derived inside the kernel when this tower arrived — it
//! wants 1.0 and the Qwen tower wants its head's inverse root width, so the
//! number is an argument and an argument can be wrong.
//!
//! Same shape as [`crate::dense::misc`]: every case compares the device against
//! the simplest possible reading of the formula, written as a plain Rust `f32`
//! loop, so a disagreement points at the kernel or at its launch geometry
//! rather than at a second, equally subtle implementation.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "kernel argument structs take i32/u32/u64 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]
#![expect(
    clippy::float_cmp,
    reason = "assertions compare known values or exact clamps"
)]
#![expect(
    clippy::too_many_arguments,
    reason = "device test launcher forwards all plane arrays and parameters"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "math formulas use q, k, v, etc."
)]
#![expect(clippy::similar_names, reason = "d_qw and d_kw mirror q and k")]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::oracle;
use crate::oracle::TOL_F32;
use crate::{Gpu, drain, f32_bytes, run};

/// Threads per threadgroup of the flat kernels, mirroring
/// `shiso_engine::launch::BLOCK_LINEAR`.
const LINEAR_BLOCK: u32 = 256;
/// Threads the rotation and the attention are launched with.
const ATTN_BLOCK: u32 = 128;
/// Floats the dense block reduction's scratch holds: 32 warp partials plus the
/// broadcast slot. `DENSE_BLOCK_SUM_SLOTS` in `dense_common.h`.
const BLOCK_SUM_SLOTS: u32 = 33;

/// The rotation kernel's fixture widths: three heads of sixty-four, which is
/// the shipped tower's head, on a 4x5 patch grid.
const HEADS: u32 = 3;
const HEAD_DIM: u32 = 64;
const GRID_H: usize = 4;
const GRID_W: usize = 5;

/// The entry points this file grades in the `dense` module.
///
/// A backend driver walks this list and asks its loader to resolve each one,
/// without a device gate: a misnamed entry is a runtime failure with no
/// compile-time warning anywhere.
pub const ENTRIES: [&str; 3] = [
    "kernel_dense_vit_qkv_norm_rope2d",
    "kernel_dense_vit_avgpool3_scale",
    "kernel_dense_clamp_f32",
];

// -- argument blocks --------------------------------------------------------

/// `dense_args_vit_rope2d`, mirroring `shiso_engine::launch::DenseVitRope2dArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VitRope2dArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    grid_w: u32,
    eps: f32,
    rope_base: f32,
}

const _: () = assert!(size_of::<VitRope2dArgs>() == 24);

/// `dense_args_vit_pool`, mirroring `shiso_engine::launch::DenseVitPoolArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VitPoolArgs {
    grid_h: u32,
    grid_w: u32,
    hidden: u32,
    scale: f32,
}

const _: () = assert!(size_of::<VitPoolArgs>() == 16);

/// `dense_args_clamp`, mirroring `shiso_engine::launch::ClampArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ClampArgs {
    count: u32,
    min: f32,
    max: f32,
}

const _: () = assert!(size_of::<ClampArgs>() == 12);

/// `vit_args_attention`, mirroring the engine's mirror of the same name. The
/// scale is the field this tower added; a mirror that drifted from the kernel
/// would launch garbage past the first four fields.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AttentionArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    plane: u32,
    scale: f32,
}

const _: () = assert!(size_of::<AttentionArgs>() == 20);

// -- helpers ----------------------------------------------------------------

/// Launches one dense entry point, naming the geometry if the driver refuses it.
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
        KernelRef::new("dense", entry),
        &LaunchArgs {
            grid,
            block,
            smem_bytes,
            args,
        },
    );
}

/// Launches one vision-set entry point. The attention under test lives in the
/// `vision` module, not the dense one.
fn launch_vision(
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

/// An f32 device buffer of `len` zeros.
fn zeros(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![0.0f32; len]))
}

/// Asserts every value agrees within [`TOL_F32`], scaled by the expected
/// magnitude.
fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = TOL_F32 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= bound,
            "{what}: element {index} is {g}, expected {w} (tolerance {bound})",
        );
    }
}

// -- the fused rotation ------------------------------------------------------

/// The whole of `kernel_dense_vit_qkv_norm_rope2d`, read straight off its
/// comment: per-head RMS norm of the three planes (q and k weighted, v not),
/// then two neox rotations over the two halves of the head, the leading half
/// by the patch's column and the trailing half by its row.
///
/// The sums are plain left folds and the rotations walk the pairs in order,
/// which is not the device's reduction order; the case is held to `TOL_F32`
/// for that reason, the same way the norm kernels are.
#[expect(
    clippy::too_many_arguments,
    reason = "the fixture's five widths and two scalars"
)]
fn rope2d_oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_w: &[f32],
    k_w: &[f32],
    rows: usize,
    grid_w: usize,
    eps: f32,
    base: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = (HEADS * HEAD_DIM) as usize;
    let half = (HEAD_DIM / 2) as usize;
    let quarter = half / 2;
    let mut out_q = q.to_vec();
    let mut out_k = k.to_vec();
    let mut out_v = v.to_vec();
    for n in 0..rows {
        let col = (n % grid_w) as f32;
        let row = (n / grid_w) as f32;
        for h in 0..HEADS as usize {
            let off = n * dim + h * HEAD_DIM as usize;
            for (plane, weight, out) in [
                (q, Some(q_w), &mut out_q),
                (k, Some(k_w), &mut out_k),
                (v, None, &mut out_v),
            ] {
                let ms: f32 = (0..HEAD_DIM as usize)
                    .map(|i| {
                        let x = plane[off + i];
                        x * x
                    })
                    .sum::<f32>()
                    / HEAD_DIM as f32;
                let inv = 1.0 / (ms + eps).sqrt();
                for i in 0..HEAD_DIM as usize {
                    let w = weight.map_or(1.0, |w| w[i]);
                    out[off + i] = plane[off + i] * inv * w;
                }
            }
            // The two rotations, each over its own half of the head, each
            // pairing `(j, j + quarter)` at its own axis's angle. The
            // normalized values are what rotate, so the staging order is
            // norm-then-rotate exactly as the kernel's shared memory is.
            for (pos, shift) in [(col, 0usize), (row, half)] {
                for j in 0..quarter {
                    let theta = pos * base.powf(-2.0 * (j as f32) / half as f32);
                    let (s, c) = (theta.sin(), theta.cos());
                    let a = &mut out_q[off + shift + j..=off + shift + j + quarter];
                    let (x0, x1) = (a[0], a[quarter]);
                    a[0] = x0 * c - x1 * s;
                    a[quarter] = x0 * s + x1 * c;
                    let b = &mut out_k[off + shift + j..=off + shift + j + quarter];
                    let (x0, x1) = (b[0], b[quarter]);
                    b[0] = x0 * c - x1 * s;
                    b[quarter] = x0 * s + x1 * c;
                }
            }
        }
    }
    (out_q, out_k, out_v)
}

/// Runs the fused kernel over the three planes and reads all three back.
fn run_rope2d(
    gpu: &Gpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_w: &[f32],
    k_w: &[f32],
    rows: usize,
    grid_w: u32,
    eps: f32,
    base: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d_q = gpu.upload(&f32_bytes(q));
    let d_k = gpu.upload(&f32_bytes(k));
    let d_v = gpu.upload(&f32_bytes(v));
    let d_qw = gpu.upload(&f32_bytes(q_w));
    let d_kw = gpu.upload(&f32_bytes(k_w));
    let args = VitRope2dArgs {
        rows: rows as u32,
        heads: HEADS,
        head_dim: HEAD_DIM,
        grid_w,
        eps,
        rope_base: base,
    };
    let smem = 4 * (BLOCK_SUM_SLOTS + 3 * HEAD_DIM);
    launch(
        gpu,
        "kernel_dense_vit_qkv_norm_rope2d",
        Extent3::new(HEADS, rows as u32, 1),
        Extent3::linear(ATTN_BLOCK),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_q.view()),
            KernelArg::Buffer(d_k.view()),
            KernelArg::Buffer(d_v.view()),
            KernelArg::Buffer(d_qw.view()),
            KernelArg::Buffer(d_kw.view()),
        ],
    );
    drain(gpu);
    (gpu.read_f32(&d_q), gpu.read_f32(&d_k), gpu.read_f32(&d_v))
}

/// The two halves of each head rotate by the two different spatial axes.
pub fn rope2d_rotates_each_part_by_its_own_axis(gpu: &Gpu) {
    let rows = GRID_H * GRID_W;
    let dim = (HEADS * HEAD_DIM) as usize;
    let q = oracle::signal(rows * dim, 11);
    let k = oracle::signal(rows * dim, 12);
    let v = oracle::signal(rows * dim, 13);
    // Non-uniform weights, so the case pins that q and k carry a learned gain
    // and v does not: with ones the weighted and unnormalized readings of the
    // value plane would differ only by the root mean square, and a kernel that
    // skipped v's normalization entirely would pass.
    let q_w = oracle::signal(HEAD_DIM as usize, 14);
    let k_w = oracle::signal(HEAD_DIM as usize, 15);
    let eps = 1e-6;
    let base = 100.0;

    let (got_q, got_k, got_v) =
        run_rope2d(gpu, &q, &k, &v, &q_w, &k_w, rows, GRID_W as u32, eps, base);
    let (want_q, want_k, want_v) = rope2d_oracle(&q, &k, &v, &q_w, &k_w, rows, GRID_W, eps, base);
    close(&got_q, &want_q, "rope2d q");
    close(&got_k, &want_k, "rope2d k");
    close(&got_v, &want_v, "rope2d v");

    // The axes are not interchangeable: the same 20 rows re-cut onto a 2-wide
    // grid give every patch except the first two a different column and a
    // different row, so the two halves of each head rotate by different angles
    // than the run above. A kernel that fed one axis to both halves — or the
    // same axis twice — would agree with the oracle only if the oracle made
    // the same mistake, and cannot agree with both runs at once.
    let (swap_q, ..) = run_rope2d(gpu, &q, &k, &v, &q_w, &k_w, rows, 2, eps, base);
    let differing = got_q
        .iter()
        .zip(&swap_q)
        .filter(|(a, b)| (*a - *b).abs() > 1e-4)
        .count();
    assert!(
        differing > rows * dim / 2,
        "a grid re-cut from 4x5 to 10x2 left {differing} of {} q values unchanged: \
         the two halves of the head are not rotating by the two different axes",
        rows * dim,
    );
}

// -- the pooling fold ---------------------------------------------------------

/// The `3 x 3` average pool folds nine patches and applies its scale.
pub fn avgpool3_averages_nine_patches_and_scales(gpu: &Gpu) {
    // A 6x9 patch grid of the shipped width: two rows of three pooled tokens,
    // and a scale that is the tower's, not a fixture's. Both extents are
    // whole multiples of three, which is what the resize rule guarantees.
    let hidden = 768usize;
    let scale = (hidden as f32).sqrt();
    let (grid_h, grid_w) = (6u32, 9u32);
    let rows_in = (grid_h as usize) * (grid_w as usize);
    let input = oracle::signal(rows_in * hidden, 21);
    let out_len = (rows_in / 9) * hidden;
    let out = zeros(gpu, out_len);
    let d_in = gpu.upload(&f32_bytes(&input));
    let args = VitPoolArgs {
        grid_h,
        grid_w,
        hidden: hidden as u32,
        scale,
    };
    launch(
        gpu,
        "kernel_dense_vit_avgpool3_scale",
        Extent3::linear((out_len as u32).div_ceil(LINEAR_BLOCK)),
        Extent3::new(LINEAR_BLOCK, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_in.view()),
        ],
    );
    drain(gpu);

    let out_w = (grid_w / 3) as usize;
    let want: Vec<f32> = (0..out_len)
        .map(|i| {
            let orow = i / hidden;
            let c = i % hidden;
            let oy = orow / out_w;
            let ox = orow % out_w;
            let mut sum = 0.0f32;
            for dy in 0..3 {
                for dx in 0..3 {
                    let sy = oy * 3 + dy;
                    let sx = ox * 3 + dx;
                    sum += input[(sy * grid_w as usize + sx) * hidden + c];
                }
            }
            (sum / 9.0) * scale
        })
        .collect();
    close(&gpu.read_f32(&out), &want, "avgpool3");
}

// -- the clamp -----------------------------------------------------------------

/// A count that is not a whole number of threadgroups: the tail guard is part
/// of what this case pins, and a multiple of the block would launch exactly
/// enough threads to hide a missing bound.
const CLAMP_N: usize = 1000;

/// The clamp is a plain elementwise selection, and safe to alias.
pub fn clamp_is_elementwise(gpu: &Gpu) {
    let input = oracle::signal(CLAMP_N, 31);
    let (min, max) = (-0.5f32, 0.25f32);
    let want: Vec<f32> = input.iter().map(|x| x.clamp(min, max)).collect();

    let out = zeros(gpu, CLAMP_N);
    let d_in = gpu.upload(&f32_bytes(&input));
    let args = ClampArgs {
        count: CLAMP_N as u32,
        min,
        max,
    };
    launch(
        gpu,
        "kernel_dense_clamp_f32",
        Extent3::linear((CLAMP_N as u32).div_ceil(LINEAR_BLOCK)),
        Extent3::new(LINEAR_BLOCK, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_in.view()),
        ],
    );
    drain(gpu);
    assert_eq!(
        gpu.read_f32(&out),
        want,
        "the clamp is not a plain selection"
    );

    // The fixture must actually reach both bounds and the pass-through, or the
    // case above proves nothing about which end is which.
    assert!(
        want.iter().filter(|x| **x == min).count() > CLAMP_N / 8
            && want.iter().filter(|x| **x == max).count() > CLAMP_N / 8
            && want.iter().any(|x| *x != min && *x != max),
        "the clamp fixture does not exercise both bounds and the interior",
    );

    // In place: the kernel documents that handing the same buffer as both
    // `out` and `in` is safe, because each element is read and written once by
    // the same thread. Running the aliased pair over a fresh copy of the
    // input must reproduce the out-of-place answer exactly.
    let d_again = gpu.upload(&f32_bytes(&input));
    launch(
        gpu,
        "kernel_dense_clamp_f32",
        Extent3::linear((CLAMP_N as u32).div_ceil(LINEAR_BLOCK)),
        Extent3::new(LINEAR_BLOCK, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_again.view()),
            KernelArg::Buffer(d_again.view()),
        ],
    );
    drain(gpu);
    let aliased = gpu.read_f32(&d_again);
    assert_eq!(aliased, want, "the in-place clamp differs from the copy");
}

// -- the attention's scale -------------------------------------------------------

/// Runs `kernel_vit_attention` over one plane of `rows` patches at a given
/// score scale and reads the output back.
fn run_attention_scaled(gpu: &Gpu, q: &[f32], k: &[f32], v: &[f32], scale: f32) -> Vec<f32> {
    let rows = q.len() / ((HEADS * HEAD_DIM) as usize);
    let out = zeros(gpu, q.len());
    let d_q = gpu.upload(&f32_bytes(q));
    let d_k = gpu.upload(&f32_bytes(k));
    let d_v = gpu.upload(&f32_bytes(v));
    let warps = ATTN_BLOCK / 32;
    let args = AttentionArgs {
        rows: rows as u32,
        heads: HEADS,
        head_dim: HEAD_DIM,
        plane: rows as u32,
        scale,
    };
    launch_vision(
        gpu,
        "kernel_vit_attention",
        Extent3::new(HEADS, rows as u32, 1),
        Extent3::linear(ATTN_BLOCK),
        4 * (HEAD_DIM * (warps + 1) + 2 * warps),
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

/// Plain two-pass softmax attention, the same oracle
/// [`crate::vision::tower`] runs, with the scale as an argument rather than a
/// derived constant.
fn attention_oracle(q: &[f32], k: &[f32], v: &[f32], scale: f32) -> Vec<f32> {
    let dim = (HEADS * HEAD_DIM) as usize;
    let rows = q.len() / dim;
    let mut out = vec![0.0f32; q.len()];
    for n in 0..rows {
        for h in 0..HEADS as usize {
            let off = h * HEAD_DIM as usize;
            let scores: Vec<f32> = (0..rows)
                .map(|p| {
                    let dot: f32 = (0..HEAD_DIM as usize)
                        .map(|d| q[n * dim + off + d] * k[p * dim + off + d])
                        .sum();
                    dot * scale
                })
                .collect();
            let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
            let den: f32 = weights.iter().sum();
            for d in 0..HEAD_DIM as usize {
                let num: f32 = weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| w * v[i * dim + off + d])
                    .sum();
                out[n * dim + off + d] = num / den;
            }
        }
    }
    out
}

/// The attention takes its softmax scale from its argument block.
pub fn vit_attention_takes_its_scale(gpu: &Gpu) {
    let rows = 12usize;
    let dim = (HEADS * HEAD_DIM) as usize;
    let q = oracle::signal(rows * dim, 41);
    let k = oracle::signal(rows * dim, 42);
    let v = oracle::signal(rows * dim, 43);

    // This tower's scale: the scores reach the softmax unscaled.
    let wide = run_attention_scaled(gpu, &q, &k, &v, 1.0);
    close(
        &wide,
        &attention_oracle(&q, &k, &v, 1.0),
        "attention scale 1.0",
    );
    // The Qwen tower's scale, which is what the kernel used to derive itself
    // at this head width. Passing it explicitly must reproduce the old
    // kernel's answer on the same input — the Qwen encoder's device cases in
    // `shiso-models` pin the same number end to end.
    let qwen_scale = 1.0 / (HEAD_DIM as f32).sqrt();
    assert_eq!(qwen_scale, 0.125, "a head width change moved the fixture");
    let narrow = run_attention_scaled(gpu, &q, &k, &v, qwen_scale);
    close(
        &narrow,
        &attention_oracle(&q, &k, &v, qwen_scale),
        "attention scale 0.125",
    );

    // And the two are not the same answer: a kernel that ignored its argument
    // would agree with whichever oracle happens to match its old constant.
    let differing = wide
        .iter()
        .zip(&narrow)
        .filter(|(a, b)| (*a - *b).abs() > 1e-3)
        .count();
    assert!(
        differing > wide.len() / 4,
        "scale 1.0 and scale 0.125 agree on {} of {} values; the kernel may be \
         ignoring its argument",
        wide.len() - differing,
        wide.len(),
    );
}
