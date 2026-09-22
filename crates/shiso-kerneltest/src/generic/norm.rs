//! The two fused RMS norms and the residual add.
//!
//! `kernel_rms_norm_mul_f32_4` is the one norm in a dense pass with nothing to
//! add - the first slot of a chunk normalizes the embedding it was handed -
//! and `kernel_add_rms_norm_mul_f32_4` is the fused pair every other slot
//! closes with. `kernel_add2_f32` is the residual add that writes the other of
//! the stream's two buffers.
//!
//! The norm cases assert BOTH destinations where there are two, and the plain
//! norm case checks the third binding slot in particular: the Metal entry is a
//! template over a fuse depth that declares two operand buffers whatever depth
//! it was built at, the host binds the weight twice so the destination lands in
//! the slot it belongs in, and a twin that declared four parameters instead of
//! five would write the destination into the unread operand and leave the real
//! one untouched. That failure reads as "the model answers, badly".
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "kernel argument structs take i32/u32/u64 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::TOL_F32;

/// Every entry point this file's cases launch.
pub const ENTRIES: [&str; 3] = [
    "kernel_rms_norm_mul_f32_4",
    "kernel_add_rms_norm_mul_f32_4",
    "kernel_add2_f32",
];

/// Threads per threadgroup of the norms, mirroring `shiso_engine::launch::WARP`.
const NORM_BLOCK: u32 = 32;
/// Threadgroup bytes their reduction needs, mirroring
/// `shiso_engine::launch::NORM_SMEM_BYTES`.
const NORM_SMEM: u32 = NORM_BLOCK * 4;
/// Threads per threadgroup of the flat add, mirroring
/// `shiso_engine::launch::BLOCK_LINEAR`.
const LINEAR_BLOCK: u32 = 256;
/// Elements of one activation row as the norms count them: `float4`.
const NORM_VEC: usize = 4;

/// Rows of the norm fixtures.
const ROWS: usize = 3;
/// Elements of one row: a whole number of `float4` and of a lane's stride.
const HIDDEN: usize = 64;
/// Epsilon inside the mean square.
const EPS: f32 = 1e-6;

/// The norm argument block (`metal/generic/norm.metal`), mirroring
/// `shiso_engine::launch::NormArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct NormArgs {
    ne00: i32,
    ne00_t: i32,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    eps: f32,
    nef1: [i32; 3],
    nef2: [i32; 3],
    nef3: [i32; 3],
    nbf1: [u64; 3],
    nbf2: [u64; 3],
    nbf3: [u64; 3],
}

const _: () = assert!(size_of::<NormArgs>() == 144);

/// The binary op's argument block (`metal/generic/bin.metal`), mirroring
/// `shiso_engine::launch::AddArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AddArgs {
    n: u32,
}

const _: () = assert!(size_of::<AddArgs>() == 4);

/// Launches one entry of the `generic` module.
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
        KernelRef::new("generic", entry),
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

/// The argument block the host packs for a `hidden`-wide, `rows`-tall norm.
fn norm_args(hidden: usize, rows: usize) -> NormArgs {
    let row = (hidden * 4) as u64;
    let planes = row * rows as u64;
    NormArgs {
        ne00: hidden as i32,
        ne00_t: (hidden / NORM_VEC) as i32,
        nb1: row,
        nb2: planes,
        nb3: planes,
        eps: EPS,
        nef1: [1, 1, 1],
        nef2: [1, 1, 1],
        nef3: [1, 1, 1],
        nbf1: [row, 0, 0],
        nbf2: [planes, 0, 0],
        nbf3: [planes, 0, 0],
    }
}

/// `dst = x * rsqrt(mean(x^2) + eps) * weight`, row by row.
///
/// The destination buffer starts as a recognisable non-zero pattern rather than
/// zeros, because the failure this case exists to catch - a twin whose
/// parameter list is one short, so the destination is bound into the unread
/// second fuse operand - leaves the destination exactly as it found it. Against
/// a zeroed buffer that reads as "the kernel wrote zeros"; against this one it
/// reads as "the kernel wrote nothing".
pub fn the_plain_norm_scales_the_row_and_multiplies_by_the_weight(gpu: &Gpu) {
    let count = ROWS * HIDDEN;
    let x = crate::oracle::signal(count, 71);
    let w = crate::oracle::signal(HIDDEN, 72);
    let poison: Vec<f32> = (0..count).map(|i| -1000.0 - i as f32).collect();
    let d_x = gpu.upload(&f32_bytes(&x));
    let d_w = gpu.upload(&f32_bytes(&w));
    let dst = gpu.upload(&f32_bytes(&poison));

    let args = norm_args(HIDDEN, ROWS);
    launch(
        gpu,
        "kernel_rms_norm_mul_f32_4",
        Extent3::new(ROWS as u32, 1, 1),
        Extent3::new(NORM_BLOCK, 1, 1),
        NORM_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_x.view()),
            KernelArg::Buffer(d_w.view()),
            // The second fuse operand this instantiation never reads; the host
            // binds the weight here for exactly this reason.
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(dst.view()),
        ],
    );
    drain(gpu);

    let mut want = vec![0.0f32; count];
    for r in 0..ROWS {
        let row = &x[r * HIDDEN..(r + 1) * HIDDEN];
        let mean: f32 = row.iter().map(|v| v * v).sum::<f32>() / (HIDDEN as f32);
        let inv = 1.0 / (mean + EPS).sqrt();
        for (i, &v) in row.iter().enumerate() {
            want[r * HIDDEN + i] = v * inv * w[i];
        }
    }
    close(&gpu.read_f32(&dst), &want, "plain norm");
}

/// `sum = a + b`, then `norm = sum * rsqrt(mean(sum^2) + eps) * weight`.
///
/// Both destinations are compared. The sum is what the next block adds into, so
/// a kernel that produced the right normalized row and dropped the sum would
/// pass a one-destination check and hand the stack a residual stream that never
/// accumulates.
pub fn the_fused_close_writes_both_the_sum_and_the_normalized_row(gpu: &Gpu) {
    let count = ROWS * HIDDEN;
    let a = crate::oracle::signal(count, 73);
    let b = crate::oracle::signal(count, 74);
    let w = crate::oracle::signal(HIDDEN, 75);
    let d_a = gpu.upload(&f32_bytes(&a));
    let d_b = gpu.upload(&f32_bytes(&b));
    let d_w = gpu.upload(&f32_bytes(&w));
    let sum = zeros(gpu, count);
    let norm = zeros(gpu, count);

    let args = norm_args(HIDDEN, ROWS);
    launch(
        gpu,
        "kernel_add_rms_norm_mul_f32_4",
        Extent3::new(ROWS as u32, 1, 1),
        Extent3::new(NORM_BLOCK, 1, 1),
        NORM_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_a.view()),
            KernelArg::Buffer(d_b.view()),
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(sum.view()),
            KernelArg::Buffer(norm.view()),
        ],
    );
    drain(gpu);

    let want_sum: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
    let mut want_norm = vec![0.0f32; count];
    for r in 0..ROWS {
        let row = &want_sum[r * HIDDEN..(r + 1) * HIDDEN];
        let mean: f32 = row.iter().map(|v| v * v).sum::<f32>() / (HIDDEN as f32);
        let inv = 1.0 / (mean + EPS).sqrt();
        for (i, &v) in row.iter().enumerate() {
            want_norm[r * HIDDEN + i] = v * inv * w[i];
        }
    }
    close(&gpu.read_f32(&sum), &want_sum, "fused close sum");
    close(&gpu.read_f32(&norm), &want_norm, "fused close norm");
}

/// Elements of the add fixture: a whole number of threadgroups plus a tail.
const ADD_N: usize = 1000;

/// `out[i] = a[i] + b[i]`, stopping at `n`.
///
/// The destination is longer than `n` and starts poisoned, because the bound is
/// the only thing between this kernel and the next allocation.
pub fn the_residual_add_writes_its_own_destination_and_stops_at_its_bound(gpu: &Gpu) {
    let a = crate::oracle::signal(ADD_N, 76);
    let b = crate::oracle::signal(ADD_N, 77);
    let d_a = gpu.upload(&f32_bytes(&a));
    let d_b = gpu.upload(&f32_bytes(&b));
    let tail = 7usize;
    let poison: Vec<f32> = (0..ADD_N + tail).map(|i| -1000.0 - i as f32).collect();
    let out = gpu.upload(&f32_bytes(&poison));

    let args = AddArgs { n: ADD_N as u32 };
    launch(
        gpu,
        "kernel_add2_f32",
        Extent3::linear((ADD_N as u32).div_ceil(LINEAR_BLOCK)),
        Extent3::new(LINEAR_BLOCK, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_a.view()),
            KernelArg::Buffer(d_b.view()),
            KernelArg::Buffer(out.view()),
        ],
    );
    drain(gpu);

    let got = gpu.read_f32(&out);
    let want: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x + y).collect();
    close(&got[..ADD_N], &want, "residual add");
    for (offset, &value) in got[ADD_N..].iter().enumerate() {
        let expected = poison[ADD_N + offset];
        assert!(
            (value - expected).abs() < f32::EPSILON,
            "element {} is past the bound and was overwritten: {value} instead of {expected}",
            ADD_N + offset
        );
    }
}
