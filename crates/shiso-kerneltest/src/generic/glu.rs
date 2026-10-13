//! The gated `SiLU`, which is the activation a qwen35 feed-forward closes on.
//!
//! `kernel_swiglu_flat_f32` is the generic module's half of the pair
//! `crates/shiso-engine/src/ffn.rs` chooses between: this curve for a qwen35
//! checkpoint and `kernel_dense_geglu_flat_f32` for a Gemma-shaped one. The
//! dense twin has been graded since Phase 7 by
//! `crates/shiso-kerneltest/src/dense/misc.rs`; this one had no case anywhere
//! in the tree, on either backend, until Phase 8 - and no CUDA entry either, so
//! a qwen35 prefill refused at its first feed-forward.
//!
//! The clamp is exercised as well as the unclamped path the engine dispatches.
//! `glu_args` sets `limit` to zero and `alpha` to one at every call site today,
//! but the field is in the block and the kernel reads it, and an entry whose
//! untaken branch is graded by nothing is an entry that will be wrong the day
//! something takes it.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the kernel argument struct takes i32/u64 dimensions while the fixture is written in usize, and every fixture dimension here is a small literal"
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::TOL_F32;

/// Every entry point this file's cases launch.
pub const ENTRIES: [&str; 1] = ["kernel_swiglu_flat_f32"];

/// Threads per threadgroup, mirroring `shiso_engine::launch::BLOCK_LINEAR`.
const LINEAR_BLOCK: u32 = 256;

/// The gate unit's argument block (`metal/generic/glu.metal`), mirroring
/// `shiso_engine::launch::GluArgs`.
///
/// The padding is spelled out because `bytemuck::Pod` refuses a struct with
/// implicit padding, and the declared layout leaves four bytes before each
/// 64-bit stride.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GluArgs {
    ne00: i32,
    pad0: [u8; 4],
    nb01: u64,
    ne10: i32,
    pad1: [u8; 4],
    nb11: u64,
    ne0: i32,
    pad2: [u8; 4],
    nb1: u64,
    i00: i32,
    i10: i32,
    alpha: f32,
    limit: f32,
}

const _: () = assert!(size_of::<GluArgs>() == 64);

/// Elements of the activation fixture: four whole threadgroups of the flat
/// kernel, which is one token's inner row at a plausible width.
const GLU_N: usize = 1024;

/// The sigmoid-weighted linear unit, as the reference writes it.
fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// An f32 device buffer of `len` zeros.
fn zeros(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![0.0f32; len]))
}

/// Asserts every value agrees within `TOL_F32`, scaled by the expected
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

/// Launches the entry over a flat row of `GLU_N` elements.
fn launch_flat(gpu: &Gpu, gate: &[f32], up: &[f32], alpha: f32, limit: f32) -> Vec<f32> {
    let d_gate = gpu.upload(&f32_bytes(gate));
    let d_up = gpu.upload(&f32_bytes(up));
    let dst = zeros(gpu, GLU_N);

    let row = (GLU_N * 4) as u64;
    let args = GluArgs {
        ne00: GLU_N as i32,
        pad0: [0; 4],
        nb01: row,
        ne10: GLU_N as i32,
        pad1: [0; 4],
        nb11: row,
        ne0: GLU_N as i32,
        pad2: [0; 4],
        nb1: row,
        i00: 0,
        i10: 0,
        alpha,
        limit,
    };
    run(
        gpu,
        KernelRef::new("generic", "kernel_swiglu_flat_f32"),
        &LaunchArgs {
            grid: Extent3::linear((GLU_N as u32).div_ceil(LINEAR_BLOCK)),
            block: Extent3::new(LINEAR_BLOCK, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d_gate.view()),
                KernelArg::Buffer(d_up.view()),
                KernelArg::Buffer(dst.view()),
            ],
        },
    );
    drain(gpu);
    gpu.read_f32(&dst)
}

/// `dst[i] = silu(gate[i]) * up[i] * alpha`, against the curve written out as a
/// plain loop.
///
/// The second half of the case asserts the fixture can tell the gate row from
/// the up row: a kernel that swapped the two sources would agree with the first
/// half only if the activation were symmetric.
///
/// # Panics
///
/// When the device disagrees with the oracle, which is what the case is for.
pub fn the_gated_silu_is_the_sigmoid_weighted_gate_times_the_up_row(gpu: &Gpu) {
    // Widened past the unit interval `signal` produces, so the exponential is
    // evaluated somewhere its curvature matters rather than only where it is
    // nearly linear.
    let gate: Vec<f32> = crate::oracle::signal(GLU_N, 31)
        .iter()
        .map(|v| v * 8.0)
        .collect();
    let up = crate::oracle::signal(GLU_N, 32);

    let got = launch_flat(gpu, &gate, &up, 1.0, 0.0);
    let want: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
    close(&got, &want, "swiglu");

    let swapped: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(u) * g).collect();
    let differing = want
        .iter()
        .zip(&swapped)
        .filter(|(a, b)| (*a - *b).abs() > 1e-3)
        .count();
    assert!(
        differing > GLU_N / 2,
        "the fixture does not tell the gate half from the up half",
    );
}

/// A positive `limit` clamps the gate above and the up row on both sides, and
/// `alpha` multiplies what comes out.
///
/// The asymmetry is the part worth pinning: the gate takes a one-sided `min`
/// and the up row a two-sided clamp, which is what the Metal body does, and a
/// twin that clamped both the same way would pass the case above and fail here.
///
/// # Panics
///
/// When the device disagrees with the oracle, which is what the case is for.
pub fn the_clamp_is_one_sided_on_the_gate_and_two_sided_on_the_up_row(gpu: &Gpu) {
    let limit = 2.0f32;
    let alpha = 0.5f32;
    // Both rows reach past the limit at both ends, so every branch of the pair
    // is taken somewhere in the row.
    let gate: Vec<f32> = crate::oracle::signal(GLU_N, 33)
        .iter()
        .map(|v| v * 6.0)
        .collect();
    let up: Vec<f32> = crate::oracle::signal(GLU_N, 34)
        .iter()
        .map(|v| v * 6.0)
        .collect();

    let got = launch_flat(gpu, &gate, &up, alpha, limit);
    let want: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| {
            let g = g.min(limit);
            let u = u.clamp(-limit, limit);
            silu(g) * u * alpha
        })
        .collect();
    close(&got, &want, "swiglu clamped");
}

/// Elements of the bench row: one token's FFN inner width at a common size.
const BENCH_GLU_N: usize = 14_336;

/// One token's gated `SiLU` over a `BENCH_GLU_N`-wide row, for [`crate::bench`].
pub(crate) fn bench_swiglu(gpu: &Gpu) -> crate::bench::Fixture {
    let n = BENCH_GLU_N;
    let d_gate = gpu.upload(&f32_bytes(&crate::oracle::signal(n, 11)));
    let d_up = gpu.upload(&f32_bytes(&crate::oracle::signal(n, 12)));
    let dst = zeros(gpu, n);
    let row = (n * 4) as u64;
    let args = GluArgs {
        ne00: n as i32,
        pad0: [0; 4],
        nb01: row,
        ne10: n as i32,
        pad1: [0; 4],
        nb11: row,
        ne0: n as i32,
        pad2: [0; 4],
        nb1: row,
        i00: 0,
        i10: 0,
        alpha: 1.0,
        limit: 0.0,
    };
    crate::bench::Fixture::new(3 * row, 0, move |gpu| {
        run(
            gpu,
            KernelRef::new("generic", "kernel_swiglu_flat_f32"),
            &LaunchArgs::new(
                Extent3::linear((n as u32).div_ceil(LINEAR_BLOCK)),
                Extent3::new(LINEAR_BLOCK, 1, 1),
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(d_gate.view()),
                    KernelArg::Buffer(d_up.view()),
                    KernelArg::Buffer(dst.view()),
                ],
            ),
        );
    })
}
