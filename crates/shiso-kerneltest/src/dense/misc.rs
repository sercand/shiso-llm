//! Device cases for the dense set's activation, scaled close, softcap and row
//! scale.
//!
//! These are the parametric helpers Gemma 4 needs and qwen35 does not: a gated
//! GELU in place of the gated `SiLU`, a residual close that scales the sum before
//! it normalizes it, the logit softcap and a plain row scale.
//!
//! Every case compares the device against the simplest possible reading of the
//! formula, written as a plain Rust `f32` loop, so a disagreement points at the
//! kernel or at its launch geometry rather than at a second, equally subtle
//! implementation.
//!
//! The `#[test]` wrappers live in each backend's own driver -
//! `crates/shiso-metal/tests/dense_misc.rs`,
//! `crates/shiso-cuda/tests/dense_misc.rs` - which open a device and call these
//! functions. [`ENTRIES`] is here rather than there because the twin list is a
//! property of the kernel set, not of a device: a backend missing an entry the
//! other exports should fail on the name, before any fixture is built.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "kernel argument structs take i32/u32/u64 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]
#![expect(
    clippy::excessive_precision,
    reason = "the activation constant is written with the digits the reference gives it, which round to the f32 the kernel holds"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::oracle::TOL_F32;

/// Threads per threadgroup of the flat kernels, mirroring
/// `shiso_engine::launch::BLOCK_LINEAR`.
const LINEAR_BLOCK: u32 = 256;
/// Threads per threadgroup of the norm, mirroring `shiso_engine::launch::WARP`.
const NORM_BLOCK: u32 = 32;
/// Threadgroup bytes the norm's reduction needs: one f32 per lane, mirroring
/// `shiso_engine::launch::NORM_SMEM_BYTES`.
const NORM_SMEM: u32 = NORM_BLOCK * 4;
/// Elements of one activation row as the norm counts them: `float4`.
const NORM_VEC: usize = 4;

/// Every entry point this module's cases launch.
pub const ENTRIES: [&str; 4] = [
    "kernel_dense_geglu_flat_f32",
    "kernel_dense_add_scale_rms_norm_mul_f32_4",
    "kernel_dense_softcap_f32",
    "kernel_dense_scale_f32",
];

// -- argument blocks --------------------------------------------------------

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

/// The normalization's argument block (`metal/generic/norm.metal`), mirroring
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

/// `dense_args_softcap`, mirroring `shiso_engine::launch::SoftcapArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SoftcapArgs {
    n: u32,
    cap: f32,
}

const _: () = assert!(size_of::<SoftcapArgs>() == 8);

/// `dense_args_scale`, mirroring `shiso_engine::launch::ScaleArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ScaleArgs {
    n: u32,
    s: f32,
}

const _: () = assert!(size_of::<ScaleArgs>() == 8);

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

/// The tanh approximation of the Gaussian error linear unit, as the reference
/// writes it.
fn gelu_tanh(v: f32) -> f32 {
    let c = 0.797_884_560_802_865_4_f32;
    0.5 * v * (1.0 + (c * (v + 0.044_715 * v * v * v)).tanh())
}

// -- the gated GELU ---------------------------------------------------------

/// Elements of the activation fixture: four whole threadgroups of the flat
/// kernel, which is one token's inner row at a plausible width.
const GLU_N: usize = 1024;

/// `dst[i] = dense_gelu_tanh(gate[i]) * up[i] * alpha`, against the tanh
/// approximation written out as a plain loop.
///
/// The second half of the case asserts the fixture can tell the gate row from
/// the up row: a kernel that swapped the two sources would agree with the first
/// half only if the activation were symmetric.
pub fn the_gated_gelu_is_the_tanh_approximation_times_the_up_row(gpu: &Gpu) {
    let gate = crate::oracle::signal(GLU_N, 21);
    let up = crate::oracle::signal(GLU_N, 22);
    let d_gate = gpu.upload(&f32_bytes(&gate));
    let d_up = gpu.upload(&f32_bytes(&up));
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
        alpha: 1.0,
        limit: 0.0,
    };
    launch(
        gpu,
        "kernel_dense_geglu_flat_f32",
        Extent3::linear((GLU_N as u32).div_ceil(LINEAR_BLOCK)),
        Extent3::new(LINEAR_BLOCK, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_gate.view()),
            KernelArg::Buffer(d_up.view()),
            KernelArg::Buffer(dst.view()),
        ],
    );
    drain(gpu);

    let want: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| gelu_tanh(g) * u)
        .collect();
    close(&gpu.read_f32(&dst), &want, "geglu");

    // The gate half is not the up half: a kernel that swapped the two sources
    // would agree with the oracle above only if the activation were symmetric,
    // and this asserts that it is not.
    let swapped: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| gelu_tanh(u) * g)
        .collect();
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

/// A gate past the point where the activation's cubic overflows still produces
/// a finite value, and the right one.
///
/// The bounded fixture above cannot reach this: a gate of about 10.7 puts the
/// tanh argument past 44, which is where a rational tanh in `exp(2y)` hands back
/// inf/inf. Gemma 4's first feed-forward produces gates of twenty routinely, and
/// one NaN element poisons the whole projection below it.
pub fn the_gated_gelu_saturates_on_a_large_gate_instead_of_overflowing_to_nan(gpu: &Gpu) {
    // The case the bounded fixture above cannot reach. The activation's cubic
    // grows fast: a gate of about 10.7 already puts the tanh argument past 44,
    // which is where a rational tanh in `exp(2y)` overflows and hands back
    // inf/inf. Gemma 4's first feed-forward produces gates of twenty routinely,
    // and one NaN element poisons an entire row of the projection below it, so
    // this is the difference between a model that answers and a model whose
    // every logit is NaN.
    // Spread across the threshold and well past it, both signs, with the
    // ordinary range still represented: the fix must not move those.
    let gate: Vec<f32> = (0..GLU_N)
        .map(|i| {
            let step = (i % 64) as f32;
            let magnitude = 0.5f32.mul_add(step, 0.5);
            if i % 2 == 0 { magnitude } else { -magnitude }
        })
        .collect();
    let up = vec![1.0f32; GLU_N];
    let d_gate = gpu.upload(&f32_bytes(&gate));
    let d_up = gpu.upload(&f32_bytes(&up));
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
        alpha: 1.0,
        limit: 0.0,
    };
    launch(
        gpu,
        "kernel_dense_geglu_flat_f32",
        Extent3::linear((GLU_N as u32).div_ceil(LINEAR_BLOCK)),
        Extent3::new(LINEAR_BLOCK, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_gate.view()),
            KernelArg::Buffer(d_up.view()),
            KernelArg::Buffer(dst.view()),
        ],
    );
    drain(gpu);

    let got = gpu.read_f32(&dst);
    for (index, (&value, &x)) in got.iter().zip(&gate).enumerate() {
        assert!(
            value.is_finite(),
            "element {index}: a gate of {x} produced {value}",
        );
    }
    // Saturation is not merely finiteness: past the threshold the activation IS
    // the identity for a positive gate and zero for a negative one, and a kernel
    // that clamped the gate itself rather than its tanh would pass the finiteness
    // check while capping the value the feed-forward carries.
    let want: Vec<f32> = gate.iter().map(|&g| gelu_tanh(g)).collect();
    close(&got, &want, "geglu over a saturating gate");
}

// -- the scaled close -------------------------------------------------------

/// Rows of the norm fixture.
const NORM_ROWS: usize = 3;
/// Elements of one norm row: a whole number of `float4` and of a simdgroup's
/// stride, which is what the kernel's work loop assumes.
const NORM_HIDDEN: usize = 64;
/// Epsilon inside the mean square.
const NORM_EPS: f32 = 1e-6;
/// The scale the fixture folds in before it normalizes.
const NORM_SCALE: f32 = 0.5;

/// `sum = (src0 + src1) * scale`, then `norm = sum * rsqrt(mean(sum^2) + eps) *
/// weight`.
///
/// The scale folded in before the store AND before the sum of squares is the
/// only thing separating this entry from `kernel_add_rms_norm_mul_f32_4`, and
/// getting it wrong produces a plausible wrong answer rather than a visible
/// one - so both outputs are compared, not just the normalized one.
pub fn the_scaled_close_scales_the_sum_before_it_stores_and_before_it_reduces(gpu: &Gpu) {
    let count = NORM_ROWS * NORM_HIDDEN;
    let a = crate::oracle::signal(count, 31);
    let b = crate::oracle::signal(count, 32);
    let w = crate::oracle::signal(NORM_HIDDEN, 33);
    let d_a = gpu.upload(&f32_bytes(&a));
    let d_b = gpu.upload(&f32_bytes(&b));
    let d_w = gpu.upload(&f32_bytes(&w));
    let sum = zeros(gpu, count);
    let norm = zeros(gpu, count);

    let row = (NORM_HIDDEN * 4) as u64;
    let planes = row * NORM_ROWS as u64;
    let args = NormArgs {
        ne00: NORM_HIDDEN as i32,
        ne00_t: (NORM_HIDDEN / NORM_VEC) as i32,
        nb1: row,
        nb2: planes,
        nb3: planes,
        eps: NORM_EPS,
        nef1: [1, 1, 1],
        nef2: [1, 1, 1],
        nef3: [1, 1, 1],
        nbf1: [row, 0, 0],
        nbf2: [planes, 0, 0],
        nbf3: [planes, 0, 0],
    };
    launch(
        gpu,
        "kernel_dense_add_scale_rms_norm_mul_f32_4",
        Extent3::new(NORM_ROWS as u32, 1, 1),
        Extent3::new(NORM_BLOCK, 1, 1),
        NORM_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_a.view()),
            KernelArg::Buffer(d_b.view()),
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(sum.view()),
            KernelArg::Buffer(norm.view()),
            KernelArg::Bytes(bytemuck::bytes_of(&NORM_SCALE)),
        ],
    );
    drain(gpu);

    // The sum the next block adds into is the scaled residual, and the
    // normalization divides by the deviation of that same scaled row -- which
    // is what makes the scale visible in the stored sum and (very nearly)
    // invisible in the normalized one.
    let want_sum: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| (x + y) * NORM_SCALE)
        .collect();
    let mut want_norm = vec![0.0f32; count];
    for r in 0..NORM_ROWS {
        let row_values = &want_sum[r * NORM_HIDDEN..(r + 1) * NORM_HIDDEN];
        let mean: f32 = row_values.iter().map(|v| v * v).sum::<f32>() / (NORM_HIDDEN as f32);
        let inv = 1.0 / (mean + NORM_EPS).sqrt();
        for (i, &v) in row_values.iter().enumerate() {
            want_norm[r * NORM_HIDDEN + i] = v * inv * w[i];
        }
    }
    close(&gpu.read_f32(&sum), &want_sum, "sum");
    close(&gpu.read_f32(&norm), &want_norm, "norm");
}

// -- the softcap ------------------------------------------------------------

/// Elements of the softcap fixture: sixteen whole threadgroups.
const CAP_N: usize = 4096;
/// The cap Gemma 4 applies to its logits row.
const CAP: f32 = 30.0;

/// `x[i] = cap * tanh(x[i] / cap)`, in place, and every result inside the cap.
///
/// The fixture is scaled well past the cap on purpose: inside `[-1, 1]` the
/// formula is indistinguishable from the identity.
pub fn the_softcap_squashes_a_row_into_the_cap(gpu: &Gpu) {
    // Scaled well past the cap on purpose: a fixture inside [-1, 1] would make
    // `cap * tanh(x / cap)` indistinguishable from the identity.
    let x: Vec<f32> = crate::oracle::signal(CAP_N, 41)
        .iter()
        .map(|v| v * 100.0)
        .collect();
    let d_x = gpu.upload(&f32_bytes(&x));
    let args = SoftcapArgs {
        n: CAP_N as u32,
        cap: CAP,
    };
    launch(
        gpu,
        "kernel_dense_softcap_f32",
        Extent3::linear((CAP_N as u32).div_ceil(LINEAR_BLOCK)),
        Extent3::new(LINEAR_BLOCK, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_x.view()),
        ],
    );
    drain(gpu);

    let want: Vec<f32> = x.iter().map(|&v| CAP * (v / CAP).tanh()).collect();
    let got = gpu.read_f32(&d_x);
    close(&got, &want, "softcap");
    for (index, &v) in got.iter().enumerate() {
        assert!(
            v.abs() < CAP,
            "element {index} is {v}, which is not inside the cap {CAP}",
        );
    }
}

// -- the row scale ----------------------------------------------------------

/// Elements the row scale covers, deliberately not a multiple of the block: the
/// tail guard is the part of a flat kernel a round fixture cannot reach.
const SCALE_N: usize = 1000;
/// Allocated elements, so the run past `SCALE_N` is a sentinel the kernel must
/// leave alone.
const SCALE_ALLOC: usize = 1024;

/// `x[i] *= s` for `i < n`, and nothing past `n`.
///
/// The bound is deliberately not a multiple of the block, because the tail
/// guard is the part of a flat kernel a round fixture cannot reach. The
/// allocation runs past the bound and carries a sentinel there.
pub fn the_row_scale_multiplies_in_place_and_stops_at_its_bound(gpu: &Gpu) {
    let mut x = crate::oracle::signal(SCALE_ALLOC, 51);
    // A sentinel past the bound, distinct from anything the signal produces.
    for v in &mut x[SCALE_N..] {
        *v = 7.0;
    }
    let d_x = gpu.upload(&f32_bytes(&x));
    // Gemma 4's embedding scale: sqrt(hidden) at hidden 2560.
    let s = (2560.0f32).sqrt();
    let args = ScaleArgs {
        n: SCALE_N as u32,
        s,
    };
    launch(
        gpu,
        "kernel_dense_scale_f32",
        Extent3::linear((SCALE_N as u32).div_ceil(LINEAR_BLOCK)),
        Extent3::new(LINEAR_BLOCK, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_x.view()),
        ],
    );
    drain(gpu);

    let want: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| if i < SCALE_N { v * s } else { v })
        .collect();
    close(&gpu.read_f32(&d_x), &want, "scale");
}
