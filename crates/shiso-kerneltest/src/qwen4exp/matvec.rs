//! The sparse family's matvec kernels: cases graded against a host restatement of the CUDA bodies.
//!
//! The dense projections the sparse stack plans outside the trellis codec:
//! the brain-float and half-precision warp-per-row kernels, the fold that adds
//! a split contraction's partial slabs, the single-precision split mat-vec and
//! its fold, and the two width conversions the BLAS tiers stage through.
//!
//! The oracle is not a plain dot product. It restates each kernel's own order -
//! which lane owns which weights, one fused multiply-add per weight with the
//! activation rows inner, the decreasing-offset `shfl_down` fold, the block
//! fold's second level, the split-major partial slabs added in split order - in
//! `f32` with [`f32::mul_add`]. So a correct port agrees to rounding noise
//! rather than to a tolerance wide enough to hide a lane reading the wrong
//! column, and the tolerance below is a few ulps of the row's absolute sum.
//!
//! The geometry (grid, block, split, chunk) is transcribed from
//! `shiso_engine::sparse::kernels` and `sparse::matmul`, which this crate may
//! not depend on; each constant names the one it mirrors.
#![expect(
    clippy::cast_possible_truncation,
    reason = "kernel arguments are u32 while the fixtures index in usize; every \
              size here is a model dimension far below u32::MAX"
)]

use half::{bf16, f16};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The module every entry lives in, mirroring `kernels::MOD_MATVEC`.
const MODULE: &str = "q4e_matvec";
/// Threads in the warp-per-row and split kernels, mirroring `kernels::BLOCK_LINEAR`.
const BLOCK_LINEAR: u32 = 256;
/// Threads in the folds, mirroring `kernels::BLOCK_NARROW`.
const BLOCK_NARROW: u32 = 128;
/// Lanes in one simdgroup (warp).
const WARP: usize = 32;
/// Output rows one warp-per-row block covers, mirroring `kernels::ROWS_PER_WARP_BLOCK`.
const ROWS_PER_BLOCK: u32 = BLOCK_LINEAR / WARP as u32;

/// Activation row counts every real-shape case runs: one (decode), an odd
/// count in the middle, and the widest decode batch.
const REAL_ROWS: [u32; 3] = [1, 5, 16];

/// An element type a template argument names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Elem {
    /// `float`.
    F32,
    /// `__half`.
    F16,
}

impl Elem {
    /// The Itanium spelling of the type in a template argument list.
    const fn mangled(self) -> &'static str {
        match self {
            Self::F32 => "f",
            Self::F16 => "6__half",
        }
    }
}

/// A leaked entry name: `KernelRef` borrows its name for `'static`, and a case
/// builds the mangled name of each instantiation it runs.
fn entry(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

/// `q4e_matmul_bf16_rows_kernel<nt>`'s exported name.
fn bf16_rows_entry(nt: u32) -> &'static str {
    entry(format!("_Z27q4e_matmul_bf16_rows_kernelILi{nt}EEv"))
}

/// `q4e_matmul_f16_rows_kernel<nt, x, o>`'s exported name.
fn f16_rows_entry(nt: u32, x: Elem, o: Elem) -> &'static str {
    entry(format!(
        "_Z26q4e_matmul_f16_rows_kernelILi{nt}E{}{}Ev",
        x.mangled(),
        o.mangled()
    ))
}

/// `q4e_matvec_combine_t_kernel<o>`'s exported name.
fn combine_entry(o: Elem) -> &'static str {
    entry(format!("_Z27q4e_matvec_combine_t_kernelI{}Ev", o.mangled()))
}

/// `kernels::f16_split`, restated.
fn f16_split(in_dim: u32, out_dim: u32) -> u32 {
    let mut split = 1u32;
    while out_dim * split < 1_024 && in_dim / (split * 2) >= 256 {
        split *= 2;
    }
    split
}

/// `matmul::half_chunk`, restated: the per-split range rounded up to a whole
/// block width, so a boundary never cuts an eight-wide load.
fn half_chunk(in_dim: u32, splits: u32) -> u32 {
    in_dim.div_ceil(splits).next_multiple_of(BLOCK_LINEAR)
}

/// `kernels::f32_split`, restated.
fn f32_split(in_dim: u32, out_dim: u32) -> u32 {
    let mut split = 1u32;
    while out_dim * split * 2 <= 512 && in_dim / (split * 2) >= 512 {
        split *= 2;
    }
    split
}

/// `warp_sum_f32`: the decreasing-offset `shfl_down` fold, lane 0's total.
///
/// Only lanes below `off` feed lane 0 at each step, so the lanes whose
/// shuffle source is out of range (and which keep their own value) never
/// matter.
fn warp_fold(mut lanes: [f32; WARP]) -> f32 {
    let mut off = WARP / 2;
    while off > 0 {
        for l in 0..off {
            lanes[l] += lanes[l + off];
        }
        off /= 2;
    }
    lanes[0]
}

/// `q4e_block_sum` over `threads` per-thread partials: each warp folds, then
/// warp 0 folds the warp totals with zeros past the last warp.
fn block_fold(partials: &[f32]) -> f32 {
    let mut warps = [0.0f32; WARP];
    for (w, chunk) in partials.chunks(WARP).enumerate() {
        let mut lanes = [0.0f32; WARP];
        lanes[..chunk.len()].copy_from_slice(chunk);
        warps[w] = warp_fold(lanes);
    }
    warp_fold(warps)
}

/// A brain-float fixture as bit patterns and as the values they widen to.
fn bf16_fixture(len: usize, seed: u32) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = signal(len, seed)
        .into_iter()
        .map(|v| bf16::from_f32(v).to_bits())
        .collect();
    let values = bits.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
    (bits, values)
}

/// A half-precision fixture as bit patterns and as the values they widen to.
fn f16_fixture(len: usize, seed: u32) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = signal(len, seed)
        .into_iter()
        .map(|v| f16::from_f32(v).to_bits())
        .collect();
    let values = bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
    (bits, values)
}

/// `u16` bit patterns as little-endian bytes.
fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A device buffer of `len` elements of `elem`, every one a NaN, so an output
/// the kernel never writes fails the comparison rather than reading as zero.
fn sentinel(gpu: &Gpu, elem: Elem, len: usize) -> DeviceBuffer {
    match elem {
        Elem::F32 => gpu.upload(&f32_bytes(&vec![f32::NAN; len])),
        Elem::F16 => gpu.upload(&u16_bytes(&vec![f16::NAN.to_bits(); len])),
    }
}

/// Reads `buffer` back as `f32`, widening a half-precision one exactly.
fn read_as_f32(gpu: &Gpu, buffer: &DeviceBuffer, elem: Elem) -> Vec<f32> {
    match elem {
        Elem::F32 => gpu.read_f32(buffer),
        Elem::F16 => gpu
            .read(buffer)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
    }
}

/// One expected output: the value the restated kernel computes and the
/// absolute sum of the terms behind it, which scales the tolerance.
#[derive(Debug, Clone, Copy)]
struct Want {
    value: f32,
    scale: f32,
}

/// Asserts `got` matches `want` element by element, naming the first element
/// that does not.
///
/// The oracle restates the kernel's order, so the only slack is for the two
/// compilers contracting or rounding differently: a few ulps of the absolute
/// sum, plus half an ulp of a half-precision result.
fn agree(got: &[f32], want: &[Want], out_elem: Elem, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: element count");
    for (k, (&g, w)) in got.iter().zip(want).enumerate() {
        let rounding = match out_elem {
            Elem::F32 => 0.0,
            Elem::F16 => w.value.abs() * 2f32.powi(-10),
        };
        let tol = 4e-7 * w.scale.max(1e-3) + rounding;
        assert!(
            (g - w.value).abs() <= tol,
            "{what}: element {k} is {g}, expected {} (tolerance {tol})",
            w.value
        );
    }
}

/// Rounds a restated result the way the kernel's `OT` store does.
fn store(v: f32, elem: Elem) -> f32 {
    match elem {
        Elem::F32 => v,
        Elem::F16 => f16::from_f32(v).to_f32(),
    }
}

// --- brain-float rows -------------------------------------------------------

/// `q4e_matmul_bf16_rows_kernel<nt>` restated: `[nt][out_dim]`.
fn bf16_rows_oracle(w: &[f32], x: &[f32], nt: usize, in_dim: usize, out_dim: usize) -> Vec<Want> {
    let mut out = vec![
        Want {
            value: 0.0,
            scale: 0.0
        };
        nt * out_dim
    ];
    let pairs = in_dim / 2;
    for row in 0..out_dim {
        let wr = &w[row * in_dim..(row + 1) * in_dim];
        for t in 0..nt {
            let xt = &x[t * in_dim..(t + 1) * in_dim];
            let mut lanes = [0.0f32; WARP];
            let mut scale = 0.0f32;
            for (lane, acc) in lanes.iter_mut().enumerate() {
                for i in (lane..pairs).step_by(WARP) {
                    *acc = wr[2 * i].mul_add(xt[2 * i], *acc);
                    *acc = wr[2 * i + 1].mul_add(xt[2 * i + 1], *acc);
                    scale += (wr[2 * i] * xt[2 * i]).abs() + (wr[2 * i + 1] * xt[2 * i + 1]).abs();
                }
            }
            out[t * out_dim + row] = Want {
                value: warp_fold(lanes),
                scale,
            };
        }
    }
    out
}

/// Runs the brain-float rows kernel over one `[out_dim x in_dim]` weight at
/// `nt` activation rows, with the engine's `matmul_rows_warp` launch.
fn check_bf16_rows(gpu: &Gpu, out_dim: u32, in_dim: u32, nt: u32) {
    let (wd, od, n) = (in_dim as usize, out_dim as usize, nt as usize);
    let (w_bits, w) = bf16_fixture(od * wd, 0x0b16 ^ out_dim);
    let x = signal(n * wd, 0x5eed ^ in_dim ^ nt);
    let want = bf16_rows_oracle(&w, &x, n, wd, od);

    let w_dev = gpu.upload(&u16_bytes(&w_bits));
    let x_dev = gpu.upload(&f32_bytes(&x));
    let out = sentinel(gpu, Elem::F32, n * od);
    run(
        gpu,
        KernelRef::new(MODULE, bf16_rows_entry(nt)),
        &LaunchArgs {
            grid: Extent3::new(out_dim.div_ceil(ROWS_PER_BLOCK), 1, 1),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(w_dev.view()),
                KernelArg::Buffer(x_dev.view()),
                KernelArg::U32(in_dim),
                KernelArg::U32(out_dim),
            ],
        },
    );
    drain(gpu);
    agree(
        &gpu.read_f32(&out),
        &want,
        Elem::F32,
        &format!("bf16 rows<{nt}> [{out_dim} x {in_dim}]"),
    );
}

/// The brain-float projection at the shapes the model runs it at: the
/// 512-wide router, the 48-wide projection, the four-output block inject over
/// the 10240-wide stream set, and a single output row.
pub fn the_bf16_rows_projection_matches_at_the_real_shapes(gpu: &Gpu) {
    for (out_dim, in_dim) in [(512, 2_560), (48, 2_560), (4, 10_240), (1, 2_560)] {
        for nt in REAL_ROWS {
            check_bf16_rows(gpu, out_dim, in_dim, nt);
        }
    }
}

/// Every one of the sixteen brain-float instantiations resolves by its
/// exported name and computes its own row count.
pub fn every_bf16_rows_instantiation_matches(gpu: &Gpu) {
    for nt in 1..=16 {
        check_bf16_rows(gpu, 48, 2_560, nt);
    }
}

// --- half-precision rows and the fold ---------------------------------------

/// `q4e_matmul_f16_rows_kernel<nt, _, _>` restated, before the store: the
/// split-major partial slabs `[split][nt][out_dim]`.
fn f16_rows_partials(
    w: &[f32],
    x: &[f32],
    shape: RowsShape,
    splits: usize,
    chunk: usize,
) -> Vec<Want> {
    let RowsShape {
        nt,
        in_dim,
        out_dim,
    } = shape;
    let mut out = Vec::with_capacity(splits * nt * out_dim);
    for s in 0..splits {
        let i0 = s * chunk;
        let i1 = (i0 + chunk).min(in_dim);
        for t in 0..nt {
            let xt = &x[t * in_dim..(t + 1) * in_dim];
            for row in 0..out_dim {
                let wr = &w[row * in_dim..(row + 1) * in_dim];
                let mut lanes = [0.0f32; WARP];
                let mut scale = 0.0f32;
                for (lane, acc) in lanes.iter_mut().enumerate() {
                    let mut i = i0 + lane * 8;
                    while i < i1 {
                        for j in i..i + 8 {
                            *acc = wr[j].mul_add(xt[j], *acc);
                            scale += (wr[j] * xt[j]).abs();
                        }
                        i += BLOCK_LINEAR as usize;
                    }
                }
                out.push(Want {
                    value: warp_fold(lanes),
                    scale,
                });
            }
        }
    }
    out
}

/// `q4e_matvec_combine_t_kernel` restated: the slabs added in split order.
fn fold(partials: &[Want], width: usize, splits: usize, out_elem: Elem) -> Vec<Want> {
    (0..width)
        .map(|k| {
            let mut value = 0.0f32;
            let mut scale = 0.0f32;
            for s in 0..splits {
                value += partials[s * width + k].value;
                scale += partials[s * width + k].scale;
            }
            Want {
                value: store(value, out_elem),
                scale,
            }
        })
        .collect()
}

/// One half-precision rows launch's shape.
#[derive(Debug, Clone, Copy)]
struct RowsShape {
    /// Activation rows (the template's `NT`).
    nt: usize,
    /// Contraction width.
    in_dim: usize,
    /// Output width.
    out_dim: usize,
}

/// Runs the half-precision rows kernel `<nt, x_elem, out_elem>` over one
/// `[out_dim x in_dim]` weight with the engine's split and chunk, then - when
/// the contraction was split - the fold into `out_elem`, exactly as
/// `matmul::project_one` sequences them.
///
/// Returns the split the launch used, so a case can assert it covered the
/// split it meant to.
fn check_f16_rows(gpu: &Gpu, out_dim: u32, in_dim: u32, nt: u32, io: (Elem, Elem)) -> u32 {
    let (x_elem, out_elem) = io;
    let shape = RowsShape {
        nt: nt as usize,
        in_dim: in_dim as usize,
        out_dim: out_dim as usize,
    };
    let splits = f16_split(in_dim, out_dim);
    let chunk = half_chunk(in_dim, splits);
    // A split slab is `OT`-typed; the engine only splits into single-precision
    // slabs, and a half-precision destination is only reached unsplit.
    assert!(
        splits == 1 || out_elem == Elem::F32,
        "a half-precision output is only exercised unsplit"
    );

    let (w_bits, w) = f16_fixture(shape.out_dim * shape.in_dim, 0x0f16 ^ out_dim);
    let (x_bytes, x) = match x_elem {
        Elem::F32 => {
            let x = signal(shape.nt * shape.in_dim, 0x5eed ^ in_dim ^ nt);
            (f32_bytes(&x), x)
        }
        Elem::F16 => {
            let (bits, x) = f16_fixture(shape.nt * shape.in_dim, 0x5eed ^ in_dim ^ nt);
            (u16_bytes(&bits), x)
        }
    };
    let partials = f16_rows_partials(&w, &x, shape, splits as usize, chunk as usize);
    let width = shape.nt * shape.out_dim;
    let want = fold(&partials, width, splits as usize, out_elem);

    let w_dev = gpu.upload(&u16_bytes(&w_bits));
    let x_dev = gpu.upload(&x_bytes);
    let out = sentinel(gpu, out_elem, width);
    let partial = (splits > 1).then(|| sentinel(gpu, Elem::F32, splits as usize * width));
    let first = partial.as_ref().unwrap_or(&out);
    run(
        gpu,
        KernelRef::new(MODULE, f16_rows_entry(nt, x_elem, out_elem)),
        &LaunchArgs {
            grid: Extent3::new(out_dim.div_ceil(ROWS_PER_BLOCK), splits, 1),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(first.view()),
                KernelArg::Buffer(w_dev.view()),
                KernelArg::Buffer(x_dev.view()),
                KernelArg::U32(in_dim),
                KernelArg::U32(out_dim),
                KernelArg::U32(chunk),
            ],
        },
    );
    let what =
        format!("f16 rows<{nt}, {x_elem:?}, {out_elem:?}> [{out_dim} x {in_dim}] split {splits}");
    if let Some(partial) = &partial {
        // The slabs themselves are checked too: a fold can hide a slab written
        // to the wrong split only if two slabs are swapped, not if one is
        // missing, and this names the slab rather than the sum.
        drain(gpu);
        agree(
            &gpu.read_f32(partial),
            &partials,
            Elem::F32,
            &format!("{what} partial slabs"),
        );
        let width_arg = nt * out_dim;
        run(
            gpu,
            KernelRef::new(MODULE, combine_entry(out_elem)),
            &LaunchArgs {
                grid: Extent3::new(width_arg.div_ceil(BLOCK_NARROW), 1, 1),
                block: Extent3::new(BLOCK_NARROW, 1, 1),
                smem_bytes: 0,
                args: &[
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(partial.view()),
                    KernelArg::U32(width_arg),
                    KernelArg::U32(splits),
                ],
            },
        );
    }
    drain(gpu);
    agree(&read_as_f32(gpu, &out, out_elem), &want, out_elem, &what);
    splits
}

/// The half-precision projection at the shapes the model runs it at, through
/// the engine's own split: the 640-wide projection splits in two, the
/// four-output inject over 10240 inputs splits 32 ways (the last twelve
/// splits starting past the contraction and contributing zeros), and a
/// 1024-wide output is wide enough not to split.
///
/// # Panics
///
/// If an output or partial slab disagrees with the restated kernel, or a
/// shape no longer splits the way the case was written for.
pub fn the_f16_rows_projection_splits_and_folds_at_the_real_shapes(gpu: &Gpu) {
    for (out_dim, in_dim, expect_split) in [(640, 2_560, 2), (4, 10_240, 32), (1_024, 2_560, 1)] {
        for nt in REAL_ROWS {
            let splits = check_f16_rows(gpu, out_dim, in_dim, nt, (Elem::F32, Elem::F32));
            assert_eq!(splits, expect_split, "[{out_dim} x {in_dim}] split");
        }
    }
}

/// Every single-precision-in, single-precision-out instantiation (one to 32
/// rows) resolves and computes its own row count, at a 48-wide output that
/// splits eight ways.
pub fn every_f32_io_f16_rows_instantiation_matches(gpu: &Gpu) {
    for nt in 1..=32 {
        check_f16_rows(gpu, 48, 2_560, nt, (Elem::F32, Elem::F32));
    }
}

/// Every half-precision-input instantiation, split eight ways and folded.
pub fn every_half_input_f16_rows_instantiation_matches(gpu: &Gpu) {
    for nt in 1..=16 {
        check_f16_rows(gpu, 48, 2_560, nt, (Elem::F16, Elem::F32));
    }
}

/// Every half-precision-output instantiation, unsplit: a 504-wide contraction
/// is too short to split and ends partway through a lane's second pass.
///
/// # Panics
///
/// If an output disagrees with the restated kernel, or the shape splits.
pub fn every_half_output_f16_rows_instantiation_matches(gpu: &Gpu) {
    for nt in 1..=16 {
        let splits = check_f16_rows(gpu, 64, 504, nt, (Elem::F32, Elem::F16));
        assert_eq!(splits, 1, "the half-output case runs unsplit");
    }
}

/// The fold into a half-precision destination rounds the single-precision sum
/// once, after the last slab.
pub fn the_half_combine_rounds_the_folded_sum(gpu: &Gpu) {
    let (splits, width) = (8usize, 4 * 640usize);
    let partial = signal(splits * width, 0xc0b1);
    let slabs: Vec<Want> = partial
        .iter()
        .map(|&v| Want {
            value: v,
            scale: v.abs(),
        })
        .collect();
    let want = fold(&slabs, width, splits, Elem::F16);
    let partial_dev = gpu.upload(&f32_bytes(&partial));
    let out = sentinel(gpu, Elem::F16, width);
    run(
        gpu,
        KernelRef::new(MODULE, combine_entry(Elem::F16)),
        &LaunchArgs {
            grid: Extent3::new((width as u32).div_ceil(BLOCK_NARROW), 1, 1),
            block: Extent3::new(BLOCK_NARROW, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(partial_dev.view()),
                KernelArg::U32(width as u32),
                KernelArg::U32(splits as u32),
            ],
        },
    );
    drain(gpu);
    agree(
        &read_as_f32(gpu, &out, Elem::F16),
        &want,
        Elem::F16,
        "half combine",
    );
}

// --- single-precision split mat-vec -----------------------------------------

/// `q4e_matvec_f32_kernel` restated: `[split][out_dim]`.
fn matvec_f32_partials(
    w: &[f32],
    x: &[f32],
    out_dim: usize,
    splits: usize,
    chunk: usize,
) -> Vec<Want> {
    let in_dim = x.len();
    let threads = BLOCK_LINEAR as usize;
    let mut out = Vec::with_capacity(splits * out_dim);
    for s in 0..splits {
        let i0 = s * chunk;
        let i1 = (i0 + chunk).min(in_dim);
        for row in 0..out_dim {
            let wr = &w[row * in_dim..(row + 1) * in_dim];
            let mut per_thread = vec![0.0f32; threads];
            let mut scale = 0.0f32;
            for (tid, acc) in per_thread.iter_mut().enumerate() {
                let mut i = i0 + tid;
                while i < i1 {
                    *acc = wr[i].mul_add(x[i], *acc);
                    scale += (wr[i] * x[i]).abs();
                    i += threads;
                }
            }
            out.push(Want {
                value: block_fold(&per_thread),
                scale,
            });
        }
    }
    out
}

/// Runs the single-precision mat-vec over one `[out_dim x in_dim]` weight with
/// the engine's split, then its fold when split. Returns the split.
fn check_matvec_f32(gpu: &Gpu, out_dim: u32, in_dim: u32) -> u32 {
    let (od, wd) = (out_dim as usize, in_dim as usize);
    let splits = f32_split(in_dim, out_dim);
    let chunk = in_dim.div_ceil(splits);
    let w = signal(od * wd, 0xf32 ^ out_dim);
    let x = signal(wd, 0x5eed ^ in_dim);
    let partials = matvec_f32_partials(&w, &x, od, splits as usize, chunk as usize);
    let want = fold(&partials, od, splits as usize, Elem::F32);

    let w_dev = gpu.upload(&f32_bytes(&w));
    let x_dev = gpu.upload(&f32_bytes(&x));
    let out = sentinel(gpu, Elem::F32, od);
    let partial = (splits > 1).then(|| sentinel(gpu, Elem::F32, splits as usize * od));
    let first = partial.as_ref().unwrap_or(&out);
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_matvec_f32_kernel"),
        &LaunchArgs {
            grid: Extent3::new(out_dim, splits, 1),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(first.view()),
                KernelArg::Buffer(w_dev.view()),
                KernelArg::Buffer(x_dev.view()),
                KernelArg::U32(in_dim),
                KernelArg::U32(chunk),
            ],
        },
    );
    let what = format!("f32 matvec [{out_dim} x {in_dim}] split {splits}");
    if let Some(partial) = &partial {
        drain(gpu);
        agree(
            &gpu.read_f32(partial),
            &partials,
            Elem::F32,
            &format!("{what} partial slabs"),
        );
        run(
            gpu,
            KernelRef::new(MODULE, "q4e_matvec_f32_combine_kernel"),
            &LaunchArgs {
                grid: Extent3::new(out_dim.div_ceil(BLOCK_NARROW), 1, 1),
                block: Extent3::new(BLOCK_NARROW, 1, 1),
                smem_bytes: 0,
                args: &[
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(partial.view()),
                    KernelArg::U32(out_dim),
                    KernelArg::U32(splits),
                ],
            },
        );
    }
    drain(gpu);
    agree(&gpu.read_f32(&out), &want, Elem::F32, &what);
    splits
}

/// The single-precision mat-vec splits a 10240-wide contraction into sixteen
/// outputs sixteen ways and folds it, and runs the 512-wide router unsplit.
///
/// # Panics
///
/// If an output or partial slab disagrees with the restated kernel, or a
/// shape no longer splits the way the case was written for.
pub fn the_f32_matvec_splits_and_folds(gpu: &Gpu) {
    assert_eq!(check_matvec_f32(gpu, 16, 10_240), 16, "[16 x 10240] split");
    assert_eq!(check_matvec_f32(gpu, 512, 2_560), 1, "[512 x 2560] split");
}

// --- width conversions ------------------------------------------------------

/// The rounding to half precision is nearest-even and the widening back is
/// exact, over a length that ends partway through a block.
///
/// # Panics
///
/// If a rounded or widened element differs from the host conversion.
pub fn the_width_conversions_round_and_widen_exactly(gpu: &Gpu) {
    let n = 5 * 2_560 + 37;
    let blocks = (n as u32).div_ceil(BLOCK_LINEAR);
    // Scaled past half's range at both ends: overflow to infinity and
    // underflow into subnormals are part of the rounding under test.
    let wide: Vec<f32> = signal(n, 0xc0de)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v * [1.0, 7.0e4, 3.0e-6, 1.0e-8][i % 4])
        .collect();
    let want_half: Vec<u16> = wide.iter().map(|&v| f16::from_f32(v).to_bits()).collect();

    let wide_dev = gpu.upload(&f32_bytes(&wide));
    let half_dev = sentinel(gpu, Elem::F16, n);
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_f32_to_f16_kernel"),
        &LaunchArgs {
            grid: Extent3::new(blocks, 1, 1),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(half_dev.view()),
                KernelArg::Buffer(wide_dev.view()),
                KernelArg::U64(n as u64),
            ],
        },
    );
    drain(gpu);
    let got_half: Vec<u16> = gpu
        .read(&half_dev)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    for (i, (&g, &w)) in got_half.iter().zip(&want_half).enumerate() {
        assert_eq!(
            g, w,
            "f32 -> f16 element {i}: {} rounded to {g:#06x}, expected {w:#06x}",
            wide[i]
        );
    }

    let back_dev = sentinel(gpu, Elem::F32, n);
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_f16_to_f32_kernel"),
        &LaunchArgs {
            grid: Extent3::new(blocks, 1, 1),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(back_dev.view()),
                KernelArg::Buffer(half_dev.view()),
                KernelArg::U64(n as u64),
            ],
        },
    );
    drain(gpu);
    for (i, (&g, &h)) in gpu.read_f32(&back_dev).iter().zip(&want_half).enumerate() {
        let w = f16::from_bits(h).to_f32();
        assert!(
            g.to_bits() == w.to_bits(),
            "f16 -> f32 element {i}: {h:#06x} widened to {g}, expected {w}"
        );
    }
    assert_eq!(got_half.len(), n, "half buffer length");
}
