//! Which projection kernel a weight is read by, and with what geometry.
//!
//! One projection - `dst[out, tok] = sum_in weight[out, in] * src[in, tok]` -
//! has half a dozen implementations here, and the choice is made by the stored
//! element type, the two dimensions and the row count. Every other file in this
//! module asks this one for a plan rather than deciding for itself, because the
//! same weight is projected from six call sites and six copies of a tier ladder
//! would be six chances to disagree about which kernel a shape belongs to.
//!
//! # Why the plan is a value and not a launch
//!
//! Every function here returns a description. That is what makes the ladder
//! checkable on a machine with no device: "a 2560-wide half-precision weight
//! into four outputs at one row splits its contraction eight ways and then
//! combines" is an assertion about a returned value, not about a trace nobody
//! can produce without hardware.
//!
//! # Two tiers this build cannot reach, and why they are refusals
//!
//! The block-quantized tiers quantize the *activation* before they multiply,
//! and the kernel that does that belongs to a different kernel set than the one
//! vendored here. The dense fallback those tiers drop through to belongs to
//! that set as well. Both are therefore a typed refusal naming the missing
//! companion rather than a silently slower path - a model that quietly
//! dequantized one tensor would produce logits nobody can attribute to a
//! kernel.

use shiso_core::DType;
use shiso_core::backend::{GemmDType, GemmOp};
use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::bind::{Args, Operand, WeightRef};
use crate::sparse::exl3::{self, Fanout, GemmShape, Staging};
use crate::sparse::kernels::{
    self, Elem, Entry, Launch, MOD_EXL3, MOD_MATVEC, TemplateArgs, blocks_for,
};
use crate::sparse::plan::{Plan, Step, TrellisOp};
use crate::sparse::scratch::Buf;
use crate::sparse::weights::{WeightDesc, WeightPart};

/// Where one projection reads from and writes to.
///
/// A struct rather than three arguments because the two operands are the same
/// type and swapping them at a call site would compile and produce a launch
/// that reads its own destination.
///
/// Deliberately NOT `#[non_exhaustive]`: every block family builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bound {
    /// The weight this projection reads.
    pub weight: WeightRef,
    /// The activation it multiplies.
    pub src: Operand,
    /// Where the result goes.
    pub dst: Operand,
}

impl Bound {
    /// One projection's three operands.
    #[must_use]
    pub const fn new(weight: WeightRef, src: Operand, dst: Operand) -> Self {
        Self { weight, src, dst }
    }

    /// The same projection advanced to the rows a later pass covers.
    #[must_use]
    const fn advanced(self, src_bytes: u64, dst_bytes: u64) -> Self {
        Self {
            weight: self.weight,
            src: self.src.advanced(src_bytes),
            dst: self.dst.advanced(dst_bytes),
        }
    }
}

/// The widest row count the per-row projections were instantiated for.
///
/// Past this a chunk is either split into passes of this width or handed to the
/// BLAS tier, depending on how wide the output is.
pub const MAX_ROWS: u32 = 16;

/// Output width at or below which a chunk stays on the per-row kernels however
/// many rows it has.
///
/// The BLAS tier rounds the activation to half precision first, and for one
/// projection in the model that rounding is not harmless: the router's choice
/// of experts can flip on it when a token's tenth and eleventh logits are a
/// hair apart, and a flipped expert moves that token's residual by percent
/// rather than by units in the last place. Every projection this narrow is
/// cheap enough to run in passes, so the rule is stated on the width rather
/// than on the tensor.
pub const NARROW_OUTPUT: u32 = 1_024;

/// Elements one block-quantized weight block holds.
const QUANT_BLOCK: u32 = 32;

/// Where a projection's partial sums are folded back together, when it has any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Combine {
    /// The entry point that folds them.
    pub entry: Entry,
    /// Its launch.
    pub launch: Launch,
    /// Partial slabs to fold, which is also the split the projection ran at.
    pub splits: u32,
}

/// One planned projection.
///
/// The variants are the tiers, in the order the ladder considers them. Each
/// carries what a launch site needs and nothing else, so a site that took the
/// wrong arm does not compile rather than passing a chunk width to a kernel
/// that has no row argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Projection {
    /// A single-precision weight, with its contraction split.
    Float {
        /// The projection itself.
        entry: Entry,
        /// Its launch.
        launch: Launch,
        /// Elements of the contraction one split covers.
        chunk: u32,
        /// The fold, absent when the contraction was not split.
        combine: Option<Combine>,
    },
    /// A half-precision weight through the warp-per-row kernel.
    HalfRows {
        /// The projection itself.
        entry: Entry,
        /// Its launch.
        launch: Launch,
        /// Elements of the contraction one split covers.
        chunk: u32,
        /// The fold, absent when the contraction was not split.
        combine: Option<Combine>,
        /// Rows this instantiation covers, which is at most [`MAX_ROWS`].
        rows: u32,
    },
    /// A half-precision weight through the BLAS tier, one call for the chunk.
    HalfBlas {
        /// Rows the call covers.
        rows: u32,
    },
    /// A brain-float weight through the warp-per-row kernel.
    BrainRows {
        /// The projection itself.
        entry: Entry,
        /// Its launch.
        launch: Launch,
        /// Rows this instantiation covers.
        rows: u32,
    },
    /// A trellis-codec weight through the cooperative kernel.
    Trellis {
        /// Tiling shape the heuristic chose.
        shape: GemmShape,
        /// Trellis width.
        bits: u32,
        /// Blocks the cooperative launch starts, all of them resident.
        blocks: u32,
    },
    /// A trellis-codec weight expanded once and multiplied by the BLAS tier.
    TrellisExpanded {
        /// Trellis width.
        bits: u32,
        /// Rows the call covers.
        rows: u32,
    },
}

/// The device facts the ladder closes over.
///
/// Three of them, and they are a struct rather than three arguments because two
/// are `bool` and would be swappable at a call site without a compile error.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller has to build one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Device {
    /// Whether a BLAS tier is available.
    pub blas: bool,
    /// Whether the stream is mid-capture, which forbids growing any scratch.
    pub capturing: bool,
    /// Blocks that can be resident at once, which bounds a cooperative grid.
    pub max_co_resident: u32,
}

/// Plans one projection.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for a stored element type whose tier needs a
/// companion kernel this build does not carry, and for a trellis weight no
/// built tiling shape divides; [`CoreError::ShapeMismatch`] when a dimension is
/// not one the chosen kernel can stride over.
pub fn plan(
    dtype: DType,
    in_dim: u32,
    out_dim: u32,
    rows: u32,
    device: Device,
) -> CoreResult<Projection> {
    if rows == 0 || in_dim == 0 || out_dim == 0 {
        return Err(CoreError::ShapeMismatch {
            field: "projection extent",
            expected: 1,
            got: 0,
        });
    }
    match dtype {
        DType::F32 => Ok(float(in_dim, out_dim)),
        DType::F16 => half(in_dim, out_dim, rows, device),
        DType::BF16 => brain(in_dim, out_dim, rows),
        DType::Exl3K4 | DType::Exl3K5 | DType::Exl3K6 => {
            trellis(dtype, in_dim, out_dim, rows, device)
        }
        // Every block-quantized tier reaches its kernel through an activation
        // quantizer, and drops through to a dense path, that belong to the
        // general kernel set rather than to this one.
        _ => Err(CoreError::Unsupported {
            what: "a block-quantized weight, whose activation quantizer is not in this kernel set",
            backend: "engine",
        }),
    }
}

/// The single-precision tier.
///
/// The split exists for one shape in particular - a very wide contraction into
/// a handful of outputs, which the one-block-per-row kernel would run on a
/// handful of blocks - and it is inert everywhere else, because the loop stops
/// immediately once the output is wide.
#[must_use]
pub fn float(in_dim: u32, out_dim: u32) -> Projection {
    let splits = kernels::f32_split(in_dim, out_dim);
    let chunk = blocks_for(u64::from(in_dim), u64::from(splits));
    Projection::Float {
        entry: Entry::plain(MOD_MATVEC, "q4e_matvec_f32_kernel"),
        launch: kernels::matvec_f32(out_dim, splits),
        chunk,
        combine: (splits > 1).then(|| Combine {
            entry: Entry::plain(MOD_MATVEC, "q4e_matvec_f32_combine_kernel"),
            launch: kernels::matvec_f32_combine(out_dim),
            splits,
        }),
    }
}

/// Elements of the contraction one half-precision split covers.
///
/// Rounded up to a whole multiple of the block width so that a split boundary
/// never falls inside one of the sixteen-byte loads the kernel issues.
#[must_use]
fn half_chunk(in_dim: u32, splits: u32) -> u32 {
    let per = blocks_for(u64::from(in_dim), u64::from(splits));
    per.next_multiple_of(kernels::BLOCK_LINEAR)
}

/// The half-precision tier.
fn half(in_dim: u32, out_dim: u32, rows: u32, device: Device) -> CoreResult<Projection> {
    // Eight halves per lane: the kernel loads sixteen bytes of weight and a
    // four-float activation at a time, so a contraction that is not a whole
    // number of eight would leave a tail no lane reads.
    if !in_dim.is_multiple_of(8) {
        return Err(CoreError::ShapeMismatch {
            field: "half-precision contraction width",
            expected: 8,
            got: u64::from(in_dim),
        });
    }
    if rows > MAX_ROWS && out_dim > NARROW_OUTPUT {
        if !device.blas {
            return Err(CoreError::Unsupported {
                what: "a wide half-precision chunk on a device with no BLAS tier",
                backend: "engine",
            });
        }
        return Ok(Projection::HalfBlas { rows });
    }
    // A chunk past the instantiated width runs in passes of that width; the
    // plan describes one pass, because every pass is the same launch.
    let pass = rows.min(MAX_ROWS);
    let splits = kernels::f16_split(in_dim, out_dim);
    let chunk = half_chunk(in_dim, splits);
    // The instantiation is the same either way: a split projection writes
    // single-precision partial slabs and an unsplit one writes the destination,
    // and the destination of every projection reached from here is
    // single-precision too. The fold below is where a narrower destination
    // would be produced, which is why the element type rides on it rather than
    // on this.
    Ok(Projection::HalfRows {
        entry: Entry::template(
            MOD_MATVEC,
            "q4e_matmul_f16_rows_kernel",
            TemplateArgs::RowsIo {
                rows: pass,
                input: Elem::F32,
                output: Elem::F32,
            },
        ),
        launch: kernels::matmul_f16_rows(out_dim, splits),
        chunk,
        combine: (splits > 1).then(|| Combine {
            entry: Entry::template(
                MOD_MATVEC,
                "q4e_matvec_combine_t_kernel",
                TemplateArgs::Elem(Elem::F32),
            ),
            launch: kernels::matvec_combine_rows(out_dim, pass),
            splits,
        }),
        rows: pass,
    })
}

/// The brain-float tier, which has no split and no single-row instantiation.
fn brain(in_dim: u32, out_dim: u32, rows: u32) -> CoreResult<Projection> {
    if !(2..=MAX_ROWS).contains(&rows) {
        return Err(CoreError::Unsupported {
            what: "a brain-float projection at a row count this kernel set was not built for",
            backend: "engine",
        });
    }
    if !in_dim.is_multiple_of(2) {
        return Err(CoreError::ShapeMismatch {
            field: "brain-float contraction width",
            expected: 2,
            got: u64::from(in_dim),
        });
    }
    Ok(Projection::BrainRows {
        entry: Entry::template(
            MOD_MATVEC,
            "q4e_matmul_bf16_rows_kernel",
            TemplateArgs::Rows(rows),
        ),
        launch: kernels::matmul_rows_warp(out_dim),
        rows,
    })
}

/// Trellis width of one of the three codec element types.
#[must_use]
pub const fn trellis_bits(dtype: DType) -> Option<u32> {
    match dtype {
        DType::Exl3K4 => Some(4),
        DType::Exl3K5 => Some(5),
        DType::Exl3K6 => Some(6),
        _ => None,
    }
}

/// The trellis tier.
fn trellis(
    dtype: DType,
    in_dim: u32,
    out_dim: u32,
    rows: u32,
    device: Device,
) -> CoreResult<Projection> {
    let bits = trellis_bits(dtype).ok_or(CoreError::Unsupported {
        what: "a trellis element type with no width",
        backend: "engine",
    })?;
    exl3::check_shape(in_dim, out_dim, bits)?;
    let gate = exl3::ReconstructGate {
        rows,
        blas: device.blas,
        capturing: device.capturing,
    };
    if exl3::reconstruct(gate, in_dim, out_dim) {
        return Ok(Projection::TrellisExpanded { bits, rows });
    }
    let shape = exl3::select_shape(in_dim, out_dim, bits, Fanout::Single).ok_or(
        CoreError::Unsupported {
            what: "a trellis weight no built tiling shape divides",
            backend: "engine",
        },
    )?;
    Ok(Projection::Trellis {
        shape,
        bits,
        blocks: exl3::gemm_blocks(in_dim, out_dim, shape, device.max_co_resident),
    })
}

/// Whether a pair of weights of identical shape and type can share one grouped
/// launch.
///
/// Only the trellis tier has one; everything else is two independent
/// projections, which is why this answers a question rather than returning a
/// plan.
#[must_use]
pub fn pairable(dtype: DType, other: DType, same_shape: bool) -> bool {
    same_shape && dtype == other && trellis_bits(dtype).is_some()
}

/// Blocks one row of a block-quantized weight holds.
///
/// Stated here because the two narrow tiers pick their block width from it, and
/// because a width that is not a whole number of blocks is a shape those
/// kernels cannot stride over at all.
#[must_use]
pub const fn quant_blocks_per_row(in_dim: u32) -> Option<u64> {
    if in_dim == 0 || !in_dim.is_multiple_of(QUANT_BLOCK) {
        return None;
    }
    Some((in_dim / QUANT_BLOCK) as u64)
}

/// Whether the single-row narrow block-quantized kernel wants this shape.
///
/// It exists for the wide-in, narrow-out projections - the stream mixes' down
/// projections and the attention key and value heads - which the general
/// warp-per-row grid runs on too few blocks. It declines any shape it does not
/// improve, which is what the two bounds say.
#[must_use]
pub fn narrow_quant_wanted(in_dim: u32, out_dim: u32) -> bool {
    quant_blocks_per_row(in_dim).is_some()
        && in_dim >= 1_024
        && out_dim != 0
        && out_dim <= NARROW_OUTPUT
}

/// Which of the two multi-row block-quantized kernels a shape takes.
///
/// The narrow one gives a whole block to each output row, which pays only when
/// the rows are few enough to fill the machine that way and long enough to keep
/// a block busy.
#[must_use]
pub fn narrow_quant_rows_wanted(out_dim: u32, blocks_per_row: u64) -> bool {
    out_dim <= NARROW_OUTPUT && blocks_per_row >= 32
}

/// Plans one projection into `out`, in launch order.
///
/// A chunk wider than the per-row kernels were instantiated for, on an output
/// narrow enough to stay off the BLAS tier, becomes several passes - and each
/// one is its own step, because each one is its own launch and the last is
/// usually narrower than the rest. Each pass reads and writes the rows it
/// covers, which is what [`Bound::advanced`] is.
///
/// # Errors
///
/// Whatever [`plan`] reports for a weight this build cannot project.
pub fn project(
    out: &mut Plan,
    weight: WeightDesc,
    bound: Bound,
    rows: u32,
    device: Device,
) -> CoreResult<()> {
    if weight.dtype == DType::F16 && rows > MAX_ROWS && weight.out_dim <= NARROW_OUTPUT {
        let mut done = 0;
        while done < rows {
            let take = (rows - done).min(MAX_ROWS);
            let src_bytes = u64::from(done) * u64::from(weight.in_dim) * 4;
            let dst_bytes = u64::from(done) * u64::from(weight.out_dim) * 4;
            project_one(
                out,
                weight,
                bound.advanced(src_bytes, dst_bytes),
                take,
                device,
            )?;
            done = done.saturating_add(take);
        }
        return Ok(());
    }
    project_one(out, weight, bound, rows, device)
}

/// Plans one pass of one projection.
#[expect(
    clippy::too_many_lines,
    reason = "one statement per kernel argument is the point: these bodies are \
              transcriptions of parameter lists, and a split that broke a list \
              across two functions would hide the order being transcribed"
)]
fn project_one(
    out: &mut Plan,
    weight: WeightDesc,
    bound: Bound,
    rows: u32,
    device: Device,
) -> CoreResult<()> {
    let w = Operand::Weight(bound.weight);
    match plan(weight.dtype, weight.in_dim, weight.out_dim, rows, device)? {
        // `out, w, x, in_dim, chunk` - and where the contraction was split the
        // projection writes the shared partial slab instead of the destination,
        // split-major, for the fold below to add up.
        Projection::Float {
            entry,
            launch,
            chunk,
            combine,
        } => {
            let dst = combine.map_or(bound.dst, |_| Operand::Scratch(Buf::ProjectionPartial));
            out.launch(
                entry,
                launch,
                Args::of(&[
                    dst,
                    w,
                    bound.src,
                    Operand::U32(weight.in_dim),
                    Operand::U32(chunk),
                ]),
            );
            if let Some(fold) = combine {
                // `out, partial, out_dim, n_split`.
                out.launch(
                    fold.entry,
                    fold.launch,
                    Args::of(&[
                        bound.dst,
                        Operand::Scratch(Buf::ProjectionPartial),
                        Operand::U32(weight.out_dim),
                        Operand::U32(fold.splits),
                    ]),
                );
            }
        }
        // `out, w, x, in_dim, out_dim, chunk`. The fold's width argument is the
        // whole partial set - this instantiation's row count times the output
        // width - and not one row of it.
        Projection::HalfRows {
            entry,
            launch,
            chunk,
            combine,
            rows: pass,
        } => {
            let dst = combine.map_or(bound.dst, |_| Operand::Scratch(Buf::ProjectionPartial));
            out.launch(
                entry,
                launch,
                Args::of(&[
                    dst,
                    w,
                    bound.src,
                    Operand::U32(weight.in_dim),
                    Operand::U32(weight.out_dim),
                    Operand::U32(chunk),
                ]),
            );
            if let Some(fold) = combine {
                out.launch(
                    fold.entry,
                    fold.launch,
                    Args::of(&[
                        bound.dst,
                        Operand::Scratch(Buf::ProjectionPartial),
                        Operand::U32(pass.saturating_mul(weight.out_dim)),
                        Operand::U32(fold.splits),
                    ]),
                );
            }
        }
        // `out, w, x, in_dim, out_dim`.
        Projection::BrainRows { entry, launch, .. } => out.launch(
            entry,
            launch,
            Args::of(&[
                bound.dst,
                w,
                bound.src,
                Operand::U32(weight.in_dim),
                Operand::U32(weight.out_dim),
            ]),
        ),
        // The weight is stored one output row at a time and the tier is
        // column-major, so the multiply reads it transposed and its leading
        // dimension counts the contraction. The activation is rounded to half
        // precision first, by its own launch, because the tier has no path
        // that reads a single-precision right operand against a
        // half-precision left one.
        Projection::HalfBlas { rows } => {
            convert_activation(out, bound.src, weight.in_dim, rows);
            out.push(Step::Blas {
                op_a: GemmOp::Transposed,
                op_b: GemmOp::AsStored,
                m: weight.out_dim,
                n: rows,
                k: weight.in_dim,
                lda: weight.in_dim,
                ldb: weight.in_dim,
                ldc: weight.out_dim,
                a: w,
                b: Operand::Scratch(Buf::HalfActivation),
                c: bound.dst,
                dtype: GemmDType::F16,
            });
        }
        Projection::Trellis {
            shape,
            bits,
            blocks,
        } => out.push(Step::Trellis {
            op: TrellisOp::Dense,
            entry: trellis_entry(kernels::EXL3_GEMM, bits, shape),
            shape,
            bits,
            blocks,
            // One group: the dense kernel reads neither `blockIdx.z` nor a
            // group width.
            groups: 1,
            block_dim: shape.block_dim(),
            smem: kernels::EXL3_SMEM_MAX,
            staging: Staging::hadamard(1, rows, weight.in_dim),
            // `A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh`
            args: Args::of(&[
                bound.src,
                Operand::Weight(part(bound.weight, WeightPart::Tiles)),
                bound.dst,
                Operand::U32(rows),
                Operand::U32(weight.in_dim),
                Operand::U32(weight.out_dim),
                Operand::CoopLocks,
                Operand::Weight(part(bound.weight, WeightPart::InputScales)),
                Operand::CoopStaging(0),
                Operand::Weight(part(bound.weight, WeightPart::OutputScales)),
            ]),
        }),
        // The expansion writes the whole weight into the staging region, the
        // conversion rounds the activation, and the multiply reads both as
        // stored: an expanded weight comes out output-major, which is the
        // layout the tier wants without a transposition.
        Projection::TrellisExpanded { bits, rows } => {
            out.push(Step::Expand {
                entry: Entry::template(
                    MOD_EXL3,
                    kernels::EXL3_RECONSTRUCT,
                    TemplateArgs::TrellisBits(bits),
                ),
                k: weight.in_dim,
                n: weight.out_dim,
                bits,
                launch: kernels::trellis_expand(weight.in_dim, weight.out_dim),
                staging: Staging::expanded(weight.in_dim, weight.out_dim),
                // `g_unpacked, g_packed, suh, svh, packed_blocks_n,
                //  packed_n_offset`
                args: Args::of(&[
                    Operand::CoopStaging(0),
                    Operand::Weight(part(bound.weight, WeightPart::Tiles)),
                    Operand::Weight(part(bound.weight, WeightPart::InputScales)),
                    Operand::Weight(part(bound.weight, WeightPart::OutputScales)),
                    Operand::I32(i32::try_from(weight.out_dim / 16).unwrap_or(i32::MAX)),
                    Operand::I32(0),
                ]),
            });
            convert_activation(out, bound.src, weight.in_dim, rows);
            out.push(Step::Blas {
                op_a: GemmOp::AsStored,
                op_b: GemmOp::AsStored,
                m: weight.out_dim,
                n: rows,
                k: weight.in_dim,
                lda: weight.out_dim,
                ldb: weight.in_dim,
                ldc: weight.out_dim,
                a: Operand::CoopStaging(0),
                b: Operand::Scratch(Buf::HalfActivation),
                c: bound.dst,
                dtype: GemmDType::F16,
            });
        }
    }
    Ok(())
}

/// The same tensor reference, narrowed to one span of a trellis payload.
const fn part(reference: WeightRef, span: WeightPart) -> WeightRef {
    reference.spanning(span)
}

/// One instantiation of a cooperative trellis projection.
pub(super) fn trellis_entry(template: &'static str, bits: u32, shape: GemmShape) -> Entry {
    Entry::template(
        MOD_EXL3,
        template,
        TemplateArgs::Trellis {
            bits,
            shape: shape.get(),
        },
    )
}

/// The rounding pass that precedes every multiply on the BLAS tier.
///
/// `out, in, n`. It is a step of the plan rather than something the tier does
/// for itself because it is a launch: it has a grid, it writes a buffer the
/// multiply then reads, and a pass that skipped it would hand the tier whatever
/// the previous chunk left in that buffer.
fn convert_activation(out: &mut Plan, src: Operand, in_dim: u32, rows: u32) {
    let elements = u64::from(rows) * u64::from(in_dim);
    out.launch(
        Entry::plain(MOD_MATVEC, "q4e_f32_to_f16_kernel"),
        kernels::elementwise(elements),
        Args::of(&[
            Operand::Scratch(Buf::HalfActivation),
            src,
            Operand::U64(elements),
        ]),
    );
}

/// Plans two projections of the same input.
///
/// A matched trellis pair shares one grouped launch; anything else is two
/// independent projections, in the order the gate precedes its up.
///
/// # Errors
///
/// Whatever [`project`] reports for either weight.
pub fn project_pair(
    out: &mut Plan,
    gate: WeightDesc,
    up: WeightDesc,
    gate_bound: Bound,
    up_bound: Bound,
    rows: u32,
    device: Device,
) -> CoreResult<()> {
    // Past the threshold the pair splits: each weight is expanded once and
    // multiplied, which is cheaper than streaming both trellises once per
    // sixteen rows, and there is no grouped form of the expanded path. Below
    // it, one grouped launch runs both over half a device each instead of two
    // cooperative launches in sequence.
    if gate.matches(&up)
        && rows <= kernels::EXL3_RECONSTRUCT_ROWS
        && let Some(bits) = trellis_bits(gate.dtype)
    {
        exl3::check_shape(gate.in_dim, gate.out_dim, bits)?;
        let fanout = Fanout::Grouped {
            slots: 2,
            input_per_slot: false,
        };
        if let Some(shape) = exl3::select_shape(gate.in_dim, gate.out_dim, bits, fanout) {
            let grid =
                exl3::mgemm_grid(gate.in_dim, gate.out_dim, shape, 2, device.max_co_resident);
            out.push(Step::Trellis {
                op: TrellisOp::Pair,
                entry: trellis_entry(kernels::EXL3_MGEMM, bits, shape),
                shape,
                bits,
                blocks: grid.per_group,
                groups: grid.concurrency,
                block_dim: shape.block_dim(),
                smem: kernels::EXL3_SMEM_MAX,
                // Two slots, each with its own transformed slab: the pair is
                // the expert fan-out with one expert per tensor, and sizing
                // this for one slot is a write past the end rather than a
                // failed launch.
                staging: Staging::hadamard(2, rows, gate.in_dim),
                // `A, B_base, B_stride, C, size_m, size_k, size_n, locks,
                //  suh_base, A_had, svh_base, B_indices, bszm_in, bszm_out,
                //  B_base2, suh_base2, svh_base2, C2, split`
                //
                // Both slots read the same input rows, so the input blocking
                // is one and the output blocking is two; the slot index list
                // is null because a dense pair has one matrix per tensor and
                // nothing to select.
                args: Args::of(&[
                    gate_bound.src,
                    Operand::Weight(part(gate_bound.weight, WeightPart::Tiles)),
                    Operand::U64(exl3::expert_tile_stride(gate.in_dim, gate.out_dim, bits)),
                    gate_bound.dst,
                    Operand::U32(rows),
                    Operand::U32(gate.in_dim),
                    Operand::U32(gate.out_dim),
                    Operand::CoopLocks,
                    Operand::Weight(part(gate_bound.weight, WeightPart::InputScales)),
                    Operand::CoopStaging(0),
                    Operand::Weight(part(gate_bound.weight, WeightPart::OutputScales)),
                    Operand::Null,
                    Operand::I32(1),
                    Operand::I32(2),
                    Operand::Weight(part(up_bound.weight, WeightPart::Tiles)),
                    Operand::Weight(part(up_bound.weight, WeightPart::InputScales)),
                    Operand::Weight(part(up_bound.weight, WeightPart::OutputScales)),
                    up_bound.dst,
                    Operand::I32(1),
                ]),
            });
            return Ok(());
        }
    }
    project(out, gate, gate_bound, rows, device)?;
    project(out, up, up_bound, rows, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GPU: Device = Device {
        blas: true,
        capturing: false,
        max_co_resident: 48,
    };

    #[test]
    fn a_wide_contraction_into_four_outputs_splits_and_then_combines() {
        // The stream mixes' injection logits: ten thousand inputs into four
        // outputs, which one block per row would run on four blocks.
        let plan = float(10_240, 4);
        let Projection::Float {
            launch,
            chunk,
            combine,
            ..
        } = plan
        else {
            panic!("a single-precision weight is a single-precision projection: {plan:?}");
        };
        assert_eq!(launch.grid.x, 4, "one block per output row");
        assert_eq!(launch.grid.y, 16, "sixteen splits of the contraction");
        assert_eq!(chunk, 640, "each split covers its share of ten thousand");
        let combine = combine.expect("a split contraction has to be folded");
        assert_eq!(combine.splits, 16);
    }

    #[test]
    fn a_square_single_precision_projection_does_not_split_and_has_no_fold() {
        let plan = float(2_560, 2_560);
        let Projection::Float {
            launch, combine, ..
        } = plan
        else {
            panic!("{plan:?}");
        };
        assert_eq!(launch.grid.y, 1);
        assert!(
            combine.is_none(),
            "an unsplit projection writes the destination itself"
        );
    }

    #[test]
    fn a_half_precision_chunk_wider_than_the_instantiated_set_runs_in_passes_of_it() {
        // A narrow output stays on the per-row kernel however wide the chunk
        // is, because the BLAS tier's rounding is what flips a routing tie.
        let plan = half(2_560, 512, 512, GPU).expect("the router is half precision");
        let Projection::HalfRows { rows, .. } = plan else {
            panic!("a narrow output never reaches the BLAS tier: {plan:?}");
        };
        assert_eq!(rows, MAX_ROWS, "one pass is the instantiated width");
    }

    #[test]
    fn a_wide_half_precision_chunk_goes_to_the_blas_tier_and_refuses_without_one() {
        let plan = half(2_560, 2_560, 512, GPU).expect("a wide chunk has a tier");
        assert_eq!(plan, Projection::HalfBlas { rows: 512 });
        let err = half(2_560, 2_560, 512, Device { blas: false, ..GPU })
            .expect_err("there is no third path for a wide chunk");
        assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
    }

    #[test]
    fn a_half_precision_split_rounds_its_share_up_to_a_whole_block() {
        // 2560 over eight splits is 320, which is not a multiple of the block
        // width; a split boundary inside one of the kernel's loads would have
        // two blocks read the same elements.
        let plan = half(2_560, 4, 1, GPU).expect("a narrow output is per-row");
        let Projection::HalfRows { chunk, .. } = plan else {
            panic!("{plan:?}");
        };
        assert_eq!(chunk, 512);
        assert!(chunk.is_multiple_of(kernels::BLOCK_LINEAR));
    }

    #[test]
    fn a_brain_float_projection_has_no_single_row_instantiation() {
        let err = brain(2_560, 128, 1).expect_err("one row is not in the built set");
        assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
        let plan = brain(2_560, 128, 4).expect("four rows is");
        let Projection::BrainRows { entry, rows, .. } = plan else {
            panic!("{plan:?}");
        };
        assert_eq!(rows, 4);
        assert_eq!(entry.name(), "q4e_matmul_bf16_rows_kernel");
    }

    #[test]
    fn a_trellis_projection_streams_at_decode_and_is_expanded_at_prefill() {
        let decode = plan(DType::Exl3K4, 2_560, 2_560, 1, GPU).expect("a decode row");
        assert!(
            matches!(decode, Projection::Trellis { bits: 4, .. }),
            "{decode:?}"
        );
        let prefill = plan(DType::Exl3K4, 2_560, 2_560, 512, GPU).expect("a prefill chunk");
        assert_eq!(prefill, Projection::TrellisExpanded { bits: 4, rows: 512 });
        // Mid-capture the scratch the expansion needs cannot be grown, so the
        // same chunk streams instead.
        let captured = plan(
            DType::Exl3K4,
            2_560,
            2_560,
            512,
            Device {
                capturing: true,
                ..GPU
            },
        )
        .expect("a captured chunk still has a path");
        assert!(
            matches!(captured, Projection::Trellis { .. }),
            "{captured:?}"
        );
    }

    #[test]
    fn a_block_quantized_weight_names_the_companion_this_build_does_not_carry() {
        for dtype in [DType::Q8_0, DType::Q4_K, DType::Q6_K, DType::Q5_0] {
            let err = plan(dtype, 2_560, 2_560, 1, GPU)
                .expect_err("no activation quantizer is vendored here");
            assert!(
                matches!(err, CoreError::Unsupported { what, .. } if what.contains("quantizer")),
                "{dtype:?}: {err:?}"
            );
        }
    }

    #[test]
    fn only_a_matched_trellis_pair_shares_one_launch() {
        assert!(pairable(DType::Exl3K4, DType::Exl3K4, true));
        assert!(!pairable(DType::Exl3K4, DType::Exl3K4, false));
        assert!(!pairable(DType::Exl3K4, DType::Exl3K5, true));
        assert!(!pairable(DType::F16, DType::F16, true));
    }

    #[test]
    fn the_narrow_block_quantized_entries_decline_the_shapes_they_do_not_improve() {
        assert!(narrow_quant_wanted(2_560, 512));
        // A short contraction: the general grid already fills the machine.
        assert!(!narrow_quant_wanted(512, 512));
        // A wide output: likewise.
        assert!(!narrow_quant_wanted(2_560, 2_560));
        // A width that is not a whole number of blocks is not a shape either
        // kernel can stride over.
        assert!(!narrow_quant_wanted(2_570, 512));
        assert_eq!(quant_blocks_per_row(2_560), Some(80));
        assert_eq!(quant_blocks_per_row(2_570), None);
        assert!(narrow_quant_rows_wanted(512, 80));
        assert!(!narrow_quant_rows_wanted(2_560, 80));
        assert!(!narrow_quant_rows_wanted(512, 16));
    }

    #[test]
    fn a_zero_extent_is_a_shape_error_rather_than_a_launch_of_nothing() {
        let err = plan(DType::F32, 0, 16, 1, GPU).expect_err("a zero contraction");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }), "{err:?}");
        let err = plan(DType::F32, 16, 16, 0, GPU).expect_err("a zero row count");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }), "{err:?}");
    }
}
