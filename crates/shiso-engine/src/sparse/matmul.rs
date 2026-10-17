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
use shiso_core::backend::{GemmDType, GemmOp, GemmReduction};
use shiso_core::error::{CoreError, CoreResult};

use crate::sparse::bind::{Args, Operand, WeightRef};
use crate::sparse::exl3::{self, Fanout, GemmShape, Staging};
use crate::sparse::kernels::{
    self, Elem, Entry, Launch, MOD_BF16, MOD_EXL3, MOD_MATVEC, MOD_MXFP8, MOD_MXFP8_A8,
    TemplateArgs, blocks_for,
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

/// The widest row count the half-precision per-row projection was
/// instantiated for, which every pass up to it takes.
///
/// Wider than [`MAX_ROWS`] because a batched verify is wider than sixteen rows
/// (four sessions at draft depth four are twenty), and past the per-row kernel
/// the stream mixes round their activation to halves on the BLAS tier and the
/// narrow projections take the strict multiply: either way a batched row's
/// arithmetic was not its own verify pass's, and identical requests in one batch
/// answered differently (measured at depth four). Every pass up to this width
/// now sums a row the one way.
pub const HALF_MAX_ROWS: u32 = 32;

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

/// Contraction width from which a narrow half-precision chunk goes to the BLAS
/// tier anyway.
///
/// The per-row kernels re-read the whole weight for every sixteen rows, which
/// for the stream mixes - the full multi-stream residual (10,240 wide) into the
/// low-rank width or the stream count - is a 6.5 MB weight read 128 times per
/// 2048-row chunk, ninety-seven times a layer stack. Measured on a GB10 that
/// was 2.5 s of a 7.5k-token prefill's 10.6 s. Those mixes round their
/// activation to half precision on the BLAS tier, as the reference does; the
/// projections [`NARROW_OUTPUT`] exists for - the router above all - contract
/// over the residual width and stay on the exact path.
pub const WIDE_CONTRACTION: u32 = 8_192;

/// Whether a half-precision chunk of `rows` must not have its activation
/// rounded, which keeps it off [`Projection::HalfBlas`].
#[must_use]
pub const fn half_exact(in_dim: u32, out_dim: u32, rows: u32) -> bool {
    rows > HALF_MAX_ROWS && out_dim <= NARROW_OUTPUT && in_dim < WIDE_CONTRACTION
}

/// Whether the strict single-precision multiply is available for a chunk.
///
/// It widens the weight into the per-stream staging, which cannot grow while
/// a stream is capturing; a capture never carries a prefill chunk, so the
/// per-row passes are what a captured chunk falls back to.
const fn exact_blas(device: Device) -> bool {
    device.blas && !device.capturing
}

/// Whether a half-precision chunk of `rows` runs as sixteen-row passes of the
/// per-row kernel rather than one BLAS call.
///
/// Measured on a GB10 before the strict tier existed, those passes - the
/// router, the recurrent gates and the shared router, sixteen rows at a time
/// with a combine behind each - were 1.2 s of a 7.5k-token prefill's 6.8 s.
#[must_use]
pub const fn half_in_passes(in_dim: u32, out_dim: u32, rows: u32, device: Device) -> bool {
    half_exact(in_dim, out_dim, device.tier_rows(rows)) && !exact_blas(device)
}

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
    /// A half-precision weight widened to single precision once and multiplied
    /// in strict single precision, one call for the chunk.
    ///
    /// The narrow projections take this rather than [`Projection::HalfBlas`]:
    /// the activation is never rounded, so the router reads the logits the
    /// per-row kernel would have produced up to the order of the sums.
    HalfExact {
        /// Rows the call covers.
        rows: u32,
    },
    /// A packed half-precision weight ([`DType::F16Pack12`]) expanded once
    /// into the per-stream staging - to the halves it was, or to the floats
    /// they are - and multiplied by the BLAS tier exactly as
    /// [`Projection::HalfBlas`] or [`Projection::HalfExact`] multiplies the
    /// stored halves.
    PackedExpanded {
        /// What the expansion writes: halves for the half-precision tier,
        /// floats for the strict one.
        wide: Elem,
        /// Rows the call covers.
        rows: u32,
    },
    /// A brain-float weight on the matrix units, a prompt chunk in one
    /// launch.
    BrainGemm {
        /// The projection itself.
        entry: Entry,
        /// Its launch.
        launch: Launch,
    },
    /// A brain-float weight through the step kernels of [`MOD_BF16`]: one
    /// launch over up to [`BF16_STEP_ROWS`] rows, each output row's
    /// contraction sliced across the warps of one block by the shape alone
    /// and folded inside it.
    BrainStep {
        /// The projection itself.
        entry: Entry,
        /// Its launch.
        launch: Launch,
        /// Warps one output row's contraction is sliced across.
        splits: u32,
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
    /// A trellis-codec weight through the decode-shaped projection: an input
    /// transform, then one n16 column group a warp over `ksplit` slices of
    /// the contraction.
    TrellisDecode {
        /// Trellis width.
        bits: u32,
        /// Slices the contraction is cut into.
        ksplit: u32,
        /// Sixteen-row tiles one pass multiplies at once.
        tiles: u32,
    },
    /// A trellis-codec weight expanded once and multiplied by the BLAS tier.
    TrellisExpanded {
        /// Trellis width.
        bits: u32,
        /// Rows the call covers.
        rows: u32,
    },
    /// An MLX affine weight, read in place: one simdgroup four output rows, one
    /// activation row a launch at decode and [`AFFINE_TOKENS`] a threadgroup
    /// over a chunk.
    Affine {
        /// The projection itself.
        entry: Entry,
        /// Its launch.
        launch: Launch,
    },
    /// An MXFP8 weight, read in place: a one-row kernel at a decode step and a
    /// multi-row kernel over a verify pass of up to [`MXFP8_STEP_ROWS`] (each
    /// one of two blockings, by output width), and the tensor-core kernel over
    /// a prompt chunk of any width. Every one takes the same argument list.
    Mxfp8 {
        /// The projection itself.
        entry: Entry,
        /// Its launch.
        launch: Launch,
    },
    /// An MXFP8 weight against MXFP8 activations, a prompt chunk under
    /// [`PrefillActivations::Lowp`]: the activations quantized into
    /// [`Buf::HalfActivation`], then the block-scaled FP8 GEMM.
    Mxfp8Lowp {
        /// The quantizer's launch.
        quant: Launch,
        /// The GEMM's launch.
        gemm: Launch,
    },
}

/// Threads an MXFP8 step-kernel block runs: four warps (`Q4E_MX_WARPS`),
/// which keep more blocks resident on a GB10 than eight-warp ones.
pub const MXFP8_THREADS: u32 = 128;

/// Output rows one block of the narrow one-row and the narrow multi-row MXFP8
/// kernels covers, a warp each.
pub const MXFP8_ROWS_PER_BLOCK: u32 = 4;

/// Output rows one block of the wide multi-row MXFP8 kernel covers, four a
/// warp: the block stages a verify pass's activations once for sixteen weight
/// rows, so the pass reads the weight about as fast as a one-row step does.
pub const MXFP8_VERIFY_ROWS_PER_BLOCK: u32 = 16;

/// Output rows one block of the wide one-row MXFP8 kernel covers, two a
/// warp.
pub const MXFP8_GEMV_ROWS_PER_BLOCK: u32 = 8;

/// The widest step pass the multi-row MXFP8 kernel takes in one launch; a
/// wider one (a batched verify) runs as passes of this width. A row's sum
/// does not depend on the width it shares, so the passes are the one launch's
/// bits.
pub const MXFP8_STEP_ROWS: u32 = 8;

/// Output rows and activation rows one tile of the MXFP8 prompt kernel covers
/// (`Q4E_MX_BM`, `Q4E_MX_BN`).
pub const MXFP8_GEMM_TILE: u32 = 128;

/// Threads the MXFP8 prompt kernel runs a tile with: sixteen warps of 32 x 32
/// (`Q4E_MX_GEMM_THREADS`).
pub const MXFP8_GEMM_THREADS: u32 = 512;

/// Contraction widths the W8A8 prompt kernel takes: its scales arrive sixteen
/// groups (512 elements) a row at a time.
pub const MXFP8_A8_K_ALIGN: u32 = 512;

/// Rows and outputs one tile of the W8A8 prompt kernel covers (`LP_A8_BM`,
/// `LP_A8_BN`).
pub const MXFP8_A8_TILE: u32 = 128;

/// Threads the W8A8 prompt kernel runs a tile with (`LP_A8_THREADS`).
pub const MXFP8_A8_THREADS: u32 = 256;

/// Dynamic shared memory the W8A8 prompt kernel takes (`LP_A8_SMEM`): four
/// stages of two 128 x 64 code tiles and two chunks of sixteen scale bytes
/// for its 256 rows.
pub const MXFP8_A8_SMEM: u32 = 4 * 2 * 128 * 64 + 2 * 256 * 16;

/// Threads one block of the MXFP8 quantizer runs, four values a thread.
pub const MXFP8_A8_QUANT_THREADS: u32 = 256;

/// Dynamic shared memory the MXFP8 prompt kernel takes (`Q4E_MX_GEMM_SMEM`):
/// a three-step ring of raw codes and activations, two converted half tiles
/// of each operand and two steps of scales. Above the 48 KiB a launch gets
/// without opting in, below the device's 99 KiB per block.
pub const MXFP8_GEMM_SMEM: u32 = 3 * (128 * 32 + 128 * 32 * 4) + 4 * (128 * 32 * 2) + 2 * 128 * 4;

/// The MXFP8 tier.
///
/// A prompt pass takes the tensor-core kernel at every width: its rows only
/// have to agree with each other across pass widths, which that kernel keeps
/// by summing a column in an order fixed by the shape. A step takes the
/// warp-per-row kernels, whose sums are the brain-float tier's on the widened
/// weight and do not move with the row count, so a verify row reproduces the
/// single-row step.
fn mxfp8(in_dim: u32, out_dim: u32, rows: u32, device: Device) -> CoreResult<Projection> {
    if crate::sparse::weights::mxfp8_bytes(in_dim, out_dim).is_none() {
        return Err(CoreError::ShapeMismatch {
            field: "mxfp8 contraction width",
            expected: u64::from(crate::sparse::weights::MXFP8_GROUP),
            got: u64::from(in_dim),
        });
    }
    // A contraction the W8A8 kernel cannot stride stays on the weight-only
    // one: a fact about the weight's shape, so every width of every prompt
    // takes the same kernel for it.
    if device.lowp_prompt() && in_dim.is_multiple_of(MXFP8_A8_K_ALIGN) {
        let quads = u64::from(rows) * u64::from(in_dim) / 4;
        return Ok(Projection::Mxfp8Lowp {
            quant: Launch::linear(
                blocks_for(quads, u64::from(MXFP8_A8_QUANT_THREADS)),
                MXFP8_A8_QUANT_THREADS,
            ),
            gemm: Launch::new(
                shiso_core::Extent3::new(
                    out_dim.div_ceil(MXFP8_A8_TILE),
                    rows.div_ceil(MXFP8_A8_TILE),
                    1,
                ),
                MXFP8_A8_THREADS,
                MXFP8_A8_SMEM,
            ),
        });
    }
    if device.row_invariant() {
        return Ok(Projection::Mxfp8 {
            entry: Entry::plain(MOD_MXFP8, "q4e_mxfp8_gemm_kernel"),
            launch: Launch::new(
                shiso_core::Extent3::new(
                    out_dim.div_ceil(MXFP8_GEMM_TILE),
                    rows.div_ceil(MXFP8_GEMM_TILE),
                    1,
                ),
                MXFP8_GEMM_THREADS,
                MXFP8_GEMM_SMEM,
            ),
        });
    }
    // The one-row and the multi-row kernels each come in two blockings,
    // chosen by the output width alone: both sum a row the same way, so the
    // choice moves no bit.
    let (entry, rows_per_block) = match rows {
        1 if out_dim > NARROW_OUTPUT => ("q4e_mxfp8_gemv_kernel", MXFP8_GEMV_ROWS_PER_BLOCK),
        1 => ("q4e_mxfp8_gemv_narrow_kernel", MXFP8_ROWS_PER_BLOCK),
        2..=MXFP8_STEP_ROWS if out_dim > NARROW_OUTPUT => {
            ("q4e_mxfp8_rows_kernel", MXFP8_VERIFY_ROWS_PER_BLOCK)
        }
        2..=MXFP8_STEP_ROWS => ("q4e_mxfp8_rows_narrow_kernel", MXFP8_ROWS_PER_BLOCK),
        _ => {
            return Err(CoreError::Unsupported {
                what: "an MXFP8 step pass wider than its multi-row kernel; it runs in passes",
                backend: "engine",
            });
        }
    };
    Ok(Projection::Mxfp8 {
        entry: Entry::plain(MOD_MXFP8, entry),
        launch: Launch::linear(out_dim.div_ceil(rows_per_block), MXFP8_THREADS),
    })
}

/// Output rows one affine projection threadgroup covers
/// (`Q4E_AFFINE_ROWS`).
pub const AFFINE_ROWS: u32 = 8;
/// Threads one affine projection threadgroup runs: two simdgroups of four
/// output rows each (`Q4E_AFFINE_SG_ROWS`).
pub const AFFINE_THREADS: u32 = 64;
/// Activation rows one multi-row affine threadgroup carries
/// (`Q4E_AFFINE_TOKENS`).
pub const AFFINE_TOKENS: u32 = 8;

/// Output rows and activation rows one tile of the prompt projection covers
/// (`Q4E_GEMM_BM`, `Q4E_GEMM_BN`).
pub const AFFINE_GEMM_TILE: u32 = 32;
/// Threads the prompt projection runs a tile with: four simdgroups.
pub const AFFINE_GEMM_THREADS: u32 = 128;

/// Slices an affine projection's contraction is split into, one pair of
/// simdgroups each: a narrow output over a wide contraction is too few
/// threadgroups to keep the device's memory busy, and a wide output already
/// is. A fact about the shape alone, so the one-row and multi-row kernels sum
/// a row the same way.
#[must_use]
pub const fn affine_splits(in_dim: u32, out_dim: u32) -> u32 {
    let sweeps = in_dim / 512;
    if out_dim.div_ceil(AFFINE_ROWS) >= 768 {
        1
    } else if sweeps >= 16 {
        4
    } else if sweeps >= 4 {
        2
    } else {
        1
    }
}

/// The affine tier: the one-row kernel for a single activation row and the
/// multi-row one for a chunk, over the same grid of output rows.
///
/// A prompt pass takes the matrix-unit kernel at every width instead: its
/// rows only have to agree with each other across chunk widths, which that
/// kernel keeps, and a chunk is where the one-row kernels' sweep order costs
/// the most.
fn affine(in_dim: u32, out_dim: u32, rows: u32, device: Device) -> CoreResult<Projection> {
    if crate::sparse::weights::affine_bytes(DType::MlxAffine4Gs64, in_dim, out_dim).is_none() {
        return Err(CoreError::ShapeMismatch {
            field: "affine contraction width",
            expected: 64,
            got: u64::from(in_dim),
        });
    }
    if device.row_invariant() {
        return Ok(Projection::Affine {
            entry: Entry::plain(MOD_MATVEC, "q4e_affine_gemm_kernel"),
            launch: Launch::new(
                shiso_core::Extent3::new(
                    out_dim.div_ceil(AFFINE_GEMM_TILE),
                    rows.div_ceil(AFFINE_GEMM_TILE),
                    1,
                ),
                AFFINE_GEMM_THREADS,
                0,
            ),
        });
    }
    let (entry, tokens) = if rows == 1 {
        ("q4e_affine_mv_kernel", 1)
    } else {
        ("q4e_affine_mm_kernel", AFFINE_TOKENS)
    };
    Ok(Projection::Affine {
        entry: Entry::plain(MOD_MATVEC, entry),
        launch: Launch::new(
            shiso_core::Extent3::new(out_dim.div_ceil(AFFINE_ROWS), rows.div_ceil(tokens), 1),
            AFFINE_THREADS * affine_splits(in_dim, out_dim),
            0,
        ),
    })
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
    /// Whether this pass's stream mixes leave the block input in half precision
    /// as well, in [`Buf::MixedHalf`]. A fact about the pass rather than the
    /// device, carried here because every planner that projects the block input
    /// already takes this struct.
    pub half_block_input: bool,
    /// Which reference pass this one's rows must reproduce. On a
    /// [`PassRole::Prompt`] pass every row comes out bit for bit as it would in
    /// a pass of any other width: every tier decision that reads a row count
    /// reads the widest one instead, and every multiply is asked for
    /// [`GemmReduction::ColumnInvariant`]. A fact about the pass, carried here
    /// for the reason `half_block_input` is.
    pub role: PassRole,
    /// Which kernels a pass's brain-float projections take: a matrix-unit
    /// one for a prompt chunk at every width where the device has it, and on
    /// a device with [`MOD_BF16`] that module's step kernels as well. A probed
    /// device fact: a kernel set without either keeps the warp-per-row kernel.
    pub prompt_matmul: PromptMatmul,
    /// The precision a prompt pass's activations enter the block-scaled
    /// weights' products at; see [`PrefillActivations`]. An operator's
    /// choice, checked against the kernel set at startup.
    pub prefill_activations: PrefillActivations,
    /// Which one-launch routed selection a decode-sized pass takes. A probed
    /// device fact, for the reason `prompt_matmul` is.
    pub route_sort: RouteSort,
}

impl Device {
    /// Whether this pass's rows are held to every other pass width.
    #[must_use]
    pub const fn row_invariant(self) -> bool {
        matches!(self.role, PassRole::Prompt { .. })
    }

    /// The reduction every multiply of this pass asks the BLAS tier for.
    #[must_use]
    pub const fn reduction(self) -> GemmReduction {
        if self.row_invariant() {
            GemmReduction::ColumnInvariant
        } else {
            GemmReduction::Fastest
        }
    }

    /// The row count a tier decision reads for a pass of `rows`: the rows
    /// themselves, or on a row-invariant pass the widest a pass can be, so
    /// that every width takes the tier a full chunk takes.
    #[must_use]
    pub const fn tier_rows(self, rows: u32) -> u32 {
        if self.row_invariant() { u32::MAX } else { rows }
    }

    /// Whether this pass quantizes its activations for the block-scaled
    /// kernels: a prompt pass under [`PrefillActivations::Lowp`]. A step or
    /// a verify never does, whatever the option, so a decoded token is the
    /// weight-only tier's answer and speculation's equality holds.
    #[must_use]
    pub const fn lowp_prompt(self) -> bool {
        self.row_invariant() && matches!(self.prefill_activations, PrefillActivations::Lowp)
    }
}

/// The precision a prompt pass's activations take into the products with the
/// block-scaled weights (the NVFP4 routed experts and the MXFP8 dense
/// projections).
///
/// A trade of accuracy for prefill speed, made by the operator
/// (`--prefill-activations`). Only prompt passes are affected: decode steps
/// and speculative verifies keep the weight-only kernels under either choice.
/// Within a prompt the choice reads no row count, so a row's bits still do not
/// depend on where the prompt was cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrefillActivations {
    /// Activations kept at half precision against the decoded weights: the
    /// weight-only kernels, the reference numerics.
    #[default]
    A16,
    /// Activations quantized to the weight's own format - NVFP4 (one `e4m3`
    /// scale per 16, a per-row global) for the routed experts, MXFP8 (one
    /// `ue8m0` scale per 32) for the dense projections - and multiplied on the
    /// block-scaled tensor-core instructions. Needs a kernel set built for a
    /// family-specific architecture (`sm_121a`); refused at startup without
    /// one.
    Lowp,
}

/// The one-launch routed selection a decode-sized pass takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RouteSort {
    /// `q4e_moe_route_sort_small_kernel` at every width it takes: every
    /// kernel set has it.
    #[default]
    Small,
    /// `q4e_moe_route_sort_tiny_kernel` for a pass of at most
    /// [`crate::sparse::moe::ROUTE_TINY_MAX_TOKENS`] rows, the small one past
    /// that: the same results at a shorter serial chain.
    Tiny,
}

/// The kernel a prompt pass's brain-float projections take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PromptMatmul {
    /// The warp-per-row kernel, sixteen rows a launch: every kernel set has it.
    #[default]
    Rows,
    /// The matrix-unit kernel, a whole chunk a launch.
    Matrix,
    /// The brain-float module of a device with tensor cores ([`MOD_BF16`]):
    /// its tensor-core GEMM for a prompt chunk, a whole chunk a launch, and
    /// its split step kernels for a decode step or a verify pass in place of
    /// the warp-per-row kernel. The one probe covers both because they are
    /// one module: a kernel set that has the GEMM has the step kernels.
    Tensor,
}

/// What one session's pass of consecutive positions is for.
///
/// The two are planned differently for one reason: a prompt's rows must come
/// out exactly as they would in a pass of any other width, and a step's rows
/// must come out exactly as a single decode step would compute them. Both are
/// the same promise - the text must not depend on how the work was cut - made
/// to two different reference passes, and no plan can keep both at once: the
/// kernels a one-row step is fast on are the ones whose sums move with the row
/// count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PassRole {
    /// A chunk of a prompt. Every tier is chosen as for the widest chunk, so
    /// that where the prompt was cut - the prefill width, a quantum shared with
    /// decoding sequences, a resume point - moves no bit of any row. See
    /// [`Device::role`].
    Prompt {
        /// The row of this pass before which the session's recurrent state is
        /// also written to the stack's snapshot buffer, when the pass carries
        /// one: the last grid point of a prompt that ends between two, which is
        /// where a longer prompt resumes. Tiers never read it.
        snapshot: Option<u32>,
    },
    /// A decode step or a speculative verify: rows a single-row step must be
    /// able to reproduce, planned on the tiers their row count reaches.
    #[default]
    Step,
}

impl PassRole {
    /// The snapshot row a prompt pass carries, if any.
    #[must_use]
    pub const fn snapshot(self) -> Option<u32> {
        match self {
            Self::Prompt { snapshot } => snapshot,
            Self::Step => None,
        }
    }
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
        DType::F16Pack12 => packed_half(in_dim, out_dim, rows, device),
        DType::BF16 => brain(in_dim, out_dim, rows, device),
        DType::Exl3K4 | DType::Exl3K5 | DType::Exl3K6 => {
            trellis(dtype, in_dim, out_dim, rows, device)
        }
        DType::MlxAffine4Gs64 => affine(in_dim, out_dim, rows, device),
        DType::Mxfp8E4M3G32 => mxfp8(in_dim, out_dim, rows, device),
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
    let tier = device.tier_rows(rows);
    if half_exact(in_dim, out_dim, tier) && exact_blas(device) {
        return Ok(Projection::HalfExact { rows });
    }
    if tier > HALF_MAX_ROWS && !half_exact(in_dim, out_dim, tier) {
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
    let pass = rows.min(HALF_MAX_ROWS);
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

/// The packed half-precision tier: [`half`]'s ladder over the twelve-bit
/// repack, the per-row kernel reading the packed planes and the two BLAS tiers
/// reading the weight expanded back to what the checkpoint stored.
fn packed_half(in_dim: u32, out_dim: u32, rows: u32, device: Device) -> CoreResult<Projection> {
    if !in_dim.is_multiple_of(crate::sparse::pack12::PACK12_COLUMN_QUANTUM) {
        return Err(CoreError::ShapeMismatch {
            field: "packed half-precision contraction width",
            expected: u64::from(crate::sparse::pack12::PACK12_COLUMN_QUANTUM),
            got: u64::from(in_dim),
        });
    }
    Ok(match half(in_dim, out_dim, rows, device)? {
        Projection::HalfExact { rows } => Projection::PackedExpanded {
            wide: Elem::F32,
            rows,
        },
        Projection::HalfBlas { rows } => Projection::PackedExpanded {
            wide: Elem::F16,
            rows,
        },
        Projection::HalfRows {
            launch,
            chunk,
            combine,
            rows,
            ..
        } => Projection::HalfRows {
            entry: Entry::template(
                MOD_MATVEC,
                "q4e_matmul_p12_rows_kernel",
                TemplateArgs::Rows(rows),
            ),
            launch: kernels::matmul_p12_rows(out_dim, launch.grid.y, rows),
            chunk,
            combine,
            rows,
        },
        other => other,
    })
}

/// The expansion of a packed half-precision weight into the staging.
fn expand_packed(out: &mut Plan, weight: WeightDesc, w: Operand, wide: Elem) {
    let elements = u64::from(weight.in_dim) * u64::from(weight.out_dim);
    let width = match wide {
        Elem::F16 => 2,
        Elem::F32 => 4,
    };
    out.push(Step::Expand {
        entry: Entry::template(
            MOD_MATVEC,
            "q4e_p12_expand_kernel",
            TemplateArgs::Elem(wide),
        ),
        k: weight.in_dim,
        n: weight.out_dim,
        // The element width being restored; this step decodes no trellis.
        bits: 16,
        launch: Launch::linear(weight.out_dim, 256),
        staging: Staging::of(elements * width),
        // `dst, w, in_dim, out_dim`
        args: Args::of(&[
            Operand::CoopStaging(0),
            w,
            Operand::U32(weight.in_dim),
            Operand::U32(weight.out_dim),
        ]),
    });
}

/// The brain-float tier, which has no split.
///
/// One row is instantiated because a draft head can carry brain-float dense
/// projections, and a draft pass is one row; a chunk wider than the
/// instantiated width is split into passes by [`project`].
///
/// A prompt pass on a device with the matrix-unit kernel takes it at every
/// width instead, as the affine tier does, for the same reason.
fn brain(in_dim: u32, out_dim: u32, rows: u32, device: Device) -> CoreResult<Projection> {
    if matches!(device.prompt_matmul, PromptMatmul::Tensor) {
        return brain_tensor(in_dim, out_dim, rows, device);
    }
    if brain_gemm(in_dim, device) {
        return Ok(Projection::BrainGemm {
            entry: Entry::plain(MOD_MATVEC, "q4e_bf16_gemm_kernel"),
            launch: Launch::new(
                shiso_core::Extent3::new(
                    out_dim.div_ceil(AFFINE_GEMM_TILE),
                    rows.div_ceil(AFFINE_GEMM_TILE),
                    1,
                ),
                AFFINE_GEMM_THREADS,
                0,
            ),
        });
    }
    if !(1..=MAX_ROWS).contains(&rows) {
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

/// Whether a brain-float projection of contraction `in_dim` takes a
/// matrix-unit kernel on this pass: a prompt pass, on a device that has one,
/// over a contraction its 32-wide slices divide.
const fn brain_gemm(in_dim: u32, device: Device) -> bool {
    matches!(
        device.prompt_matmul,
        PromptMatmul::Matrix | PromptMatmul::Tensor
    ) && device.row_invariant()
        && in_dim.is_multiple_of(32)
}

/// The widest pass the brain-float step kernels of [`MOD_BF16`] take in one
/// launch; a wider step pass runs as passes of this width, which are the one
/// launch's bits because a row's sum does not depend on the rows beside it.
/// The half-precision tier's width ([`HALF_MAX_ROWS`]), for the reason given
/// there: a batched verify is wider than sixteen rows.
pub const BF16_STEP_ROWS: u32 = HALF_MAX_ROWS;

/// Warps a brain-float step-kernel block runs (`Q4E_BF_WARPS`).
pub const BF16_STEP_WARPS: u32 = 8;

/// Threads a brain-float step-kernel block runs.
pub const BF16_STEP_THREADS: u32 = BF16_STEP_WARPS * kernels::WARP;

/// Output rows one warp of the brain-float step kernel for `rows` activation
/// rows owns (`R` in `bf16.cu`): one at a decode step, two over a verify pass.
/// Sharing each activation load between two output rows is what keeps a
/// verify pass bound by the weight read.
#[must_use]
pub const fn bf16_warp_rows(rows: u32) -> u32 {
    if rows <= 1 { 1 } else { 2 }
}

/// Warps one output row's contraction is sliced across in a brain-float step
/// projection: 1, 2, 4 or 8, a function of the shape alone so that every pass
/// width sums a row the same way. A narrow output is sliced until about two
/// thousand warps carry it, each slice at least one full step of the warp's
/// loads: the stream mix down (320 outputs over 10,240) on one warp a row is
/// 320 warps, too few to keep the memory busy.
#[must_use]
pub const fn bf16_split(in_dim: u32, out_dim: u32) -> u32 {
    let mut split = 1;
    while split < BF16_STEP_WARPS
        && out_dim.saturating_mul(split) < 2_048
        && in_dim / (split * 2) >= 256
    {
        split *= 2;
    }
    split
}

/// Output rows and activation rows one tile of the brain-float prompt GEMM
/// covers (`Q4E_BF_BM`, `Q4E_BF_BN`).
pub const BF16_GEMM_TILE: u32 = 128;

/// Threads the brain-float prompt GEMM runs a tile with: sixteen warps of
/// 32 x 32 (`Q4E_BF_GEMM_THREADS`).
pub const BF16_GEMM_THREADS: u32 = 512;

/// Dynamic shared memory the brain-float prompt GEMM takes
/// (`Q4E_BF_GEMM_SMEM(0)`): a three-step ring of weight tiles and raw
/// activations, and two rounded activation tiles. Above the 48 KiB a launch
/// gets without opting in, below the device's 99 KiB per block.
pub const BF16_GEMM_SMEM: u32 = 3 * (128 * 32 * 2 + 128 * 32 * 4) + 2 * (128 * 32 * 2);

/// Dynamic shared memory the split-activation prompt GEMM takes
/// (`Q4E_BF_GEMM_SMEM(1)`): a two-step ring, and two pairs of rounded tiles
/// (the activations' brain-float part and what rounding left).
pub const BF16_GEMM_SPLIT_SMEM: u32 = 2 * (128 * 32 * 2 + 128 * 32 * 4) + 4 * (128 * 32 * 2);

/// The brain-float tier on a device with [`MOD_BF16`].
///
/// A prompt pass takes the tensor-core GEMM at every width. The projections
/// [`half_exact`] keeps off the rounded half-precision multiply - narrow
/// outputs over the residual width, the router above all - take the split
/// GEMM, which carries each activation as two brain floats, sixteen
/// significant bits, so a near-tie between two experts is not decided by the
/// rounding; the rest round their activations to brain floats, as the
/// checkpoint's own arithmetic does.
///
/// A step pass takes the split step kernels, whose sums do not move with the
/// row count: a verify row reproduces the single-row step bit for bit. A
/// prompt pass over a contraction the GEMM's slices do not divide takes them
/// too, which keeps its promise for the same reason.
fn brain_tensor(in_dim: u32, out_dim: u32, rows: u32, device: Device) -> CoreResult<Projection> {
    if brain_gemm(in_dim, device) {
        let (entry, smem) = if half_exact(in_dim, out_dim, device.tier_rows(rows)) {
            ("q4e_bf16_gemm_split_kernel", BF16_GEMM_SPLIT_SMEM)
        } else {
            ("q4e_bf16_gemm_kernel", BF16_GEMM_SMEM)
        };
        return Ok(Projection::BrainGemm {
            entry: Entry::plain(MOD_BF16, entry),
            launch: Launch::new(
                shiso_core::Extent3::new(
                    out_dim.div_ceil(BF16_GEMM_TILE),
                    rows.div_ceil(BF16_GEMM_TILE),
                    1,
                ),
                BF16_GEMM_THREADS,
                smem,
            ),
        });
    }
    // Eight brain floats a lane: the kernels load sixteen bytes of weight at
    // a time.
    if !in_dim.is_multiple_of(8) {
        return Err(CoreError::ShapeMismatch {
            field: "brain-float contraction width",
            expected: 8,
            got: u64::from(in_dim),
        });
    }
    let entry = match rows {
        1 => "q4e_bf16_gemv_kernel",
        2..=8 => "q4e_bf16_rows_kernel",
        9..=BF16_STEP_ROWS => "q4e_bf16_rows_wide_kernel",
        _ => {
            return Err(CoreError::Unsupported {
                what: "a brain-float step pass wider than its widest kernel; it runs in passes",
                backend: "engine",
            });
        }
    };
    let splits = bf16_split(in_dim, out_dim);
    let rows_per_block = BF16_STEP_WARPS / splits * bf16_warp_rows(rows);
    Ok(Projection::BrainStep {
        entry: Entry::plain(MOD_BF16, entry),
        launch: Launch::linear(out_dim.div_ceil(rows_per_block), BF16_STEP_THREADS),
        splits,
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
        rows: device.tier_rows(rows),
        blas: device.blas,
        capturing: device.capturing,
    };
    if exl3::reconstruct(gate, in_dim, out_dim) {
        return Ok(Projection::TrellisExpanded { bits, rows });
    }
    if let Some(ksplit) = decode_split(
        in_dim,
        out_dim,
        device.tier_rows(rows),
        device.max_co_resident,
    ) {
        return Ok(Projection::TrellisDecode {
            bits,
            ksplit,
            tiles: if rows > 16 { 2 } else { 1 },
        });
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

/// Rows up to which a dense trellis projection takes the decode-shaped launch.
///
/// The same bound the cooperative kernel holds below the expansion, and it has
/// to be: every pass up to it sums a row the same way, so a short prompt
/// prefilled alone and one prefilled in a pass beside other sessions' prompts
/// get the same bits. Drawn at 32 instead, a prompt's text depended on what it
/// was batched with (measured: two identical requests in one batch diverged).
pub const TRELLIS_DECODE_MAX_ROWS: u32 = kernels::EXL3_RECONSTRUCT_ROWS;

/// Width at or below which one side of a dense trellis weight has to be for
/// the decode-shaped launch to take it.
///
/// The cooperative kernel runs one block an SM, so a weight narrow on either
/// side leaves most of each block idle behind its lock chain: the attention
/// key and value projections (2560 -> 512) ran at 60 GB/s there, the shared
/// expert's (640 -> 2560) at 90, the indexer's under 50. On the wide
/// projections - the recurrent input and output, the attention query - the
/// cooperative kernel's decode loop is the faster of the two by a fifth on a
/// GB10, where a six-bit trellis is bound by its decode rather than by the
/// bytes (it slows in proportion when the SM clock is pinned lower), so those
/// stay where they are. Measured in a standalone harness at one to sixteen
/// rows.
pub const TRELLIS_DECODE_NARROW: u32 = 1_024;

/// Warps the decode-shaped projection wants in flight, which is what the
/// contraction is sliced to reach: about eight an SM on a 48-SM device, where
/// fewer left a narrow output's weight streaming at a third of the bandwidth.
const TRELLIS_DECODE_WARPS: u32 = 384;

/// The contraction slices a dense trellis projection takes on the
/// decode-shaped launch, or `None` when this shape and row count do not take
/// it.
///
/// The fewest slices that reach [`TRELLIS_DECODE_WARPS`] and make the block
/// count a multiple of `sms`, the device's one-block-an-SM residency: the
/// blocks are all resident at once, so an SM holding one more than its
/// neighbours is the launch's tail (measured: a 160-block grid on 48 SMs ran
/// the recurrent input projection 17% slower than the cooperative kernel's 48
/// balanced blocks). A function of the shape and the device alone, so every
/// row count below the bound sums a row the same way.
#[must_use]
pub fn decode_split(in_dim: u32, out_dim: u32, rows: u32, sms: u32) -> Option<u32> {
    let group = 16 * kernels::EXL3_DGEMV_WARPS;
    if rows > TRELLIS_DECODE_MAX_ROWS
        || (out_dim > TRELLIS_DECODE_NARROW && in_dim > TRELLIS_DECODE_NARROW)
        || !out_dim.is_multiple_of(128)
        || !in_dim.is_multiple_of(128)
        || !128_u32.is_multiple_of(group)
    {
        return None;
    }
    let groups = out_dim / 16;
    let blocks = out_dim / group;
    let stages = in_dim / kernels::EXL3_DGEMV_KT;
    let want = TRELLIS_DECODE_WARPS.div_ceil(groups).clamp(1, stages);
    let sms = sms.max(1);
    (want..=stages)
        .find(|split| (blocks * split).is_multiple_of(sms))
        .or(Some(want))
}

/// Whether one weight of a pair takes the decode-shaped launch at `rows`,
/// which is what splits the pair into two ordinary launches.
#[must_use]
pub fn pair_decoded(weight: &WeightDesc, rows: u32, device: Device) -> bool {
    trellis_bits(weight.dtype).is_some()
        && decode_split(
            weight.in_dim,
            weight.out_dim,
            device.tier_rows(rows),
            device.max_co_resident,
        )
        .is_some()
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
    let passes = match weight.dtype {
        DType::F16 | DType::F16Pack12 => {
            half_in_passes(weight.in_dim, weight.out_dim, rows, device)
        }
        // The brain-float tier has no BLAS path, so passes are its only way
        // through a chunk - a draft head's catch-up over the prompt is one.
        DType::BF16 => {
            let widest = if matches!(device.prompt_matmul, PromptMatmul::Tensor) {
                BF16_STEP_ROWS
            } else {
                MAX_ROWS
            };
            rows > widest && !brain_gemm(weight.in_dim, device)
        }
        // A step pass wider than the multi-row kernel - a batched verify -
        // runs in passes of it; a prompt chunk is one tensor-core launch.
        DType::Mxfp8E4M3G32 => rows > MXFP8_STEP_ROWS && !device.row_invariant(),
        _ => false,
    };
    if passes {
        let width = match weight.dtype {
            DType::F16 | DType::F16Pack12 => HALF_MAX_ROWS,
            DType::BF16 if matches!(device.prompt_matmul, PromptMatmul::Tensor) => BF16_STEP_ROWS,
            DType::Mxfp8E4M3G32 => MXFP8_STEP_ROWS,
            _ => MAX_ROWS,
        };
        let mut done = 0;
        while done < rows {
            let take = (rows - done).min(width);
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

/// Matrices one MXFP8 group launch takes (`Q4E_MXG_PARAMS`).
pub const MXFP8_GROUP_SLOTS: usize = 4;

/// Output rows in all at or below which an MXFP8 group takes its one-row-a-
/// warp entries. The shared expert's pair (1280 rows) measured 22.5 us on two
/// rows a warp and 18.7 us on the pipelined one row a warp; the recurrent and
/// attention groups (over 13,000 rows) are fastest on two.
pub const MXFP8_GROUP_NARROW: u32 = 4_096;

/// Whether projections of one input with these weights take MXFP8 group
/// launches at `rows`: every weight MXFP8 over one contraction width, on a
/// step or verify pass.
///
/// A prompt pass keeps the tensor-core kernel per matrix, whose tiles gain
/// nothing from sharing a launch. A planner that reorders projections to put
/// them side by side asks this first, so that a weight in any other format
/// keeps its place in the chain.
#[must_use]
pub fn groups(weights: &[WeightDesc], device: Device) -> bool {
    weights.len() > 1
        && !device.row_invariant()
        && weights.iter().all(|w| {
            w.dtype == DType::Mxfp8E4M3G32
                && w.in_dim == weights[0].in_dim
                && crate::sparse::weights::mxfp8_bytes(w.in_dim, w.out_dim).is_some()
        })
}

/// The MXFP8 group entry for `rows` activation rows over matrices of `total`
/// output rows in all, and the output rows one of its blocks covers.
///
/// The blockings sum a row identically, so the choice is made for bandwidth
/// alone and by the shape alone, like the single-matrix kernels'.
#[must_use]
pub const fn mxfp8_group_entry(rows: u32, total: u32) -> (&'static str, u32) {
    // A verify block stages the pass's activations once for its rows, so one
    // row a warp re-stages them four times as often: past the single-matrix
    // kernels' narrow width that costs more than the extra warps gain.
    match rows {
        1 if total > MXFP8_GROUP_NARROW => {
            ("q4e_mxfp8_gemv_group_kernel", MXFP8_GEMV_ROWS_PER_BLOCK)
        }
        1 => ("q4e_mxfp8_gemv_group_narrow_kernel", MXFP8_ROWS_PER_BLOCK),
        _ if total > NARROW_OUTPUT => ("q4e_mxfp8_rows_group_kernel", MXFP8_VERIFY_ROWS_PER_BLOCK),
        _ => ("q4e_mxfp8_rows_group_narrow_kernel", MXFP8_ROWS_PER_BLOCK),
    }
}

/// Plans projections that all read one input, in order.
///
/// Where [`groups`] allows it, up to [`MXFP8_GROUP_SLOTS`] matrices share one
/// launch that reads the activation once and writes each matrix's own
/// destination, in passes of [`MXFP8_STEP_ROWS`] rows; a row of every matrix
/// is summed exactly as its own launch would sum it, so the group moves no
/// bit. Anything else is the projections one by one, as [`project`] plans
/// them.
///
/// # Errors
///
/// Whatever [`project`] reports for a weight this build cannot project, and
/// [`CoreError::ShapeMismatch`] for members that do not share their input.
pub fn project_group(
    out: &mut Plan,
    members: &[(WeightDesc, Bound)],
    rows: u32,
    device: Device,
) -> CoreResult<()> {
    let weights: Vec<WeightDesc> = members.iter().map(|(w, _)| *w).collect();
    if rows == 0 || !groups(&weights, device) {
        for &(weight, bound) in members {
            project(out, weight, bound, rows, device)?;
        }
        return Ok(());
    }
    if members.iter().any(|(_, b)| b.src != members[0].1.src) {
        return Err(CoreError::ShapeMismatch {
            field: "group projection input",
            expected: 1,
            got: 2,
        });
    }
    let in_dim = weights[0].in_dim;
    for slots in members.chunks(MXFP8_GROUP_SLOTS) {
        let mut done = 0;
        while done < rows {
            let take = (rows - done).min(MXFP8_STEP_ROWS);
            let src_bytes = u64::from(done) * u64::from(in_dim) * 4;
            let total: u32 = slots.iter().map(|(w, _)| w.out_dim).sum();
            let (entry, per_block) = mxfp8_group_entry(take, total);
            let blocks: u32 = slots
                .iter()
                .map(|(w, _)| w.out_dim.div_ceil(per_block))
                .sum();
            let mut ops = Vec::with_capacity(4 * MXFP8_GROUP_SLOTS + 3);
            for slot in 0..MXFP8_GROUP_SLOTS {
                if let Some(&(weight, bound)) = slots.get(slot) {
                    let dst_bytes = u64::from(done) * u64::from(weight.out_dim) * 4;
                    let b = bound.advanced(src_bytes, dst_bytes);
                    ops.extend([
                        b.dst,
                        Operand::Weight(b.weight),
                        Operand::Weight(part(b.weight, WeightPart::GroupScales)),
                        Operand::U32(weight.out_dim),
                    ]);
                } else {
                    ops.extend([Operand::Null, Operand::Null, Operand::Null, Operand::U32(0)]);
                }
            }
            ops.extend([
                members[0].1.src.advanced(src_bytes),
                Operand::U32(in_dim),
                Operand::U32(take),
            ]);
            out.launch(
                Entry::plain(MOD_MXFP8, entry),
                Launch::linear(blocks, MXFP8_THREADS),
                Args::of(&ops),
            );
            done += take;
        }
    }
    Ok(())
}

/// Plans one half-precision projection on the BLAS tier whose input may
/// already be half precision and whose output may be.
///
/// The stream mixes of a prefill chunk take this: the normed residual is
/// written as halves by the norm that produces it, so the rounding pass the
/// tier otherwise opens with - a whole read and write of the widest activation
/// in the model, ninety-seven times a chunk - has nothing left to do, and the
/// expansion's halves are what the collapse reads.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when the weight would not be planned onto
/// [`Projection::HalfBlas`] at this row count, which the caller's choice of
/// precision is supposed to have ruled out.
pub fn project_on_blas(
    out: &mut Plan,
    weight: WeightDesc,
    bound: Bound,
    rows: u32,
    device: Device,
    io: (Elem, Elem),
) -> CoreResult<()> {
    let a = match plan(weight.dtype, weight.in_dim, weight.out_dim, rows, device)? {
        Projection::HalfBlas { .. } => Operand::Weight(bound.weight),
        Projection::PackedExpanded {
            wide: Elem::F16, ..
        } => {
            expand_packed(out, weight, Operand::Weight(bound.weight), Elem::F16);
            Operand::CoopStaging(0)
        }
        _ => {
            return Err(CoreError::Unsupported {
                what: "a half-precision stream mix whose weight is not on the BLAS tier",
                backend: "engine",
            });
        }
    };
    let (input, output) = io;
    let b = match input {
        Elem::F16 => bound.src,
        Elem::F32 => {
            convert_activation(out, bound.src, weight.in_dim, rows);
            Operand::Scratch(Buf::HalfActivation)
        }
    };
    out.push(Step::Blas {
        op_a: GemmOp::Transposed,
        op_b: GemmOp::AsStored,
        m: weight.out_dim,
        n: rows,
        k: weight.in_dim,
        lda: weight.in_dim,
        ldb: weight.in_dim,
        ldc: weight.out_dim,
        a,
        b,
        c: bound.dst,
        dtype: match output {
            Elem::F16 => GemmDType::F16ToF16,
            Elem::F32 => GemmDType::F16,
        },
        reduction: device.reduction(),
    });
    Ok(())
}

/// Whether a chunk's stream mixes can keep their normed residual and their
/// expansion in half precision: every weight of every mix is half precision
/// and lands on the BLAS tier at this row count, where the activation is
/// rounded to halves anyway.
#[must_use]
pub fn half_streams<'a>(
    mixes: impl IntoIterator<Item = &'a crate::sparse::weights::MixDesc>,
    rows: u32,
    device: Device,
) -> bool {
    let on_blas = |weight: &WeightDesc| {
        matches!(weight.dtype, DType::F16 | DType::F16Pack12)
            && matches!(
                plan(weight.dtype, weight.in_dim, weight.out_dim, rows, device),
                Ok(Projection::HalfBlas { .. }
                    | Projection::PackedExpanded {
                        wide: Elem::F16,
                        ..
                    })
            )
    };
    let mut any = false;
    for mix in mixes {
        any = true;
        if !on_blas(&mix.down) || !on_blas(&mix.up) || !mix.inject.as_ref().is_none_or(on_blas) {
            return false;
        }
    }
    any
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
        // `out, w, x, in_dim, out_dim, rows`.
        Projection::BrainGemm { entry, launch } => out.launch(
            entry,
            launch,
            Args::of(&[
                bound.dst,
                w,
                bound.src,
                Operand::U32(weight.in_dim),
                Operand::U32(weight.out_dim),
                Operand::U32(rows),
            ]),
        ),
        // `out, w, x, in_dim, out_dim, splits, rows`.
        Projection::BrainStep {
            entry,
            launch,
            splits,
        } => out.launch(
            entry,
            launch,
            Args::of(&[
                bound.dst,
                w,
                bound.src,
                Operand::U32(weight.in_dim),
                Operand::U32(weight.out_dim),
                Operand::U32(splits),
                Operand::U32(rows),
            ]),
        ),
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
            let b = half_input(out, bound.src, weight.in_dim, rows, device);
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
                b,
                c: bound.dst,
                dtype: GemmDType::F16,
                reduction: device.reduction(),
            });
        }
        // The widening writes the weight into the staging region, output-major
        // as stored, and the multiply reads it transposed exactly as the
        // half-precision tier reads the weight in place.
        Projection::HalfExact { rows } => {
            let elements = u64::from(weight.in_dim) * u64::from(weight.out_dim);
            out.push(Step::Expand {
                entry: Entry::plain(MOD_MATVEC, "q4e_f16_to_f32_kernel"),
                k: weight.in_dim,
                n: weight.out_dim,
                // The element width being widened; this step decodes no trellis.
                bits: 16,
                launch: kernels::elementwise(elements),
                staging: Staging::of(elements * 4),
                // `out, in, n`
                args: Args::of(&[Operand::CoopStaging(0), w, Operand::U64(elements)]),
            });
            out.push(Step::Blas {
                op_a: GemmOp::Transposed,
                op_b: GemmOp::AsStored,
                m: weight.out_dim,
                n: rows,
                k: weight.in_dim,
                lda: weight.in_dim,
                ldb: weight.in_dim,
                ldc: weight.out_dim,
                a: Operand::CoopStaging(0),
                b: bound.src,
                c: bound.dst,
                dtype: GemmDType::F32Strict,
                reduction: device.reduction(),
            });
        }
        // The expansion, then the multiply the stored halves would have taken,
        // reading the staging where it would have read the weight.
        Projection::PackedExpanded { wide, rows } => {
            expand_packed(out, weight, w, wide);
            match wide {
                Elem::F16 => {
                    let b = half_input(out, bound.src, weight.in_dim, rows, device);
                    out.push(Step::Blas {
                        op_a: GemmOp::Transposed,
                        op_b: GemmOp::AsStored,
                        m: weight.out_dim,
                        n: rows,
                        k: weight.in_dim,
                        lda: weight.in_dim,
                        ldb: weight.in_dim,
                        ldc: weight.out_dim,
                        a: Operand::CoopStaging(0),
                        b,
                        c: bound.dst,
                        dtype: GemmDType::F16,
                        reduction: device.reduction(),
                    });
                }
                Elem::F32 => out.push(Step::Blas {
                    op_a: GemmOp::Transposed,
                    op_b: GemmOp::AsStored,
                    m: weight.out_dim,
                    n: rows,
                    k: weight.in_dim,
                    lda: weight.in_dim,
                    ldb: weight.in_dim,
                    ldc: weight.out_dim,
                    a: Operand::CoopStaging(0),
                    b: bound.src,
                    c: bound.dst,
                    dtype: GemmDType::F32Strict,
                    reduction: device.reduction(),
                }),
            }
        }
        // `out, w, scales, biases, x, in_dim, out_dim, rows`
        Projection::Affine { entry, launch } => out.launch(
            entry,
            launch,
            Args::of(&[
                bound.dst,
                w,
                Operand::Weight(part(bound.weight, WeightPart::AffineScales)),
                Operand::Weight(part(bound.weight, WeightPart::AffineBiases)),
                bound.src,
                Operand::U32(weight.in_dim),
                Operand::U32(weight.out_dim),
                Operand::U32(rows),
            ]),
        ),
        // The quantizer `q, x, rows, k` into the half-precision tier's
        // staging - which only that tier's rounding pass and the BLAS call
        // right after it ever use, never across another projection, and which
        // is sized for the widest contraction at two bytes an element, against
        // the 33/32 of a byte this layout takes - then the GEMM
        // `out, codes, scales, xq, in_dim, out_dim, rows`.
        Projection::Mxfp8Lowp { quant, gemm } => {
            out.launch(
                Entry::plain(MOD_MXFP8_A8, "q4e_mxfp8_a8_quant_kernel"),
                quant,
                Args::of(&[
                    Operand::Scratch(Buf::HalfActivation),
                    bound.src,
                    Operand::U32(rows),
                    Operand::U32(weight.in_dim),
                ]),
            );
            out.launch(
                Entry::plain(MOD_MXFP8_A8, "q4e_mxfp8_a8_gemm_kernel"),
                gemm,
                Args::of(&[
                    bound.dst,
                    w,
                    Operand::Weight(part(bound.weight, WeightPart::GroupScales)),
                    Operand::Scratch(Buf::HalfActivation),
                    Operand::U32(weight.in_dim),
                    Operand::U32(weight.out_dim),
                    Operand::U32(rows),
                ]),
            );
        }
        // `out, codes, scales, x, in_dim, out_dim, rows`
        Projection::Mxfp8 { entry, launch } => out.launch(
            entry,
            launch,
            Args::of(&[
                bound.dst,
                w,
                Operand::Weight(part(bound.weight, WeightPart::GroupScales)),
                bound.src,
                Operand::U32(weight.in_dim),
                Operand::U32(weight.out_dim),
                Operand::U32(rows),
            ]),
        ),
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
        // The input transform into the staging, then the slices, whose last
        // block per output tile folds the partial sums behind the transform.
        Projection::TrellisDecode {
            bits,
            ksplit,
            tiles,
        } => {
            let staging = Staging::decode(rows, weight.in_dim, weight.out_dim, ksplit);
            let partials = Staging::decode_partials(rows, weight.in_dim);
            let warps = u64::from(rows) * u64::from(weight.in_dim / 128);
            out.push(Step::Staged {
                entry: Entry::template(
                    MOD_EXL3,
                    kernels::EXL3_DGEMV_HAD,
                    TemplateArgs::Untemplated,
                ),
                launch: Launch::linear(blocks_for(warps, 8), 256),
                staging,
                // `A, A_had, suh, size_m, size_k`
                args: Args::of(&[
                    bound.src,
                    Operand::CoopStaging(0),
                    Operand::Weight(part(bound.weight, WeightPart::InputScales)),
                    Operand::I32(i32::try_from(rows).unwrap_or(i32::MAX)),
                    Operand::I32(i32::try_from(weight.in_dim).unwrap_or(i32::MAX)),
                ]),
            });
            let group = 16 * kernels::EXL3_DGEMV_WARPS;
            out.push(Step::Staged {
                entry: Entry::template(
                    MOD_EXL3,
                    kernels::EXL3_DGEMV,
                    TemplateArgs::TrellisTiles { bits, tiles },
                ),
                launch: Launch::new(
                    shiso_core::Extent3::new(weight.out_dim / group, ksplit, 1),
                    kernels::EXL3_DGEMV_WARPS * 32,
                    kernels::exl3_dgemv_smem(bits, tiles),
                ),
                staging,
                // `A_had, B, C, part, tickets, svh, size_m, size_k, size_n,
                //  ksplit`
                args: Args::of(&[
                    Operand::CoopStaging(0),
                    Operand::Weight(part(bound.weight, WeightPart::Tiles)),
                    bound.dst,
                    Operand::CoopStaging(partials),
                    Operand::CoopLocks,
                    Operand::Weight(part(bound.weight, WeightPart::OutputScales)),
                    Operand::I32(i32::try_from(rows).unwrap_or(i32::MAX)),
                    Operand::I32(i32::try_from(weight.in_dim).unwrap_or(i32::MAX)),
                    Operand::I32(i32::try_from(weight.out_dim).unwrap_or(i32::MAX)),
                    Operand::I32(i32::try_from(ksplit).unwrap_or(i32::MAX)),
                ]),
            });
        }
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
            let b = half_input(out, bound.src, weight.in_dim, rows, device);
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
                b,
                c: bound.dst,
                dtype: GemmDType::F16,
                reduction: device.reduction(),
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
    convert_into(out, src, in_dim, rows);
}

/// The half-precision operand a BLAS-tier projection of `src` reads: the copy
/// the stream mix already wrote when `src` is the whole block input and the
/// pass has one, and otherwise a rounding pass into the shared half buffer.
fn half_input(out: &mut Plan, src: Operand, in_dim: u32, rows: u32, device: Device) -> Operand {
    if device.half_block_input && src == Operand::Scratch(Buf::Mixed) {
        return Operand::Scratch(Buf::MixedHalf);
    }
    convert_into(out, src, in_dim, rows);
    Operand::Scratch(Buf::HalfActivation)
}

/// The rounding launch itself.
fn convert_into(out: &mut Plan, src: Operand, in_dim: u32, rows: u32) {
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
    // An MXFP8 pair is one group launch on a step or verify pass: the same
    // sums as the two launches, at one reading of the input.
    if groups(&[gate, up], device) {
        return project_group(out, &[(gate, gate_bound), (up, up_bound)], rows, device);
    }
    // Narrow enough for the decode-shaped launch, the pair is two of those, at
    // every row count that launch covers, so that a row is summed one way
    // below the bound whatever else shares its pass. Serially the two cost a
    // little more than the grouped launch (23 against 22 us for the shared
    // expert); the routed block runs them as a branch beside its own launches,
    // where the grouped launch could not go: it is cooperative, and its grid
    // has to be resident alone.
    if pair_decoded(&gate, rows, device) && pair_decoded(&up, rows, device) {
        project(out, gate, gate_bound, rows, device)?;
        return project(out, up, up_bound, rows, device);
    }
    // Past the threshold the pair splits: each weight is expanded once and
    // multiplied, which is cheaper than streaming both trellises once per
    // sixteen rows, and there is no grouped form of the expanded path. Below
    // it, one grouped launch runs both over half a device each instead of two
    // cooperative launches in sequence.
    if gate.matches(&up)
        && device.tier_rows(rows) <= kernels::EXL3_RECONSTRUCT_ROWS
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
        half_block_input: false,
        role: PassRole::Step,
        prompt_matmul: PromptMatmul::Rows,
        prefill_activations: PrefillActivations::A16,
        route_sort: RouteSort::Tiny,
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
    fn a_narrow_half_precision_chunk_is_multiplied_exactly_or_in_passes() {
        // A narrow output is never rounded onto the half-precision tier,
        // because that rounding is what flips a routing tie: it takes the
        // strict single-precision multiply where there is one, and passes of
        // the per-row kernel where there is not - or where a capture forbids
        // widening the weight into the staging.
        let plan = half(2_560, 512, 512, GPU).expect("the router is half precision");
        assert_eq!(plan, Projection::HalfExact { rows: 512 });
        for device in [
            Device { blas: false, ..GPU },
            Device {
                capturing: true,
                ..GPU
            },
        ] {
            let plan = half(2_560, 512, 512, device).expect("the router has a per-row tier");
            let Projection::HalfRows { rows, .. } = plan else {
                panic!("without the strict multiply a narrow output runs in passes: {plan:?}");
            };
            assert_eq!(rows, HALF_MAX_ROWS, "one pass is the instantiated width");
        }
    }

    #[test]
    fn a_stream_mix_contraction_goes_to_the_half_precision_tier_however_narrow() {
        // The whole multi-stream residual into the low-rank width: re-read for
        // every sixteen rows it was most of a prefill, and its activation is
        // rounded to halves anyway, as the reference does.
        let plan = half(10_240, 320, 512, GPU).expect("a stream mix has a tier");
        assert_eq!(plan, Projection::HalfBlas { rows: 512 });
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
    fn a_prompt_pass_takes_the_brain_float_matrix_kernel_at_every_width() {
        let prompt = Device {
            prompt_matmul: PromptMatmul::Matrix,
            role: PassRole::Prompt { snapshot: None },
            ..GPU
        };
        for rows in [1, 5, 512, 4096] {
            let plan = brain(2_560, 512, rows, prompt).expect("the device has it");
            let Projection::BrainGemm { entry, launch } = plan else {
                panic!("{plan:?}");
            };
            assert_eq!(entry.name(), "q4e_bf16_gemm_kernel");
            assert_eq!(launch.grid.y, rows.div_ceil(AFFINE_GEMM_TILE));
        }
        // Without the kernel, or on a step, the row kernel as before.
        for device in [
            Device {
                prompt_matmul: PromptMatmul::Rows,
                ..prompt
            },
            Device {
                prompt_matmul: PromptMatmul::Matrix,
                ..GPU
            },
        ] {
            let plan = brain(2_560, 512, 4, device).expect("the built set covers it");
            assert!(matches!(plan, Projection::BrainRows { .. }), "{plan:?}");
        }
    }

    #[test]
    fn a_brain_float_projection_runs_one_row_and_splits_a_chunk_into_passes() {
        for want in [1, 4, MAX_ROWS] {
            let plan = brain(2_560, 128, want, GPU).expect("the built set covers it");
            let Projection::BrainRows { entry, rows, .. } = plan else {
                panic!("{plan:?}");
            };
            assert_eq!(rows, want);
            assert_eq!(entry.name(), "q4e_matmul_bf16_rows_kernel");
        }
        let err =
            brain(2_560, 128, MAX_ROWS + 1, GPU).expect_err("one launch is at most the width");
        assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
        // A draft head's catch-up chunk: two full passes and a narrower tail.
        let mut out = Plan::new();
        let weight = WeightRef::layer(
            crate::sparse::bind::LayerRef::target(0),
            crate::sparse::bind::Role::AttnQuery,
        );
        let bound = Bound::new(
            weight,
            Operand::Scratch(Buf::Residual),
            Operand::Scratch(Buf::Embed),
        );
        project(
            &mut out,
            WeightDesc::new(DType::BF16, 2_560, 12_288),
            bound,
            2 * MAX_ROWS + 3,
            GPU,
        )
        .expect("a brain-float chunk runs in passes");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn a_tensor_core_device_takes_the_brain_float_gemm_on_a_prompt_pass() {
        let prompt = Device {
            prompt_matmul: PromptMatmul::Tensor,
            role: PassRole::Prompt { snapshot: None },
            ..GPU
        };
        // The stream mixes and the n-gram projections round their activations;
        // the router and the shared router, narrow over the residual width,
        // carry them as two brain floats. The choice reads the shape only, so
        // every width of a prompt takes the same kernel.
        for (out_dim, in_dim, want, smem) in [
            (320, 10_240, "q4e_bf16_gemm_kernel", BF16_GEMM_SMEM),
            (10_240, 320, "q4e_bf16_gemm_kernel", BF16_GEMM_SMEM),
            (4, 10_240, "q4e_bf16_gemm_kernel", BF16_GEMM_SMEM),
            (248_320, 2_560, "q4e_bf16_gemm_kernel", BF16_GEMM_SMEM),
            (
                512,
                2_560,
                "q4e_bf16_gemm_split_kernel",
                BF16_GEMM_SPLIT_SMEM,
            ),
            (1, 2_560, "q4e_bf16_gemm_split_kernel", BF16_GEMM_SPLIT_SMEM),
        ] {
            for rows in [1, 7, 2048] {
                let plan = brain(in_dim, out_dim, rows, prompt).expect("the device has it");
                let Projection::BrainGemm { entry, launch } = plan else {
                    panic!("{plan:?}");
                };
                assert_eq!(entry.name(), want, "[{out_dim} x {in_dim}] at {rows}");
                assert_eq!(entry.module(), MOD_BF16);
                assert_eq!(launch.grid.x, out_dim.div_ceil(BF16_GEMM_TILE));
                assert_eq!(launch.grid.y, rows.div_ceil(BF16_GEMM_TILE));
                assert_eq!(launch.smem, smem);
            }
        }
        const { assert!(BF16_GEMM_SMEM <= 101_376 && BF16_GEMM_SPLIT_SMEM <= 101_376) };
    }

    #[test]
    fn a_tensor_core_device_steps_brain_floats_on_the_split_kernels() {
        let step = Device {
            prompt_matmul: PromptMatmul::Tensor,
            ..GPU
        };
        for (rows, want, per_warp) in [
            (1, "q4e_bf16_gemv_kernel", 1),
            (2, "q4e_bf16_rows_kernel", 2),
            (8, "q4e_bf16_rows_kernel", 2),
            (9, "q4e_bf16_rows_wide_kernel", 2),
            (BF16_STEP_ROWS, "q4e_bf16_rows_wide_kernel", 2),
        ] {
            // The stream mix down: narrow, so each row is sliced eight ways.
            let plan = brain(10_240, 320, rows, step).expect("a step pass");
            let Projection::BrainStep {
                entry,
                launch,
                splits,
            } = plan
            else {
                panic!("{plan:?}");
            };
            assert_eq!(entry.name(), want);
            assert_eq!(splits, 8);
            assert_eq!(launch.grid.x, 320 / per_warp);
            assert_eq!(launch.block.x, BF16_STEP_THREADS);
            // The output head: wide, so one warp a row.
            let plan = brain(2_560, 248_320, rows, step).expect("a step pass");
            let Projection::BrainStep { launch, splits, .. } = plan else {
                panic!("{plan:?}");
            };
            assert_eq!(splits, 1);
            assert_eq!(launch.grid.x, 248_320 / (8 * per_warp));
        }
        // The split is the shape's alone.
        assert_eq!(bf16_split(2_560, 512), 4);
        assert_eq!(bf16_split(10_240, 4), 8);
        assert_eq!(bf16_split(320, 10_240), 1);
        assert_eq!(bf16_split(2_560, 2_560), 1);
        let err = brain(2_560, 512, BF16_STEP_ROWS + 1, step).expect_err("one launch at most");
        assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
        // A batched verify wider than one launch: passes of the widest one.
        let weight = WeightRef::layer(
            crate::sparse::bind::LayerRef::target(0),
            crate::sparse::bind::Role::AttnQuery,
        );
        let bound = Bound::new(
            weight,
            Operand::Scratch(Buf::Residual),
            Operand::Scratch(Buf::Embed),
        );
        let mut out = Plan::new();
        project(
            &mut out,
            WeightDesc::new(DType::BF16, 2_560, 12_288),
            bound,
            BF16_STEP_ROWS + 5,
            step,
        )
        .expect("a wide verify runs in passes");
        let names: Vec<_> = out
            .steps()
            .iter()
            .filter_map(Step::entry)
            .map(|e| e.name())
            .collect();
        assert_eq!(names, ["q4e_bf16_rows_wide_kernel", "q4e_bf16_rows_kernel"]);
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
    fn an_mxfp8_step_takes_the_row_kernels_and_a_prompt_the_tensor_core_kernel() {
        let one = plan(DType::Mxfp8E4M3G32, 2_560, 10_240, 1, GPU).expect("a decode row");
        let Projection::Mxfp8 { entry, launch } = one else {
            panic!("{one:?}");
        };
        assert_eq!(entry.name(), "q4e_mxfp8_gemv_kernel");
        assert_eq!(entry.module(), MOD_MXFP8);
        assert_eq!(launch.grid.x, 10_240 / MXFP8_GEMV_ROWS_PER_BLOCK);
        assert_eq!(launch.block.x, MXFP8_THREADS);
        // A narrow output keeps one row a warp.
        let narrow = plan(DType::Mxfp8E4M3G32, 2_560, 512, 1, GPU).expect("a decode row");
        let Projection::Mxfp8 { entry, launch } = narrow else {
            panic!("{narrow:?}");
        };
        assert_eq!(entry.name(), "q4e_mxfp8_gemv_narrow_kernel");
        assert_eq!(launch.grid.x, 512 / MXFP8_ROWS_PER_BLOCK);
        for rows in 2..=MXFP8_STEP_ROWS {
            let many = plan(DType::Mxfp8E4M3G32, 6_144, 2_560, rows, GPU).expect("a verify pass");
            let Projection::Mxfp8 { entry, launch } = many else {
                panic!("{many:?}");
            };
            assert_eq!(entry.name(), "q4e_mxfp8_rows_kernel");
            // The grid is the output's alone, whatever the row count.
            assert_eq!(launch.grid.x, 2_560 / MXFP8_VERIFY_ROWS_PER_BLOCK);
            assert_eq!(launch.grid.y, 1);
            let narrow = plan(DType::Mxfp8E4M3G32, 2_560, 48, rows, GPU).expect("a verify pass");
            let Projection::Mxfp8 { entry, launch } = narrow else {
                panic!("{narrow:?}");
            };
            assert_eq!(entry.name(), "q4e_mxfp8_rows_narrow_kernel");
            assert_eq!(launch.grid.x, 48 / MXFP8_ROWS_PER_BLOCK);
        }
        let err = plan(DType::Mxfp8E4M3G32, 2_560, 512, MXFP8_STEP_ROWS + 1, GPU)
            .expect_err("one step launch is at most the multi-row width");
        assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
        // A prompt pass takes the tensor-core kernel at every width, one row
        // included, so that where the prompt was cut moves no bit.
        let prompt = Device {
            role: PassRole::Prompt { snapshot: None },
            ..GPU
        };
        for rows in [1, 7, 64, 333, 4_096] {
            let p = plan(DType::Mxfp8E4M3G32, 2_560, 640, rows, prompt).expect("a prompt chunk");
            let Projection::Mxfp8 { entry, launch } = p else {
                panic!("{p:?}");
            };
            assert_eq!(entry.name(), "q4e_mxfp8_gemm_kernel");
            assert_eq!(launch.grid.x, 640 / MXFP8_GEMM_TILE);
            assert_eq!(launch.grid.y, rows.div_ceil(MXFP8_GEMM_TILE));
            assert_eq!(launch.block.x, MXFP8_GEMM_THREADS);
            assert_eq!(launch.smem, MXFP8_GEMM_SMEM);
        }
        let err = plan(DType::Mxfp8E4M3G32, 2_570, 640, 1, GPU)
            .expect_err("a contraction that is not whole groups");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }), "{err:?}");
    }

    #[test]
    fn low_precision_activations_take_the_block_scaled_kernel_on_prompt_rows_only() {
        let lowp = Device {
            prefill_activations: PrefillActivations::Lowp,
            ..GPU
        };
        let prompt = Device {
            role: PassRole::Prompt { snapshot: None },
            ..lowp
        };
        // Every prompt width takes the W8A8 kernel: the choice reads the
        // weight's shape, never the row count.
        for rows in [1, 7, 333, 4_096] {
            let p = plan(DType::Mxfp8E4M3G32, 2_560, 12_288, rows, prompt).expect("a prompt chunk");
            let Projection::Mxfp8Lowp { quant, gemm } = p else {
                panic!("{p:?}");
            };
            assert_eq!(gemm.grid.x, 12_288 / MXFP8_A8_TILE);
            assert_eq!(gemm.grid.y, rows.div_ceil(MXFP8_A8_TILE));
            assert_eq!((gemm.block.x, gemm.smem), (MXFP8_A8_THREADS, MXFP8_A8_SMEM));
            assert_eq!(
                u64::from(quant.grid.x) * u64::from(quant.block.x) * 4,
                (u64::from(rows) * 2_560).next_multiple_of(4 * u64::from(MXFP8_A8_QUANT_THREADS)),
                "four values a thread over every activation"
            );
        }
        // A step or a verify keeps the weight-only kernels under the option.
        for rows in [1, 5] {
            let p = plan(DType::Mxfp8E4M3G32, 2_560, 12_288, rows, lowp).expect("a step");
            assert!(matches!(p, Projection::Mxfp8 { .. }), "{p:?}");
        }
        // A contraction the W8A8 kernel cannot stride keeps the weight-only
        // prompt kernel, at every width.
        let p = plan(DType::Mxfp8E4M3G32, 2_592, 640, 333, prompt).expect("a prompt chunk");
        let Projection::Mxfp8 { entry, .. } = p else {
            panic!("{p:?}");
        };
        assert_eq!(entry.name(), "q4e_mxfp8_gemm_kernel");
        // And the projection is two launches: the quantizer, then the GEMM,
        // through the half-precision staging.
        let weight = WeightRef::layer(
            crate::sparse::bind::LayerRef::target(0),
            crate::sparse::bind::Role::AttnQuery,
        );
        let bound = Bound::new(
            weight,
            Operand::Scratch(Buf::Residual),
            Operand::Scratch(Buf::Embed),
        );
        let mut out = Plan::new();
        project(
            &mut out,
            WeightDesc::new(DType::Mxfp8E4M3G32, 2_560, 12_288),
            bound,
            333,
            prompt,
        )
        .expect("a prompt chunk");
        let names: Vec<_> = out
            .steps()
            .iter()
            .filter_map(Step::entry)
            .map(|e| e.name())
            .collect();
        assert_eq!(
            names,
            ["q4e_mxfp8_a8_quant_kernel", "q4e_mxfp8_a8_gemm_kernel"]
        );
    }

    #[test]
    fn a_wide_mxfp8_verify_runs_in_passes_and_a_prompt_in_one_launch() {
        let weight = WeightRef::layer(
            crate::sparse::bind::LayerRef::target(0),
            crate::sparse::bind::Role::AttnQuery,
        );
        let bound = Bound::new(
            weight,
            Operand::Scratch(Buf::Residual),
            Operand::Scratch(Buf::Embed),
        );
        let desc = WeightDesc::new(DType::Mxfp8E4M3G32, 2_560, 12_288);
        // Twenty rows: two full passes and a four-row tail.
        let mut out = Plan::new();
        project(&mut out, desc, bound, 20, GPU).expect("a batched verify");
        assert_eq!(out.len(), 3);
        let names: Vec<_> = out
            .steps()
            .iter()
            .filter_map(Step::entry)
            .map(|e| e.name())
            .collect();
        assert_eq!(names, ["q4e_mxfp8_rows_kernel"; 3]);
        let mut out = Plan::new();
        let prompt = Device {
            role: PassRole::Prompt { snapshot: None },
            ..GPU
        };
        project(&mut out, desc, bound, 333, prompt).expect("a prompt chunk");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn a_zero_extent_is_a_shape_error_rather_than_a_launch_of_nothing() {
        let err = plan(DType::F32, 0, 16, 1, GPU).expect_err("a zero contraction");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }), "{err:?}");
        let err = plan(DType::F32, 16, 16, 0, GPU).expect_err("a zero row count");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }), "{err:?}");
    }
}
