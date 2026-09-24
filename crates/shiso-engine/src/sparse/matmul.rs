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
    /// An MLX affine weight, read in place: one simdgroup an output row, one
    /// activation row a launch at decode and [`AFFINE_TOKENS`] a threadgroup
    /// over a chunk.
    Affine {
        /// The projection itself.
        entry: Entry,
        /// Its launch.
        launch: Launch,
    },
}

/// Output rows one affine projection threadgroup covers
/// (`Q4E_AFFINE_ROWS`).
pub const AFFINE_ROWS: u32 = 8;
/// Activation rows one multi-row affine threadgroup carries
/// (`Q4E_AFFINE_TOKENS`).
pub const AFFINE_TOKENS: u32 = 8;

/// The affine tier: the one-row kernel for a single activation row and the
/// multi-row one for a chunk, over the same grid of output rows.
fn affine(in_dim: u32, out_dim: u32, rows: u32) -> CoreResult<Projection> {
    if crate::sparse::weights::affine_bytes(DType::MlxAffine4Gs64, in_dim, out_dim).is_none() {
        return Err(CoreError::ShapeMismatch {
            field: "affine contraction width",
            expected: 64,
            got: u64::from(in_dim),
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
            32 * AFFINE_ROWS,
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
        DType::BF16 => brain(in_dim, out_dim, rows),
        DType::Exl3K4 | DType::Exl3K5 | DType::Exl3K6 => {
            trellis(dtype, in_dim, out_dim, rows, device)
        }
        DType::MlxAffine4Gs64 => affine(in_dim, out_dim, rows),
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
fn brain(in_dim: u32, out_dim: u32, rows: u32) -> CoreResult<Projection> {
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
        DType::BF16 => rows > MAX_ROWS,
        _ => false,
    };
    if passes {
        let width = if matches!(weight.dtype, DType::F16 | DType::F16Pack12) {
            HALF_MAX_ROWS
        } else {
            MAX_ROWS
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
    fn a_brain_float_projection_runs_one_row_and_splits_a_chunk_into_passes() {
        for want in [1, 4, MAX_ROWS] {
            let plan = brain(2_560, 128, want).expect("the built set covers it");
            let Projection::BrainRows { entry, rows, .. } = plan else {
                panic!("{plan:?}");
            };
            assert_eq!(rows, want);
            assert_eq!(entry.name(), "q4e_matmul_bf16_rows_kernel");
        }
        let err = brain(2_560, 128, MAX_ROWS + 1).expect_err("one launch is at most the width");
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
