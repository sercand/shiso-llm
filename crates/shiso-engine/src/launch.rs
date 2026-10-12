//! Every kernel the forward pass dispatches, described once.
//!
//! A kernel is named, argued and launched exactly as its module records it. The
//! qwen35 entries come from `shiso-metal/metal/qwen35/KERNELS.md`; the generic
//! entries come from the argument block the kernel itself declares. Nothing here
//! invents a name, rounds a grid into a shape a kernel was not written for, or
//! shares an argument block between two kernels that merely have the same fields.
//!
//! # Why this is a file of its own
//!
//! It is the only place a kernel name is spelled, so "does the engine dispatch a
//! kernel that does not exist" is one grep against `KERNELS.md` rather than a
//! survey of six call sites.

#![expect(
    dead_code,
    reason = "this module describes every kernel the stack can dispatch; the paths a given checkpoint does not take are still part of the description"
)]

use bytemuck::{Pod, Zeroable};

use shiso_core::Extent3;
use shiso_core::backend::{Backend, FnConst, FnConstValue, KernelArg, KernelRef, LaunchArgs};
use shiso_core::dtype::DType;

/// One tensor argument in a kernel signature: its element type and its shape.
///
/// A signature is the list of these a kernel takes, in parameter order. Two
/// signatures are interchangeable only when they agree entry for entry, because
/// a kernel locates the bytes it reads by position and shape rather than by
/// name: a mismatch would reinterpret live activations instead of missing a
/// cache, so a mismatch is a bug to report and never a fallback to take.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct TensorDesc {
    /// Element type of the tensor.
    pub dtype: DType,
    /// Shape of the tensor. An empty shape describes a scalar.
    pub shape: Vec<i64>,
}

impl TensorDesc {
    /// Elements the shape covers: the product of its dimensions, where an empty
    /// shape counts as one. A negative dimension is read as zero, so a
    /// malformed descriptor covers nothing instead of wrapping.
    #[must_use]
    pub(crate) fn num_elements(&self) -> usize {
        self.shape
            .iter()
            .map(|&dim| usize::try_from(dim.max(0)).unwrap_or(usize::MAX))
            .fold(1usize, usize::saturating_mul)
    }

    /// Blocks the tensor occupies. A quantized format packs
    /// [`DType::block_elems`] elements into [`DType::block_bytes`], and a
    /// partial tail block still costs a whole one.
    #[must_use]
    pub(crate) fn num_blocks(&self) -> usize {
        let elems = usize::try_from(self.dtype.block_elems().get()).unwrap_or(1);
        self.num_elements().div_ceil(elems)
    }

    /// Bytes the tensor occupies, counted in whole blocks. The wide quantized
    /// formats do not have a whole number of bytes per element, so multiplying
    /// elements by a per-element size would round the wrong way on every one of
    /// them.
    #[must_use]
    pub(crate) fn num_bytes(&self) -> usize {
        let block = usize::try_from(self.dtype.block_bytes().get()).unwrap_or(usize::MAX);
        self.num_blocks().saturating_mul(block)
    }
}

use shiso_core::error::{CoreError, CoreResult};

/// Function-constant index of the simdgroup count in the extended-matvec family.
const EXT_NSG_INDEX: u16 = 600;

/// Function-constant index of the per-lane input split in the same family.
const EXT_NXPSG_INDEX: u16 = 601;

/// Simdgroups per extended-matvec threadgroup, as the engine dispatches them.
///
/// Held in the width the launch arithmetic needs; the function constant takes
/// it as an `i16` at [`ext_consts`], which is the one place the sign matters.
///
/// # This number is spelled three ways and only two of them are here
///
/// Anyone raising it - the 2026-09-21 decode-throughput plan proposed eight,
/// and `docs/handoff/cuda-decode-perf.md` §5 records why the measurement said
/// not to - has to move all three together:
///
/// 1. this constant, which [`ext_block`] and [`ext_rows_per_threadgroup`] read,
///    and so the grid and the rows-per-block arithmetic with it;
/// 2. the **three** `FnConstValue::I16(2)` literals below, one each in
///    `EXT_CONSTS_16`, `EXT_CONSTS_8` and `EXT_CONSTS_4`, which repeat it
///    rather than read it - [`ext_consts`] already derives from the constant,
///    so this file spells the number two ways today. A grep for `I16(2)` finds
///    a fourth, in `PLAIN_CONSTS[1]`: that one is the plain family's
///    simdgroup count at its second entry and has nothing to do with this
///    constant. Counting it as a fourth spelling of `EXT_NSG` would have a
///    raise edit a table that must not move;
/// 3. `SHISO_EXT_NSG` in `kernels/cuda/generic/mul_mv_ext.cuh`.
///
/// **On CUDA the third is the only one the kernel obeys.** That backend
/// specializes by entry name and ignores the function constant entirely -
/// `SHISO_EXT_NSG` is a `#define` resolved when the module is compiled - so
/// moving this constant alone changes the grid and the block extent while the
/// kernel keeps mapping `blockIdx.x` through the old product. The launch
/// succeeds. Some output rows are then written twice and others never, which
/// is a wrong answer at full speed and not a crash.
pub(crate) const EXT_NSG: u32 = 2;

/// Lanes in one simdgroup, which is also the block width of the row kernels.
pub const WARP: u32 = 32;

/// Threads per threadgroup of the linear generic kernels.
pub const BLOCK_LINEAR: u32 = 256;

/// Threadgroup memory the norm reduction needs: one f32 per lane.
pub(crate) const NORM_SMEM_BYTES: u32 = WARP * 4;

/// Threads per threadgroup of the qwen35 head kernels.
pub const BLOCK_128: u32 = 128;

/// Bytes of one f32 activation element.
pub const F32_BYTES: u64 = 4;

/// Module the qwen35 kernels are registered under.
pub const Q35: &str = "qwen35";

/// The module the copied generic kernels are registered under.
pub const GENERIC: &str = "generic";

/// The `qwen35moe` module: the two codecs `modelopt` writes and the routed
/// feed-forward that reads them.
///
/// Its own image rather than entries in [`Q35`], because the two families
/// share their recurrent and attention kernels - which this family dispatches
/// through that module, unchanged - and share nothing else. On this backend a
/// module name is the key of a separately compiled image, so an entry the
/// engine asks for under this name has to be in an image of this name.
pub const Q35MOE: &str = "qwen35moe";

/// Module the parametric dense kernels are registered under.
///
/// The set a checkpoint that is not qwen35 dispatches: the same widths arrive as
/// arguments rather than as constants, so one entry point serves every model of
/// that shape. The module is a label rather than a lookup key — one flat library,
/// so the entry point already names the function — but a launch that records
/// which set its kernel came from is one that can be audited against the kernel
/// set that was compiled in.
pub const DENSE: &str = "dense";

/// Module the rotated-checkpoint kernels are registered under: the blockwise
/// rotation and the matmuls over ternary and 2-bit affine weights.
pub const TERNARY: &str = "ternary";

/// Arguments of `kernel_ternary_mv_*`, mirroring `ternary_args_mv`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TernaryMvArgs {
    /// Inputs a row.
    pub in_dim: u32,
    /// Output rows.
    pub out_dim: u32,
    /// Activation rows.
    pub n_tok: u32,
    /// Interleaved contraction slices one threadgroup carries.
    pub splits: u32,
}

/// Arguments of `kernel_ternary_get_rows_*`, mirroring `ternary_args_rows`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TernaryRowsArgs {
    /// Elements a row.
    pub in_dim: u32,
    /// Rows gathered.
    pub n_tok: u32,
}

/// Output rows one ternary matvec threadgroup covers.
pub const TERNARY_ROWS: u32 = 8;

/// The suffix the ternary entries spell for `dtype`, or `None` for a format
/// that set does not read.
#[must_use]
pub const fn ternary_tag(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::Pq2_0 => Some("pq2_0"),
        DType::Ptq1_0 => Some("ptq1_0"),
        DType::MlxAffine2Gs128 => Some("mlx2"),
        _ => None,
    }
}

/// `kernel_ternary_mv_<tag>_f32_t1`: the one-row matvec.
///
/// Built at one token a launch only. Two rows and up go to
/// [`ternary_mmv_name`], which decodes each weight once for every row, so a
/// token-blocked matvec would be compiled and never dispatched.
#[must_use]
pub const fn ternary_mv_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::Pq2_0 => Some("kernel_ternary_mv_pq2_0_f32_t1"),
        DType::Ptq1_0 => Some("kernel_ternary_mv_ptq1_0_f32_t1"),
        DType::MlxAffine2Gs128 => Some("kernel_ternary_mv_mlx2_f32_t1"),
        _ => None,
    }
}

/// Tokens one threadgroup of the prompt-width ternary matmul covers.
pub const TERNARY_MM_TOKENS: u32 = 32;

/// Output rows one threadgroup of either ternary matmul covers.
pub const TERNARY_MM_ROWS: u32 = 32;

/// `kernel_ternary_mmv_<tag>_f32`: two to eight rows, the weights decoded
/// once and multiplied against every token at once.
#[must_use]
pub const fn ternary_mmv_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::Pq2_0 => Some("kernel_ternary_mmv_pq2_0_f32"),
        DType::Ptq1_0 => Some("kernel_ternary_mmv_ptq1_0_f32"),
        DType::MlxAffine2Gs128 => Some("kernel_ternary_mmv_mlx2_f32"),
        _ => None,
    }
}

/// `kernel_ternary_mm_<tag>_f32_n32`: a prompt chunk, 32 tokens a tile.
#[must_use]
pub const fn ternary_mm_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::Pq2_0 => Some("kernel_ternary_mm_pq2_0_f32_n32"),
        DType::Ptq1_0 => Some("kernel_ternary_mm_ptq1_0_f32_n32"),
        DType::MlxAffine2Gs128 => Some("kernel_ternary_mm_mlx2_f32_n32"),
        _ => None,
    }
}

/// `kernel_ternary_get_rows_<tag>_f32`.
#[must_use]
pub const fn ternary_rows_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::Pq2_0 => Some("kernel_ternary_get_rows_pq2_0_f32"),
        DType::Ptq1_0 => Some("kernel_ternary_get_rows_ptq1_0_f32"),
        DType::MlxAffine2Gs128 => Some("kernel_ternary_get_rows_mlx2_f32"),
        _ => None,
    }
}

/// Contraction slices a ternary matvec threadgroup carries for a weight of
/// this shape: doubled while the grid is under 256 threadgroups and the
/// contraction still has a 512-input sweep for every slice, at most eight.
/// A function of the shape alone, so a row sums the same way at every pass
/// width.
#[must_use]
pub const fn ternary_splits(in_dim: u32, out_dim: u32) -> u32 {
    let tgs = out_dim.div_ceil(TERNARY_ROWS);
    let sweeps = in_dim.div_ceil(512);
    let mut s = 1;
    while tgs * s < 256 && s * 2 <= 8 && sweeps >= s * 2 {
        s *= 2;
    }
    s
}

/// Module the vision tower's kernels are registered under.
///
/// The engine dispatches exactly one of them: the splice that overwrites a
/// placeholder token's embedding row with the row the tower produced for it.
/// The rest of that set belongs to the encoder crate, which is where the tower
/// runs.
pub(crate) const VISION: &str = "vision";

/// Threadgroup memory of the recurrent kernel.
///
/// There is deliberately no lane count beside it. How many threads a recurrent
/// threadgroup carries is a register-budget question the device answers - a
/// constant here asked 512 of a part that allows 448, and every launch of the
/// block was refused - so the width comes from the family's own shape table,
/// which measures it, and a future launch site has nothing to reach for here.
/// The memory is not the same kind of number: it is bound by the launch and
/// fixed by the kernel source, so it stays a constant.
pub const RECURRENT_SMEM_BYTES: u32 = 1_156;

/// Threadgroup memory of the chunked recurrence, on the same footing.
pub const CHUNK_SMEM_BYTES: u32 = 16_896;

/// Tokens one chunk run covers.
pub const GDN_CHUNK_RUN: u32 = 64;

/// The shortest chunk the chunked recurrence is worth running at all.
///
/// Below this the triangular solve costs more than the token-at-a-time walk it
/// replaces, and the two paths would disagree by more than the walk's own
/// rounding on such a short span.
pub const GDN_CHUNK_MIN: u32 = 64;

/// Tokens per attention tile in the prefill kernel.
pub const ATTN_PREFILL_TILE: u32 = 16;

/// Threadgroup memory of a 128-lane head kernel: the block reduction's slots.
///
/// A block reduction writes one partial per lane plus the fold of the last
/// simdgroup, so 33 slots is the whole requirement and a taller request would
/// only ask the driver for memory no lane touches.
pub const HEAD_SMEM_BYTES: u32 = 132;

/// Threadgroup memory of the two norm-and-rotary kernels.
///
/// Both hold one head of the mix in threadgroup memory on top of the reduction's
/// slots, which is what the rotary pass reads and rewrites in place.
pub const NORM_ROPE_SMEM_BYTES: u32 = 1_156;

/// Epsilon inside the recurrent block's per-head mean square.
///
/// The normalization is a plain root-mean-square over one head with the weight
/// applied raw, so this epsilon is the only additive guard the head has; it is a
/// fixed part of the block, not a value the checkpoint carries.
pub const GDN_NORM_EPS: f32 = 1e-6;

/// Threadgroup memory of the decode attention kernel.
pub const ATTN_DECODE_SMEM_BYTES: u32 = 8_256;

/// Threadgroup memory of the prefill attention kernel.
pub const ATTN_PREFILL_SMEM_BYTES: u32 = 16_384;

/// The widest head the dense attention set stages.
///
/// `DENSE_ATTN_HD` in `metal/dense/dense_common.h`. Every threadgroup
/// allocation below is sized by it rather than by a layer's own `head_dim`,
/// because the kernels index their staging rows by the constant; a layer whose
/// head is wider would read and write past the end of memory that was allocated
/// correctly for the constant. A kernel cannot refuse that, so
/// [`crate::dense::DenseKernels`] refuses the shape on the host.
pub const DENSE_ATTN_HEAD_DIM: u32 = 512;

/// Threadgroup memory of the dense set's two norm-and-rotate kernels.
///
/// `4 * (DENSE_BLOCK_SUM_SLOTS + DENSE_ATTN_HD)`: the block reduction's 33 slots
/// plus the widest head the set stages, which is what the rotary pass reads and
/// rewrites in place. Sized by the compile-time stride rather than by the layer's
/// own `head_dim`, because the kernel indexes the staging row by a constant and
/// a launch that asked for less would be a launch the widest layer overruns.
pub const DENSE_NORM_ROPE_SMEM_BYTES: u32 = 4 * (33 + DENSE_ATTN_HEAD_DIM);

/// Threadgroup memory of the dense decode softmax.
///
/// One accumulator row per simdgroup - `8 * 512` floats - plus the two 8-slot
/// partial arrays the order-independent merge reads.
pub const DENSE_ATTN_DECODE_SMEM_BYTES: u32 = 8 * DENSE_ATTN_HEAD_DIM * 4 + 2 * 8 * 4;

/// Tokens per query tile in the dense prefill softmax.
///
/// Also the threshold: at or above one tile the prefill entry is dispatched, and
/// below it the decode entry, which would otherwise stage a key tile for tokens a
/// single decode threadgroup covers in one pass.
pub const DENSE_ATTN_PREFILL_TILE: u32 = 16;

/// Keys per tile of the dense prefill softmax, one per lane of a simdgroup's low
/// half.
pub const DENSE_ATTN_KT: u32 = 16;

/// Threadgroup memory of the dense prefill softmax: the transposed key tile as
/// `half`, at the widest head the set carries.
pub const DENSE_ATTN_PREFILL_SMEM_BYTES: u32 = DENSE_ATTN_HEAD_DIM * DENSE_ATTN_KT * 2;

/// Weights in one per-layer embedding row, which is one `q6_K` block exactly.
///
/// `DENSE_PLE_DIM` in `metal/dense/dense_common.h`. Not a width the shape may
/// choose: the prepare kernel stages a decoded row of this size in threadgroup
/// memory and the row IS a block, so a table of another width has no kernel here
/// and is refused on the host.
pub const DENSE_PLE_DIM: u32 = 256;

/// Pins the restated width to `DType::Q6_K`'s own: the two are written as
/// separate literals above rather than one derived from the other, because the
/// kernel-facing constant has to be a plain `u32` a `FnConstValue` can carry,
/// so this assertion is what keeps a change to the format's block width from
/// silently drifting out of step with the width the kernel was sized for.
const _: () = assert!(DENSE_PLE_DIM == DType::Q6_K.block_elems().get());

/// Bytes one `q6_K` block occupies: `ql[128]`, `qh[64]`, `scales[16]`, `d`.
///
/// One per layer slot per token id is the whole of a per-layer embedding table's
/// row, so this is also the stride the streaming reader and the prepare kernel
/// both derive a row offset from.
pub const DENSE_PLE_BLOCK_BYTES: u64 = 210;

/// Pins the restated stride to `DType::Q6_K`'s own, the same way
/// `DENSE_PLE_DIM` is pinned to its block width above.
const _: () = assert!(DENSE_PLE_BLOCK_BYTES == DType::Q6_K.block_bytes().get() as u64);

/// Threadgroup memory of the per-layer prepare kernel.
///
/// `4 * (DENSE_BLOCK_SUM_SLOTS + DENSE_PLE_DIM)`: the block reduction's 33 slots,
/// then the decoded row staged behind them. The row is staged because one `q6_K`
/// decode call produces sixteen weights, so decoding per element would throw
/// fifteen sixteenths of the unpacking away.
pub const DENSE_PLE_SMEM_BYTES: u32 = 4 * (33 + DENSE_PLE_DIM);

/// Elements in one `q6_K` block, which is the span a gather threadgroup unpacks.
pub(crate) const QK_K: u32 = 256;

/// Threads per block of the `q6_K` gather.
pub const GET_ROWS_BLOCK: u32 = 256;

/// Tile width of the prefill GEMM, in output rows.
const GEMM_TILE_ROWS: u32 = 64;

/// Tile height of the same GEMM, in tokens.
const GEMM_TILE_TOKENS: u32 = 32;

/// Threads one prefill GEMM threadgroup runs: four simdgroups.
pub const GEMM_BLOCK: u32 = 128;

/// Threadgroup memory the prefill GEMM asks for.
///
/// Two claims on the same allocation, and the larger one wins. Staging the tiles
/// needs 6144 bytes: a 64-by-32 `f16` weight tile at offset zero and a 32-by-32
/// `f16` activation tile behind it. The epilogue that writes a *partial* tile
/// reuses the same bytes as `float`, and its furthest write lands at element
/// 2047 of that view - the last simdgroup's base of 1056, plus its last 8-by-8
/// store at stride 64 - so it needs 8192. A full tile never takes that path,
/// which is why the device fixture passes with less; a projection whose output
/// width is not a whole number of 64 rows does take it, and this model has two.
pub(crate) const GEMM_SMEM_BYTES: u32 = 8_192;

/// The prefill GEMM's two function constants.
///
/// The indices are the kernel's own (`FC_MUL_MM` is 700), not the matvec family's:
/// a pipeline built from a function whose required constants are unset does not
/// fail politely, it aborts the process, so a wrong index here is not a wrong
/// answer but a crash.
///
/// They are not a matched pair, despite sitting next to each other.
///
/// 700 guards the *input* tail. False, because every input width this model
/// projects from is a whole number of the kernel's 32-wide K tile, and true would
/// select a load path that reads the weight without dequantizing it - which is not
/// a tail guard at all for a block format.
///
/// 701 guards the *output* tile, and it must be true. With it false the kernel
/// always takes its unguarded store and writes a full 64-by-32 tile whatever the
/// destination's real extent is. A chunk of five tokens has 5 rows, so every
/// launch overruns; the narrowest projection here is 32 wide, where the kernel
/// wrote 2048 floats into a 160-float buffer. The damage lands in whatever was
/// allocated next, which is why the symptom was a forward pass that gave a
/// different answer on every run rather than a crash.
const GEMM_CONSTANTS: [FnConst; 2] = [
    FnConst {
        index: 700,
        value: FnConstValue::Bool(false),
    },
    FnConst {
        index: 701,
        value: FnConstValue::Bool(true),
    },
];

/// The prefill GEMM's specialization, which is the same table for every launch.
#[must_use]
pub const fn mm_consts() -> &'static [FnConst] {
    &GEMM_CONSTANTS
}

// ---------------------------------------------------------------------------------------
// Argument blocks, in the field order each kernel declares.
// ---------------------------------------------------------------------------------------

/// The normalization kernel's argument block (`metal/generic/norm.metal`).
///
/// The `nef`/`nbf` triples describe the normalization weight as a broadcast
/// tensor: `nef1` of ones means every activation row multiplies by the same
/// weight row, which is what a qwen35 norm weight is.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct NormArgs {
    /// Elements in one activation row.
    pub ne00: i32,
    /// Elements per weight row, equal to [`NormArgs::ne00`] for a qwen35 norm.
    pub ne00_t: i32,
    /// Bytes between activation rows.
    pub nb1: u64,
    /// Stride between activation planes.
    pub nb2: u64,
    /// Its successor.
    pub nb3: u64,
    /// Epsilon added inside the mean square.
    pub eps: f32,
    /// Weight extent on the three broadcast axes: `1` on the row axis makes one
    /// weight row serve every activation row.
    pub nef1: [i32; 3],
    /// Its second axis.
    pub nef2: [i32; 3],
    /// And third.
    pub nef3: [i32; 3],
    /// Weight strides on the first axis.
    pub nbf1: [u64; 3],
    /// And on the second.
    pub nbf2: [u64; 3],
    /// And on the third.
    pub nbf3: [u64; 3],
}

const _: () = assert!(size_of::<NormArgs>() == 144);

/// The gate-unit kernel kernel's argument block (`metal/generic/glu.metal`).
///
/// `kernel_swiglu_flat_f32` walks `ne00` elements with no row structure at all, so
/// the two sources must be laid out alike for one flat index to name the same
/// element in both. A block's gate and up projections are laid out alike, at
/// `[n_tok][intermediate]`, which is what lets one flat index name one element in
/// both.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GluArgs {
    /// Elements the activation covers: tokens times the inner width.
    pub ne00: i32,
    /// The four bytes the declared layout leaves before the first stride.
    pub pad0: [u8; 4],
    /// Bytes between rows of the gating source.
    pub nb01: u64,
    /// Width of one row of the up source.
    pub ne10: i32,
    /// The four bytes the declared layout leaves before the next stride.
    pub pad1: [u8; 4],
    /// Bytes between its rows.
    pub nb11: u64,
    /// Elements the activation covers.
    pub ne0: i32,
    /// The four bytes the declared layout leaves before the output stride.
    pub pad2: [u8; 4],
    /// Bytes between output rows.
    pub nb1: u64,
    /// Element offset of the gating source.
    pub i00: i32,
    /// Element offset of the up source.
    pub i10: i32,
    /// Multiplier applied to the product.
    pub alpha: f32,
    /// Symmetric clamp, or zero for none.
    pub limit: f32,
}

const _: () = assert!(size_of::<GluArgs>() == 64);

/// The row-scale kernel's argument block (`metal/dense/dense_misc.metal`).
///
/// Flat: the kernel takes its element index straight from the grid, so `n` is the
/// whole run the caller wants multiplied and not a row width it strides by.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ScaleArgs {
    /// Elements the scale covers; anything past it is left alone.
    pub n: u32,
    /// The multiplier.
    pub s: f32,
}

const _: () = assert!(size_of::<ScaleArgs>() == 8);

/// The softcap kernel's argument block (`metal/dense/dense_misc.metal`).
///
/// Flat, for the same reason [`ScaleArgs`] is: the logits row the cap applies to
/// is one contiguous run.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct SoftcapArgs {
    /// Elements the cap covers.
    pub n: u32,
    /// The cap itself: `cap * tanh(x / cap)`, which is `x` near zero and bounded
    /// by `cap` at either tail.
    pub cap: f32,
}

const _: () = assert!(size_of::<SoftcapArgs>() == 8);

/// The extended matvec kernel's argument block (`metal/generic/mul_mv_ext.metal`), shared by the
/// `q4_K`, `q8_0`, `q5_K`, `q6_K` and `f16` instantiations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct MulMvExtArgs {
    /// Elements in one weight row: the projection's input width.
    pub ne00: i32,
    /// Weight rows, which is the projection's output width.
    pub ne01: i32,
    /// Broadcast planes: one for a two-dimensional weight.
    pub ne02: i32,
    /// Four bytes the C layout leaves before the stride run.
    pub pad0: i32,
    /// Bytes per weight element, which for a block codec is one whole block.
    pub nb00: u64,
    /// Bytes per weight row.
    pub nb01: u64,
    /// Stride between weight planes.
    pub nb02: u64,
    /// Its successor, equal to [`MulMvExtArgs::nb02`] at one plane.
    pub nb03: u64,
    /// Elements per activation row.
    pub ne10: i32,
    /// Activation rows.
    pub ne11: i32,
    /// Activation planes.
    pub ne12: i32,
    /// Four bytes the C layout leaves before the activation strides.
    pub pad1: i32,
    /// Bytes per activation element.
    pub nb10: u64,
    /// Bytes per activation row.
    pub nb11: u64,
    /// Stride between activation planes.
    pub nb12: u64,
    /// Its successor.
    pub nb13: u64,
    /// Elements per output row.
    pub ne0: i32,
    /// Output rows.
    pub ne1: i32,
    /// Broadcast ratio on the second axis: one for a plain product.
    pub r2: i16,
    /// And on the third.
    pub r3: i16,
    /// Trailing padding to the struct's eight-byte alignment.
    pub pad2: [u8; 4],
}

const _: () = assert!(size_of::<MulMvExtArgs>() == 112);

/// The prefill GEMM kernel's argument block (`metal/qwen35/gemm_q4k.metal`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct MulMmArgs {
    /// Elements per weight row.
    pub ne00: i32,
    /// Weight planes.
    pub ne02: i32,
    /// Bytes per weight row.
    pub nb01: u64,
    /// Stride between weight planes.
    pub nb02: u64,
    /// Its successor.
    pub nb03: u64,
    /// Activation planes.
    pub ne12: i32,
    /// Four bytes the C layout leaves before the activation strides.
    pub pad0: i32,
    /// Bytes per activation element.
    pub nb10: u64,
    /// Bytes per activation row.
    pub nb11: u64,
    /// Stride between activation planes.
    pub nb12: u64,
    /// Its successor.
    pub nb13: u64,
    /// Elements per output row.
    pub ne0: i32,
    /// Output rows.
    pub ne1: i32,
    /// Broadcast ratio on the second axis.
    pub r2: i16,
    /// And on the third.
    pub r3: i16,
    /// Trailing padding to the struct's eight-byte alignment.
    pub pad2: [i8; 4],
}

const _: () = assert!(size_of::<MulMmArgs>() == 88);

/// The staging kernel's argument block (`kernels/cuda/generic/dequant_plane.cuh`).
///
/// Three fields and no strides beyond the row: the destination is a dense f32
/// plane this engine allocates rather than a tensor read out of a file, so
/// there is nothing to describe about it that `ne00` does not already say.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct DequantPlaneArgs {
    /// Elements per weight row.
    pub ne00: i32,
    /// Weight rows.
    pub ne01: i32,
    /// Bytes per weight row, which for a block codec is a whole number of
    /// blocks.
    pub nb01: u64,
}

const _: () = assert!(size_of::<DequantPlaneArgs>() == 16);

/// The plain (single-token) matvec argument block
/// (`metal/generic/mul_mv_q8_0.metal`), shared by the `f32`, `f16` and `q8_0`
/// instantiations.
///
/// The codecs differ; the host-side description of "a weight matrix, an
/// activation row, an output row" does not. This is a different struct from
/// [`MulMvArgs`], not a prefix of it: the extended family carries the
/// input-split and row-blocking fields the plain one has no use for, so the two
/// are never interchangeable at a launch.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct MulMvArgs {
    /// Elements per weight row.
    pub ne00: i32,
    /// Weight rows.
    pub ne01: i32,
    /// Weight planes.
    pub ne02: i32,
    /// The four bytes the declared layout leaves before the first stride.
    pub pad0: [u8; 4],
    /// Bytes per weight element.
    pub nb00: u64,
    /// Bytes between weight rows.
    pub nb01: u64,
    /// Stride between weight planes.
    pub nb02: u64,
    /// Its successor.
    pub nb03: u64,
    /// Elements per activation row.
    pub ne10: i32,
    /// Activation rows.
    pub ne11: i32,
    /// Activation planes.
    pub ne12: i32,
    /// The four bytes the layout leaves before the activation strides.
    pub pad1: [u8; 4],
    /// Bytes per activation element.
    pub nb10: u64,
    /// Bytes per activation row.
    pub nb11: u64,
    /// Stride between activation planes.
    pub nb12: u64,
    /// Its successor.
    pub nb13: u64,
    /// Elements per output row.
    pub ne0: i32,
    /// Output rows.
    pub ne1: i32,
    /// Output rows one threadgroup produces. The dense kernel reads this back as
    /// a switch over its own instantiations rather than as a template parameter,
    /// so it is a launch argument here and not a name suffix.
    pub nr0: i32,
    /// Broadcast ratio on the second axis: one for a plain product.
    pub r2: i16,
    /// And on the third.
    pub r3: i16,
}

const _: () = assert!(size_of::<MulMvArgs>() == 112);

/// The row-gather kernel's argument block (`metal/generic/get_rows.metal`), which the
/// `q6_K` gather shares: the codec differs, the host-side description does not.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct GetRowsArgs {
    /// Elements in one vocabulary row.
    pub n_embd: i32,
    /// Rows the id may name.
    pub n_vocab: i32,
    /// Tokens the launch covers.
    pub n_tokens: i32,
    /// Four bytes the C layout leaves before the stride run.
    pub pad0: i32,
    /// Bytes per vocabulary row of the source.
    pub src_row_bytes: u64,
    /// Bytes per output row, in f32.
    pub dst_row_bytes: u64,
    /// Bytes between ids in the id row.
    pub token_stride: u64,
}

const _: () = assert!(size_of::<GetRowsArgs>() == 40);

/// `q35_args_gdn_conv`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ConvArgs {
    /// Channels the mix carries.
    pub channels: u32,
    /// Width of the causal window, including the token being written.
    pub conv_kernel: u32,
    /// Tokens in the chunk.
    pub n_tok: u32,
}

const _: () = assert!(size_of::<ConvArgs>() == 12);

/// `q35_args_gdn_l2norm`, run twice per chunk: once over the query half of the
/// mix and once over the key half, told apart only by [`L2NormArgs::offset`].
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct L2NormArgs {
    /// Width of one head.
    pub head_dim: u32,
    /// Heads this half of the mix carries.
    pub n_heads: u32,
    /// Floats between mix rows.
    pub stride: u32,
    /// Where this half starts inside a row.
    pub offset: u32,
}

const _: () = assert!(size_of::<L2NormArgs>() == 16);

/// `q35_args_gdn_gates`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GateArgs {
    /// Value heads, which is what the gates are per.
    pub n_head: u32,
    /// Tokens in the chunk.
    pub n_tok: u32,
}

const _: () = assert!(size_of::<GateArgs>() == 8);

/// `q35_args_gdn_out_gate`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OutGateArgs {
    /// Width of one head.
    pub head_dim: u32,
    /// Value heads.
    pub n_head: u32,
    /// Epsilon inside the head normalization.
    pub eps: f32,
}

const _: () = assert!(size_of::<OutGateArgs>() == 12);

/// `q35_args_gdn_recurrent`, which the chunked form shares field for field.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RecurrentArgs {
    /// Width of one head.
    pub head_dim: u32,
    /// Key/query heads.
    pub n_head_k: u32,
    /// Value heads.
    pub n_head_v: u32,
    /// Floats to the key half of a mix row.
    pub k_offset: u32,
    /// Floats to the value half.
    pub v_offset: u32,
    /// Floats between mix rows.
    pub stride: u32,
    /// Tokens this launch walks.
    pub n_tok: u32,
}

const _: () = assert!(size_of::<RecurrentArgs>() == 28);

/// `q35_args_q_norm_rope`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct QNormRopeArgs {
    /// Width of one query head.
    pub head_dim: u32,
    /// Query heads.
    pub n_head: u32,
    /// Leading dimensions rotary is applied to.
    pub n_rot: u32,
    /// Tokens in the chunk.
    pub n_tok: u32,
    /// Rotary frequency base.
    pub rope_base: f32,
    /// Epsilon inside the per-head mean square.
    pub eps: f32,
}

const _: () = assert!(size_of::<QNormRopeArgs>() == 24);

/// `q35_args_k_norm_rope`, which also writes the key/value cache.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct KNormRopeArgs {
    /// Width of one key head.
    pub head_dim: u32,
    /// Key/value heads.
    pub n_head_kv: u32,
    /// Leading dimensions rotary is applied to.
    pub n_rot: u32,
    /// Tokens in the chunk.
    pub n_tok: u32,
    /// Rotary frequency base.
    pub rope_base: f32,
    /// Epsilon inside the per-head mean square.
    pub eps: f32,
    /// Positions per cache page.
    pub page_tokens: u32,
    /// Entries between one row's page table and the next, or `0` when every
    /// row of this launch reads the table the buffer names.
    ///
    /// A batched decode step carries one row per session and two sessions
    /// share no page, so a launch that read one table for every row would
    /// answer every row past the first out of row zero's conversation -
    /// fluently, at the right shape, through launches that all succeed. Zero
    /// is the shared case rather than a sentinel, because `pages + t * 0` is
    /// `pages` for every row.
    pub page_stride: u32,
}

const _: () = assert!(size_of::<KNormRopeArgs>() == 32);

/// `q35_args_attn_decode` and `q35_args_attn_prefill`, which are one block.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AttnArgs {
    /// Width of one head.
    pub head_dim: u32,
    /// Query heads.
    pub n_head: u32,
    /// Key/value heads.
    pub n_head_kv: u32,
    /// Tokens in the chunk.
    pub n_tok: u32,
    /// Positions per cache page.
    pub page_tokens: u32,
    /// Entries between one row's page table and the next, or `0` when every
    /// row of this launch reads the table the buffer names.
    ///
    /// A batched decode step carries one row per session and two sessions
    /// share no page, so a launch that read one table for every row would
    /// answer every row past the first out of row zero's conversation -
    /// fluently, at the right shape, through launches that all succeed. Zero
    /// is the shared case rather than a sentinel, because `pages + t * 0` is
    /// `pages` for every row.
    pub page_stride: u32,
}

const _: () = assert!(size_of::<AttnArgs>() == 24);

/// `dense_args_q_norm_rope` (`metal/dense/dense_norm_rope.metal`).
///
/// The rotated width and the frequency width are separate fields because they
/// are separate questions: a layer whose ladder is derived over the whole head
/// but only whose leading slice rotates pairs `i` with `i + freq_dims / 2`, not
/// with `i + n_rot / 2`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DenseQNormRopeArgs {
    /// Width of one query head.
    pub head_dim: u32,
    /// Query heads.
    pub n_head: u32,
    /// Leading dimensions rotary is applied to.
    pub n_rot: u32,
    /// Width the frequency ladder is computed over.
    pub freq_dims: u32,
    /// Tokens in the chunk.
    pub n_tok: u32,
    /// Rotary frequency base.
    pub rope_base: f32,
    /// Epsilon inside the per-head mean square.
    pub eps: f32,
}

const _: () = assert!(size_of::<DenseQNormRopeArgs>() == 28);

/// `dense_args_kv_norm_rope`, which also writes both halves of the cache.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DenseKvNormRopeArgs {
    /// Width of one key head.
    pub head_dim: u32,
    /// Key/value heads.
    pub n_head_kv: u32,
    /// Leading dimensions rotary is applied to.
    pub n_rot: u32,
    /// Width the frequency ladder is computed over.
    pub freq_dims: u32,
    /// Tokens in the chunk.
    pub n_tok: u32,
    /// Rotary frequency base.
    pub rope_base: f32,
    /// Epsilon inside the per-head mean square.
    pub eps: f32,
    /// Positions per cache page.
    pub page_tokens: u32,
    /// Non-zero when the value is normalized before it is cached. The
    /// normalization carries no weight, which is the whole of the difference
    /// between the two halves of a cache row.
    pub v_norm: u32,
    /// Entries between one row's page table and the next, or `0` when every
    /// row of this launch reads the table the buffer names.
    ///
    /// A batched decode step carries one row per session and two sessions
    /// share no page, so a launch that read one table for every row would
    /// answer every row past the first out of row zero's conversation -
    /// fluently, at the right shape, through launches that all succeed. Zero
    /// is the shared case rather than a sentinel, because `pages + t * 0` is
    /// `pages` for every row.
    pub page_stride: u32,
}

const _: () = assert!(size_of::<DenseKvNormRopeArgs>() == 40);

/// `dense_args_attn`, which both dense softmax entry points read.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DenseAttnArgs {
    /// Width of one head.
    pub head_dim: u32,
    /// Query heads.
    pub n_head: u32,
    /// Key/value heads.
    pub n_head_kv: u32,
    /// Tokens in the chunk.
    pub n_tok: u32,
    /// Positions per cache page.
    pub page_tokens: u32,
    /// Positions a query may look back over; `0` is no window at all.
    pub window: u32,
    /// The number every score is multiplied by, which this family's layers carry
    /// rather than deriving from the head width.
    pub scale: f32,
    /// Entries between one row's page table and the next, or `0` when every
    /// row of this launch reads the table the buffer names.
    ///
    /// A batched decode step carries one row per session and two sessions
    /// share no page, so a launch that read one table for every row would
    /// answer every row past the first out of row zero's conversation -
    /// fluently, at the right shape, through launches that all succeed. Zero
    /// is the shared case rather than a sentinel, because `pages + t * 0` is
    /// `pages` for every row.
    pub page_stride: u32,
}

const _: () = assert!(size_of::<DenseAttnArgs>() == 32);

/// `dense_args_ple`: the per-layer embedding mix, one threadgroup per
/// (token, layer slot).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DensePleArgs {
    /// Tokens in the chunk.
    pub n_tok: u32,
    /// Layer slots, which is how many `dim`-wide spans one token's row carries.
    pub n_layers: u32,
    /// Weights in one slot's row, which must be [`DENSE_PLE_DIM`].
    pub dim: u32,
    /// Bytes one token's WHOLE row occupies: one `q6_K` block per slot. A row
    /// identifier becomes an offset through this, so a stride one block out
    /// reads every row after the first from the middle of its predecessor.
    pub row_bytes: u32,
    /// Non-zero when the rows buffer is the whole table and the token id picks
    /// the row; zero when it is the chunk's own rows in token order.
    pub resident: u32,
    /// Epsilon inside the projection's mean square.
    pub eps: f32,
    /// What the projection is multiplied by BEFORE it is normalized, so the
    /// scale reaches the magnitudes the normalization divides by as well as the
    /// value it then multiplies.
    pub proj_scale: f32,
    /// What the dequantized embedding is multiplied by.
    pub embed_scale: f32,
    /// What their sum is multiplied by.
    pub mix_scale: f32,
}

const _: () = assert!(size_of::<DensePleArgs>() == 36);

/// `dense_args_ple_gate`: the gate multiply at one layer slot.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DensePleGateArgs {
    /// Tokens in the chunk.
    pub n_tok: u32,
    /// Layer slots the prepared block carries, which is its row stride.
    pub n_layers: u32,
    /// Weights in one slot's row.
    pub dim: u32,
    /// Which slot of that block this launch reads.
    pub layer: u32,
}

const _: () = assert!(size_of::<DensePleGateArgs>() == 16);

/// `dense_args_vit_rope2d`: per-head `RMSNorm` of the three attention planes and
/// the two-axis rotation of the query and the key, for the Gemma 4 vision tower
/// (`metal/dense/dense_vision.metal`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DenseVitRope2dArgs {
    /// Patch rows in the item.
    pub rows: u32,
    /// Attention heads.
    pub heads: u32,
    /// Weights in one head.
    pub head_dim: u32,
    /// Columns in the patch grid: a row's column and row are `row % grid_w` and
    /// `row / grid_w`, and the two halves of the head rotate by those two.
    pub grid_w: u32,
    /// Epsilon inside the per-head mean square.
    pub eps: f32,
    /// Base of the rotation's frequency ladder.
    pub rope_base: f32,
}

const _: () = assert!(size_of::<DenseVitRope2dArgs>() == 24);

/// `dense_args_vit_pool`: the `3 x 3` average pool that folds a patch block into
/// one token, scaled by a constant (`metal/dense/dense_vision.metal`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DenseVitPoolArgs {
    /// Patch rows before pooling.
    pub grid_h: u32,
    /// Patch columns before pooling.
    pub grid_w: u32,
    /// Weights in one row.
    pub hidden: u32,
    /// What the mean is multiplied by (`sqrt(hidden)` in the reference).
    pub scale: f32,
}

const _: () = assert!(size_of::<DenseVitPoolArgs>() == 16);

/// `dense_args_clamp`: the elementwise clamp every clamped vision projection is
/// built from (`metal/dense/dense_vision.metal`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ClampArgs {
    /// Elements the clamp covers.
    pub count: u32,
    /// Lower bound.
    pub min: f32,
    /// Upper bound.
    pub max: f32,
}

const _: () = assert!(size_of::<ClampArgs>() == 12);

/// `dense_args_affine`: projection dimensions, token count and bit width for
/// MLX affine matvec and GEMM kernels (`metal/dense/dense_affine.metal`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DenseAffineArgs {
    /// Input dimension (columns of weight, elements of x).
    pub in_dim: u32,
    /// Output dimension (rows of weight, elements of y).
    pub out_dim: u32,
    /// Token count.
    pub n_tok: u32,
    /// Quantization bits (4 or 8).
    pub bits: u32,
}

const _: () = assert!(size_of::<DenseAffineArgs>() == 16);

/// `dense_args_embed_affine`: hidden width and token count for
/// MLX affine token embedding gather (`metal/dense/dense_affine_rows.metal`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DenseEmbedAffineArgs {
    /// Hidden dimension (columns of table).
    pub hidden: u32,
    /// Token count.
    pub n_tok: u32,
}

const _: () = assert!(size_of::<DenseEmbedAffineArgs>() == 8);

/// The generic set's elementwise-add argument block
/// (`metal/generic/bin.metal`), which the two- and three-operand adds share.
///
/// One field, and it is the element count: both kernels take their index
/// straight from the grid and neither strides a row, so there is nothing else
/// for the block to carry.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AddArgs {
    /// Elements the add covers; anything past it is left alone.
    pub n: u32,
}

const _: () = assert!(size_of::<AddArgs>() == 4);

/// `vit_args_splice`: the seam between the vision tower and the residual stream.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct SpliceArgs {
    /// Token rows in the chunk, which is the destination row count.
    pub rows: u32,
    /// Residual-stream width.
    pub hidden: u32,
    /// Rows in the encoder block, so an index past it is refused rather than
    /// read.
    pub src_rows: u32,
}

/// `q35_args_attn_gate`: the sigmoid gate applied to attention's output before
/// the output projection.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AttnGateArgs {
    /// Width of one head.
    pub head_dim: u32,
    /// Query heads.
    pub n_head: u32,
    /// Tokens in the chunk.
    pub n_tok: u32,
}

const _: () = assert!(size_of::<AttnGateArgs>() == 12);

// ---------------------------------------------------------------------------------------
// Extended-matvec dispatch math.
// ---------------------------------------------------------------------------------------

/// Input elements each lane takes a turn on.
///
/// A width divisible by 256 with fewer than three tokens gets the widest split;
/// below that the extra lanes cost more in bookkeeping than the shorter run saves,
/// so the same weight is read two different ways depending only on how many
/// tokens are in flight.
#[must_use]
pub(crate) const fn ext_nxpsg(in_dim: u64, n_tok: u32) -> u32 {
    if in_dim.is_multiple_of(256) && n_tok < 3 {
        16
    } else if in_dim.is_multiple_of(128) {
        8
    } else {
        4
    }
}

/// Token rows one extended-matvec threadgroup takes.
///
/// The family is instantiated at one to five rows and the mapping is a table, not
/// a formula: seven and eight tokens go four at a time, because no instantiation
/// exists at those widths, and so does any wider batch, which is the remainder a
/// prefill chunk leaves.
#[must_use]
pub(crate) const fn ext_r1ptg(n_tok: u32) -> u32 {
    match n_tok {
        0 | 1 => 1,
        2 => 2,
        3 | 6 => 3,
        5 => 5,
        _ => 4,
    }
}

/// Output rows one extended-matvec threadgroup produces.
///
/// The lanes of a simdgroup split the input by [`ext_nxpsg`], so a group covers
/// `32 / nxpsg` rows at a time and every simdgroup in the block works on its own
/// set of the same rows.
#[must_use]
pub(crate) const fn ext_rows_per_threadgroup(nxpsg: u32) -> u32 {
    WARP / nxpsg * EXT_NSG
}

/// The specialization every extended-matvec launch carries.
///
/// Both counts are `u32` everywhere else, and the driver wants them signed:
/// a lane count that does not fit `i16` is a bug in this file, not an input,
/// so it says so rather than wrapping.
#[must_use]
pub(crate) fn ext_consts(nxpsg: u32) -> [FnConst; 2] {
    [
        FnConst {
            index: EXT_NSG_INDEX,
            value: FnConstValue::I16(i16::try_from(EXT_NSG).expect("simdgroup count fits i16")),
        },
        FnConst {
            index: EXT_NXPSG_INDEX,
            value: FnConstValue::I16(i16::try_from(nxpsg).expect("lane split fits i16")),
        },
    ]
}

/// The `q4_K` matvec instantiations, indexed by rows per threadgroup minus one.
const EXT_Q4_K: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_q4_K_f32_r1_1"),
    Some("kernel_mul_mv_ext_q4_K_f32_r1_2"),
    Some("kernel_mul_mv_ext_q4_K_f32_r1_3"),
    Some("kernel_mul_mv_ext_q4_K_f32_r1_4"),
    Some("kernel_mul_mv_ext_q4_K_f32_r1_5"),
];

/// The `q4_0` matvec instantiations, which is how every Gemma 4 linear layer is
/// read: that checkpoint stores all of them in this format and none in any
/// other.
const EXT_Q4_0: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_q4_0_f32_r1_1"),
    Some("kernel_mul_mv_ext_q4_0_f32_r1_2"),
    Some("kernel_mul_mv_ext_q4_0_f32_r1_3"),
    Some("kernel_mul_mv_ext_q4_0_f32_r1_4"),
    Some("kernel_mul_mv_ext_q4_0_f32_r1_5"),
];

/// `q8_0` has no single-row instantiation: one row goes through the plain
/// `kernel_mul_mv_q8_0_f32`, which is a different kernel with a different block.
const EXT_Q8_0: [Option<&str>; 5] = [
    None,
    Some("kernel_mul_mv_ext_q8_0_f32_r1_2"),
    Some("kernel_mul_mv_ext_q8_0_f32_r1_3"),
    Some("kernel_mul_mv_ext_q8_0_f32_r1_4"),
    Some("kernel_mul_mv_ext_q8_0_f32_r1_5"),
];

/// Added by the qwen35 kernel set, which is what lets a `q5_K` projection run
/// with no host-side repack.
const EXT_Q5_K: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_q5_K_f32_r1_1"),
    Some("kernel_mul_mv_ext_q5_K_f32_r1_2"),
    Some("kernel_mul_mv_ext_q5_K_f32_r1_3"),
    Some("kernel_mul_mv_ext_q5_K_f32_r1_4"),
    Some("kernel_mul_mv_ext_q5_K_f32_r1_5"),
];

/// The embedding's format, and the tied output projection's.
const EXT_Q6_K: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_q6_K_f32_r1_1"),
    Some("kernel_mul_mv_ext_q6_K_f32_r1_2"),
    Some("kernel_mul_mv_ext_q6_K_f32_r1_3"),
    Some("kernel_mul_mv_ext_q6_K_f32_r1_4"),
    Some("kernel_mul_mv_ext_q6_K_f32_r1_5"),
];

/// The `q2_K` instantiations, the widest of the sub-4-bit formats the Saluki
/// export mixes.
const EXT_Q2_K: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_q2_K_f32_r1_1"),
    Some("kernel_mul_mv_ext_q2_K_f32_r1_2"),
    Some("kernel_mul_mv_ext_q2_K_f32_r1_3"),
    Some("kernel_mul_mv_ext_q2_K_f32_r1_4"),
    Some("kernel_mul_mv_ext_q2_K_f32_r1_5"),
];

/// The `iq2_xxs` instantiations.
const EXT_IQ2_XXS: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_iq2_xxs_f32_r1_1"),
    Some("kernel_mul_mv_ext_iq2_xxs_f32_r1_2"),
    Some("kernel_mul_mv_ext_iq2_xxs_f32_r1_3"),
    Some("kernel_mul_mv_ext_iq2_xxs_f32_r1_4"),
    Some("kernel_mul_mv_ext_iq2_xxs_f32_r1_5"),
];

/// The `iq2_xs` instantiations.
const EXT_IQ2_XS: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_iq2_xs_f32_r1_1"),
    Some("kernel_mul_mv_ext_iq2_xs_f32_r1_2"),
    Some("kernel_mul_mv_ext_iq2_xs_f32_r1_3"),
    Some("kernel_mul_mv_ext_iq2_xs_f32_r1_4"),
    Some("kernel_mul_mv_ext_iq2_xs_f32_r1_5"),
];

/// The `iq2_s` instantiations.
const EXT_IQ2_S: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_iq2_s_f32_r1_1"),
    Some("kernel_mul_mv_ext_iq2_s_f32_r1_2"),
    Some("kernel_mul_mv_ext_iq2_s_f32_r1_3"),
    Some("kernel_mul_mv_ext_iq2_s_f32_r1_4"),
    Some("kernel_mul_mv_ext_iq2_s_f32_r1_5"),
];

/// The `iq3_xxs` instantiations.
const EXT_IQ3_XXS: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_iq3_xxs_f32_r1_1"),
    Some("kernel_mul_mv_ext_iq3_xxs_f32_r1_2"),
    Some("kernel_mul_mv_ext_iq3_xxs_f32_r1_3"),
    Some("kernel_mul_mv_ext_iq3_xxs_f32_r1_4"),
    Some("kernel_mul_mv_ext_iq3_xxs_f32_r1_5"),
];

/// The `iq3_s` instantiations.
const EXT_IQ3_S: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_iq3_s_f32_r1_1"),
    Some("kernel_mul_mv_ext_iq3_s_f32_r1_2"),
    Some("kernel_mul_mv_ext_iq3_s_f32_r1_3"),
    Some("kernel_mul_mv_ext_iq3_s_f32_r1_4"),
    Some("kernel_mul_mv_ext_iq3_s_f32_r1_5"),
];

/// The `iq1_s` instantiations, the narrowest format the export stores a
/// projection in.
const EXT_IQ1_S: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_iq1_s_f32_r1_1"),
    Some("kernel_mul_mv_ext_iq1_s_f32_r1_2"),
    Some("kernel_mul_mv_ext_iq1_s_f32_r1_3"),
    Some("kernel_mul_mv_ext_iq1_s_f32_r1_4"),
    Some("kernel_mul_mv_ext_iq1_s_f32_r1_5"),
];

/// The `iq1_m` instantiations, which the export pairs with `iq1_s` at the same
/// role width: a gate and an up projection of one shape in two codecs, which is
/// why each dispatches off its own table rather than a shared one.
const EXT_IQ1_M: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_iq1_m_f32_r1_1"),
    Some("kernel_mul_mv_ext_iq1_m_f32_r1_2"),
    Some("kernel_mul_mv_ext_iq1_m_f32_r1_3"),
    Some("kernel_mul_mv_ext_iq1_m_f32_r1_4"),
    Some("kernel_mul_mv_ext_iq1_m_f32_r1_5"),
];

/// The `iq4_xs` instantiations, the 4-bit end of the mixed set.
const EXT_IQ4_XS: [Option<&str>; 5] = [
    Some("kernel_mul_mv_ext_iq4_xs_f32_r1_1"),
    Some("kernel_mul_mv_ext_iq4_xs_f32_r1_2"),
    Some("kernel_mul_mv_ext_iq4_xs_f32_r1_3"),
    Some("kernel_mul_mv_ext_iq4_xs_f32_r1_4"),
    Some("kernel_mul_mv_ext_iq4_xs_f32_r1_5"),
];

/// One plain-matvec launch: the entry point plus the geometry it was compiled
/// for.
///
/// The plain family is what a single token falls back to when the extended
/// family has no instantiation at one row, which is exactly the decode case for
/// the formats whose extended table starts at two rows. Its geometry is not the
/// extended one: the block is the same `(32, nsg)` shape, but the rows are
/// divided by [`PlainMv::rows_per_threadgroup`] on `x` with the token on `y`
/// and the batch on `z` rather than by the two-axis product the extended
/// family uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlainMv {
    /// Entry point.
    pub(crate) name: &'static str,
    /// Simdgroups per threadgroup.
    pub(crate) nsg: u32,
    /// Output rows one threadgroup produces, which is the grid divisor.
    pub(crate) rows_per_threadgroup: u32,
    /// Threadgroup memory the lane reduction needs.
    pub(crate) smem_bytes: u32,
}

/// Simdgroups per threadgroup of the dense plain matvec.
///
/// One simdgroup per 128 input elements, capped at eight: past that the split
/// costs more in the cross-simdgroup reduction than the shorter run saves.
const PLAIN_NSG_CAP: u32 = 8;

/// Output rows one threadgroup produces, and the grid divisor, of every plain
/// instantiation that reduces an output row across its simdgroups.
///
/// Both families agree on two: the `q8_0` kernel fixes it at compile time and
/// the dense kernels switch on the `nr0` launch argument, whose only accepted
/// values are two and four.
const PLAIN_ROWS: u32 = 2;

/// Output rows one threadgroup produces for the short instantiations, where one
/// lane takes a whole output row and nothing is reduced across the block.
const PLAIN_SHORT_ROWS: u32 = 32;

/// Simdgroups per threadgroup of the `q8_0` plain matvec.
const Q8_MV_NSG: u32 = 4;

/// Threadgroup memory the plain-matvec reduction needs: two f32 per lane.
const PLAIN_SMEM_BYTES: u32 = WARP * 2 * 4;

/// The plain matvec for one weight format at one input width.
///
/// `None` means the format has no plain instantiation, which is the quantized
/// formats other than `q8_0`: those always have an extended instantiation at one
/// row, so they never need this family.
#[must_use]
pub(crate) fn plain_mv(dtype: DType, in_dim: u64) -> Option<PlainMv> {
    if dtype == DType::Q8_0 {
        return Some(PlainMv {
            name: "kernel_mul_mv_q8_0_f32",
            nsg: Q8_MV_NSG,
            rows_per_threadgroup: PLAIN_ROWS,
            smem_bytes: PLAIN_SMEM_BYTES,
        });
    }

    let wide = match dtype {
        DType::F32 => "kernel_mul_mv_f32_f32",
        DType::F16 => "kernel_mul_mv_f16_f32",
        _ => return None,
    };

    if in_dim < 32 {
        return Some(PlainMv {
            name: if dtype == DType::F32 {
                "kernel_mul_mv_f32_f32_short"
            } else {
                "kernel_mul_mv_f16_f32_short"
            },
            nsg: 1,
            rows_per_threadgroup: PLAIN_SHORT_ROWS,
            smem_bytes: 0,
        });
    }

    let nsg = u32::try_from(in_dim.div_ceil(128))
        .unwrap_or(PLAIN_NSG_CAP)
        .min(PLAIN_NSG_CAP);
    if in_dim.is_multiple_of(4) {
        return Some(PlainMv {
            name: if dtype == DType::F32 {
                "kernel_mul_mv_f32_f32_4"
            } else {
                "kernel_mul_mv_f16_f32_4"
            },
            nsg,
            rows_per_threadgroup: PLAIN_ROWS,
            smem_bytes: PLAIN_SMEM_BYTES,
        });
    }

    Some(PlainMv {
        name: wide,
        nsg,
        rows_per_threadgroup: PLAIN_ROWS,
        smem_bytes: PLAIN_SMEM_BYTES,
    })
}

/// Grid of a plain-matvec launch.
///
/// Every kernel in the family reads the same three axes: `x` picks a block of
/// output rows, `y` the token and `z` the batch. A single-token decode gets a
/// `y` extent of one rather than a missing axis, which is why the row count is
/// not the whole grid.
#[must_use]
pub(crate) const fn plain_mv_grid(
    out_dim: u32,
    rows_per_threadgroup: u32,
    n_tok: u32,
    n_batch: u32,
) -> Extent3 {
    Extent3::new(out_dim.div_ceil(rows_per_threadgroup), n_tok, n_batch)
}

/// Block of a plain-matvec launch: one warp per simdgroup.
#[must_use]
pub(crate) const fn plain_block(nsg: u32) -> Extent3 {
    Extent3::new(WARP, nsg, 1)
}

/// The specialization table of a plain matvec: the simdgroup count alone.
///
/// The plain kernel reads the same `nsg` constant the extended family reads and
/// has no use for the input-split constant, so only the first slot is filled.
#[must_use]
pub(crate) fn plain_consts(nsg: u32) -> [FnConst; 1] {
    [FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(i16::try_from(nsg).expect("simdgroup count fits i16")),
    }]
}

/// The matvec entry point for a weight format at a chosen row blocking.
///
/// `None` means the format has no instantiation at that width, which the caller
/// reports rather than works around: a projection silently skipped is a model
/// that answers confidently and wrongly.
///
/// The packed affine formats are absent on purpose. Their entry points take
/// [`DenseAffineArgs`] and five buffers rather than [`MulMvExtArgs`] and three,
/// so a name returned here would be launched against the wrong argument block.
/// [`crate::projection::projection`] routes them to their own launcher before
/// this table is ever consulted.
#[must_use]
pub const fn ext_name(dtype: DType, r1ptg: u32) -> Option<&'static str> {
    let table: [Option<&'static str>; 5] = match dtype {
        DType::Q4_K => EXT_Q4_K,
        DType::Q4_0 => EXT_Q4_0,
        DType::Q8_0 => EXT_Q8_0,
        DType::Q5_K => EXT_Q5_K,
        DType::Q6_K => EXT_Q6_K,
        // The mixed i-quant set of the Saluki export: every one of them is
        // instantiated at all five row blockings, so none of these tables has
        // a `None` slot the way `q8_0`'s does.
        DType::Q2_K => EXT_Q2_K,
        DType::IQ2_XXS => EXT_IQ2_XXS,
        DType::IQ2_XS => EXT_IQ2_XS,
        DType::IQ2_S => EXT_IQ2_S,
        DType::IQ3_XXS => EXT_IQ3_XXS,
        DType::IQ3_S => EXT_IQ3_S,
        DType::IQ1_S => EXT_IQ1_S,
        DType::IQ1_M => EXT_IQ1_M,
        DType::IQ4_XS => EXT_IQ4_XS,
        // `f16` has no extended instantiation on either backend, and a row here
        // named four kernels no library holds: every chunk of two to five tokens
        // over an `f16` weight failed in the driver. It takes the plain matvec
        // at every width below the GEMM, as the by-name backend always has.
        _ => return None,
    };
    if r1ptg == 0 || r1ptg > 5 {
        return None;
    }
    table[r1ptg as usize - 1]
}

/// The `q4_0` matvec instantiations a device that cannot specialize at launch
/// has to ask for by name, indexed by `[r1ptg - 1][nxpsg]`.
///
/// Fifteen names for what is one kernel on Metal. There is no choice about it:
/// a CUDA module holds one symbol per instantiation, so the lane split that is
/// a pipeline constant there is part of the symbol here. The three splits are
/// the three [`ext_nxpsg`] can return and no others; a fourth would be a name
/// this table does not hold and a `None` from [`ext_name_for`], which is the
/// answer a caller reports rather than works around.
const EXT_Q4_0_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_q4_0_f32_r1_1_nx4",
        "kernel_mul_mv_ext_q4_0_f32_r1_1_nx8",
        "kernel_mul_mv_ext_q4_0_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_0_f32_r1_2_nx4",
        "kernel_mul_mv_ext_q4_0_f32_r1_2_nx8",
        "kernel_mul_mv_ext_q4_0_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_0_f32_r1_3_nx4",
        "kernel_mul_mv_ext_q4_0_f32_r1_3_nx8",
        "kernel_mul_mv_ext_q4_0_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_0_f32_r1_4_nx4",
        "kernel_mul_mv_ext_q4_0_f32_r1_4_nx8",
        "kernel_mul_mv_ext_q4_0_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_0_f32_r1_5_nx4",
        "kernel_mul_mv_ext_q4_0_f32_r1_5_nx8",
        "kernel_mul_mv_ext_q4_0_f32_r1_5_nx16",
    ],
];

/// The `q6_K` instantiations, spelled the same way and for the same reason.
///
/// A second table rather than a second axis on the first: the two codecs are
/// reached through different modules ([`ext_module`] sends `q6_K` to [`Q35`]
/// and `q4_0` to [`GENERIC`]), and a single table indexed by format would
/// invite a caller to forget that.
///
/// Gemma 4 needs it because its output projection is the vocabulary table tied
/// back, and that table is `q6_K` in the GGUF container whatever the linear
/// layers are packed at. `q6_K` has no tiled product, so every width of that
/// product goes through this family rather than through a GEMM at the wide end.
const EXT_Q6_K_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_q6_K_f32_r1_1_nx4",
        "kernel_mul_mv_ext_q6_K_f32_r1_1_nx8",
        "kernel_mul_mv_ext_q6_K_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_q6_K_f32_r1_2_nx4",
        "kernel_mul_mv_ext_q6_K_f32_r1_2_nx8",
        "kernel_mul_mv_ext_q6_K_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_q6_K_f32_r1_3_nx4",
        "kernel_mul_mv_ext_q6_K_f32_r1_3_nx8",
        "kernel_mul_mv_ext_q6_K_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_q6_K_f32_r1_4_nx4",
        "kernel_mul_mv_ext_q6_K_f32_r1_4_nx8",
        "kernel_mul_mv_ext_q6_K_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_q6_K_f32_r1_5_nx4",
        "kernel_mul_mv_ext_q6_K_f32_r1_5_nx8",
        "kernel_mul_mv_ext_q6_K_f32_r1_5_nx16",
    ],
];

/// The `q4_K` matvec instantiations, indexed the same way.
///
/// The bulk of a qwen35 `Q4_K_M` container: 131 of its 426 tensors, every
/// linear layer of every block. Unlike `q5_K` this codec *does* have a tiled
/// product, so a prefill chunk of two tokens or more never reaches this table -
/// but a decode step is one token, and at one token
/// `crates/shiso-engine/src/projection.rs` takes the matvec. Without this table
/// `ext_name_for` answered `None` on a backend that specializes by name, the
/// projection fell through to `plain_projection`, and a decode refused with
/// "`Q4_K` has no plain matvec at this width" - a prefill that worked followed
/// by a first token that did not.
const EXT_Q4_K_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_q4_K_f32_r1_1_nx4",
        "kernel_mul_mv_ext_q4_K_f32_r1_1_nx8",
        "kernel_mul_mv_ext_q4_K_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_K_f32_r1_2_nx4",
        "kernel_mul_mv_ext_q4_K_f32_r1_2_nx8",
        "kernel_mul_mv_ext_q4_K_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_K_f32_r1_3_nx4",
        "kernel_mul_mv_ext_q4_K_f32_r1_3_nx8",
        "kernel_mul_mv_ext_q4_K_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_K_f32_r1_4_nx4",
        "kernel_mul_mv_ext_q4_K_f32_r1_4_nx8",
        "kernel_mul_mv_ext_q4_K_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_K_f32_r1_5_nx4",
        "kernel_mul_mv_ext_q4_K_f32_r1_5_nx8",
        "kernel_mul_mv_ext_q4_K_f32_r1_5_nx16",
    ],
];

/// The `q5_K` instantiations, spelled the same way and for the same reason.
///
/// A third table rather than a third axis, on the rule the `q6_K` one states:
/// [`ext_module`] sends this codec to [`Q35`] beside `q6_K`, and a single table
/// indexed by format would invite a caller to forget which module a name lives
/// in.
///
/// qwen35 needs it because a `Q4_K_M` container keeps 48 tensors at `q5_K`, and
/// `q5_K` has no tiled product - [`mm_name`] returns `None` for it - so every
/// width of that projection comes through this family rather than through a
/// GEMM at the wide end. Without this table `ext_name_for` answered `None` on a
/// backend that specializes by name, the projection fell through to
/// `plain_projection`, and the pass refused with "`Q5_K` has no plain matvec
/// at this width" - which named the family that could not serve it rather than
/// the one that should have.
const EXT_Q5_K_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_q5_K_f32_r1_1_nx4",
        "kernel_mul_mv_ext_q5_K_f32_r1_1_nx8",
        "kernel_mul_mv_ext_q5_K_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_q5_K_f32_r1_2_nx4",
        "kernel_mul_mv_ext_q5_K_f32_r1_2_nx8",
        "kernel_mul_mv_ext_q5_K_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_q5_K_f32_r1_3_nx4",
        "kernel_mul_mv_ext_q5_K_f32_r1_3_nx8",
        "kernel_mul_mv_ext_q5_K_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_q5_K_f32_r1_4_nx4",
        "kernel_mul_mv_ext_q5_K_f32_r1_4_nx8",
        "kernel_mul_mv_ext_q5_K_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_q5_K_f32_r1_5_nx4",
        "kernel_mul_mv_ext_q5_K_f32_r1_5_nx8",
        "kernel_mul_mv_ext_q5_K_f32_r1_5_nx16",
    ],
];

/// The `q2_K` instantiations, spelled the same way and for the same reason as
/// the four above: a CUDA module holds one symbol per instantiation, so the
/// lane split that is a function constant on Metal is part of the symbol there.
///
/// `q2_K` and the eight i-quant tables below are the mixed set of the Saluki
/// export. All nine decode through the `kq` walk - one whole 32-weight
/// sub-block a lane, the twin of `kernel_mul_mv_ext_kq_f32_disp` in
/// `crates/shiso-metal/metal/generic/mul_mv_ext.metal` - and all fifteen
/// `(r1ptg, nxpsg)` pairs are instantiated in
/// `kernels/cuda/generic/generic.cu`, because a pair that is merely absent is
/// a decode that resolves at one prompt width and refuses at another. Without
/// these tables [`ext_name_for`] answered `None` for them on a backend that
/// specializes by name, the projection fell through to `plain_projection`,
/// and the pass refused with "`Q2_K` has no plain matvec at this width".
const EXT_Q2_K_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_q2_K_f32_r1_1_nx4",
        "kernel_mul_mv_ext_q2_K_f32_r1_1_nx8",
        "kernel_mul_mv_ext_q2_K_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_q2_K_f32_r1_2_nx4",
        "kernel_mul_mv_ext_q2_K_f32_r1_2_nx8",
        "kernel_mul_mv_ext_q2_K_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_q2_K_f32_r1_3_nx4",
        "kernel_mul_mv_ext_q2_K_f32_r1_3_nx8",
        "kernel_mul_mv_ext_q2_K_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_q2_K_f32_r1_4_nx4",
        "kernel_mul_mv_ext_q2_K_f32_r1_4_nx8",
        "kernel_mul_mv_ext_q2_K_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_q2_K_f32_r1_5_nx4",
        "kernel_mul_mv_ext_q2_K_f32_r1_5_nx8",
        "kernel_mul_mv_ext_q2_K_f32_r1_5_nx16",
    ],
];

/// The `iq2_xxs` instantiations.
const EXT_IQ2_XXS_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_1_nx4",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_1_nx8",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_2_nx4",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_2_nx8",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_3_nx4",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_3_nx8",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_4_nx4",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_4_nx8",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_5_nx4",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_5_nx8",
        "kernel_mul_mv_ext_iq2_xxs_f32_r1_5_nx16",
    ],
];

/// The `iq2_xs` instantiations.
const EXT_IQ2_XS_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_iq2_xs_f32_r1_1_nx4",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_1_nx8",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_xs_f32_r1_2_nx4",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_2_nx8",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_xs_f32_r1_3_nx4",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_3_nx8",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_xs_f32_r1_4_nx4",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_4_nx8",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_xs_f32_r1_5_nx4",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_5_nx8",
        "kernel_mul_mv_ext_iq2_xs_f32_r1_5_nx16",
    ],
];

/// The `iq2_s` instantiations.
const EXT_IQ2_S_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_iq2_s_f32_r1_1_nx4",
        "kernel_mul_mv_ext_iq2_s_f32_r1_1_nx8",
        "kernel_mul_mv_ext_iq2_s_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_s_f32_r1_2_nx4",
        "kernel_mul_mv_ext_iq2_s_f32_r1_2_nx8",
        "kernel_mul_mv_ext_iq2_s_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_s_f32_r1_3_nx4",
        "kernel_mul_mv_ext_iq2_s_f32_r1_3_nx8",
        "kernel_mul_mv_ext_iq2_s_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_s_f32_r1_4_nx4",
        "kernel_mul_mv_ext_iq2_s_f32_r1_4_nx8",
        "kernel_mul_mv_ext_iq2_s_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq2_s_f32_r1_5_nx4",
        "kernel_mul_mv_ext_iq2_s_f32_r1_5_nx8",
        "kernel_mul_mv_ext_iq2_s_f32_r1_5_nx16",
    ],
];

/// The `iq3_xxs` instantiations.
const EXT_IQ3_XXS_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_1_nx4",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_1_nx8",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_2_nx4",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_2_nx8",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_3_nx4",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_3_nx8",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_4_nx4",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_4_nx8",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_5_nx4",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_5_nx8",
        "kernel_mul_mv_ext_iq3_xxs_f32_r1_5_nx16",
    ],
];

/// The `iq3_s` instantiations.
const EXT_IQ3_S_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_iq3_s_f32_r1_1_nx4",
        "kernel_mul_mv_ext_iq3_s_f32_r1_1_nx8",
        "kernel_mul_mv_ext_iq3_s_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq3_s_f32_r1_2_nx4",
        "kernel_mul_mv_ext_iq3_s_f32_r1_2_nx8",
        "kernel_mul_mv_ext_iq3_s_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq3_s_f32_r1_3_nx4",
        "kernel_mul_mv_ext_iq3_s_f32_r1_3_nx8",
        "kernel_mul_mv_ext_iq3_s_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq3_s_f32_r1_4_nx4",
        "kernel_mul_mv_ext_iq3_s_f32_r1_4_nx8",
        "kernel_mul_mv_ext_iq3_s_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq3_s_f32_r1_5_nx4",
        "kernel_mul_mv_ext_iq3_s_f32_r1_5_nx8",
        "kernel_mul_mv_ext_iq3_s_f32_r1_5_nx16",
    ],
];

/// The `iq1_s` instantiations.
const EXT_IQ1_S_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_iq1_s_f32_r1_1_nx4",
        "kernel_mul_mv_ext_iq1_s_f32_r1_1_nx8",
        "kernel_mul_mv_ext_iq1_s_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq1_s_f32_r1_2_nx4",
        "kernel_mul_mv_ext_iq1_s_f32_r1_2_nx8",
        "kernel_mul_mv_ext_iq1_s_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq1_s_f32_r1_3_nx4",
        "kernel_mul_mv_ext_iq1_s_f32_r1_3_nx8",
        "kernel_mul_mv_ext_iq1_s_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq1_s_f32_r1_4_nx4",
        "kernel_mul_mv_ext_iq1_s_f32_r1_4_nx8",
        "kernel_mul_mv_ext_iq1_s_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq1_s_f32_r1_5_nx4",
        "kernel_mul_mv_ext_iq1_s_f32_r1_5_nx8",
        "kernel_mul_mv_ext_iq1_s_f32_r1_5_nx16",
    ],
];

/// The `iq1_m` instantiations. The export also stores the vocabulary table at
/// `iq1_m`; that gather is a separate entry (`kernel_get_rows_iq1_m_f32`),
/// not this family.
const EXT_IQ1_M_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_iq1_m_f32_r1_1_nx4",
        "kernel_mul_mv_ext_iq1_m_f32_r1_1_nx8",
        "kernel_mul_mv_ext_iq1_m_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq1_m_f32_r1_2_nx4",
        "kernel_mul_mv_ext_iq1_m_f32_r1_2_nx8",
        "kernel_mul_mv_ext_iq1_m_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq1_m_f32_r1_3_nx4",
        "kernel_mul_mv_ext_iq1_m_f32_r1_3_nx8",
        "kernel_mul_mv_ext_iq1_m_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq1_m_f32_r1_4_nx4",
        "kernel_mul_mv_ext_iq1_m_f32_r1_4_nx8",
        "kernel_mul_mv_ext_iq1_m_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq1_m_f32_r1_5_nx4",
        "kernel_mul_mv_ext_iq1_m_f32_r1_5_nx8",
        "kernel_mul_mv_ext_iq1_m_f32_r1_5_nx16",
    ],
];

/// The `iq4_xs` instantiations, the 4-bit end of the mixed set.
const EXT_IQ4_XS_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_iq4_xs_f32_r1_1_nx4",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_1_nx8",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq4_xs_f32_r1_2_nx4",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_2_nx8",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq4_xs_f32_r1_3_nx4",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_3_nx8",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq4_xs_f32_r1_4_nx4",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_4_nx8",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_iq4_xs_f32_r1_5_nx4",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_5_nx8",
        "kernel_mul_mv_ext_iq4_xs_f32_r1_5_nx16",
    ],
];

/// Where a lane split sits in [`EXT_Q4_0_BY_NAME`]'s second axis.
///
/// `None` for a split the family is not built with, which is every value except
/// the three [`ext_nxpsg`] returns.
const fn nxpsg_slot(nxpsg: u32) -> Option<usize> {
    match nxpsg {
        4 => Some(0),
        8 => Some(1),
        16 => Some(2),
        _ => None,
    }
}

/// The matvec entry point, spelled the way `by_name` asks for.
///
/// `by_name` comes from [`shiso_core::Backend::specializes_by_name`] and is the
/// whole of the difference. `false` is [`ext_name`]: one name per row blocking,
/// with the lane split travelling beside it as a function constant. `true` folds
/// the split into the name, because the device it describes has one compiled
/// symbol per instantiation and refuses a constant at launch rather than
/// launching a variant the caller did not ask for.
///
/// Thirteen formats have the by-name spelling - the `q4_0` every Gemma 4 linear
/// layer is packed at, the `q6_K` its tied output head is, the `q4_K` and
/// `q5_K` a qwen35 `Q4_K_M` container keeps 131 and 48 tensors at, and the
/// nine mixed sub-4-bit formats of the Saluki export - and `None` for any
/// other is a deliberate refusal rather than an omission: the remaining
/// formats have no instantiation under that spelling anywhere in the tree,
/// and returning the unspecialized name would hand back an entry whose lane
/// split is whatever the module happened to be built with.
#[must_use]
pub const fn ext_name_for(
    dtype: DType,
    r1ptg: u32,
    nxpsg: u32,
    by_name: bool,
) -> Option<&'static str> {
    if !by_name {
        return ext_name(dtype, r1ptg);
    }
    if r1ptg == 0 || r1ptg > 5 {
        return None;
    }
    let table = match dtype {
        DType::Q4_0 => &EXT_Q4_0_BY_NAME,
        DType::Q4_K => &EXT_Q4_K_BY_NAME,
        DType::Q5_K => &EXT_Q5_K_BY_NAME,
        DType::Q6_K => &EXT_Q6_K_BY_NAME,
        DType::Q2_K => &EXT_Q2_K_BY_NAME,
        DType::IQ2_XXS => &EXT_IQ2_XXS_BY_NAME,
        DType::IQ2_XS => &EXT_IQ2_XS_BY_NAME,
        DType::IQ2_S => &EXT_IQ2_S_BY_NAME,
        DType::IQ3_XXS => &EXT_IQ3_XXS_BY_NAME,
        DType::IQ3_S => &EXT_IQ3_S_BY_NAME,
        DType::IQ1_S => &EXT_IQ1_S_BY_NAME,
        DType::IQ1_M => &EXT_IQ1_M_BY_NAME,
        DType::IQ4_XS => &EXT_IQ4_XS_BY_NAME,
        _ => return None,
    };
    match nxpsg_slot(nxpsg) {
        Some(slot) => Some(table[r1ptg as usize - 1][slot]),
        None => None,
    }
}

/// The prefill GEMM's specialization, for a backend that may not have one.
///
/// [`mm_consts`] is one fixed table - the input tail broadcast off, the output
/// tile guard on - and never varies with the launch, so a device that cannot
/// take a constant at launch compiles the one variant the engine ever asks for
/// and is handed nothing. The GEMM therefore needs no second spelling, which is
/// the difference between it and the matvec above.
#[must_use]
pub const fn mm_consts_for(by_name: bool) -> &'static [FnConst] {
    if by_name { &[] } else { &GEMM_CONSTANTS }
}

/// The plain matvec's specialization, for a backend that may not have one.
///
/// The plain family's only constant is the simdgroup count, and that count is
/// also the launch block's second extent ([`plain_block`]), so a device that
/// refuses a launch-time constant reads it off the launch instead and is handed
/// nothing. Same shape as [`mm_consts_for`] and for the same reason; the
/// difference from the matvec above is that nothing about the entry point
/// changes, only where the kernel reads one number from.
///
/// # Errors
///
/// `None` when `by_name` is false and `nsg` is past the family's cap, which is
/// the caller's error to report rather than a value to clamp.
#[must_use]
pub(crate) fn plain_consts_for(nsg: u32, by_name: bool) -> Option<&'static [FnConst]> {
    if by_name {
        return Some(&[]);
    }
    plain_consts_static(nsg)
}

/// The prefill GEMM for a weight format, when one exists.
///
/// Four formats have a GEMM. `q5_K` and `q6_K` do not, which is affordable
/// because those two are the embedding and the tied output rather than the bulk of
/// a block. The packed affine formats are absent for the reason given on
/// [`ext_name`].
#[must_use]
pub const fn mm_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::Q4_K => Some("kernel_mul_mm_q4_K_f32"),
        DType::Q4_0 => Some("kernel_mul_mm_q4_0_f32"),
        DType::Q8_0 => Some("kernel_mul_mm_q8_0_f32"),
        DType::F16 => Some("kernel_mul_mm_f16_f32"),
        _ => None,
    }
}

/// The entry that widens a whole weight plane of `dtype` to f32.
///
/// **Exactly the codecs [`mm_name`] answers for, and that is a requirement
/// rather than a coincidence.** `projection::mm_projection` asks the vendor
/// matrix library first and falls back to the tile, so a codec with a tile and
/// no stager would silently never take the faster path, and one with a stager
/// and no tile would have no fallback on a device with no matrix tier at all.
/// The two tables are read together and a divergence between them is invisible
/// at any single prompt length.
#[must_use]
pub const fn dequant_plane_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::Q4_K => Some("kernel_dequant_plane_q4_K_f32"),
        DType::Q4_0 => Some("kernel_dequant_plane_q4_0_f32"),
        DType::Q8_0 => Some("kernel_dequant_plane_q8_0_f32"),
        DType::F16 => Some("kernel_dequant_plane_f16_f32"),
        _ => None,
    }
}

/// Threads one staging block runs, mirroring `SHISO_DEQUANT_BLOCK`.
///
/// The kernel's stride loop is written in terms of this constant rather than
/// `blockDim.x`, so a launch that disagreed would leave part of every row
/// unwritten - and an unwritten element of a staged plane is not a zero, it is
/// whatever the previous tensor left in the slab.
pub(crate) const DEQUANT_BLOCK: u32 = 256;

/// Grid of a staging launch: one block per weight row.
#[must_use]
pub(crate) const fn dequant_plane_grid(out_dim: u32) -> Extent3 {
    Extent3::new(out_dim, 1, 1)
}

/// Grid of an extended-matvec launch: rows over the output width, tokens over the
/// row blocking.
#[must_use]
pub(crate) const fn ext_grid(out_dim: u32, n_tok: u32, nxpsg: u32, r1ptg: u32) -> Extent3 {
    Extent3::new(
        out_dim.div_ceil(ext_rows_per_threadgroup(nxpsg)),
        n_tok.div_ceil(r1ptg),
        1,
    )
}

/// Grid of a prefill GEMM: tiles of tokens across, output tiles down.
#[must_use]
pub(crate) const fn mm_grid(out_dim: u32, n_tok: u32) -> Extent3 {
    Extent3::new(
        n_tok.div_ceil(GEMM_TILE_TOKENS),
        out_dim.div_ceil(GEMM_TILE_ROWS),
        1,
    )
}

// ---------------------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------------------

/// Issues one launch.
///
/// # Errors
///
/// Whatever the backend reports for a launch it cannot encode or accept. The
/// error names the backend's own operation; [`launch_failure`] is what wraps it
/// in the name of the kernel that failed.
pub fn run(
    backend: &dyn Backend,
    kernel: KernelRef,
    grid: Extent3,
    block: Extent3,
    smem_bytes: u32,
    args: &[KernelArg<'_>],
) -> CoreResult<()> {
    backend.run(
        kernel,
        &LaunchArgs {
            grid,
            block,
            smem_bytes,
            args,
        },
    )
}

/// Reports a launch failure under the name of the kernel that failed.
#[must_use]
pub fn launch_failure(kernel: &'static str, source: CoreError) -> CoreError {
    CoreError::Backend {
        backend: "engine",
        op: kernel,
        source: Box::new(source),
    }
}

/// The module one extended-matvec instantiation was compiled from.
///
/// Only the `q5_K` and `q6_K` instantiations are added by the qwen35 kernel set;
/// every other format comes from the generic one. The module is a label rather
/// than a lookup key -- one flat library, so the entry point already names the
/// function -- but a launch that records where its kernel came from is one that
/// can be audited against the kernel set that was compiled in.
#[must_use]
pub(crate) const fn ext_module(dtype: DType) -> &'static str {
    match dtype {
        DType::Q5_K | DType::Q6_K => Q35,
        _ => GENERIC,
    }
}

/// Threads per threadgroup of an extended-matvec launch.
///
/// The kernel derives each lane's row from its simdgroup index, so the block has
/// to be exactly `EXT_NSG` simdgroups wide: a wider threadgroup would run lanes
/// the row arithmetic never accounts for and every output row would be written by
/// threads the grid does not expect.
#[must_use]
pub const fn ext_block() -> Extent3 {
    Extent3::new(WARP, EXT_NSG, 1)
}

/// The specialization of a 16-lane input split.
const EXT_CONSTS_16: [FnConst; 2] = [
    FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(2),
    },
    FnConst {
        index: EXT_NXPSG_INDEX,
        value: FnConstValue::I16(16),
    },
];

/// The specialization of an 8-lane input split.
const EXT_CONSTS_8: [FnConst; 2] = [
    FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(2),
    },
    FnConst {
        index: EXT_NXPSG_INDEX,
        value: FnConstValue::I16(8),
    },
];

/// The specialization of a 4-lane input split.
const EXT_CONSTS_4: [FnConst; 2] = [
    FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(2),
    },
    FnConst {
        index: EXT_NXPSG_INDEX,
        value: FnConstValue::I16(4),
    },
];

/// The specialization table of an extended-matvec launch, by input split.
///
/// [`KernelRef::consts`] is `&'static`, so a launch cannot carry a table built at
/// the call site: the three splits `ext_nxpsg` can return are the three tables
/// above. `the_static_tables_are_the_builders` fails if a literal there and
/// `ext_consts` ever disagree, which is what makes the literals safe to keep.
#[must_use]
pub const fn ext_consts_static(nxpsg: u32) -> Option<&'static [FnConst]> {
    match nxpsg {
        16 => Some(&EXT_CONSTS_16),
        8 => Some(&EXT_CONSTS_8),
        4 => Some(&EXT_CONSTS_4),
        _ => None,
    }
}

/// The specialization table of a plain matvec, indexed by simdgroup count.
///
/// Slot `n` is the table for `n` simdgroups, so slot zero is the count no
/// threadgroup has and the lookup below shifts its index instead of carrying a
/// table no threadgroup can be built with.
const PLAIN_CONSTS: [[FnConst; 1]; 8] = [
    [FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(1),
    }],
    [FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(2),
    }],
    [FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(3),
    }],
    [FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(4),
    }],
    [FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(5),
    }],
    [FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(6),
    }],
    [FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(7),
    }],
    [FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(8),
    }],
];

/// The plain family's table for a simdgroup count.
///
/// `None` for a count the family cannot be built with, which is a caller bug
/// rather than a fallback to take: [`plain_mv`] only ever returns a count the cap
/// allows, and a matvec launched with the wrong simdgroup count reads past its
/// own threadgroup.
#[must_use]
pub(crate) fn plain_consts_static(nsg: u32) -> Option<&'static [FnConst]> {
    let index = usize::try_from(nsg).ok()?.checked_sub(1)?;
    PLAIN_CONSTS.get(index).map(<[FnConst; 1]>::as_slice)
}
#[cfg(test)]
mod tests {
    use super::{
        EXT_NSG_INDEX, GENERIC, PLAIN_NSG_CAP, PLAIN_ROWS, PLAIN_SHORT_ROWS, PLAIN_SMEM_BYTES,
        TensorDesc, WARP, ext_module, ext_name, ext_name_for, plain_block, plain_consts, plain_mv,
        plain_mv_grid,
    };
    use shiso_core::backend::FnConstValue;
    use shiso_core::dtype::DType;
    fn desc(dtype: DType, shape: &[i64]) -> TensorDesc {
        TensorDesc {
            dtype,
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn elements_are_the_product_of_the_dimensions() {
        let row = desc(DType::F32, &[8, 2560]);
        assert_eq!(row.num_elements(), 8 * 2560);
        assert_eq!(
            row.num_bytes(),
            8 * 2560 * 4,
            "an f32 block is one element of four bytes, so blocks and elements agree"
        );
    }

    #[test]
    fn an_empty_shape_is_one_element_rather_than_none() {
        let scalar = desc(DType::F16, &[]);
        assert_eq!(scalar.num_elements(), 1);
        assert_eq!(scalar.num_bytes(), 2);
    }

    #[test]
    fn a_partial_tail_block_costs_a_whole_one() {
        // q4_K packs 256 elements into 144 bytes. A row of 300 elements is two
        // blocks: a per-element byte count would round it to one block and
        // under-allocate every quantized weight the loader sizes this way.
        let row = desc(DType::Q4_K, &[300]);
        assert_eq!(row.num_blocks(), 2);
        assert_eq!(row.num_bytes(), 288);
    }

    #[test]
    fn a_negative_dimension_covers_nothing() {
        let broken = desc(DType::Q8_0, &[-1, 4]);
        assert_eq!(broken.num_elements(), 0);
        assert_eq!(broken.num_blocks(), 0);
        assert_eq!(
            broken.num_bytes(),
            0,
            "a malformed descriptor cannot ask for a huge allocation"
        );
    }

    #[test]
    fn every_plain_matvec_row_blocking_is_two() {
        for dtype in [DType::F32, DType::F16] {
            for in_dim in [32u64, 2560, 2561] {
                let mv = plain_mv(dtype, in_dim).expect("a dense format has a plain matvec");
                assert_eq!(
                    mv.rows_per_threadgroup, PLAIN_ROWS,
                    "{dtype:?} at {in_dim} elements: the dense kernels switch on nr0, whose \
                     only accepted values are two and four"
                );
                assert_eq!(
                    mv.smem_bytes, PLAIN_SMEM_BYTES,
                    "{dtype:?} at {in_dim}: the reduction stages one f32 per lane per row"
                );
            }
        }
    }

    #[test]
    fn a_narrow_input_gives_each_lane_its_own_row() {
        let mv = plain_mv(DType::F32, 16).expect("a short row still has a plain matvec");
        assert_eq!(mv.name, "kernel_mul_mv_f32_f32_short");
        assert_eq!(mv.nsg, 1, "one lane per row needs no second simdgroup");
        assert_eq!(mv.rows_per_threadgroup, PLAIN_SHORT_ROWS);
        assert_eq!(
            mv.smem_bytes, 0,
            "a lane writes its own row, so nothing is staged across the block"
        );
    }

    #[test]
    fn a_width_divisible_by_four_takes_the_vectorized_kernel() {
        assert_eq!(
            plain_mv(DType::F32, 2560).expect("f32").name,
            "kernel_mul_mv_f32_f32_4"
        );
        assert_eq!(
            plain_mv(DType::F16, 2560).expect("f16").name,
            "kernel_mul_mv_f16_f32_4"
        );
        assert_eq!(
            plain_mv(DType::F32, 2561).expect("f32").name,
            "kernel_mul_mv_f32_f32"
        );
        assert_eq!(
            plain_mv(DType::F16, 2561).expect("f16").name,
            "kernel_mul_mv_f16_f32"
        );
    }

    #[test]
    fn simdgroups_track_the_input_and_then_stop() {
        assert_eq!(plain_mv(DType::F32, 128).expect("one group").nsg, 1);
        assert_eq!(plain_mv(DType::F32, 129).expect("rounded up").nsg, 2);
        assert_eq!(
            plain_mv(DType::F32, 1024).expect("at the cap").nsg,
            PLAIN_NSG_CAP
        );
        assert_eq!(
            plain_mv(DType::F32, u64::from(u32::MAX))
                .expect("a width too wide for a lane count")
                .nsg,
            PLAIN_NSG_CAP,
            "an absurd width takes the cap rather than wrapping"
        );
    }

    #[test]
    fn a_quantized_format_other_than_q8_0_has_no_plain_matvec() {
        for dtype in [DType::Q4_K, DType::Q5_K, DType::Q6_K] {
            assert!(
                plain_mv(dtype, 2560).is_none(),
                "{dtype:?} always has an extended instantiation at one row"
            );
        }
    }

    #[test]
    fn the_grid_keeps_the_token_and_batch_axes() {
        let one = plain_mv_grid(2560, PLAIN_ROWS, 1, 1);
        assert_eq!((one.x, one.y, one.z), (1280, 1, 1));
        let wide = plain_mv_grid(2561, PLAIN_ROWS, 3, 2);
        assert_eq!(
            (wide.x, wide.y, wide.z),
            (1281, 3, 2),
            "the kernel reads the token on y and the batch on z even at one token"
        );
        let block = plain_block(4);
        assert_eq!((block.x, block.y, block.z), (WARP, 4, 1));
    }

    #[test]
    fn a_plain_matvec_specializes_with_one_constant() {
        let consts = plain_consts(4);
        assert_eq!(
            consts.len(),
            1,
            "the plain family has no input-split constant"
        );
        assert_eq!(consts[0].index, EXT_NSG_INDEX);
        match consts[0].value {
            FnConstValue::I16(nsg) => assert_eq!(i32::from(nsg), 4),
            other => panic!("the plain family specializes with a short, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "simdgroup count fits i16")]
    fn a_simdgroup_count_that_cannot_be_a_short_is_a_bug() {
        assert!(plain_consts(u32::from(u16::MAX)).is_empty());
    }

    /// The nine mixed i-quant formats of the Saluki export, each with the tag
    /// its kernel entry point spells between `ext_` and `_f32`.
    ///
    /// The tags are written out rather than derived from [`DType::type_name`]
    /// because `q2_K` is the one format whose entry point does not spell the
    /// format the way the dtype names itself: a derivation would have renamed
    /// every `q2_K` entry to lowercase, and the miss would surface as a driver
    /// failure at launch rather than here.
    const SALUKI_MATVEC_TAGS: [(DType, &str); 9] = [
        (DType::Q2_K, "q2_K"),
        (DType::IQ2_XXS, "iq2_xxs"),
        (DType::IQ2_XS, "iq2_xs"),
        (DType::IQ2_S, "iq2_s"),
        (DType::IQ3_XXS, "iq3_xxs"),
        (DType::IQ3_S, "iq3_s"),
        (DType::IQ1_S, "iq1_s"),
        (DType::IQ1_M, "iq1_m"),
        (DType::IQ4_XS, "iq4_xs"),
    ];

    #[test]
    fn the_mixed_i_quants_dispatch_at_every_row_blocking() {
        for (dtype, tag) in SALUKI_MATVEC_TAGS {
            for r1ptg in 1..=5 {
                let want = format!("kernel_mul_mv_ext_{tag}_f32_r1_{r1ptg}");
                assert_eq!(
                    ext_name(dtype, r1ptg),
                    Some(want.as_str()),
                    "{dtype:?} at {r1ptg} rows per threadgroup must name the entry point \
                     the generic kernel set exports"
                );
            }
            assert_eq!(
                ext_name(dtype, 0),
                None,
                "{dtype:?} has no zero-row table slot"
            );
            assert_eq!(
                ext_name(dtype, 6),
                None,
                "{dtype:?} is not instantiated at six rows per threadgroup"
            );
            assert_eq!(
                ext_module(dtype),
                GENERIC,
                "{dtype:?} was compiled into the generic set, not the qwen35 one"
            );
        }
    }

    /// The same nine formats under the by-name spelling, at every pair the
    /// tables hold.
    ///
    /// The name is written from the tag rather than copied out of a table
    /// because that is the claim worth grading: the entry point is the
    /// constant spelling's name with `_nx4`, `_nx8` or `_nx16` appended, and
    /// nothing else changes. `ext_matvec_entries` then checks that each of
    /// these names has an instantiation in the CUDA generic set, which is
    /// what makes the pair of tables and sources a checked thing rather than
    /// two lists somebody keeps in their heads.
    #[test]
    fn the_mixed_i_quants_have_by_name_spellings() {
        for (dtype, tag) in SALUKI_MATVEC_TAGS {
            for r1ptg in 1..=5 {
                for nxpsg in [4u32, 8, 16] {
                    let want = format!("kernel_mul_mv_ext_{tag}_f32_r1_{r1ptg}_nx{nxpsg}");
                    assert_eq!(
                        ext_name_for(dtype, r1ptg, nxpsg, true),
                        Some(want.as_str()),
                        "{dtype:?} at {r1ptg} rows and {nxpsg} lanes must name the \
                         instantiation the CUDA generic set exports"
                    );
                }
            }
            assert_eq!(
                ext_name_for(dtype, 0, 4, true),
                None,
                "{dtype:?} has no zero-row table slot"
            );
            assert_eq!(
                ext_name_for(dtype, 6, 4, true),
                None,
                "{dtype:?} is not instantiated at six rows per threadgroup"
            );
            assert_eq!(
                ext_name_for(dtype, 1, 32, true),
                None,
                "{dtype:?} is not instantiated at a 32-lane split"
            );
        }
    }
}
