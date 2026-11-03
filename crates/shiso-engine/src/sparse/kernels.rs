//! Every kernel the sparse stack dispatches: its name, and the grid, block and
//! shared-memory size its launcher computes.
//!
//! This file is a mirror of one document - the kernel set's own name and
//! launch-geometry table - and it is the only place in this crate where a
//! kernel of that set is spelled. "Does the stack dispatch a kernel that does
//! not exist" is therefore one comparison against that table rather than a
//! survey of a dozen call sites, and "is a grid computed the way its launcher
//! computes it" is a unit test against a transcribed formula rather than an
//! argument about a number.
//!
//! # Two kinds of entry, and why only one of them is a `&'static str`
//!
//! Most of these kernels carry C linkage, so the driver answers to the name
//! written here verbatim and a [`KernelRef`] names them directly.
//!
//! The rest are explicit instantiations of C++ templates. Those get mangled
//! names that depend on the compiler's mangling of the argument list, and the
//! kernel set's table deliberately does not write them down: a mangled name
//! that is subtly wrong fails at lookup with "not found", and one that is
//! subtly wrong *and happens to match a different instantiation* does not fail
//! at all. So this crate never spells one. It names the template and the
//! arguments it was instantiated with - [`Instantiation`] - and resolves that
//! pair through an [`EntryTable`] captured off the built module, which is the
//! only place those names can honestly come from.
//!
//! # The constants are the kernels' own
//!
//! `TILE`, `SPLIT`, `WARPS` and the rest are compile-time constants defined in
//! the kernel text, and the launch formulas are written in terms of them. They
//! are transcribed here rather than folded into numbers so that a formula reads
//! the way its launcher reads and a changed kernel constant is a one-line edit
//! rather than an archaeology exercise.
//!
//! # Shared memory past the default
//!
//! Several rows exceed the 48 KiB per-block default and have to be opted into
//! before the launch. Nothing here does that: the backend derives the opt-in
//! from the launch itself - the first launch asking for more than the default
//! raises that entry point's ceiling - so a table of which kernels need it
//! would be a second copy of a fact the size already carries.

use shiso_core::Extent3;
use shiso_core::backend::KernelRef;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::page::KvElem;

// ---------------------------------------------------------------------------
// modules
// ---------------------------------------------------------------------------

/// Hyper-connection stream mixing.
pub const MOD_HC: &str = "q4e_hc";
/// Routing, expert sorting and the routed feed-forward's own stages.
pub const MOD_MOE: &str = "q4e_moe";
/// The gated recurrent block.
pub const MOD_GDN: &str = "q4e_gdn";
/// The gated recurrence of a prefill chunk, on tensor cores.
pub const MOD_GDN_TC: &str = "q4e_gdn_tc";
/// The same prefill chunk's whole recurrent block between its projections in
/// two launches: the convolution, norms, gates and run-independent half in one,
/// the carry and the output gate in the other, bit for bit what the
/// [`MOD_GDN_TC`] chain computes. Not an upstream unit, for the reason
/// [`MOD_MODELOPT`] is not.
pub const MOD_GDN_FUSED: &str = "q4e_gdn_fused";
/// The n-gram embedding block.
pub const MOD_PLE: &str = "q4e_ple";
/// Dense projections at every stored element type.
pub const MOD_MATVEC: &str = "q4e_matvec";
/// Dense attention and the query-side rotary.
pub const MOD_QSA: &str = "q4e_qsa";
/// Selected-block attention.
pub const MOD_QSA_SPARSE: &str = "q4e_qsa_sparse";
/// The indexer that chooses the selected blocks.
pub const MOD_IDX: &str = "q4e_idx";
/// Elementwise helpers and the greedy row pick.
pub const MOD_MISC: &str = "q4e_misc";
/// The `modelopt` codecs this family's NVIDIA exports store: the routed
/// stages over an NVFP4 bank, and the FP8 n-gram row decode. Not an upstream
/// unit, which is why it is a module of its own rather than entries in
/// [`MOD_MOE`] or [`MOD_PLE`].
pub const MOD_MODELOPT: &str = "q4e_modelopt";
/// This family's own variants of two ported paths: the tensor-core attention
/// with exact softmax weights, and the fused decode row of a recurrent layer.
/// Not an upstream unit, for the reason [`MOD_MODELOPT`] is not.
pub const MOD_EXACT: &str = "q4e_exact";
/// The key/value store and every attention reader of the K and V planes, over
/// a cache of FP8 E4M3 bytes. Not an upstream unit, for the reason
/// [`MOD_MODELOPT`] is not; see [`CacheKernel`].
pub const MOD_KV8: &str = "q4e_kv8";

/// Dense projections over an MXFP8 weight, read as stored: the decode and
/// verify kernels, which sum a row as the brain-float tier sums the widened
/// weight, and the tensor-core prompt kernel. Not an upstream unit, for the
/// reason [`MOD_MODELOPT`] is not.
pub const MOD_MXFP8: &str = "q4e_mxfp8";
/// The indexer's pooled key pass over a per-session ring of raw keys, which is
/// what lets a page carry no raw key row per position. Not an upstream unit,
/// for the reason [`MOD_MODELOPT`] is not.
pub const MOD_IDXRING: &str = "q4e_idxring";

/// The routed NVFP4 expert stages of a prompt pass with NVFP4 activations
/// (W4A4) on the block-scaled FP4 instruction, and their quantizer and
/// brain-float combine. Present only in a kernel set built for a
/// family-specific architecture (`kernels/cuda/q4e_lowp`); see
/// [`crate::sparse::matmul::PrefillActivations`].
pub const MOD_NVFP4_A4: &str = "q4e_nvfp4_a4";
/// The dense MXFP8 projections of a prompt pass with MXFP8 activations
/// (W8A8) on the block-scaled FP8 instruction, and their quantizer. Present
/// where [`MOD_NVFP4_A4`] is.
pub const MOD_MXFP8_A8: &str = "q4e_mxfp8_a8";
/// The entries `--prefill-activations lowp` launches, as (module, entry):
/// every one has to resolve before the option is accepted.
pub const LOWP_ENTRIES: [(&str, &str); 7] = [
    (MOD_NVFP4_A4, "q4e_nvfp4_a4_quant_kernel"),
    (MOD_NVFP4_A4, "q4e_nvfp4_a4_quant_bf16_kernel"),
    (MOD_NVFP4_A4, "q4e_nvfp4_a4_gate_up_kernel"),
    (MOD_NVFP4_A4, "q4e_nvfp4_a4_down_kernel"),
    (MOD_NVFP4_A4, "q4e_nvfp4_a4_combine_kernel"),
    (MOD_MXFP8_A8, "q4e_mxfp8_a8_quant_kernel"),
    (MOD_MXFP8_A8, "q4e_mxfp8_a8_gemm_kernel"),
];

/// Dense projections over a brain-float weight, read as stored, on a device
/// with tensor cores: split step kernels that sum a row one way whatever the
/// pass width, and the tensor-core prompt GEMM. Not an upstream unit, for the
/// reason [`MOD_MODELOPT`] is not; the ported brain-float kernel in
/// [`MOD_MATVEC`] is what a kernel set without this module keeps.
pub const MOD_BF16: &str = "q4e_bf16";
/// The routed block's step launch: a decode step's or a verify's router,
/// routed selection and shared expert as one grid of work items, bit for bit
/// what the [`MOD_BF16`], [`MOD_MOE`] and [`MOD_MXFP8`] launches it replaces
/// compute. Not an upstream unit, for the reason [`MOD_MODELOPT`] is not; see
/// [`crate::sparse::moe::fused_route`].
pub const MOD_FFN: &str = "q4e_ffn";
/// The logits stages run before any pick: today, the suppress stage.
pub const MOD_SAMPLE: &str = "sample";
/// The row gather that opens a pass, at each stored element type of a
/// vocabulary table.
pub const MOD_EMBED: &str = "q4e_embed";
/// The trellis-codec projections, which launch cooperatively.
pub const MOD_EXL3: &str = "exl3";
/// The vision tower's own module, which this family reaches for exactly one
/// entry: the splice that overwrites a placeholder token's embedding with the
/// row the tower produced. Everything else in it runs above the stack.
pub const MOD_VISION: &str = crate::launch::VISION;

// ---------------------------------------------------------------------------
// the kernels' own compile-time constants
// ---------------------------------------------------------------------------

/// Threads in the linear elementwise kernels, and the width most of the set
/// uses.
pub const BLOCK_LINEAR: u32 = 256;
/// Threads in the narrow per-row kernels.
pub const BLOCK_NARROW: u32 = 128;
/// Lanes in one warp, which is also the width of the shortest block here.
pub const WARP: u32 = 32;
/// Threads in the two kernels that reduce a whole row in one block.
pub const BLOCK_WIDE: u32 = 1_024;

/// Streams one fused hyper-connection combine may carry.
///
/// The combine's block width is fixed, so the residual it can hold is bounded;
/// a stack whose stream width exceeds this has no fused combine at all and the
/// launcher refuses the call rather than truncating.
pub const HC_FUSE_MAX: u32 = 16;

/// Lanes per state column in the recurrent step.
pub const GDN_SPLIT: u32 = 4;
/// Tokens one chunked recurrence run covers.
pub const GDN_CHUNK: u32 = 64;
/// Query rows the chunked recurrence stages at a time in its output phase.
pub const GDN_QTILE: u32 = 32;
/// Lanes per value column in the chunked recurrence.
pub const GDN_CSPLIT: u32 = 8;
/// The one head width both recurrent kernels' state tiles are sized for.
///
/// Their launchers reject any other, so this is a precondition rather than a
/// tuning parameter: the shared-memory formulas below are only meaningful at
/// this width.
pub const GDN_STATE_HEAD_DIM: u32 = 128;

/// Width of one stored indexer key row.
pub const IDX_DIM: u32 = 128;
/// Positions one pooled indexer key block averages over.
pub const IDX_R: u32 = 4;
/// Query rows one scorer block covers.
pub const IDX_QT: u32 = 16;
/// Key blocks one scalar scorer block covers.
pub const IDX_BT: u32 = 64;
/// Key blocks one tensor-core scorer block covers.
pub const IDX_TC_BT: u32 = 128;
/// Shared memory the tensor-core scorer reserves.
pub const IDX_TC_SMEM: u32 = (16 * 4 + IDX_TC_BT) * IDX_DIM * 2;
/// Head count the tensor-core scorer is written for; every other count runs
/// the scalar one.
pub const IDX_TC_HEADS: u32 = 4;
/// Query rows one selection pass covers.
///
/// A prefill chunk wider than this is selected in passes of this width, each
/// reading the position of its own last row.
pub const IDX_QUERY_BLOCK: u32 = 512;
/// Slots one expanded selection writes per query row: the selected positions
/// plus three words of bookkeeping.
pub const IDX_WIDTH: u32 = 2_048 + 3;
/// Raw indexer key rows one session keeps per attention rank, each position at
/// slot `position % IDX_RING`.
///
/// The pooled pass is the only reader of a raw key, and it reads the keys of
/// the block a pass starts inside - at most `IDX_R - 1` positions behind the
/// pass's first. Between passes those are the only raw keys worth keeping, and
/// a ring this deep keeps them through the furthest rewind a speculative step
/// makes: a verify settles anywhere inside its `MAX_DRAFT + 1` rows, and the
/// draft head's catch-up starts up to `MAX_DRAFT` positions behind its chain.
/// With every pass starting at most `D` positions behind the highest written,
/// a slot still holds the key a pass reads when the ring is `D + IDX_R - 1`
/// deep (see `idxring.cu`); [`RING_REWIND`] is that `D`.
pub const IDX_RING: u32 = 16;
/// The furthest any pass starts behind the highest position its rank has
/// written: a full verify of `MAX_DRAFT` drafts and its committed token.
pub const RING_REWIND: u32 = crate::sparse::mtp::MAX_DRAFT + 1;
/// The tap row that asks the pooled pass for no copy into a checkpoint.
pub const IDX_NO_TAP: u32 = u32::MAX;
const _: () = assert!(
    IDX_RING >= RING_REWIND + IDX_R - 1 && IDX_RING.is_multiple_of(IDX_R),
    "the raw key ring must outlast a speculative rewind"
);

/// Warps in the dense attention block.
pub const ATTN_WARPS: u32 = 8;
/// Query rows one tiled attention block covers, which is also its floor: below
/// this row count the tiled path is not taken.
pub const ATTN_QT: u32 = 16;
/// Keys per tile, one per lane.
pub const ATTN_KT: u32 = 32;
/// Warps in the tiled and split attention blocks.
pub const ATTN_TILE_WARPS: u32 = 8;
/// Query rows one sparse split block covers.
pub const ATTN_QG: u32 = 16;
/// Key splits per (key head, query group) in the split path.
pub const ATTN_SPLITS: u32 = 48;
/// The one head width the sparse partial slab is sized for.
pub const ATTN_SPLIT_HD: u32 = 256;
/// Splits the gather path runs.
pub const ATTN_GSPLITS: u32 = 8;
/// Keys per tile in the tensor-core attention block.
pub const TC_KT: u32 = 64;
/// Head width the tensor-core attention block is written for.
pub const TC_HD: u32 = 256;
/// Shared memory the tensor-core attention block reserves.
pub const TC_SMEM: u32 = (16 + 2 * TC_KT) * TC_HD * 2 + 16 * TC_KT * 4 + 16 * TC_KT * 2 + 16 * 4;
/// Shared memory the exact tensor-core attention reserves: the ported block's,
/// plus a second f16 weight tile for the low half of every softmax weight.
pub const TC_EXACT_SMEM: u32 = TC_SMEM + 16 * TC_KT * 2;

/// Rows past which a trellis projection is expanded once and multiplied by the
/// BLAS tier instead of streamed by the cooperative kernel.
pub const EXL3_RECONSTRUCT_ROWS: u32 = 144;

/// Threads one block of the fused routed trellis launch carries.
///
/// A constant of the kernel rather than of the tiling shape, and the difference
/// matters: `exl3_moe_kernel` is `__launch_bounds__` at this width and reads it
/// back as `block_threads` to divide its own work, while its sibling
/// projections take their width from the shape they were instantiated for. It
/// is `EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16` = `256 * 32 / 16` in the
/// kernel's own terms; the routed launch is built at one output tile, so there
/// is no shape to read it from.
pub const EXL3_MOE_BLOCK_DIM: u32 = 512;

/// Dynamic shared memory every cooperative trellis launch reserves.
///
/// Not a tuning parameter: the instances are compiled for this budget and the
/// device attribute that admits it is raised from the launch itself, so a
/// launch asking for less would run an instance whose tiles do not fit.
pub const EXL3_SMEM_MAX: u32 = 90 * 1024;

/// The trellis projections' own entry names, unmangled.
///
/// The dense projection.
pub const EXL3_GEMM: &str = "exl3_gemm_kernel";
/// Several stacked weights behind one launch, one slot each.
pub const EXL3_MGEMM: &str = "exl3_mgemm_kernel";
/// The fused routed block: gate, activation and down for every routed column.
pub const EXL3_MOE: &str = "exl3_moe_kernel";
/// The one-off expansion of a trellis weight into half precision.
pub const EXL3_RECONSTRUCT: &str = "reconstruct_had_kernel";
/// The weight-stationary routed block's first launch: every routed column's
/// input, scaled and Hadamard-transformed for gate and up, in expert order.
///
/// Not a template, but its name is C++-mangled all the same, so it is read off
/// the module exactly as the instantiations are.
pub const EXL3_MOE_WS_PREP: &str = "exl3_moe_ws_prep";
/// The inverse of the expert sort, which the by-token prep writes through.
pub const EXL3_MOE_WS_INVERT: &str = "exl3_moe_ws_invert";
/// The weight-stationary block's input rows, produced one token at a time.
pub const EXL3_MOE_WS_PREP_TOK: &str = "exl3_moe_ws_prep_tok";
/// The weight-stationary routed block's gate and up projections, with the
/// activation and the down projection's input transform as their epilogue.
pub const EXL3_MOE_WS_GATEUP: &str = "exl3_moe_ws_gateup";
/// The weight-stationary routed block's down projection.
pub const EXL3_MOE_WS_DOWN: &str = "exl3_moe_ws_down";
/// The same down projection writing its slot outputs as halves.
pub const EXL3_MOE_WS_DOWN_H: &str = "exl3_moe_ws_down_h";
/// The weighted sum of the half slot outputs, with the gated shared expert
/// added in the same pass.
pub const EXL3_MOE_WS_COMBINE: &str = "exl3_moe_ws_combine";
/// The decode-shaped routed block's gate and up projections: raw sums only,
/// over the active experts in 16-row passes.
pub const EXL3_MOE_WSD_GATEUP: &str = "exl3_moe_wsd_gateup";
/// The decode-shaped block's input rows, one warp per (token, slot, chunk).
pub const EXL3_MOE_WSD_PREP: &str = "exl3_moe_wsd_prep";
/// Slots the decode-shaped combine stages, one warp each (`WSD_MAX_USED`).
pub const EXL3_MOE_WSD_MAX_USED: u32 = 16;
/// The gate and up epilogue the weight-stationary launch runs in place, as a
/// launch of its own over the decode-shaped block's raw sums.
pub const EXL3_MOE_WSD_ACT: &str = "exl3_moe_wsd_act";
/// The decode-shaped routed block's down projection: raw sums only.
pub const EXL3_MOE_WSD_DOWN: &str = "exl3_moe_wsd_down";
/// The down epilogue and the weighted combine in one launch, over the
/// decode-shaped block's raw sums.
pub const EXL3_MOE_WSD_COMBINE: &str = "exl3_moe_wsd_combine";
/// Warps one decode-shaped block carries, one n16 column group each
/// (`SHS_WSD_WARPS`).
pub const EXL3_MOE_WSD_WARPS: u32 = 4;
/// Pipeline stages the decode-shaped instances were built with.
const EXL3_MOE_WSD_STAGES: u32 = 4;

/// Dynamic shared memory one decode-shaped block needs.
///
/// `WsdSmem<bits, two, STAGES, WARPS>::total`, restated: per stage a 16 x 64
/// half-precision activation tile and four by `WARPS` trellis tiles, for one
/// matrix or for the gate and up pair.
#[must_use]
pub const fn exl3_moe_wsd_smem(bits: u32, two: bool) -> u32 {
    let tile_words = 256 * bits / 16;
    let a_stage = 16 * 64;
    let b_stage = (64 / 16) * EXL3_MOE_WSD_WARPS * tile_words;
    let matrices = if two { 2 } else { 1 };
    EXL3_MOE_WSD_STAGES * matrices * (a_stage * 2 + b_stage * 2)
}
/// The decode-shaped dense projection's input transform.
pub const EXL3_DGEMV_HAD: &str = "exl3_dgemv_had";
/// The decode-shaped dense projection itself.
pub const EXL3_DGEMV: &str = "exl3_dgemv";
/// Warps one decode-shaped dense block carries (`SHS_DG_WARPS`).
pub const EXL3_DGEMV_WARPS: u32 = 4;
/// Pipeline stages the decode-shaped dense instances were built with.
const EXL3_DGEMV_STAGES: u32 = 4;
/// K one pipeline stage of the decode-shaped dense projection covers
/// (`DG_KT`), which a slice of the contraction has to be a multiple of.
pub const EXL3_DGEMV_KT: u32 = 64;

/// Dynamic shared memory one decode-shaped dense block needs.
///
/// `DgSmem<bits, STAGES, WARPS, MT>::total`, restated: per stage `tiles`
/// 16 x 64 half-precision activation tiles and four by `WARPS` trellis tiles.
#[must_use]
pub const fn exl3_dgemv_smem(bits: u32, tiles: u32) -> u32 {
    let tile_words = 256 * bits / 16;
    let a_stage = tiles * 16 * EXL3_DGEMV_KT;
    let b_stage = (EXL3_DGEMV_KT / 16) * EXL3_DGEMV_WARPS * tile_words;
    EXL3_DGEMV_STAGES * (a_stage * 2 + b_stage * 2)
}

/// Threads one block of every weight-stationary launch carries (`WS_THREADS`).
pub const EXL3_MOE_WS_THREADS: u32 = 256;
/// Output columns one weight-stationary block owns.
pub const EXL3_MOE_WS_TILE_N: u32 = 128;
/// Pipeline stages the weight-stationary instances were built with.
const EXL3_MOE_WS_STAGES: u32 = 3;

/// Dynamic shared memory one weight-stationary block needs.
///
/// `WsSmem<bits, two, STAGES>::total`, restated: per stage, a 64 x 64
/// half-precision activation tile and four by eight 16 x 16 trellis tiles, for
/// one matrix or for the gate and up pair; and never less than the two
/// 32 x 128 single-precision tiles the epilogue stages its output through.
#[must_use]
pub const fn exl3_moe_ws_smem(bits: u32, two: bool) -> u32 {
    let tile_words = 256 * bits / 16;
    let a_stage = 64 * 64;
    let b_stage = (64 / 16) * 8 * tile_words;
    let matrices = if two { 2 } else { 1 };
    let bytes = EXL3_MOE_WS_STAGES * matrices * (a_stage * 2 + b_stage * 2);
    let epilogue = 2 * 32 * 128 * 4;
    if bytes > epilogue { bytes } else { epilogue }
}

/// Bytes of one single-precision element.
pub const F32_BYTES: u64 = 4;
/// Bytes of one half-precision element, which is what the cache stores.
pub const F16_BYTES: u64 = 2;
/// Bytes of one `int32`, which is what a token id, a position and a page slot
/// each are.
pub const I32_BYTES: u64 = 4;

// ---------------------------------------------------------------------------
// naming an entry point
// ---------------------------------------------------------------------------

/// The element type a template was instantiated at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Elem {
    /// Single precision.
    F32,
    /// Half precision.
    F16,
}

/// The arguments one template was instantiated with.
///
/// A closed set rather than a formatted string: an instantiation this build
/// never produced is then a variant that cannot be constructed rather than a
/// lookup that fails at request time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum TemplateArgs {
    /// One element type: the stream norm, the collapse, the fused combine, and
    /// the partial-sum combine.
    Elem(Elem),
    /// A row count: the narrow and warp-per-row projections at a fixed
    /// blocking.
    Rows(u32),
    /// A row count and the element types the projection reads and writes.
    RowsIo {
        /// Rows the instantiation blocks over.
        rows: u32,
        /// Element type of the activation it reads.
        input: Elem,
        /// Element type of the destination it writes.
        output: Elem,
    },
    /// A stored element type id, for the routed down projection.
    StoredType(u32),
    /// A trellis width and the tiling shape number, for the two cooperative
    /// projections.
    ///
    /// The shape is a number rather than a [`crate::sparse::exl3::GemmShape`]
    /// so that this enum stays a plain key: the shape's own type carries the
    /// tile widths, and a table row is matched on the number the instantiation
    /// was built with.
    Trellis {
        /// Trellis width the instance decodes.
        bits: u32,
        /// Tiling shape number the instance was built for.
        shape: u8,
    },
    /// A trellis width alone: the fused routed block, whose output tile is
    /// fixed, and the expansion.
    TrellisBits(u32),
    /// A trellis width and the m16 tiles one pass holds, for the
    /// decode-shaped dense projection.
    TrellisTiles {
        /// Trellis width the instance decodes.
        bits: u32,
        /// Sixteen-row tiles one pass of the instance multiplies at once.
        tiles: u32,
    },
    /// No arguments at all: an entry point with C++ linkage that is not a
    /// template, whose mangled name has to be read off the module like one.
    Untemplated,
    /// A stored element type id and the block count one weight row holds.
    StoredTypeBlocks {
        /// The stored element type id the weights carry.
        stored: u32,
        /// Blocks in one weight row, which the kernel keeps in registers.
        blocks: u32,
    },
}

/// One instantiation of a template kernel: which template, and with what.
///
/// This is deliberately not a name. The driver's name for an instantiation is
/// mangled and has to be read off the built module; what identifies the kernel
/// to a reader, and what a captured name is matched back against, is the pair
/// below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Instantiation {
    /// Module the template's translation unit was compiled into.
    pub module: &'static str,
    /// The template's own name, unmangled.
    pub template: &'static str,
    /// The arguments it was instantiated with.
    pub args: TemplateArgs,
}

impl Instantiation {
    /// Names one instantiation.
    #[must_use]
    pub const fn new(module: &'static str, template: &'static str, args: TemplateArgs) -> Self {
        Self {
            module,
            template,
            args,
        }
    }
}

/// An entry point the stack can launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Entry {
    /// A C-linkage entry point, which the driver answers to by this name.
    Plain(KernelRef),
    /// One instantiation of a template, resolved through an [`EntryTable`].
    Template(Instantiation),
}

/// The kernels that touch the attention key and value planes, named by the job
/// they do rather than by entry point.
///
/// Each one has an entry per cache element, and they share everything else:
/// the grid, the block, the shared-memory reservation and the argument list.
/// So a launch site names the job and the element and gets the entry, and the
/// geometry helpers below serve both. An enum rather than a name lookup, so a
/// launch site cannot ask for the fp8 twin of a kernel that has none and get
/// the f16 one back - every match below is exhaustive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheKernel {
    /// The key norm, rotary and store of one sequence's rows.
    Store,
    /// The same store with one page table per row.
    StoreBatch,
    /// The always-correct dense attention.
    Attention,
    /// The dense attention with one page table per row.
    AttentionBatch,
    /// The tiled dense attention.
    Tiled,
    /// The split-key attention.
    Split,
    /// The gathered attention over a selection.
    Gather,
    /// The exact tensor-core attention, dense or gathered.
    TensorCore,
}

impl CacheKernel {
    /// Every cache kernel, in no particular order.
    pub const ALL: [Self; 8] = [
        Self::Store,
        Self::StoreBatch,
        Self::Attention,
        Self::AttentionBatch,
        Self::Tiled,
        Self::Split,
        Self::Gather,
        Self::TensorCore,
    ];

    /// The entry that does this job over a cache of `kv`.
    #[must_use]
    pub const fn entry(self, kv: KvElem) -> Entry {
        let (module, name) = self.name(kv);
        Entry::plain(module, name)
    }

    /// The module and entry name that do this job over a cache of `kv`.
    #[must_use]
    pub const fn name(self, kv: KvElem) -> (&'static str, &'static str) {
        match kv {
            KvElem::Fp8E4m3 => (
                MOD_KV8,
                match self {
                    Self::Store => "q4e_qsa_store_kv_fp8_kernel",
                    Self::StoreBatch => "q4e_qsa_store_kv_batch_fp8_kernel",
                    Self::Attention => "q4e_qsa_attention_fp8_kernel",
                    Self::AttentionBatch => "q4e_qsa_attention_batch_fp8_kernel",
                    Self::Tiled => "q4e_qsa_attention_tiled_fp8_kernel",
                    Self::Split => "q4e_qsa_attention_split_fp8_kernel",
                    Self::Gather => "q4e_qsa_attention_gather_fp8_kernel",
                    Self::TensorCore => "q4e_qsa_attention_tc_exact_fp8_kernel",
                },
            ),
            KvElem::F16 => match self {
                Self::Store => (MOD_QSA, "q4e_qsa_store_kv_kernel"),
                Self::StoreBatch => (MOD_QSA, "q4e_qsa_store_kv_batch_kernel"),
                Self::Attention => (MOD_QSA, "q4e_qsa_attention_kernel"),
                Self::AttentionBatch => (MOD_QSA, "q4e_qsa_attention_batch_kernel"),
                Self::Tiled => (MOD_QSA, "q4e_qsa_attention_tiled_kernel"),
                Self::Split => (MOD_QSA_SPARSE, "q4e_qsa_attention_split_kernel"),
                Self::Gather => (MOD_QSA_SPARSE, "q4e_qsa_attention_gather_kernel"),
                Self::TensorCore => (MOD_EXACT, "q4e_qsa_attention_tc_exact_kernel"),
            },
        }
    }
}

/// The `(module, entry)` pairs a cache element needs beyond the f16 set, which
/// a stack probes before it accepts the element; `None` for f16, whose kernels
/// are the ones every other path already requires.
#[must_use]
pub fn kv_entries(kv: KvElem) -> Option<[(&'static str, &'static str); 8]> {
    if kv == KvElem::F16 {
        return None;
    }
    Some(CacheKernel::ALL.map(|kernel| kernel.name(kv)))
}

impl Entry {
    /// A C-linkage entry point in `module`.
    #[must_use]
    pub const fn plain(module: &'static str, entry: &'static str) -> Self {
        Self::Plain(KernelRef::new(module, entry))
    }

    /// One instantiation of `template` in `module`.
    #[must_use]
    pub const fn template(
        module: &'static str,
        template: &'static str,
        args: TemplateArgs,
    ) -> Self {
        Self::Template(Instantiation::new(module, template, args))
    }

    /// The name this entry is reported by, which for an instantiation is the
    /// template rather than a mangled name nothing here knows.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Plain(kernel) => kernel.entry,
            Self::Template(inst) => inst.template,
        }
    }

    /// The module the entry lives in.
    #[must_use]
    pub const fn module(&self) -> &'static str {
        match self {
            Self::Plain(kernel) => kernel.module,
            Self::Template(inst) => inst.module,
        }
    }
}

/// The captured mangled names of every template instantiation in a build.
///
/// A build-time artifact, not something this crate can produce: the names come
/// out of the compiled module's own symbol table, so the table is empty on a
/// machine that has never compiled the kernels and every template dispatch
/// there is a typed refusal rather than a guessed name.
///
/// `#[non_exhaustive]` is deliberately absent: a build script has to construct
/// one of these, and the attribute would only stop it.
#[derive(Debug, Clone, Copy, Default)]
pub struct EntryTable {
    rows: &'static [(Instantiation, &'static str)],
}

impl EntryTable {
    /// Wraps a captured table.
    #[must_use]
    pub const fn new(rows: &'static [(Instantiation, &'static str)]) -> Self {
        Self { rows }
    }

    /// How many instantiations the build captured.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the build captured nothing, which is the state of a host with no
    /// device toolchain.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Resolves one entry to a launchable reference.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when the entry is a template instantiation
    /// this build did not capture a name for. That is a missing build artifact
    /// rather than a missing kernel, and naming it that way is what keeps a
    /// reader from going looking for a kernel that is present.
    pub fn resolve(&self, entry: Entry) -> CoreResult<KernelRef> {
        match entry {
            Entry::Plain(kernel) => Ok(kernel),
            Entry::Template(want) => self
                .rows
                .iter()
                .find(|(have, _)| *have == want)
                .map(|(_, name)| KernelRef::new(want.module, name))
                .ok_or(CoreError::Unsupported {
                    what: "a template instantiation whose entry name this build never captured",
                    backend: "engine",
                }),
        }
    }
}

// ---------------------------------------------------------------------------
// launch geometry
// ---------------------------------------------------------------------------

/// The three numbers a launch is made of.
///
/// Held together rather than returned as a tuple because two of them are
/// three-dimensional extents that read identically at a call site and would be
/// swappable without a compile error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Launch {
    /// Blocks.
    pub grid: Extent3,
    /// Threads per block.
    pub block: Extent3,
    /// Dynamic shared memory in bytes.
    pub smem: u32,
}

impl Launch {
    /// A one-dimensional launch with no dynamic shared memory.
    #[must_use]
    pub const fn linear(blocks: u32, threads: u32) -> Self {
        Self {
            grid: Extent3::linear(blocks),
            block: Extent3::linear(threads),
            smem: 0,
        }
    }

    /// A launch with an explicit grid and dynamic shared memory.
    #[must_use]
    pub const fn new(grid: Extent3, threads: u32, smem: u32) -> Self {
        Self {
            grid,
            block: Extent3::linear(threads),
            smem,
        }
    }

    /// Total threads this launch starts, for a sizing assertion.
    #[must_use]
    pub const fn threads(&self) -> u64 {
        self.grid.volume() * self.block.volume()
    }
}

/// Blocks covering `n` units at `per` units each.
///
/// The launchers write this as `(n + per - 1) / per`; where one of them writes
/// something that is *not* a ceiling, the function computing that grid says so
/// and does not call this.
#[must_use]
pub fn blocks_for(n: u64, per: u64) -> u32 {
    if per == 0 {
        return 0;
    }
    u32::try_from(n.div_ceil(per)).unwrap_or(u32::MAX)
}

/// A linear elementwise launch over `n` elements.
///
/// Eight of these kernels share this one shape, which is why they share one
/// function: the scale-and-activate, the gated activation, the elementwise add,
/// the row broadcast and the conversion helpers all read their bound from an
/// argument and index by a flat thread id.
#[must_use]
pub fn elementwise(n: u64) -> Launch {
    Launch::linear(blocks_for(n, u64::from(BLOCK_LINEAR)), BLOCK_LINEAR)
}

/// One block per token row for the splice, whatever the residual width: the
/// kernel's own loop strides by `blockDim.x`.
#[must_use]
pub fn splice(n_tok: u32) -> Launch {
    Launch::linear(n_tok, BLOCK_LINEAR)
}

// --- hyper-connection streams ----------------------------------------------

/// The residual broadcast that opens a pass.
#[must_use]
pub fn hc_init(hidden: u32, streams: u32, n_tok: u32) -> Launch {
    elementwise(u64::from(n_tok) * u64::from(streams) * u64::from(hidden))
}

/// One block per (token, stream) pair for the stream norm.
#[must_use]
pub const fn hc_norm(streams: u32, n_tok: u32) -> Launch {
    Launch::linear(n_tok.saturating_mul(streams), BLOCK_LINEAR)
}

/// One block per token for the stream collapse.
#[must_use]
pub const fn hc_collapse(n_tok: u32) -> Launch {
    Launch::linear(n_tok, BLOCK_LINEAR)
}

/// The fused residual update and next-block norm.
///
/// The second grid extent is the launcher's own switch: one block per token
/// once there are enough tokens to fill the machine that way, one block per
/// (token, stream) below it.
#[must_use]
pub const fn hc_combine(streams: u32, n_tok: u32) -> Launch {
    let y = if n_tok >= BLOCK_LINEAR { 1 } else { streams };
    Launch::new(Extent3::new(n_tok, y, 1), BLOCK_LINEAR, 0)
}

/// Whether the fused combine can carry this residual at all.
///
/// Its block width is fixed, so it covers at most `BLOCK_LINEAR * HC_FUSE_MAX`
/// elements of one stream; past that the launcher refuses rather than leaving a
/// tail unwritten.
#[must_use]
pub const fn hc_combine_fits(hidden: u32) -> bool {
    hidden as u64 <= BLOCK_LINEAR as u64 * HC_FUSE_MAX as u64
}

// --- routed feed-forward ----------------------------------------------------

/// One block per token for the router, with the expert logits staged in shared
/// memory for the selection rounds.
#[must_use]
pub const fn moe_route(experts: u32, n_tok: u32) -> Launch {
    Launch::new(
        Extent3::linear(n_tok),
        BLOCK_LINEAR,
        experts.saturating_mul(4),
    )
}

/// A per-token combine over the residual width. The shared expert's add has the
/// same shape and shares this.
#[must_use]
pub fn moe_combine(hidden: u32, n_tok: u32) -> Launch {
    Launch::new(
        Extent3::new(
            blocks_for(u64::from(hidden), u64::from(BLOCK_LINEAR)),
            n_tok,
            1,
        ),
        BLOCK_LINEAR,
        0,
    )
}

/// The counting pass of the expert sort, over the routed columns.
#[must_use]
pub fn moe_sort_count(rows: u64) -> Launch {
    elementwise(rows)
}

/// The scan that turns per-expert counts into bounds.
///
/// One warp, one block: the scan is over the expert count and serial by
/// construction, so a wider launch would be threads waiting on each other.
#[must_use]
pub const fn moe_sort_scan() -> Launch {
    Launch::linear(1, WARP)
}

/// The scatter that places each routed column in its expert's run.
#[must_use]
pub fn moe_sort_scatter(rows: u64) -> Launch {
    elementwise(rows)
}

/// Output rows one down-projection block covers.
///
/// The launcher fixes the block at [`BLOCK_NARROW`] threads and gives each warp
/// one output row, so this is derived from the block width rather than being a
/// constant of its own.
pub const MOE_DOWN_ROWS_PER_BLOCK: u32 = BLOCK_NARROW / WARP;

/// The per-column routed down projection.
#[must_use]
pub fn moe_down(out_dim: u32, rows: u32) -> Launch {
    Launch::new(
        Extent3::new(
            blocks_for(u64::from(out_dim), u64::from(MOE_DOWN_ROWS_PER_BLOCK)),
            rows,
            1,
        ),
        BLOCK_NARROW,
        0,
    )
}

/// The expert-grouped routed down projection.
///
/// The second extent spans only the experts that received a column. The routed
/// column count bounds that list, so a chunk narrower than the expert count can
/// never activate more experts than it has columns.
#[must_use]
pub fn moe_down_grouped(out_dim: u32, rows: u32, experts: u32) -> Launch {
    Launch::new(
        Extent3::new(
            blocks_for(u64::from(out_dim), u64::from(MOE_DOWN_ROWS_PER_BLOCK)),
            rows.min(experts),
            1,
        ),
        BLOCK_NARROW,
        0,
    )
}

// --- the recurrent block ----------------------------------------------------

/// The causal convolution over the recurrent projection's channels.
#[must_use]
pub fn gdn_conv(channels: u32) -> Launch {
    Launch::linear(
        blocks_for(u64::from(channels), u64::from(BLOCK_LINEAR)),
        BLOCK_LINEAR,
    )
}

/// The same convolution with one sequence's window per row.
#[must_use]
pub fn gdn_conv_batch(channels: u32, n_row: u32) -> Launch {
    Launch::new(
        Extent3::new(
            blocks_for(u64::from(channels), u64::from(BLOCK_LINEAR)),
            n_row,
            1,
        ),
        BLOCK_LINEAR,
        0,
    )
}

/// One block per (head, token) for the query and key normalization.
#[must_use]
pub const fn gdn_l2norm(heads: u32, n_tok: u32) -> Launch {
    Launch::new(Extent3::new(heads, n_tok, 1), BLOCK_NARROW, 0)
}

/// The decay and gate pass, one thread per (head, token).
#[must_use]
pub fn gdn_gates(heads: u32, n_tok: u32) -> Launch {
    elementwise(u64::from(heads) * u64::from(n_tok))
}

/// Shared memory the sequential recurrence reserves: one state tile with a
/// float of row padding that keeps its shared reads off a single bank.
#[must_use]
pub const fn gdn_recurrent_smem(head_dim: u32) -> u32 {
    head_dim
        .saturating_mul(head_dim.saturating_add(1))
        .saturating_mul(4)
}

/// The sequential recurrence, one block per value head.
#[must_use]
pub const fn gdn_recurrent(head_dim: u32, v_heads: u32) -> Launch {
    Launch::new(
        Extent3::linear(v_heads),
        head_dim.saturating_mul(GDN_SPLIT),
        gdn_recurrent_smem(head_dim),
    )
}

/// The fused decode row of a recurrent layer: the sequential recurrence's
/// grid, block and state tile, which it carries unchanged.
#[must_use]
pub const fn gdn_step(head_dim: u32, v_heads: u32) -> Launch {
    gdn_recurrent(head_dim, v_heads)
}

/// The same recurrence with one sequence's state per row.
#[must_use]
pub const fn gdn_recurrent_batch(head_dim: u32, v_heads: u32, n_row: u32) -> Launch {
    Launch::new(
        Extent3::new(v_heads, n_row, 1),
        head_dim.saturating_mul(GDN_SPLIT),
        gdn_recurrent_smem(head_dim),
    )
}

/// Shared memory the chunked recurrence reserves.
///
/// Five staged tiles and two rows of per-chunk scalars, at the one head width
/// the kernel's tiles are sized for.
#[must_use]
pub const fn gdn_chunk_smem() -> u32 {
    let c = GDN_CHUNK;
    let d = GDN_STATE_HEAD_DIM;
    (c * d + c * (d + 1) + c * c + GDN_QTILE * d + 2 * c) * 4
}

/// The chunked recurrence, one block per value head.
#[must_use]
pub const fn gdn_chunk(head_dim: u32, v_heads: u32) -> Launch {
    Launch::new(
        Extent3::linear(v_heads),
        head_dim.saturating_mul(GDN_CSPLIT),
        gdn_chunk_smem(),
    )
}

/// Tokens one run of the tensor-core recurrence covers (`Q4E_GT_C`).
pub const GDN_TC_RUN: u32 = 64;

/// Runs of the tensor-core recurrence a chunk of `n_tok` rows is cut into.
#[must_use]
pub const fn gdn_tc_runs(n_tok: u32) -> u32 {
    n_tok.div_ceil(GDN_TC_RUN)
}

/// The run-independent half of the tensor-core recurrence: one block per
/// (value head, run), heads fastest.
#[must_use]
pub const fn gdn_tc_prep(v_heads: u32, n_tok: u32) -> Launch {
    let c = GDN_TC_RUN;
    Launch::new(
        Extent3::new(v_heads, gdn_tc_runs(n_tok), 1),
        512,
        2 * c * 136 * 2 + c * 68 * 4 + 3 * c * 4,
    )
}

/// Value heads per key head the fused prompt kernels are written for.
pub const GDN_FUSED_HEADS_PER_KEY: u32 = 3;

/// Convolution taps the fused prompt kernels are written for.
pub const GDN_FUSED_CONV_TAPS: u32 = 4;

/// Shared memory the fused prep reserves (`Q4E_GF_PREP_SMEM`): two padded
/// half tiles, three float solve tiles and the three heads' decays and write
/// strengths.
pub const GDN_FUSED_PREP_SMEM: u32 = {
    let c = GDN_TC_RUN;
    let r = GDN_FUSED_HEADS_PER_KEY;
    2 * c * 136 * 2 + r * c * 68 * 4 + 2 * r * c * 4
};

/// Shared memory the fused scan reserves (`Q4E_GF_SCAN_SMEM`): the f32 query
/// and key rows, w, T and P, and the decays and write scales of one run.
pub const GDN_FUSED_SCAN_SMEM: u32 = {
    let c = GDN_TC_RUN;
    let d = GDN_STATE_HEAD_DIM;
    2 * c * d * 4 + c * d * 2 + 2 * c * c * 2 + 2 * c * 4
};

/// The fused prep: one block per (key head, run), heads fastest.
#[must_use]
pub const fn gdn_fused_prep(qk_heads: u32, n_tok: u32) -> Launch {
    Launch::new(
        Extent3::new(qk_heads, gdn_tc_runs(n_tok), 1),
        512,
        GDN_FUSED_PREP_SMEM,
    )
}

/// The fused carry and output gate: one block per value head.
#[must_use]
pub const fn gdn_fused_scan(v_heads: u32) -> Launch {
    Launch::new(Extent3::linear(v_heads), 256, GDN_FUSED_SCAN_SMEM)
}

/// The convolution of a prefill chunk, parallel over 64-token tiles.
#[must_use]
pub const fn gdn_conv_par(channels: u32, n_tok: u32) -> Launch {
    Launch::new(
        Extent3::new(channels.div_ceil(BLOCK_LINEAR), n_tok.div_ceil(64), 1),
        BLOCK_LINEAR,
        0,
    )
}

/// The carry of the tensor-core recurrence: one block per value head.
#[must_use]
pub const fn gdn_tc_scan(v_heads: u32) -> Launch {
    let c = GDN_TC_RUN;
    let d = GDN_STATE_HEAD_DIM;
    Launch::new(
        Extent3::linear(v_heads),
        512,
        d * 136 * 2 + 2 * c * 136 * 2 + 2 * c * 4,
    )
}

/// The output gate, one block per (token, head).
#[must_use]
pub const fn gdn_out_gate(heads: u32, n_tok: u32) -> Launch {
    Launch::linear(n_tok.saturating_mul(heads), BLOCK_NARROW)
}

// --- the n-gram block -------------------------------------------------------

/// The row decode, one warp-wide block per (token, head).
#[must_use]
pub const fn ple_dequant(heads: u32, n_tok: u32) -> Launch {
    Launch::linear(n_tok.saturating_mul(heads), WARP)
}

/// The trellis-codec row decode, whose block width is the head width it is
/// reachable at.
#[must_use]
pub const fn ple_dequant_trellis(heads: u32, n_tok: u32, head_dim: u32) -> Launch {
    Launch::linear(n_tok.saturating_mul(heads), head_dim)
}

/// The gated value, one block per (token, stream).
#[must_use]
pub const fn ple_gated_value(streams: u32, n_tok: u32) -> Launch {
    Launch::new(Extent3::new(n_tok, streams, 1), BLOCK_LINEAR, 0)
}

/// The dilated convolution over the gated value's channels.
#[must_use]
pub fn ple_conv(channels: u32) -> Launch {
    gdn_conv(channels)
}

/// The same convolution with one sequence's window per row.
#[must_use]
pub fn ple_conv_batch(channels: u32, n_row: u32) -> Launch {
    gdn_conv_batch(channels, n_row)
}

// --- dense projections ------------------------------------------------------

/// Threads the two narrow block-quantized projections take.
///
/// A row short enough that the wide block would leave most of its threads idle
/// gets the narrow one; the threshold is the block count of one weight row.
#[must_use]
pub const fn narrow_block(blocks_per_row: u64) -> u32 {
    if blocks_per_row <= BLOCK_NARROW as u64 {
        BLOCK_NARROW
    } else {
        BLOCK_LINEAR
    }
}

/// The single-row narrow block-quantized projection, one block per output row.
#[must_use]
pub const fn matvec_narrow(out_dim: u32, blocks_per_row: u64) -> Launch {
    Launch::linear(out_dim, narrow_block(blocks_per_row))
}

/// Output rows one warp-per-row projection block covers.
///
/// Derived from the block width: [`BLOCK_LINEAR`] threads over [`WARP`] lanes
/// is eight warps, one output row each.
pub const ROWS_PER_WARP_BLOCK: u32 = BLOCK_LINEAR / WARP;

/// The multi-row warp-per-row projection.
#[must_use]
pub fn matmul_rows_warp(out_dim: u32) -> Launch {
    Launch::linear(
        blocks_for(u64::from(out_dim), u64::from(ROWS_PER_WARP_BLOCK)),
        BLOCK_LINEAR,
    )
}

/// The half-precision multi-row projection, which splits the contraction.
#[must_use]
pub fn matmul_f16_rows(out_dim: u32, n_split: u32) -> Launch {
    Launch::new(
        Extent3::new(
            blocks_for(u64::from(out_dim), u64::from(ROWS_PER_WARP_BLOCK)),
            n_split,
            1,
        ),
        BLOCK_LINEAR,
        0,
    )
}

/// Output rows one warp of the packed half-precision projection owns at
/// `rows` activation rows: `Q4E_P12_WARP_ROWS` in the kernel, restated.
///
/// Past a few activation rows a warp that owned one output row re-read every
/// activation row for it and ran at a quarter of the bandwidth; owning two or
/// four shares each activation load between them.
#[must_use]
pub const fn p12_warp_rows(rows: u32) -> u32 {
    if rows <= 4 {
        1
    } else if rows <= 16 {
        2
    } else {
        4
    }
}

/// The packed half-precision multi-row projection, which splits the
/// contraction.
///
/// A warp that owns one output row runs in blocks of four warps: on the
/// narrow-contraction mixes a whole row is one step of the loop, and the
/// smaller block keeps more of them resident (measured on a GB10, 28 us to 25
/// on a 320 x 10240 mix). A warp that owns several needs every register the
/// eight-warp block leaves it.
#[must_use]
pub fn matmul_p12_rows(out_dim: u32, n_split: u32, rows: u32) -> Launch {
    let per_warp = p12_warp_rows(rows);
    let threads = if per_warp == 1 {
        BLOCK_NARROW
    } else {
        BLOCK_LINEAR
    };
    Launch::new(
        Extent3::new(
            blocks_for(u64::from(out_dim), u64::from(threads / WARP * per_warp)),
            n_split,
            1,
        ),
        threads,
        0,
    )
}

/// The single-precision projection, which splits the contraction.
#[must_use]
pub const fn matvec_f32(out_dim: u32, n_split: u32) -> Launch {
    Launch::new(Extent3::new(out_dim, n_split, 1), BLOCK_LINEAR, 0)
}

/// The combine that folds a single-precision split's partial sums.
#[must_use]
pub fn matvec_f32_combine(out_dim: u32) -> Launch {
    Launch::linear(
        blocks_for(u64::from(out_dim), u64::from(BLOCK_NARROW)),
        BLOCK_NARROW,
    )
}

/// The combine that folds a half-precision split's partial sums.
///
/// Its grid covers the whole partial set - the projection's row count times the
/// output width - rather than one row, which is the one place a template
/// argument reaches a launch formula, and it is the *caller's* argument rather
/// than its own.
#[must_use]
pub fn matvec_combine_rows(out_dim: u32, rows: u32) -> Launch {
    Launch::linear(
        blocks_for(
            u64::from(rows) * u64::from(out_dim),
            u64::from(BLOCK_NARROW),
        ),
        BLOCK_NARROW,
    )
}

/// The contraction split the single-precision projection's launcher computes.
///
/// It is not an input: the launcher doubles the split while the resulting block
/// count still fits the machine and each split still has a full pass of work.
#[must_use]
pub fn f32_split(in_dim: u32, out_dim: u32) -> u32 {
    let (in_dim, out_dim) = (u64::from(in_dim), u64::from(out_dim));
    let mut split = 1u64;
    while out_dim * split * 2 <= 512 && in_dim / (split * 2) >= 512 {
        split *= 2;
    }
    u32::try_from(split).unwrap_or(1)
}

/// The contraction split the half-precision projection's launcher computes.
#[must_use]
pub fn f16_split(in_dim: u32, out_dim: u32) -> u32 {
    let (in_dim, out_dim) = (u64::from(in_dim), u64::from(out_dim));
    let mut split = 1u64;
    while out_dim * split < 1_024 && in_dim / (split * 2) >= 256 {
        split *= 2;
    }
    u32::try_from(split).unwrap_or(1)
}

// --- dense attention --------------------------------------------------------

/// The query normalization and rotary, one block per (head, token).
#[must_use]
pub const fn qsa_q_norm_rope(heads: u32, n_tok: u32, head_dim: u32) -> Launch {
    Launch::new(
        Extent3::new(heads, n_tok, 1),
        BLOCK_NARROW,
        head_dim.saturating_mul(4),
    )
}

/// The key/value store, one block per (key head, token).
#[must_use]
pub const fn qsa_store_kv(kv_heads: u32, n_tok: u32, head_dim: u32) -> Launch {
    Launch::new(
        Extent3::new(kv_heads, n_tok, 1),
        BLOCK_NARROW,
        head_dim.saturating_mul(4),
    )
}

/// The same key/value store with one sequence's page table per row.
///
/// The same geometry as the scalar entry at a matching row count: the batched
/// sibling differs only in reading its page table out of a per-row array, which
/// is an argument rather than a grid.
#[must_use]
pub const fn qsa_store_kv_batch(kv_heads: u32, n_row: u32, head_dim: u32) -> Launch {
    qsa_store_kv(kv_heads, n_row, head_dim)
}

/// The always-correct dense attention, one block per (head, token).
#[must_use]
pub const fn qsa_attention(heads: u32, n_tok: u32, head_dim: u32) -> Launch {
    Launch::new(
        Extent3::new(heads, n_tok, 1),
        WARP.saturating_mul(ATTN_WARPS),
        ATTN_WARPS.saturating_mul(head_dim).saturating_mul(4),
    )
}

/// The same dense attention with one sequence's page table per row.
#[must_use]
pub const fn qsa_attention_batch(heads: u32, n_row: u32, head_dim: u32) -> Launch {
    qsa_attention(heads, n_row, head_dim)
}

/// Shared memory the tiled dense attention reserves.
#[must_use]
pub const fn qsa_tiled_smem(head_dim: u32) -> u32 {
    2 * head_dim * ATTN_KT * 2 + ATTN_QT * head_dim * 4 + ATTN_TILE_WARPS * ATTN_KT * 4
}

/// The tiled dense attention, one block per (head, query tile).
#[must_use]
pub fn qsa_attention_tiled(heads: u32, n_tok: u32, head_dim: u32) -> Launch {
    Launch::new(
        Extent3::new(heads, blocks_for(u64::from(n_tok), u64::from(ATTN_QT)), 1),
        WARP.saturating_mul(ATTN_TILE_WARPS),
        qsa_tiled_smem(head_dim),
    )
}

/// Whether the tiled path's shape preconditions hold.
///
/// Its launcher refuses the path below the query-tile width or at a head width
/// its tiles cannot cover, and falls back to the always-correct kernel.
#[must_use]
pub const fn qsa_tiled_eligible(n_tok: u32, head_dim: u32) -> bool {
    n_tok >= ATTN_QT && head_dim.is_multiple_of(WARP)
}

/// Shared memory the two sparse query-group kernels reserve.
#[must_use]
pub const fn qsa_sparse_smem() -> u32 {
    ATTN_SPLIT_HD * ATTN_KT * 2 * 2 + ATTN_QG * ATTN_SPLIT_HD * 2 + ATTN_TILE_WARPS * ATTN_KT * 4
}

/// The split attention over every visible key, one block per (query group,
/// split, key head).
#[must_use]
pub fn qsa_attention_split(kv_heads: u32, n_tok: u32) -> Launch {
    // One block per row, whatever the group width: a row's key range and its
    // split are its own, so it sums exactly as it would alone.
    Launch::new(
        Extent3::new(n_tok, ATTN_SPLITS, kv_heads),
        BLOCK_LINEAR,
        qsa_sparse_smem(),
    )
}

/// The gathered attention over the selected blocks.
#[must_use]
pub const fn qsa_attention_gather(kv_heads: u32, n_tok: u32) -> Launch {
    Launch::new(
        Extent3::new(ATTN_GSPLITS, kv_heads, n_tok),
        BLOCK_LINEAR,
        qsa_sparse_smem(),
    )
}

/// Splits the tensor-core attention runs on the sparse entry.
///
/// It depends on the row count alone, deliberately, so a captured graph never
/// sees a different geometry on replay.
#[must_use]
pub fn tc_splits_sparse(kv_heads: u32, n_tok: u32) -> u32 {
    if u64::from(n_tok) * u64::from(kv_heads) >= 64 {
        1
    } else {
        ATTN_GSPLITS
    }
}

/// The tensor-core attention.
#[must_use]
pub const fn qsa_attention_tc(kv_heads: u32, n_tok: u32, splits: u32) -> Launch {
    Launch::new(Extent3::new(splits, kv_heads, n_tok), BLOCK_LINEAR, TC_SMEM)
}

/// The exact tensor-core attention: the ported kernel's grid and block, with
/// the larger shared-memory reservation of its second weight tile.
#[must_use]
pub const fn qsa_attention_tc_exact(kv_heads: u32, n_tok: u32, splits: u32) -> Launch {
    Launch::new(
        Extent3::new(splits, kv_heads, n_tok),
        BLOCK_LINEAR,
        TC_EXACT_SMEM,
    )
}

/// The combine that folds a split attention's partial slabs.
///
/// Its geometry is the same at all three of its call sites; only the split
/// count it is handed differs, and it is skipped entirely when the tensor-core
/// path ran at a single split, because that path wrote the destination itself.
#[must_use]
pub const fn qsa_attention_combine(heads: u32, n_tok: u32) -> Launch {
    Launch::linear(heads.saturating_mul(n_tok), ATTN_SPLIT_HD)
}

// --- the indexer ------------------------------------------------------------

/// The pooled indexer key pass.
///
/// The grid is **not** a ceiling. At most `n_tok / r` key blocks complete in one
/// forward pass and one more covers the partial block, so the launcher writes a
/// truncating divide plus one. Rounding up instead launches the same count at
/// every row count that is not a multiple of the pooling ratio and a different
/// one when it is - which is the case that matters, because it is the only case
/// where the two disagree.
#[must_use]
pub const fn idx_pool(n_tok: u32, compress: u32) -> Launch {
    let blocks = match n_tok.checked_div(compress) {
        Some(whole) => whole + 1,
        None => 0,
    };
    Launch::linear(blocks, BLOCK_NARROW)
}

/// The pooled indexer key pass over the raw key ring: the pooled pass's own
/// grid, whose block 0 always exists because it is the one that writes the
/// ring.
#[must_use]
pub const fn idx_ring_pool(n_tok: u32, compress: u32) -> Launch {
    idx_pool(n_tok, compress)
}

/// The indexer query normalization and rotary, one block per (head, token).
#[must_use]
pub const fn idx_q(heads: u32, n_tok: u32) -> Launch {
    Launch::new(Extent3::new(heads, n_tok, 1), BLOCK_NARROW, 0)
}

/// Shared memory the scalar scorer reserves: the staged query tile plus the key
/// block tile.
#[must_use]
pub const fn idx_score_smem(heads: u32) -> u32 {
    IDX_QT
        .saturating_mul(heads)
        .saturating_add(IDX_BT)
        .saturating_mul(IDX_DIM)
        .saturating_mul(2)
}

/// The scalar block scorer, one block per (key-block tile, query tile).
#[must_use]
pub fn idx_score(heads: u32, n_tok: u32, max_blocks: u32) -> Launch {
    Launch::new(
        Extent3::new(
            blocks_for(u64::from(max_blocks), u64::from(IDX_BT)),
            blocks_for(u64::from(n_tok), u64::from(IDX_QT)),
            1,
        ),
        BLOCK_LINEAR,
        idx_score_smem(heads),
    )
}

/// The tensor-core block scorer, whose key tile is wider than the scalar one's.
///
/// The two `x` extents are therefore not interchangeable, which is why this is
/// a separate function rather than a flag on the one above.
#[must_use]
pub fn idx_score_tc(n_tok: u32, max_blocks: u32) -> Launch {
    Launch::new(
        Extent3::new(
            blocks_for(u64::from(max_blocks), u64::from(IDX_TC_BT)),
            blocks_for(u64::from(n_tok), u64::from(IDX_QT)),
            1,
        ),
        BLOCK_LINEAR,
        IDX_TC_SMEM,
    )
}

/// The block selection, one wide block per query row.
#[must_use]
pub const fn idx_topk(n_tok: u32) -> Launch {
    Launch::linear(n_tok, BLOCK_WIDE)
}

/// The expansion of selected blocks into positions, one block per query row.
#[must_use]
pub const fn idx_expand(n_tok: u32) -> Launch {
    Launch::linear(n_tok, BLOCK_LINEAR)
}

// --- helpers ----------------------------------------------------------------

/// The suppress stage, one block per logits row: each block writes a few
/// dozen ids, so a wide block would be threads with nothing to do.
#[must_use]
pub const fn suppress_rows(rows: u32) -> Launch {
    Launch::linear(rows, BLOCK_LINEAR)
}

/// The greedy pick, one wide block per logits row.
#[must_use]
pub const fn argmax_rows(rows: u32) -> Launch {
    Launch::linear(rows, BLOCK_WIDE)
}

/// Entries [`Buf::RowTop`](crate::sparse::scratch::Buf::RowTop) keeps of a
/// logits row.
///
/// A keyed draw with a top-k stage looks at the row's top `k` and nothing
/// else - its nucleus and its floor are cut from those - so a caller whose
/// `k` is at most this draws from these exactly what it would from the whole
/// row. Sixty-four covers the top-k every checkpoint here ships (twenty, forty)
/// at 512 bytes a row against the megabyte the row is.
pub const ROW_TOP: u32 = 64;

/// The top of each logits row, one wide block per row.
#[must_use]
pub const fn top_rows(rows: u32) -> Launch {
    Launch::linear(rows, BLOCK_WIDE)
}

/// The one-off expansion of a trellis weight: one block per tile of the
/// expanded half-precision result.
///
/// A truncating divide rather than a ceiling, and deliberately so: the tier
/// that reaches this is gated on both dimensions being a whole number of tiles
/// already, so a remainder here would mean the gate let through a shape it
/// should have refused.
#[must_use]
pub const fn trellis_expand(k: u32, n: u32) -> Launch {
    Launch::new(
        Extent3::new(
            n / crate::sparse::exl3::RECONSTRUCT_TILE_N,
            k / crate::sparse::exl3::RECONSTRUCT_TILE_K,
            1,
        ),
        crate::sparse::exl3::RECONSTRUCT_THREADS,
        0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_recorded_shared_memory_sizes_are_what_the_formulas_evaluate_to() {
        // Both figures are recorded beside the kernels, and both are reached
        // only at the one head width the state tiles are sized for. A formula
        // that drifted from them would be a launch failure on the first token
        // of a real model and nothing at all in a test that only asked whether
        // the arithmetic ran.
        assert_eq!(gdn_recurrent_smem(GDN_STATE_HEAD_DIM), 66_048);
        assert_eq!(gdn_chunk_smem(), 99_072);
    }

    #[test]
    fn the_sparse_attention_slabs_sit_under_the_default_and_the_tensor_core_one_does_not() {
        assert_eq!(qsa_sparse_smem(), 41_984);
        assert_eq!(TC_SMEM, 79_936);
        assert_eq!(TC_EXACT_SMEM, 81_984);
        assert_eq!(qsa_tiled_smem(256), 50_176);
        assert_eq!(IDX_TC_SMEM, 49_152);
        assert_eq!(idx_score_smem(4), 32_768);
    }

    #[test]
    fn the_pooling_grid_is_a_truncating_divide_plus_one_and_not_a_ceiling() {
        // The two agree everywhere except at a multiple of the ratio, which is
        // exactly where a ceiling would be wrong: 256 positions complete 64
        // blocks and the launcher still asks for the 65th, which covers the
        // partial block the next token starts.
        assert_eq!(idx_pool(256, IDX_R).grid.x, 65);
        assert_eq!(blocks_for(256, u64::from(IDX_R)), 64);
        assert_eq!(idx_pool(255, IDX_R).grid.x, 64);
        assert_eq!(blocks_for(255, u64::from(IDX_R)), 64);
        assert_eq!(idx_pool(1, IDX_R).grid.x, 1);
    }

    #[test]
    fn an_unresolvable_template_names_the_missing_build_artifact_rather_than_a_kernel() {
        let empty = EntryTable::default();
        let entry = Entry::template(MOD_HC, "q4e_hc_norm_kernel", TemplateArgs::Elem(Elem::F32));
        let err = empty
            .resolve(entry)
            .expect_err("a table with no captured names cannot resolve a template");
        assert!(
            matches!(err, CoreError::Unsupported { what, .. } if what.contains("never captured")),
            "{err:?}"
        );
    }

    #[test]
    fn a_plain_entry_resolves_without_a_table_at_all() {
        let empty = EntryTable::default();
        let entry = Entry::plain(MOD_MISC, "q4e_add2_kernel");
        let kernel = empty.resolve(entry).expect("a C-linkage name is its own");
        assert_eq!(kernel.entry, "q4e_add2_kernel");
        assert_eq!(kernel.module, MOD_MISC);
    }

    #[test]
    fn the_contraction_splits_stop_where_their_launchers_stop() {
        // The single-precision split doubles while the block count stays inside
        // 512 and each split keeps a 512-element pass. The injection logits are
        // the shape it exists for: a very wide contraction into four outputs.
        assert_eq!(f32_split(10_240, 4), 16);
        // A projection wide on the output side never splits at all.
        assert_eq!(f32_split(2_560, 2_560), 1);
        // The half-precision rule is the looser of the two.
        assert_eq!(f16_split(2_560, 4), 8);
        assert_eq!(f16_split(2_560, 2_560), 1);
    }

    #[test]
    fn the_fused_combine_refuses_a_stream_wider_than_its_block_can_cover() {
        assert!(hc_combine_fits(2_560));
        assert!(hc_combine_fits(4_096));
        assert!(!hc_combine_fits(4_097));
    }

    #[test]
    fn the_combine_grid_drops_its_stream_extent_once_a_chunk_fills_the_machine() {
        assert_eq!(hc_combine(4, 1).grid, Extent3::new(1, 4, 1));
        assert_eq!(hc_combine(4, 255).grid, Extent3::new(255, 4, 1));
        assert_eq!(hc_combine(4, 256).grid, Extent3::new(256, 1, 1));
    }

    #[test]
    fn the_grouped_down_projection_spans_only_the_experts_a_chunk_can_have_reached() {
        // Ten routed slots over one token can activate at most ten experts, so
        // the grid is ten blocks deep and not five hundred and twelve.
        assert_eq!(moe_down_grouped(2_560, 10, 512).grid.y, 10);
        assert_eq!(moe_down_grouped(2_560, 5_120, 512).grid.y, 512);
    }
}
