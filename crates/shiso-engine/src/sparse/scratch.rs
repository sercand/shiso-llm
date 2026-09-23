//! Every activation buffer a pass writes, sized once for the widest chunk.
//!
//! Scratch is allocated for the widest chunk the stack will run and reused by
//! every layer. The alternative is per-layer allocation, which for the expert
//! stage alone would be tens of mebibytes multiplied by the layer count, for
//! buffers that never outlive a launch.
//!
//! # Why the buffers are an enum and not forty fields
//!
//! Every one of them is sized by the same three inputs - the geometry, the chunk
//! width and the logit row count - and the *sizes* are the part worth checking
//! without a device. Naming them as an enum with one sizing function puts all
//! forty formulas in one place a test can walk end to end, and makes "is there a
//! buffer this pass writes that nothing allocated" a question the type system
//! answers rather than one a reader has to.
//!
//! # The widest buffer, and why it is the one it is
//!
//! It is neither the logits nor the residual. Over a wide chunk it is the
//! selected-block attention's partial slabs - one head width of accumulator per
//! (query, slice), which at eight slices is eight accumulators per query head
//! per token - and behind it the routed feed-forward's per-column output, one
//! row of the residual width per (token, expert slot). [`widest`] answers which
//! and by how much, because the order between them moves with the expert count
//! and the head count, and a caller choosing a chunk width wants the number
//! rather than a rule of thumb about it.

use std::sync::Arc;

use shiso_core::backend::{Backend, DEVICE_POINTER_BYTES};
use shiso_core::buffer::{DeviceBuffer, DeviceView, alloc_tagged};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ledger::MemTag;
use shiso_core::page::MROPE_BACK;

use crate::sparse::inputs::MROPE_AXES;
use crate::sparse::kernels::{
    ATTN_GSPLITS, ATTN_QT, ATTN_SPLITS, F16_BYTES, F32_BYTES, I32_BYTES, IDX_DIM, IDX_WIDTH,
};
use crate::sparse::shape::SparseShape;

/// Floats the shared contraction-split partial slab holds.
///
/// The half-precision split is the wider of the two users: its launcher splits
/// until the block count approaches a thousand while each split keeps a full
/// pass, and the partial set it writes is the instantiated row count times the
/// output width. This is the bound that set stays inside.
pub const PROJECTION_PARTIAL_FLOATS: u32 = 2_048 * 16;

/// Floats one sparse attention partial slab occupies.
///
/// A head width of accumulator plus the running maximum, the running sum and the
/// padding that keeps two slabs off one bank.
pub const PART_STRIDE: u64 = 256 + 8;

/// One activation buffer a pass writes.
///
/// The order is the order a pass first touches them, which is also the order
/// they are allocated in, so a device that runs out of memory runs out at a
/// buffer a reader can place in the pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Buf {
    /// The chunk's token ids, as the row the gather indexes with.
    Ids,
    /// The chunk's absolute positions.
    Positions,
    /// The chunk's rotary position table, with the prefix rows in front.
    Mrope,
    /// One `int32` per token: the encoder row that token's embedding is
    /// replaced by, or a negative number for a token that keeps its own.
    ///
    /// Written only by a pass carrying media, and read only by the splice that
    /// follows the row gather. It is an activation buffer rather than a
    /// per-request one because it belongs to one chunk and its address must not
    /// move under a capture - and it is allocated for every stack rather than
    /// for the media ones, because the media ones are not known at allocation
    /// time and one `int32` a token is the cheapest buffer here.
    SpliceMap,
    /// The gathered embedding rows.
    Embed,
    /// The residual, every stream.
    Residual,
    /// The normed residual a mix reads.
    Normed,
    /// One mixed vector per token: what a block actually reads.
    Mixed,
    /// The same block input in half precision, which the stream mixes of a
    /// prefill chunk write beside it for the projections that read it on the
    /// half-precision tier.
    MixedHalf,
    /// The low-rank vector inside a stream mix.
    LowRank,
    /// The expansion back to the full stream width.
    Expanded,
    /// The per-stream logits a combine injects.
    Inject,
    /// A block's contribution, before the combine folds it in.
    BlockOut,
    /// The recurrent block's fused query, key and value projection.
    GdnQkv,
    /// The same rows after the causal convolution.
    GdnConv,
    /// Per-head decay.
    GdnDecay,
    /// Per-head write strength.
    GdnBeta,
    /// The decay projection's output, which the gate pass reads.
    GdnAlphaProj,
    /// The write-strength projection's output, which the gate pass reads.
    GdnBetaProj,
    /// The recurrence's output.
    GdnAttn,
    /// The tensor-core recurrence's `u` per (run, head), in half precision.
    GdnTcU,
    /// Its `w` per (run, head), in half precision.
    GdnTcW,
    /// Its masked, scaled query-key products per (run, head), in half
    /// precision.
    GdnTcP,
    /// Its cumulative log decays per (run, head).
    GdnTcG,
    /// That output after the head norm and gate.
    GdnGated,
    /// The recurrent block's output-gate projection.
    GdnGate,
    /// The attention query projection, output gate rows included.
    AttnQueryFull,
    /// The query alone, normed and rotated.
    AttnQuery,
    /// The per-head output gate the query projection carried.
    AttnGate,
    /// The key rows this chunk stores.
    AttnKey,
    /// The value rows this chunk stores.
    AttnValue,
    /// Attention's output, before the gate.
    AttnOut,
    /// The partial slabs a split or gathered attention folds.
    AttnPartial,
    /// The indexer's key rows.
    IdxKey,
    /// The indexer's query projection.
    IdxQuery,
    /// That query, normed and rotated, in half precision.
    IdxQueryNorm,
    /// One score per (query row, key block).
    IdxScore,
    /// The selected key blocks.
    IdxSelected,
    /// How many blocks each query row selected.
    IdxCount,
    /// The selected blocks expanded into positions.
    IdxTokens,
    /// How many positions each query row's expansion holds.
    IdxTokenCount,
    /// One logit per (token, expert).
    MoeRouter,
    /// The experts each token was routed to.
    MoeIds,
    /// The weight each routed expert contributes with.
    MoeWeights,
    /// Per-expert column counts.
    MoeCounts,
    /// Per-expert column bounds, after the scan.
    MoeBounds,
    /// The experts that received a column.
    MoeActive,
    /// The routed columns, sorted by expert.
    MoeSorted,
    /// The draft head's normed view of the target hidden state it starts from.
    DraftNormed,
    /// The draft head's projection of that hidden state, per stream.
    DraftHidden,
    /// A segmented verify's residual rows, held for the draft head's catch-up
    /// passes: each of those overwrites the residual, and every session's
    /// catch-up reads its own verify rows.
    ResidualHold,
    /// A batch of sessions' staged draft rows, side by side, which a batched
    /// draft pass opens from.
    DraftStage,
    /// The routed gate projection, one row per column.
    MoeGate,
    /// The routed up projection, one row per column.
    MoeUp,
    /// The activated inner rows the down projection reads, one per column.
    MoeMid,
    /// The down projection's output, one row per column.
    MoeDown,
    /// The weight-stationary routed block's gate input: every routed column's
    /// row, scaled and transformed, in half precision and expert order.
    MoeWsGate,
    /// The same rows for the up projection, which has its own input scales.
    MoeWsUp,
    /// The half-precision down input the gate and up launch leaves behind.
    MoeWsAct,
    /// Each routed column's row in expert order: the sort's inverse.
    MoeWsPos,
    /// Each routed column's input row, replicated once per slot, for the
    /// grouped launch of a pass of a few rows.
    MoeRep,
    /// The shared expert's gate projection.
    SharedGate,
    /// The shared expert's up projection.
    SharedUp,
    /// The shared expert's activated inner rows.
    SharedMid,
    /// The shared expert's output.
    SharedOut,
    /// The shared expert's scalar gate.
    SharedLogit,
    /// The decoded n-gram rows.
    PleEmbed,
    /// The n-gram keys.
    PleKey,
    /// The residual the n-gram keys are gated against.
    PleQuery,
    /// The n-gram values.
    PleValue,
    /// The gated value.
    PleGated,
    /// The gated value after the dilated convolution.
    PleConv,
    /// The per-row pointer arrays the batched kernels dereference.
    ///
    /// One array per kind of per-session state a batched launch addresses - the
    /// page table, the n-gram convolution window, and each recurrent slot's
    /// convolution window and state matrix - laid out back to back, each one
    /// slot per row of the widest decode batch. See [`crate::sparse::rows`] for
    /// the layout and for who fills it.
    ///
    /// It is an activation buffer rather than a per-session one on purpose: the
    /// contents belong to one step, the address must not move under a capture,
    /// and every row of a batch reads the same array.
    RowPointers,
    /// The logits block.
    Logits,
    /// The greedy pick of every logits row.
    Argmax,
    /// The chunk's activation rounded to half precision, for the one tier that
    /// cannot read a single-precision operand.
    ///
    /// One buffer for every projection that reaches that tier rather than one
    /// each: it is written by the rounding pass immediately before the multiply
    /// that reads it and by nothing else, so it is dead the moment that
    /// multiply returns. It is sized for the widest contraction any projection
    /// in the stack has, which is also what keeps its address still under a
    /// capture.
    HalfActivation,
    /// The partial sums a contraction-split projection writes before its fold.
    ///
    /// One slab for every split projection in the stack rather than one each:
    /// a split's partials are read by the fold that immediately follows it and
    /// by nothing else, so the buffer is dead the moment that fold returns. It
    /// is sized for the widest split any projection here takes and never
    /// resized, which is also what keeps its address still under a capture.
    ProjectionPartial,
}

impl Buf {
    /// Every buffer, in allocation order.
    pub const ALL: [Self; 77] = [
        Self::Ids,
        Self::Positions,
        Self::Mrope,
        Self::SpliceMap,
        Self::Embed,
        Self::Residual,
        Self::Normed,
        Self::Mixed,
        Self::MixedHalf,
        Self::LowRank,
        Self::Expanded,
        Self::Inject,
        Self::BlockOut,
        Self::GdnQkv,
        Self::GdnConv,
        Self::GdnDecay,
        Self::GdnBeta,
        Self::GdnAlphaProj,
        Self::GdnBetaProj,
        Self::GdnAttn,
        Self::GdnTcU,
        Self::GdnTcW,
        Self::GdnTcP,
        Self::GdnTcG,
        Self::GdnGated,
        Self::GdnGate,
        Self::AttnQueryFull,
        Self::AttnQuery,
        Self::AttnGate,
        Self::AttnKey,
        Self::AttnValue,
        Self::AttnOut,
        Self::AttnPartial,
        Self::IdxKey,
        Self::IdxQuery,
        Self::IdxQueryNorm,
        Self::IdxScore,
        Self::IdxSelected,
        Self::IdxCount,
        Self::IdxTokens,
        Self::IdxTokenCount,
        Self::MoeRouter,
        Self::MoeIds,
        Self::MoeWeights,
        Self::MoeCounts,
        Self::MoeBounds,
        Self::MoeActive,
        Self::MoeSorted,
        Self::DraftNormed,
        Self::DraftHidden,
        Self::ResidualHold,
        Self::DraftStage,
        Self::MoeGate,
        Self::MoeUp,
        Self::MoeMid,
        Self::MoeDown,
        Self::MoeWsGate,
        Self::MoeWsUp,
        Self::MoeWsAct,
        Self::MoeWsPos,
        Self::MoeRep,
        Self::SharedGate,
        Self::SharedUp,
        Self::SharedMid,
        Self::SharedOut,
        Self::SharedLogit,
        Self::PleEmbed,
        Self::PleKey,
        Self::PleQuery,
        Self::PleValue,
        Self::PleGated,
        Self::PleConv,
        Self::RowPointers,
        Self::Logits,
        Self::Argmax,
        Self::HalfActivation,
        Self::ProjectionPartial,
    ];

    /// Bytes this buffer occupies for a chunk of `rows` tokens.
    #[must_use]
    #[expect(
        clippy::match_same_arms,
        reason = "one arm per buffer group is the point: two buffers that happen \
                  to be the same width today are still two different formulas, and \
                  merging them would hide the next divergence"
    )]
    pub fn bytes(self, shape: &SparseShape, rows: u32, logit_rows: u32, ctx: u32) -> u64 {
        let rows = u64::from(rows);
        let per_token = |width: u32| rows * u64::from(width) * F32_BYTES;
        let run_rows = rows.next_multiple_of(u64::from(crate::sparse::kernels::GDN_TC_RUN));
        let columns = shape.moe.slot_rows(u32::try_from(rows).unwrap_or(u32::MAX));
        match self {
            Self::Ids | Self::Positions | Self::SpliceMap => rows * I32_BYTES,
            // The rotary table carries a prefix in front of the chunk's first
            // token, because the pooled-key pass derives a pooled row's position
            // by rounding down to the compression boundary and can name a
            // position a few slots before the chunk starts.
            Self::Mrope => (rows + u64::from(MROPE_BACK)) * u64::from(MROPE_AXES) * I32_BYTES,
            Self::Embed | Self::Mixed | Self::BlockOut => per_token(shape.hidden),
            Self::Residual | Self::Normed | Self::Expanded => per_token(shape.stream_dim()),
            Self::LowRank => per_token(shape.stream_rank),
            Self::Inject => per_token(shape.streams),
            Self::GdnQkv | Self::GdnConv => per_token(shape.gdn_in_dim()),
            Self::GdnDecay | Self::GdnBeta | Self::GdnAlphaProj | Self::GdnBetaProj => {
                per_token(shape.gdn.v_heads)
            }
            Self::GdnAttn | Self::GdnGated | Self::GdnGate => per_token(shape.gdn_v_dim()),
            // Whole runs: the last one is padded to the run width, and the
            // prep writes every row of it.
            Self::GdnTcU | Self::GdnTcW => run_rows * u64::from(shape.gdn_v_dim()) * F16_BYTES,
            Self::GdnTcP => {
                run_rows
                    * u64::from(shape.gdn.v_heads)
                    * u64::from(crate::sparse::kernels::GDN_TC_RUN)
                    * F16_BYTES
            }
            Self::GdnTcG => run_rows * u64::from(shape.gdn.v_heads) * F32_BYTES,
            Self::AttnQueryFull => per_token(shape.q_dim()) * 2,
            Self::AttnQuery | Self::AttnGate | Self::AttnOut => per_token(shape.q_dim()),
            Self::AttnKey | Self::AttnValue => per_token(shape.kv_dim()),
            // Two paths write this buffer and the larger of them decides its
            // size; sizing for one alone is an out-of-bounds write on the
            // other. The gathered path runs a few slices over the whole chunk.
            // The split path runs many more slices, but only ever below the
            // tiled path's floor - so it is sized once, for the widest batch it
            // can take, which is also what keeps its address still under a
            // capture.
            Self::AttnPartial => {
                let gathered = rows * u64::from(ATTN_GSPLITS);
                let split = u64::from(ATTN_QT - 1) * u64::from(ATTN_SPLITS);
                u64::from(shape.gqa.q_heads) * gathered.max(split) * PART_STRIDE * F32_BYTES
            }
            Self::IdxKey => per_token(shape.indexer.head_dim),
            Self::IdxQuery => per_token(shape.indexer_q_dim()),
            Self::IdxQueryNorm => {
                rows * u64::from(shape.indexer.heads) * u64::from(IDX_DIM) * F16_BYTES
            }
            // The context in blocks, not the budget: the scorer ranks every
            // block the context can hold and the top-k keeps `budget_blocks` of
            // them, so a row sized by the budget is a row that never ranks
            // anything past it. See `IndexerShape::score_blocks`.
            Self::IdxScore => rows * u64::from(shape.indexer.score_blocks(ctx)) * F32_BYTES,
            Self::IdxSelected => rows * u64::from(shape.indexer.budget_blocks()) * I32_BYTES,
            Self::IdxCount | Self::IdxTokenCount => rows * I32_BYTES,
            Self::IdxTokens => rows * u64::from(IDX_WIDTH) * I32_BYTES,
            // Sized by the chunk, not by the widest draft chain: the pass that
            // fills the head's cache as a prompt is read runs the head over
            // every row of the chunk. Sized at nine rows, a prompt longer than
            // that failed its first chunk's projection and - worse - had its
            // normalization write past the end of this buffer first.
            Self::DraftNormed | Self::DraftHidden => per_token(shape.stream_dim()),
            Self::DraftStage => {
                u64::from(crate::sparse::stack::MAX_DECODE_BATCH)
                    * u64::from(shape.stream_dim())
                    * F32_BYTES
            }
            Self::ResidualHold => u64::from(logit_rows) * u64::from(shape.stream_dim()) * F32_BYTES,
            Self::MoeRouter => per_token(shape.moe.experts),
            Self::MoeIds => rows * u64::from(shape.moe.used) * I32_BYTES,
            Self::MoeWeights => rows * u64::from(shape.moe.used) * F32_BYTES,
            // One extra slot past the expert count: the scan writes the total
            // there, and the grouped down projection reads it as the length of
            // the active list.
            Self::MoeCounts | Self::MoeBounds | Self::MoeActive => {
                (u64::from(shape.moe.experts) + 1) * I32_BYTES
            }
            Self::MoeSorted => columns * I32_BYTES,
            Self::MoeGate | Self::MoeUp | Self::MoeMid => {
                columns * u64::from(shape.moe.intermediate) * F32_BYTES
            }
            Self::MoeDown => columns * u64::from(shape.hidden) * F32_BYTES,
            Self::MoeWsGate | Self::MoeWsUp => columns * u64::from(shape.hidden) * F16_BYTES,
            Self::MoeWsAct => columns * u64::from(shape.moe.intermediate) * F16_BYTES,
            Self::MoeWsPos => columns * I32_BYTES,
            // Only the grouped launch writes it, and only below its row bounds.
            Self::MoeRep => {
                let grouped = [
                    crate::sparse::bind::WeightSet::Target,
                    crate::sparse::bind::WeightSet::Draft,
                ]
                .into_iter()
                .map(crate::sparse::moe::grouped_max)
                .max()
                .unwrap_or(0);
                shape
                    .moe
                    .slot_rows(u32::try_from(rows.min(u64::from(grouped))).unwrap_or(0))
                    * u64::from(shape.hidden)
                    * F32_BYTES
            }
            Self::SharedGate | Self::SharedUp | Self::SharedMid => {
                per_token(shape.moe.intermediate)
            }
            Self::SharedOut => per_token(shape.hidden),
            Self::SharedLogit => rows * F32_BYTES,
            Self::PleEmbed | Self::PleValue => per_token(shape.hidden),
            Self::PleKey | Self::PleQuery | Self::PleGated | Self::PleConv => {
                per_token(shape.stream_dim())
            }
            // One slot per row of the widest decode batch in every array, so
            // the stride is the same at every batch width and an address a
            // launch was planned against never moves with the row count.
            Self::RowPointers => {
                u64::from(crate::sparse::rows::arrays(shape))
                    * u64::from(logit_rows)
                    * DEVICE_POINTER_BYTES
            }
            Self::Logits => u64::from(logit_rows) * u64::from(shape.vocab) * F32_BYTES,
            Self::Argmax => u64::from(logit_rows) * I32_BYTES,
            // The two split projections bound their own partial sets: the
            // single-precision one keeps the output width times the split at or
            // below 512 floats, and the half-precision one at or below the
            // instantiated row count times two thousand and forty-eight. The
            // larger of the two is the size, because one slab serves both.
            // The widest contraction any projection in the stack has is the
            // residual across all its streams, which the stream mixes read
            // whole. Sizing this for a narrower one is an out-of-bounds write
            // by the rounding pass, not a failed launch.
            Self::HalfActivation => rows * u64::from(shape.stream_dim()) * F16_BYTES,
            Self::MixedHalf => rows * u64::from(shape.hidden) * F16_BYTES,
            Self::ProjectionPartial => u64::from(PROJECTION_PARTIAL_FLOATS) * F32_BYTES,
        }
    }

    /// What the memory ledger counts this buffer under: the indexer's and
    /// the experts' workspaces apart from the rest, because the indexer's
    /// score row is the one buffer that grows with the context.
    #[must_use]
    pub const fn tag(self) -> MemTag {
        match self {
            Self::IdxKey
            | Self::IdxQuery
            | Self::IdxQueryNorm
            | Self::IdxScore
            | Self::IdxSelected
            | Self::IdxCount
            | Self::IdxTokenCount
            | Self::IdxTokens => MemTag::IndexerScratch,
            Self::MoeRouter
            | Self::MoeIds
            | Self::MoeWeights
            | Self::MoeCounts
            | Self::MoeBounds
            | Self::MoeActive
            | Self::MoeSorted
            | Self::MoeGate
            | Self::MoeUp
            | Self::MoeMid
            | Self::MoeDown
            | Self::MoeWsGate
            | Self::MoeWsUp
            | Self::MoeWsAct
            | Self::MoeWsPos
            | Self::MoeRep
            | Self::SharedGate
            | Self::SharedUp
            | Self::SharedMid
            | Self::SharedOut
            | Self::SharedLogit => MemTag::MoeScratch,
            _ => MemTag::Activations,
        }
    }

    /// Whether this buffer is only reached by a stack that folds in an n-gram
    /// block.
    #[must_use]
    pub const fn is_ngram(self) -> bool {
        matches!(
            self,
            Self::PleEmbed
                | Self::PleKey
                | Self::PleQuery
                | Self::PleValue
                | Self::PleGated
                | Self::PleConv
        )
    }
}

/// Bytes every buffer a chunk needs, without allocating any of them.
///
/// The sizing is the checkable part, so it is a function of its own: a stack
/// that will not fit is a number a caller can compute before it opens a device.
#[must_use]
pub fn total_bytes(shape: &SparseShape, rows: u32, logit_rows: u32, ctx: u32) -> u64 {
    Buf::ALL
        .iter()
        .filter(|buf| shape.ple.is_some() || !buf.is_ngram())
        .map(|buf| buf.bytes(shape, rows, logit_rows, ctx))
        .fold(0u64, u64::saturating_add)
}

/// The buffer that costs the most, and what it costs.
///
/// Worth having as an accessor rather than as a comment: which buffer dominates
/// changes with the expert count, and a caller choosing a chunk width wants the
/// number rather than a rule of thumb about it.
#[must_use]
pub fn widest(shape: &SparseShape, rows: u32, logit_rows: u32, ctx: u32) -> (Buf, u64) {
    Buf::ALL
        .iter()
        .filter(|buf| shape.ple.is_some() || !buf.is_ngram())
        .map(|buf| (*buf, buf.bytes(shape, rows, logit_rows, ctx)))
        .max_by_key(|(_, bytes)| *bytes)
        .unwrap_or((Buf::Ids, 0))
}

/// Every activation buffer, resident.
#[derive(Debug)]
pub struct Scratch {
    buffers: Vec<DeviceBuffer>,
    rows: u32,
    logit_rows: u32,
}

impl Scratch {
    /// Allocates every buffer a chunk of `rows` tokens needs.
    ///
    /// `logit_rows` is sized separately because the logits block is indexed by
    /// batch member rather than by token: a prefill writes one row of it and a
    /// decode step writes one per session.
    ///
    /// `ctx` is the run's context ceiling, and only the indexer's score buffer
    /// reads it: that buffer's row spans every block the context can hold,
    /// because a block the scorer has no slot for is a block the selection can
    /// never choose.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when an allocation fails.
    pub fn new(
        backend: &Arc<dyn Backend>,
        shape: &SparseShape,
        rows: u32,
        logit_rows: u32,
        ctx: u32,
    ) -> CoreResult<Self> {
        let buffers = Buf::ALL
            .iter()
            .map(|buf| {
                let bytes = if buf.is_ngram() && shape.ple.is_none() {
                    0
                } else {
                    buf.bytes(shape, rows, logit_rows, ctx)
                };
                alloc_tagged(backend, buf.tag(), bytes)
            })
            .collect::<CoreResult<Vec<_>>>()?;
        Ok(Self {
            buffers,
            rows,
            logit_rows,
        })
    }

    /// Rows the scratch was sized for.
    #[must_use]
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Logit rows the scratch was sized for.
    #[must_use]
    pub const fn logit_rows(&self) -> u32 {
        self.logit_rows
    }

    /// One buffer, whole.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when the buffer list is shorter than the
    /// enum, which can only happen if the two were built from different
    /// versions of this file.
    pub fn view(&self, buf: Buf) -> CoreResult<DeviceView<'_>> {
        self.buffer(buf).map(DeviceBuffer::view)
    }

    /// One buffer from a byte offset on.
    ///
    /// Several kernels are launched more than once over disjoint halves of one
    /// projection - the two normalizations of the recurrent block's fused
    /// query and key rows are the clearest case - and where the launcher passes
    /// the offset as a scalar this is not used, but where it advances the
    /// pointer instead this is what that advance is.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when the offset runs past the buffer, and
    /// whatever [`Scratch::view`] reports.
    pub fn view_from(&self, buf: Buf, offset: u64) -> CoreResult<DeviceView<'_>> {
        let buffer = self.buffer(buf)?;
        let len = buffer.len().checked_sub(offset).ok_or({
            CoreError::OutOfBounds {
                offset,
                len: 0,
                buffer_bytes: buffer.len(),
            }
        })?;
        buffer.slice(offset, len)
    }

    /// The allocation behind one buffer.
    fn buffer(&self, buf: Buf) -> CoreResult<&DeviceBuffer> {
        let index = Buf::ALL
            .iter()
            .position(|other| *other == buf)
            .unwrap_or(usize::MAX);
        self.buffers.get(index).ok_or(CoreError::OutOfBounds {
            offset: index as u64,
            len: 1,
            buffer_bytes: self.buffers.len() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{GdnShape, LayerKind};
    use crate::sparse::shape::{GqaShape, IndexerShape, MoeShape, MropeSections};

    /// The context the cases size against. Only the indexer's score buffer
    /// reads it, and it is deliberately larger than the selection budget: that
    /// is the case where the two differ.
    const CTX: u32 = 16_384;

    fn shape() -> SparseShape {
        SparseShape {
            layers: vec![
                LayerKind::Gdn,
                LayerKind::Gdn,
                LayerKind::Gdn,
                LayerKind::Attn,
            ],
            hidden: 2_560,
            vocab: 248_320,
            streams: 4,
            stream_rank: 320,
            gqa: GqaShape::new(24, 2, 256, 64),
            gdn: GdnShape::new(48, 16, 128, 4),
            moe: MoeShape::new(512, 10, 1, 640),
            indexer: IndexerShape::new(4, 128, 2_048, 4),
            ple: None,
            mrope: MropeSections::new(11, 11, 10),
            rms_eps: 1.0e-6,
            rope_theta: 1.0e7,
        }
    }

    #[test]
    fn every_buffer_is_named_exactly_once_in_the_allocation_order() {
        let mut seen = Buf::ALL.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            Buf::ALL.len(),
            "a buffer listed twice would be allocated twice and read as one"
        );
    }

    #[test]
    fn the_rotary_table_carries_its_prefix_rows() {
        let shape = shape();
        assert_eq!(
            Buf::Mrope.bytes(&shape, 1, 1, CTX),
            (1 + u64::from(MROPE_BACK)) * u64::from(MROPE_AXES) * I32_BYTES
        );
    }

    #[test]
    fn the_widest_buffer_at_a_prefill_chunk_is_the_selected_attention_slab() {
        let shape = shape();
        let (buf, bytes) = widest(&shape, 512, 8, CTX);
        assert_eq!(buf, Buf::AttnPartial);
        // Twenty-four query heads, 512 tokens, eight slices, one slab each.
        assert_eq!(bytes, 24 * 512 * u64::from(ATTN_GSPLITS) * PART_STRIDE * 4);
        // The routed column output is the next one down, and it is the one that
        // grows with the expert count rather than with the head count.
        assert_eq!(
            Buf::MoeDown.bytes(&shape, 512, 8, CTX),
            512 * 10 * 2_560 * 4
        );
    }

    #[test]
    fn a_decode_step_is_dominated_by_the_fixed_split_slab_and_then_the_logits() {
        let shape = shape();
        let (buf, bytes) = widest(&shape, 1, 8, CTX);
        // The split path's slab does not shrink with the row count: it is
        // allocated once for the widest batch that path can take, so that a
        // captured graph never sees its address move. At one row it is
        // therefore larger than everything the row count does scale.
        assert_eq!(buf, Buf::AttnPartial);
        assert_eq!(
            bytes,
            24 * u64::from(ATTN_QT - 1) * u64::from(ATTN_SPLITS) * PART_STRIDE * 4
        );
        // Eight logit rows of a quarter-million vocabulary is the next one down.
        assert_eq!(Buf::Logits.bytes(&shape, 1, 8, CTX), 8 * 248_320 * 4);
    }

    #[test]
    fn a_stack_with_no_n_gram_block_pays_nothing_for_its_buffers() {
        let shape = shape();
        let without = total_bytes(&shape, 512, 8, CTX);
        let with = {
            let mut shape = shape.clone();
            shape.ple = Some(crate::sparse::shape::PleShape::new(
                shiso_core::ids::LayerIdx::new(1),
                3,
                8,
                160,
                4,
            ));
            total_bytes(&shape, 512, 8, CTX)
        };
        assert!(
            with > without,
            "the n-gram buffers cost something when the block exists"
        );
    }

    #[test]
    fn the_sparse_partial_slab_covers_both_paths_that_write_it() {
        let shape = shape();
        // At one row the gathered path wants eight slabs a head; the split path
        // wants forty-eight for each of the fifteen rows it can take. Sizing
        // for the first alone would be an out-of-bounds write on the second.
        assert_eq!(
            Buf::AttnPartial.bytes(&shape, 1, 1, CTX),
            24 * u64::from(ATTN_QT - 1) * u64::from(ATTN_SPLITS) * PART_STRIDE * 4
        );
        // Over a chunk the gathered path is the larger of the two, and from
        // there the buffer grows with the row count.
        assert_eq!(
            Buf::AttnPartial.bytes(&shape, 512, 1, CTX),
            24 * 512 * u64::from(ATTN_GSPLITS) * PART_STRIDE * 4
        );
    }

    #[test]
    fn the_expert_bookkeeping_has_one_slot_past_the_expert_count() {
        let shape = shape();
        assert_eq!(Buf::MoeBounds.bytes(&shape, 512, 8, CTX), (512 + 1) * 4);
        assert_eq!(Buf::MoeActive.bytes(&shape, 1, 1, CTX), (512 + 1) * 4);
    }

    /// A score row spans the context, and the buffer it lives in is sized by the
    /// same number the launch is given. Two places, one formula: when they
    /// disagree the launch reads and writes past the buffer it was handed.
    #[test]
    fn the_score_buffer_is_strided_by_the_context_and_not_by_the_budget() {
        let shape = shape();
        let blocks = shape.indexer.score_blocks(CTX);
        assert!(blocks > shape.indexer.budget_blocks());
        assert_eq!(
            Buf::IdxScore.bytes(&shape, 512, 8, CTX),
            512 * u64::from(blocks) * F32_BYTES
        );
        // The kept list is the budget's, and stays that way.
        assert_eq!(
            Buf::IdxSelected.bytes(&shape, 512, 8, CTX),
            512 * u64::from(shape.indexer.budget_blocks()) * I32_BYTES
        );
    }
}
