//! A resident hybrid stack: its weights, its sessions, and the lifecycle around
//! the forward pass.
//!
//! [`HybridStack::new`] is the only way to build one, and it takes weights that
//! are already on the device, so a value of this type always means a stack that
//! can run. There is no half-built state to guard against and no second
//! initialization step to forget.
//!
//! # What this owns, and what belongs to the crate above it
//!
//! It owns everything that outlives a launch: the weights, the session table, the
//! activation scratch, and the cache the sessions draw pages from. It owns none of
//! the identity - which file the weights came from, what its tensors were called,
//! which token ends a turn. A crate that knows a checkpoint family supplies the
//! geometry and the imported weights; this runs them.
//!
//! The order the blocks run in belongs to [`crate::layers`], and each block family
//! does its own launching. Sampling is not here at all: it is arithmetic over a
//! row the caller has already read back, so it stays a free function.
//!
//! # Chunking is this file's, not the caller's
//!
//! Scratch is sized for [`MAX_PREFILL_CHUNK`] rows. A prompt longer than that is
//! split here rather than refused, because the split is invisible in the result:
//! each sub-chunk advances the session by its own rows and the recurrent state and
//! the cache carry the prefix across the boundary exactly as they carry it across a
//! caller-chosen one.

#[cfg(test)]
#[path = "stack_tests.rs"]
mod tests;

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::Instant;

use shiso_core::backend::Backend;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::{DecodeStep, Logits, PromptRun};
use shiso_core::{DType, DeviceView, GraphKey};
use shiso_kvstore::{CacheStats, CommitOpts, KvCache, Reason};

use crate::blocks::{BlockKernels, Trail};
use crate::kv::{self, DeviceKv};
use crate::launch::F32_BYTES;
use crate::layers::{self, Geometry, PleTable, ResidualIndex, Scratch, Weights};
use crate::options::EngineOptions;
use crate::ple_rows::{PleRows, issue_rows};
use crate::rows::RowPages;
use crate::sessions::Session;
use crate::shape::{HybridShape, MediaPositions, NormScheme};
use crate::vision::{MediaChunk, Streams, VisionPrefill, Walk};

/// Rows one forward pass may carry.
///
/// Every activation buffer is allocated at this width, so it is the one number
/// that decides the scratch bill. A prompt longer than this is split rather than
/// refused.
pub const MAX_PREFILL_CHUNK: u32 = 512;

/// Sessions one decode step may carry, which is also the height of the logits
/// block.
///
/// A batch wider than this is an error rather than a silent split: the caller
/// chose the batch and the rows it gets back are indexed by its own order, so
/// quietly running half of it would misalign every row.
pub const MAX_DECODE_BATCH: u32 = 8;

/// Prompt runs one batched prompt pass may carry, which with
/// [`MAX_DECODE_BATCH`] sets the height of the logits block.
///
/// A pass answers one logits row per run, so this is a bound on rows the
/// block holds rather than on work: sixteen rows of a 65,536-entry vocabulary
/// are four mebibytes. A wider batch is refused rather than split, for the
/// reason [`MAX_DECODE_BATCH`] gives.
pub const MAX_PROMPT_RUNS: u32 = 16;

/// Everything that lives on the device, in construction order.
///
/// Held behind one `Option` rather than one per field: [`Model::teardown`] has to
/// be able to release device memory and report what it could not release, and a
/// field-by-field `Option` would put a "has this been torn down" question at every
/// use site instead of one.
pub(crate) struct Resident {
    pub(crate) weights: Weights,
    pub(crate) kv: DeviceKv,
    pub(crate) cache: Arc<KvCache>,
    pub(crate) scratch: Scratch,
    /// The block a batched step's page tables are packed into.
    ///
    /// Sized once, beside the scratch and for the same reason: a step that
    /// allocated would be a step whose cost depends on the allocator, and a
    /// launch would be able to race a reallocation.
    pub(crate) rows: RowPages,
    /// The per-layer embedding table, for a family that has one.
    ///
    /// Held here rather than in [`Weights`] because a streamed table is a file
    /// descriptor and a reader thread rather than a device allocation, and
    /// because teardown has to release it in construction order with the rest of
    /// the device state.
    pub(crate) ple: Option<PleTable>,
}

/// Host buffers the transfer paths reuse.
///
/// All three are large enough that allocating them per call would dominate the
/// call: a checkpoint blob is tens of mebibytes and a logits row is a megabyte.
#[derive(Debug, Default)]
pub(crate) struct Staging {
    /// The recurrent stack, as a checkpoint blob.
    pub(crate) ckpt: Vec<u8>,
    /// One logits row, for the checkpoint that carries one.
    pub(crate) logits: Vec<u8>,
    /// The chunk's token ids and positions, as `int32` rows.
    ids: Vec<u8>,
    /// The chunk's per-layer embedding rows, still quantized.
    ///
    /// The gather thread reads into a buffer of its own — it cannot borrow this
    /// one — so what happens here is a move rather than a fill: the awaited
    /// bytes land in this slot, the copy reads them, and the buffer they
    /// displaced is freed. That is one allocation a chunk rather than two, and
    /// no copy at all on the host side of the transfer.
    ple: Vec<u8>,
    /// The multimodal position streams and the splice map of the chunk in
    /// flight. Empty, and never written, for a request with no media.
    streams: Streams,
}

/// A multimodal prefill part way through its chunks.
///
/// The walk is the part that cannot be rebuilt per chunk: it carries the running
/// multimodal position, which after a single image is already a different number
/// from the linear one, and it carries how far into the encoder block the splice
/// has got. A prompt split across sub-chunks has to see the same stream it would
/// have seen unsplit, and this is what makes the split invisible.
#[derive(Debug)]
struct MediaRun<'a> {
    prefill: VisionPrefill<'a>,
    walk: Walk<'a>,
}

impl MediaRun<'_> {
    /// What the pass needs to know about the encoder block.
    fn chunk(&self) -> MediaChunk<'_> {
        MediaChunk {
            rows: self.prefill.rows(),
            row_count: self.prefill.row_count(),
        }
    }
}

/// A resident hybrid recurrent/attention stack.
///
/// The fields are `pub(crate)` because the checkpoint tier in [`crate::ckpt`] is this
/// type's other half rather than a client of it: saving a session reads its frontier,
/// its recurrent stack and the cache at the same moment, and a set of accessors that
/// handed those out one borrow at a time would be a longer way of saying the same
/// thing. Nothing outside the crate can see any of them.
pub struct HybridStack {
    pub(crate) backend: Arc<dyn Backend>,
    pub(crate) shape: HybridShape,
    pub(crate) opts: EngineOptions,
    pub(crate) geometry: Geometry,
    /// The block launches this checkpoint family supplies.
    ///
    /// Built once by the crate that knows which family this is and read at
    /// every block. It sits here rather than in [`Geometry`] because
    /// [`Geometry`] is the architecture: two families of this block structure
    /// share every width in it and dispatch different kernels over them.
    pub(crate) kernels: Arc<dyn BlockKernels>,
    /// Recurrent slots, which is how many states a session allocates.
    pub(crate) recurrent_slots: usize,
    pub(crate) resident: Option<Resident>,
    pub(crate) sessions: HashMap<SessionId, Session>,
    /// Order sessions were opened in, so teardown can close them in reverse.
    pub(crate) order: Vec<SessionId>,
    counter: NonZeroU64,
    /// The draft head, once a loader has bound one.
    ///
    /// `None` is the common case and is what makes a step plain: a build with
    /// no head decodes one token a step exactly as it always did.
    pub(crate) draft: Option<crate::draft::DraftHead>,
    /// The block drafter, once a loader has bound one.
    ///
    /// Exclusive with [`HybridStack::draft`]: a stack drafts one way.
    pub(crate) block: Option<crate::block_draft::BlockDrafter>,
    /// The block verified and not yet settled, when there is one.
    pub(crate) pending: Option<PendingBlock>,
    /// The ids no greedy pick this stack takes may land on; see
    /// [`HybridStack::suppress_tokens`].
    pub(crate) suppressed: Vec<TokenId>,
    pub(crate) staging: Staging,
    /// The disk tier, once a caller has turned it on.
    ///
    /// `None` until [`HybridStack::ckpt_enable`] is called, which is the whole of
    /// "checkpointing is off by default": this crate reads no environment and opens
    /// no directory of its own.
    pub(crate) ckpt: Option<crate::ckpt::Checkpoint>,
    /// Start of the clock the cache prices entries against.
    ///
    /// A monotonic origin rather than a wall clock: the cache only ever compares
    /// two of these to each other, and a wall clock that steps backwards would
    /// make a fresh entry look older than one it displaced.
    clock: Instant,
    /// Graph slots this stack has handed out, by the decode shape each belongs
    /// to.
    ///
    /// **A decode step of this family is around 1,100 launches**, and `nsys`
    /// measures 2.4 microseconds of dead device time between consecutive ones
    /// on this box - 2.57 ms of a 19.0 ms step, 13.5% of it, spent by the host
    /// getting out of its own way. A captured graph replays the whole sequence
    /// from one submission.
    ///
    /// The key is `(rows, the ordered session ids)` and the session list is
    /// load-bearing rather than defensive: `layers::forward` hands each GDN
    /// block its session's OWN recurrent state by address, so a captured step
    /// belongs to the exact ordered set of sessions it was captured over. This
    /// is the same rule `sparse/island.rs` states for a batched island, which
    /// declines to capture at all; here the set goes in the key instead,
    /// because the case that matters - one stream, one session, one row - has
    /// a stable set and is the one the throughput target is graded at.
    graph_slots: HashMap<DecodeShape, GraphKey>,
    /// The next unused graph slot number.
    next_graph_slot: u64,
    /// Which scratch buffer a decode pass leaves its residual in.
    ///
    /// Recorded from the pass that captured, and reused by every replay of it.
    /// A replay runs the launches and returns nothing, so this is the one piece
    /// of the pass's result that does not come back out of the device - and it
    /// is a function of the layer list alone, which does not change while a
    /// model is resident. A replay found with none of these recorded would be a
    /// slot committed by something other than the code below, so it refuses
    /// rather than guessing a buffer.
    decode_residual: Option<ResidualIndex>,
    /// The final normalized rows kept for each session a decision head will
    /// read, `hidden` floats per prompt token, in the order they were read.
    captures: HashMap<SessionId, Vec<f32>>,
}

/// A verified block waiting to be told how much of it the caller kept.
#[derive(Debug)]
pub(crate) struct PendingBlock {
    /// Whose round it is.
    pub(crate) session: SessionId,
    /// The token the round committed at its frontier.
    pub(crate) first: TokenId,
    /// The session's frontier before the round.
    pub(crate) frontier: u32,
    /// The drafts the round verified.
    pub(crate) drafts: Vec<TokenId>,
    /// Where they came from.
    pub(crate) source: crate::mtp::DraftSource,
    /// When the round began, for its cost.
    pub(crate) started: Instant,
}

/// What a captured decode step belongs to: its width, its sessions, and
/// whether its rotation reads the three multimodal streams.
///
/// See [`HybridStack::graph_slots`] for why the session list is in here. The
/// rotation is too because the two rules are different launches, and a
/// replay of one for the other would rotate past a picture as though it were
/// text.
type DecodeShape = (u32, Vec<SessionId>, bool);

impl std::fmt::Debug for HybridStack {
    /// Names the model by its geometry and its live sessions.
    ///
    /// Never the buffers: a device allocation has no host-readable content, so
    /// printing one would produce an identifier that means nothing to a reader and
    /// a line long enough to bury the fields that do.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridStack")
            .field("backend", &self.backend.backend_name())
            .field("shape", &self.shape)
            .field("opts", &self.opts)
            .field("geometry", &self.geometry)
            .field("kernels", &self.kernels)
            .field("recurrent_slots", &self.recurrent_slots)
            .field("resident", &self.resident.is_some())
            .field("sessions", &self.sessions.len())
            .field("order", &self.order)
            .field("counter", &self.counter)
            .field("draft", &self.draft.is_some())
            .field("block", &self.block.is_some())
            .field("pending", &self.pending.as_ref().map(|p| p.session))
            .field("suppressed", &self.suppressed.len())
            .field("staging", &self.staging)
            .field("ckpt", &self.ckpt)
            .field("clock", &self.clock)
            .field("graph_slots", &self.graph_slots.len())
            .field("next_graph_slot", &self.next_graph_slot)
            .field("decode_residual", &self.decode_residual)
            .field("captures", &self.captures.len())
            .finish()
    }
}

/// The pieces of `layout` as the pass reads them, each in its run's session
/// and the trunk forked into `forks`.
fn segment_runs<'s>(
    runs: &[PromptRun<'_>],
    layout: &RunLayout,
    sessions: &'s HashMap<SessionId, Session>,
    forks: &'s [&'s Session],
) -> CoreResult<Vec<layers::SegmentRun<'s>>> {
    layout
        .pieces
        .iter()
        .map(|piece| {
            let id = runs[piece.run].session;
            Ok(layers::SegmentRun {
                session: sessions.get(&id).ok_or(CoreError::UnknownSession(id))?,
                start: piece.start,
                rows: piece.rows,
                forks: if piece.trunk { forks } else { &[] },
            })
        })
        .collect()
}

/// Shortest opening several fresh prompts must share before a batched pass
/// reads it once: below this the copies that fork it cost about what the
/// rows they save do.
const MIN_SHARED_TRUNK: usize = 8;

/// One piece of a batched prompt pass: rows of one run, read in its session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunPiece {
    /// Which run the rows belong to.
    run: usize,
    /// The session position of the first row.
    start: u32,
    /// How many rows.
    rows: u32,
    /// Whether this is the shared trunk, forked into every other run.
    trunk: bool,
}

/// How a batched prompt pass lays its runs out.
///
/// Runs whose sessions are all fresh and whose prompts open with the same
/// tokens read that opening once: it runs in the first run's session and is
/// forked into the others, and each run then reads only what follows it. A
/// run keeps at least one row of its own, which is the row its answer is read
/// from.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RunLayout {
    /// The ids of every row, in pass order.
    tokens: Vec<TokenId>,
    /// The session position of every row, in pass order.
    positions: Vec<u32>,
    /// The pieces, in pass order.
    pieces: Vec<RunPiece>,
    /// Tokens of the shared trunk, zero when nothing is shared.
    shared: u32,
    /// Each run's last row in the pass, in run order.
    last_rows: Vec<u32>,
}

impl RunLayout {
    /// Lays `runs` out over sessions standing at `starts`.
    fn new(runs: &[PromptRun<'_>], starts: &[u32]) -> Self {
        let fresh = runs.len() > 1 && starts.iter().all(|&at| at == 0);
        let shared = if fresh {
            let shortest = runs.iter().map(|run| run.tokens.len()).min().unwrap_or(0);
            let first = runs.first().map_or(&[][..], |run| run.tokens);
            let common = (0..shortest.saturating_sub(1))
                .take_while(|&at| runs.iter().all(|run| run.tokens[at] == first[at]))
                .count();
            if common >= MIN_SHARED_TRUNK {
                common
            } else {
                0
            }
        } else {
            0
        };
        let mut layout = Self {
            tokens: Vec::new(),
            positions: Vec::new(),
            pieces: Vec::new(),
            shared: u32::try_from(shared).unwrap_or(u32::MAX),
            last_rows: Vec::with_capacity(runs.len()),
        };
        if shared > 0 {
            layout.push(0, &runs[0].tokens[..shared], starts[0], true);
        }
        for (index, (run, start)) in runs.iter().zip(starts).enumerate() {
            let from = start.saturating_add(layout.shared);
            layout.push(index, &run.tokens[shared..], from, false);
            layout.last_rows.push(layout.rows().saturating_sub(1));
        }
        layout
    }

    /// Appends one piece.
    fn push(&mut self, run: usize, tokens: &[TokenId], start: u32, trunk: bool) {
        let rows = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        self.tokens.extend_from_slice(tokens);
        self.positions
            .extend((0..rows).map(|row| start.saturating_add(row)));
        self.pieces.push(RunPiece {
            run,
            start,
            rows,
            trunk,
        });
    }

    /// Rows the pass carries.
    fn rows(&self) -> u32 {
        u32::try_from(self.tokens.len()).unwrap_or(u32::MAX)
    }

    /// The rotary position of every row, in pass order: its session position
    /// less its run's entry of `shifts`.
    fn rotations(&self, shifts: &[u32]) -> Vec<u32> {
        self.pieces
            .iter()
            .flat_map(|piece| {
                let shift = shifts.get(piece.run).copied().unwrap_or(0);
                (0..piece.rows)
                    .map(move |row| piece.start.saturating_add(row).saturating_sub(shift))
            })
            .collect()
    }
}

/// Uploads the three per-row inputs a decode step reads: the token ids, the
/// positions they land at, and the bound each row may attend up to.
///
/// Lifted out of [`HybridStack::run_batch`] so that the launch sequence there
/// reads as one block, which matters more than it used to: everything above it
/// writes resident scratch at fixed addresses and stays OUTSIDE the capture,
/// and everything below goes in. A reader has to be able to see where that line
/// falls.
/// Each run's rotary shift, in run order: past a picture a run's rows rotate
/// at their position less the shift the picture left, as its solo read's
/// would. Zero under the linear position rule, which has no such shift.
fn run_shifts(
    sessions: &HashMap<SessionId, Session>,
    geometry: &Geometry,
    runs: &[PromptRun<'_>],
) -> CoreResult<Vec<u32>> {
    runs.iter()
        .map(|run| {
            let live = sessions
                .get(&run.session)
                .ok_or(CoreError::UnknownSession(run.session))?;
            Ok(if geometry.media_positions == MediaPositions::Mrope3D {
                live.rope_shift()
            } else {
                0
            })
        })
        .collect()
}

/// Uploads a batched prompt pass's rows as [`upload_decode_rows`] does, and
/// its three rotary streams when a run of it stands past a picture - each row
/// at its position less its run's entry of `shifts` - and says whether it
/// uploaded those. Only such a pass takes the three streams, so a text-only
/// batch launches what it always did.
fn upload_run_rows(
    backend: &dyn Backend,
    staging: &mut Vec<u8>,
    scratch: &Scratch,
    layout: &RunLayout,
    shifts: &[u32],
) -> CoreResult<bool> {
    upload_decode_rows(backend, staging, scratch, &layout.tokens, &layout.positions)?;
    if shifts.iter().all(|shift| *shift == 0) {
        return Ok(false);
    }
    let rotations = layout.rotations(shifts);
    for dst in scratch.mrope() {
        upload_row(
            backend,
            staging,
            dst,
            rotations
                .iter()
                .map(|at| i32::try_from(*at).unwrap_or(i32::MAX)),
        )?;
    }
    Ok(true)
}

fn upload_decode_rows(
    backend: &dyn Backend,
    staging: &mut Vec<u8>,
    scratch: &Scratch,
    tokens: &[TokenId],
    positions: &[u32],
) -> CoreResult<()> {
    upload_row(
        backend,
        staging,
        scratch.ids(),
        tokens.iter().map(|t| i32::try_from(*t).unwrap_or(0)),
    )?;
    upload_row(
        backend,
        staging,
        scratch.positions(),
        positions
            .iter()
            .map(|at| i32::try_from(*at).unwrap_or(i32::MAX)),
    )?;
    // A decode row is causal and text-only: what it may read is its own
    // position and everything before it, which is the position row again.
    upload_row(
        backend,
        staging,
        scratch.bound(),
        positions
            .iter()
            .map(|at| i32::try_from(*at).unwrap_or(i32::MAX)),
    )
}

/// What one pool page and one session of this stack cost on the device.
///
/// A family that drafts binds its head after the stack is built, with a pool
/// of the same page count, so a speculative stack counts one more attention
/// layer's key and value runs per page for it: the head is one layer.
fn pool_cost(
    shape: &HybridShape,
    layout: &kv::KvLayout,
    cache: &shiso_kvstore::CacheGeometry,
    opts: EngineOptions,
) -> crate::options::PoolCost {
    let head = if opts.spec.is_speculative() {
        layout
            .run_bytes
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .saturating_mul(kv::RUNS_PER_LAYER)
    } else {
        0
    };
    let records = u64::from(opts.spec.rollback_slots());
    let recurrent = u64::from(shape.recurrent_count())
        .saturating_mul(kv::conv_state_bytes(shape).saturating_add(kv::state_bytes(shape)));
    crate::options::PoolCost {
        page_bytes: layout.page_bytes().saturating_add(head),
        session_bytes: recurrent.saturating_mul(records.saturating_add(1)),
        fixed_bytes: u64::from(opts.kv_ckpt_slots).saturating_mul(cache.ckpt_bytes),
    }
}

impl HybridStack {
    /// Builds a stack over weights that are already on the device.
    ///
    /// The weights arrive imported rather than as a path, and that is the whole
    /// seam: which file they came from, what its tensors were called and whether
    /// the bytes were wrapped or copied are decisions this crate has no way to
    /// make and no reason to. What it needs is the geometry and the resident
    /// tensors, and with both in hand a stack either exists and can run or the
    /// construction failed.
    ///
    /// `embed` is the embedding table's storage format, which selects the
    /// row-gather entry point. It is an argument rather than a field of the shape
    /// because it is a property of the file rather than of the architecture.
    ///
    /// `kernels` is the family's own block launcher. It arrives built rather
    /// than chosen here for the same reason the weights do: nothing in this
    /// crate names a checkpoint family, and picking the kernels would be
    /// naming one.
    ///
    /// `ple` is the per-layer embedding table, and it has to agree with the
    /// shape: a table handed to a stack that states none would be read by
    /// nothing, and a shape that states one without a table would dispatch a
    /// branch over a buffer nobody filled. Both are refused here rather than
    /// discovered at the first chunk.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when no row-gather kernel reads `embed` or the
    /// per-layer rows are in a format no prepare kernel decodes,
    /// [`CoreError::ShapeMismatch`] when the weight list and the shape disagree
    /// about the layer count or a slot's kind or about the per-layer table, and
    /// [`CoreError::Backend`] when a device allocation fails.
    pub fn new(
        backend: &Arc<dyn Backend>,
        shape: HybridShape,
        weights: Weights,
        embed: DType,
        kernels: Arc<dyn BlockKernels>,
        ple: Option<PleTable>,
        opts: EngineOptions,
    ) -> CoreResult<Self> {
        // Before anything is derived from it: every helper below reads the shape
        // without re-checking it, so the one place that does check is here.
        //
        // Two different questions, asked in this order. The first is whether the
        // shape contradicts itself, which is wrong for every family. The second
        // is whether THESE kernels can run a shape that is perfectly coherent -
        // a head wider than their staging, say - which only the family that owns
        // them can answer, and which no kernel can refuse once it is launched.
        shape.validate()?;
        kernels.validate(&shape)?;
        let slots = u64::try_from(weights.layers.len()).unwrap_or(u64::MAX);
        if slots != u64::from(shape.n_layers()) {
            return Err(CoreError::ShapeMismatch {
                field: "layer count",
                expected: u64::from(shape.n_layers()),
                got: slots,
            });
        }
        // The shape and the table are two statements about the same fact, and a
        // pass that met them disagreeing would either read a buffer nobody
        // filled or hold a table nothing reads.
        if shape.ple.is_some() != ple.is_some() {
            return Err(CoreError::ShapeMismatch {
                field: "per-layer table",
                expected: u64::from(shape.ple.is_some()),
                got: u64::from(ple.is_some()),
            });
        }
        // The branch writes the post-norm scratch, which only a stack that
        // normalizes its blocks' outputs allocates. Refused here rather than at
        // the first prefill, because a stack that cannot run its own shape
        // should not become a value that says it can.
        if shape.ple.is_some() && shape.norms != NormScheme::PrePost {
            return Err(CoreError::Unsupported {
                what: "a per-layer embedding table on a stack that does not normalize its blocks' \
                       outputs, whose post-norm scratch the branch's projection needs",
                backend: "engine",
            });
        }
        let geometry = Geometry::new(&shape, embed)?;
        validate_ple(&geometry, ple.as_ref())?;
        let recurrent_slots = usize::try_from(shape.recurrent_count()).unwrap_or(0);
        let scratch = Scratch::new(
            backend,
            &geometry,
            MAX_PREFILL_CHUNK,
            MAX_DECODE_BATCH.max(MAX_PROMPT_RUNS),
        )?;
        let rows = RowPages::new(backend, opts.ctx, MAX_DECODE_BATCH)?;
        // The pool after the scratch, so that a pool sized to the device is
        // sized from what the weights and the scratch left of it.
        let mut opts = opts;
        let layout = kv::KvLayout::from_shape(&shape);
        let cache_geometry = kv::geometry(&shape);
        opts.fit_pool(
            backend.as_ref(),
            pool_cost(&shape, &layout, &cache_geometry, opts),
        )?;
        let kv = DeviceKv::open(backend, &layout, opts.kv_pool_pages)?;
        let cache = KvCache::open(
            &cache_geometry,
            Box::new(kv.clone()),
            opts.kv_pool_pages,
            opts.kv_ckpt_slots,
            None,
        )
        .map_err(cache_failure)?;
        tracing::info!(
            layers = shape.n_layers(),
            recurrent = recurrent_slots,
            ctx = opts.ctx,
            pool_pages = opts.kv_pool_pages,
            "stack resident"
        );
        Ok(Self {
            backend: Arc::clone(backend),
            shape,
            opts,
            geometry,
            kernels,
            recurrent_slots,
            resident: Some(Resident {
                weights,
                kv,
                cache: Arc::new(cache),
                scratch,
                rows,
                ple,
            }),
            sessions: HashMap::new(),
            order: Vec::new(),
            counter: NonZeroU64::MIN,
            draft: None,
            block: None,
            pending: None,
            suppressed: Vec::new(),
            staging: Staging::default(),
            ckpt: None,
            clock: Instant::now(),
            graph_slots: HashMap::new(),
            next_graph_slot: 0,
            decode_residual: None,
            captures: HashMap::new(),
        })
    }

    /// Every width this stack derives a launch from.
    ///
    /// Public because a draft head is built at the target's widths and by the
    /// crate that knows the family, and that crate should not be re-deriving
    /// them from the shape it already handed over.
    #[must_use]
    pub const fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// The device half, or the typed failure a torn-down stack answers with.
    pub(crate) fn resident(&self) -> CoreResult<&Resident> {
        self.resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })
    }

    /// Seconds since the model was loaded, which is the clock the cache prices
    /// entries against.
    pub(crate) fn now(&self) -> f64 {
        self.clock.elapsed().as_secs_f64()
    }

    /// Runs one chunk through the stack, leaving the normalized rows in scratch.
    ///
    /// The session is advanced first: the pass writes cache rows, and a row that
    /// has not been claimed may be handed to another session between the claim and
    /// the launch that fills it.
    fn run_chunk(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: Option<&mut MediaRun<'_>>,
    ) -> CoreResult<u32> {
        self.run_rows(session, tokens, media, Trail::None)
    }

    /// Runs one chunk, and says whether the recurrent slots record a per-row
    /// trail as they go.
    ///
    /// `trail` is [`Trail::Record`] only for a verify pass, which is the only
    /// pass a rejection can wind back. It is a separate entry point rather
    /// than an argument on [`HybridStack::run_chunk`] so that neither the
    /// prefill path nor the decode loop has to name a value that is
    /// [`Trail::None`] for both of them.
    #[expect(
        clippy::too_many_lines,
        reason = "chunk execution coordinates token uploads, PLE staging and layer forward pass"
    )]
    fn run_rows(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: Option<&mut MediaRun<'_>>,
        trail: Trail,
    ) -> CoreResult<u32> {
        let n_tok = u32::try_from(tokens.len()).map_err(|_| CoreError::ContextOverflow {
            session,
            cap: self.opts.ctx,
        })?;
        let capacity = self.opts.ctx;
        let now = self.now();
        let Self {
            backend,
            geometry,
            kernels,
            resident,
            sessions,
            staging,
            block,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        let live = sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        // Before the session is advanced: a refusal in there must leave the
        // session exactly as it found it. The reads go first for the opposite
        // reason - the enqueue below is what hides their latency.
        let ple_tokens: Vec<TokenId> = if let Some(run) = &media {
            let pads = run.prefill.pads();
            tokens
                .iter()
                .map(|&t| if pads.matches(t) { TokenId::new(0) } else { t })
                .collect()
        } else {
            tokens.to_vec()
        };
        let rows = open_ple(resident, geometry, &ple_tokens)?;
        let start = live.written();
        // A text chunk past a picture rotates at its sequence position less
        // the shift the picture left; a media chunk's walk already carries it.
        let shift = if media.is_none() && geometry.media_positions == MediaPositions::Mrope3D {
            live.rope_shift()
        } else {
            0
        };
        let room = kv::Room {
            cache: &resident.cache,
            now,
        };
        live.advance(tokens, capacity, room)?;
        upload_row(
            backend.as_ref(),
            &mut staging.ids,
            resident.scratch.ids(),
            ple_tokens.iter().map(|t| i32::try_from(*t).unwrap_or(0)),
        )?;
        upload_row(
            backend.as_ref(),
            &mut staging.ids,
            resident.scratch.positions(),
            (0..n_tok).map(|t| i32::try_from(start.saturating_add(t)).unwrap_or(i32::MAX)),
        )?;
        // The four multimodal rows come out of one walk, so the positions and
        // the row the splice copies cannot disagree about where a placeholder
        // run began.
        let media = match media {
            None => {
                // A text chunk is causal: every token may read every key up to
                // its own position and no further, which is the position row
                // again. One four-byte-a-token upload rather than a branch at
                // the launch, so the block is handed one row whatever the
                // request was.
                upload_row(
                    backend.as_ref(),
                    &mut staging.ids,
                    resident.scratch.bound(),
                    (0..n_tok).map(|t| i32::try_from(start.saturating_add(t)).unwrap_or(i32::MAX)),
                )?;
                if shift > 0 {
                    // Text has one rotary position on every stream.
                    for dst in resident.scratch.mrope() {
                        upload_row(
                            backend.as_ref(),
                            &mut staging.ids,
                            dst,
                            (0..n_tok).map(|t| {
                                i32::try_from(start.saturating_add(t).saturating_sub(shift))
                                    .unwrap_or(i32::MAX)
                            }),
                        )?;
                    }
                }
                None
            }
            Some(run) => {
                staging.streams.clear();
                run.walk.chunk(tokens, &mut staging.streams);
                let scratch = &resident.scratch;
                // The three streams are uploaded whichever rule the family
                // states - under the linear one they are the position row
                // repeated, and the rotation is not pointed at them. Three
                // adjacent four-byte-a-token copies against a branch here, and
                // what decides whether they are read is the geometry.
                for (dst, stream) in scratch.mrope().into_iter().zip([
                    &staging.streams.t,
                    &staging.streams.h,
                    &staging.streams.w,
                ]) {
                    upload_row(
                        backend.as_ref(),
                        &mut staging.ids,
                        dst,
                        stream.iter().copied(),
                    )?;
                }
                // No bound may name a cache row this chunk has not written:
                // the rows past it hold whatever the previous occupant of those
                // pages left, and attention over them is fluent and wrong.
                //
                // Nothing the walk emits can. Both of its `out.bound.push`
                // sites in `vision::mrope` push the token's OWN sequence
                // position, and a chunk has by definition written every row up
                // to its own last token. This line used to clamp, because a
                // media item's tokens were once widened to the item's last
                // position and an item wider than a whole chunk then put the
                // first half's bound past what that half had written. Attention
                // is causal on both position rules now, the widening is gone,
                // and the clamp went with it - so what is left here is the
                // assertion that the property it used to enforce still holds.
                debug_assert!(
                    staging.streams.bound.iter().all(|&at| {
                        u32::try_from(at).is_ok_and(|at| at < start.saturating_add(n_tok))
                    }),
                    "a bound names a cache row this chunk has not written: \
                     chunk covers {start}..{}, bounds {:?}",
                    start.saturating_add(n_tok),
                    staging.streams.bound,
                );
                upload_row(
                    backend.as_ref(),
                    &mut staging.ids,
                    scratch.bound(),
                    staging.streams.bound.iter().copied(),
                )?;
                upload_row(
                    backend.as_ref(),
                    &mut staging.ids,
                    scratch.splice_map(),
                    staging.streams.map.iter().copied(),
                )?;
                Some(run.chunk())
            }
        };
        // The await site: everything above has overlapped the reads.
        if let Some(rows) = rows {
            stage_ple(backend.as_ref(), resident, &mut staging.ple, rows)?;
        }
        let residual = layers::forward(
            backend.as_ref(),
            &resident.weights,
            geometry,
            &resident.scratch,
            kernels.as_ref(),
            &resident.kv,
            layers::PassRows::Chunk(live),
            resident.ple.as_ref(),
            n_tok,
            media,
            layers::PassExtras {
                trail,
                taps: block.as_ref().map(crate::block_draft::BlockDrafter::sink),
                rotary: shift > 0,
            },
        )?;
        // A plain pass's rows are committed as they run, so they become the
        // drafter's context now. A verify pass's are not: which of them stay
        // is decided after it, and the drafter absorbs exactly those.
        if !trail.records()
            && let Some(drafter) = block.as_mut()
        {
            drafter.absorb(backend, live.kv().view(), 0, n_tok, start)?;
        }
        live.push_history(tokens);
        // Recorded on the session and not on the stack: the row a draft head
        // starts from is this session's last, and a stack-wide latch would name
        // whichever session ran last instead. `n_tok` rows means row
        // `n_tok - 1` is the last.
        live.record_pass(residual, n_tok.saturating_sub(1));
        // Any pass that is not a verify leaves a residual of its own, so the
        // row the head staged is one the session has moved past. Re-staged on
        // the next speculative step rather than here, because a build with no
        // head would otherwise be paying a copy per chunk for a buffer nothing
        // reads.
        if !trail.records()
            && let Some(head) = self.draft.as_mut()
        {
            head.stale();
        }
        Ok(n_tok)
    }

    /// Runs one decode step over `steps` as a single pass of `steps.len()`
    /// rows.
    ///
    /// This is the whole of what Phase 4 changed, so it is worth stating what
    /// the old shape was and why it cost what it did. `decode_batch` used to
    /// call [`HybridStack::run_chunk`] once per member: one complete forward
    /// pass per row, with the logits rows stitched together afterwards. Every
    /// projection therefore saw `n_tok = 1`, took the matvec branch, and
    /// **re-read the whole weight matrix once per session** - so four
    /// concurrent streams read a two-gigabyte checkpoint four times a step and
    /// aggregate throughput was the same at eight streams as at one.
    ///
    /// One pass over B rows reads it once. What that costs instead is stated
    /// where it is paid: [`crate::blocks::GdnStates::PerRow`] for the recurrent
    /// state, which is per session and does not amortize, and
    /// [`crate::attn::Pages`] for the page table, which is per session and
    /// costs one small copy a row.
    ///
    /// # The order below is the whole correctness argument
    ///
    /// 1. **Advance every member**, which claims the page its next token writes
    ///    and re-uploads that session's own table. A refusal here has advanced
    ///    some members and not others, which is why `decode_batch` checks every
    ///    handle before calling this.
    /// 2. **Write the inputs** - the ids, the positions and the causal bounds -
    ///    from the frontiers the advance just moved.
    /// 3. **Pack the page tables**, after the advance and before anything is
    ///    issued. `sparse/run.rs`'s rule, for the reason `rows.rs` gives:
    ///    packed before the advance the block holds each table as it was one
    ///    token ago, which is right until the step crosses a page boundary;
    ///    packed after the pass every launch has already read whatever the slot
    ///    last held.
    /// 4. **Run the pass**, and record on each session the residual row it can
    ///    be drafted from.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`] for a handle this stack does not hold,
    /// [`CoreError::ContextOverflow`] when a member's step reaches past the
    /// context, and whatever the device reports.
    /// The graph slot a decode step of this shape belongs to, allocating one
    /// on first sight.
    ///
    /// `None` declines capture, and there are two reasons to:
    ///
    /// - **The backend offers no graphs.** Nothing to file under.
    /// - **The slot table is full.** `GRAPH_CACHE_CAP` bounds the device side
    ///   of this; the bound here is on the host map, and it exists so that a
    ///   caller whose session set really is unbounded - a server opening and
    ///   closing a session per request - stops allocating slot numbers rather
    ///   than growing a map forever. Past the bound every step runs eagerly,
    ///   which is what it did before this existed.
    ///
    /// The shape includes the ordered session ids and not merely their count;
    /// [`HybridStack::graph_slots`] says why.
    fn decode_slot(&mut self, n_tok: u32, steps: &[DecodeStep]) -> Option<GraphKey> {
        /// Distinct decode shapes one stack will file graphs for.
        ///
        /// Generous against the shapes a decode step actually takes - eight
        /// widths times the session sets a batch is assembled from - and far
        /// under the device cache's own cap, so this is the bound that trips
        /// first and it trips as a slowdown rather than a device error.
        const MAX_SLOTS: usize = 256;

        self.backend.as_graph()?;
        let rotary = matches!(self.shape.media_positions, MediaPositions::Mrope3D)
            && steps.iter().any(|step| {
                self.sessions
                    .get(&step.session)
                    .is_some_and(|live| live.rope_shift() > 0)
            });
        let shape: DecodeShape = (
            n_tok,
            steps.iter().map(|step| step.session).collect(),
            rotary,
        );
        if let Some(key) = self.graph_slots.get(&shape) {
            return Some(*key);
        }
        if self.graph_slots.len() >= MAX_SLOTS {
            return None;
        }
        let key = GraphKey::new(self.next_graph_slot);
        self.next_graph_slot = self.next_graph_slot.saturating_add(1);
        self.graph_slots.insert(shape, key);
        Some(key)
    }

    /// Drops every captured decode step and the slots they were filed under.
    ///
    /// Called wherever an address a capture may have baked in can have moved.
    /// A captured graph names the addresses it was captured with and nothing
    /// updates them afterwards, so the cost of being wrong here is not a slow
    /// step but a step that reads a freed buffer.
    ///
    /// The slot numbers go with the graphs rather than being reused: a slot
    /// whose graph was invalidated but whose number a later shape inherited
    /// would replay one shape's launches for another.
    pub(crate) fn invalidate_decode_graphs(&mut self) {
        if let Some(graphs) = self.backend.as_graph() {
            graphs.invalidate_all();
        }
        self.graph_slots.clear();
        self.decode_residual = None;
    }

    #[expect(
        clippy::too_many_lines,
        reason = "a batched pass coordinates the advance, the row uploads, the page-table pack and the captured forward, in that order"
    )]
    fn run_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<()> {
        let n_tok = u32::try_from(steps.len()).unwrap_or(u32::MAX);
        let capacity = self.opts.ctx;
        let tokens: Vec<TokenId> = steps.iter().map(|step| step.token).collect();
        // Before `self` is taken apart: the slot this step's launches belong
        // to, allocated on first sight of a shape and reused forever after.
        let slot = self.decode_slot(n_tok, steps);
        let now = self.now();
        let taps_sink = self
            .block
            .as_ref()
            .map(crate::block_draft::BlockDrafter::sink);
        let Self {
            backend,
            geometry,
            kernels,
            resident,
            sessions,
            staging,
            decode_residual,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        // Before any session is advanced: the one refusal this can make must
        // leave every member exactly as it found it.
        let rows = open_ple(resident, geometry, &tokens)?;
        // Each member's own next position, read before the advance moves it,
        // and where it rotates: past a picture, that position less the shift
        // the picture left.
        let mut positions = Vec::with_capacity(steps.len());
        let mut rotations = Vec::with_capacity(steps.len());
        for step in steps {
            let live = sessions
                .get_mut(&step.session)
                .ok_or(CoreError::UnknownSession(step.session))?;
            positions.push(live.written());
            let shift = if geometry.media_positions == MediaPositions::Mrope3D {
                live.rope_shift()
            } else {
                0
            };
            rotations.push(live.written().saturating_sub(shift));
            let room = kv::Room {
                cache: &resident.cache,
                now,
            };
            live.advance(std::slice::from_ref(&step.token), capacity, room)?;
        }
        upload_decode_rows(
            backend.as_ref(),
            &mut staging.ids,
            &resident.scratch,
            &tokens,
            &positions,
        )?;
        // Only a step with a member past a picture takes the three streams,
        // so a text-only batch launches what it always did.
        let rotary = rotations.iter().zip(&positions).any(|(r, p)| r != p);
        if rotary {
            for dst in resident.scratch.mrope() {
                upload_row(
                    backend.as_ref(),
                    &mut staging.ids,
                    dst,
                    rotations
                        .iter()
                        .map(|at| i32::try_from(*at).unwrap_or(i32::MAX)),
                )?;
            }
        }
        // Immutable from here: the pass reads every member at once, and the
        // advance above is the last thing that could have moved one.
        let members: Vec<&Session> = steps
            .iter()
            .map(|step| {
                sessions
                    .get(&step.session)
                    .ok_or(CoreError::UnknownSession(step.session))
            })
            .collect::<CoreResult<_>>()?;
        let pages = resident
            .rows
            .pack(backend.as_ref(), members.iter().map(|m| m.kv().view()))?;
        if let Some(rows) = rows {
            stage_ple(backend.as_ref(), resident, &mut staging.ple, rows)?;
        }
        // Everything above this line is host work against fixed addresses -
        // three row uploads into resident scratch and a page-table pack into a
        // resident block - so it stays OUTSIDE the capture and a replay reads
        // whatever it last wrote. What goes inside is the launch sequence
        // alone.
        //
        // `encode` is called at most once per step by `island::run`, and a
        // capture RECORDS launches rather than running them - which is the
        // property this pass needs and would otherwise have to check for
        // itself. The GDN blocks update their recurrent state in place, so a
        // protocol that ran the region twice on a failed capture would advance
        // every recurrent slot twice and answer wrongly rather than slowly.
        let mut produced = None;
        let ran = crate::sparse::island::run(backend.as_graph(), slot, || {
            produced = Some(layers::forward(
                backend.as_ref(),
                &resident.weights,
                geometry,
                &resident.scratch,
                kernels.as_ref(),
                &resident.kv,
                layers::PassRows::Batch {
                    members: &members,
                    pages,
                },
                resident.ple.as_ref(),
                n_tok,
                None,
                // A batch records no trail: a rollback winds one session back
                // along one sequence, and these rows are not one sequence. The
                // recurrent family refuses the combination; nothing here builds
                // it.
                layers::PassExtras {
                    trail: Trail::None,
                    taps: taps_sink,
                    rotary,
                },
            )?);
            Ok(())
        })?;
        drop(members);
        let residual = match produced {
            Some(index) => {
                *decode_residual = Some(index);
                index
            }
            // A replay ran the launches and returned nothing, so the one part
            // of the pass's result that does not come back out of the device
            // comes from the pass that captured. It is a function of the layer
            // list alone and the list does not change while a model is
            // resident. `None` here would mean a slot was committed by
            // something other than the branch above, which is a refusal rather
            // than a guess at a buffer.
            None => decode_residual.ok_or(CoreError::Unsupported {
                what: "a replayed decode step with no residual recorded for it",
                backend: "engine",
            })?,
        };
        tracing::trace!(rows = n_tok, ?ran, "decode step");
        // Row `i` of the pass is `steps[i]`, so that is the row session `i` can
        // be drafted from - and it is recorded on the session rather than on
        // the stack, which is what stops a speculative step for one member
        // reading the hidden state of whichever member happened to run last.
        for (row, step) in steps.iter().enumerate() {
            let row = u32::try_from(row).unwrap_or(u32::MAX);
            let live = sessions
                .get_mut(&step.session)
                .ok_or(CoreError::UnknownSession(step.session))?;
            live.push_history(std::slice::from_ref(&step.token));
            live.record_pass(residual, row);
        }
        // The head's staged row is now a row every member has moved past.
        if let Some(head) = self.draft.as_mut() {
            head.stale();
        }
        Ok(())
    }

    /// Hands a block drafter each member's row of the batch that just ran.
    ///
    /// Every row of a plain decode step is committed, and is its session's
    /// context from now on - at that session's own position, which the advance
    /// just moved one past, through that session's own page table.
    fn absorb_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<()> {
        let Self {
            backend,
            sessions,
            block,
            ..
        } = self;
        let Some(drafter) = block.as_mut() else {
            return Ok(());
        };
        for (row, step) in steps.iter().enumerate() {
            let live = sessions
                .get(&step.session)
                .ok_or(CoreError::UnknownSession(step.session))?;
            let at = live.written().saturating_sub(1);
            let row = u32::try_from(row).unwrap_or(u32::MAX);
            drafter.absorb(backend, live.kv().view(), row, 1, at)?;
        }
        Ok(())
    }

    /// Runs one verify pass: `rows` rows, recording a per-row trail.
    ///
    /// A thin name over [`HybridStack::run_rows`] so that `draft.rs` does not
    /// reach for a private entry point with a boolean nobody else passes.
    pub(crate) fn run_verify_rows(
        &mut self,
        session: SessionId,
        rows: &[TokenId],
    ) -> CoreResult<()> {
        self.run_rows(session, rows, None, Trail::Record)
            .map(|_rows| ())
    }

    /// Projects rows `0 .. n` of the pass that just ran into logits rows
    /// `0 .. n`, in one call.
    pub(crate) fn write_logits_rows(&self, n: u32) -> CoreResult<()> {
        let resident = self.resident()?;
        layers::logits_rows(
            self.backend.as_ref(),
            &resident.weights,
            &self.geometry,
            &resident.scratch,
            n,
        )
    }

    /// Projects row `token` of the pass that just ran into logits row `row`.
    pub(crate) fn write_logits_row(&self, token: u32, row: u32) -> CoreResult<()> {
        self.write_logits(token, row)
    }

    /// Registers the span a speculative step settled on.
    ///
    /// No logits row travels with it. The row a resume would want is the one
    /// the step's own caller is about to sample from, and it is written into
    /// the block at an index chosen by the verify pass rather than by the
    /// batch - so handing the cache one of them would be handing it a row
    /// indexed by the wrong thing.
    pub(crate) fn commit_step(&mut self, session: SessionId) -> CoreResult<()> {
        self.commit(session, false, None)
    }

    /// Appends the chunk's final normalized rows to `session`'s capture, when
    /// a decision head asked for them.
    fn capture_rows(&mut self, session: SessionId, n_tok: u32) -> CoreResult<()> {
        let Some(rows) = self.captures.get_mut(&session) else {
            return Ok(());
        };
        let hidden = self.geometry.hidden as usize;
        let take = hidden * n_tok as usize;
        let resident = self.resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        let view = resident
            .scratch
            .normed()
            .slice(0, (take * size_of::<f32>()) as u64)?;
        let mut bytes = vec![0u8; take * size_of::<f32>()];
        self.backend.copy_d2h(view, &mut bytes)?;
        rows.extend(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b)),
        );
        Ok(())
    }

    /// Keeps the final normalized row of every token `session` reads from now
    /// on, for a decision head to read once the prompt is complete.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`] for a closed or unknown id.
    pub fn capture_hidden(&mut self, session: SessionId) -> CoreResult<()> {
        if !self.sessions.contains_key(&session) {
            return Err(CoreError::UnknownSession(session));
        }
        self.captures.insert(session, Vec::new());
        Ok(())
    }

    /// The rows [`HybridStack::capture_hidden`] kept for `session`, which stops
    /// keeping them: `hidden` floats per token read since the capture began.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`] when nothing was being captured for it.
    pub fn take_hidden(&mut self, session: SessionId) -> CoreResult<Vec<f32>> {
        self.captures
            .remove(&session)
            .ok_or(CoreError::UnknownSession(session))
    }

    /// Writes one logits row from the chunk's last normalized row.
    fn write_logits(&self, token: u32, row: u32) -> CoreResult<()> {
        let resident = self.resident()?;
        layers::logits_row(
            self.backend.as_ref(),
            &resident.weights,
            &self.geometry,
            &resident.scratch,
            token,
            row,
        )
    }

    /// Advances a session by `tokens`, splitting anything wider than the scratch.
    ///
    /// Returns the row index inside the scratch that the last token's normalized
    /// output landed in, which is the row a caller that wants logits projects.
    ///
    /// `with_logits` says whether the final chunk's commit carries the prompt-final
    /// logits row. The projection has to happen *before* that commit rather than
    /// after it, because the commit is the only moment the cache will take the row:
    /// a second commit at the same position is a no-op, so a row projected afterwards
    /// would never reach the checkpoint and every resume from it would have to re-run
    /// a decode step to get one.
    fn advance(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        with_logits: bool,
        media: Option<&VisionPrefill<'_>>,
    ) -> CoreResult<u32> {
        if tokens.is_empty() {
            return Err(CoreError::ShapeMismatch {
                field: "prompt chunk",
                expected: 1,
                got: 0,
            });
        }
        // An empty item list is the text path, not a degenerate multimodal one:
        // a caller that builds the same request shape every time gets the old
        // launches back by handing over no items.
        let media = media.filter(|prefill| !prefill.is_empty());
        let live = self
            .sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        if media.is_some() {
            live.mark_media();
        }
        let start = live.written();
        let floor = self.hint_floor();
        self.hint_branch_point(session, tokens, floor)?;
        let live = self
            .sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        // A session that already read a picture - this prompt's earlier
        // chunks, or the prefix a resume stood it on - continues the rotary
        // stream where that picture left it, not at its sequence position.
        let rotary = start.saturating_sub(live.rope_shift());
        let positions = self.shape.media_positions;
        let mut run = media.map(|prefill| MediaRun {
            prefill: *prefill,
            walk: Walk::new(prefill.grids(), prefill.pads(), start, positions).at_rotary(rotary),
        });
        let width = usize::try_from(MAX_PREFILL_CHUNK).unwrap_or(usize::MAX);
        let mut last = 0;
        let mut rest = tokens;
        while !rest.is_empty() {
            // A chunk may not end inside a placeholder run: a token of a media
            // span may be allowed to read a key from the end of that span, and a
            // key the chunk never reaches is a cache row nothing has written.
            //
            // It also stops where a picture ends, so the recurrence sums the
            // runs after a picture from its end in every read of this prompt -
            // cold, resumed or batched - whichever boundaries this one had.
            let take = match run.as_ref() {
                Some(run) => crate::media_path::picture_end_within(
                    rest,
                    run.walk.split_point(rest, width),
                    run.prefill.pads(),
                ),
                None => rest.len().min(width),
            };
            let live = self
                .sessions
                .get(&session)
                .ok_or(CoreError::UnknownSession(session))?;
            let start = live.written();
            // A chunk also stops at a shared boundary inside it, so the state
            // there is committed - and checkpointed - before the rest is read:
            // a sibling sharing exactly that much stands on it. See
            // `chunk_stop` for which boundaries count and where it cuts.
            let end = start.saturating_add(u32::try_from(take).unwrap_or(u32::MAX));
            let take = self
                .chunk_stop(session, start, end, floor, run.is_some())?
                .and_then(|at| usize::try_from(at - start).ok())
                .unwrap_or(take);
            let (head, tail) = rest.split_at(take);
            let n_tok = self.run_chunk(session, head, run.as_mut())?;
            // Before anything else touches the scratch: the draft head's warm
            // pass reuses the normalized rows this pass just left there.
            self.capture_rows(session, n_tok)?;
            // The draft head's own cache, over the rows this chunk just left in
            // the residual. Without it the head attends to key and value rows
            // nothing wrote - it still answers, because attention reads
            // whatever was last at those addresses, so the only thing that
            // shows is a draft the target rarely agrees with: a build that is
            // slower for speculating and correct either way.
            self.warm_draft(session, head, start)?;
            last = n_tok.saturating_sub(1);
            let final_chunk = tail.is_empty();
            let row = (final_chunk && with_logits).then_some(0);
            if row.is_some() {
                self.write_logits(last, 0)?;
            }
            // Before the commit: a checkpoint taken here is resumed at the
            // shift its key's spans give, which is the one the walk reached.
            if let Some(run) = run.as_ref() {
                self.sessions
                    .get_mut(&session)
                    .ok_or(CoreError::UnknownSession(session))?
                    .set_rope_shift(run.walk.shift());
            }
            self.commit(session, final_chunk, row)?;
            rest = tail;
        }
        // The count the items carry is the count the rendered prompt used, and
        // nothing below re-expands a placeholder run. So the walk having handed
        // out every encoder row is the statement that the prompt and the tower
        // described the same request - the one mismatch that otherwise succeeds,
        // stays fluent and answers about a rearranged picture.
        //
        // Asked only once the prompt is complete: a caller splitting its own
        // prompt has more placeholders coming.
        if with_logits && let Some(run) = run.as_ref() {
            let consumed = u64::from(run.walk.rows_consumed());
            let available = u64::from(run.prefill.row_count());
            if consumed != available {
                return Err(CoreError::ShapeMismatch {
                    field: "vision placeholder tokens",
                    expected: available,
                    got: consumed,
                });
            }
        }
        Ok(last)
    }

    /// The closest a media prompt's message boundary or a score's branch point
    /// may lie to where a read or its resume began and still be stopped at: the
    /// tier's floor.
    ///
    /// A checkpoint is worth the prefill between it and the one below it, and
    /// a few hundred tokens of that is not worth a snapshot. A score's own
    /// boundaries are the host's call - its siblings wait on exactly those -
    /// and are taken wherever they fall.
    fn hint_floor(&self) -> u32 {
        self.ckpt.map_or(1, |crate::ckpt::Checkpoint { options }| {
            options.min_tokens.max(1)
        })
    }

    /// Adds where a scoring read of `tokens` parts from the paths the cache
    /// holds to its boundaries, when that lies at least `floor` past where the
    /// read or its resume began and clears every picture.
    ///
    /// A score's end is answered and dropped, so without this a later request
    /// sharing everything up to that point - an agent's next step, whose
    /// history grew by one line - would find nothing to stand on short of the
    /// shared boundaries its own siblings asked for.
    fn hint_branch_point(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        floor: u32,
    ) -> CoreResult<()> {
        let Self {
            ckpt,
            resident,
            sessions,
            ..
        } = self;
        let live = sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let (Some(_), Some(resident)) = (ckpt.as_ref(), resident.as_ref()) else {
            return Ok(());
        };
        if !live.scoring() {
            return Ok(());
        }
        let start = live.written();
        let len = start.saturating_add(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
        let branch = resident.cache.branch_point(&live.cache_path_with(tokens));
        let lowest = start.max(live.resumed_at()).saturating_add(floor);
        if branch > start
            && branch < len
            && branch >= lowest
            && !live.key().is_some_and(|key| key.splits(branch))
        {
            live.add_hint(branch);
        }
        Ok(())
    }

    /// Where a chunk of `session` from `start` stops short of `end`: the first
    /// shared boundary inside it that the read may stop at, moved back onto
    /// the recurrence's grid where that still clears every picture. `None` is
    /// not a refusal: no boundary lies inside, and the chunk runs to `end`.
    ///
    /// In a media prompt (`media`) a boundary counts only where the content
    /// key lets the state there be filed and the cut clears every picture: a
    /// chunk may not end inside a run. The chunked recurrence walks its runs
    /// from a chunk's start, so a stop between two grid points shifts every run
    /// after it, and the prompt - and every prompt resumed from the checkpoint
    /// taken there - sums differently from a read that never stopped. The
    /// resume re-reads the few positions between the grid point and the
    /// boundary instead. Past a picture the grid counts from its end, which
    /// every read stops at ([`shiso_core::media::run_floor`]), so the boundary
    /// just past a shared picture moves back onto that end.
    ///
    /// A text prompt that is not a score stops at its first message boundary
    /// inside the chunk, exactly where it falls: the rule plain conversations
    /// have always had, which their next turn resumes on. A media prompt's
    /// boundaries count from `floor` past where its read or its resume began -
    /// not past each chunk's start, which no boundary inside a chunk as wide
    /// as the floor could ever clear.
    fn chunk_stop(
        &mut self,
        session: SessionId,
        start: u32,
        end: u32,
        floor: u32,
        media: bool,
    ) -> CoreResult<Option<u32>> {
        let live = self
            .sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        if !live.scoring() && !media {
            return Ok(live.hints_between(start, end).next());
        }
        let lowest = if live.scoring() {
            start
        } else {
            start.max(live.resumed_at().saturating_add(floor - 1))
        };
        let clears =
            |live: &Session, at: u32| !media || live.key().is_some_and(|key| !key.splits(at));
        let spans = live.key().map_or(&[][..], |key| key.spans.as_slice());
        // Each boundary moves onto its grid point, which may be the chunk's
        // own end - a picture's end, which every read cuts at, with the
        // questions parting just past it - so the commit there is the shared
        // one. A grid point at or behind where this chunk starts is one an
        // earlier cut already took, and the next boundary is asked instead of
        // stopping off the grid.
        let Some((hint, at)) = live.hints_between(lowest, u32::MAX).find_map(|hint| {
            let at = shiso_core::media::run_floor(hint, spans, crate::launch::GDN_CHUNK_RUN);
            (at > lowest && at <= end && clears(live, at)).then_some((hint, at))
        }) else {
            return Ok(None);
        };
        if at != hint {
            live.move_hint(hint, at);
        }
        Ok(Some(at))
    }

    /// Registers the span the session just wrote, and admits a checkpoint when the
    /// cache says this position is worth one.
    ///
    /// The snapshot is asked for before it is taken: a checkpoint blob is a
    /// tens-of-mebibytes read back off the device, so paying it speculatively on
    /// every committed span would cost more than the prefill it saves.
    ///
    /// `logits_row` names which row of the logits block is this session's, or `None`
    /// when the step produced no row. A committed checkpoint without one cannot be
    /// resumed from without re-running a step to get it, which is the whole reason
    /// the row travels with the state rather than beside it.
    fn commit(
        &mut self,
        session: SessionId,
        earned: bool,
        logits_row: Option<u32>,
    ) -> CoreResult<()> {
        let now = self.now();
        let Self {
            backend,
            resident,
            sessions,
            staging,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        let live = sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let q = live.written();
        // Whether this position is worth a checkpoint, and the cheap half of
        // that question is asked first.
        //
        // The cache answers `next_admission` with the end of what it has been
        // handed when nothing earlier stops it - and at a decode step that is
        // simply the next position, so every single token looks like an
        // admission point. Taking it at face value costs the whole recurrent
        // stack read back to the host per token: tens of mebibytes, across one
        // device-to-host copy per recurrent slot, each of which flushes the
        // command buffer and waits. Measured, that alone is most of a decode
        // step.
        //
        // So a snapshot is only considered at a boundary that is worth one on
        // its own terms: the end of a prompt, or a stride multiple. A prompt's
        // end is admitted outright - the cache's earliest stop is the branch
        // point, which every rendered prompt passes on its template's opening,
        // so gating the end on it stored nothing but first prompts and exact
        // repeats - and a stride position still has to be the cache's own stop.
        // A session that read media is checkpointed only under its content key
        // and never inside a picture: its placeholder runs cannot tell two
        // pictures apart, and a resume there would start halfway through one.
        //
        // A shared boundary is admitted as a prompt's end is, whether the
        // chunk ended there or the prompt did. A scoring session's prompt
        // end is answered and dropped, so it takes a checkpoint at its
        // shared boundaries and nowhere else: a snapshot at its end or at
        // its strides is one nothing resumes, and in a full store it takes
        // the slot a sibling's boundary needed.
        let hint = live.is_hint(q);
        let (earned, periodic) = if live.scoring() {
            (hint, false)
        } else {
            (earned || hint, q > 0 && q.is_multiple_of(MAX_PREFILL_CHUNK))
        };
        let admit = (earned || periodic)
            && live.keyed_at(q)
            && (earned
                || resident.cache.next_admission(
                    live.kv().frontier(),
                    live.cache_path(),
                    &[],
                    MAX_PREFILL_CHUNK,
                ) == Some(q));
        if admit {
            snapshot(backend.as_ref(), live, &mut staging.ckpt)?;
            if let Some(row) = logits_row {
                let width = logits_row_bytes(resident);
                let at = u64::from(row).saturating_mul(width);
                let row = resident.scratch.logits().slice(at, width)?;
                staging
                    .logits
                    .resize(usize::try_from(row.len()).unwrap_or(0), 0);
                backend.copy_d2h(row, &mut staging.logits)?;
            }
        }
        // A prompt's end goes to the disk tier as soon as it is admitted: cold
        // when this process prefilled the whole prompt, a continuation when it
        // extended a resumed one.
        let reason = if live.resumed_at() == 0 {
            Reason::Cold
        } else {
            Reason::Continued
        };
        let scoring = live.scoring();
        let (history, kv) = live.commit_parts();
        let mut opts = CommitOpts::span(&[]);
        if admit {
            opts = opts.checkpoint(staging.ckpt.as_slice(), earned);
            // A score's boundaries go before a conversation's when slots run
            // out: a decision burst must not drain the pool chats resume from.
            if scoring {
                opts = opts.scoring();
            }
            if logits_row.is_some() {
                opts = opts.logits_row(staging.logits.as_slice());
            }
            if earned {
                opts = opts.persist(reason);
            }
        }
        resident
            .cache
            .commit(kv.frontier_mut(), history, q, opts, now)
            .map_err(cache_failure)
    }
}

/// Starts a chunk's per-layer reads, or bounds its ids against a resident table.
///
/// `None` is not a refusal and not an empty read: it is a chunk with nothing to
/// await, which is every chunk of a stack with no per-layer table and every
/// chunk of one whose table is resident. Only a streamed table has rows in
/// flight, because only a streamed table has rows the host has to fetch.
///
/// Called before the session is advanced, so that the one refusal it can make
/// leaves the session exactly as it found it.
fn open_ple(
    resident: &Resident,
    geometry: &Geometry,
    tokens: &[TokenId],
) -> CoreResult<Option<PleRows>> {
    match resident.ple.as_ref() {
        None => Ok(None),
        Some(PleTable::Streamed { stream, .. }) => {
            issue_rows(stream, tokens.iter().map(|t| u64::from(t.get())).collect()).map(Some)
        }
        // A resident table is indexed by the token id inside the kernel, which
        // cannot refuse one past the end: it would read whatever follows the
        // table and decode it to finite weights. The bound is checked here
        // instead, over ids the caller already walks.
        Some(PleTable::Resident { .. }) => {
            let vocab = geometry.ple.as_ref().map_or(0, |ple| ple.vocab);
            for token in tokens {
                if token.get() >= vocab {
                    return Err(CoreError::OutOfBounds {
                        offset: u64::from(token.get()),
                        len: 1,
                        buffer_bytes: u64::from(vocab),
                    });
                }
            }
            Ok(None)
        }
    }
}

/// Checks a per-layer table against the geometry the stack would run it at.
///
/// Every disagreement here is silent at runtime, which is the whole reason the
/// function exists. The prepare kernel turns a row identifier into an offset by
/// striding `layers * 210` bytes and decodes what it finds as `q6_K`: a table
/// whose rows are one block narrower is read from the middle of its own
/// predecessor from the second row on, and a table in another format is decoded
/// as if it were this one. Both produce finite weights at every element, so
/// nothing downstream can notice.
///
/// A resident table is stored the way a projection weight is: `in_dim` is the
/// width of one row, which for this table is one token's whole row
/// (`layers * dim`, Gemma 4 E4B's 10752), and `out_dim` is the number of rows,
/// which is the table's own vocabulary (262144). That orientation is what makes
/// [`Weight::row_bytes`] the row stride the kernel strides by.
///
/// [`Weight::row_bytes`]: crate::projection::Weight::row_bytes
fn validate_ple(geometry: &Geometry, ple: Option<&PleTable>) -> CoreResult<()> {
    let (Some(widths), Some(table)) = (geometry.ple.as_ref(), ple) else {
        return Ok(());
    };
    let mismatch = |field: &'static str, expected: u64, got: u64| CoreError::ShapeMismatch {
        field,
        expected,
        got,
    };
    let wrong_format = CoreError::Unsupported {
        what: "a per-layer embedding row in a format no prepare kernel decodes",
        backend: "engine",
    };
    let rows = match table {
        PleTable::Streamed { stream, row_dtype } => {
            if *row_dtype != DType::Q6_K && *row_dtype != DType::MlxAffine4Gs64 {
                return Err(wrong_format);
            }
            // The reader's stride and the kernel's are two numbers derived from
            // two places, and the staging is filled by one and read by the
            // other.
            let want = widths.row_bytes_for(*row_dtype);
            let got = u64::from(stream.row_bytes());
            if got != want {
                return Err(mismatch("per-layer row bytes", want, got));
            }
            stream.n_rows()
        }
        PleTable::Resident { table } => {
            if table.dtype != DType::Q6_K && table.dtype != DType::MlxAffine4Gs64 {
                return Err(wrong_format);
            }
            let want = widths.block_elems();
            let got = u64::from(table.in_dim);
            if got != want {
                return Err(mismatch("per-layer row width", want, got));
            }
            u64::from(table.out_dim)
        }
    };
    // The row count is what a token id is bounded against, so a table that
    // holds fewer rows than the shape says would have that bound admit ids the
    // table cannot answer for.
    if rows != u64::from(widths.vocab) {
        return Err(mismatch(
            "per-layer table rows",
            u64::from(widths.vocab),
            rows,
        ));
    }
    Ok(())
}

/// Waits for a chunk's per-layer rows and copies them into the staging.
///
/// The destination is the front of the row and its length is the gather's own,
/// not the whole buffer: the staging is sized for the widest chunk, and the
/// prepare kernel reads exactly `n_tok` rows out of its front.
///
/// The awaited bytes are MOVED into `host` rather than copied through it: the
/// gather thread cannot borrow a buffer this side of the join, so a fill would
/// be a second copy of a megabytes-wide chunk for nothing.
fn stage_ple(
    backend: &dyn Backend,
    resident: &Resident,
    host: &mut Vec<u8>,
    rows: PleRows,
) -> CoreResult<()> {
    let dst = resident
        .scratch
        .ple_stream()
        .ok_or(CoreError::ShapeMismatch {
            field: "per-layer staging",
            expected: 1,
            got: 0,
        })?;
    *host = rows.await_rows()?;
    let len = u64::try_from(host.len()).unwrap_or(u64::MAX);
    if len > dst.len() {
        return Err(CoreError::OutOfBounds {
            offset: 0,
            len,
            buffer_bytes: dst.len(),
        });
    }
    backend.copy_h2d(dst.slice(0, len)?, host)
}

/// Bytes of one logits row.
fn logits_row_bytes(resident: &Resident) -> u64 {
    u64::from(resident.weights.output_matrix().out_dim).saturating_mul(F32_BYTES)
}

/// Copies the session's whole recurrent stack into `blob`.
///
/// The layout is the ordinal order the layers name, each slot's convolution window
/// followed by its state matrix. It is the layout a resume reads back, and the two
/// agree because this is the only function that writes it.
pub(crate) fn snapshot(
    backend: &dyn Backend,
    session: &mut Session,
    blob: &mut Vec<u8>,
) -> CoreResult<()> {
    blob.clear();
    for ordinal in 0..session.recurrent_slots() {
        let ordinal = u32::try_from(ordinal).unwrap_or(u32::MAX);
        let state = session.gdn(ordinal)?;
        append(backend, blob, state.conv())?;
        append(backend, blob, state.state())?;
    }
    Ok(())
}

/// Reads one device range onto the end of `blob`.
fn append(backend: &dyn Backend, blob: &mut Vec<u8>, view: DeviceView<'_>) -> CoreResult<()> {
    let start = blob.len();
    let len = usize::try_from(view.len()).unwrap_or(0);
    blob.resize(start.saturating_add(len), 0);
    backend.copy_d2h(view, &mut blob[start..])
}

/// Uploads one `int32` row, reusing `host` as the staging buffer.
pub(crate) fn upload_row(
    backend: &dyn Backend,
    host: &mut Vec<u8>,
    dst: DeviceView<'_>,
    values: impl Iterator<Item = i32>,
) -> CoreResult<()> {
    host.clear();
    for value in values {
        host.extend_from_slice(&value.to_le_bytes());
    }
    let len = u64::try_from(host.len()).unwrap_or(u64::MAX);
    if len > dst.len() {
        return Err(CoreError::OutOfBounds {
            offset: 0,
            len,
            buffer_bytes: dst.len(),
        });
    }
    backend.copy_h2d(dst.slice(0, len)?, host)
}

/// Presents a cache failure in the core vocabulary.
pub(crate) fn cache_failure(source: shiso_kvstore::KvError) -> CoreError {
    CoreError::Backend {
        backend: "engine",
        op: "kv cache",
        source: Box::new(source),
    }
}

impl HybridStack {
    /// The geometry this stack runs.
    #[must_use]
    pub const fn shape(&self) -> &HybridShape {
        &self.shape
    }

    /// Tokens one session may hold, prompt included.
    #[must_use]
    pub const fn context_capacity(&self) -> u32 {
        self.opts.ctx
    }

    /// Pages the KV pool was allocated with.
    ///
    /// The count a draft head bound to this stack sizes its own pool by. It is
    /// the resolved one, which differs from the options the stack was built
    /// with when those asked for a share of the device.
    #[must_use]
    pub const fn pool_pages(&self) -> u32 {
        self.opts.kv_pool_pages
    }

    /// What the cache currently holds: pages, checkpoint slots, and the disk tier
    /// when one is open.
    ///
    /// `None` only when the stack has been torn down. Everything else is a real
    /// measurement, which is the point: an operator looking at a cache report has
    /// no way to tell a zero that was measured from a zero that means "nobody
    /// asked", so the two are different answers here.
    #[must_use]
    pub fn cache_stats(&self) -> Option<CacheStats> {
        Some(self.resident.as_ref()?.cache.stats())
    }

    /// What the per-layer reader has done, for a stack that streams its table.
    ///
    /// `None` when there is nothing that could have read a row: a stack that has
    /// been torn down, a family with no per-layer table, and a table held
    /// resident, which is read by the kernel off the device and never by a
    /// reader at all. A zeroed statistic in any of those cases would claim a
    /// measurement that nothing took, which is the same distinction
    /// [`Self::cache_stats`] makes.
    #[must_use]
    pub fn ple_stats(&self) -> Option<shiso_ple::PleStats> {
        match self.resident.as_ref()?.ple.as_ref()? {
            PleTable::Streamed { stream, .. } => Some(stream.stats()),
            PleTable::Resident { .. } => None,
        }
    }

    /// Recurrent slots, which is how many states a session allocates.
    #[must_use]
    pub const fn recurrent_slots(&self) -> usize {
        self.recurrent_slots
    }

    /// Allocates a fresh, empty session.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the device state a session needs cannot be
    /// created, and whatever the cache reports for a frontier it cannot hand out.
    pub fn open_session(&mut self) -> CoreResult<SessionId> {
        let capacity = self.opts.ctx;
        let recurrent = self.recurrent_slots;
        let shape = self.shape.clone();
        let frontier = self.resident()?.cache.cold_frontier();
        let id = SessionId::next(&mut self.counter).ok_or(CoreError::Backend {
            backend: "engine",
            op: "open_session",
            source: Box::new(std::io::Error::other("session id space exhausted")),
        })?;
        let session = Session::open(
            &self.backend,
            id,
            frontier,
            capacity,
            &shape,
            recurrent,
            self.opts.spec,
        )?;
        self.sessions.insert(id, session);
        self.order.push(id);
        Ok(id)
    }

    /// Releases a session's state. A second call for the same handle is
    /// [`CoreError::UnknownSession`], not a silent success.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`] when the handle is not one this stack minted
    /// or has already been closed.
    pub fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        let live = self
            .sessions
            .remove(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        self.order.retain(|id| *id != session);
        self.captures.remove(&session);
        // Every captured decode step names this session's recurrent state and
        // its page references by address, and both are about to be released.
        // Dropped BEFORE the session is, so that no window exists in which a
        // replay could name freed storage.
        self.invalidate_decode_graphs();
        // Dropping the session releases its recurrent stack and drops its frontier,
        // which is what returns the page references to the pool.
        drop(live);
        Ok(())
    }

    /// Advances a session by a non-final prompt chunk, producing nothing.
    ///
    /// A chunk wider than the scratch is split here rather than refused: the split
    /// is invisible in the result, because the recurrent state and the cache carry
    /// the prefix across an internal boundary exactly as across a caller's.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::ContextOverflow`] when the
    /// chunk reaches past the context, and [`CoreError::Backend`] on a device
    /// failure.
    pub fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        self.advance(session, tokens, false, None).map(|_last| ())
    }

    /// Advances a session by a non-final prompt chunk that carries media.
    ///
    /// As [`HybridStack::prefill_chunk`], and the placeholder rows are replaced
    /// by `media`'s. The row-count agreement is only checked once the prompt is
    /// complete, because a caller splitting its own prompt still has
    /// placeholders to come; a request that never reaches
    /// [`HybridStack::prefill_last_media`] is never checked.
    ///
    /// # Errors
    ///
    /// As [`HybridStack::prefill_chunk`].
    pub fn prefill_chunk_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: &VisionPrefill<'_>,
    ) -> CoreResult<()> {
        self.advance(session, tokens, false, Some(media))
            .map(|_last| ())
    }

    /// Advances a session by its final prompt chunk and returns the last-token row.
    ///
    /// # Errors
    ///
    /// As [`HybridStack::prefill_chunk`], plus whatever the cache reports for the
    /// commit that follows.
    pub fn prefill_last(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
    ) -> CoreResult<Logits<'_>> {
        self.advance(session, tokens, true, None)?;
        self.prompt_logits()
    }

    /// Advances a session by its final prompt chunk, splicing `media` over the
    /// placeholder positions, and returns the last-token row.
    ///
    /// The tokens are the rendered prompt with every placeholder already
    /// expanded to its item's count: this splices, it does not expand. A prompt
    /// whose placeholder run does not match what the items claim is refused
    /// here rather than answered, which is the one mismatch that would otherwise
    /// produce a fluent answer about a rearranged picture.
    ///
    /// # Errors
    ///
    /// As [`HybridStack::prefill_last`], plus [`CoreError::ShapeMismatch`] when
    /// the prompt's placeholder run and the encoder's rows disagree.
    pub fn prefill_last_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: &VisionPrefill<'_>,
    ) -> CoreResult<Logits<'_>> {
        self.advance(session, tokens, true, Some(media))?;
        self.prompt_logits()
    }

    /// The single logits row a completed prompt left in the block.
    fn prompt_logits(&self) -> CoreResult<Logits<'_>> {
        let resident = self.resident()?;
        let row = logits_row_bytes(resident);
        Logits::new(
            resident.scratch.logits().slice(0, row)?,
            1,
            self.geometry.vocab,
        )
    }

    /// Advances every member by one token. Row `i` of the result is `steps[i]`,
    /// and an empty slice is `Ok` with zero rows rather than an error.
    ///
    /// **One row per session.** A batch naming the same handle twice is
    /// refused rather than answered: the two rows would share one recurrent
    /// state and one row of the packed page table, so the second would fold
    /// into a state the first had already moved and read a table written for a
    /// position it is not at. A caller that wants two tokens of one session
    /// wants a chunk, which is what the prefill path is.
    ///
    /// # What "checked before any of them runs" does and does not cover
    ///
    /// The membership and uniqueness checks below are complete before the
    /// first session is advanced, so a batch that fails **either** of them
    /// advances nobody. That is not true of every refusal this can make:
    /// [`CoreError::ContextOverflow`] is raised by the advance itself, member
    /// by member, so a batch whose fifth member has no room leaves the first
    /// four a token ahead with no logits row to show for it. That was equally
    /// true of the per-member loop this replaced and it is not a regression -
    /// it is stated because the sentence above reads broader than it is.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] for a batch wider than the logits block or
    /// one naming a session twice, [`CoreError::UnknownSession`] when any step
    /// names a handle this stack does not hold - both checked for every member
    /// before any of them runs - [`CoreError::ContextOverflow`] when a member's
    /// step reaches past the context, which is raised mid-advance as above, and
    /// [`CoreError::Backend`] on a device failure.
    pub fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        let rows = u32::try_from(steps.len()).unwrap_or(u32::MAX);
        if rows > MAX_DECODE_BATCH {
            return Err(CoreError::Unsupported {
                what: "a decode batch wider than the logits block",
                backend: "engine",
            });
        }
        // Every member is checked before any of them runs: a batch that failed
        // half way would have advanced some sessions by a token the caller never
        // gets a row for.
        for (at, step) in steps.iter().enumerate() {
            if !self.sessions.contains_key(&step.session) {
                return Err(CoreError::UnknownSession(step.session));
            }
            // A quadratic scan over at most `MAX_DECODE_BATCH` handles, which
            // is eight: a set would allocate once a step to answer a question
            // about eight values.
            if steps[..at].iter().any(|seen| seen.session == step.session) {
                return Err(CoreError::Unsupported {
                    what: "a decode batch naming one session twice, whose two rows would \
                           share a recurrent state and a page table",
                    backend: "engine",
                });
            }
        }
        // An empty batch runs nothing, which used to be true by the loop
        // never turning. It is a guard now, because one pass over zero rows is
        // a real pass: every launch would go out at `n_tok = 0` and the norm
        // would divide by a width it was handed as zero.
        if !steps.is_empty() {
            self.run_batch(steps)?;
            self.absorb_batch(steps)?;
        }
        for (row, step) in steps.iter().enumerate() {
            let row = u32::try_from(row).unwrap_or(u32::MAX);
            // Row `i` of the pass is step `i`, in the caller's own order, so
            // the logits row a session commits is its own.
            self.write_logits(row, row)?;
            // The row this step just produced travels with the checkpoint, so a
            // session saved mid-generation resumes standing on its own frontier row
            // rather than having to re-run the step that made it.
            self.commit(step.session, false, Some(row))?;
        }
        let resident = self.resident()?;
        let bytes = logits_row_bytes(resident).saturating_mul(u64::from(rows));
        Logits::new(
            resident.scratch.logits().slice(0, bytes)?,
            rows,
            self.geometry.vocab,
        )
    }

    /// Reads several sessions' final prompt runs in one pass and returns one
    /// logits row per run, in run order: row `r` is the row
    /// [`HybridStack::prefill_last`] would return for `runs[r]`, and every
    /// session is left where that read would leave it.
    ///
    /// The runs are laid end to end as one pass ([`layers::PassRows::Segments`]):
    /// the projections, norms and feed-forward read every row at once, and
    /// attention and the recurrent blocks read each run through its own
    /// session's chunk path. A projection's output row is a function of its
    /// input row alone, so a run's rows are the ones its solo read computes.
    ///
    /// # Errors
    ///
    /// Every refusal comes before any session is advanced.
    /// [`CoreError::Unsupported`] with [`shiso_core::NO_BATCHED_PROMPT_PASS`]
    /// for a stack that drafts (a draft head warms per chunk, which this pass
    /// does not), and without it for more runs than [`MAX_PROMPT_RUNS`], a
    /// pass wider than [`MAX_PREFILL_CHUNK`] rows once a shared trunk is
    /// counted once, a session that read media or keeps a
    /// decision capture, and a session named twice; [`CoreError::ShapeMismatch`]
    /// for no runs or an empty one; [`CoreError::UnknownSession`];
    /// [`CoreError::ContextOverflow`]; and whatever the device reports.
    pub fn prefill_last_many(&mut self, runs: &[PromptRun<'_>]) -> CoreResult<Logits<'_>> {
        self.check_prompt_runs(runs)?;
        let capacity = self.opts.ctx;
        let now = self.now();
        let last_rows = {
            let Self {
                backend,
                geometry,
                kernels,
                resident,
                sessions,
                staging,
                ..
            } = self;
            let resident = resident.as_ref().ok_or(CoreError::Unsupported {
                what: "a model whose device state teardown already released",
                backend: "engine",
            })?;
            let starts = runs
                .iter()
                .map(|run| {
                    sessions
                        .get(&run.session)
                        .map(Session::written)
                        .ok_or(CoreError::UnknownSession(run.session))
                })
                .collect::<CoreResult<Vec<_>>>()?;
            let layout = RunLayout::new(runs, &starts);
            // Measured on the layout rather than the runs: a trunk read once
            // is what lets several prompts of a few hundred tokens share one
            // pass at all.
            if layout.rows() > MAX_PREFILL_CHUNK {
                return Err(CoreError::Unsupported {
                    what: "a batched prompt pass wider than one prefill chunk",
                    backend: "engine",
                });
            }
            let rows = open_ple(resident, geometry, &layout.tokens)?;
            let shifts = run_shifts(sessions, geometry, runs)?;
            for run in runs {
                let room = kv::Room {
                    cache: &resident.cache,
                    now,
                };
                sessions
                    .get_mut(&run.session)
                    .ok_or(CoreError::UnknownSession(run.session))?
                    .advance(run.tokens, capacity, room)?;
            }
            let rotary = upload_run_rows(
                backend.as_ref(),
                &mut staging.ids,
                &resident.scratch,
                &layout,
                &shifts,
            )?;
            if let Some(rows) = rows {
                stage_ple(backend.as_ref(), resident, &mut staging.ple, rows)?;
            }
            // The sessions the trunk is forked into: every run's but the first.
            let forks: Vec<&Session> = if layout.shared > 0 {
                runs.iter()
                    .skip(1)
                    .map(|run| {
                        sessions
                            .get(&run.session)
                            .ok_or(CoreError::UnknownSession(run.session))
                    })
                    .collect::<CoreResult<_>>()?
            } else {
                Vec::new()
            };
            let segments = segment_runs(runs, &layout, sessions, &forks)?;
            let residual = layers::forward(
                backend.as_ref(),
                &resident.weights,
                geometry,
                &resident.scratch,
                kernels.as_ref(),
                &resident.kv,
                layers::PassRows::Segments(&segments),
                resident.ple.as_ref(),
                layout.rows(),
                None,
                layers::PassExtras {
                    trail: Trail::None,
                    taps: None,
                    rotary,
                },
            )?;
            drop(segments);
            drop(forks);
            for (run, last) in runs.iter().zip(&layout.last_rows) {
                let live = sessions
                    .get_mut(&run.session)
                    .ok_or(CoreError::UnknownSession(run.session))?;
                live.push_history(run.tokens);
                live.record_pass(residual, *last);
            }
            layout.last_rows
        };
        self.answer_prompt_runs(runs, &last_rows)
    }

    /// Each run's last row of the pass that just ran into its own logits row,
    /// and the commit a solo final chunk makes, carrying that row; then the
    /// block of those rows.
    fn answer_prompt_runs(
        &mut self,
        runs: &[PromptRun<'_>],
        last_rows: &[u32],
    ) -> CoreResult<Logits<'_>> {
        for ((row, run), last) in (0u32..).zip(runs).zip(last_rows) {
            self.write_logits(*last, row)?;
            self.commit(run.session, true, Some(row))?;
        }
        let rows = u32::try_from(runs.len()).unwrap_or(u32::MAX);
        let resident = self.resident()?;
        let bytes = logits_row_bytes(resident).saturating_mul(u64::from(rows));
        Logits::new(
            resident.scratch.logits().slice(0, bytes)?,
            rows,
            self.geometry.vocab,
        )
    }

    /// The refusals of [`HybridStack::prefill_last_many`], all of them before
    /// anything is advanced or launched.
    fn check_prompt_runs(&self, runs: &[PromptRun<'_>]) -> CoreResult<()> {
        // A fact about this stack rather than the call: a draft head warms its
        // own cache chunk by chunk and a block drafter absorbs every pass's
        // rows, and this pass does neither.
        if self.draft.is_some() || self.block.is_some() {
            return Err(CoreError::Unsupported {
                what: shiso_core::NO_BATCHED_PROMPT_PASS,
                backend: "engine",
            });
        }
        if runs.is_empty() || runs.iter().any(|run| run.tokens.is_empty()) {
            return Err(CoreError::ShapeMismatch {
                field: "batched prompt runs",
                expected: 1,
                got: 0,
            });
        }
        if runs.len() > MAX_PROMPT_RUNS as usize {
            return Err(CoreError::Unsupported {
                what: "a batched prompt pass of more runs than the logits block holds",
                backend: "engine",
            });
        }
        for (at, run) in runs.iter().enumerate() {
            let live = self
                .sessions
                .get(&run.session)
                .ok_or(CoreError::UnknownSession(run.session))?;
            if runs[..at].iter().any(|seen| seen.session == run.session) {
                return Err(CoreError::Unsupported {
                    what: "a batched prompt pass naming one session twice",
                    backend: "engine",
                });
            }
            // A session that read media joins once its pictures are behind it:
            // what is left is text, rotated as its solo read would rotate it.
            let picture_ahead = live.carries_media()
                && live
                    .key()
                    .is_none_or(|key| key.spans.iter().any(|span| span.end() > live.written()));
            if picture_ahead || self.captures.contains_key(&run.session) {
                return Err(CoreError::Unsupported {
                    what: "a batched prompt run over a session with a picture left to read \
                           or that keeps a decision capture",
                    backend: "engine",
                });
            }
            let next = u64::from(live.written())
                .saturating_add(u64::try_from(run.tokens.len()).unwrap_or(u64::MAX));
            if next > u64::from(self.opts.ctx) {
                return Err(CoreError::ContextOverflow {
                    session: run.session,
                    cap: self.opts.ctx,
                });
            }
        }
        Ok(())
    }

    /// Releases device resources, reporting failures.
    ///
    /// Sessions go first and in reverse open order, because a session holds page
    /// references the cache below it is about to be asked to release; then the
    /// resident state in reverse construction order.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the device could not be brought to rest.
    pub fn teardown(&mut self) -> CoreResult<()> {
        let mut first = Ok(());
        // Before any of it is released: a captured decode step names the
        // weights, the scratch, the page block and every member's recurrent
        // state by address, and teardown frees all of them.
        self.invalidate_decode_graphs();
        // Sessions go in reverse open order, because a session holds page
        // references the cache below it is about to be asked to release.
        while let Some(id) = self.order.pop() {
            if let Some(live) = self.sessions.remove(&id) {
                drop(live);
            }
        }
        self.sessions.clear();
        if let Some(resident) = self.resident.take() {
            let Resident {
                weights,
                kv,
                cache,
                scratch,
                rows,
                ple,
            } = resident;
            // Reverse construction order: the per-layer table (whose reader owns
            // a descriptor and a prefetch thread), then the packed page-table
            // block, then scratch, then the cache that holds the pool handle,
            // then the pool, then the weights.
            drop(ple);
            drop(rows);
            drop(scratch);
            drop(cache);
            drop(kv);
            drop(weights);
        }
        if let Err(error) = self.backend.sync() {
            first = Err(error);
        }
        tracing::info!("model torn down");
        first
    }
}

impl Drop for HybridStack {
    /// Releases whatever teardown did not, and says so.
    ///
    /// Every device buffer frees itself regardless, so this is about ordering and
    /// visibility rather than about whether the memory comes back: a model that was
    /// dropped without an explicit teardown had no chance to report a device that
    /// refused to let go.
    fn drop(&mut self) {
        if self.resident.is_none() {
            return;
        }
        if let Err(error) = self.teardown() {
            tracing::warn!(%error, "teardown from drop could not report cleanly");
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::{PromptRun, RunLayout, RunPiece};
    use shiso_core::ids::{SessionId, TokenId};

    fn ids(values: &[u32]) -> Vec<TokenId> {
        values.iter().copied().map(TokenId::new).collect()
    }

    fn runs(prompts: &[Vec<TokenId>]) -> Vec<PromptRun<'_>> {
        let mut counter = std::num::NonZeroU64::MIN;
        prompts
            .iter()
            .map(|tokens| PromptRun {
                session: SessionId::next(&mut counter).expect("a handle"),
                tokens,
            })
            .collect()
    }

    #[test]
    fn fresh_runs_sharing_an_opening_read_it_once() {
        let opening: Vec<u32> = (1..=10).collect();
        let prompts: Vec<Vec<TokenId>> = [vec![20, 21], vec![30], vec![40, 41, 42]]
            .into_iter()
            .map(|tail| ids(&[opening.clone(), tail].concat()))
            .collect();
        let layout = RunLayout::new(&runs(&prompts), &[0, 0, 0]);
        assert_eq!(layout.shared, 10);
        assert_eq!(
            layout.pieces,
            vec![
                RunPiece {
                    run: 0,
                    start: 0,
                    rows: 10,
                    trunk: true
                },
                RunPiece {
                    run: 0,
                    start: 10,
                    rows: 2,
                    trunk: false
                },
                RunPiece {
                    run: 1,
                    start: 10,
                    rows: 1,
                    trunk: false
                },
                RunPiece {
                    run: 2,
                    start: 10,
                    rows: 3,
                    trunk: false
                },
            ]
        );
        assert_eq!(layout.last_rows, vec![11, 12, 15]);
        assert_eq!(layout.rows(), 16);
        assert_eq!(&layout.positions[10..13], &[10, 11, 10]);
    }

    #[test]
    fn a_run_keeps_its_own_last_row_and_a_short_opening_is_not_shared() {
        // One prompt is the other's opening: the shared part stops one short
        // of it, so it still has the row its answer is read from.
        let long: Vec<u32> = (1..=12).collect();
        let prompts = vec![ids(&long), ids(&long[..9])];
        let layout = RunLayout::new(&runs(&prompts), &[0, 0]);
        assert_eq!(layout.shared, 8);
        assert_eq!(layout.last_rows, vec![11, 12]);

        let prompts = vec![ids(&[1, 2, 3, 9]), ids(&[1, 2, 3, 8])];
        let layout = RunLayout::new(&runs(&prompts), &[0, 0]);
        assert_eq!(layout.shared, 0);
        assert_eq!(layout.last_rows, vec![3, 7]);
    }

    #[test]
    fn a_resumed_session_shares_nothing() {
        let opening: Vec<u32> = (1..=10).collect();
        let prompts = vec![
            ids(&[opening.clone(), vec![5]].concat()),
            ids(&[opening, vec![6]].concat()),
        ];
        let layout = RunLayout::new(&runs(&prompts), &[0, 64]);
        assert_eq!(layout.shared, 0);
        assert_eq!(layout.positions[11], 64);
    }
}
