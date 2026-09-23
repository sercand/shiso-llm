//! A resident sparse stack: its weights, its sessions, and the pass it plans.
//!
//! [`SparseStack::new`] is the only way to build one, and it takes weights that
//! are already on the device, so a value of this type always means a stack that
//! can run. There is no half-built state to guard against and no second
//! initialization step to forget.
//!
//! # What this owns, and what belongs to the crate above it
//!
//! It owns everything that outlives a launch: the weights, the session table,
//! the activation scratch, and the cache the sessions draw pages from. It owns
//! none of the identity - which file the weights came from, what its tensors
//! were called, which token ends a turn. A crate that knows a checkpoint family
//! supplies the geometry and the imported weights; this runs them.
//!
//! # The pass is planned here and issued next door
//!
//! [`SparseStack::plan_chunk`] returns the ordered work a chunk would do - every
//! launch, its grid, its shared memory, and which of them belong to which
//! captured island - rather than issuing it. The reason is in
//! [`crate::sparse::plan`]: the decisions are what a port gets wrong, and they
//! are checkable on a machine with no device, while the numbers are not.
//!
//! [`crate::sparse::run`] is the other half - the file that claims the cache
//! rows, starts the n-gram gather, writes the host inputs, lands the gathered
//! rows, issues every step and advances the session, in that order. It is an
//! `impl` block on this same type, split out for the per-file line cap, as
//! [`crate::sparse::ckpt`] is.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, DeviceView, alloc_tagged};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, SessionId, TokenId};
use shiso_core::ledger::MemTag;
use shiso_core::page::PAGE_TOKENS;
use shiso_kvstore::{Frontier, KvCache};

use crate::options::EngineOptions;
use crate::sessions::clear_range;
use crate::shape::LayerKind;
use crate::sparse::island::{GraphPolicy, IslandKey};
use crate::sparse::kernels::F32_BYTES;
use crate::sparse::kernels::IDX_QUERY_BLOCK;
use crate::sparse::matmul::Device;
use crate::sparse::moe::RowOrigin;
use crate::sparse::mtp::{Rollback, SpecDepth};
use crate::sparse::paged::{
    PagePlanes, PagedKv, SessionPages, conv_state_bytes, ple_window_bytes, state_bytes,
};
use crate::sparse::plan::Plan;
use crate::sparse::ple::{self, PleGather};
use crate::sparse::recurrent::ChunkPolicy;
use crate::sparse::scratch::{Buf, Scratch};
use crate::sparse::shape::SparseShape;
use crate::sparse::weights::{ModelDesc, SparseWeights};

/// Rows one pass may carry.
///
/// Every activation buffer is allocated at this width, so it is the one number
/// that decides the scratch bill. A prompt longer than this is split rather
/// than refused.
///
/// 2048 rather than 512 because the routed experts are read once per pass: at
/// 512 rows each expert's weights were streamed and decoded four times for a
/// 2048-token scheduler chunk, and prefill ran at 550-610 tok/s on a GB10
/// against 1040-1090 for the reference at 2048. The scratch this costs is
/// about 2 GB, on a device where the weights are 62.
pub const MAX_PREFILL_CHUNK: u32 = 8192;

/// Sessions one decode step may carry, which is also the height of the logits
/// block and the width every per-row pointer array is packed at.
pub const MAX_DECODE_BATCH: u32 = 8;

/// Rows of logits one pass may produce: a segmented verify's every row, which
/// is a decode batch's sequences times one verify each.
pub const MAX_LOGIT_ROWS: u32 = 32;

/// Everything a sparse stack needs that is not geometry and not a weight.
///
/// Composes the shared knobs rather than restating them: the context length and
/// the two cache counts mean the same thing for every model in this crate, and
/// two copies of them would be two chances for the defaults to drift apart.
///
/// Deliberately NOT `#[non_exhaustive]`: a model crate builds one.
#[derive(Debug, Clone, Copy)]
pub struct SparseOptions {
    /// The knobs every model in this crate shares.
    pub engine: EngineOptions,
    /// When the chunked recurrence is allowed to run.
    pub chunk: ChunkPolicy,
    /// Drafts one decode step may carry.
    ///
    /// It is here rather than on a request because it sizes state: every session
    /// allocates one per-position record per drafted row for every recurrent
    /// slot, and a depth that could change between two steps would change the
    /// slab a launch was already planned against.
    pub spec: SpecDepth,
    /// When a pass may be captured and replayed.
    pub graphs: GraphPolicy,
    /// Blocks the device can hold resident at once, which bounds every
    /// cooperative grid.
    ///
    /// A measured device fact rather than an environment knob: it is the
    /// multiprocessor count times the occupancy of the trellis kernels at their
    /// shared-memory budget, and the caller that opened the device is the only
    /// party that can ask.
    pub co_resident_blocks: u32,
    /// Whether the device granted the tiled attention kernel its shared memory.
    pub tiled_attention: bool,
    /// Whether it granted the tensor-core kernels theirs.
    pub tensor_core: bool,
    /// The entry names this build captured for the template kernels.
    ///
    /// Empty on a host that never compiled the kernels, which makes every
    /// template dispatch a typed refusal naming the missing build artifact
    /// rather than a guessed name.
    pub entries: crate::sparse::kernels::EntryTable,
    /// Bytes the streamed n-gram reader may hold rows in.
    ///
    /// A per-stream fact rather than a process one: the table is read a
    /// hundred-odd bytes at a time out of tens of gibibytes, so the cache is
    /// the whole of what keeps a decode step off the device, and two stacks on
    /// two devices want two answers.
    pub ngram_cache_bytes: u64,
    /// Which reader that table is opened through, and how deep it runs.
    pub ngram_reader: shiso_ple::PleConfig,
    /// Rows the widest pass may carry, which is what every activation buffer
    /// is allocated at: a multiple of [`IDX_QUERY_BLOCK`] from one block up
    /// to [`MAX_PREFILL_CHUNK`].
    ///
    /// The scratch is linear in this - about 1.3 MiB a row on the published
    /// geometry, 10 GiB at 8192 - so it is the knob that trades prefill speed
    /// (every routed expert is read once per pass) for key/value pool. A
    /// prompt longer than this is split rather than refused.
    pub prefill_rows: u32,
}

impl SparseOptions {
    /// The defaults a server starts with.
    ///
    /// The block budget defaults to zero, which makes every cooperative grid a
    /// single block: correct, slow, and obviously a placeholder rather than a
    /// plausible-looking number that would hide a caller who never measured.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            engine: EngineOptions::new(),
            chunk: ChunkPolicy::new(),
            spec: SpecDepth::NONE,
            graphs: GraphPolicy {
                spec_depth: 0,
                capture_verify: false,
            },
            co_resident_blocks: 0,
            tiled_attention: true,
            tensor_core: true,
            entries: crate::sparse::kernels::EntryTable::new(&[]),
            ngram_cache_bytes: NGRAM_CACHE_BYTES,
            ngram_reader: shiso_ple::PleConfig::DEFAULT,
            prefill_rows: MAX_PREFILL_CHUNK,
        }
    }
}

/// Bytes the streamed n-gram row cache holds by default.
///
/// Half a gibibyte, which is the measured working set of a decode step's
/// gathers over a table too large to hold any other way: the rows a turn
/// revisits are the ones its recent grams name, and that set is small and hot
/// while the table it is drawn from is tens of gibibytes.
pub const NGRAM_CACHE_BYTES: u64 = 512 << 20;

impl SparseOptions {
    /// The [`SparseOptions::prefill_rows`] a scheduler whose widest prefill
    /// chunk is `widest` tokens needs: rounded up to a whole query block, and
    /// capped at [`MAX_PREFILL_CHUNK`], past which the stack splits a chunk
    /// itself.
    #[must_use]
    pub fn prefill_rows_for(widest: usize) -> u32 {
        let widest = u32::try_from(widest).unwrap_or(u32::MAX);
        widest
            .clamp(IDX_QUERY_BLOCK, MAX_PREFILL_CHUNK)
            .next_multiple_of(IDX_QUERY_BLOCK)
    }
}

impl Default for SparseOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// One recurrent slot's persistent state.
///
/// The two checkpoint slabs are present together or not at all: they are
/// allocated by the same decision - whether this session may speculate - and a
/// rollback that restored a convolution window without its state matrix would
/// leave the slot describing two different positions.
#[derive(Debug)]
pub(crate) struct RecurrentState {
    pub(crate) conv: DeviceBuffer,
    pub(crate) state: DeviceBuffer,
    rollback: Option<RecurrentRollback>,
}

/// One recurrent slot's per-position records, kept while a draft is outstanding.
#[derive(Debug)]
struct RecurrentRollback {
    conv: DeviceBuffer,
    state: DeviceBuffer,
}

/// What a session holds because it may speculate.
///
/// One struct rather than three optional fields on the session, because the
/// three are allocated by one decision and a session holding two of them would
/// be a rollback that restores part of a position.
#[derive(Debug)]
pub(crate) struct SpecState {
    /// Drafts a step may carry, which is how many records each slab holds.
    depth: SpecDepth,
    /// The n-gram block's per-position windows, when the stack has an n-gram
    /// block.
    ngram_ckpt: Option<DeviceBuffer>,
    /// The residual row the draft head consumes on the next step.
    ///
    /// Staged out of the shared activation scratch rather than read from it,
    /// because the next thing to write that scratch is another session's chunk:
    /// the draft for the last committed row cannot run until the token after it
    /// arrives, and that arrives after the hand-off.
    pub(crate) pending: DeviceBuffer,
    /// Whether [`SpecState::pending`] holds a row, and therefore whether the
    /// draft head can run at all on the next step.
    pub(crate) pending_valid: bool,
}

/// One live session.
#[derive(Debug)]
pub struct SparseSession {
    id: SessionId,
    pub(crate) pages: SessionPages,
    pub(crate) states: Vec<RecurrentState>,
    /// The n-gram block's dilated window, when the stack has one.
    pub(crate) ngram_window: Option<DeviceBuffer>,
    /// What this session holds because it may speculate, or `None` on a session
    /// opened for plain decode.
    pub(crate) spec: Option<SpecState>,
    pub(crate) written: u32,
    pub(crate) history: Vec<TokenId>,
}

impl SparseSession {
    /// Absolute position of the next token this session writes.
    #[must_use]
    pub const fn written(&self) -> u32 {
        self.written
    }

    /// The tokens this session has consumed, oldest first.
    #[must_use]
    pub fn history(&self) -> &[TokenId] {
        &self.history
    }

    /// The page table a cache-touching launch reads.
    #[must_use]
    pub fn pages(&self) -> DeviceView<'_> {
        self.pages.view()
    }

    /// The `ordinal`-th recurrent slot's convolution window and state matrix, in
    /// the order a checkpoint blob lays them out.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when no slot carries that ordinal.
    pub fn recurrent(&self, ordinal: u32) -> CoreResult<(DeviceView<'_>, DeviceView<'_>)> {
        let index = usize::try_from(ordinal).unwrap_or(usize::MAX);
        let state = self.states.get(index).ok_or(CoreError::OutOfBounds {
            offset: u64::from(ordinal),
            len: 1,
            buffer_bytes: self.states.len() as u64,
        })?;
        Ok((state.conv.view(), state.state.view()))
    }

    /// The `ordinal`-th recurrent slot's per-position rollback records.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when no slot carries that ordinal, and
    /// [`CoreError::Unsupported`] when this session was not opened for
    /// speculation and therefore has no records to roll back to.
    pub fn recurrent_rollback(&self, ordinal: u32) -> CoreResult<(DeviceView<'_>, DeviceView<'_>)> {
        let index = usize::try_from(ordinal).unwrap_or(usize::MAX);
        let slot = self.states.get(index).ok_or(CoreError::OutOfBounds {
            offset: u64::from(ordinal),
            len: 1,
            buffer_bytes: self.states.len() as u64,
        })?;
        let rollback = slot.rollback.as_ref().ok_or(CoreError::Unsupported {
            what: "the per-position recurrent records of a session that was not opened for \
                   speculation",
            backend: "engine",
        })?;
        Ok((rollback.conv.view(), rollback.state.view()))
    }

    /// The frontier a commit extends and the path it extends it with.
    ///
    /// Both at once, because a commit needs to advance one while reading the
    /// other and they are fields of the same session: handed out one accessor at
    /// a time, the caller would have to copy the path, and the path is as long
    /// as the conversation.
    pub(crate) const fn commit_parts(&mut self) -> (&mut Frontier, &[TokenId]) {
        (self.pages.frontier_mut(), self.history.as_slice())
    }

    /// The n-gram block's window, when this stack has one.
    #[must_use]
    pub fn ngram_window(&self) -> Option<DeviceView<'_>> {
        self.ngram_window.as_ref().map(DeviceBuffer::view)
    }

    /// The n-gram block's per-position rollback records, when this session has
    /// both an n-gram block and a draft outstanding.
    #[must_use]
    pub fn ngram_window_rollback(&self) -> Option<DeviceView<'_>> {
        self.spec
            .as_ref()?
            .ngram_ckpt
            .as_ref()
            .map(DeviceBuffer::view)
    }

    /// Drafts one step of this session may carry.
    #[must_use]
    pub fn spec_depth(&self) -> SpecDepth {
        self.spec
            .as_ref()
            .map_or(SpecDepth::NONE, |spec| spec.depth)
    }

    /// The staged residual row the draft head consumes next, or `None` when
    /// this session has none - it does not speculate, or nothing has been
    /// staged since the last flush.
    ///
    /// Without a staged row there is no draft and therefore no proposal to
    /// gate the prompt lookup with, which is why speculation reports it rather
    /// than drafting from a buffer that holds the previous request's rows.
    #[must_use]
    pub fn pending_draft(&self) -> Option<DeviceView<'_>> {
        let spec = self.spec.as_ref()?;
        spec.pending_valid.then(|| spec.pending.view())
    }

    /// How many recurrent slots this session carries state for.
    #[must_use]
    pub fn recurrent_slots(&self) -> usize {
        self.states.len()
    }

    /// The handle this session was opened under.
    ///
    /// The one place it is read is the execution context a captured island is
    /// filed under: two sessions share the activation scratch and differ only
    /// in the state they own alone, so the handle is what tells two otherwise
    /// identical captures apart.
    #[must_use]
    pub const fn id(&self) -> SessionId {
        self.id
    }
}

/// Where one island's steps sit inside a pass's plan.
///
/// A range rather than a plan of its own, because the whole point of a pass is
/// that its steps are in one order: an island that owned its steps would let a
/// caller run them in a different one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct IslandSpan {
    /// What identifies this island's captured graph.
    pub key: IslandKey,
    /// Index of its first step in the pass's plan.
    pub start: usize,
    /// How many steps it covers.
    pub len: usize,
}

/// One planned pass through the stack.
#[derive(Debug)]
#[non_exhaustive]
pub struct Pass {
    /// Every step, in order.
    pub plan: Plan,
    /// Where each island sits, in the order they run.
    pub islands: Vec<IslandSpan>,
    /// Whether this row count may be captured and replayed at all.
    pub capturable: bool,
    /// Whether the indexer's selection was live for this pass.
    pub selected: bool,
    /// Rows the pass was planned for.
    pub rows: u32,
    /// Whether those rows are one sequence's consecutive positions or one row
    /// per sequence.
    ///
    /// It travels with the pass because it decides which of two kernels a
    /// state-touching launch is: a pass planned for a batch and issued against
    /// one session would bind per-row arrays nobody packed.
    pub origin: RowOrigin,
    /// Rows in the encoder block this pass splices out of, or `None` for a
    /// text pass.
    ///
    /// It travels with the pass for the reason [`Pass::origin`] does: a pass
    /// planned with a splice and issued against a binder that names no encoder
    /// block is a pass whose first launch reads a pointer nobody set, and a
    /// pass planned without one and issued for a media chunk leaves every
    /// placeholder token holding the embedding of its own placeholder id -
    /// which is a fluent answer about nothing in the picture.
    pub media_rows: Option<u32>,
    /// The gathers the n-gram block consumes, one per row, in row order.
    ///
    /// They travel with the pass because they are started before the layer loop
    /// and awaited inside it: a handle that lived only at the block would have
    /// to be created there, which is the one place the read latency cannot be
    /// hidden. Empty on a stack that folds in no n-gram block, and one per row
    /// otherwise - a decode batch's rows belong to different sequences, so each
    /// hashes its own trailing grams.
    pub gather: Vec<PleGather>,
}

/// Everything that lives on the device, in construction order.
///
/// The fields are `pub(crate)` because the checkpoint tier in
/// [`crate::sparse::ckpt`] is this type's own second half, split out for the
/// per-file line cap rather than because it is a layer above.
#[derive(Debug)]
pub(crate) struct Resident {
    pub(crate) weights: SparseWeights,
    pub(crate) kv: PagedKv,
    pub(crate) cache: Arc<KvCache>,
    pub(crate) scratch: Scratch,
    /// The streamed n-gram table and its staging, on a stack that folds an
    /// n-gram block in.
    ///
    /// Present exactly when the geometry names an n-gram slot: the constructor
    /// refuses either without the other, because a slot with no table decodes
    /// its layer out of whatever the allocator left behind and a table with no
    /// slot is a reader nothing ever asks.
    pub(crate) ngram: Option<ple::NgramTier>,
}

/// The host buffers this stack moves bytes through, reused between calls.
///
/// One allocation each rather than a fresh `Vec` per call: the state blob is
/// megabytes and a resume is on the path a request waits on, and the input rows
/// are rebuilt for every chunk of every session.
#[derive(Debug, Default)]
pub(crate) struct Staging {
    /// The recurrent stack, the n-gram window and the pending draft row.
    pub(crate) ckpt: Vec<u8>,
    /// The frontier logits row, which travels beside the blob rather than in it.
    pub(crate) logits: Vec<u8>,
    /// The token ids, positions and rotary table one chunk is run with.
    pub(crate) inputs: crate::sparse::inputs::InputRows,
    /// The session path a gather hashes its row identifiers out of: the prefix
    /// the session has written, then the chunk about to run.
    ///
    /// Reused rather than built per call for the reason the input rows are: the
    /// n-gram hash indexes the path by absolute position, so it needs the whole
    /// of it, and a decode step would otherwise allocate a conversation-length
    /// vector per token.
    pub(crate) path: Vec<TokenId>,
}

/// A resident sparse mixture-of-experts stack.
///
/// The fields are `pub(crate)` for the reason [`Resident`]'s are: the checkpoint
/// tier is this type's own second half.
pub struct SparseStack {
    pub(crate) backend: Arc<dyn Backend>,
    pub(crate) shape: SparseShape,
    pub(crate) opts: SparseOptions,
    pub(crate) desc: ModelDesc,
    pub(crate) device: Device,
    pub(crate) recurrent_slots: usize,
    pub(crate) resident: Option<Resident>,
    pub(crate) sessions: HashMap<SessionId, SparseSession>,
    pub(crate) order: Vec<SessionId>,
    counter: NonZeroU64,
    /// The host buffers a save and a resume move bytes through.
    pub(crate) staging: Staging,
    /// The clock the cache prices entries against: seconds since this stack was
    /// built, which is monotonic and needs no wall clock.
    clock: std::time::Instant,
    /// The disk tier, once a caller has opened a directory over it.
    pub(crate) ckpt: Option<crate::ckpt::Checkpoint>,
    /// The residual tap, while a caller has one armed; see
    /// [`crate::sparse::tap`].
    pub(crate) tap: Option<crate::sparse::tap::ResidualTap>,
}

impl std::fmt::Debug for SparseStack {
    /// Names the stack by its geometry and its live sessions.
    ///
    /// Never the buffers: a device allocation has no host-readable content, so
    /// printing one would produce an identifier that means nothing to a reader
    /// and a line long enough to bury the fields that do.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseStack")
            .field("backend", &self.backend.backend_name())
            .field("shape", &self.shape)
            .field("opts", &self.opts)
            .field("device", &self.device)
            .field("recurrent_slots", &self.recurrent_slots)
            .field("resident", &self.resident.is_some())
            .field("sessions", &self.sessions.len())
            .field("order", &self.order)
            .field("counter", &self.counter)
            .field("checkpointing", &self.ckpt.is_some())
            .field("tapped", &self.tap.is_some())
            .field("staging", &self.staging)
            .field("clock", &self.clock)
            .field("layers", &self.desc.layers.len())
            .finish()
    }
}

/// The line a resident stack announces itself with: its geometry, and what
/// its cache costs per token, per session and per checkpoint - the figures a
/// capacity question starts from.
fn log_resident(shape: &SparseShape, opts: &SparseOptions, kv: &PagedKv, ckpt_bytes: u64) {
    tracing::info!(
        layers = shape.n_layers(),
        recurrent = shape.count(LayerKind::Gdn),
        experts = shape.moe.experts,
        streams = shape.streams,
        ctx = opts.engine.ctx,
        pool_pages = opts.engine.kv_pool_pages,
        pool_tokens = u64::from(opts.engine.kv_pool_pages) * u64::from(PAGE_TOKENS),
        page_bytes = kv.page_bytes(),
        kv_bytes_per_token = kv.page_bytes() / u64::from(PAGE_TOKENS),
        session_bytes = session_bytes(shape, opts.spec),
        ckpt_bytes,
        ckpt_slots = opts.engine.kv_ckpt_slots,
        "sparse stack resident"
    );
}

/// Refuses a pass width the stack cannot be built at.
///
/// The floor is one query block, which is also what keeps every decode and
/// verify pass (at most [`MAX_LOGIT_ROWS`] rows) inside the scratch; the
/// ceiling is the width the kernels' own tables were sized for.
fn check_prefill_rows(rows: u32) -> CoreResult<()> {
    if !(IDX_QUERY_BLOCK..=MAX_PREFILL_CHUNK).contains(&rows)
        || !rows.is_multiple_of(IDX_QUERY_BLOCK)
    {
        return Err(CoreError::InvalidValue {
            key: "prefill_rows",
            value: format!(
                "{rows} is not a multiple of {IDX_QUERY_BLOCK} between {IDX_QUERY_BLOCK} and \
                 {MAX_PREFILL_CHUNK}"
            ),
        });
    }
    Ok(())
}

/// Device bytes one session allocates when it opens.
///
/// Every recurrent slot's window and state, one rollback record of each per
/// drafted row, the n-gram window and its records, and the staged draft row -
/// what [`SparseStack::open_session`] allocates, counted the same way.
fn session_bytes(shape: &SparseShape, spec: SpecDepth) -> u64 {
    let records = u64::from(spec.rollback_slots());
    let copies = if spec.is_speculative() {
        records.saturating_add(1)
    } else {
        1
    };
    let recurrent = u64::from(shape.count(LayerKind::Gdn))
        .saturating_mul(conv_state_bytes(shape).saturating_add(state_bytes(shape)));
    let pending = if spec.is_speculative() {
        u64::from(shape.stream_dim()).saturating_mul(F32_BYTES)
    } else {
        0
    };
    recurrent
        .saturating_add(ple_window_bytes(shape))
        .saturating_mul(copies)
        .saturating_add(pending)
}

impl SparseStack {
    /// Builds a stack over weights that are already on the device.
    ///
    /// `ngram` is the streamed table the n-gram block gathers its rows from. It
    /// is an argument because it is not a weight: it is a reader over a file
    /// nothing makes resident, opened by whichever crate knows which tensor of
    /// which checkpoint holds it. It is required exactly when the geometry
    /// names an n-gram slot, and that is checked here rather than left to the
    /// pass: a slot with no table would decode a whole layer's contribution to
    /// the residual out of whatever the allocator last left at that address,
    /// and every launch of that pass would still succeed.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the weight list and the shape disagree
    /// about the layer count or a slot's kind, [`CoreError::Unsupported`] when
    /// the geometry and the table disagree about whether there is an n-gram
    /// block at all, whatever [`SparseShape::validate`] and
    /// [`ple::NgramTier::open`] report, and
    /// [`CoreError::Backend`] when a device allocation fails.
    pub fn new(
        backend: &Arc<dyn Backend>,
        shape: SparseShape,
        weights: SparseWeights,
        opts: SparseOptions,
        ngram: Option<ple::NgramTable>,
    ) -> CoreResult<Self> {
        shape.validate()?;
        check_layers(&shape, &weights)?;
        check_prefill_rows(opts.prefill_rows)?;
        // Everything resident so far is weights: the ledger's second mark.
        if let Some(ledger) = backend.ledger() {
            ledger.mark("after_weights", backend.memory_info());
        }
        let recurrent_slots = usize::try_from(shape.count(LayerKind::Gdn)).unwrap_or(0);
        // One extra rank of planes when the checkpoint brought a draft head,
        // because the head runs the same indexer over its own rows.
        let planes = PagePlanes {
            indexer: true,
            draft: weights.draft.is_some(),
        };
        let scratch = Scratch::new(
            backend,
            &shape,
            opts.prefill_rows,
            MAX_LOGIT_ROWS,
            opts.engine.ctx,
        )?;
        let ngram = match (shape.ple, ngram) {
            (Some(ple), Some(table)) => {
                crate::sparse::memory::watch_ngram(backend, &table, opts.ngram_cache_bytes);
                Some(ple::NgramTier::open(
                    backend,
                    ple,
                    table,
                    opts.prefill_rows,
                )?)
            }
            (None, None) => None,
            (Some(_), None) => {
                return Err(CoreError::Unsupported {
                    what: "a geometry that folds in an n-gram block with no streamed table to \
                           gather its rows from, which would decode that layer's contribution \
                           to the residual out of unwritten memory",
                    backend: "engine",
                });
            }
            (None, Some(_)) => {
                return Err(CoreError::Unsupported {
                    what: "a streamed n-gram table for a geometry with no n-gram slot, which is \
                           a reader no launch in the pass would ever ask",
                    backend: "engine",
                });
            }
        };
        // The pool last, so that a pool sized to the device is sized from what
        // the weights, the scratch and the n-gram tier left of it.
        let mut opts = opts;
        let geometry = crate::sparse::paged::cache_geometry(&shape, planes, opts.spec);
        opts.engine.fit_pool(
            backend.as_ref(),
            crate::options::PoolCost {
                page_bytes: geometry.page_bytes,
                session_bytes: session_bytes(&shape, opts.spec),
                fixed_bytes: u64::from(opts.engine.kv_ckpt_slots)
                    .saturating_mul(geometry.ckpt_bytes),
            },
        )?;
        let kv = PagedKv::open(backend, &shape, planes, opts.engine.kv_pool_pages)?;
        let cache = KvCache::open(
            &geometry,
            Box::new(kv.clone()),
            opts.engine.kv_pool_pages,
            opts.engine.kv_ckpt_slots,
            None,
        )
        .map_err(cache_failure)?;
        let cache = Arc::new(cache);
        crate::sparse::memory::watch_cache(backend, &cache, kv.page_bytes(), geometry.ckpt_bytes);
        let device = Device {
            blas: backend.as_blas().is_some(),
            capturing: false,
            max_co_resident: opts.co_resident_blocks,
            half_block_input: false,
        };
        let desc = weights.desc(ngram.as_ref().map(|tier| tier.table().row()));
        log_resident(&shape, &opts, &kv, geometry.ckpt_bytes);
        Ok(Self {
            backend: Arc::clone(backend),
            shape,
            opts,
            desc,
            device,
            recurrent_slots,
            resident: Some(Resident {
                weights,
                kv,
                cache,
                scratch,
                ngram,
            }),
            sessions: HashMap::new(),
            order: Vec::new(),
            counter: NonZeroU64::MIN,
            staging: Staging::default(),
            clock: std::time::Instant::now(),
            ckpt: None,
            tap: None,
        })
    }

    /// Arms the residual tap: every single-session chunk pass from here on
    /// copies its last row's residual after the prologue and after every
    /// island, and runs without device graphs. See [`crate::sparse::tap`].
    ///
    /// Arming again discards whatever the previous tap held.
    ///
    /// # Errors
    ///
    /// Whatever [`crate::sparse::tap::ResidualTap::new`] refuses.
    pub fn arm_residual_tap(&mut self) -> CoreResult<()> {
        self.tap = Some(crate::sparse::tap::ResidualTap::new(
            self.shape.streams,
            self.shape.hidden,
        )?);
        Ok(())
    }

    /// Disarms the residual tap and returns what the last tapped pass copied.
    ///
    /// `None` means no tap was armed.
    pub fn take_residual_tap(&mut self) -> Option<crate::sparse::tap::ResidualTap> {
        self.tap.take()
    }

    /// The geometry this stack runs.
    #[must_use]
    pub const fn shape(&self) -> &SparseShape {
        &self.shape
    }

    /// Tokens one session may hold, prompt included.
    #[must_use]
    pub const fn context_capacity(&self) -> u32 {
        self.opts.engine.ctx
    }

    /// Rows the widest pass this stack runs may carry; a longer prompt is
    /// split into chunks of at most this many.
    #[must_use]
    pub const fn prefill_rows(&self) -> u32 {
        self.opts.prefill_rows
    }

    /// Recurrent slots, which is how many states a session allocates.
    #[must_use]
    pub const fn recurrent_slots(&self) -> usize {
        self.recurrent_slots
    }

    /// The device facts every projection tier closes over.
    #[must_use]
    pub const fn device(&self) -> Device {
        self.device
    }

    /// The device this stack runs on.
    ///
    /// Public because the session state a caller moves bytes into and out of -
    /// a staged draft row, a restored checkpoint - lives behind this stack's own
    /// allocations, and a caller holding a second handle on the device would be
    /// a second party able to free them.
    #[must_use]
    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// One of the shared activation buffers a pass writes.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when the stack's device state has already been
    /// released, and whatever the scratch reports for a buffer this geometry has
    /// none of.
    pub fn scratch_view(&self, buf: Buf) -> CoreResult<DeviceView<'_>> {
        self.resident()?.scratch.view(buf)
    }

    /// The widths every plan is built from.
    #[must_use]
    pub const fn desc(&self) -> &ModelDesc {
        &self.desc
    }

    /// Seconds since this stack was built, which is the clock the cache prices
    /// entries against.
    pub(crate) fn now(&self) -> f64 {
        self.clock.elapsed().as_secs_f64()
    }

    /// The device half, or the typed failure a torn-down stack answers with.
    pub(crate) fn resident(&self) -> CoreResult<&Resident> {
        self.resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a stack whose device state teardown already released",
            backend: "engine",
        })
    }

    /// Allocates a fresh, empty session.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the device state a session needs cannot be
    /// created, and whatever the cache reports for a frontier it cannot hand
    /// out.
    pub fn open_session(&mut self) -> CoreResult<SessionId> {
        let capacity = self.opts.engine.ctx;
        let frontier = self.resident()?.cache.cold_frontier();
        let id = SessionId::next(&mut self.counter).ok_or(CoreError::Backend {
            backend: "engine",
            op: "open_session",
            source: Box::new(std::io::Error::other("session id space exhausted")),
        })?;
        // One record per drafted row. Zero at plain-decode depth, which is what
        // makes the slabs absent rather than empty: a zero-byte allocation is
        // still an address a kernel would write to.
        let records = u64::from(self.opts.spec.rollback_slots());
        let states = (0..self.recurrent_slots)
            .map(|_| {
                let conv_bytes = conv_state_bytes(&self.shape);
                let state_bytes = state_bytes(&self.shape);
                let rollback = if self.opts.spec.is_speculative() {
                    Some(RecurrentRollback {
                        conv: alloc_tagged(
                            &self.backend,
                            MemTag::RecurrentRollback,
                            records.saturating_mul(conv_bytes),
                        )?,
                        state: alloc_tagged(
                            &self.backend,
                            MemTag::RecurrentRollback,
                            records.saturating_mul(state_bytes),
                        )?,
                    })
                } else {
                    None
                };
                // A fresh allocation is whatever the last tenant of that
                // address left behind - the device allocator hands back raw
                // pages and does not clear them. A recurrent slot is read
                // before it is written on a session's first chunk, so without
                // this the fold carries a stranger's state and still answers.
                // The dense family clears its stack for the same reason.
                let conv = alloc_tagged(&self.backend, MemTag::RecurrentState, conv_bytes)?;
                let state = alloc_tagged(&self.backend, MemTag::RecurrentState, state_bytes)?;
                clear_range(&self.backend, &conv)?;
                clear_range(&self.backend, &state)?;
                if let Some(slabs) = rollback.as_ref() {
                    clear_range(&self.backend, &slabs.conv)?;
                    clear_range(&self.backend, &slabs.state)?;
                }
                Ok(RecurrentState {
                    conv,
                    state,
                    rollback,
                })
            })
            .collect::<CoreResult<Vec<_>>>()?;
        let window_bytes = ple_window_bytes(&self.shape);
        let ngram_window = self
            .shape
            .ple
            .map(|_| alloc_tagged(&self.backend, MemTag::SessionOther, window_bytes))
            .transpose()?;
        if let Some(window) = ngram_window.as_ref() {
            clear_range(&self.backend, window)?;
        }
        let spec = if self.opts.spec.is_speculative() {
            // The n-gram block's per-position records are cleared for the
            // reason its recurrent siblings are, and not clearing them here
            // while clearing those would be the asymmetry, not the fix: a
            // rollback reads them, and a device allocator hands back whatever
            // was last at that address.
            let ngram_ckpt = self
                .shape
                .ple
                .map(|_| {
                    alloc_tagged(
                        &self.backend,
                        MemTag::RecurrentRollback,
                        records.saturating_mul(window_bytes),
                    )
                })
                .transpose()?;
            if let Some(slab) = ngram_ckpt.as_ref() {
                clear_range(&self.backend, slab)?;
            }
            Some(SpecState {
                depth: self.opts.spec,
                ngram_ckpt,
                pending: alloc_tagged(
                    &self.backend,
                    MemTag::SessionOther,
                    u64::from(self.shape.stream_dim()).saturating_mul(F32_BYTES),
                )?,
                pending_valid: false,
            })
        } else {
            None
        };
        let pages = SessionPages::open(&self.backend, frontier, capacity)?;
        let session = SparseSession {
            id,
            pages,
            states,
            ngram_window,
            spec,
            written: 0,
            history: Vec::new(),
        };
        self.sessions.insert(id, session);
        self.order.push(id);
        Ok(id)
    }

    /// Releases a session's state. A second call for the same handle is
    /// [`CoreError::UnknownSession`], not a silent success.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`] when the handle is not one this stack
    /// minted or has already been closed.
    pub fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        let live = self
            .sessions
            .remove(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        self.order.retain(|id| *id != session);
        // Dropping the session releases its recurrent stack and drops its
        // frontier, which is what returns the page references to the pool.
        drop(live);
        Ok(())
    }

    /// One live session.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`] when the handle is not one this stack
    /// minted.
    pub fn session(&self, session: SessionId) -> CoreResult<&SparseSession> {
        self.sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))
    }

    /// Claims the cache rows the next `n_tok` tokens will write, and says
    /// whether anything captured against the old page table has to go.
    ///
    /// The claim happens before the pass that fills those rows, not after: a
    /// page the frontier has not claimed may be handed to another session at any
    /// moment.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::ContextOverflow`] when the
    /// chunk reaches past the context, and whatever the pool reports.
    pub fn reserve(&mut self, session: SessionId, n_tok: u32) -> CoreResult<bool> {
        let capacity = self.opts.engine.ctx;
        let live = self
            .sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let next = live
            .written
            .checked_add(n_tok)
            .ok_or(CoreError::ContextOverflow {
                session,
                cap: capacity,
            })?;
        if next > capacity {
            return Err(CoreError::ContextOverflow {
                session,
                cap: capacity,
            });
        }
        let change = live.pages.ensure(next)?;
        Ok(change == crate::sparse::paged::TableChange::Republished)
    }

    /// Records that a session advanced by `tokens`, and registers the span it
    /// just wrote with the cache.
    ///
    /// Separate from [`SparseStack::reserve`] because the rows are claimed
    /// before the pass and the path is extended after it: a session that
    /// recorded the tokens first and then failed to run them would believe it
    /// held a prefix nothing computed.
    ///
    /// `earned` says this span ends a prompt rather than being one more step of
    /// a generation. It travels separately from `frontier_row` because the two
    /// answer different questions and only look like one: see
    /// [`SparseStack::commit`].
    ///
    /// `frontier_row` names which row of the logits block this session's last
    /// position produced, or `None` for a chunk that produced none. It is not a
    /// convenience: a checkpoint admitted without that row cannot be resumed
    /// from without re-running a step, which is the one thing the disk tier
    /// promises not to do - so the row travels with the state rather than beside
    /// it, and the parameter is what makes leaving it out a decision rather than
    /// an omission.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and whatever the cache reports for a span
    /// it could not register.
    pub fn advance(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        earned: bool,
        frontier_row: Option<u32>,
    ) -> CoreResult<()> {
        let n_tok = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        {
            let live = self
                .sessions
                .get_mut(&session)
                .ok_or(CoreError::UnknownSession(session))?;
            live.history.extend_from_slice(tokens);
            live.written = live.written.saturating_add(n_tok);
        }
        self.commit(session, earned, frontier_row)
    }

    /// Puts a session back where it stood before the drafts the target rejected.
    ///
    /// Every piece of per-position state the verify pass advanced is copied back
    /// from the record the sequential kernels wrote for the last accepted row:
    /// every recurrent slot's convolution window and state matrix, and the n-gram
    /// block's dilated window. The key/value cache is deliberately *not* rolled
    /// back, and that is not an omission - the rejected rows' cache entries are
    /// at positions the session's frontier has not advanced past, so the next
    /// pass overwrites them before anything reads them.
    ///
    /// A [`Rollback`] cannot be constructed for a step whose draft was accepted
    /// whole, so reaching this function always means there is state to restore.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::Unsupported`] when the session
    /// was not opened for speculation and has no records to restore from, and
    /// whatever the device reports for a copy.
    pub fn spec_rollback(&self, session: SessionId, rollback: Rollback) -> CoreResult<()> {
        let live = self.session(session)?;
        let conv_bytes = conv_state_bytes(&self.shape);
        let state_bytes = state_bytes(&self.shape);
        for ordinal in 0..u32::try_from(live.states.len()).unwrap_or(u32::MAX) {
            let (conv, state) = live.recurrent(ordinal)?;
            let (conv_ckpt, state_ckpt) = live.recurrent_rollback(ordinal)?;
            self.backend.copy_d2d(
                conv,
                conv_ckpt.slice(rollback.offset(conv_bytes), conv_bytes)?,
            )?;
            self.backend.copy_d2d(
                state,
                state_ckpt.slice(rollback.offset(state_bytes), state_bytes)?,
            )?;
        }
        if let (Some(window), Some(records)) = (live.ngram_window(), live.ngram_window_rollback()) {
            let stride = ple_window_bytes(&self.shape);
            self.backend
                .copy_d2d(window, records.slice(rollback.offset(stride), stride)?)?;
        }
        tracing::debug!(%session, slot = rollback.slot(), "speculative state rolled back");
        Ok(())
    }

    /// Stages row `row` of the pass's residual as the row the draft head reads
    /// on the next step.
    ///
    /// It is copied out of the shared activation scratch rather than read from
    /// it, because the next thing to write that scratch is another session's
    /// chunk: the draft for the last committed row cannot run until the token
    /// after it arrives, and that arrives after the hand-off.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::Unsupported`] when the session
    /// does not speculate, and whatever the device reports for the copy.
    pub fn stage_draft_row(&mut self, session: SessionId, row: u32) -> CoreResult<()> {
        let width = u64::from(self.shape.stream_dim()).saturating_mul(F32_BYTES);
        // Split borrows: the staged row belongs to the session and the rows it
        // comes from belong to the stack's shared scratch, so both are held at
        // once rather than one accessor at a time.
        let Self {
            backend,
            resident,
            sessions,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a stack whose device state teardown already released",
            backend: "engine",
        })?;
        let residual = resident
            .scratch
            .view(Buf::Residual)?
            .slice(u64::from(row).saturating_mul(width), width)?;
        let live = sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let spec = live.spec.as_mut().ok_or(CoreError::Unsupported {
            what: "staging a draft row on a session that was not opened for speculation",
            backend: "engine",
        })?;
        backend.copy_d2d(spec.pending.view(), residual)?;
        spec.pending_valid = true;
        Ok(())
    }

    /// Forgets the staged draft row, which is what a session does when the rows
    /// it was standing on are no longer the rows it is standing on.
    ///
    /// The bytes are left alone: what makes a row unusable is that nothing can
    /// say which position it belongs to, and clearing the flag is what says that.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`].
    pub fn discard_draft_row(&mut self, session: SessionId) -> CoreResult<()> {
        let live = self
            .sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        if let Some(spec) = live.spec.as_mut() {
            spec.pending_valid = false;
        }
        Ok(())
    }

    /// The operand resolver one session's launches bind through.
    ///
    /// Public because the two halves of a pass - deciding it and issuing it -
    /// are separate on purpose, and a caller checking what a step would be
    /// handed needs the same resolver the issuing half uses rather than a
    /// second one that could drift from it.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and [`CoreError::Unsupported`] when the
    /// stack's device state has already been released.
    pub fn binder(&self, session: SessionId) -> CoreResult<crate::sparse::bind::Binder<'_>> {
        self.binder_rows(&[session])
    }

    /// The operand resolver a decode batch's launches bind through, one session
    /// per row and in row order.
    ///
    /// The row order is the whole of what this adds, and it is not a detail:
    /// every per-row array the batched kernels dereference is indexed by
    /// `blockIdx.y`, so a binder assembled in a different order than the one the
    /// inputs were written in would hand row one's state to row zero.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::Unsupported`] for an empty
    /// batch, one wider than the logits block, or a stack whose device state
    /// teardown already released, and [`CoreError::InvalidValue`] when one
    /// session appears twice.
    pub fn binder_rows(
        &self,
        sessions: &[SessionId],
    ) -> CoreResult<crate::sparse::bind::Binder<'_>> {
        self.binder_rows_media(sessions, None)
    }

    /// The same resolver, also naming the encoder block a multimodal pass
    /// splices out of.
    ///
    /// Separate from [`SparseStack::binder_rows`] rather than an optional
    /// argument on it, because every existing caller is a text pass and a
    /// caller that had to pass `None` would be a caller that could pass
    /// `Some` by accident.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::binder_rows`].
    pub fn binder_rows_media<'a>(
        &'a self,
        sessions: &[SessionId],
        media: Option<DeviceView<'a>>,
    ) -> CoreResult<crate::sparse::bind::Binder<'a>> {
        distinct(sessions)?;
        let resident = self.resident()?;
        let rows = sessions
            .iter()
            .map(|id| self.session(*id))
            .collect::<CoreResult<Vec<_>>>()?;
        let binder = crate::sparse::bind::Binder::batched(
            &resident.scratch,
            &resident.weights,
            &rows,
            &resident.kv,
        )?;
        let binder = match resident.ngram.as_ref() {
            Some(tier) => binder.with_ngram(tier.views()),
            None => binder,
        };
        let binder = match media {
            Some(rows) => binder.with_media(rows),
            None => binder,
        };
        Ok(match self.backend.as_coop() {
            Some(coop) => binder.with_coop(coop),
            None => binder,
        })
    }

    /// Issues a planned pass for one session.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], whatever the binder reports for an
    /// operand this session cannot name, and whatever the executor reports for
    /// a tier this build cannot issue.
    pub fn execute(
        &self,
        session: SessionId,
        pass: &Pass,
    ) -> CoreResult<crate::sparse::exec::PassReport> {
        self.execute_rows(&[session], pass)
    }

    /// Issues a planned pass for a decode batch, one session per row.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::execute`], plus [`CoreError::ShapeMismatch`] when the
    /// pass was planned for a different number of rows than it is being issued
    /// against - which would bind per-row arrays one row short of what a launch
    /// reads.
    pub fn execute_rows(
        &self,
        sessions: &[SessionId],
        pass: &Pass,
    ) -> CoreResult<crate::sparse::exec::PassReport> {
        self.execute_rows_media(sessions, pass, None)
    }

    /// The same, also naming the encoder block the pass splices out of.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::execute_rows`], plus [`CoreError::Unsupported`] when
    /// what the pass was planned with and what it is issued with disagree -
    /// a splice over a block nobody named, or a media chunk whose pass has no
    /// splice in it. Both leave every launch succeeding.
    pub fn execute_rows_media<'a>(
        &'a self,
        sessions: &[SessionId],
        pass: &Pass,
        media: Option<DeviceView<'a>>,
    ) -> CoreResult<crate::sparse::exec::PassReport> {
        self.execute_rows_tapped(sessions, pass, media, None)
    }

    /// The same pass, copying residual rows into `tap` when one is given.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::execute_rows_media`], plus whatever
    /// [`crate::sparse::exec::Exec::run_pass_tapped`] reports for the copy.
    pub(crate) fn execute_rows_tapped<'a>(
        &'a self,
        sessions: &[SessionId],
        pass: &Pass,
        media: Option<DeviceView<'a>>,
        tap: Option<&mut crate::sparse::tap::ResidualTap>,
    ) -> CoreResult<crate::sparse::exec::PassReport> {
        if pass.media_rows.is_some() != media.is_some() {
            return Err(CoreError::Unsupported {
                what: "a pass whose splice and whose encoder block disagree about whether \
                       this chunk carries a picture",
                backend: "engine",
            });
        }
        let rows = u32::try_from(sessions.len()).unwrap_or(u32::MAX);
        if let RowOrigin::Segmented(runs) = pass.origin
            && runs.count() != rows
        {
            return Err(CoreError::ShapeMismatch {
                field: "segmented pass sessions",
                expected: u64::from(runs.count()),
                got: u64::from(rows),
            });
        }
        if pass.origin == RowOrigin::Unrelated && pass.rows != rows {
            return Err(CoreError::ShapeMismatch {
                field: "decode batch rows",
                expected: u64::from(pass.rows),
                got: u64::from(rows),
            });
        }
        // A pass still holding its gather is a pass whose rows nobody fetched.
        // Issuing it would bind the staging buffer anyway and decode one
        // layer's whole contribution to the residual out of whatever was last
        // at that address, with every launch succeeding.
        if !pass.gather.is_empty() {
            return Err(CoreError::Unsupported {
                what: "a pass whose streamed n-gram rows have not been gathered, which would \
                       decode one layer out of memory nothing wrote",
                backend: "engine",
            });
        }
        // Built here rather than travelling from the caller, and built from the
        // pass rather than beside it: the height the kernel bounds its reads by
        // is then the same number the plan was made with, and the residual
        // width is the stack's own. A second copy of either is a copy that can
        // be the one that is wrong.
        let args = pass.media_rows.map(|src_rows| crate::launch::SpliceArgs {
            rows: pass.rows,
            hidden: self.shape.hidden,
            src_rows,
        });
        let binder = self.binder_rows_media(sessions, media)?;
        let binder = match args.as_ref() {
            Some(args) => binder.with_splice(args),
            None => binder,
        };
        crate::sparse::exec::Exec::new(self.backend.as_ref(), self.opts.entries, binder)
            .run_pass_tapped(pass, self.backend.as_graph(), tap)
    }

    /// The executor one session's launches are issued through.
    ///
    /// Public for the same reason [`SparseStack::binder`] is: a caller checking
    /// what one step does needs the executor the whole pass uses rather than a
    /// second one assembled beside it.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::binder`].
    pub fn executor(&self, session: SessionId) -> CoreResult<crate::sparse::exec::Exec<'_>> {
        self.executor_rows(&[session])
    }

    /// The executor a decode batch's launches are issued through.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::binder_rows`].
    pub fn executor_rows(
        &self,
        sessions: &[SessionId],
    ) -> CoreResult<crate::sparse::exec::Exec<'_>> {
        Ok(crate::sparse::exec::Exec::new(
            self.backend.as_ref(),
            self.opts.entries,
            self.binder_rows(sessions)?,
        ))
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
                ngram,
            } = resident;
            drop(ngram);
            drop(scratch);
            drop(cache);
            drop(kv);
            drop(weights);
        }
        if let Err(error) = self.backend.sync() {
            first = Err(error);
        }
        tracing::info!("sparse stack torn down");
        first
    }
}

impl Drop for SparseStack {
    /// Releases whatever teardown did not, and says so.
    fn drop(&mut self) {
        if self.resident.is_none() {
            return;
        }
        if let Err(error) = self.teardown() {
            tracing::warn!(%error, "teardown from drop could not report cleanly");
        }
    }
}

/// Checks that the imported weights and the geometry agree slot for slot.
///
/// Separate from the constructor because it is a different kind of statement:
/// the constructor allocates, this one refuses. A slot whose kind disagrees
/// would otherwise run one block family's kernels over another's tensors and
/// produce finite numbers.
fn check_layers(shape: &SparseShape, weights: &SparseWeights) -> CoreResult<()> {
    let slots = u64::try_from(weights.layers.len()).unwrap_or(u64::MAX);
    if slots != u64::from(shape.n_layers()) {
        return Err(CoreError::ShapeMismatch {
            field: "layer count",
            expected: u64::from(shape.n_layers()),
            got: slots,
        });
    }
    for (index, layer) in weights.layers.iter().enumerate() {
        let slot = u32::try_from(index).unwrap_or(u32::MAX);
        let want = shape
            .layer_kind(LayerIdx::new(slot))
            .ok_or(CoreError::ShapeMismatch {
                field: "layer kind",
                expected: u64::from(shape.n_layers()),
                got: u64::from(slot),
            })?;
        if want != layer.block.kind() {
            return Err(CoreError::InvalidValue {
                key: "layer kind",
                value: format!(
                    "slot {slot} holds {:?} but the geometry says {want:?}",
                    layer.block.kind()
                ),
            });
        }
        if shape.has_ple(LayerIdx::new(slot)) != layer.ngram.is_some() {
            return Err(CoreError::InvalidValue {
                key: "n-gram slot",
                value: format!("slot {slot} disagrees with the geometry about the n-gram block"),
            });
        }
    }
    Ok(())
}

/// Refuses a batch that names one session twice.
///
/// Two rows of one batch on one session would have two rows of one launch
/// writing that session's recurrent state and its cache rows at once, through
/// the same slot of every per-row array. Nothing fails: the two rows race, one
/// of them wins, and the loser's token is answered out of the winner's state.
///
/// One function rather than one per caller. The planner refuses first and the
/// resolver refuses again, and a rule written down twice is a rule that can be
/// relaxed in one place.
///
/// Speculation is deliberately *not* refused beside it, and it is worth saying
/// why, because the batched kernels take no per-position checkpoint slab and a
/// batched step therefore records nothing. Nothing reads those records in
/// between: a rollback restores the state the pass it is rolling back wrote,
/// and the pass it rolls back is always a speculative verify - several rows of
/// one session, which is a chunk and takes the sequential kernels. A batch and
/// a verify cannot be the same pass.
///
/// # Errors
///
/// [`CoreError::InvalidValue`] naming the session that appears twice.
pub(crate) fn distinct(sessions: &[SessionId]) -> CoreResult<()> {
    for (index, id) in sessions.iter().enumerate() {
        if sessions[..index].contains(id) {
            return Err(CoreError::InvalidValue {
                key: "decode batch",
                value: format!("session {id} appears in more than one row"),
            });
        }
    }
    Ok(())
}

/// Presents a cache failure in the core vocabulary.
fn cache_failure(source: shiso_kvstore::KvError) -> CoreError {
    CoreError::Backend {
        backend: "engine",
        op: "kv cache",
        source: Box::new(source),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pass_width_is_a_whole_number_of_query_blocks_up_to_the_ceiling() {
        for rows in [IDX_QUERY_BLOCK, 2048, 4096, MAX_PREFILL_CHUNK] {
            assert!(check_prefill_rows(rows).is_ok(), "{rows}");
        }
        assert_eq!(SparseOptions::prefill_rows_for(1), IDX_QUERY_BLOCK);
        assert_eq!(SparseOptions::prefill_rows_for(3000), 3072);
        assert_eq!(SparseOptions::prefill_rows_for(2048), 2048);
        assert_eq!(SparseOptions::prefill_rows_for(1 << 20), MAX_PREFILL_CHUNK);
        for rows in [
            0,
            32,
            IDX_QUERY_BLOCK + 1,
            3000,
            MAX_PREFILL_CHUNK + IDX_QUERY_BLOCK,
        ] {
            assert!(check_prefill_rows(rows).is_err(), "{rows}");
        }
    }
}
