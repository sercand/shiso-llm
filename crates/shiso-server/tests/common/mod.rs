//! A device and a model that exist entirely in host memory.
//!
//! Every test in this crate that needs a model needs one that answers a known
//! sequence and counts what was done to it, on a machine with no GPU. The same
//! shape of fake is used one crate down for the ownership contract; it is
//! duplicated rather than shared because a fake exported from a crate becomes
//! part of that crate's API and grows to serve every consumer of it.
//!
//! # Why the fake batches
//!
//! The scheduler's whole claim is that one forward pass serves several
//! sequences, and a fake that answered one row whatever it was asked would make
//! every batching assertion vacuous - the test would pass on a loop that ran the
//! sequences one after another. So `decode_batch` here returns exactly one row
//! per step, in step order, and each session advances its own cursor through the
//! script. A test that asserted on row order would fail if the coordinator ever
//! paired a row with the wrong sequence, which is the bug worth having a fake
//! for.
//!
//! # Why every call is written down
//!
//! Fairness and residency are claims about the *order and the width* of calls,
//! not about their results. [`Audit::calls`] is that order, and it is the only
//! way a test can say "no prompt quantum ran between two of this sequence's
//! decode steps" without reaching inside the coordinator.

#![allow(
    dead_code,
    reason = "one harness serves every integration test here, and no single test uses all of it"
)]

use std::collections::{HashMap, VecDeque};
use std::io;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Condvar, Mutex};
use shiso_core::{
    Backend, CoreError, CoreResult, DecodeStep, DeviceBuffer, DeviceView, KernelRef, LaunchArgs,
    Logits, Model, ModelInfo, RawBuffer, SessionId, StopTokens, TokenId, alloc,
};
use shiso_engine::PersistRefusal;
use shiso_engine::Reason;
use shiso_engine::ckpt::{Checkpointed, CkptConfig, ResumeInfo, ResumeSource, SaveOutcome};
use shiso_server::scheduler::{
    EngineFailure, EngineHandle, SchedulerConfig, spawn_engine, spawn_engine_configured,
    spawn_engine_with_checkpoints,
};
use shiso_server::tokenizer::Vocabulary;

/// A vocabulary small enough that a logits row is a handful of bytes.
const VOCAB: u32 = 64;

/// The id that ends a turn.
///
/// Not exported: a scripted model runs out of script and ends its turn by
/// itself, so no test has to name this, and one that did would be asserting on
/// the fake rather than on the server.
pub(crate) const EOS: u32 = 63;

/// Tokens one fake session may hold.
pub(crate) const CAPACITY: u32 = 256;

/// Rows the fake's logits buffer can hold, which bounds the batch it can answer.
const MAX_ROWS: u32 = 16;

/// A device that is a map of byte vectors.
#[derive(Debug, Default)]
pub(crate) struct FakeBackend {
    allocations: Mutex<HashMap<u64, Vec<u8>>>,
    next_id: AtomicU64,
}

impl FakeBackend {
    /// A device with nothing allocated on it.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn fail(op: &'static str, detail: String) -> CoreError {
        CoreError::Backend {
            backend: "fake",
            op,
            source: Box::new(io::Error::other(detail)),
        }
    }

    fn range_of(
        &self,
        view: DeviceView<'_>,
        len: u64,
        op: &'static str,
    ) -> CoreResult<Range<usize>> {
        let start = usize::try_from(view.offset())
            .map_err(|_| Self::fail(op, "offset too large".to_owned()))?;
        let width =
            usize::try_from(len).map_err(|_| Self::fail(op, "length too large".to_owned()))?;
        let table = self.allocations.lock();
        let buffer = table
            .get(&view.id().get())
            .ok_or_else(|| Self::fail(op, format!("no live allocation {}", view.id())))?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| Self::fail(op, "range overflows".to_owned()))?;
        if end > buffer.len() {
            return Err(CoreError::OutOfBounds {
                offset: view.offset(),
                len,
                buffer_bytes: buffer.len() as u64,
            });
        }
        Ok(start..end)
    }
}

impl Backend for FakeBackend {
    fn backend_name(&self) -> &'static str {
        "fake"
    }

    fn alloc_raw(&self, bytes: u64) -> CoreResult<RawBuffer> {
        let width = usize::try_from(bytes)
            .map_err(|_| Self::fail("alloc_raw", format!("{bytes} bytes do not fit")))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let id = NonZeroU64::new(id)
            .ok_or_else(|| Self::fail("alloc_raw", "id space exhausted".to_owned()))?;
        self.allocations.lock().insert(id.get(), vec![0u8; width]);
        Ok(RawBuffer::new(id, bytes))
    }

    fn free_raw(&self, raw: RawBuffer) -> CoreResult<()> {
        if self.allocations.lock().remove(&raw.id().get()).is_some() {
            return Ok(());
        }
        Err(Self::fail(
            "free_raw",
            format!("{} was already free", raw.id()),
        ))
    }

    fn copy_h2d(&self, dst: DeviceView<'_>, src: &[u8]) -> CoreResult<()> {
        let range = self.range_of(dst, src.len() as u64, "copy_h2d")?;
        let mut table = self.allocations.lock();
        let buffer = table
            .get_mut(&dst.id().get())
            .ok_or_else(|| Self::fail("copy_h2d", "the allocation vanished".to_owned()))?;
        buffer[range].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, src: DeviceView<'_>, dst: &mut [u8]) -> CoreResult<()> {
        let range = self.range_of(src, dst.len() as u64, "copy_d2h")?;
        let table = self.allocations.lock();
        let buffer = table
            .get(&src.id().get())
            .ok_or_else(|| Self::fail("copy_d2h", "the allocation vanished".to_owned()))?;
        dst.copy_from_slice(&buffer[range]);
        Ok(())
    }

    fn copy_d2d(&self, dst: DeviceView<'_>, src: DeviceView<'_>) -> CoreResult<()> {
        let source = self.range_of(src, src.len(), "copy_d2d")?;
        let copied = {
            let table = self.allocations.lock();
            let buffer = table
                .get(&src.id().get())
                .ok_or_else(|| Self::fail("copy_d2d", "the allocation vanished".to_owned()))?;
            buffer[source].to_vec()
        };
        let target = self.range_of(dst, dst.len(), "copy_d2d")?;
        let mut table = self.allocations.lock();
        let buffer = table
            .get_mut(&dst.id().get())
            .ok_or_else(|| Self::fail("copy_d2d", "the allocation vanished".to_owned()))?;
        buffer[target].copy_from_slice(&copied);
        Ok(())
    }

    fn run(&self, _kernel: KernelRef, _args: &LaunchArgs<'_>) -> CoreResult<()> {
        Ok(())
    }

    fn sync(&self) -> CoreResult<()> {
        Ok(())
    }
}

/// One call the model was asked to make.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Call {
    /// A batched decode step over this many sequences.
    Decode(usize),
    /// A prompt quantum of this many tokens.
    Prefill {
        /// Tokens the quantum covered.
        tokens: usize,
        /// Whether it was the chunk that produces logits.
        last: bool,
    },
}

/// One offer to the checkpoint tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SaveNote {
    /// Positions the session had run when it was offered.
    pub(crate) tokens: usize,
    /// Why the coordinator said it was storing it.
    pub(crate) reason: Reason,
}

/// Everything a test can still observe about a model that has been moved onto
/// the engine thread.
#[derive(Debug, Default)]
pub(crate) struct Audit {
    /// Explicit teardowns. An `Arc` of its own so that the older entry point can
    /// hand a test just this counter without the rest.
    pub(crate) teardowns: Arc<AtomicU64>,
    /// Sessions opened.
    pub(crate) opens: AtomicU64,
    /// Sessions closed.
    pub(crate) closes: AtomicU64,
    /// Sessions the model believes are open right now.
    pub(crate) live: Mutex<Vec<SessionId>>,
    /// Every call, in order.
    pub(crate) calls: Mutex<Vec<Call>>,
    /// Frontiers the coordinator offered to the checkpoint tier.
    pub(crate) saves: Mutex<Vec<SaveNote>>,
    /// Resumes the coordinator asked for, by prompt length.
    pub(crate) resumes: Mutex<Vec<usize>>,
    /// Message boundaries the coordinator handed over, per session, in order.
    pub(crate) hints: Mutex<Vec<Vec<u32>>>,
    /// Every prompt chunk the model was asked to read, whole: one entry per
    /// `prefill_chunk`/`prefill_last`, the chunk's ids in order. A scoring
    /// test asserts the exact ids the prompt was read with; a decode step
    /// reads no prompt and adds nothing here.
    pub(crate) prefill_prompts: Mutex<Vec<Vec<u32>>>,
    /// Shutdown sweeps.
    pub(crate) sweeps: AtomicU64,
}

impl Audit {
    /// The widths of every batched decode step, in order.
    pub(crate) fn decode_widths(&self) -> Vec<usize> {
        self.calls
            .lock()
            .iter()
            .filter_map(|call| match call {
                Call::Decode(width) => Some(*width),
                Call::Prefill { .. } => None,
            })
            .collect()
    }

    /// The token counts of every prompt quantum, in order.
    pub(crate) fn prefill_chunks(&self) -> Vec<usize> {
        self.calls
            .lock()
            .iter()
            .filter_map(|call| match call {
                Call::Prefill { tokens, .. } => Some(*tokens),
                Call::Decode(_) => None,
            })
            .collect()
    }

    /// The widest batch any one step covered.
    pub(crate) fn widest_batch(&self) -> usize {
        self.decode_widths().into_iter().max().unwrap_or(0)
    }
}

/// A latch the model waits on, so a test can hold the engine thread still.
///
/// Permits rather than a flag: a test that needs the coordinator to get exactly
/// so far and no further cannot express that with "open" and "closed", and a
/// sleep long enough to stand in for it is the thing that makes a suite flaky on
/// a loaded machine.
#[derive(Debug)]
pub(crate) struct Gate {
    permits: Mutex<u64>,
    ready: Condvar,
}

impl Gate {
    /// A gate nothing may pass until a test says so.
    pub(crate) fn closed() -> Arc<Self> {
        Arc::new(Self {
            permits: Mutex::new(0),
            ready: Condvar::new(),
        })
    }

    /// Lets `count` more calls through.
    pub(crate) fn allow(&self, count: u64) {
        let mut permits = self.permits.lock();
        *permits = permits.saturating_add(count);
        self.ready.notify_all();
    }

    /// Lets everything through from now on.
    pub(crate) fn open(&self) {
        let mut permits = self.permits.lock();
        *permits = u64::MAX;
        self.ready.notify_all();
    }

    /// Blocks until a permit is available and takes one.
    fn pass(&self) {
        let mut permits = self.permits.lock();
        while *permits == 0 {
            self.ready.wait(&mut permits);
        }
        if *permits != u64::MAX {
            *permits -= 1;
        }
    }
}

/// A model that answers a scripted sequence.
///
/// The script is written into the logits row as a spike at the scripted id, so
/// the real sampler picks it: the test then exercises the sampling path rather
/// than bypassing it, and a change that breaks argmax fails here too.
#[derive(Debug)]
pub(crate) struct FakeModel {
    info: ModelInfo,
    backend: Arc<dyn Backend>,
    logits: Option<DeviceBuffer>,
    script: Vec<u32>,
    /// Whole logits rows to answer prefills with, in order: `prefill_last`
    /// pops the front row and writes it verbatim instead of scripting a
    /// spike. A scoring test scripts the exact row whose label log-probabilities
    /// it will assert on; the queue is per-model, matching the scripted-answer
    /// convention that every session reads the same script.
    prefill_rows: VecDeque<Vec<f32>>,
    /// How far through the script each session is. Per session rather than
    /// global, so two sequences in one batch answer independently and a test can
    /// tell "row `i` belongs to `steps[i]`" from "every row is the same".
    cursors: HashMap<SessionId, usize>,
    open: Vec<SessionId>,
    next_session: NonZeroU64,
    audit: Arc<Audit>,
    gate: Option<Arc<Gate>>,
}

impl FakeModel {
    /// A model that will answer `script` to every session, then end its turn.
    pub(crate) fn new(backend: &Arc<dyn Backend>, script: Vec<u32>) -> (Self, Arc<Audit>) {
        let info = ModelInfo {
            context_capacity: CAPACITY,
            vocab: VOCAB,
            hidden: 8,
            n_layers: 2,
            stop_tokens: StopTokens::one(TokenId::new(EOS)),
        };
        let logits = alloc(backend, info.logits_row_bytes() * u64::from(MAX_ROWS))
            .expect("a logits block allocates");
        let audit = Arc::new(Audit::default());
        (
            Self {
                info,
                backend: Arc::clone(backend),
                logits: Some(logits),
                script,
                prefill_rows: VecDeque::new(),
                cursors: HashMap::new(),
                open: Vec::new(),
                next_session: NonZeroU64::MIN,
                audit: Arc::clone(&audit),
                gate: None,
            },
            audit,
        )
    }

    /// The same model, answering the next `rows.len()` prefills with these
    /// whole logits rows, in order, before falling back to the script.
    pub(crate) fn with_prefill_rows(mut self, rows: Vec<Vec<f32>>) -> Self {
        self.prefill_rows = VecDeque::from(rows);
        self
    }

    /// The same model, held still by `gate` on every decode step.
    pub(crate) fn with_gate(mut self, gate: Arc<Gate>) -> Self {
        self.gate = Some(gate);
        self
    }

    /// Writes one session's next scripted token into `row`.
    fn write_row(&mut self, session: SessionId, row: u32) -> CoreResult<()> {
        let at = self.cursors.entry(session).or_insert(0);
        let token = self.script.get(*at).copied().unwrap_or(EOS);
        *at += 1;
        let width = usize::try_from(VOCAB).expect("a small vocabulary fits a host index");
        let mut scores = vec![0f32; width];
        if let Some(slot) = scores.get_mut(usize::try_from(token).unwrap_or(0)) {
            *slot = 1.0;
        }
        let bytes: Vec<u8> = scores.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = self.logits.as_ref().ok_or(CoreError::Backend {
            backend: "fake",
            op: "write_row",
            source: Box::new(io::Error::other("teardown released the logits row")),
        })?;
        let offset = u64::from(row) * self.info.logits_row_bytes();
        self.backend
            .copy_h2d(buffer.slice(offset, bytes.len() as u64)?, &bytes)
    }

    /// Writes `row` verbatim into logits row `row_index`, through the same
    /// host-to-device copy a scripted spike takes.
    fn write_explicit_row(&self, row: &[f32], row_index: u32) -> CoreResult<()> {
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = self.logits.as_ref().ok_or(CoreError::Backend {
            backend: "fake",
            op: "write_explicit_row",
            source: Box::new(io::Error::other("teardown released the logits row")),
        })?;
        let offset = u64::from(row_index) * self.info.logits_row_bytes();
        self.backend
            .copy_h2d(buffer.slice(offset, bytes.len() as u64)?, &bytes)
    }

    /// The first `rows` rows, as a borrowed logits block.
    fn block(&self, rows: u32) -> CoreResult<Logits<'_>> {
        let buffer = self.logits.as_ref().ok_or(CoreError::Backend {
            backend: "fake",
            op: "block",
            source: Box::new(io::Error::other("teardown released the logits row")),
        })?;
        Logits::new(
            buffer.slice(0, self.info.logits_row_bytes() * u64::from(rows))?,
            rows,
            self.info.vocab,
        )
    }
}

impl Model for FakeModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn open_session(&mut self) -> CoreResult<SessionId> {
        let session = SessionId::next(&mut self.next_session).ok_or(CoreError::Backend {
            backend: "fake",
            op: "open_session",
            source: Box::new(io::Error::other("session id space exhausted")),
        })?;
        self.open.push(session);
        self.audit.opens.fetch_add(1, Ordering::Relaxed);
        self.audit.live.lock().push(session);
        Ok(session)
    }

    fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        let Some(at) = self.open.iter().position(|held| *held == session) else {
            return Err(CoreError::UnknownSession(session));
        };
        self.open.remove(at);
        self.cursors.remove(&session);
        self.audit.closes.fetch_add(1, Ordering::Relaxed);
        self.audit.live.lock().retain(|held| *held != session);
        Ok(())
    }

    fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        if !self.open.contains(&session) {
            return Err(CoreError::UnknownSession(session));
        }
        self.audit.calls.lock().push(Call::Prefill {
            tokens: tokens.len(),
            last: false,
        });
        self.audit
            .prefill_prompts
            .lock()
            .push(tokens.iter().map(|t| t.get()).collect());
        Ok(())
    }

    fn prefill_last(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>> {
        if !self.open.contains(&session) {
            return Err(CoreError::UnknownSession(session));
        }
        self.audit.calls.lock().push(Call::Prefill {
            tokens: tokens.len(),
            last: true,
        });
        self.audit
            .prefill_prompts
            .lock()
            .push(tokens.iter().map(|t| t.get()).collect());
        // A scripted row answers verbatim - it is what a scoring test reads
        // back - and only once: the queue drains and the script's spikes take
        // over, so a test that scripted one row for the score still gets
        // scripted answers from every later prefill.
        match self.prefill_rows.pop_front() {
            Some(row) => self.write_explicit_row(&row, 0)?,
            None => self.write_row(session, 0)?,
        }
        self.block(1)
    }

    fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        for step in steps {
            if !self.open.contains(&step.session) {
                return Err(CoreError::UnknownSession(step.session));
            }
        }
        if let Some(gate) = self.gate.clone() {
            gate.pass();
        }
        self.audit.calls.lock().push(Call::Decode(steps.len()));
        for (row, step) in steps.iter().enumerate() {
            let row = u32::try_from(row).unwrap_or(u32::MAX);
            self.write_row(step.session, row)?;
        }
        self.block(u32::try_from(steps.len()).unwrap_or(MAX_ROWS))
    }

    fn teardown(&mut self) -> CoreResult<()> {
        self.audit.teardowns.fetch_add(1, Ordering::Relaxed);
        self.logits = None;
        Ok(())
    }
}

/// The same model with a checkpoint tier that records what it was asked.
///
/// It resumes whatever `hit` says it should - nothing by default - and counts
/// the positions each session has run, which is what a save stores: the
/// session's own history, never a path the caller hands over.
#[derive(Debug)]
pub(crate) struct FakeCheckpointed {
    inner: FakeModel,
    enabled: bool,
    /// Positions each open session has run.
    written: HashMap<SessionId, usize>,
    /// How many prompt tokens a resume reports reusing, when it reports a hit.
    hit: Option<u32>,
}

impl FakeCheckpointed {
    /// A checkpointing model whose tier is already on.
    pub(crate) fn new(backend: &Arc<dyn Backend>, script: Vec<u32>) -> (Self, Arc<Audit>) {
        let (inner, audit) = FakeModel::new(backend, script);
        (
            Self {
                inner,
                enabled: true,
                written: HashMap::new(),
                hit: None,
            },
            audit,
        )
    }

    /// The same model, held still by `gate` on every decode step.
    pub(crate) fn with_gate(mut self, gate: Arc<Gate>) -> Self {
        self.inner = self.inner.with_gate(gate);
        self
    }

    /// The same model, reporting a resume of `tokens` prompt tokens to every
    /// prompt at least that long.
    pub(crate) const fn with_hit(mut self, tokens: u32) -> Self {
        self.hit = Some(tokens);
        self
    }
}

impl Model for FakeCheckpointed {
    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }

    fn open_session(&mut self) -> CoreResult<SessionId> {
        let session = self.inner.open_session()?;
        self.written.insert(session, 0);
        Ok(session)
    }

    fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        self.written.remove(&session);
        self.inner.close_session(session)
    }

    fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        self.inner.prefill_chunk(session, tokens)?;
        *self.written.entry(session).or_insert(0) += tokens.len();
        Ok(())
    }

    fn prefill_last(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>> {
        *self.written.entry(session).or_insert(0) += tokens.len();
        self.inner.prefill_last(session, tokens)
    }

    fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        for step in steps {
            *self.written.entry(step.session).or_insert(0) += 1;
        }
        self.inner.decode_batch(steps)
    }

    fn teardown(&mut self) -> CoreResult<()> {
        self.inner.teardown()
    }
}

impl Checkpointed for FakeCheckpointed {
    fn kv_enable(&mut self, _config: &CkptConfig) -> CoreResult<()> {
        self.enabled = true;
        Ok(())
    }

    fn kv_checkpoint_enabled(&self) -> bool {
        self.enabled
    }

    fn kv_resume(
        &mut self,
        session: SessionId,
        prompt: &[TokenId],
    ) -> CoreResult<Option<ResumeInfo<'_>>> {
        self.inner.audit.resumes.lock().push(prompt.len());
        let Some(hit) = self
            .hit
            .filter(|hit| usize::try_from(*hit).is_ok_and(|n| n <= prompt.len()))
        else {
            return Ok(None);
        };
        let reused = usize::try_from(hit).unwrap_or(prompt.len());
        self.written.insert(session, reused);
        // A resume of the whole prompt stands on the row its next token is
        // drawn from; the fake writes one, as the real tier restores one.
        let logits = if reused == prompt.len() {
            self.inner.write_row(session, 0)?;
            Some(self.inner.block(1)?)
        } else {
            None
        };
        Ok(Some(ResumeInfo::new(
            hit,
            prompt[..reused].to_vec(),
            logits,
            ResumeSource::Memory,
        )))
    }

    fn kv_save(&mut self, session: SessionId, reason: Reason) -> CoreResult<SaveOutcome> {
        if !self.inner.open.contains(&session) {
            return Err(CoreError::UnknownSession(session));
        }
        self.inner.audit.saves.lock().push(SaveNote {
            tokens: self.written.get(&session).copied().unwrap_or(0),
            reason,
        });
        // Refused rather than stored, because a fake that claimed to have
        // stored a file would let a test assert on a store this harness does
        // not have. What is being pinned is the offer and its reason.
        Ok(SaveOutcome::Refused(PersistRefusal::NoCheckpoint))
    }

    fn kv_set_hints(&mut self, _session: SessionId, hints: &[u32]) -> CoreResult<()> {
        self.inner.audit.hints.lock().push(hints.to_vec());
        Ok(())
    }

    fn kv_shutdown_sweep(&mut self) -> usize {
        self.inner.audit.sweeps.fetch_add(1, Ordering::Relaxed);
        0
    }
}

/// Starts an engine thread over a model that answers `script`.
///
/// The third value counts teardowns, and is the only thing a test can still
/// observe about the model once it has been moved onto the engine thread. It is
/// a bare counter rather than a named audit type because a named type with one
/// field is a type every test that ignores the counter has to be told to
/// ignore.
///
/// `vocabulary` is what turns the scripted ids into text. `None` is what the
/// boundary tests want - ids reach the caller and the text is empty - and a
/// vocabulary is what a wire test needs, because the thing an adapter encodes
/// is the text.
pub(crate) fn fake_engine(
    script: Vec<u32>,
    vocabulary: Option<Arc<Vocabulary>>,
) -> (
    EngineHandle,
    std::thread::JoinHandle<Result<(), EngineFailure>>,
    Arc<AtomicU64>,
) {
    let backend: Arc<dyn Backend> = FakeBackend::new();
    let (model, audit) = FakeModel::new(&backend, script);
    let teardowns = Arc::clone(&audit.teardowns);
    let (engine, join) = spawn_engine(backend, Box::new(model), vocabulary, "fake-model", CAPACITY);
    (engine, join, teardowns)
}

/// Starts an engine thread over a model that answers prefills with these whole
/// logits rows, in order, and an empty script after them: a scoring pass reads
/// its scripted row, and anything that generates afterwards ends its turn on
/// the script's exhaustion. The whole audit, rather than just the teardown
/// count, because a scoring test asserts on the prompts the model read.
pub(crate) fn fake_engine_rows(
    rows: Vec<Vec<f32>>,
    vocabulary: Option<Arc<Vocabulary>>,
) -> (
    EngineHandle,
    std::thread::JoinHandle<Result<(), EngineFailure>>,
    Arc<Audit>,
) {
    let backend: Arc<dyn Backend> = FakeBackend::new();
    let (model, audit) = FakeModel::new(&backend, Vec::new());
    let model = model.with_prefill_rows(rows);
    let (engine, join) = spawn_engine(backend, Box::new(model), vocabulary, "fake-model", CAPACITY);
    (engine, join, audit)
}

/// Starts an engine thread with a chosen configuration and the whole audit.
pub(crate) fn scheduled_engine(
    script: Vec<u32>,
    cfg: SchedulerConfig,
    gate: Option<Arc<Gate>>,
) -> (
    EngineHandle,
    std::thread::JoinHandle<Result<(), EngineFailure>>,
    Arc<Audit>,
) {
    let backend: Arc<dyn Backend> = FakeBackend::new();
    let (model, audit) = FakeModel::new(&backend, script);
    let model = match gate {
        Some(gate) => model.with_gate(gate),
        None => model,
    };
    let (engine, join) =
        spawn_engine_configured(backend, Box::new(model), None, "fake-model", CAPACITY, cfg);
    (engine, join, audit)
}

/// Starts an engine thread over a model whose checkpoint tier is on.
pub(crate) fn checkpointed_engine(
    script: Vec<u32>,
    cfg: SchedulerConfig,
    gate: Option<Arc<Gate>>,
) -> (
    EngineHandle,
    std::thread::JoinHandle<Result<(), EngineFailure>>,
    Arc<Audit>,
) {
    checkpointed_engine_hitting(script, cfg, gate, None)
}

/// As [`checkpointed_engine`], with the tier reporting a resume of `hit`
/// prompt tokens to every prompt at least that long.
pub(crate) fn checkpointed_engine_hitting(
    script: Vec<u32>,
    cfg: SchedulerConfig,
    gate: Option<Arc<Gate>>,
    hit: Option<u32>,
) -> (
    EngineHandle,
    std::thread::JoinHandle<Result<(), EngineFailure>>,
    Arc<Audit>,
) {
    let backend: Arc<dyn Backend> = FakeBackend::new();
    let (model, audit) = FakeCheckpointed::new(&backend, script);
    let model = match gate {
        Some(gate) => model.with_gate(gate),
        None => model,
    };
    let model = match hit {
        Some(tokens) => model.with_hit(tokens),
        None => model,
    };
    let (engine, join) = spawn_engine_with_checkpoints(
        backend,
        Box::new(model),
        shiso_server::media::MediaTowers::default(),
        None,
        "fake-model",
        CAPACITY,
        cfg,
        (
            shiso_parsers::ReasoningScheme::None,
            shiso_parsers::ToolCallScheme::None,
        ),
    );
    (engine, join, audit)
}

/// Starts an engine thread over a scripted model with the parsers named.
///
/// The checkpointing fake rather than the plain one, because that is the shape
/// the factory hands a real server and the only [`spawn_engine`] entry that
/// takes a scheme pair at all.
pub(crate) fn parsed_engine(
    script: Vec<u32>,
    vocabulary: Option<Arc<Vocabulary>>,
    parsers: (
        shiso_parsers::ReasoningScheme,
        shiso_parsers::ToolCallScheme,
    ),
) -> (
    EngineHandle,
    std::thread::JoinHandle<Result<(), EngineFailure>>,
) {
    let backend: Arc<dyn Backend> = FakeBackend::new();
    let (model, _audit) = FakeCheckpointed::new(&backend, script);
    spawn_engine_with_checkpoints(
        backend,
        Box::new(model),
        shiso_server::media::MediaTowers::default(),
        vocabulary,
        "fake-model",
        CAPACITY,
        SchedulerConfig::new(),
        parsers,
    )
}

/// Feeds the gate until `want` sessions are resident, or gives up.
///
/// A progress loop rather than a sleep long enough to "probably" be enough: each
/// permit lets the coordinator finish one decode step and go round its loop
/// once, which is its only chance to admit whatever is waiting. The test
/// therefore drives the coordinator forward one step at a time until it has
/// admitted what the test needs, and never asserts on how fast a machine ran.
pub(crate) async fn wait_for_live(audit: &Audit, gate: &Gate, want: usize) {
    for _ in 0..2_000 {
        if audit.live.lock().len() >= want {
            return;
        }
        gate.allow(1);
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    panic!(
        "only {} of {want} sequences became resident",
        audit.live.lock().len()
    );
}

/// Waits until the model has been asked to do `want` things.
///
/// A session exists from the moment it is opened, which is before the prompt
/// has been read and before the stream's opening frame has been sent - so a test
/// that acted on residency alone could act in the window where the turn has not
/// started yet, and would be testing a different case on the runs where it lost
/// the race.
pub(crate) async fn wait_for_calls(audit: &Audit, want: usize) {
    for _ in 0..2_000 {
        if audit.calls.lock().len() >= want {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    panic!(
        "only {} of {want} calls were made",
        audit.calls.lock().len()
    );
}

/// Waits until the queue has seen `want` requests.
///
/// What "the coordinator is busy and these are waiting behind it" looks like
/// from outside: the counter moves when a request enters the queue, which is
/// before anything can admit it.
pub(crate) async fn wait_for_queued(engine: &EngineHandle, want: u64) {
    for _ in 0..2_000 {
        if engine.stats().requests() >= want {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    panic!(
        "only {} of {want} requests reached the queue",
        engine.stats().requests()
    );
}
