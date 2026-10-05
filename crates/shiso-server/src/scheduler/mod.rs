//! The one thread that owns the model, and the loop it runs.
//!
//! # Why an OS thread and not a task
//!
//! A decode step is a blocking device call that takes tens to hundreds of
//! milliseconds and does not yield. Run on a runtime worker it starves every
//! other connection on that worker, and the usual escape hatch - a blocking
//! pool - does not apply here, because the model is not `Sync` and must stay on
//! one thread for its whole life. So the model gets a dedicated
//! [`std::thread`], and the queue is read with `blocking_recv`, which is the
//! documented way to read an async channel from a thread that is not a runtime
//! worker.
//!
//! # Fail fast, before the listener binds
//!
//! The caller loads the model on its own thread and only then calls
//! [`spawn_engine`]. There is no readiness handshake and no "loading" state to
//! report, because nothing is serving yet when the load happens: a checkpoint
//! that cannot be loaded ends the process with a non-zero status having never
//! accepted a connection. `Model: Send` is what makes moving the loaded model
//! onto this thread legal.
//!
//! # What one tick does
//!
//! Admit whatever fits, cut whatever outlived its deadline, run one batched
//! decode step over every sequence that is ready, read one bounded quantum of
//! one sequence's prompt, send what that produced, and retire whatever stopped.
//! The order is the fairness rule: an answer already streaming gets its token
//! before a newly arrived prompt is read, and a prompt is read in pieces small
//! enough that no answer waits more than one piece for its next token.
//!
//! # Why the channel is the wait primitive
//!
//! When nothing is resident and nothing is queued, the loop blocks on the
//! channel. It does not poll, and it does not hold a second queue behind a
//! condition variable: the channel already *is* a bounded queue with a blocking
//! read, and a second one in front of it would only add a place where the
//! bound - and therefore the point at which a caller is told the server is
//! overloaded - could be enforced twice or not at all.
//!
//! # Flex requests wait for standard work to drain
//!
//! A flex request is served only while no standard request has work. One that
//! arrives while standard work is resident is *held*: it gets its stream at
//! once, so a streaming caller sees its headers and the keepalives that follow,
//! but no session, until the standard work has drained. Held requests are
//! admitted oldest first, and a standard request that comes off the queue
//! behind one is admitted straight past it. One that is already resident is
//! paused by the planner (see `batching.rs`) rather than evicted, and resumes
//! where it stood.
//!
//! Flex sequences never take the last slot while there is more than one: a
//! standard request arriving at a table full of flex answers would otherwise
//! wait for one of them to finish, which is the one thing a standard caller
//! was promised it would not do.
//!
//! # Who closes a session
//!
//! This thread, and nothing else. A session is opened at admission and closed
//! exactly once on the retirement path below, whatever ended the sequence: a
//! finished turn, a deadline, a client that hung up, a device that failed. A
//! client that could cause a session to be released would be a client that can
//! free device state in the middle of somebody else's forward pass.

mod batching;
mod cancel;
pub mod clock;
mod config;
mod lifecycle;
#[cfg(test)]
mod media_key_tests;
pub mod queue;
mod report;
mod score_pass;
mod stats;
pub mod stop;

use std::collections::VecDeque;
use std::io;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use shiso_core::{
    Backend, CoreError, DecodeStep, MediaDigest, MediaKey, MediaSpan, Model, PadTokens, SessionId,
    StopTokens, TokenId, VisionGrid, VisionPrefill,
};
use shiso_engine::Reason;
use shiso_engine::ckpt::Checkpointed;
use shiso_ir::{EventStream, StopReason, StreamEvent};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

use crate::decision;
use crate::error::AppError;
use crate::media::{MediaItem, MediaTowers};
use crate::media_cache::{CachedRows, MediaCache};
use crate::tokenizer::Vocabulary;
use crate::vision::{EncodedMedia, VisionTower};
use shiso_audio::AudioTower;
use shiso_parsers::{ReasoningScheme, ToolCallScheme};

use batching::{PrefillStep, SlotTable};
use cancel::{Delivery, send_all, send_event};
use lifecycle::{Ending, SeqSlot, SlotSetup};

pub use clock::{Clock, FakeClock, SystemClock};
pub use config::{
    DEFAULT_BOUNDARY_ALIGN, DEFAULT_BOUNDARY_TRIM, DEFAULT_CONTINUED_INTERVAL,
    DEFAULT_DRAIN_TIMEOUT, DEFAULT_MAX_NUM_SEQS, DEFAULT_MAX_QUEUED, DEFAULT_MIXED_QUANTUM,
    DEFAULT_PUMP_BYTES, DEFAULT_QUEUE_ADMISSION_MS, DEFAULT_SOLO_CHUNK, PREFILL_GRID,
    SchedulerConfig, StorePolicy,
};
pub use queue::{
    EngineGone, EngineHandle, EngineRequest, GenerateRequest, QUEUE_CAPACITY, ScoreOutcome,
    ScoreRequest,
};
pub use stats::{SchedulerStats, Snapshot, Window};
pub use stop::{StopStep, StopWatcher};

/// Sequences one decode step carries, which is the widest batch the engines'
/// logits block and per-row pointer arrays are built for.
const DECODE_GROUP_ROWS: usize = shiso_engine::MAX_DECODE_BATCH as usize;

/// Events buffered for a client that is not reading fast enough.
const EVENT_QUEUE: usize = 64;

/// How much more of the queued checkpoint pages one pump call may copy when no
/// sequence is resident: nothing is waiting on the device.
const IDLE_PUMP_FACTOR: u64 = 8;

/// How long an idle pump waits for the writer to free staging before it
/// polls the request queue and tries again.
const IDLE_PUMP_WAIT: std::time::Duration = std::time::Duration::from_millis(5);

/// The loop could not finish cleanly.
///
/// The join handle yields this rather than `()` so that a coordinator which
/// died on a device error does not look identical to one that shut down because
/// its queue closed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EngineFailure {
    /// The model reported an error while releasing its device resources.
    #[error("the model could not be torn down")]
    Teardown(#[source] CoreError),
}

/// The model, and whether it has a disk tier.
///
/// Two variants because not every model can checkpoint, and the one that can
/// must still be usable through the plain contract - every forward pass goes
/// through [`Model`] whichever arm this is. The checkpointing methods are
/// reachable only through [`EngineModel::ckpt_mut`], which answers `None` for a
/// model that does not have them *and* for one whose tier was never opened, so
/// a scheduler cannot ask a question the model has no answer to.
#[derive(Debug)]
enum EngineModel {
    /// A model with no disk tier. What a harness drives, and what a family that
    /// never gains one would be served through.
    Plain(Box<dyn Model>),
    /// A model that can store and resume a prefix.
    Checkpointed(Box<dyn Checkpointed>),
}

impl EngineModel {
    /// The forward pass, which is the same contract either way.
    fn as_model_mut(&mut self) -> &mut dyn Model {
        match self {
            Self::Plain(model) => model.as_mut(),
            Self::Checkpointed(model) => model.as_mut(),
        }
    }

    /// The disk tier, when this model has one that is switched on.
    fn ckpt_mut(&mut self) -> Option<&mut dyn Checkpointed> {
        match self {
            Self::Plain(_) => None,
            Self::Checkpointed(model) => {
                if model.kv_checkpoint_enabled() {
                    Some(model.as_mut())
                } else {
                    None
                }
            }
        }
    }

    /// What the cache underneath holds, tier on or off; `None` for a model
    /// that keeps no paged cache.
    fn kv_stats(&self) -> Option<shiso_kvstore::CacheStats> {
        match self {
            Self::Plain(_) => None,
            Self::Checkpointed(model) => model.kv_cache_stats(),
        }
    }
}

/// One row of a batched decode step, bound to the sequence it belongs to.
///
/// A struct per row rather than two lists that have to stay the same length:
/// "row `i` of the result is `steps[i]`" is the contract a batching bug would
/// break by silently handing one caller another caller's token.
#[derive(Debug, Clone, Copy)]
struct BatchRow {
    slot: usize,
    step: DecodeStep,
}

/// A flex request that has its stream and is waiting for a session.
///
/// The reply has already been answered, so what is held is the request and the
/// end of the stream its events will go down. A sink the caller has dropped is
/// a request nobody is waiting for any more, and it is let go without ever
/// opening a session.
struct HeldFlex {
    request: GenerateRequest,
    sink: mpsc::Sender<StreamEvent>,
}

/// Everything the engine thread owns.
struct Coordinator {
    backend: Arc<dyn Backend>,
    model: EngineModel,
    /// The vision tower, when the operator started this server with one.
    ///
    /// Owned by this thread because what it produces is device memory this
    /// thread is about to splice into a forward pass. `None` is a text-only
    /// server: no tower was named, none was loaded, and a request that carries
    /// a picture is refused rather than answered from its words alone.
    vision: Option<VisionTower>,
    /// The audio tower, when the tower container carries one. Shared with the
    /// renderer, which reads its frontend and token rule; only this thread
    /// runs it.
    audio: Option<Arc<dyn AudioTower>>,
    /// The towers' output by content digest, shared with the renderer, which
    /// reads it to skip decoding a picture whose rows are resident. `None`
    /// encodes every item every time.
    media_cache: Option<Arc<MediaCache>>,
    vocabulary: Option<Arc<Vocabulary>>,
    /// The placeholder ids a prompt's media runs are written in, resolved once
    /// at boot from the tower's placement and this checkpoint's vocabulary.
    ///
    /// Resolved here rather than per prefill because it is the same answer
    /// every time, and resolved from the tower rather than from a constant
    /// because the marker is the family's and the id is the vocabulary's. A
    /// server with no tower has no media to splice, so the default - neither id
    /// - is the honest value rather than a placeholder.
    pads: PadTokens,
    model_id: Arc<str>,
    ctx: u32,
    stop: StopTokens,
    cfg: SchedulerConfig,
    stats: Arc<SchedulerStats>,
    parsers: (ReasoningScheme, ToolCallScheme),
    slots: SlotTable,
    /// One logits row, read back from the device and reused every step.
    row: Vec<u8>,
    /// The same row as floats, reused for the same reason.
    scores: Vec<f32>,
    next_id: u64,
    /// Decode steps run so far: a prompt whose prefill saw none of them ran on
    /// the device alone, and its wall time is a fair measure of prefill speed.
    decode_steps: u64,
    /// Whether the queue has closed and the loop is draining.
    closing: bool,
    /// Whether every sequence draws by the keyed rule: true on a server whose
    /// model speculates, so sampled sequences can verify drafts by equality.
    keyed: bool,
    /// Flex requests waiting for standard work to drain, oldest first.
    ///
    /// Unbounded, by decision: a flex caller asked to wait rather than to be
    /// refused, and each entry is a prompt behind an `Arc` and a channel. The
    /// bound on how much can be *queued* is still the channel's; this is only
    /// what has come off it.
    held_flex: VecDeque<HeldFlex>,
    /// The periodic info line.
    report: report::StatsLog,
    /// What the batched prompt pass has shown so far; see [`score_pass`].
    score_pass: score_pass::ScorePass,
}

/// Starts the engine thread over a loaded model.
///
/// The returned [`JoinHandle`] must be kept and joined after the listener
/// stops: teardown runs when the queue closes, and a process that exits without
/// joining races its own teardown.
///
/// `vocabulary` is what turns generated ids into the text a stream carries.
/// `None` is legal and produces a stream of ids with empty text, which is what
/// a harness driving a scripted model wants and is strictly more than a failed
/// request would be.
///
/// # Panics
///
/// When the operating system refuses to start a thread. There is no useful
/// recovery: the process was started to serve a model and cannot, and a caller
/// handed an error here would have nothing to do with it but exit.
pub fn spawn_engine(
    backend: Arc<dyn Backend>,
    model: Box<dyn Model>,
    vocabulary: Option<Arc<Vocabulary>>,
    model_id: &str,
    ctx: u32,
) -> (EngineHandle, JoinHandle<Result<(), EngineFailure>>) {
    spawn_engine_configured(
        backend,
        model,
        vocabulary,
        model_id,
        ctx,
        SchedulerConfig::new(),
    )
}

/// Starts the engine thread over a plain model with a chosen configuration.
///
/// # Panics
///
/// As [`spawn_engine`].
pub fn spawn_engine_configured(
    backend: Arc<dyn Backend>,
    model: Box<dyn Model>,
    vocabulary: Option<Arc<Vocabulary>>,
    model_id: &str,
    ctx: u32,
    cfg: SchedulerConfig,
) -> (EngineHandle, JoinHandle<Result<(), EngineFailure>>) {
    start(
        backend,
        EngineModel::Plain(model),
        MediaTowers::default(),
        vocabulary,
        model_id,
        ctx,
        cfg,
        (ReasoningScheme::None, ToolCallScheme::None),
    )
}

/// Starts the engine thread over a model that can checkpoint.
///
/// `vision` is the tower this server was started with, or `None` for a text-only
/// server. It is taken here rather than installed later because the thread that
/// runs it is the thread this call creates, and a tower handed over afterwards
/// would be a second initialisation step with a window in between where a
/// request carrying a picture would be refused by a server that has one.
///
/// Whether anything is actually stored is the model's own answer: a checkpoint
/// tier that was never opened reports itself as off, and the coordinator then
/// never asks it anything. That is why there is no second flag here - a
/// scheduler told "checkpointing is on" by its caller and "off" by the model
/// would have two answers to one question.
///
/// # Panics
///
/// As [`spawn_engine`].
#[expect(
    clippy::too_many_arguments,
    reason = "the scheduler takes the device, the model, the media towers, the vocabulary, the identity, the context size, the runtime config and the stream parsers"
)]
pub fn spawn_engine_with_checkpoints(
    backend: Arc<dyn Backend>,
    model: Box<dyn Checkpointed>,
    towers: MediaTowers,
    vocabulary: Option<Arc<Vocabulary>>,
    model_id: &str,
    ctx: u32,
    cfg: SchedulerConfig,
    parsers: (ReasoningScheme, ToolCallScheme),
) -> (EngineHandle, JoinHandle<Result<(), EngineFailure>>) {
    start(
        backend,
        EngineModel::Checkpointed(model),
        towers,
        vocabulary,
        model_id,
        ctx,
        cfg,
        parsers,
    )
}

/// Builds the queue, the counters and the thread.
#[expect(
    clippy::too_many_arguments,
    reason = "the scheduler takes the device, the model, the media towers, the vocabulary, the identity, the context size, the runtime config and the stream parsers"
)]
fn start(
    backend: Arc<dyn Backend>,
    mut model: EngineModel,
    towers: MediaTowers,
    vocabulary: Option<Arc<Vocabulary>>,
    model_id: &str,
    ctx: u32,
    cfg: SchedulerConfig,
    parsers: (ReasoningScheme, ToolCallScheme),
) -> (EngineHandle, JoinHandle<Result<(), EngineFailure>>) {
    let model_id: Arc<str> = Arc::from(model_id);
    let stats = Arc::new(SchedulerStats::new());
    let (handle, rx) = queue::channel(Arc::clone(&model_id), ctx, &cfg, Arc::clone(&stats));
    let info = *model.as_model_mut().info();
    let vocab = usize::try_from(info.vocab).unwrap_or(usize::MAX);
    let max_num_seqs = cfg.max_num_seqs;
    let MediaTowers {
        vision,
        audio,
        cache: media_cache,
    } = towers;
    let pads = match (vision.as_ref(), vocabulary.as_ref()) {
        (Some(tower), Some(vocabulary)) => vocabulary.media_ids(&tower.placement()),
        _ => PadTokens::default(),
    };
    let keyed = model.as_model_mut().spec_mut().is_some();
    let report = report::StatsLog::new(cfg.stats_interval, &stats, cfg.clock.now());
    let coordinator = Coordinator {
        backend,
        model,
        vision,
        audio,
        media_cache,
        vocabulary,
        pads,
        model_id,
        ctx,
        stop: info.stop_tokens,
        cfg,
        parsers,
        stats,
        slots: SlotTable::new(max_num_seqs),
        row: vec![0u8; vocab.saturating_mul(4)],
        scores: vec![0f32; vocab],
        next_id: 1,
        decode_steps: 0,
        keyed,
        closing: false,
        held_flex: VecDeque::new(),
        report,
        score_pass: score_pass::ScorePass::default(),
    };
    let join = std::thread::Builder::new()
        .name("shiso-engine".to_owned())
        .spawn(move || coordinator.run(rx))
        .expect("the engine thread starts");
    (handle, join)
}

impl Coordinator {
    /// The loop.
    fn run(mut self, mut rx: mpsc::Receiver<EngineRequest>) -> Result<(), EngineFailure> {
        tracing::info!(
            max_num_seqs = self.cfg.max_num_seqs,
            max_queued = self.cfg.max_queued,
            mixed_quantum = self.cfg.mixed_quantum,
            solo_chunk = self.cfg.solo_chunk,
            checkpointing = self.model.ckpt_mut().is_some(),
            "the scheduler is running"
        );

        loop {
            self.admit_waiting(&mut rx);
            // After the queue, so a standard request that was queued behind a
            // held flex one is resident before the flex one is considered. An
            // empty table always takes a held request, so a table still empty
            // here has nothing held either and the blocking read below cannot
            // strand one.
            self.admit_held_flex();
            if self.slots.is_empty() {
                if self.closing {
                    break;
                }
                // Checkpoint pages still queued for disk are copied while
                // nothing else wants the device, a staging budget at a time,
                // polling the queue between rounds so a request that arrives
                // meanwhile is not kept waiting.
                if self.pump_idle(&mut rx) {
                    continue;
                }
                // The line for the work that just ended, before the wait that
                // may last hours: otherwise its rates would be averaged over
                // the idle time too, whenever the next request woke the loop.
                self.log_stats(rx.len());
                // Nothing to run and nothing waiting: block until something
                // arrives. An idle server burns no CPU, which a poll would not
                // manage and which matters because this thread is pinned for the
                // life of the process.
                match rx.blocking_recv() {
                    Some(request) => self.admit(request),
                    None => self.closing = true,
                }
                continue;
            }
            self.stats.set_pending(rx.len() + self.held_flex.len());
            self.stats.set_active(self.slots.len());
            self.tick();
            if self.report.due(self.cfg.clock.now()) {
                self.log_stats(rx.len());
            }
        }

        self.shutdown();
        tracing::info!("the engine queue closed; tearing down the model");
        self.model
            .as_model_mut()
            .teardown()
            .map_err(EngineFailure::Teardown)?;
        tracing::info!("shutdown complete");
        Ok(())
    }

    /// Prints the periodic line for the interval that ends now.
    fn log_stats(&mut self, queued: usize) {
        let gauges = report::Gauges {
            running: self.slots.len(),
            waiting: queued + self.held_flex.len(),
            kv: self.model.kv_stats(),
        };
        let now = self.cfg.clock.now();
        self.report.emit(&self.stats, now, &gauges);
    }

    /// Admits as many waiting requests as there are free slots.
    fn admit_waiting(&mut self, rx: &mut mpsc::Receiver<EngineRequest>) {
        while self.slots.has_room() && !self.closing {
            match rx.try_recv() {
                Ok(request) => self.admit(request),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.closing = true;
                    break;
                }
            }
        }
    }

    /// Gives one request a stream, and then a session and a slot - or, for a
    /// flex request that cannot run yet, a place in the held queue. A scoring
    /// request takes neither path: it gets a session, a slot and its answer.
    fn admit(&mut self, request: EngineRequest) {
        match request {
            EngineRequest::Generate { request, reply } => {
                let Some(sink) = Self::reply_stream(&request, reply) else {
                    return;
                };
                if request.is_flex() && !(self.held_flex.is_empty() && self.flex_fits()) {
                    tracing::debug!(
                        held = self.held_flex.len() + 1,
                        "a flex request is held until standard work drains"
                    );
                    self.held_flex.push_back(HeldFlex { request, sink });
                    return;
                }
                self.open(request, sink);
            }
            EngineRequest::Score { request, reply } => self.open_score(request, reply),
        }
    }

    /// Whether a flex request could take a slot now: there is room, no
    /// standard sequence is resident, and flex sequences would still leave one
    /// slot free.
    fn flex_fits(&self) -> bool {
        let resident = self.slots.slots();
        let standard = resident
            .iter()
            .any(|slot| !slot.is_flex() && !slot.is_done());
        let flex = resident.iter().filter(|slot| slot.is_flex()).count();
        let cap = self.cfg.max_num_seqs.saturating_sub(1).max(1);
        self.slots.has_room() && !standard && flex < cap
    }

    /// Admits held flex requests, oldest first, for as long as one fits.
    ///
    /// A held request whose caller has gone is dropped first, wherever it is in
    /// the queue, so it neither opens a session nor holds up the ones behind it.
    fn admit_held_flex(&mut self) {
        self.held_flex.retain(|held| {
            let waiting = !held.sink.is_closed();
            if !waiting {
                tracing::debug!("a held flex request was abandoned before it started");
            }
            waiting
        });
        while !self.held_flex.is_empty() && self.flex_fits() {
            let Some(held) = self.held_flex.pop_front() else {
                break;
            };
            tracing::debug!(
                still_held = self.held_flex.len(),
                "a held flex request is admitted"
            );
            self.open(held.request, held.sink);
        }
    }

    /// Answers a request's reply with its stream, and returns the end its
    /// events go down - or `None` when there is nothing to serve: an empty
    /// prompt, which is told so on its stream, or a caller already gone.
    fn reply_stream(
        request: &GenerateRequest,
        reply: oneshot::Sender<Result<EventStream, AppError>>,
    ) -> Option<mpsc::Sender<StreamEvent>> {
        let (sink, events) = mpsc::channel(EVENT_QUEUE);
        let stream: EventStream = Box::pin(ReceiverStream::new(events));

        if request.prompt_tokens.is_empty() {
            // Nothing to run a forward pass over. Reported as a terminal
            // failure on the stream rather than as a refusal of the queue,
            // because the queue accepted it and a caller holding a stream that
            // silently ends has no way to tell why.
            if reply.send(Ok(stream)).is_ok() {
                send_event(
                    &sink,
                    StreamEvent::Failed {
                        message: "the rendered prompt is empty".to_owned(),
                    },
                );
            }
            return None;
        }
        if reply.send(Ok(stream)).is_err() {
            // The caller gave up between queueing and admission. Nothing has
            // been generated and no session has been opened, so there is
            // nothing to unwind.
            tracing::debug!("a queued request was abandoned before it started");
            return None;
        }
        Some(sink)
    }

    /// A session for a new sequence, or `None` after telling its client why
    /// none could be opened.
    fn open_session_for(&mut self, sink: &mpsc::Sender<StreamEvent>) -> Option<SessionId> {
        match self.model.as_model_mut().open_session() {
            Ok(session) => Some(session),
            Err(error) => {
                tracing::warn!(error = %error.chain(), "a session could not be opened");
                send_event(
                    sink,
                    StreamEvent::Failed {
                        message: error.to_string(),
                    },
                );
                None
            }
        }
    }

    /// The declared tools a sequence of `request` proposes a call's fixed text
    /// from, or `None` when it declared none.
    ///
    /// The scaffold spells the Qwen 3.5 XML body; under any other tool scheme
    /// it would propose text the model never writes.
    fn scaffold_for(&self, request: &GenerateRequest) -> Option<Arc<shiso_parsers::ToolScaffold>> {
        (self.parsers.1 == ToolCallScheme::Qwen35)
            .then(|| Arc::clone(&request.tool_scaffold))
            .filter(|scaffold| !scaffold.is_empty())
    }

    /// Gives a request whose stream is answered a session and a slot.
    fn open(&mut self, request: GenerateRequest, sink: mpsc::Sender<StreamEvent>) {
        // What a server without the right tower would refuse is refused before
        // a session is opened, because it is cheaper to refuse with nothing
        // standing on it. A request whose pictures cannot be encoded is FAILED
        // rather than served from its words: the prompt still carries the
        // placeholder run, so serving it would run the model over a few hundred
        // embedding rows that stand for nothing and hand back fluent prose
        // about a picture nobody looked at.
        if let Some(message) = self.media_refusal(&request.media) {
            tracing::warn!(%message, "a request carrying media could not be served");
            send_event(&sink, StreamEvent::Failed { message });
            return;
        }
        let key = self.media_key(&request);
        let items = Arc::clone(&request.media);
        let key_digests = Arc::clone(&request.media_digests);

        let Some(session) = self.open_session_for(&sink) else {
            return;
        };
        self.stats.admit();

        let now = self.cfg.clock.now();
        let id = format!("{:016x}", self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs());
        if send_event(
            &sink,
            StreamEvent::Start {
                id: id.clone(),
                model: self.model_id.to_string(),
                created,
            },
        ) == Delivery::Closed
        {
            self.release(session);
            return;
        }

        let budget = self.budget_for(request.prompt_tokens.len(), request.max_tokens);
        // Cloned off the request before it is moved into the slot; it is an
        // `Arc<str>`, so the clone is a refcount and not the prompt.
        let prompt_text = Arc::clone(&request.prompt_text);
        let string_params = Arc::clone(&request.string_params);
        let scaffold = self.scaffold_for(&request);
        let mut slot = SeqSlot::new(
            request,
            SlotSetup {
                id,
                session,
                sink,
                started_at: now,
                deadline: self.cfg.request_timeout.map(|window| now + window),
                budget,
                keyed: self.keyed,
                suppressed: Arc::clone(&self.cfg.suppressed),
                score_reply: None,
            },
            None,
            // The rendered prompt and not just the schemes: a template can
            // spend the reasoning opener itself - Qwen 3.5's closes the
            // generation prompt with `<think>\n` - and then the turn begins
            // inside the trace with no opener left for the model to emit. The
            // declared string parameters ride along so a call's values are
            // read with the types the caller declared.
            shiso_parsers::parser_for_prompt_with(
                self.parsers.0,
                self.parsers.1,
                &prompt_text,
                string_params,
            ),
        );
        slot.set_scaffold(scaffold);
        if let Some((path, spans)) = &key {
            self.key_session(&mut slot, path, spans);
        }
        self.try_resume(&mut slot);
        // The tower runs after the resume, over only the items the resume did
        // not cover: a conversation re-sends its pictures every turn, and the
        // ones a stored prefix already read would be encoded for nothing. A
        // resume never ends inside a picture, so an item is covered whole or
        // not at all.
        if !items.is_empty() && !slot.is_done() && !slot.prompt_done() {
            let from = slot.prompt_cursor();
            let covered = key.as_ref().map_or(0, |(_, spans)| {
                spans
                    .iter()
                    .take_while(|span| usize::try_from(span.end()).unwrap_or(usize::MAX) <= from)
                    .count()
            });
            let digests = key_digests.get(covered..).unwrap_or_default();
            match self.encode_media(items.get(covered..).unwrap_or_default(), digests) {
                Ok(Some(media)) => slot.attach_media(media),
                Ok(None) => {}
                Err(message) => {
                    tracing::warn!(%message, "a request carrying media could not be served");
                    slot.fail(message);
                }
            }
        }
        tracing::info!(
            id = %slot.id,
            prompt_tokens = slot.prompt.len(),
            cached_tokens = slot.cached_tokens(),
            max_tokens = budget,
            media = items.len(),
            running = self.slots.len() + 1,
            "request admitted"
        );
        if let Some(refused) = self.slots.admit(slot) {
            // Unreachable while admission only happens with a free slot, and
            // still handled rather than asserted: a sequence whose session was
            // opened and then dropped would leak device state, which is the one
            // failure nobody would see until the pool ran out.
            tracing::error!(
                slot = refused.id,
                "a sequence was admitted with no slot for it"
            );
            self.release(refused.session);
        }
    }

    /// Gives a scoring request a session and a slot; its reply is answered by
    /// the prefill terminus, or by the retirement path if the prompt never
    /// finishes reading.
    ///
    /// Deliberately bare beside [`Coordinator::open`]: a scoring prompt is
    /// read once and kept by nobody, so the media checks, the checkpoint
    /// resume, the scaffold and the disk store a generation passes through
    /// are all work this request will never benefit from - and the store,
    /// left in, would file a one-question prompt under a key no follow-up
    /// will ever ask for. The empty prompt and empty label set are refused
    /// here as well as at the handle: the HTTP surface and the handle both
    /// refuse first, and the scheduler refuses too, because a forward pass
    /// over nothing must not be reachable from any caller. Media encoding
    /// does run here, because the tower must splice its rows before the one
    /// prefill reads them, on the scheduler thread where all model calls
    /// live.
    fn open_score(
        &mut self,
        request: ScoreRequest,
        reply: oneshot::Sender<Result<ScoreOutcome, AppError>>,
    ) {
        if request.prompt_tokens.is_empty() || request.labels.is_empty() {
            let _ = reply.send(Err(AppError::bad_request(
                "a scoring request needs a prompt and at least one label",
                None,
                "invalid_request",
            )));
            return;
        }
        // The sink of a stream nobody will read: a scoring sequence answers
        // through the reply, and a dropped receiver is what says so to every
        // send the retirement path makes anyway.
        let (sink, events) = mpsc::channel(EVENT_QUEUE);
        drop(events);
        // Opened directly rather than through `open_session_for`, which hands
        // its failure to a stream this sequence does not have: here the
        // failure's own words ride the reply.
        let session = match self.model.as_model_mut().open_session() {
            Ok(session) => session,
            Err(error) => {
                tracing::warn!(error = %error.chain(), "a scoring session could not be opened");
                let _ = reply.send(Err(AppError::internal(
                    "the scoring session could not be opened",
                    error,
                )));
                return;
            }
        };
        self.stats.admit();

        let now = self.cfg.clock.now();
        let id = format!("{:016x}", self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        // Cloned off the request before it moves into the slot: the tower must
        // run over the items after the slot exists, so the encode can attach
        // its rows to it.
        let items = Arc::clone(&request.media);
        let digests = Arc::clone(&request.media_digests);
        tracing::info!(
            id,
            prompt_tokens = request.prompt_tokens.len(),
            labels = request.labels.len(),
            media = items.len(),
            running = self.slots.len() + 1,
            "scoring request admitted"
        );
        let mut slot = SeqSlot::new_score(
            request,
            SlotSetup {
                id,
                session,
                sink,
                started_at: now,
                deadline: self.cfg.request_timeout.map(|window| now + window),
                budget: 0,
                keyed: false,
                // A score reads label rows and never draws.
                suppressed: Arc::from([]),
                score_reply: Some(reply),
            },
        );
        slot.set_checkpoint_floor(self.cfg.store.min_tokens);
        // The generation encode step in its plain form: a score never resumes
        // and is never keyed, so there is no covered prefix to subtract and no
        // resume to run ahead of. A failure fails the slot, whose still-held
        // reply carries the message through the retirement path.
        if !items.is_empty() {
            match self.encode_media(&items, &digests) {
                Ok(Some(media)) => slot.attach_media(media),
                Ok(None) => {}
                Err(message) => {
                    tracing::warn!(%message, "a scoring request carrying media could not be served");
                    slot.fail(message);
                }
            }
        }
        if let Some(refused) = self.slots.admit(slot) {
            // Unreachable while admission only happens with a free slot;
            // handled for the same reason `open` handles it - a dropped slot
            // would leak its session. Its reply comes back as an engine
            // failure through the dropped channel.
            tracing::error!(
                slot = refused.id,
                "a scoring sequence was admitted with no slot for it"
            );
            self.release(refused.session);
        }
    }

    /// Why a request's media cannot be looked at on this server, when it
    /// cannot: the checkpoint has no media path, or a tower it needs is
    /// missing. The message is written for whoever sent the request and names
    /// the flag an operator would change.
    fn media_refusal(&mut self, items: &[MediaItem]) -> Option<String> {
        if items.is_empty() {
            return None;
        }
        if self.model.as_model_mut().media_mut().is_none() {
            return Some(
                "this checkpoint has no media path, so the attached media cannot be \
                 looked at"
                    .to_owned(),
            );
        }
        if self.vision.is_none() && items.iter().any(|item| !item.is_audio()) {
            return Some(
                "this server was started without a vision tower, so the attached \
                 media cannot be looked at; start it with --vision naming the tower"
                    .to_owned(),
            );
        }
        if self.audio.is_none() && items.iter().any(MediaItem::is_audio) {
            return Some(
                "this server has no audio tower, so the attached audio cannot be \
                 heard; start it with --vision naming a container that carries one"
                    .to_owned(),
            );
        }
        None
    }

    /// The prefix-cache key of a media request's prompt. See [`media_key`].
    fn media_key(&self, request: &GenerateRequest) -> Option<(Vec<TokenId>, Vec<MediaSpan>)> {
        if request.media.is_empty() {
            return None;
        }
        let merge = self
            .vision
            .as_ref()
            .map_or(1, VisionTower::spatial_merge_size);
        let grids: Vec<VisionGrid> = request.media.iter().map(|item| item.grid(merge)).collect();
        media_key(
            &request.prompt_tokens,
            &grids,
            &request.media_digests,
            self.pads,
        )
    }

    /// Asks the model to file `slot`'s path under its media key, and marks the
    /// slot keyed when it agrees. A model that refuses keeps the slot off the
    /// tier, which is the answer the refusal means.
    fn key_session(&mut self, slot: &mut SeqSlot, path: &[TokenId], spans: &[MediaSpan]) {
        let Some(ckpt) = self.model.ckpt_mut() else {
            return;
        };
        let key = match MediaKey::new(path, spans) {
            Ok(key) => key,
            Err(error) => {
                tracing::warn!(%error, "a media prompt's cache key was malformed; it will not be cached");
                return;
            }
        };
        match ckpt.kv_set_media_key(slot.session, &slot.prompt, key) {
            Ok(()) => slot.set_keyed(),
            Err(error) => {
                tracing::debug!(error = %error.chain(), "the model does not key media prompts; this one will not be cached");
            }
        }
    }

    /// Runs each item's tower over a request's media, or explains why it
    /// cannot.
    ///
    /// Items are encoded one at a time in prompt order and gathered into one
    /// block, because the splice consumes rows in the order the pad runs
    /// appear and a picture and a clip interleave there. Both towers loop per
    /// item inside anyway, so nothing is lost by not batching a kind.
    ///
    /// `Ok(None)` is a text-only request: nothing was attached, so nothing was
    /// encoded, and the sequence takes the launches it took before any of this
    /// existed. The `Err` is the message the caller's stream fails with, written
    /// for whoever sent the request and naming the flag an operator would change.
    fn encode_media(
        &mut self,
        items: &[MediaItem],
        digests: &[MediaDigest],
    ) -> Result<Option<EncodedMedia>, String> {
        if items.is_empty() {
            return Ok(None);
        }
        if let Some(message) = self.media_refusal(items) {
            return Err(message);
        }
        let started = Instant::now();
        let mut parts = Vec::with_capacity(items.len());
        let mut ran = 0usize;
        for (at, item) in items.iter().enumerate() {
            // The request thread's lease first, then the cache - another
            // request may have encoded the same picture since this one was
            // rendered - and the tower only when neither has it.
            let digest = digests.get(at).copied();
            let cached = match item {
                MediaItem::Encoded(item) => Some(Arc::clone(&item.rows)),
                _ => digest
                    .zip(self.media_cache.as_ref())
                    .and_then(|(digest, cache)| cache.rows(&digest)),
            };
            if let Some(rows) = cached {
                parts.push(rows);
                continue;
            }
            let rows = self.run_tower(item)?;
            ran += 1;
            parts.push(match (digest, self.media_cache.as_ref()) {
                (Some(digest), Some(cache)) => cache.insert(digest, rows),
                _ => Arc::new(rows),
            });
        }
        let encoded =
            EncodedMedia::from_parts(&self.backend, parts).map_err(|error| error.to_string())?;
        tracing::debug!(
            items = items.len(),
            encoded = ran,
            reused = items.len() - ran,
            rows = encoded.pad_total(),
            ms = started.elapsed().as_secs_f64() * 1000.0,
            "a request's media was encoded"
        );
        Ok(Some(encoded))
    }

    /// Runs one decoded item through its tower.
    fn run_tower(&self, item: &MediaItem) -> Result<CachedRows, String> {
        let (encoded, audio) = match item {
            MediaItem::Vision(item) => {
                let Some(tower) = self.vision.as_ref() else {
                    return Err(
                        "this server was started without a vision tower, so the attached \
                         media cannot be looked at; start it with --vision naming the tower"
                            .to_owned(),
                    );
                };
                let encoded = tower
                    .encode_rows(&self.backend, std::slice::from_ref(item))
                    .map_err(|error| error.to_string())?;
                (encoded, false)
            }
            MediaItem::Audio(clip) => {
                let Some(tower) = self.audio.as_ref() else {
                    return Err(
                        "this server has no audio tower, so the attached audio cannot be \
                         heard; start it with --vision naming a container that carries one"
                            .to_owned(),
                    );
                };
                let encoded = tower
                    .encode(&self.backend, std::slice::from_ref(clip))
                    .map_err(|error| error.to_string())?;
                (encoded, true)
            }
            MediaItem::Encoded(_) => {
                return Err("an item that was already encoded reached the tower".to_owned());
            }
        };
        let grid = match encoded.grids.as_slice() {
            [grid] => *grid,
            grids => {
                return Err(format!("one item encoded to {} grids", grids.len()));
            }
        };
        Ok(CachedRows::new(encoded.rows, grid, audio))
    }

    /// How many tokens a turn over this prompt may generate.
    ///
    /// The smaller of what the caller asked for and what is left of the
    /// session's context. A budget that ignored the context would overflow half
    /// way through a turn, which reaches the caller as a failure rather than as
    /// a truncated answer.
    fn budget_for(&self, prompt_tokens: usize, asked: u32) -> u32 {
        let prompt = u32::try_from(prompt_tokens).unwrap_or(u32::MAX);
        asked.min(self.ctx.saturating_sub(prompt))
    }

    /// Stands a fresh sequence on a stored prefix, when there is one, and tells
    /// the model where its prompt's message boundaries are.
    fn try_resume(&mut self, slot: &mut SeqSlot) {
        if !slot.checkpointable() {
            return;
        }
        let Self {
            model,
            backend,
            row,
            scores,
            stats,
            vocabulary,
            cfg,
            stop,
            ..
        } = self;
        let Some(ckpt) = model.ckpt_mut() else {
            return;
        };
        let hints = message_hints(&slot.prompt, *stop);
        if let Err(error) = ckpt.kv_set_hints(slot.session, &hints) {
            tracing::debug!(error = %error.chain(), "the prompt's message boundaries were not recorded");
        }
        let whole_prompt = slot.prompt.len();
        let reused = match ckpt.kv_resume(slot.session, &slot.prompt) {
            Err(error) => {
                tracing::warn!(error = %error.chain(), "a checkpoint resume failed; the prompt will be prefilled");
                return;
            }
            Ok(None) => return,
            Ok(Some(info)) => {
                let reused = info.tokens_reused;
                if usize::try_from(reused).unwrap_or(usize::MAX) >= whole_prompt {
                    // The stored payload carries the row the original run
                    // produced, so a full hit is already standing on the answer
                    // its next sample reads and no step is re-run to recover it.
                    let read = info
                        .frontier_logits
                        .as_ref()
                        .ok_or(CoreError::Unsupported {
                            what: "a resume of a whole prompt without its logits row",
                            backend: "scheduler",
                        })
                        .and_then(|logits| logits.row(0))
                        .and_then(|view| backend.copy_d2h(view, row));
                    if let Err(error) = read {
                        // The session already stands on the prefix, so a
                        // prefill from here would run the prompt twice; the
                        // sequence fails instead of answering from that.
                        tracing::warn!(error = %error.chain(), "a resumed frontier row did not read back");
                        slot.fail(error.to_string());
                        return;
                    }
                }
                tracing::debug!(
                    slot = slot.id,
                    tokens_reused = reused,
                    source = ?info.source,
                    "a stored prefix was resumed"
                );
                reused
            }
        };
        slot.resumed(reused);
        stats.resumed(u64::from(reused));
        if slot.prompt_done() {
            start_decoding(
                row,
                scores,
                vocabulary.as_deref(),
                *stop,
                cfg.clock.now(),
                stats,
                slot,
            );
        }
    }

    /// The quantum `step` as it runs: a score's first quantum resumes first
    /// (see [`Coordinator::resume_score`]), and `None` when there is nothing
    /// to read - no such slot, or a resume that failed it or already answered
    /// it with a full hit.
    ///
    /// The plan was made before the resume moved the cursor, so a quantum
    /// planned as a middle piece can now reach the prompt's end - and the end
    /// is the one piece that must read its row.
    fn ready_step(&mut self, step: PrefillStep) -> Option<Quantum> {
        let epoch = self.score_pass.shared_reads;
        if let Some(shared) = self
            .slots
            .get_mut(step.slot)
            .and_then(|slot| slot.settle_score_resume(epoch))
        {
            self.resume_score(step.slot, shared);
        }
        let slot = self.slots.slots().get(step.slot)?;
        if slot.is_done() || slot.prompt_done() {
            return None;
        }
        let (start, end) = slot.prompt_window(step.tokens);
        if start >= end {
            return None;
        }
        Some(Quantum {
            step: PrefillStep {
                last: step.last || end == slot.prompt.len(),
                ..step
            },
            session: slot.session,
            prompt: Arc::clone(&slot.prompt),
            window: start..end,
        })
    }

    /// Stands a score on a stored prefix at its first prefill quantum, and
    /// names the point its sibling questions diverge so the stack checkpoints
    /// it there.
    ///
    /// The score's own boundaries are left out: its message ends lie inside
    /// its own question, past where any sibling shares it, and a checkpoint
    /// there would be a recurrent read-back nothing resumes. A full hit
    /// answers from the stored row, as [`Coordinator::try_resume`] does.
    fn resume_score(&mut self, at: usize, shared: u32) {
        let Self {
            model,
            backend,
            row,
            scores,
            stats,
            slots,
            ..
        } = self;
        let Some(slot) = slots.get_mut(at) else {
            return;
        };
        if !slot.checkpointable() {
            return;
        }
        let Some(ckpt) = model.ckpt_mut() else {
            return;
        };
        if let Err(error) = ckpt.kv_set_scoring(slot.session) {
            tracing::debug!(error = %error.chain(), "a score was not marked as one");
        }
        let hints: &[u32] = if shared > 0 { &[shared] } else { &[] };
        if let Err(error) = ckpt.kv_set_hints(slot.session, hints) {
            tracing::debug!(error = %error.chain(), "the shared prefix of a score was not recorded");
        }
        let whole_prompt = slot.prompt.len();
        let reused = match ckpt.kv_resume(slot.session, &slot.prompt) {
            Err(error) => {
                // A sibling waiting on a checkpoint asks again each time one
                // may have been written; only its first failure is news.
                if slot.score_resume_retried() {
                    tracing::debug!(error = %error.chain(), "a waiting score's checkpoint resume failed again");
                } else {
                    tracing::warn!(error = %error.chain(), "a score's checkpoint resume failed; the prompt will be prefilled");
                }
                return;
            }
            Ok(None) => return,
            Ok(Some(info)) => {
                let reused = info.tokens_reused;
                if usize::try_from(reused).unwrap_or(usize::MAX) >= whole_prompt {
                    let read = info
                        .frontier_logits
                        .as_ref()
                        .ok_or(CoreError::Unsupported {
                            what: "a resume of a whole prompt without its logits row",
                            backend: "scheduler",
                        })
                        .and_then(|logits| logits.row(0))
                        .and_then(|view| backend.copy_d2h(view, row));
                    if let Err(error) = read {
                        tracing::warn!(error = %error.chain(), "a resumed frontier row did not read back");
                        slot.fail(error.to_string());
                        return;
                    }
                }
                tracing::debug!(
                    slot = slot.id,
                    tokens_reused = reused,
                    shared,
                    source = ?info.source,
                    "a score resumed a stored prefix"
                );
                reused
            }
        };
        slot.resumed(reused);
        stats.resumed(u64::from(reused));
        if slot.prompt_done() {
            finish_scoring(row, scores, slot);
        }
    }

    /// Copies queued checkpoint pages off the device while no sequence is
    /// resident. Returns `true` when a request arrived meanwhile and was
    /// admitted, so the loop runs it instead of blocking.
    fn pump_idle(&mut self, rx: &mut mpsc::Receiver<EngineRequest>) -> bool {
        let budget = self.cfg.store.pump_bytes.saturating_mul(IDLE_PUMP_FACTOR);
        loop {
            let Some(ckpt) = self.model.ckpt_mut() else {
                return false;
            };
            let queued = match ckpt.kv_pump(budget) {
                Ok(report) => {
                    if report.pages > 0 || report.entries > 0 {
                        tracing::trace!(?report, "idle pump");
                    }
                    report.queued
                }
                Err(error) => {
                    tracing::warn!(error = %error.chain(), "a queued checkpoint could not be copied");
                    0
                }
            };
            if queued == 0 {
                return false;
            }
            match rx.try_recv() {
                Ok(request) => {
                    self.admit(request);
                    return true;
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.closing = true;
                    return true;
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    // Staging is full until the writer lands what it has; a
                    // short wait lets it, and the queue is polled again after.
                    std::thread::sleep(IDLE_PUMP_WAIT);
                }
            }
        }
    }

    /// One tick.
    fn tick(&mut self) {
        let now = self.cfg.clock.now();
        cancel::enforce_deadlines(&self.stats, self.slots.slots_mut(), now);
        let plan = batching::plan(&self.planned_facts(), &self.cfg);
        for (at, slot) in self.slots.slots_mut().iter_mut().enumerate() {
            if slot.is_flex() {
                slot.mark_paused(plan.paused.contains(&at), now);
            }
        }
        if plan.is_idle() {
            // Something is resident and none of it is decoding or reading
            // prompt. By construction that cannot happen - a sequence leaves
            // prefill standing on a token, and one that stops is retired at the
            // end of the tick - so reaching it means an invariant broke, and
            // looping on it would burn the one engine thread at full speed
            // forever. Ending those sequences is the failure mode that is
            // visible to the caller instead of to a fan.
            self.unstick();
        }
        // In groups no wider than the engine's decode batch: `--max-num-seqs`
        // may admit more sequences than one pass carries, and a batch wider
        // than its logits block is refused outright.
        for group in plan.decode.chunks(DECODE_GROUP_ROWS) {
            self.decode_step(group);
        }
        if !plan.prefill_batch.is_empty() {
            let quantum = batching::quantum(!plan.decode.is_empty(), &self.cfg);
            self.prefill_batch_step(&plan.prefill_batch, quantum);
        } else if let Some(step) = plan.prefill {
            self.read_alone(step);
        }
        self.store_continued();
        self.retire();
        self.pump();
    }

    /// Copies a bounded slice of queued checkpoint pages off the device, between
    /// steps: the write-behind's one foot on the scheduler thread.
    fn pump(&mut self) {
        let budget = self.cfg.store.pump_bytes;
        let Some(ckpt) = self.model.ckpt_mut() else {
            return;
        };
        if let Err(error) = ckpt.kv_pump(budget) {
            tracing::warn!(error = %error.chain(), "a queued checkpoint could not be copied");
        }
    }

    /// One decode step, speculative when this build and this sequence allow it.
    ///
    /// Three conditions, and all of them are necessary. The model has to carry a
    /// draft head. The step has to be *one* sequence, because a verify pass is
    /// one sequence's consecutive positions and there is no batched form of it
    /// here. And the sampling has to be greedy, because a draft is accepted by
    /// comparing it against the target's greedy pick: with a temperature the
    /// caller might not have chosen the token the comparison accepted on, and
    /// the committed prefix would be one the model never sampled.
    ///
    /// Anything else takes the batched path below, which is the same work a
    /// build with no head does.
    fn decode_step(&mut self, rows: &[usize]) {
        self.decode_steps = self.decode_steps.wrapping_add(1);
        if let [only] = rows
            && self.speculative(*only)
        {
            self.speculative_step(*only);
            return;
        }
        // Several sequences, all greedy, on a build with a draft head: one
        // speculative step for all of them, which a family that can verify
        // several sequences in one pass does in one pass over its weights.
        if rows.len() > 1 && rows.iter().all(|at| self.speculative(*at)) {
            // A sampled member is settled by the two-phase step, one session
            // at a time; an all-greedy batch keeps the batched step. In
            // opportunistic mode, speculative batching runs for all members.
            if rows.iter().all(|at| self.greedy(*at)) || !self.cfg.mtp_exact_sampling {
                self.speculative_batch_step(rows);
            } else {
                for at in rows {
                    self.speculative_step(*at);
                }
            }
            return;
        }
        self.batched_decode_step(rows);
    }

    /// One speculative step over several sequences.
    fn speculative_batch_step(&mut self, rows: &[usize]) {
        let batch: Vec<(usize, DecodeStep)> = rows
            .iter()
            .filter_map(|at| {
                self.slots
                    .slots()
                    .get(*at)
                    .and_then(SeqSlot::decode_step)
                    .map(|step| (*at, step))
            })
            .collect();
        if batch.is_empty() {
            return;
        }
        let steps: Vec<(SessionId, TokenId)> = batch
            .iter()
            .map(|(_, step)| (step.session, step.token))
            .collect();
        let started = Instant::now();
        let results = {
            let Some(spec) = self.model.as_model_mut().spec_mut() else {
                return;
            };
            match spec.speculate_batch(&steps) {
                Ok(results) => results,
                Err(error) => {
                    tracing::error!(%error, rows = batch.len(), "a batched speculative decode step failed");
                    for (at, _) in &batch {
                        self.fail_slot(*at, &error, "a batched speculative decode step failed");
                    }
                    return;
                }
            }
        };
        tracing::debug!(
            rows = batch.len(),
            produced = results
                .iter()
                .map(|step| step.produced.len())
                .sum::<usize>(),
            accepted = results.iter().map(|step| step.accepted).sum::<u32>(),
            ms = started.elapsed().as_secs_f64() * 1e3,
            "a batched speculative step"
        );
        let now = self.cfg.clock.now();
        for ((at, _), step) in batch.iter().zip(results) {
            self.stats.speculated(step.accepted, step.drafted);
            self.stats.decoded(step.produced.len());
            let Self {
                slots,
                vocabulary,
                stats,
                stop,
                ..
            } = self;
            let Some(slot) = slots.get_mut(*at) else {
                continue;
            };
            slot.speculated(step.accepted, step.drafted);
            let mut events = Vec::new();
            for token in step.produced {
                if slot.is_done() {
                    break;
                }
                events.extend(slot.accept(token, vocabulary.as_deref(), *stop, now));
            }
            send_all(stats, slot, events);
        }
    }

    /// Whether this row's next step can be speculative: a greedy sequence on
    /// a model that drafts, a sampled one whose tokens are keyed and so can
    /// be checked against a draft by equality, or an opportunistic sequence
    /// when exact sampling is not requested.
    fn speculative(&mut self, at: usize) -> bool {
        let eligible = self.slots.slots().get(at).is_some_and(|slot| {
            slot.sampling().is_greedy() || slot.is_keyed() || !self.cfg.mtp_exact_sampling
        });
        eligible && self.model.as_model_mut().spec_mut().is_some()
    }

    /// Whether this row samples greedily.
    fn greedy(&self, at: usize) -> bool {
        self.slots
            .slots()
            .get(at)
            .is_some_and(|slot| slot.sampling().is_greedy())
    }

    /// One speculative step over one sequence.
    ///
    /// The two-phase step where the model has one: the drafted block and the
    /// target's rows come back, and the sequence itself chooses each token -
    /// by its own rule at that position, tool-call tags and all - keeping the
    /// drafts up to the first it did not choose. The greedy step otherwise,
    /// and a plain step for a sampled sequence on a model with neither.
    fn speculative_step(&mut self, at: usize) {
        let Some(step) = self.slots.slots().get(at).and_then(SeqSlot::decode_step) else {
            return;
        };
        let (hint, proposal) = match self.slots.get_mut(at) {
            Some(slot) => (
                slot.draft_hint(),
                slot.scaffold_proposal(self.vocabulary.as_deref()),
            ),
            None => return,
        };
        let verified = match self.model.as_model_mut().spec_mut() {
            Some(spec) => spec.verify_block(step.session, step.token, hint, &proposal),
            None => return,
        };
        match verified {
            Ok(Some(block)) => {
                self.settle_two_phase(at, step, block);
                return;
            }
            Ok(None) => {}
            Err(error) => {
                tracing::error!(%error, "a speculative verify failed");
                self.fail_slot(at, &error, "a speculative verify failed");
                return;
            }
        }
        if !self.greedy(at) && self.cfg.mtp_exact_sampling {
            self.batched_decode_step(&[at]);
            return;
        }
        let (produced, accepted, drafted) = {
            let Some(spec) = self.model.as_model_mut().spec_mut() else {
                return;
            };
            match spec.speculate(step.session, step.token) {
                Ok(step) => {
                    self.stats.speculated(step.accepted, step.drafted);
                    (step.produced, step.accepted, step.drafted)
                }
                Err(error) => {
                    tracing::error!(%error, "a speculative decode step failed");
                    self.fail_slot(at, &error, "a speculative decode step failed");
                    return;
                }
            }
        };
        self.stats.decoded(produced.len());
        let now = self.cfg.clock.now();
        let Self {
            slots,
            vocabulary,
            stats,
            stop,
            ..
        } = self;
        let Some(slot) = slots.get_mut(at) else {
            return;
        };
        slot.speculated(accepted, drafted);
        let mut events = Vec::new();
        // A step can produce several tokens and the sequence can end inside
        // them: the ones past the end were committed to the session, which is
        // about to be retired, and emitting them would put text after the end of
        // the answer.
        for token in produced {
            if slot.is_done() {
                break;
            }
            events.extend(slot.accept(token, vocabulary.as_deref(), *stop, now));
        }
        send_all(stats, slot, events);
    }

    /// Chooses the tokens of a verified block for the sequence at `at` and
    /// settles the model on how many drafts it kept.
    fn settle_two_phase(
        &mut self,
        at: usize,
        step: DecodeStep,
        mut block: shiso_core::model::VerifiedBlock,
    ) {
        let now = self.cfg.clock.now();
        let drafted = u32::try_from(block.drafts.len()).unwrap_or(u32::MAX);
        let mut accepted = 0u32;
        let mut produced = 0usize;
        {
            let Self {
                slots,
                vocabulary,
                stats,
                stop,
                ..
            } = self;
            let Some(slot) = slots.get_mut(at) else {
                return;
            };
            let mut events = Vec::new();
            // A copy of the drafts, so the block can lend each row out to be
            // masked while they are compared against.
            let drafts = block.drafts.clone();
            for (i, draft) in drafts
                .iter()
                .map(Some)
                .chain(std::iter::once(None))
                .enumerate()
            {
                if slot.is_done() {
                    break;
                }
                let token = slot.choose(block.row_mut(i));
                events.extend(slot.accept(token, vocabulary.as_deref(), *stop, now));
                produced += 1;
                if draft == Some(&token) {
                    accepted += 1;
                } else {
                    break;
                }
            }
            slot.speculated(accepted, drafted);
            send_all(stats, slot, events);
        }
        self.stats.speculated(accepted, drafted);
        self.stats.decoded(produced);
        let settled = match self.model.as_model_mut().spec_mut() {
            Some(spec) => spec.settle_block(step.session, block, accepted),
            None => return,
        };
        if let Err(error) = settled {
            tracing::error!(%error, "a speculative settle failed");
            self.fail_slot(at, &error, "a speculative settle failed");
        }
    }

    /// One batched decode step over every ready sequence.
    fn batched_decode_step(&mut self, rows: &[usize]) {
        let batch: Vec<BatchRow> = rows
            .iter()
            .filter_map(|at| {
                self.slots
                    .slots()
                    .get(*at)
                    .and_then(SeqSlot::decode_step)
                    .map(|step| BatchRow { slot: *at, step })
            })
            .collect();
        if batch.is_empty() {
            return;
        }
        let steps: Vec<DecodeStep> = batch.iter().map(|row| row.step).collect();

        let sampled = {
            let Self {
                model,
                backend,
                slots,
                row,
                scores,
                ..
            } = self;
            let logits = match model.as_model_mut().decode_batch(&steps) {
                Ok(logits) => logits,
                Err(error) => {
                    let message = error.to_string();
                    tracing::error!(error = %error.chain(), rows = batch.len(), "a batched decode step failed");
                    for entry in &batch {
                        if let Some(slot) = slots.get_mut(entry.slot) {
                            slot.fail(message.clone());
                        }
                    }
                    return;
                }
            };
            let mut sampled = Vec::with_capacity(batch.len());
            for (index, entry) in batch.iter().enumerate() {
                let at = u32::try_from(index).unwrap_or(u32::MAX);
                let read = logits
                    .row(at)
                    .and_then(|view| backend.copy_d2h(view, row.as_mut_slice()));
                if let Err(error) = read {
                    tracing::error!(error = %error.chain(), row = index, "a decode row did not read back");
                    break;
                }
                let Some(slot) = slots.get_mut(entry.slot) else {
                    continue;
                };
                sampled.push((entry.slot, slot.choose(fill(row, scores))));
            }
            sampled
        };

        self.stats.decoded(sampled.len());
        let now = self.cfg.clock.now();
        for (at, token) in sampled {
            let Self {
                slots,
                vocabulary,
                stats,
                stop,
                ..
            } = self;
            let Some(slot) = slots.get_mut(at) else {
                continue;
            };
            let events = slot.accept(token, vocabulary.as_deref(), *stop, now);
            send_all(stats, slot, events);
        }
    }

    /// One bounded quantum of one sequence's prompt.
    fn prefill_step(&mut self, step: PrefillStep) {
        let Some(quantum) = self.ready_step(step) else {
            return;
        };
        let (step, session, prompt) = (quantum.step, quantum.session, quantum.prompt);
        let chunk = &prompt[quantum.window];
        let covered = chunk.len();

        let carries_media = self
            .slots
            .slots()
            .get(step.slot)
            .is_some_and(|slot| slot.media().is_some());

        if !step.last {
            if carries_media {
                // Unreachable: the planner gives a media sequence its whole
                // prompt in one quantum. Handled rather than asserted because
                // the alternative to failing here is a second call that restarts
                // the placeholder walk, and that is the failure that answers
                // fluently about a rearranged picture instead of erroring.
                let error = CoreError::Unsupported {
                    what: "a media prompt read in pieces",
                    backend: "scheduler",
                };
                self.fail_slot(step.slot, &error, "a media prompt was split");
                return;
            }
            let started = Instant::now();
            if let Err(error) = self.model.as_model_mut().prefill_chunk(session, chunk) {
                self.fail_slot(step.slot, &error, "a prompt chunk could not be read");
                return;
            }
            let decode_steps = self.decode_steps;
            if let Some(slot) = self.slots.get_mut(step.slot) {
                slot.advance_prefill(covered);
                slot.time_prefill(covered, started.elapsed(), decode_steps);
            }
            self.stats.prefilled(covered as u64);
            return;
        }
        let started = Instant::now();

        {
            let Self {
                model,
                backend,
                slots,
                pads,
                row,
                ..
            } = self;
            let hidden = model.as_model_mut().info().hidden;
            let media = slots.slots().get(step.slot).and_then(SeqSlot::media);
            let read = match media {
                None => model.as_model_mut().prefill_last(session, chunk),
                Some(media) => {
                    // The placeholder ids were resolved at boot, from the same
                    // placement the renderer expanded the prompt with: they are
                    // the ids the pad run is written in, and a second spelling
                    // of them would be a second answer that can disagree with
                    // the first.
                    VisionPrefill::new(media.rows().view(), media.grids(), *pads, hidden).and_then(
                        |prefill| {
                            model
                                .as_model_mut()
                                .media_mut()
                                .ok_or(CoreError::Unsupported {
                                    what: "media in a prompt this checkpoint cannot splice",
                                    backend: "scheduler",
                                })?
                                .prefill_media(session, chunk, &prefill)
                        },
                    )
                }
            };
            let read = read.and_then(|logits| {
                logits
                    .row(0)
                    .and_then(|view| backend.copy_d2h(view, row.as_mut_slice()))
            });
            if let Err(error) = read {
                self.fail_slot(step.slot, &error, "a prompt could not be read");
                return;
            }
        }
        self.finish_prompt(step.slot, covered, started);

        let Self {
            slots,
            row,
            scores,
            vocabulary,
            stats,
            cfg,
            stop,
            ..
        } = self;
        let now = cfg.clock.now();
        if let Some(slot) = slots.get_mut(step.slot) {
            if slot.is_score() {
                finish_scoring(row, scores, slot);
            } else {
                start_decoding(row, scores, vocabulary.as_deref(), *stop, now, stats, slot);
            }
        }
    }

    /// Ends every resident sequence the loop has no work for.
    ///
    /// See the call site: this is unreachable, and it is here rather than as a
    /// debug assertion because the alternative to ending those sequences is an
    /// engine thread spinning at full speed for the life of the process.
    fn unstick(&mut self) {
        for slot in self.slots.slots_mut() {
            if slot.is_done() {
                continue;
            }
            tracing::error!(
                slot = slot.id,
                "a resident sequence is neither decoding nor reading prompt; ending it"
            );
            slot.fail("the scheduler has no work for this sequence".to_owned());
        }
    }

    /// Marks one sequence as having failed underneath.
    fn fail_slot(&mut self, at: usize, error: &CoreError, what: &str) {
        tracing::error!(error = %error.chain(), "{what}");
        let message = error.to_string();
        if let Some(slot) = self.slots.get_mut(at) {
            slot.fail(message);
        }
    }

    /// Accounts a prompt's last chunk: the cursor, the counters, and the prefill
    /// rate the disk tier prices a resume against.
    ///
    /// A prompt whose chunks ran with no decode step between them timed the
    /// device alone, and the last chunk's row read waited for all of it: a fair
    /// sample of what a prefill costs. The prompt's end itself was queued for
    /// disk by the model's own commit.
    fn finish_prompt(&mut self, at: usize, covered: usize, started: Instant) {
        let decode_steps = self.decode_steps;
        let timed = self.slots.get_mut(at).and_then(|slot| {
            slot.advance_prefill(covered);
            slot.time_prefill(covered, started.elapsed(), decode_steps);
            slot.solo_prefill(decode_steps)
        });
        self.stats.prefilled(covered as u64);
        if let Some((tokens, elapsed)) = timed
            && let Some(ckpt) = self.model.ckpt_mut()
        {
            ckpt.kv_observe_prefill(tokens, elapsed.as_secs_f64());
        }
    }

    /// Files a growing conversation's frontier on the configured interval.
    ///
    /// Off by default: the end of every turn is filed anyway, and a checkpoint
    /// mid-answer is a snapshot of the whole recurrent stack for a position the
    /// turn's end supersedes. What the interval buys is a crash mid-answer that
    /// resumes part-way through it.
    fn store_continued(&mut self) {
        if self.cfg.store.continued_interval == 0 || self.model.ckpt_mut().is_none() {
            return;
        }
        let policy = self.cfg.store;
        let due: Vec<usize> = self
            .slots
            .slots()
            .iter()
            .enumerate()
            .filter(|(_at, slot)| {
                // A score's prompt is read once and kept by nobody: filing a
                // mid-prefill score frontier would put a one-question prompt
                // on the disk tier under a key no follow-up will ever ask
                // for. The retirement site already refuses a score; this is
                // the other half of that ruling, for the interval path.
                !slot.is_done()
                    && !slot.is_score()
                    && slot.checkpointable()
                    && policy.stores_continued(slot.written_len(), slot.stored_len, false)
            })
            .map(|(at, _slot)| at)
            .collect();
        for at in due {
            let Some(slot) = self.slots.slots().get(at) else {
                continue;
            };
            let (session, written) = (slot.session, slot.written_len());
            if self.save(session, Reason::Continued)
                && let Some(slot) = self.slots.get_mut(at)
            {
                slot.stored_len = written;
            }
        }
    }

    /// Offers the checkpoint at a session's frontier to the disk tier.
    ///
    /// The model's own gates decide whether anything is queued: the floor, an
    /// entry already stored, a session that read media. None of those are
    /// re-implemented here, because a scheduler that decided them too would be
    /// a second answer that can disagree with the first.
    fn save(&mut self, session: SessionId, reason: Reason) -> bool {
        let Self { model, stats, .. } = self;
        let Some(ckpt) = model.ckpt_mut() else {
            return false;
        };
        match ckpt.kv_save(session, reason) {
            Ok(outcome) => {
                if outcome.stored().is_some() {
                    stats.checkpoint_written();
                    tracing::debug!(?reason, ?outcome, "a frontier was offered to disk");
                    return true;
                }
                tracing::debug!(?outcome, ?reason, "a frontier was not stored");
                false
            }
            Err(error) => {
                // A store that fails costs the next request a prefill and
                // nothing else, so it is reported and the turn continues.
                tracing::warn!(error = %error.chain(), ?reason, "a frontier could not be stored");
                false
            }
        }
    }

    /// Sends the terminal events of everything that stopped, stores it, and
    /// closes its session.
    fn retire(&mut self) {
        let finished = self.slots.take_finished();
        if finished.is_empty() {
            return;
        }
        let now = self.cfg.clock.now();
        for mut slot in finished {
            let ending = slot.ending.clone();
            // The other half of the reply-exactly-once invariant for scoring
            // sequences: `Ok` is sent at the prefill terminus, and a reply
            // still held here belongs to a sequence that ended without ever
            // reading its prompt to the end - a failed read, a deadline, a
            // table refusal - which its caller must hear about as exactly
            // that. The scheduler fabricates no `AppError` for a generation
            // turn - an error there travels as a stream event, or nowhere -
            // and scoring is the deliberate exception: with no stream, the
            // reply is the only place this failure can be said at all.
            if let Some(reply) = slot.take_score_reply() {
                let error = match ending.as_ref() {
                    Some(Ending::Failed(message)) => AppError::internal(
                        "a scoring prompt could not be read",
                        io::Error::other(message.clone()),
                    ),
                    _ => AppError::internal(
                        "the scoring sequence ended without an answer",
                        io::Error::other("the prompt was never read to its end"),
                    ),
                };
                let _ = reply.send(Err(error));
            }
            let events = slot.terminal_events(now);
            let ttft = slot.ttft();
            if !events.is_empty() {
                send_all(&self.stats, &mut slot, events);
            }
            match &ending {
                Some(Ending::Stopped(_)) => self.stats.completed(ttft),
                Some(Ending::Failed(_)) => self.stats.failed(),
                _ => {}
            }
            // A scoring prompt is read once and kept by nobody: no prefix on
            // the disk tier - nothing will resume a one-question prompt whose
            // answer the caller already has - and no turn replayed for the
            // renderer, because a scoring sequence generated no turn.
            if !slot.is_score() {
                self.store_on_retire(&slot, ending.as_ref());
                self.remember_turn(&slot);
            }
            log_retired(&slot, ending.as_ref(), now);
            self.release(slot.session);
        }
        self.stats.set_active(self.slots.len());
    }

    /// Keeps a finished turn's ids for the renderer, so the next prompt in its
    /// conversation carries the ids this turn ran rather than a re-encoding of
    /// their text that can split differently.
    fn remember_turn(&self, slot: &SeqSlot) {
        let (Some(replay), Some(vocabulary)) = (&self.cfg.turn_replay, &self.vocabulary) else {
            return;
        };
        if let Some((text, tokens)) = slot.finished_turn(vocabulary) {
            replay.record(text, tokens);
        }
    }

    /// Stores a retiring sequence's prefix, when the policy says to.
    ///
    /// The reason says why the state is leaving memory, which is what an
    /// operator reading a cache directory needs: a turn that finished is a
    /// continuation somebody may resume, a turn that was cut is state being
    /// discarded, and a process on its way down is neither.
    fn store_on_retire(&mut self, slot: &SeqSlot, ending: Option<&Ending>) {
        if !slot.checkpointable() || self.model.ckpt_mut().is_none() {
            return;
        }
        if matches!(ending, Some(Ending::Failed(_))) {
            // A turn that failed mid-pass may be standing on state that is not
            // what its tokens describe, and a file that claims otherwise is
            // worse than no file.
            return;
        }
        if !self
            .cfg
            .store
            .stores_continued(slot.written_len(), slot.stored_len, true)
        {
            return;
        }
        let reason = if self.closing {
            Reason::Shutdown
        } else if matches!(ending, Some(Ending::Stopped(_))) {
            Reason::Continued
        } else {
            Reason::Evict
        };
        if self.save(slot.session, reason) && reason == Reason::Shutdown {
            self.stats.swept(1);
        }
    }

    /// Closes one session. The only caller of `close_session` in this crate.
    fn release(&mut self, session: SessionId) {
        match self.model.as_model_mut().close_session(session) {
            Ok(()) => self.stats.session_closed(),
            Err(error) => {
                // Counted anyway: the coordinator is done with this session
                // whatever the model says, and a counter that stopped matching
                // the opens would report a leak that is really a failed release.
                self.stats.session_closed();
                tracing::error!(error = %error.chain(), %session, "a session could not be released");
            }
        }
    }

    /// The tail: terminal events for anything still live, the sweep, then the
    /// sessions.
    fn shutdown(&mut self) {
        // Nothing is held by the time the loop ends - an empty table takes the
        // next held request - so this is the safety net, not the path: a held
        // caller is told why its stream ended rather than seeing it close.
        for held in self.held_flex.drain(..) {
            send_event(
                &held.sink,
                StreamEvent::Failed {
                    message: "the server shut down before this flex request ran".to_owned(),
                },
            );
        }
        let mut live = self.slots.drain();
        if !live.is_empty() {
            tracing::info!(
                sequences = live.len(),
                "the queue closed with sequences still running; ending them"
            );
            let now = self.cfg.clock.now();
            for slot in &mut live {
                if !slot.is_done() {
                    slot.stop(StopReason::EndTurn);
                }
                // The drain does not run `retire` - the loop only ends with
                // the table empty, so anything here is the safety net - and a
                // score whose reply only dropped with its slot would come back
                // to its caller as "the engine dropped the request", which is
                // true and less useful than the truth from here. The same
                // reply-exactly-once invariant as `retire`: only an
                // unanswered score still holds its reply at this point.
                if let Some(reply) = slot.take_score_reply() {
                    let _ = reply.send(Err(AppError::internal(
                        "the scoring sequence ended without an answer",
                        io::Error::other("the server shut down before the prompt was read"),
                    )));
                }
                let events = slot.terminal_events(now);
                send_all(&self.stats, slot, events);
            }
        }
        // Unconditional, because "there is nothing left to sweep" is the
        // engine's answer to give and not this loop's to assume: a session this
        // table does not know about is exactly the kind the sweep exists for.
        self.sweep();
        for slot in &live {
            self.release(slot.session);
        }
    }

    /// Stores every session that is still open on the way down, then waits for
    /// every queued write to land.
    ///
    /// The sweep ordinarily stores nothing new, because the retirement path has
    /// already offered each sequence as it ended. It is the safety net for the
    /// sessions that path never reached. The drain is bounded: the unit that
    /// runs this server gives the process a fixed time to stop, and a write
    /// still queued when the bound passes is abandoned rather than killed
    /// halfway.
    fn sweep(&mut self) {
        let timeout = self.cfg.store.drain_timeout;
        let Self { model, stats, .. } = self;
        let Some(ckpt) = model.ckpt_mut() else {
            return;
        };
        let written = ckpt.kv_shutdown_sweep();
        stats.swept(written as u64);
        let started = Instant::now();
        let landed = ckpt.kv_drain(timeout);
        tracing::info!(
            landed,
            drain_ms = started.elapsed().as_millis(),
            "the checkpoint queue was drained on the way down"
        );
    }
}

/// The one info line a finished request ends with.
fn log_retired(slot: &SeqSlot, ending: Option<&Ending>, now: Instant) {
    let usage = slot.usage(now);
    let ending = match ending {
        Some(Ending::Stopped(reason)) => format!("{reason:?}"),
        Some(Ending::Cancelled) => "cancelled".to_owned(),
        Some(Ending::Failed(_)) => "failed".to_owned(),
        None => "unknown".to_owned(),
    };
    let (drafted, accepted, steps) = slot.speculation();
    #[expect(
        clippy::cast_precision_loss,
        reason = "ratios printed for a human to read to one decimal place"
    )]
    let (spec_accept, spec_tokens_per_step) = if steps == 0 {
        (None, None)
    } else {
        (
            (drafted > 0).then(|| format!("{:.1}%", accepted as f64 / drafted as f64 * 100.0)),
            Some(format!("{:.2}", (accepted + steps) as f64 / steps as f64)),
        )
    };
    tracing::info!(
        id = %slot.id,
        %ending,
        prompt_tokens = usage.prompt_tokens,
        cached_tokens = usage.cached_prompt_tokens,
        generated = usage.completion_tokens,
        ttft_ms = format_args!("{:.0}", usage.time_to_first_token_ms),
        prefill_tps = slot
            .prefill_rate()
            .map(|rate| tracing::field::display(format!("{rate:.1}"))),
        decode_tps = format_args!("{:.1}", usage.tokens_per_second),
        spec_accept = spec_accept.map(tracing::field::display),
        spec_tokens_per_step = spec_tokens_per_step.map(tracing::field::display),
        total_ms = format_args!(
            "{:.0}",
            now.saturating_duration_since(slot.started_at()).as_secs_f64() * 1e3
        ),
        "request finished"
    );
}

/// A media prompt's prefix-cache key: its placeholder runs written in the
/// items' content ids, and where each run sits.
///
/// `None` - the request is then kept off the checkpoint tier, as every media
/// request once was - for a prompt whose runs cannot be matched to its items
/// one for one: a run of the wrong length, a stray placeholder, or an item with
/// no digest. Each of those would key some positions under the wrong picture,
/// and a key that is wrong is worse than none.
fn media_key(
    prompt: &[TokenId],
    grids: &[VisionGrid],
    digests: &[MediaDigest],
    pads: PadTokens,
) -> Option<(Vec<TokenId>, Vec<MediaSpan>)> {
    if grids.is_empty() || digests.len() != grids.len() {
        return None;
    }
    let mut path = prompt.to_vec();
    let mut spans = Vec::with_capacity(grids.len());
    let mut at = 0usize;
    for (grid, digest) in grids.iter().zip(digests) {
        let start = (at..prompt.len()).find(|p| pads.matches(prompt[*p]))?;
        let end = start.checked_add(usize::try_from(grid.pad_count()).ok()?)?;
        if !prompt.get(start..end)?.iter().all(|t| pads.matches(*t)) {
            return None;
        }
        for (k, slot) in path[start..end].iter_mut().enumerate() {
            *slot = digest.key_id(u32::try_from(k).ok()?);
        }
        spans.push(MediaSpan::new(u32::try_from(start).ok()?, *grid));
        at = end;
    }
    if prompt[at..].iter().any(|t| pads.matches(*t)) {
        return None;
    }
    Some((path, spans))
}

/// Message boundaries of a prompt worth a checkpoint of their own: the end of
/// its first message and the start of its last.
///
/// A boundary is the position just past an end-of-turn token. The first one is
/// where a system prompt ends - shared by every session of an agent, and worth
/// a checkpoint on first sight. The start of the final message is what a
/// regenerated or edited last message branches from. The boundary just before
/// the generation prompt is left out: the prompt's own end is a handful of
/// tokens later and is checkpointed anyway, so a stop there would buy a chunk
/// split and a snapshot and no reuse. Every boundary in between is either one
/// of these for an earlier request or covered by the turn-end checkpoints a
/// conversation already leaves.
fn message_hints(prompt: &[TokenId], stop: StopTokens) -> Vec<u32> {
    let len = prompt.len();
    let mut boundaries: Vec<u32> = prompt
        .iter()
        .enumerate()
        .filter(|(at, token)| stop.contains(**token) && at + 1 < len)
        .map(|(at, _)| u32::try_from(at + 1).unwrap_or(u32::MAX))
        .collect();
    boundaries.pop();
    let first = boundaries.first().copied();
    let last = boundaries
        .last()
        .copied()
        .filter(|last| Some(*last) != first);
    first.into_iter().chain(last).collect()
}

/// Hands a sequence the first token of its turn and sends what that produced.
///
/// A free function rather than a method because it is called with a sequence
/// that is in the slot table and with one that has not been admitted yet - a
/// resume can hand a turn its first token before it has a slot - and a method
/// would have to borrow the coordinator whole in both cases.
fn start_decoding(
    row: &[u8],
    scores: &mut [f32],
    vocabulary: Option<&Vocabulary>,
    stop: StopTokens,
    now: Instant,
    stats: &SchedulerStats,
    slot: &mut SeqSlot,
) {
    let token = slot.choose(fill(row, scores));
    let events = slot.accept(token, vocabulary, stop, now);
    send_all(stats, slot, events);
}

/// Reads the labels off a scoring prompt's last row and sends its answer.
///
/// The scoring half of [`Coordinator::prefill_step`]'s terminus, beside
/// [`start_decoding`] rather than inline: the two arms of that terminus share
/// only the row, and a sequence that generates and a sequence that scores end
/// their prompt in opposite ways - one stands on a sampled token, the other
/// is over. The `fill` borrow ends with the log-probability vector, so
/// nothing borrowed is alive across the send.
///
/// This send is where the reply-exactly-once invariant opens: `Ok` leaves
/// here, and the retirement path answers only a reply this one left behind
/// (see [`Coordinator::retire`]). A failed send is a caller that hung up
/// mid-prefill; the sequence still retires and frees its session either way.
fn finish_scoring(row: &[u8], scores: &mut [f32], slot: &mut SeqSlot) {
    let label_logprobs = decision::math::label_logprobs(fill(row, scores), slot.score_labels());
    slot.finish_score();
    if let Some(reply) = slot.take_score_reply() {
        let _ = reply.send(Ok(ScoreOutcome { label_logprobs }));
    }
}

/// One prefill quantum as [`Coordinator::ready_step`] hands it over: the
/// step as it will run, and the session, prompt and non-empty window of it
/// that it reads. `prompt_window` clamps the window to the prompt, so
/// indexing with it cannot go out of bounds.
struct Quantum {
    step: PrefillStep,
    session: SessionId,
    prompt: Arc<[TokenId]>,
    window: std::ops::Range<usize>,
}

/// Reads one logits row out of its bytes into `scores`.
fn fill<'a>(row: &[u8], scores: &'a mut [f32]) -> &'a mut [f32] {
    for (slot, bytes) in scores.iter_mut().zip(row.as_chunks::<4>().0) {
        *slot = f32::from_le_bytes(*bytes);
    }
    scores
}
