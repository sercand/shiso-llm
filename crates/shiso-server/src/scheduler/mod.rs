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
pub mod queue;
mod stats;
pub mod stop;

use std::collections::VecDeque;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use shiso_core::{
    Backend, CoreError, DecodeStep, Model, PadTokens, SamplingParams, Seed, SessionId, StopTokens,
    TokenId, VisionPrefill,
};
use shiso_engine::Reason;
use shiso_engine::ckpt::Checkpointed;
use shiso_ir::{EventStream, StopReason, StreamEvent};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::tokenizer::Vocabulary;
use crate::vision::{EncodedMedia, VisionTower};
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
pub use queue::{EngineGone, EngineHandle, EngineRequest, GenerateRequest, QUEUE_CAPACITY};
pub use stats::SchedulerStats;
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
    /// Flex requests waiting for standard work to drain, oldest first.
    ///
    /// Unbounded, by decision: a flex caller asked to wait rather than to be
    /// refused, and each entry is a prompt behind an `Arc` and a channel. The
    /// bound on how much can be *queued* is still the channel's; this is only
    /// what has come off it.
    held_flex: VecDeque<HeldFlex>,
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
        None,
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
    reason = "the scheduler takes the device, the model, the vision tower, the vocabulary, the identity, the context size, the runtime config and the stream parsers"
)]
pub fn spawn_engine_with_checkpoints(
    backend: Arc<dyn Backend>,
    model: Box<dyn Checkpointed>,
    vision: Option<VisionTower>,
    vocabulary: Option<Arc<Vocabulary>>,
    model_id: &str,
    ctx: u32,
    cfg: SchedulerConfig,
    parsers: (ReasoningScheme, ToolCallScheme),
) -> (EngineHandle, JoinHandle<Result<(), EngineFailure>>) {
    start(
        backend,
        EngineModel::Checkpointed(model),
        vision,
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
    reason = "the scheduler takes the device, the model, the vision tower, the vocabulary, the identity, the context size, the runtime config and the stream parsers"
)]
fn start(
    backend: Arc<dyn Backend>,
    mut model: EngineModel,
    vision: Option<VisionTower>,
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
    let pads = match (vision.as_ref(), vocabulary.as_ref()) {
        (Some(tower), Some(vocabulary)) => vocabulary.media_ids(&tower.placement()),
        _ => PadTokens::default(),
    };
    let coordinator = Coordinator {
        backend,
        model,
        vision,
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
        closing: false,
        held_flex: VecDeque::new(),
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
    /// flex request that cannot run yet, a place in the held queue.
    fn admit(&mut self, request: EngineRequest) {
        let EngineRequest::Generate { request, reply } = request;
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
        reply: tokio::sync::oneshot::Sender<Result<EventStream, crate::error::AppError>>,
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

    /// Gives a request whose stream is answered a session and a slot.
    fn open(&mut self, request: GenerateRequest, sink: mpsc::Sender<StreamEvent>) {
        // The tower runs before a session is opened, because everything it can
        // refuse is cheaper to refuse with nothing standing on it. A request
        // whose pictures cannot be encoded is FAILED rather than served from its
        // words: the prompt still carries the placeholder run, so serving it
        // would run the model over a few hundred embedding rows that stand for
        // nothing and hand back fluent prose about a picture nobody looked at.
        let media = match self.encode_media(&request) {
            Ok(media) => media,
            Err(message) => {
                tracing::warn!(%message, "a request carrying media could not be served");
                send_event(&sink, StreamEvent::Failed { message });
                return;
            }
        };

        let session = match self.model.as_model_mut().open_session() {
            Ok(session) => session,
            Err(error) => {
                tracing::warn!(error = %error.chain(), "a session could not be opened");
                send_event(
                    &sink,
                    StreamEvent::Failed {
                        message: error.to_string(),
                    },
                );
                return;
            }
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
        let mut slot = SeqSlot::new(
            request,
            SlotSetup {
                id,
                session,
                sink,
                started_at: now,
                deadline: self.cfg.request_timeout.map(|window| now + window),
                budget,
            },
            media,
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
        self.try_resume(&mut slot);
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

    /// Runs the tower over a request's media, or explains why it cannot.
    ///
    /// `Ok(None)` is a text-only request: nothing was attached, so nothing was
    /// encoded, and the sequence takes the launches it took before any of this
    /// existed. The `Err` is the message the caller's stream fails with, written
    /// for whoever sent the request and naming the flag an operator would change.
    fn encode_media(&mut self, request: &GenerateRequest) -> Result<Option<EncodedMedia>, String> {
        if request.media.is_empty() {
            return Ok(None);
        }
        if self.model.as_model_mut().media_mut().is_none() {
            return Err(
                "this checkpoint has no media path, so the attached media cannot be \
                 looked at"
                    .to_owned(),
            );
        }
        let Some(tower) = self.vision.as_ref() else {
            return Err(
                "this server was started without a vision tower, so the attached media \
                 cannot be looked at; start it with --vision naming the tower"
                    .to_owned(),
            );
        };
        let started = Instant::now();
        let encoded = tower
            .encode(&self.backend, &request.media)
            .map_err(|error| error.to_string())?;
        tracing::debug!(
            items = request.media.len(),
            rows = encoded.pad_total(),
            ms = started.elapsed().as_secs_f64() * 1000.0,
            "a request's media was encoded"
        );
        Ok(Some(encoded))
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
        let plan = batching::plan(&self.slots.facts(), &self.cfg);
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
        if let Some(step) = plan.prefill {
            self.prefill_step(step);
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
            self.speculative_batch_step(rows);
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

    /// Whether this row's next step can be speculative.
    fn speculative(&mut self, at: usize) -> bool {
        let greedy = self
            .slots
            .slots()
            .get(at)
            .is_some_and(|slot| slot.sampling().is_greedy());
        greedy && self.model.as_model_mut().spec_mut().is_some()
    }

    /// One speculative step over one sequence.
    fn speculative_step(&mut self, at: usize) {
        let Some(step) = self.slots.slots().get(at).and_then(SeqSlot::decode_step) else {
            return;
        };
        let produced = {
            let Some(spec) = self.model.as_model_mut().spec_mut() else {
                return;
            };
            match spec.speculate(step.session, step.token) {
                Ok(step) => {
                    self.stats.speculated(step.accepted, step.drafted);
                    step.produced
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
                let (params, seed) = slot.draw();
                sampled.push((entry.slot, sample(row, scores, &params, seed)));
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
        let Some(slot) = self.slots.slots().get(step.slot) else {
            return;
        };
        let session = slot.session;
        let prompt = Arc::clone(&slot.prompt);
        let (start, end) = slot.prompt_window(step.tokens);
        let Some(chunk) = prompt.get(start..end) else {
            return;
        };
        let covered = chunk.len();
        if covered == 0 {
            return;
        }

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
            start_decoding(row, scores, vocabulary.as_deref(), *stop, now, stats, slot);
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
                !slot.is_done()
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
            self.store_on_retire(&slot, ending.as_ref());
            self.remember_turn(&slot);
            tracing::debug!(
                slot = slot.id,
                emitted = slot.generated().len(),
                cached = slot.cached_tokens(),
                ?ending,
                "a sequence retired"
            );
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
    let token = {
        let (params, seed) = slot.draw();
        sample(row, scores, &params, seed)
    };
    let events = slot.accept(token, vocabulary, stop, now);
    send_all(stats, slot, events);
}

/// Reads one logits row out of its bytes and picks a token from it.
fn sample(row: &[u8], scores: &mut [f32], params: &SamplingParams, seed: &mut Seed) -> TokenId {
    for (slot, bytes) in scores.iter_mut().zip(row.as_chunks::<4>().0) {
        *slot = f32::from_le_bytes(*bytes);
    }
    shiso_engine::sample_logits(scores, params, seed)
}
