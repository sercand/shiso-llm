//! The channel between the handlers and the one thread that owns the model.
//!
//! # Why a channel and not a lock around the model
//!
//! A `Box<dyn Model>` is `Send` and deliberately not `Sync`: exactly one thread
//! owns it, which is what lets its session bookkeeping be plain mutable state
//! instead of a lock. Handing handlers a `Mutex<Box<dyn Model>>` would undo
//! that and, worse, would put an unbounded blocking section inside an async
//! task - a decode that takes a hundred milliseconds is a hundred milliseconds
//! a runtime worker is not polling anything else.
//!
//! So the boundary is a message. A handler sends a request and awaits a reply
//! channel; the engine thread does the blocking work. Nothing in `api/` ever
//! calls a trait method on the model, and there is no way for it to, because it
//! has no reference to one.
//!
//! # Why the reply is a stream, and why it is sent before the first token
//!
//! The reply is an [`EventStream`], not a finished [`shiso_ir::Response`], and
//! the engine answers the oneshot with it *before* generating anything. A
//! streaming client should see the response head as soon as the request is
//! accepted, and a blocking client folds the same stream into a response with
//! [`shiso_ir::ResponseBuilder`] - one production path, two encodings of it,
//! which is what makes a field that only the blocking path fills in impossible.
//!
//! A scoring request is the deliberate exception: it generates nothing to
//! stream, so its reply carries the answer itself, sent when the prompt has
//! been read.
//!
//! # Why admission is bounded, and what a full queue answers
//!
//! The channel has a capacity, so a server under more load than it can carry
//! has to do something. Blocking the caller until a slot frees would turn
//! overload into unbounded latency, which reaches a client as a timeout it
//! cannot distinguish from a hung server. So the wait is bounded by
//! [`super::SchedulerConfig::queue_admission`] and what comes back after it is
//! a refusal that says the server is overloaded - a status a client library
//! already knows how to back off from.
//!
//! A flex request is the exception. What its caller bought is capacity nobody
//! else is using, at whatever latency that costs, so it waits for room in the
//! queue for as long as it takes and is never told the server is overloaded.
//! Its caller still cancels by going away, which drops the wait. A scoring
//! request is the exception to that exception: its wait is bounded in every
//! class, because what waits is an answer rather than a receipt, and a held
//! score buys its caller nothing (see [`EngineHandle::score`]).

use std::sync::Arc;
use std::time::Duration;

use shiso_core::{DecisionQuestion, SamplingParams, Seed, TokenId};
use shiso_ir::{EventStream, ServiceClass};
use tokio::sync::{mpsc, oneshot};

use crate::error::AppError;
use crate::media::MediaItem;
use crate::scheduler::stats::SchedulerStats;
use shiso_core::media::MediaDigest;

/// Requests the queue holds when nothing said otherwise.
///
/// Kept as a named constant because the fixed-capacity entry point - the one a
/// harness uses, which takes no configuration - still needs a number, and a
/// literal at that call site would be a capacity with no name to argue about.
pub const QUEUE_CAPACITY: usize = 256;

/// One unit of work for the engine thread.
///
/// It started as an enum with one variant and stays an enum because the
/// channel is the only path into the coordinator: a second kind of message -
/// scoring a prompt, and later a cache administration request - would
/// otherwise mean rewriting every call site. Cancellation is deliberately
/// *not* such a message: a caller cancels by dropping the stream, which the
/// coordinator observes on its next send, so a cancel variant would be a
/// second way to say something the first way already says unambiguously.
///
/// `Debug` is written out rather than derived: the reply channel carries an
/// [`EventStream`], which is a boxed trait object with no `Debug` of its own,
/// and a message whose `Debug` shows the request but not the channel is more
/// useful in a log than no `Debug` at all.
#[non_exhaustive]
pub enum EngineRequest {
    /// Generate a turn.
    Generate {
        /// What to generate.
        request: GenerateRequest,
        /// Where the stream is handed back.
        ///
        /// Paired with the request by [`EngineHandle::generate`] rather than
        /// carried inside it, so no caller can construct a request whose reply
        /// channel belongs to a different request.
        reply: oneshot::Sender<Result<EventStream, AppError>>,
    },
    /// Score a prompt against a set of labels: one prefill, nothing generated.
    ///
    /// Unlike [`Self::Generate`], the reply carries the *answer* rather than a
    /// receipt of admission: it is sent when the prompt has been read to its
    /// end, with the labels' log-probabilities, or with an error when the
    /// sequence ends any other way. A scoring sequence has no stream, so the
    /// reply is the one place its outcome can be said.
    Score {
        /// What to score.
        request: ScoreRequest,
        /// Where the answer goes.
        ///
        /// Paired with the request by [`EngineHandle::score`], for the same
        /// reason [`GenerateRequest`] replies are.
        reply: oneshot::Sender<Result<ScoreOutcome, AppError>>,
    },
}

impl std::fmt::Debug for EngineRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generate { request, .. } => f
                .debug_struct("Generate")
                .field("request", request)
                .finish_non_exhaustive(),
            Self::Score { request, .. } => f
                .debug_struct("Score")
                .field("request", request)
                .finish_non_exhaustive(),
        }
    }
}

/// What one generation needs.
#[derive(Debug)]
#[non_exhaustive]
pub struct GenerateRequest {
    /// The rendered prompt, already tokenized.
    ///
    /// `Arc<[T]>` rather than `Arc<Vec<T>>`: one indirection instead of two,
    /// and the length is part of the pointer rather than behind it.
    pub prompt_tokens: Arc<[TokenId]>,
    /// The rendered prompt as text, which the stream parser reads to tell
    /// whether the template already opened a reasoning block.
    ///
    /// Empty when the surface had no rendered text to offer. The checkpoint
    /// tier does not read it: it keys by `prompt_tokens`.
    pub prompt_text: Arc<str>,
    /// Resolved sampling settings. The request's own preferences were turned
    /// into these at the edge, because only the edge knows the server's preset.
    pub sampling: SamplingParams,
    /// The PRNG stream this generation draws from.
    pub seed: Seed,
    /// The token budget for the generated turn.
    pub max_tokens: u32,
    /// Sequences that end the turn when the model produces them.
    ///
    /// Honoured here rather than by an encoder, because a sequence can straddle
    /// two tokens and a streaming encoder has already sent the first half by the
    /// time the second arrives. One implementation, in the one place that sees
    /// the text before anyone else does.
    pub stop: Arc<[String]>,
    /// Pre-expanded media items, in the order the prompt refers to them.
    ///
    /// Carried as `Arc<[_]>` because the payload is megabytes of `f32` and the
    /// request crosses a thread boundary; cloning it to cross would double the
    /// largest allocation in the request path.
    pub media: Arc<[MediaItem]>,
    /// One digest per item of `media`, in the same order: what the prefix
    /// cache files a media prompt's placeholder runs under.
    ///
    /// Computed where the request is built rather than on the engine thread,
    /// because hashing a picture's rows is megabytes of work and that thread
    /// is every other sequence's decode loop.
    pub media_digests: Arc<[MediaDigest]>,
    /// Which declared tool parameters are strings, so the tool-call reader keeps
    /// them as strings. Empty for a request with no tools.
    ///
    /// The XML call body carries no types; without this the reader guesses
    /// from the text, and a `path` of `123` or file content that parses as JSON
    /// reaches the tool as something it never declared.
    pub string_params: Arc<shiso_parsers::StringParams>,
    /// The declared tools' names and parameters, from which a speculative
    /// step proposes a tool call's fixed text. Empty for a request with no
    /// tools.
    pub tool_scaffold: Arc<shiso_parsers::ToolScaffold>,
    /// Whether this generation runs beside standard work or only in its
    /// absence. See [`ServiceClass::Flex`] for what the scheduler does with it.
    pub service_class: ServiceClass,
}

impl GenerateRequest {
    /// A generation of `prompt_tokens` with no media.
    #[must_use]
    pub fn new(
        prompt_tokens: Arc<[TokenId]>,
        sampling: SamplingParams,
        seed: Seed,
        max_tokens: u32,
    ) -> Self {
        Self {
            prompt_tokens,
            prompt_text: Arc::from(""),
            sampling,
            seed,
            max_tokens,
            stop: Arc::from(Vec::new()),
            media: Arc::from(Vec::new()),
            media_digests: Arc::from(Vec::new()),
            string_params: Arc::default(),
            tool_scaffold: Arc::default(),
            service_class: ServiceClass::Standard,
        }
    }

    /// The same generation, with the rendered prompt text attached.
    #[must_use]
    pub fn with_text(mut self, text: Arc<str>) -> Self {
        self.prompt_text = text;
        self
    }

    /// The same generation, with stop sequences attached.
    #[must_use]
    pub fn with_stop(mut self, stop: Arc<[String]>) -> Self {
        self.stop = stop;
        self
    }

    /// The same generation, with media attached, each item beside its digest.
    ///
    /// The digests come from whoever decoded the items, which already hashed
    /// each one once and remembered it. A count that does not match is kept
    /// rather than refused, and it is safe: the scheduler then files the
    /// request under no key, as though it carried no digests at all.
    #[must_use]
    pub fn with_media(mut self, media: Arc<[MediaItem]>, digests: Arc<[MediaDigest]>) -> Self {
        self.media = media;
        self.media_digests = digests;
        self
    }

    /// The same generation, with the request's string-typed tool parameters
    /// attached.
    #[must_use]
    pub fn with_string_params(mut self, strings: Arc<shiso_parsers::StringParams>) -> Self {
        self.string_params = strings;
        self
    }

    /// The same generation, with the declared tools' scaffold attached.
    #[must_use]
    pub fn with_tool_scaffold(mut self, scaffold: Arc<shiso_parsers::ToolScaffold>) -> Self {
        self.tool_scaffold = scaffold;
        self
    }

    /// The same generation, served in `class`.
    #[must_use]
    pub const fn with_service_class(mut self, class: ServiceClass) -> Self {
        self.service_class = class;
        self
    }

    /// Whether this generation is served only when no standard one has work.
    #[must_use]
    pub const fn is_flex(&self) -> bool {
        matches!(self.service_class, ServiceClass::Flex)
    }
}

/// What one scoring pass needs.
///
/// The pass reads `prompt_tokens` once - a prefill, no decode steps - and
/// answers with the log-probability of each of `labels` at the prompt's last
/// position. Nothing is generated: the labels are scored, never sampled, and
/// a scoring sequence never speaks on a stream.
#[derive(Debug)]
#[non_exhaustive]
pub struct ScoreRequest {
    /// The rendered prompt, already tokenized.
    ///
    /// `Arc<[T]>` for the same reason [`GenerateRequest::prompt_tokens`] is:
    /// one indirection, and the length beside the pointer.
    pub prompt_tokens: Arc<[TokenId]>,
    /// The label ids to read from the prompt's last row. The answer is one
    /// log-probability per entry, in this order.
    pub labels: Arc<[TokenId]>,
    /// Whether this score runs beside standard work or only in its absence.
    /// See [`ServiceClass::Flex`] for what the scheduler does with it; a flex
    /// score is still answered through its reply, never held for one.
    pub service_class: ServiceClass,
    /// Pre-expanded media items, in the order the prompt refers to them.
    ///
    /// The same names [`GenerateRequest::media`] carries for the same content:
    /// a scoring prompt can carry a picture, and the pad runs it was rendered
    /// with must be spliced before the one prefill reads them.
    ///
    /// [`GenerateRequest::media`]: GenerateRequest::media
    pub media: Arc<[MediaItem]>,
    /// One digest per item of `media`, in the same order.
    ///
    /// A score is never keyed on the prefix cache, so these ride along for
    /// the media cache lookup in `encode_media` and nothing else.
    pub media_digests: Arc<[MediaDigest]>,
    /// How many leading tokens this prompt shares with the other questions
    /// of its request, zero when it shares none worth naming.
    ///
    /// The scheduler checkpoints the prompt there, so the sibling questions
    /// read after it resume from that point and read only their own tails.
    pub shared_prefix: u32,
    /// The schema a decision head scores, when this pass is answered by the
    /// model's head rather than by label rows: spans of `prompt_tokens`, one
    /// per question and one per option. `labels` is then empty.
    pub decision: Option<Arc<[DecisionQuestion]>>,
}

impl ScoreRequest {
    /// A scoring pass over `prompt_tokens` reading `labels`, standard class,
    /// no media.
    #[must_use]
    pub fn new(prompt_tokens: Arc<[TokenId]>, labels: Arc<[TokenId]>) -> Self {
        Self {
            prompt_tokens,
            labels,
            service_class: ServiceClass::Standard,
            media: Arc::from(Vec::new()),
            media_digests: Arc::from(Vec::new()),
            shared_prefix: 0,
            decision: None,
        }
    }

    /// A pass over `prompt_tokens` answered by the model's decision head over
    /// `questions`, standard class, no media.
    #[must_use]
    pub fn decision(prompt_tokens: Arc<[TokenId]>, questions: Arc<[DecisionQuestion]>) -> Self {
        Self {
            decision: Some(questions),
            ..Self::new(prompt_tokens, Arc::from(Vec::new()))
        }
    }

    /// The same scoring request, sharing its first `tokens` with the other
    /// questions of its request.
    #[must_use]
    pub const fn with_shared_prefix(mut self, tokens: u32) -> Self {
        self.shared_prefix = tokens;
        self
    }

    /// The same scoring request with media attached, each item beside its
    /// digest.
    #[must_use]
    pub fn with_media(mut self, media: Arc<[MediaItem]>, digests: Arc<[MediaDigest]>) -> Self {
        self.media = media;
        self.media_digests = digests;
        self
    }

    /// The same scoring pass, served in `class`.
    #[must_use]
    pub const fn with_service_class(mut self, class: ServiceClass) -> Self {
        self.service_class = class;
        self
    }

    /// Whether this score is served only when no standard work has the device.
    #[must_use]
    pub const fn is_flex(&self) -> bool {
        matches!(self.service_class, ServiceClass::Flex)
    }
}

/// The answer to one [`ScoreRequest`].
#[derive(Debug)]
pub struct ScoreOutcome {
    /// The log-probability of each label, in the order the request gave them.
    /// Empty for a decision pass.
    pub label_logprobs: Vec<f64>,
    /// One logit per option, per question, for a decision pass; empty for a
    /// label pass.
    pub option_logits: Vec<Vec<f32>>,
    /// Prompt tokens a stored prefix covered rather than this pass reading
    /// them: a sibling question's checkpoint, or an earlier request's.
    pub cached_tokens: u64,
}

/// The handlers' end of the boundary.
///
/// Cheap to clone - a channel sender, two snapshot fields and two `Arc`s - so it
/// lives inside the shared application state and every handler holds the same
/// queue.
#[derive(Debug, Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<EngineRequest>,
    model_id: Arc<str>,
    ctx: u32,
    max_num_seqs: usize,
    admission: Duration,
    stats: Arc<SchedulerStats>,
}

impl EngineHandle {
    /// Builds a handle over an existing sender.
    pub(crate) fn new(
        tx: mpsc::Sender<EngineRequest>,
        model_id: Arc<str>,
        ctx: u32,
        max_num_seqs: usize,
        admission: Duration,
        stats: Arc<SchedulerStats>,
    ) -> Self {
        Self {
            tx,
            model_id,
            ctx,
            max_num_seqs,
            admission,
            stats,
        }
    }

    /// The model this queue serves, for a handler that has to name it.
    ///
    /// A snapshot rather than a round trip: the name does not change while the
    /// process runs, and a `/v1/models` request should not have to queue behind
    /// a decode to learn it.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Tokens one session may hold, as the server was configured.
    #[must_use]
    pub const fn ctx(&self) -> u32 {
        self.ctx
    }

    /// Sequences the coordinator behind this queue holds at once.
    ///
    /// A snapshot for the same reason the model name is one: it does not change
    /// while the process runs, and an operator endpoint asking for it should not
    /// have to queue behind a decode.
    #[must_use]
    pub const fn max_num_seqs(&self) -> usize {
        self.max_num_seqs
    }

    /// The counters the coordinator behind this queue is filling in.
    #[must_use]
    pub fn stats(&self) -> &Arc<SchedulerStats> {
        &self.stats
    }

    /// Queues a generation and waits for its stream.
    ///
    /// Returns once the engine has accepted the request, which is before the
    /// first token exists. What is awaited here is admission, not completion.
    ///
    /// # Errors
    ///
    /// [`AppError::Overloaded`] when the queue stayed full for the whole
    /// admission window - the server has more work than it can hold, and a
    /// client that backs off and retries will be served. Never for a flex
    /// request, which waits for room instead. [`AppError::Internal`]
    /// when the engine thread is gone - a closed channel or a dropped reply -
    /// because a request that cannot reach the model at all is a server failure
    /// and not something a retry will fix.
    pub async fn generate(&self, request: GenerateRequest) -> Result<EventStream, AppError> {
        self.stats.request_queued();
        let (reply, answer) = oneshot::channel();
        let flex = request.is_flex();
        let send = self.tx.send(EngineRequest::Generate { request, reply });
        let delivered = if flex {
            Ok(send.await)
        } else {
            tokio::time::timeout(self.admission, send).await
        };
        match delivered {
            Ok(Ok(())) => {}
            // The channel closed under us. Nothing was generated, and nothing
            // will be: the thread that would have answered is gone.
            Ok(Err(_closed)) => {
                return Err(AppError::internal("the engine queue is closed", EngineGone));
            }
            Err(_elapsed) => {
                self.stats.request_rejected();
                tracing::warn!(
                    waited_ms = self.admission.as_millis(),
                    "the scheduler queue stayed full for the whole admission window; \
                     refusing the request rather than making the caller wait without a bound"
                );
                return Err(AppError::Overloaded {
                    what: "the scheduler queue is full",
                });
            }
        }
        answer
            .await
            .map_err(|_dropped| AppError::internal("the engine dropped the request", EngineGone))?
    }

    /// Queues a scoring pass and waits for its answer.
    ///
    /// Unlike [`EngineHandle::generate`], what is awaited here is completion,
    /// not admission: the reply is sent when the prompt has been read to its
    /// end, carrying the labels' log-probabilities.
    ///
    /// The wait for room in the queue is bounded by the admission window for
    /// *every* service class, flex included. The held-flex lane exists so a
    /// streaming caller sees its head and keepalives while it waits; a score
    /// has nothing to show while it waits, and its answer is a prefill that
    /// holds a slot for milliseconds - waiting one behind long work would make
    /// the caller pay a generation's latency for a scoring pass's answer.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] when the prompt or the label set is empty -
    /// a scoring pass with nothing to read, or nothing to read *for*, answers
    /// nothing. The HTTP surface refuses these first; the refusal is repeated
    /// here so no in-crate caller can queue a request the engine would have
    /// to invent an answer for. [`AppError::Overloaded`] when the queue stayed
    /// full for the whole admission window. [`AppError::Internal`] when the
    /// engine thread is gone - a closed channel, or a reply dropped with the
    /// sequence that held it.
    pub async fn score(&self, request: ScoreRequest) -> Result<ScoreOutcome, AppError> {
        let unanswerable = match &request.decision {
            Some(questions) => questions.is_empty(),
            None => request.labels.is_empty(),
        };
        if request.prompt_tokens.is_empty() || unanswerable {
            return Err(AppError::bad_request(
                "a scoring request needs a prompt and at least one label",
                None,
                "invalid_request",
            ));
        }
        self.stats.request_queued();
        let (reply, answer) = oneshot::channel();
        let delivered = tokio::time::timeout(
            self.admission,
            self.tx.send(EngineRequest::Score { request, reply }),
        )
        .await;
        match delivered {
            Ok(Ok(())) => {}
            // The channel closed under us. Nothing was read, and nothing will
            // be: the thread that would have answered is gone.
            Ok(Err(_closed)) => {
                return Err(AppError::internal("the engine queue is closed", EngineGone));
            }
            Err(_elapsed) => {
                self.stats.request_rejected();
                tracing::warn!(
                    waited_ms = self.admission.as_millis(),
                    "the scheduler queue stayed full for the whole admission window; \
                     refusing the score rather than making the caller wait without a bound"
                );
                return Err(AppError::Overloaded {
                    what: "the scheduler queue is full",
                });
            }
        }
        answer
            .await
            .map_err(|_dropped| AppError::internal("the engine dropped the request", EngineGone))?
    }
}

/// The engine thread is not there any more.
///
/// A unit error type rather than a string, so the two places that report it
/// cannot drift apart in wording, and so the cause an operator sees in the log
/// is a type they can grep for.
#[derive(Debug, thiserror::Error)]
#[error("the engine thread is not running")]
pub struct EngineGone;

/// Builds the channel the engine thread reads and the handle the router holds.
pub(crate) fn channel(
    model_id: Arc<str>,
    ctx: u32,
    cfg: &super::SchedulerConfig,
    stats: Arc<SchedulerStats>,
) -> (EngineHandle, mpsc::Receiver<EngineRequest>) {
    let (tx, rx) = mpsc::channel(cfg.max_queued.max(1));
    (
        EngineHandle::new(
            tx,
            model_id,
            ctx,
            cfg.max_num_seqs,
            cfg.queue_admission,
            stats,
        ),
        rx,
    )
}
