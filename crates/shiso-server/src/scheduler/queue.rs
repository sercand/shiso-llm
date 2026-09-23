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
//! # Why admission is bounded, and what a full queue answers
//!
//! The channel has a capacity, so a server under more load than it can carry
//! has to do something. Blocking the caller until a slot frees would turn
//! overload into unbounded latency, which reaches a client as a timeout it
//! cannot distinguish from a hung server. So the wait is bounded by
//! [`super::SchedulerConfig::queue_admission`] and what comes back after it is
//! a refusal that says the server is overloaded - a status a client library
//! already knows how to back off from.

use std::sync::Arc;
use std::time::Duration;

use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_ir::EventStream;
use shiso_vision::VisionItem;
use tokio::sync::{mpsc, oneshot};

use crate::error::AppError;
use crate::scheduler::stats::SchedulerStats;

/// Requests the queue holds when nothing said otherwise.
///
/// Kept as a named constant because the fixed-capacity entry point - the one a
/// harness uses, which takes no configuration - still needs a number, and a
/// literal at that call site would be a capacity with no name to argue about.
pub const QUEUE_CAPACITY: usize = 256;

/// One unit of work for the engine thread.
///
/// An enum with one variant. It stays an enum because the channel is the only
/// path into the coordinator and a second kind of message - a cache
/// administration request, say - would otherwise mean rewriting every call
/// site. Cancellation is deliberately *not* such a message: a caller cancels by
/// dropping the stream, which the coordinator observes on its next send, so a
/// cancel variant would be a second way to say something the first way already
/// says unambiguously.
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
}

impl std::fmt::Debug for EngineRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generate { request, .. } => f
                .debug_struct("Generate")
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
    /// The rendered prompt as text, which is what the checkpoint tier keys on.
    ///
    /// Empty when the surface had no rendered text to offer, and that is a
    /// cold start rather than an error: an entry keyed by nothing could never be
    /// named by a later request. Deriving it here by decoding the ids would
    /// produce a string that is *nearly* the prompt, and a cache keyed on nearly
    /// the prompt occasionally answers with somebody else's conversation.
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
    pub media: Arc<[VisionItem]>,
    /// Which declared tool parameters are strings, so the tool-call reader keeps
    /// them as strings. Empty for a request with no tools.
    ///
    /// The XML call body carries no types; without this the reader guesses
    /// from the text, and a `path` of `123` or file content that parses as JSON
    /// reaches the tool as something it never declared.
    pub string_params: Arc<shiso_parsers::StringParams>,
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
            string_params: Arc::default(),
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

    /// The same generation, with media attached.
    #[must_use]
    pub fn with_media(mut self, media: Arc<[VisionItem]>) -> Self {
        self.media = media;
        self
    }

    /// The same generation, with the request's string-typed tool parameters
    /// attached.
    #[must_use]
    pub fn with_string_params(mut self, strings: Arc<shiso_parsers::StringParams>) -> Self {
        self.string_params = strings;
        self
    }
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
    /// client that backs off and retries will be served. [`AppError::Internal`]
    /// when the engine thread is gone - a closed channel or a dropped reply -
    /// because a request that cannot reach the model at all is a server failure
    /// and not something a retry will fix.
    pub async fn generate(&self, request: GenerateRequest) -> Result<EventStream, AppError> {
        self.stats.request_queued();
        let (reply, answer) = oneshot::channel();
        let send = self.tx.send(EngineRequest::Generate { request, reply });
        match tokio::time::timeout(self.admission, send).await {
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
