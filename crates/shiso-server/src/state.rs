//! What every handler is given.
//!
//! # One model per process, bound once
//!
//! The state is a plain struct behind an `Arc` and is bound to the router with
//! `with_state` exactly once. There is deliberately no indirection for swapping
//! the model while the process runs: this binary serves one checkpoint, chosen
//! on the command line, and a hot-swap slot would put a lock and a "which
//! model is loaded right now" question into every handler in exchange for a
//! capability nothing here offers. Restarting the process is the swap.
//!
//! Anyone tempted to add that indirection later should add a second binary or a
//! supervisor instead; this note is here so the absence reads as a decision
//! rather than as an oversight.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::auth::AuthConfig;
use crate::response_store::ResponseStore;
use crate::scheduler::{EngineHandle, SchedulerStats};
use crate::tokenizer::Vocabulary;
use crate::tokenizer::render_prompt::PromptRenderer;

/// The checkpoint tier, as the engine opened it.
///
/// A description rather than a handle. The cache itself belongs to the model,
/// on the one thread that owns it, and a handler holding a second reference to
/// it would be a handler that can read device-backed state while a forward pass
/// is writing it. What an operator can be shown without that is where the tier
/// is, what it was allowed to use, and what the scheduler has done with it -
/// which is what [`crate::api::misc_handlers::cache_body`] reports.
#[derive(Debug, Clone)]
pub struct CheckpointTier {
    /// The directory entries are written to.
    pub dir: PathBuf,
    /// The ceiling it was opened with, in bytes.
    pub budget_bytes: u64,
    /// The prefix floor below which nothing is stored.
    pub min_tokens: u32,
}

/// Everything the handlers share.
#[derive(Debug)]
pub struct AppState {
    /// The queue into the engine thread. Handlers never touch the model itself.
    pub engine: EngineHandle,
    /// The name this checkpoint answers to on the wire.
    pub model_id: String,
    /// Tokens one session may hold, as the server was configured.
    pub ctx: u32,
    /// The cache, when this server holds one of its own.
    ///
    /// Not the engine's: the engine's cache lives behind the model on the
    /// engine thread. This is the handle a test or an embedder passes in, and it
    /// is what lets `/cache` report measured numbers at all.
    pub kv: Option<Arc<shiso_kvstore::KvCache>>,
    /// The checkpoint tier the engine opened, when `--kvstore-dir` was given.
    pub ckpt: Option<CheckpointTier>,
    /// The counters the engine thread is filling in.
    pub sched: Arc<SchedulerStats>,
    /// The bearer tokens that may call the inference routes, when the server
    /// was given any. `None` is an open server.
    pub auth: Option<AuthConfig>,
    /// The vocabulary, for the edge that has to render and encode a prompt.
    pub vocabulary: Option<Arc<Vocabulary>>,
    /// The chat template and vocabulary, bound together.
    ///
    /// `None` when the checkpoint shipped no chat template the server could
    /// find. Every generation route needs one, so they answer 503 rather than
    /// inventing a prompt format; everything else still serves.
    pub renderer: Option<Arc<PromptRenderer>>,
    /// The turns a caller can continue or fetch again.
    ///
    /// Always present, because an empty store and an absent one answer every
    /// question the same way and only one of them needs a branch at each use.
    pub response_store: Arc<ResponseStore>,
    /// When the process started serving, for the uptime metric.
    pub started_at: Instant,
}

impl AppState {
    /// The state for a server that has a model and no disk tier.
    #[must_use]
    pub fn new(engine: EngineHandle, model_id: String, ctx: u32) -> Self {
        Self {
            sched: Arc::clone(engine.stats()),
            engine,
            model_id,
            ctx,
            kv: None,
            ckpt: None,
            auth: None,
            vocabulary: None,
            renderer: None,
            response_store: ResponseStore::from_env(),
            started_at: Instant::now(),
        }
    }

    /// The same state with the disk tier attached.
    #[must_use]
    pub fn with_cache(mut self, kv: Arc<shiso_kvstore::KvCache>) -> Self {
        self.kv = Some(kv);
        self
    }

    /// The same state with the checkpoint tier described.
    #[must_use]
    pub fn with_checkpoint_tier(mut self, tier: CheckpointTier) -> Self {
        self.ckpt = Some(tier);
        self
    }

    /// The same state with the auth gate closed.
    #[must_use]
    pub fn with_auth(mut self, auth: AuthConfig) -> Self {
        self.auth = Some(auth);
        self
    }

    /// The same state with a vocabulary attached.
    #[must_use]
    pub fn with_vocabulary(mut self, vocabulary: Arc<Vocabulary>) -> Self {
        self.vocabulary = Some(vocabulary);
        self
    }

    /// The same state with a prompt renderer attached.
    #[must_use]
    pub fn with_renderer(mut self, renderer: Arc<PromptRenderer>) -> Self {
        self.renderer = Some(renderer);
        self
    }

    /// The same state with a particular response store.
    ///
    /// The bound and the expiry are read from the environment by default; this
    /// is how a test pins them instead of setting a process-wide variable that
    /// every other concurrently running test would also see.
    #[must_use]
    pub fn with_response_store(mut self, store: Arc<ResponseStore>) -> Self {
        self.response_store = store;
        self
    }

    /// Seconds this process has been serving.
    #[must_use]
    pub fn uptime_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }
}
