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

use crate::error::AppError;
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

/// Where `/metrics` and `/debug/memory` read the memory ledger from.
///
/// The backend rather than the model: the ledger is a set of atomics the
/// backend owns and a device query, neither of which touches anything the
/// engine thread is writing. Its probes take the cache's own lock for the
/// length of one statistics read, which is what `/cache` would cost too.
#[derive(Debug, Clone)]
pub struct MemorySource {
    backend: Arc<dyn shiso_core::Backend>,
}

impl MemorySource {
    /// A source over `backend`'s ledger; `None` when it keeps none.
    #[must_use]
    pub fn new(backend: Arc<dyn shiso_core::Backend>) -> Option<Self> {
        backend.ledger()?;
        Some(Self { backend })
    }

    /// Everything the ledger knows now, with the device's figure now.
    #[must_use]
    pub fn report(&self) -> Option<shiso_core::LedgerReport> {
        let ledger = self.backend.ledger()?;
        Some(ledger.report(self.backend.memory_info()))
    }
}

/// Everything the handlers share.
#[derive(Debug)]
pub struct AppState {
    /// The queue into the engine thread, when a language model is loaded.
    /// Handlers never touch the model itself; they reach this through
    /// [`AppState::engine`], which answers the typed "no model" error for a
    /// server that serves transcription alone.
    pub engine: Option<EngineHandle>,
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
    /// The sampling a caller who names nothing gets: the checkpoint's
    /// generation config, or neutral when it ships none.
    pub sampling_preset: shiso_core::SamplingParams,
    /// The memory ledger, when the backend keeps one.
    pub memory: Option<MemorySource>,
    /// The transcription service, when a recognizer is loaded.
    pub audio: Option<crate::audio::AudioService>,
    /// The image generation service, when a generator is loaded.
    ///
    /// Handlers never touch the generator itself; they reach this through
    /// [`AppState::image`], which answers the typed "no image model" error
    /// for a server that serves language or transcription alone.
    pub image: Option<crate::image::ImageService>,
}

impl AppState {
    /// The state for a server that has a model and no disk tier.
    #[must_use]
    pub fn new(engine: EngineHandle, model_id: String, ctx: u32) -> Self {
        Self {
            sched: Arc::clone(engine.stats()),
            engine: Some(engine),
            model_id,
            ctx,
            kv: None,
            ckpt: None,
            vocabulary: None,
            renderer: None,
            response_store: ResponseStore::from_env(),
            started_at: Instant::now(),
            sampling_preset: shiso_core::SamplingParams::neutral(),
            memory: None,
            audio: None,
            image: None,
        }
    }

    /// The state for a server with no language model: every generation route
    /// answers [`AppError::NoModel`], and everything else serves.
    #[must_use]
    pub fn without_language_model() -> Self {
        Self {
            sched: Arc::new(SchedulerStats::new()),
            engine: None,
            model_id: String::new(),
            ctx: 0,
            kv: None,
            ckpt: None,
            vocabulary: None,
            renderer: None,
            response_store: ResponseStore::from_env(),
            started_at: Instant::now(),
            sampling_preset: shiso_core::SamplingParams::neutral(),
            memory: None,
            audio: None,
            image: None,
        }
    }

    /// The same state with the transcription service attached.
    #[must_use]
    pub fn with_audio(mut self, audio: crate::audio::AudioService) -> Self {
        self.audio = Some(audio);
        self
    }

    /// The same state with the image generation service attached.
    #[must_use]
    pub fn with_image(mut self, image: crate::image::ImageService) -> Self {
        self.image = Some(image);
        self
    }

    /// The image queue.
    ///
    /// # Errors
    ///
    /// [`AppError::NoImageModel`] when no image generator is loaded.
    pub fn image(&self) -> Result<&crate::image::ImageService, AppError> {
        self.image.as_ref().ok_or(AppError::NoImageModel)
    }

    /// The engine queue.
    ///
    /// # Errors
    ///
    /// [`AppError::NoModel`] when no language model is loaded.
    pub fn engine(&self) -> Result<&EngineHandle, AppError> {
        self.engine.as_ref().ok_or(AppError::NoModel)
    }

    /// The same state with the memory ledger attached.
    #[must_use]
    pub fn with_memory(mut self, memory: MemorySource) -> Self {
        self.memory = Some(memory);
        self
    }

    /// The same state with a sampling preset other than neutral.
    #[must_use]
    pub const fn with_sampling_preset(mut self, preset: shiso_core::SamplingParams) -> Self {
        self.sampling_preset = preset;
        self
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
