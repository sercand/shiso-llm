//! [`Runtime`]: the engine, in-process, behind one handle.
//!
//! `shiso serve` boots a model and puts HTTP on top. [`Runtime`] boots the same
//! model through the same functions and puts nothing on top: the caller hands
//! it [`shiso_ir`] requests and reads [`shiso_ir`] responses. It is a facade
//! over the typed cores ([`crate::api::pipeline`], [`crate::image`],
//! [`crate::decision`]) and owns nothing they do not: the engine thread, the
//! image worker and the model-home lock.
//!
//! # Which executor
//!
//! The async methods run on the caller's executor, whatever it is: a tokio
//! task, `futures::executor::block_on`, a UI toolkit's. The cores need two
//! things from tokio that a foreign executor does not provide - a timer for
//! the admission deadline and a handle for `spawn_blocking` - so the runtime
//! owns a one-worker tokio runtime whose only job is to be the context each
//! poll enters. The work itself is on the engine thread and the image worker;
//! the caller's task only awaits channels, which are executor-neutral.
//!
//! # Blocking
//!
//! The `*_blocking` methods park the calling thread. They work from a plain
//! thread and from a `spawn_blocking` thread; from an async worker they return
//! an [`AppError`] with code [`BLOCKING_IN_ASYNC`] instead of stalling the
//! executor. tokio refuses the nested `block_on` with a panic, which is caught
//! and reported the same way (its message still reaches the panic hook).
//!
//! # Sharing and dropping
//!
//! Methods take `&self`, so one `Runtime` serves any number of threads, and
//! it is not `Clone`: wrap it in an `Arc` to share it. It cannot be dropped
//! while a call borrows it, which is why there is no "call in flight at drop".
//! A call whose future was dropped half-way, or a stream whose sink said stop,
//! leaves a request in the engine; [`Drop`] closes the queue and joins the
//! engine thread, which finishes what it holds first. Drop therefore blocks as
//! long as one engine step; do not drop it on an async worker you cannot
//! afford to stall.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_ir::decision::{Choice, ChoiceValue, Input, Question, QuestionKind, ReadoutState};
use shiso_ir::image::{Edit, Generate, Images, Step};
use shiso_ir::{Message, Request, Response, StreamEvent};
use tokio::runtime::Handle;

use crate::api::pipeline::{self, PromptField};
use crate::boot::{self, EngineThread};
use crate::boot_error::BootError;
use crate::error::AppError;
use crate::hub::{self, InstalledModel, PullError};
use crate::options::{ConfigError, ExplicitlySet, ServeOptions};
use crate::state::AppState;

/// The [`AppError::code`] of a `*_blocking` call made from an async worker.
pub const BLOCKING_IN_ASYNC: &str = "blocking_in_async_context";

/// Where a [`Runtime`]'s model comes from.
#[derive(Debug, Clone, Copy)]
pub enum Source<'a> {
    /// A registry reference (`name` or `name:variant`), installed in the model
    /// home, pulled first when [`RuntimeOptions::pull_if_missing`] says so.
    Registry(&'a str),
    /// A directory: an installed variant (it holds a `recipe.json`) or a
    /// checkpoint, as [`hub::resolve_serve_dir`] reads it. Never touches the
    /// network.
    Dir(&'a Path),
}

/// How a [`Runtime`] is opened.
#[derive(Debug, Clone, Default)]
pub struct RuntimeOptions {
    /// The model home. `None` is the default home (`~/.shiso`); a [`Source::Dir`]
    /// does not read it.
    pub home: Option<PathBuf>,
    /// Download a [`Source::Registry`] model that is not installed. Off by
    /// default: opening never reaches the network unless asked to.
    pub pull_if_missing: bool,
    /// The server settings the model boots with: context, sequences, cache
    /// and so on. A registry recipe fills the ones `explicit` does not name.
    pub serve: ServeOptions,
    /// The fields of `serve` the caller set on purpose, which a recipe's
    /// defaults must not replace (`ctx`, `max_num_seqs`, ...).
    pub explicit: ExplicitlySet,
}

/// Why [`Runtime::open`] failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OpenError {
    /// The model is not installed (and pulling was off), or the pull failed.
    #[error(transparent)]
    Pull(#[from] PullError),
    /// The options do not describe something servable.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// The device, a checkpoint or a worker did not come up.
    #[error(transparent)]
    Boot(#[from] BootError),
    /// The model loaded nothing a [`Runtime`] serves: an embedding or
    /// transcription model, or a directory with no language or image model.
    #[error("nothing to run: the model is not a language or image model")]
    NothingLoaded,
    /// The private tokio runtime could not start.
    #[error("could not start the runtime's executor")]
    Executor(#[source] std::io::Error),
    /// `open` was called from an async task, which it would stall.
    #[error("Runtime::open blocks; call it from a thread that may block, not from an async task")]
    BlockingInAsync,
    /// The throwaway pass that warms the device failed.
    #[error("the warm-up pass failed")]
    WarmUp(#[source] AppError),
}

/// Joins the engine thread when dropped, after the queue has closed.
#[derive(Debug)]
struct EngineGuard(Option<EngineThread>);

impl Drop for EngineGuard {
    fn drop(&mut self) {
        match self.0.take().map(std::thread::JoinHandle::join) {
            None | Some(Ok(Ok(()))) => {}
            Some(Ok(Err(error))) => {
                tracing::error!(error = %shiso_core::chain_of(&error), "the engine did not shut down cleanly");
            }
            Some(Err(_panicked)) => tracing::error!("the engine thread panicked"),
        }
    }
}

/// A tokio runtime that is shut down without waiting, so dropping it is legal
/// on any thread (a plain drop panics inside an async context).
#[derive(Debug)]
struct Executor {
    handle: Handle,
    runtime: Option<tokio::runtime::Runtime>,
}

impl Executor {
    fn start() -> std::io::Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("shiso-runtime")
            .enable_all()
            .build()?;
        Ok(Self {
            handle: runtime.handle().clone(),
            runtime: Some(runtime),
        })
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

/// Polls `future` inside `handle`'s context on every poll, so timers and
/// `spawn_blocking` work under any executor, and the future stays in the
/// caller's task (it need not be `'static`, and dropping it cancels it).
async fn within<F: Future>(handle: Handle, future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    std::future::poll_fn(|cx| {
        let _entered = handle.enter();
        future.as_mut().poll(cx)
    })
    .await
}

/// The language model, image model and decision head of one model, in this
/// process. See the [module documentation](self).
#[derive(Debug)]
pub struct Runtime {
    // Declaration order is drop order: the state's last handle closes the
    // engine queue, then the engine thread is joined, then the executor goes,
    // then the model-home lock is released.
    state: Arc<AppState>,
    _engine: EngineGuard,
    executor: Executor,
    _installed: Option<InstalledModel>,
}

impl Runtime {
    /// Opens the model `source` names and warms it.
    ///
    /// A registry model is looked up in the model home (and pulled when
    /// [`RuntimeOptions::pull_if_missing`]), its components and recipe defaults
    /// fill [`RuntimeOptions::serve`], and the model boots through the same
    /// functions `shiso serve` uses. The install's lock is held until the
    /// `Runtime` drops, so `shiso rm` waits for it.
    ///
    /// Then one throwaway pass runs through whatever loaded - a one-question
    /// decision on a decision model, one token on a language model with a chat
    /// template, one small step on an image model - so the device's first-use
    /// cost (kernel and buffer set-up, about 1.5 s on Metal) is paid here and
    /// the first real call is as fast as any other. `open` takes that much
    /// longer. A failed decision or text pass fails the open; a failed image
    /// pass is logged and ignored, because that request is built from the
    /// model's own limits and a refusal of it says nothing about the real ones.
    ///
    /// `progress(done, total)` reports a pull's bytes; returning `false`
    /// cancels it ([`PullError::Cancelled`]). It is not called for a
    /// [`Source::Dir`] or an installed model.
    ///
    /// Blocks while the weights load: call it from a thread that may block.
    ///
    /// # Errors
    ///
    /// [`OpenError::Pull`] for a model that is not installed or cannot be
    /// pulled; [`OpenError::Config`] and [`OpenError::Boot`] for settings or
    /// files that will not boot; [`OpenError::NothingLoaded`];
    /// [`OpenError::BlockingInAsync`]; [`OpenError::WarmUp`].
    pub fn open(
        source: Source<'_>,
        options: RuntimeOptions,
        progress: &mut dyn FnMut(u64, u64) -> bool,
    ) -> Result<Self, OpenError> {
        // Before any weight moves: loading and then failing the warm-up for
        // being on the wrong thread would be the expensive place to find out.
        let executor = Executor::start().map_err(OpenError::Executor)?;
        if try_block_on(&executor.handle, async {}).is_none() {
            return Err(OpenError::BlockingInAsync);
        }
        let RuntimeOptions {
            home,
            pull_if_missing,
            mut serve,
            explicit,
        } = options;
        let installed = match source {
            Source::Dir(dir) => {
                hub::resolve_serve_dir(dir, &mut serve, &explicit)?;
                None
            }
            Source::Registry(name) => {
                let home = home.unwrap_or_else(shiso_hub::HubConfig::default_home);
                let config = hub::config_for_home(&home);
                // The registry's spelling is lowercase; an embedder need not know that.
                let name = name.to_lowercase();
                match hub::resolve_serve_model(&name, &mut serve, &explicit, &config) {
                    Err(PullError::NotInstalled(_)) if pull_if_missing => {
                        // Held across the second resolve so the variant cannot
                        // be removed in between; both locks are shared.
                        let pulled = hub::pull(&name, &config, progress)?;
                        let installed =
                            hub::resolve_serve_model(&name, &mut serve, &explicit, &config)?;
                        drop(pulled);
                        installed
                    }
                    other => other?,
                }
            }
        };
        serve.validate()?;

        // The image model loads before the language model for the reason
        // `shiso serve` gives: a KV pool sized to free memory must see its
        // weights already resident.
        let image = boot::boot_image(&serve)?;
        let (mut state, engine) = match serve.model.as_deref() {
            Some(model) if shiso_models::embedder::is_embedder(model) => {
                (AppState::without_language_model(), None)
            }
            Some(model) => {
                let (state, join) = boot::boot_language_model(&serve, model)?;
                (state, Some(join))
            }
            None => (AppState::without_language_model(), None),
        };
        if let Some(image) = image {
            state = state.with_image(image);
        }
        let runtime = Self::assemble(state, engine, installed, executor)?;
        runtime.warm_up()?;
        Ok(runtime)
    }

    /// A runtime over a state its caller built, with the engine thread that
    /// serves it (joined when the runtime drops), for an embedder or a test
    /// that boots by hand. Nothing is warmed.
    ///
    /// # Errors
    ///
    /// [`OpenError::NothingLoaded`] for a state with neither a language model
    /// nor an image model, and [`OpenError::Executor`].
    pub fn from_state(state: AppState, engine: Option<EngineThread>) -> Result<Self, OpenError> {
        match Executor::start() {
            Ok(executor) => Self::assemble(state, engine, None, executor),
            Err(error) => {
                // The state first: its queue closing is what lets the join return.
                drop(state);
                drop(EngineGuard(engine));
                Err(OpenError::Executor(error))
            }
        }
    }

    fn assemble(
        state: AppState,
        engine: Option<EngineThread>,
        installed: Option<InstalledModel>,
        executor: Executor,
    ) -> Result<Self, OpenError> {
        // Wrapped first so an early return still joins the thread.
        let engine = EngineGuard(engine);
        if state.engine.is_none() && state.image.is_none() {
            // The state first: its queue closing is what lets the join return.
            drop(state);
            drop(engine);
            return Err(OpenError::NothingLoaded);
        }
        Ok(Self {
            state: Arc::new(state),
            _engine: engine,
            executor,
            _installed: installed,
        })
    }

    /// Whether a language model is loaded: [`Self::generate`] and
    /// [`Self::decide`] answer [`AppError::NoModel`] when not.
    #[must_use]
    pub fn has_language_model(&self) -> bool {
        self.state.engine.is_some()
    }

    /// Whether an image model is loaded: [`Self::images`] and
    /// [`Self::edit_image`] answer [`AppError::NoImageModel`] when not.
    #[must_use]
    pub fn has_image_model(&self) -> bool {
        self.state.image.is_some()
    }

    /// Runs one chat turn and returns the whole answer.
    ///
    /// # Errors
    ///
    /// [`AppError::NoModel`] with no language model; [`AppError::NotReady`]
    /// when the model ships no chat template; [`AppError::ContextOverflow`]
    /// for a prompt that leaves no room to answer; otherwise as
    /// [`pipeline::start`] and [`crate::api::blocking::collect_response`].
    pub async fn generate(&self, request: &Request) -> Result<Response, AppError> {
        within(self.executor.handle.clone(), async {
            let started = pipeline::start(&self.state, request, PromptField::Messages).await?;
            crate::api::blocking::collect_response(started.events).await
        })
        .await
    }

    /// [`Self::generate`], parking the calling thread.
    ///
    /// # Errors
    ///
    /// As [`Self::generate`], and [`BLOCKING_IN_ASYNC`] from an async worker.
    pub fn generate_blocking(&self, request: &Request) -> Result<Response, AppError> {
        self.block(self.generate(request))
    }

    /// Runs one chat turn, handing each [`StreamEvent`] to `sink` as it
    /// happens. `sink` returns `false` to cancel: the request is dropped and
    /// the call returns `Ok`.
    ///
    /// A refusal that arrives before the first call to `sink` is the request's;
    /// a failure after that is carried by the stream as [`StreamEvent`]s.
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::generate`] that precede the first event.
    pub async fn generate_stream(
        &self,
        request: &Request,
        mut sink: impl FnMut(StreamEvent) -> bool + Send,
    ) -> Result<(), AppError> {
        use futures::StreamExt as _;
        within(self.executor.handle.clone(), async {
            let mut events = pipeline::start(&self.state, request, PromptField::Messages)
                .await?
                .events;
            while let Some(event) = events.next().await {
                if !sink(event) {
                    break;
                }
            }
            Ok(())
        })
        .await
    }

    /// Makes the pictures of a text-to-image request.
    ///
    /// # Errors
    ///
    /// [`AppError::NoImageModel`] with no image model; otherwise as
    /// [`crate::image::generate`].
    pub async fn images(&self, request: Generate) -> Result<Images, AppError> {
        within(
            self.executor.handle.clone(),
            crate::image::generate(&self.state, request),
        )
        .await
    }

    /// [`Self::images`], handing each [`Step`] to `sink`; `false` stops.
    ///
    /// # Errors
    ///
    /// As [`Self::images`].
    pub async fn images_stream(
        &self,
        request: Generate,
        sink: impl FnMut(Step) -> bool + Send,
    ) -> Result<(), AppError> {
        within(
            self.executor.handle.clone(),
            crate::image::generate_stream(&self.state, request, sink),
        )
        .await
    }

    /// Edits the reference pictures of `request` into one picture.
    ///
    /// # Errors
    ///
    /// As [`Self::images`], and [`AppError::BadRequest`] for references the
    /// model cannot take.
    pub async fn edit_image(&self, request: Edit) -> Result<Images, AppError> {
        within(
            self.executor.handle.clone(),
            crate::image::edit(&self.state, request),
        )
        .await
    }

    /// [`Self::edit_image`], handing each [`Step`] to `sink`; `false` stops.
    ///
    /// # Errors
    ///
    /// As [`Self::edit_image`].
    pub async fn edit_image_stream(
        &self,
        request: Edit,
        sink: impl FnMut(Step) -> bool + Send,
    ) -> Result<(), AppError> {
        within(
            self.executor.handle.clone(),
            crate::image::edit_stream(&self.state, request, sink),
        )
        .await
    }

    /// Scores the questions of `request`: one answer per question, each with
    /// its probabilities.
    ///
    /// Concurrent calls are answered one prompt at a time by the engine, so an
    /// answer does not depend on who else is asking.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] for a request with no questions or one the
    /// model cannot be asked, and the errors of [`crate::decision::decide`];
    /// [`AppError::NoModel`] with no language model.
    pub async fn decide(
        &self,
        request: shiso_ir::decision::Request,
    ) -> Result<shiso_ir::decision::Response, AppError> {
        within(
            self.executor.handle.clone(),
            crate::decision::decide(&self.state, request),
        )
        .await
    }

    /// [`Self::decide`], parking the calling thread.
    ///
    /// # Errors
    ///
    /// As [`Self::decide`], and [`BLOCKING_IN_ASYNC`] from an async worker.
    pub fn decide_blocking(
        &self,
        request: shiso_ir::decision::Request,
    ) -> Result<shiso_ir::decision::Response, AppError> {
        self.block(self.decide(request))
    }

    /// Parks the calling thread on `future`, or refuses when that would stall
    /// an async worker.
    fn block<T>(&self, future: impl Future<Output = Result<T, AppError>>) -> Result<T, AppError> {
        try_block_on(&self.executor.handle, future).unwrap_or_else(|| Err(blocking_in_async()))
    }

    /// The throwaway pass that warms the device failed.
    /// The throwaway pass [`Self::open`] documents.
    fn warm_up(&self) -> Result<(), OpenError> {
        let state = &*self.state;
        if state.decision_layout.is_some() || state.decision_readout.is_some() {
            self.decide_blocking(warm_up_decision())
                .map_err(OpenError::WarmUp)?;
        } else if state.engine.is_some() && state.renderer.is_some() {
            let mut request = Request::new(state.model_id.clone(), vec![Message::user("warm-up")]);
            request.max_tokens = 1;
            self.generate_blocking(&request)
                .map_err(OpenError::WarmUp)?;
        }
        if state.image.is_some() {
            let request = Generate {
                prompt: "warm-up".to_owned(),
                n: Some(1),
                size: Some("256x256".to_owned()),
                aspect_ratio: None,
                steps: Some(1),
                seed: Some(0),
                output_format: None,
                output_compression: None,
            };
            if let Err(error) = self.block(self.images(request)) {
                tracing::warn!(%error, "the image warm-up pass failed; the first image will be slower");
            }
        }
        Ok(())
    }
}

/// Runs `future` to completion on `handle` from the calling thread, or `None`
/// when that thread is driving async tasks.
///
/// There is no way to ask that question: a blocking-pool thread holds the
/// runtime's handle and has a task id, yet may block, while a `block_on` main
/// future is no task and may not. Tokio refuses the nested `block_on` by
/// panicking before it polls anything, which this turns into the `None` it
/// means. Only that refusal is caught; any other panic goes on.
fn try_block_on<F: Future>(handle: &Handle, future: F) -> Option<F::Output> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle.block_on(future)))
        .map_err(|payload| {
            let nested = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .is_some_and(|message| message.contains("within a runtime"));
            if !nested {
                std::panic::resume_unwind(payload);
            }
        })
        .ok()
}

/// The error of a blocking call made where it would stall an executor.
fn blocking_in_async() -> AppError {
    AppError::bad_request(
        "this call blocks; call it from a plain thread or a spawn_blocking thread, or use the async method",
        None,
        BLOCKING_IN_ASYNC,
    )
}

/// The smallest valid decision request: the warm-up pass.
fn warm_up_decision() -> shiso_ir::decision::Request {
    let choice = |value: &str| Choice {
        value: ChoiceValue::Str(value.to_owned()),
        description: None,
    };
    shiso_ir::decision::Request {
        input: Input::Text("warm-up".to_owned()),
        readout_state: ReadoutState::Input,
        questions: vec![Question {
            name: "warm_up".to_owned(),
            instructions: "pick one".to_owned(),
            kind: QuestionKind::Choice(vec![choice("a"), choice("b")]),
            probability_answer: false,
        }],
        temperature: None,
        preserve_thinking: None,
        token_ids: false,
    }
}
