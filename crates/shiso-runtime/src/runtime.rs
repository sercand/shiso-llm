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
//! task, `futures::executor::block_on`, a UI toolkit's. The cores need a tokio
//! timer for the engine's admission deadline, which a foreign executor does not
//! provide, so the runtime owns a one-worker tokio runtime that drives timers
//! and is the context each poll enters. The work itself is on the engine thread
//! and the image worker; the caller's task only awaits channels, which are
//! executor-neutral.
//!
//! # Blocking
//!
//! The `*_blocking` methods spawn the call onto that private runtime and park
//! the calling thread on a standard channel until it answers. They never call
//! `block_on` on the caller's thread, so they cannot panic whatever the caller
//! is: a plain thread, a `spawn_blocking` thread, or an async worker. On an
//! async worker they block that worker for the length of the call, as any
//! blocking I/O does; prefer the async methods there. The single refusal is a
//! call made on this runtime's own worker thread (reached by a task spawned
//! from a stream sink, which runs in the runtime's context): it would wait for
//! itself, so it returns an error. Other threads of that runtime's blocking
//! pool, and any thread of another `Runtime`, may call.
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
use crate::private_runtime::PrivateRuntime;
use crate::state::AppState;

/// Where a [`Runtime`]'s model comes from.
#[derive(Debug, Clone, Copy)]
pub enum Source<'a> {
    /// A registry reference (`name` or `name:variant`), installed in the model
    /// home, pulled first when [`RuntimeOptions::pull_if_missing`] says so.
    Registry(&'a str),
    /// A directory: an installed variant (it holds a `recipe.json`) or a
    /// checkpoint, as [`hub::resolve_serve_dir`] reads it. Never touches the
    /// network and takes no lock: if the directory is a registry install, the
    /// caller keeps the [`InstalledModel`] from [`hub::resolve`] or
    /// [`hub::pull`] alive for as long as the `Runtime`, or `shiso rm` or a
    /// reinstall can remove the files under it.
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
    ///
    /// The KV cache pool is the exception to "the server's defaults": left
    /// unset (`kv_cache_tokens` and `gpu_memory_utilization` both `None`), the
    /// runtime sizes it to `2 x ctx x max_num_seqs` tokens (the sessions and an equal prefix cache), not the server's 131072,
    /// which would sit resident for an embedder that never has that many
    /// sessions. To raise it, set `serve.kv_cache_tokens` (a token count) or
    /// `serve.gpu_memory_utilization` (a share of device memory) yourself; the
    /// pool must hold at least one `ctx`. `shiso serve` is unchanged.
    ///
    /// `bind` and `port` are the server's listener settings; they are never
    /// used here, and `port` must still be non-zero to validate.
    ///
    /// `explicit` is a set of field names, matched exactly (`"ctx"`,
    /// `"max_num_seqs"`): a misspelled name is ignored, and a field changed in
    /// `serve` but not named there is replaced by a recipe default when the
    /// recipe has one.
    pub serve: ServeOptions,
    /// The fields of `serve` the caller set on purpose, which a recipe's
    /// defaults must not replace (`ctx`, `max_num_seqs`, ...).
    pub explicit: ExplicitlySet,
}

/// Why [`Runtime::open`] failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OpenError {
    /// The model could not be made available: it is not installed (and pulling
    /// was off), the pull failed, or, for a [`Source::Dir`], its `recipe.json`
    /// is unreadable ([`PullError::Inconsistent`]) or a component option was
    /// also set ([`PullError::ComponentFlagConflict`]).
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

thread_local! {
    /// The id of the [`Executor`] whose worker this thread is, or 0. A blocking
    /// call must not wait on its own runtime's worker: that worker is the one
    /// that would have to run it. Blocking-pool threads and other runtimes'
    /// threads may wait, so this is an id and not a flag, and it is set from
    /// the park hook, which only workers fire.
    static WORKER_OF: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The private runtime plus the id its worker carries in [`WORKER_OF`].
#[derive(Debug)]
struct Executor {
    id: u64,
    runtime: PrivateRuntime,
}

impl Executor {
    fn start() -> std::io::Result<Self> {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (parked, first_park) = std::sync::mpsc::sync_channel(1);
        let runtime = PrivateRuntime::build(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .thread_name("shiso-runtime")
                .on_thread_park(move || {
                    WORKER_OF.set(id);
                    let _ = parked.try_send(());
                })
                .enable_all(),
        )?;
        // The idle worker parks at once; waiting for that means the thread is
        // marked before any task can run on it.
        let _ = first_park.recv_timeout(std::time::Duration::from_secs(5));
        Ok(Self { id, runtime })
    }

    const fn handle(&self) -> &Handle {
        self.runtime.handle()
    }
}

/// Polls `future` inside `handle`'s context on every poll, so tokio timers
/// work under any executor, and the future stays in the caller's task (it need
/// not be `'static`, and dropping it cancels it).
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
    /// longer. The image pass asks for the smallest picture the model's limits
    /// allow (the smallest side, at the model's fixed step count or one step).
    /// A failed decision or text pass fails the open; a failed image pass is
    /// logged and ignored, because it has only been exercised against mocks.
    ///
    /// `progress(done, total)` reports a pull's bytes; returning `false`
    /// cancels it ([`PullError::Cancelled`]). It is not called for a
    /// [`Source::Dir`] or an installed model.
    ///
    /// Blocks while the weights load (and the pull, if any): call it from a
    /// thread that may block. From an async worker it blocks that worker.
    ///
    /// # Errors
    ///
    /// [`OpenError::Pull`] for a model that is not installed or cannot be
    /// pulled; [`OpenError::Config`] and [`OpenError::Boot`] for settings or
    /// files that will not boot; [`OpenError::NothingLoaded`];
    /// [`OpenError::WarmUp`].
    pub fn open(
        source: Source<'_>,
        options: RuntimeOptions,
        progress: &mut dyn FnMut(u64, u64) -> bool,
    ) -> Result<Self, OpenError> {
        let executor = Executor::start().map_err(OpenError::Executor)?;
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
                // The registry's spelling is lowercase and an embedder need not
                // know that; a string that is a path is left as it is, which is
                // what `shiso serve` does with it.
                let name = match hub::classify_model_arg(name) {
                    hub::ModelArgKind::RegistryRef(reference) => reference.to_lowercase(),
                    hub::ModelArgKind::Path(_) => name.to_owned(),
                };
                match hub::resolve_serve_model(&name, &mut serve, &explicit, &config) {
                    Err(PullError::NotInstalled(_)) if pull_if_missing => {
                        // Held across the second resolve so the variant cannot
                        // be removed in between; both locks are shared.
                        let pulled = hub::pull_for(serve.device, &name, &config, progress)?;
                        let installed =
                            hub::resolve_serve_model(&name, &mut serve, &explicit, &config)?;
                        drop(pulled);
                        installed
                    }
                    other => other?,
                }
            }
        };
        // A server holds many sessions' worth of cache and its default pool is
        // sized so (131072 tokens); an embedder asking one question at a time
        // does not need that resident. Unless the caller chose a pool (tokens or
        // a memory share), it gets twice what `max_num_seqs` sessions of `ctx`
        // tokens fill: the second share is the prefix cache, which a request
        // that resumes from a shared prefix needs beside its own session (with
        // one share a long two-question prompt ran out of pages at one slot).
        if serve.kv_cache_tokens.is_none() && serve.gpu_memory_utilization.is_none() {
            serve.kv_cache_tokens = Some(
                u64::from(serve.ctx)
                    .saturating_mul(u64::try_from(serve.max_num_seqs).unwrap_or(u64::MAX))
                    .saturating_mul(2),
            );
        }
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
    /// Drop closes the queue by dropping the state and then joins `engine`, so
    /// a caller that keeps its own clone of the engine handle (or of anything
    /// in the state that holds one) makes the drop wait for it forever.
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
    pub async fn generate(&self, request: Request) -> Result<Response, AppError> {
        within(self.executor.handle().clone(), async {
            chat(&self.state, &request).await
        })
        .await
    }

    /// [`Self::generate`], parking the calling thread (see the
    /// [module documentation](self#blocking)).
    ///
    /// # Errors
    ///
    /// As [`Self::generate`].
    pub fn generate_blocking(&self, request: Request) -> Result<Response, AppError> {
        self.block(move |state| async move { chat(&state, &request).await })
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
        request: Request,
        mut sink: impl FnMut(StreamEvent) -> bool + Send,
    ) -> Result<(), AppError> {
        use futures::StreamExt as _;
        within(self.executor.handle().clone(), async {
            let mut events = pipeline::start(&self.state, &request, PromptField::Messages)
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
            self.executor.handle().clone(),
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
            self.executor.handle().clone(),
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
            self.executor.handle().clone(),
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
            self.executor.handle().clone(),
            crate::image::edit_stream(&self.state, request, sink),
        )
        .await
    }

    /// Scores the questions of `request`: one answer per question, each with
    /// its probabilities.
    ///
    /// Concurrent calls may share a batched pass in the engine. On a backend
    /// that batches in half precision an answer can then differ from the one a
    /// lone call gets in the low bits (up to about 2e-4 in a probability,
    /// measured on d1-3b; none on a decision-head model); the chosen option is
    /// stable.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] (code `invalid_request`, param `questions`) for
    /// a request with no questions, or one the model cannot be asked, and the
    /// errors of [`crate::decision::decide`]; [`AppError::NoModel`] with no
    /// language model.
    pub async fn decide(
        &self,
        request: shiso_ir::decision::Request,
    ) -> Result<shiso_ir::decision::Response, AppError> {
        require_questions(&request)?;
        within(
            self.executor.handle().clone(),
            crate::decision::decide(&self.state, request),
        )
        .await
    }

    /// [`Self::decide`], parking the calling thread (see the
    /// [module documentation](self#blocking)).
    ///
    /// # Errors
    ///
    /// As [`Self::decide`].
    pub fn decide_blocking(
        &self,
        request: shiso_ir::decision::Request,
    ) -> Result<shiso_ir::decision::Response, AppError> {
        require_questions(&request)?;
        self.block(move |state| async move { crate::decision::decide(&state, request).await })
    }

    /// Runs the call `make` builds on the private runtime and parks the calling
    /// thread on a channel until it answers. Nothing here calls `block_on`, so
    /// no caller can make it panic; the task owns what it needs, which is why
    /// `make` is handed the state.
    ///
    /// The one caller that cannot wait is this runtime's own worker thread,
    /// which would be waiting for itself: a sink runs inside the runtime's tokio
    /// context, so a task it spawns lands there. That is refused with an error.
    /// Its blocking-pool threads (`spawn_blocking` from a sink) may wait, and so
    /// may another runtime's threads, workers included.
    ///
    /// The task drops its state clone after the answer is sent, so a caller
    /// that drops the runtime at once joins the engine while the clone is still
    /// held; the executor, declared after the engine guard, is alive to drop it.
    fn block<T, F>(&self, make: impl FnOnce(Arc<AppState>) -> F) -> Result<T, AppError>
    where
        T: Send + 'static,
        F: Future<Output = Result<T, AppError>> + Send + 'static,
    {
        if WORKER_OF.get() == self.executor.id {
            return Err(AppError::bad_request(
                "a blocking call was made on the runtime's own worker thread (a task spawned from a sink?); use the async method there",
                None,
                "blocking_on_runtime_worker",
            ));
        }
        let (answer, answered) = std::sync::mpsc::channel();
        let call = make(Arc::clone(&self.state));
        self.executor.handle().spawn(async move {
            let _ = answer.send(call.await);
        });
        // A closed channel is a task that never answered: it panicked, or the
        // executor stopped.
        answered.recv().unwrap_or_else(|_| {
            Err(AppError::internal(
                "the runtime's executor ended the call without an answer",
                std::io::Error::other("the call's task was dropped"),
            ))
        })
    }

    /// The throwaway pass [`Self::open`] documents.
    fn warm_up(&self) -> Result<(), OpenError> {
        let state = &*self.state;
        if state.decision_layout.is_some() || state.decision_readout.is_some() {
            self.decide_blocking(warm_up_decision())
                .map_err(OpenError::WarmUp)?;
        } else if state.engine.is_some() && state.renderer.is_some() {
            let mut request = Request::new(state.model_id.clone(), vec![Message::user("warm-up")]);
            request.max_tokens = 1;
            self.generate_blocking(request).map_err(OpenError::WarmUp)?;
        }
        if let Some(image) = state.image.as_ref() {
            let request = warm_up_image(image.limits());
            if let Err(error) = self
                .block(move |state| async move { crate::image::generate(&state, request).await })
            {
                tracing::warn!(%error, "the image warm-up pass failed; the first image will be slower");
            }
        }
        Ok(())
    }
}

/// One chat turn over `state`, folded into its response.
async fn chat(state: &AppState, request: &Request) -> Result<Response, AppError> {
    let started = pipeline::start(state, request, PromptField::Messages).await?;
    crate::api::blocking::collect_response(started.events).await
}

/// A decision request with no questions asks nothing; refused here rather than
/// in the shared core, whose answer on the wire routes stays as it was.
fn require_questions(request: &shiso_ir::decision::Request) -> Result<(), AppError> {
    if request.questions.is_empty() {
        return Err(AppError::bad_request(
            "a decision request needs at least one question",
            Some("questions"),
            "invalid_request",
        ));
    }
    Ok(())
}

/// The smallest picture the limits allow: the smallest side on the size grid
/// (a multiple of zero is no grid, so the side stands as it is), at the fixed
/// step count or one step.
fn warm_up_image(limits: &shiso_vision::generator::ImageLimits) -> Generate {
    let multiple = limits.size_multiple.max(1);
    let side = (limits.min_side.div_ceil(multiple) * multiple).max(multiple);
    Generate {
        prompt: "warm-up".to_owned(),
        n: Some(1),
        size: Some(format!("{side}x{side}")),
        aspect_ratio: None,
        steps: Some(limits.fixed_steps.unwrap_or(1)),
        seed: Some(0),
        output_format: None,
        output_compression: None,
    }
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

#[cfg(test)]
mod tests {
    use super::warm_up_image;
    use shiso_vision::generator::ImageLimits;

    fn limits(multiple: u32, min_side: u32, fixed: Option<u32>) -> ImageLimits {
        ImageLimits {
            size_multiple: multiple,
            min_side,
            max_side: 4096,
            default_size: (1024, 1024),
            default_steps: 4,
            fixed_steps: fixed,
        }
    }

    #[test]
    fn the_warm_up_picture_is_the_smallest_the_limits_allow() {
        let size = |l: &ImageLimits| warm_up_image(l).size.expect("a size");
        assert_eq!(size(&limits(16, 256, None)), "256x256");
        assert_eq!(
            size(&limits(16, 250, None)),
            "256x256",
            "rounded up to the grid"
        );
        assert_eq!(size(&limits(64, 100, None)), "128x128");
        assert_eq!(size(&limits(16, 0, None)), "16x16", "never a zero side");
        assert_eq!(size(&limits(0, 250, None)), "250x250", "no grid: no panic");
        assert_eq!(warm_up_image(&limits(16, 256, None)).steps, Some(1));
        assert_eq!(warm_up_image(&limits(16, 256, Some(4))).steps, Some(4));
    }
}
