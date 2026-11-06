//! 3D splat generation: defaults, request translation, and the queueing
//! service.
//!
//! The shape is [`crate::image`]'s: a wire handler turns a request into a
//! validated [`SplatJob`], the service queues it on a bounded channel, and one
//! worker thread runs it on the device. What differs is the vocabulary - steps
//! and Gaussians instead of a size - and that a splat's cost is not bounded by
//! a per-request token count.

mod core;
mod worker;

pub use core::build_job;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use shiso_core::ids::Seed;
use shiso_ir::splat::Generate;
use shiso_vision::splat::{
    GAUSSIANS_PER_ANCHOR, GeneratedSplat, MAX_GAUSSIANS, MAX_STEPS, MIN_GAUSSIANS, SplatGenError,
    SplatGenerator, SplatJob, SplatLimits, SplatStepEvent,
};
use tokio::sync::{mpsc, oneshot};

use crate::error::AppError;

/// What a splat request that names no step count or Gaussian count gets.
///
/// The server-side counterpart of the wire's optional `steps` and
/// `num_gaussians`: the loaded model's [`SplatLimits`] say what it was built
/// to run at, and the operator may move either value at boot through
/// [`SplatOverrides`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplatDefaults {
    /// Denoising steps.
    pub steps: u32,
    /// How many Gaussians to place.
    pub num_gaussians: u32,
}

impl SplatDefaults {
    /// What the model itself defaults to.
    #[must_use]
    pub const fn of(limits: &SplatLimits) -> Self {
        Self {
            steps: limits.default_steps,
            num_gaussians: limits.default_num_gaussians,
        }
    }
}

/// The operator's explicit choices, each `None` when the flag was not given.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SplatOverrides {
    /// `--splat-default-steps`.
    pub steps: Option<u32>,
    /// `--splat-default-num-gaussians`.
    pub num_gaussians: Option<u32>,
}

/// An operator override the loaded model cannot honour.
///
/// Refused at boot rather than clamped: a default that silently moved is not
/// the default that was configured, and one the model refuses would fail
/// every request that relied on it.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum OverrideError {
    /// The step count is outside what any splat model runs.
    #[error(
        "--splat-default-steps {steps} is not servable: steps must be between 1 and {MAX_STEPS}"
    )]
    Steps {
        /// Steps as configured.
        steps: u32,
    },
    /// The Gaussian count is outside the family's range or off its grid.
    #[error(
        "--splat-default-num-gaussians {count} is not servable: it must be a multiple of \
         {GAUSSIANS_PER_ANCHOR} between {MIN_GAUSSIANS} and {MAX_GAUSSIANS}"
    )]
    Gaussians {
        /// The count as configured.
        count: u32,
    },
}

impl SplatOverrides {
    /// The model's defaults with these overrides applied.
    ///
    /// # Errors
    ///
    /// [`OverrideError`] when an override is one the model would refuse.
    pub fn resolve(self, limits: SplatLimits) -> Result<SplatDefaults, OverrideError> {
        let mut defaults = SplatDefaults::of(&limits);
        if let Some(steps) = self.steps {
            if steps == 0 || steps > MAX_STEPS {
                return Err(OverrideError::Steps { steps });
            }
            defaults.steps = steps;
        }
        if let Some(count) = self.num_gaussians {
            if !(MIN_GAUSSIANS..=MAX_GAUSSIANS).contains(&count)
                || !count.is_multiple_of(GAUSSIANS_PER_ANCHOR)
            {
                return Err(OverrideError::Gaussians { count });
            }
            defaults.num_gaussians = count;
        }
        Ok(defaults)
    }
}

/// Judges a request's knobs without a loaded model.
///
/// The wire calls this before it asks for the service, so a malformed request
/// is a 400 whether or not a splat model is loaded. The defaults come from
/// [`SplatLimits::default`] because none of the checks depend on them: an
/// omitted knob is filled from a default that is valid by construction.
///
/// # Errors
///
/// [`AppError::BadRequest`] naming the first field that is out of range, or
/// `image` when the request carries no bytes.
pub fn validate(request: &Generate, defaults: SplatLimits) -> Result<(), AppError> {
    build_job(request.clone(), SplatDefaults::of(&defaults), Seed::new(0)).map(|_job| ())
}

/// The HTTP failure a [`SplatGenError`] answers with.
///
/// The caller-side refusals (an undecodable picture, an alpha that matting or
/// erosion emptied, a knob outside its range) are the request's fault and
/// answer 400; the device-side failures answer 500 with the cause kept as a
/// real source for the log.
#[must_use]
pub fn refuse(error: SplatGenError) -> AppError {
    let message = error.to_string();
    match &error {
        SplatGenError::UndecodableImage(_) | SplatGenError::EmptyAlpha => {
            AppError::bad_request(message, Some("image"), "invalid_request")
        }
        SplatGenError::InvalidSteps { .. } => {
            AppError::bad_request(message, Some("steps"), "invalid_request")
        }
        SplatGenError::InvalidGaussians { .. } => {
            AppError::bad_request(message, Some("num_gaussians"), "invalid_request")
        }
        SplatGenError::InvalidCfg { .. } => {
            AppError::bad_request(message, Some("cfg_strength"), "invalid_request")
        }
        SplatGenError::InvalidRescale { .. } => {
            AppError::bad_request(message, Some("rescale_t"), "invalid_request")
        }
        SplatGenError::InvalidCropMargin { .. } => {
            AppError::bad_request(message, Some("crop_margin"), "invalid_request")
        }
        SplatGenError::Matting(_)
        | SplatGenError::Backend(_)
        | SplatGenError::Checkpoint(_)
        | SplatGenError::Encoding(_) => AppError::internal("splat generation failed", error),
    }
}

/// A queued generation and where its answer and its progress go.
#[derive(Debug)]
struct Queued {
    job: SplatJob,
    /// Where step events go for a streaming caller; `None` for a plain one.
    progress: Option<mpsc::UnboundedSender<SplatStepEvent>>,
    reply: oneshot::Sender<Result<GeneratedSplat, AppError>>,
}

/// What the worker thread is started with.
#[derive(Debug)]
pub struct SplatServiceDesc {
    /// The generator, loaded on the splat backend.
    pub generator: Arc<dyn SplatGenerator>,
    /// The name the generator answers to on the wire.
    pub model_id: String,
    /// The most generations that may wait in the queue beyond the running
    /// one; the next is refused.
    pub max_queue: usize,
    /// Where per-generation profile reports go; `None` unless `--profile-dir`.
    pub profile: Option<crate::profile::ProfileSink>,
}

/// The handle the HTTP handlers hold.
#[derive(Debug, Clone)]
pub struct SplatService {
    tx: mpsc::Sender<Queued>,
    model_id: String,
    limits: SplatLimits,
    defaults: SplatDefaults,
}

impl SplatService {
    /// Starts the worker thread.
    ///
    /// The thread is not joined: it ends when the last handle is dropped,
    /// which is when the process is done serving, and the queue closing is
    /// what ends its loop.
    ///
    /// # Errors
    ///
    /// The operating system's refusal to start a thread.
    pub fn start(desc: SplatServiceDesc) -> std::io::Result<Self> {
        // A queue of zero would admit nothing beyond the running job and
        // refuse everything else; the floor of one is the image service's, and
        // the operator's flag is refused at boot rather than reaching here.
        let (tx, rx) = mpsc::channel(desc.max_queue.max(1));
        let limits = desc.generator.limits();
        let service = Self {
            tx,
            model_id: desc.model_id.clone(),
            limits,
            defaults: SplatDefaults::of(&limits),
        };
        std::thread::Builder::new()
            .name("shiso-splat".to_owned())
            .spawn(move || worker::run_worker(&desc, rx))?;
        Ok(service)
    }

    /// The same service with the operator's default step and Gaussian counts
    /// applied over the model's own.
    ///
    /// # Errors
    ///
    /// [`OverrideError`] for an override the model would refuse.
    pub fn with_overrides(mut self, overrides: SplatOverrides) -> Result<Self, OverrideError> {
        self.defaults = overrides.resolve(self.limits)?;
        Ok(self)
    }

    /// What a request that names no steps or Gaussians gets.
    #[must_use]
    pub const fn defaults(&self) -> SplatDefaults {
        self.defaults
    }

    /// The name the generator answers to on the wire.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Queues a generation and waits for the splat.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] for a job [`SplatJob::validate`] refuses -
    /// refused here, before it costs a queue slot. [`AppError::Overloaded`]
    /// when the queue is full: a caller that backs off and retries is served.
    /// [`AppError::Internal`] when the worker thread is gone, or the generator
    /// failed on the device side.
    pub async fn generate(&self, job: SplatJob) -> Result<GeneratedSplat, AppError> {
        self.submit(job, None).await
    }

    /// Queues a generation, forwards each step event to `progress` while it
    /// runs, and waits for the splat.
    ///
    /// A caller that stops reading `progress` does not stop the generation;
    /// its final answer still arrives on the returned future.
    ///
    /// # Errors
    ///
    /// The same refusals as [`Self::generate`].
    pub async fn generate_stream(
        &self,
        job: SplatJob,
        progress: mpsc::UnboundedSender<SplatStepEvent>,
    ) -> Result<GeneratedSplat, AppError> {
        self.submit(job, Some(progress)).await
    }

    /// The one path both entry points take: admit, then await the answer.
    async fn submit(
        &self,
        job: SplatJob,
        progress: Option<mpsc::UnboundedSender<SplatStepEvent>>,
    ) -> Result<GeneratedSplat, AppError> {
        self.admit(job, progress)?.await
    }

    /// Validates and queues a generation without waiting for the splat: the
    /// synchronous half of [`Self::generate`], for a caller that must know
    /// whether it was admitted before it commits to anything - the stream
    /// handler, which answers a refusal with a status code and only then
    /// spends its 200.
    ///
    /// The returned future is the answer, exactly what [`Self::generate`]
    /// would have awaited.
    ///
    /// # Errors
    ///
    /// The admission refusals of [`Self::generate`]: [`AppError::BadRequest`]
    /// for a job [`SplatJob::validate`] refuses, and [`AppError::Overloaded`]
    /// when the queue is full. Both are in hand without an await.
    pub fn admit(
        &self,
        job: SplatJob,
        progress: Option<mpsc::UnboundedSender<SplatStepEvent>>,
    ) -> Result<SplatAnswer, AppError> {
        // Checked before the queue: a job the generator would refuse must not
        // cost a waiting slot.
        job.validate().map_err(refuse)?;
        let (reply, answer) = oneshot::channel();
        // `try_send`, not `send`: a full queue is refused now rather than
        // awaited, because a splat that arrives behind a queue nobody bounded
        // is an answer no client is still waiting for.
        self.tx
            .try_send(Queued {
                job,
                progress,
                reply,
            })
            .map_err(|err| match err {
                mpsc::error::TrySendError::Full(_queued) => AppError::Overloaded {
                    what: "the splat queue is full",
                },
                mpsc::error::TrySendError::Closed(_queued) => {
                    AppError::internal("the splat queue is closed", WorkerGone)
                }
            })?;
        Ok(Box::pin(async move {
            answer.await.map_err(|_dropped| {
                AppError::internal("the splat worker dropped the request", WorkerGone)
            })?
        }))
    }
}

/// The answer to an admitted generation: what the worker reports, or the
/// 500 saying it is gone.
///
/// Boxed so [`SplatService::admit`] can hand the same future to every caller:
/// the worker-gone mapping lives in `admit` alone, and the two entry points
/// that await it cannot drift apart in wording.
pub type SplatAnswer = Pin<Box<dyn Future<Output = Result<GeneratedSplat, AppError>> + Send>>;

/// The worker thread is not there any more.
///
/// A unit error type rather than a string, so the two places that report it
/// cannot drift apart in wording.
#[derive(Debug, thiserror::Error)]
#[error("the splat worker thread is not running")]
struct WorkerGone;

/// A generation panicked inside the generator.
///
/// The worker catches the panic so one bad job does not end the thread; this
/// is what the caller's 500 keeps as its cause.
#[derive(Debug, thiserror::Error)]
#[error("the splat generator panicked")]
struct GeneratorPanicked;

#[cfg(test)]
mod tests {
    use shiso_core::ids::Seed;
    use shiso_ir::splat::{Generate, Mode};
    use shiso_vision::splat::SplatLimits;

    use super::{OverrideError, SplatDefaults, SplatOverrides, core};
    use crate::error::AppError;

    fn request() -> Generate {
        Generate {
            image: vec![1],
            mode: Mode::Synthetic,
            seed: None,
            steps: None,
            cfg_strength: None,
            rescale_t: None,
            num_gaussians: None,
            crop_margin: None,
        }
    }

    #[test]
    fn an_omitted_field_takes_the_model_default() {
        let defaults = SplatDefaults::of(&SplatLimits::default());
        let job = core::build_job(
            Generate {
                mode: Mode::Real,
                seed: Some(9),
                ..request()
            },
            defaults,
            Seed::new(9),
        )
        .expect("defaults are valid");
        assert_eq!((job.steps, job.num_gaussians), (50, 262_144));
        assert!(
            (job.crop_margin - 1.7).abs() < f32::EPSILON,
            "real mode picks the loose crop"
        );
    }

    #[test]
    fn an_explicit_margin_beats_the_mode() {
        let job = core::build_job(
            Generate {
                mode: Mode::Real,
                seed: Some(1),
                crop_margin: Some(1.4),
                ..request()
            },
            SplatDefaults::of(&SplatLimits::default()),
            Seed::new(1),
        )
        .expect("valid");
        assert!((job.crop_margin - 1.4).abs() < f32::EPSILON);
    }

    #[test]
    fn an_override_the_model_would_refuse_fails_at_boot() {
        let over = SplatOverrides {
            steps: Some(101),
            num_gaussians: None,
        };
        assert!(matches!(
            over.resolve(SplatLimits::default()),
            Err(OverrideError::Steps { .. })
        ));
        let over = SplatOverrides {
            steps: None,
            num_gaussians: Some(32_769),
        };
        assert!(matches!(
            over.resolve(SplatLimits::default()),
            Err(OverrideError::Gaussians { .. })
        ));
    }

    #[test]
    fn an_empty_image_is_a_client_error_on_the_image_field() {
        let err = core::build_job(
            Generate {
                image: Vec::new(),
                ..request()
            },
            SplatDefaults::of(&SplatLimits::default()),
            Seed::new(1),
        )
        .expect_err("empty image is refused");
        match err {
            AppError::BadRequest {
                param: Some(field), ..
            } => assert_eq!(field, "image"),
            other => panic!("expected a field-scoped bad request, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_alpha_is_a_client_error_on_the_image_field() {
        let err = super::refuse(shiso_vision::splat::SplatGenError::EmptyAlpha);
        assert!(matches!(
            err,
            AppError::BadRequest {
                param: Some(ref field),
                ..
            } if field == "image"
        ));
    }
}

#[cfg(test)]
mod service_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use shiso_core::ids::Seed;
    use shiso_vision::splat::{
        GeneratedSplat, SplatGenError, SplatGenerator, SplatJob, SplatLimits, SplatMode,
        SplatStepEvent,
    };
    use tokio::sync::mpsc;

    use super::{SplatService, SplatServiceDesc};
    use crate::error::AppError;

    fn job() -> SplatJob {
        SplatJob {
            image: vec![0x89, b'P', b'N', b'G'],
            mode: SplatMode::Synthetic,
            seed: Seed::new(42),
            steps: 1,
            cfg_strength: 3.0,
            rescale_t: 3.0,
            num_gaussians: 262_144,
            crop_margin: 1.2,
        }
    }

    #[derive(Debug)]
    struct Fake {
        calls: AtomicUsize,
        panic: bool,
        steps: u32,
    }

    impl Fake {
        fn new(panic: bool) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                panic,
                steps: 1,
            })
        }
    }

    impl SplatGenerator for Fake {
        fn model_id(&self) -> &'static str {
            "fake"
        }

        fn limits(&self) -> SplatLimits {
            SplatLimits::default()
        }

        fn generate(
            &self,
            _job: &SplatJob,
            progress: Option<&dyn Fn(SplatStepEvent)>,
        ) -> Result<GeneratedSplat, SplatGenError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(!self.panic, "boom");
            if let Some(callback) = progress {
                callback(SplatStepEvent {
                    step: 0,
                    total_steps: self.steps,
                });
            }
            Ok(GeneratedSplat {
                bytes: b"ply".to_vec(),
                media_type: "model/ply",
                num_gaussians: 262_144,
                completion_tokens: 614_700,
            })
        }
    }

    fn start(fake: &Arc<Fake>) -> SplatService {
        SplatService::start(SplatServiceDesc {
            generator: {
                let generator: Arc<dyn SplatGenerator> = Arc::<Fake>::clone(fake);
                generator
            },
            model_id: "fake".to_owned(),
            max_queue: 1,
            profile: None,
        })
        .expect("thread")
    }

    #[tokio::test]
    async fn a_queued_job_runs_on_the_worker_thread() {
        let fake = Fake::new(false);
        let service = start(&fake);
        let out = service.generate(job()).await.expect("runs");
        assert_eq!(out.media_type, "model/ply");
        assert_eq!(fake.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_bad_job_is_refused_before_it_costs_a_queue_slot() {
        let fake = Fake::new(false);
        let service = start(&fake);
        let mut bad = job();
        bad.steps = 0;
        let Err(err) = service.admit(bad, None) else {
            panic!("a zero-step job must not be admitted");
        };
        assert!(matches!(
            err,
            AppError::BadRequest {
                param: Some(ref field),
                ..
            } if field == "steps"
        ));
        assert_eq!(fake.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_panic_ends_the_job_not_the_thread() {
        let fake = Fake::new(true);
        let service = start(&fake);
        assert!(matches!(
            service.generate(job()).await,
            Err(AppError::Internal { .. })
        ));
        assert!(
            service.generate(job()).await.is_err(),
            "the second job still runs"
        );
        assert_eq!(fake.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn progress_events_reach_a_streaming_caller() {
        let fake = Fake::new(false);
        let service = start(&fake);
        let (tx, mut rx) = mpsc::unbounded_channel();
        let out = service.generate_stream(job(), tx).await.expect("runs");
        assert_eq!(out.completion_tokens, 614_700);
        let event = rx.recv().await.expect("one event");
        assert_eq!((event.step, event.total_steps), (0, 1));
        assert!(
            rx.recv().await.is_none(),
            "the channel closes after the job"
        );
    }

    /// A generator that reports it has started and then waits until the test
    /// releases it: how a job is caught occupying the worker while the queue
    /// behind it fills.
    #[derive(Debug)]
    struct Blocking {
        started: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
        gate: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl SplatGenerator for Blocking {
        fn model_id(&self) -> &'static str {
            "blocking"
        }

        fn limits(&self) -> SplatLimits {
            SplatLimits::default()
        }

        fn generate(
            &self,
            _job: &SplatJob,
            _progress: Option<&dyn Fn(SplatStepEvent)>,
        ) -> Result<GeneratedSplat, SplatGenError> {
            if let Some(started) = self
                .started
                .lock()
                .expect("the started lock is never poisoned")
                .take()
            {
                let _sent = started.send(());
            }
            let _released = self
                .gate
                .lock()
                .expect("the gate lock is never poisoned")
                .recv();
            Ok(GeneratedSplat {
                bytes: b"ply".to_vec(),
                media_type: "model/ply",
                num_gaussians: 262_144,
                completion_tokens: 1,
            })
        }
    }

    #[tokio::test]
    async fn a_full_queue_is_overloaded_rather_than_awaited() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (gate_tx, gate_rx) = std::sync::mpsc::channel();
        let generator = Arc::new(Blocking {
            started: std::sync::Mutex::new(Some(started_tx)),
            gate: std::sync::Mutex::new(gate_rx),
        });
        let boxed: Arc<dyn SplatGenerator> = Arc::<Blocking>::clone(&generator);
        let service = SplatService::start(SplatServiceDesc {
            generator: boxed,
            model_id: "blocking".to_owned(),
            max_queue: 1,
            profile: None,
        })
        .expect("thread");

        let first = tokio::spawn({
            let service = service.clone();
            async move { service.generate(job()).await }
        });
        started_rx
            .await
            .expect("the first job reaches the generator");

        // The queue holds one beyond the running job; the next is refused now
        // rather than awaited, because an answer behind a queue nobody bounded
        // is one no client is still waiting for.
        let queued = service
            .admit(job(), None)
            .expect("the queue has room for one");
        let Err(err) = service.admit(job(), None) else {
            panic!("a full queue must refuse the third job");
        };
        assert!(matches!(err, AppError::Overloaded { .. }), "{err:?}");

        drop(queued);
        gate_tx.send(()).expect("the gate is open");
        let _ = first.await.expect("the first job finishes");
    }

    #[tokio::test]
    async fn the_service_reports_the_model_identity_and_defaults() {
        let fake = Fake::new(false);
        let service = start(&fake);
        assert_eq!(service.model_id(), "fake");
        assert_eq!(
            service.defaults().steps,
            SplatLimits::default().default_steps
        );
    }
}
