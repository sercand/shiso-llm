//! Image generation: the service the HTTP handlers submit to, and the thread
//! that runs it.
//!
//! The generator runs on a thread of its own, over a backend instance of its
//! own, beside the engine and audio threads: a diffusion step and a chat turn
//! must not share a command stream, because a device that records one stream
//! for replay (graph capture) would record the other's launches into it.
//!
//! One generation runs at a time, in arrival order: a denoising step already
//! saturates the device, so a second job beside it would only make both
//! slower while doubling the peak the KV pool was sized against. Callers
//! beyond the running one wait in a bounded queue; a request that finds the
//! queue full is refused with [`AppError::Overloaded`] rather than queued
//! without a bound, because a caller that backs off and retries is served,
//! while one that waits forever holds a connection for an answer that arrives
//! too late to be worth anything.

mod worker;

pub mod size;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageLimits, ImageStepEvent,
};
use tokio::sync::{mpsc, oneshot};

use crate::error::AppError;

/// What an image request that names no geometry or step count gets.
///
/// The server-side counterpart of the wire's optional `size` and
/// `aspect_ratio`: the loaded model's [`ImageLimits`] say what it was built to
/// run at, and the operator may move either value at boot
/// (`--image-default-size`, `--image-default-steps`) through
/// [`ImageOverrides`]. A caller that names nothing inherits the result,
/// exactly as a caller that names no temperature inherits
/// [`AppState::sampling_preset`](crate::state::AppState::sampling_preset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDefaults {
    /// Denoising steps.
    pub steps: u32,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

impl ImageDefaults {
    /// What the model itself defaults to: its fixed schedule when it has one,
    /// else its default step count, and its default size.
    #[must_use]
    pub fn of(limits: &ImageLimits) -> Self {
        Self {
            steps: limits.fixed_steps.unwrap_or(limits.default_steps),
            width: limits.default_size.0,
            height: limits.default_size.1,
        }
    }
}

/// The operator's explicit choices, each `None` when the flag was not given.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageOverrides {
    /// `--image-default-steps`.
    pub steps: Option<u32>,
    /// `--image-default-size`, already parsed.
    pub size: Option<(u32, u32)>,
}

/// An operator override the loaded model cannot honour.
///
/// Refused at boot rather than clamped: a default that silently moved is not
/// the default that was configured, and one the model refuses would fail
/// every request that relied on it.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum OverrideError {
    /// The size is off the model's grid or outside its bounds.
    #[error(
        "--image-default-size {width}x{height} is not servable by this model: both sides must be \
         a multiple of {multiple} within {min}..={max}"
    )]
    Size {
        /// Width as configured.
        width: u32,
        /// Height as configured.
        height: u32,
        /// The model's grid.
        multiple: u32,
        /// The model's smallest side.
        min: u32,
        /// The model's largest side.
        max: u32,
    },
    /// The model runs one schedule and the flag names another.
    #[error(
        "--image-default-steps {steps} contradicts this model's fixed schedule of {fixed} steps"
    )]
    Steps {
        /// Steps as configured.
        steps: u32,
        /// The only count the model runs.
        fixed: u32,
    },
}

impl ImageOverrides {
    /// The model's defaults with these overrides applied.
    ///
    /// # Errors
    ///
    /// [`OverrideError`] when an override is one the model would refuse.
    pub fn resolve(self, limits: &ImageLimits) -> Result<ImageDefaults, OverrideError> {
        let mut defaults = ImageDefaults::of(limits);
        if let Some((width, height)) = self.size {
            let side_ok = |side: u32| {
                (limits.min_side..=limits.max_side).contains(&side)
                    && side.is_multiple_of(limits.size_multiple)
            };
            if !side_ok(width) || !side_ok(height) {
                return Err(OverrideError::Size {
                    width,
                    height,
                    multiple: limits.size_multiple,
                    min: limits.min_side,
                    max: limits.max_side,
                });
            }
            defaults.width = width;
            defaults.height = height;
        }
        if let Some(steps) = self.steps {
            if let Some(fixed) = limits.fixed_steps
                && steps != fixed
            {
                return Err(OverrideError::Steps { steps, fixed });
            }
            defaults.steps = steps;
        }
        Ok(defaults)
    }
}

/// The limits the size grammar judges a request against when no image model
/// is loaded: the request is still refused as malformed before the 503, and
/// no model exists to ask.
pub const UNLOADED_LIMITS: ImageLimits = ImageLimits {
    size_multiple: 16,
    min_side: 256,
    max_side: 2048,
    default_size: (1024, 1024),
    default_steps: 4,
    fixed_steps: None,
};

/// A queued generation and where its answer and its progress go.
#[derive(Debug)]
struct Queued {
    job: ImageGenerationJob,
    /// Where step events go for a streaming caller; `None` for a plain one.
    progress: Option<mpsc::UnboundedSender<ImageStepEvent>>,
    reply: oneshot::Sender<Result<GeneratedImage, AppError>>,
}

/// What the worker thread is started with.
#[derive(Debug)]
pub struct ImageServiceDesc {
    /// The generator, loaded on the image backend.
    pub generator: Arc<dyn ImageGenerator>,
    /// The name the generator answers to on the wire.
    pub model_id: String,
    /// The most generations that may wait in the queue beyond the running
    /// one; the next is refused.
    pub max_queue: usize,
}

/// The handle the HTTP handlers hold.
#[derive(Debug, Clone)]
pub struct ImageService {
    tx: mpsc::Sender<Queued>,
    model_id: String,
    limits: ImageLimits,
    defaults: ImageDefaults,
}

impl ImageService {
    /// Starts the worker thread.
    ///
    /// The thread is not joined: it ends when the last handle is dropped,
    /// which is when the process is done serving, and the queue closing is
    /// what ends its loop.
    ///
    /// # Errors
    ///
    /// The operating system's refusal to start a thread.
    pub fn start(desc: ImageServiceDesc) -> std::io::Result<Self> {
        // A queue of zero would admit nothing beyond the running job and
        // refuse everything else; the floor of one is the audio service's,
        // and `ServeArgs::validate` is what refuses `--image-max-queue 0`
        // for the operator's path.
        let (tx, rx) = mpsc::channel(desc.max_queue.max(1));
        let limits = desc.generator.limits();
        let service = Self {
            tx,
            model_id: desc.model_id.clone(),
            limits,
            defaults: ImageDefaults::of(&limits),
        };
        std::thread::Builder::new()
            .name("shiso-image".to_owned())
            .spawn(move || worker::run_worker(&desc, rx))?;
        Ok(service)
    }

    /// The same service with the operator's default size and step count
    /// applied over the model's own.
    ///
    /// # Errors
    ///
    /// [`OverrideError`] for an override the model would refuse.
    pub fn with_overrides(mut self, overrides: ImageOverrides) -> Result<Self, OverrideError> {
        self.defaults = overrides.resolve(&self.limits)?;
        Ok(self)
    }

    /// What the loaded model accepts.
    #[must_use]
    pub const fn limits(&self) -> &ImageLimits {
        &self.limits
    }

    /// What a request that names no size or steps gets.
    #[must_use]
    pub const fn defaults(&self) -> &ImageDefaults {
        &self.defaults
    }

    /// The name the generator answers to on the wire.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Queues a generation and waits for the image.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] for a job [`ImageGenerationJob::validate`]
    /// refuses - refused here, before it costs a queue slot.
    /// [`AppError::Overloaded`] when the queue is full: the server has more
    /// image work than it holds, and a caller that backs off and retries is
    /// served. [`AppError::Internal`] when the worker thread is gone, or the
    /// generator failed on the device side.
    pub async fn generate(&self, job: ImageGenerationJob) -> Result<GeneratedImage, AppError> {
        self.submit(job, None).await
    }

    /// Queues a generation, forwards each step event to `progress` while it
    /// runs, and waits for the image.
    ///
    /// A caller that stops reading `progress` does not stop the generation;
    /// its final answer still arrives on the returned future.
    ///
    /// # Errors
    ///
    /// The same refusals as [`Self::generate`].
    pub async fn generate_stream(
        &self,
        job: ImageGenerationJob,
        progress: mpsc::UnboundedSender<ImageStepEvent>,
    ) -> Result<GeneratedImage, AppError> {
        self.submit(job, Some(progress)).await
    }

    /// The one path both entry points take: admit, then await the answer.
    async fn submit(
        &self,
        job: ImageGenerationJob,
        progress: Option<mpsc::UnboundedSender<ImageStepEvent>>,
    ) -> Result<GeneratedImage, AppError> {
        self.admit(job, progress)?.await
    }

    /// Validates and queues a generation without waiting for the image:
    /// the synchronous half of [`Self::generate`], for a caller that must
    /// know whether it was admitted before it commits to anything - the
    /// stream handler, which answers a refusal with a status code and only
    /// then spends its 200.
    ///
    /// The returned future is the answer, exactly what [`Self::generate`]
    /// would have awaited.
    ///
    /// # Errors
    ///
    /// The admission refusals of [`Self::generate`]: [`AppError::BadRequest`]
    /// for a job [`ImageGenerationJob::validate`] refuses, and
    /// [`AppError::Overloaded`] when the queue is full. Both are in hand
    /// without an await.
    pub fn admit(
        &self,
        job: ImageGenerationJob,
        progress: Option<mpsc::UnboundedSender<ImageStepEvent>>,
    ) -> Result<ImageAnswer, AppError> {
        // Checked before the queue: a job the generator would refuse must not
        // cost a waiting slot.
        job.validate(&self.limits).map_err(refuse)?;
        let (reply, answer) = oneshot::channel();
        // `try_send`, not `send`: a full queue is refused now rather than
        // awaited, because an image that arrives behind a queue nobody
        // bounded is an answer no client is still waiting for.
        self.tx
            .try_send(Queued {
                job,
                progress,
                reply,
            })
            .map_err(|err| match err {
                mpsc::error::TrySendError::Full(_queued) => AppError::Overloaded {
                    what: "the image queue is full",
                },
                mpsc::error::TrySendError::Closed(_queued) => {
                    AppError::internal("the image queue is closed", WorkerGone)
                }
            })?;
        Ok(Box::pin(async move {
            answer.await.map_err(|_dropped| {
                AppError::internal("the image worker dropped the request", WorkerGone)
            })?
        }))
    }
}

/// The answer to an admitted generation: what the worker reports, or the
/// 500 saying it is gone.
///
/// Boxed so [`ImageService::admit`] can hand the same future to every
/// caller: the worker-gone mapping lives in `admit` alone, and the two
/// entry points that await it cannot drift apart in wording.
pub type ImageAnswer = Pin<Box<dyn Future<Output = Result<GeneratedImage, AppError>> + Send>>;

/// The HTTP failure an [`ImageGenError`] answers with.
///
/// The generator's caller-side refusals (a blank or over-long prompt, dimensions off the
/// model's grid, a step count off its schedule, a zero step count, a size whose decode is
/// over the host's memory budget, a format it does not encode, undecodable
/// reference bytes) are the request's fault and answer 400; its device-side
/// failures (compute, encoding, a checkpoint that went bad) are this
/// server's fault and answer 500 with the cause kept as a real source for
/// the log.
#[must_use]
pub fn refuse(error: ImageGenError) -> AppError {
    let message = error.to_string();
    match &error {
        ImageGenError::EmptyPrompt | ImageGenError::PromptTooLong { .. } => {
            AppError::bad_request(message, Some("prompt"), "invalid_request")
        }
        ImageGenError::InvalidDimensions { .. } | ImageGenError::DecodeTooLarge { .. } => {
            AppError::bad_request(message, Some("size"), "invalid_request")
        }
        ImageGenError::InvalidSteps { .. } | ImageGenError::StepsFixed { .. } => {
            AppError::bad_request(message, Some("steps"), "invalid_request")
        }
        ImageGenError::UnsupportedFormat(_) => {
            AppError::bad_request(message, Some("output_format"), "invalid_request")
        }
        ImageGenError::ReferenceImage(_) => {
            AppError::bad_request(message, Some("image"), "invalid_request")
        }
        ImageGenError::Backend(_) | ImageGenError::Encoding(_) | ImageGenError::Checkpoint(_) => {
            AppError::internal("image generation failed", error)
        }
    }
}

/// The worker thread is not there any more.
///
/// A unit error type rather than a string, so the two places that report it
/// cannot drift apart in wording, and so the cause an operator sees in the
/// log is a type they can grep for.
#[derive(Debug, thiserror::Error)]
#[error("the image worker thread is not running")]
struct WorkerGone;

/// A generation panicked inside the generator.
///
/// The worker catches the panic so one bad job does not end the thread; this
/// is what the caller's 500 keeps as its cause.
#[derive(Debug, thiserror::Error)]
#[error("the image generator panicked")]
struct GeneratorPanicked;

#[cfg(test)]
mod tests {
    use shiso_vision::generator::ImageLimits;

    use super::{ImageDefaults, ImageOverrides, OverrideError};

    const LIMITS: ImageLimits = ImageLimits {
        size_multiple: 32,
        min_side: 256,
        max_side: 2048,
        default_size: (1024, 1024),
        default_steps: 40,
        fixed_steps: None,
    };

    /// With no flag given, the model's own numbers are the defaults, and a
    /// fixed schedule wins over its default step count.
    #[test]
    fn the_model_supplies_the_defaults_when_no_flag_is_given() {
        let defaults = ImageOverrides::default().resolve(&LIMITS).expect("none");
        assert_eq!(
            defaults,
            ImageDefaults {
                steps: 40,
                width: 1024,
                height: 1024
            }
        );
        let fixed = ImageLimits {
            fixed_steps: Some(8),
            ..LIMITS
        };
        assert_eq!(ImageDefaults::of(&fixed).steps, 8);
    }

    /// An explicit flag moves the default but not past what the model accepts.
    #[test]
    fn an_override_the_model_would_refuse_is_refused_at_boot() {
        let moved = ImageOverrides {
            steps: Some(12),
            size: Some((768, 1344)),
        }
        .resolve(&LIMITS)
        .expect("servable");
        assert_eq!((moved.steps, moved.width, moved.height), (12, 768, 1344));
        let off_grid = ImageOverrides {
            size: Some((1008, 1008)),
            ..ImageOverrides::default()
        };
        assert!(matches!(
            off_grid.resolve(&LIMITS),
            Err(OverrideError::Size { multiple: 32, .. })
        ));
        let fixed = ImageLimits {
            fixed_steps: Some(8),
            ..LIMITS
        };
        let wrong_steps = ImageOverrides {
            steps: Some(4),
            ..ImageOverrides::default()
        };
        assert_eq!(
            wrong_steps.resolve(&fixed),
            Err(OverrideError::Steps { steps: 4, fixed: 8 })
        );
    }
}
