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

use std::sync::Arc;

use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageStepEvent,
};
use tokio::sync::{mpsc, oneshot};

use crate::error::AppError;

/// Denoising steps a request that names none gets unless `--image-default-steps`
/// says otherwise.
pub const DEFAULT_STEPS: u32 = 4;

/// The size a request that names none gets unless `--image-default-size` says
/// otherwise, in the `WxH` spelling the flag takes.
pub const DEFAULT_SIZE: &str = "1024x1024";

/// What an image request that names no geometry or step count gets.
///
/// The server-side counterpart of the wire's optional `size` and
/// `aspect_ratio`: the operator chooses these once at boot
/// (`--image-default-size`, `--image-default-steps`) and a caller that names
/// nothing inherits them, exactly as a caller that names no temperature
/// inherits [`AppState::sampling_preset`](crate::state::AppState::sampling_preset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageDefaults {
    /// Denoising steps.
    pub steps: u32,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

impl Default for ImageDefaults {
    /// The published defaults: [`DEFAULT_STEPS`] and the parsed form of
    /// [`DEFAULT_SIZE`]. A test pins that the two spellings stay one fact.
    fn default() -> Self {
        Self {
            steps: DEFAULT_STEPS,
            width: 1024,
            height: 1024,
        }
    }
}

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
        let service = Self {
            tx,
            model_id: desc.model_id.clone(),
        };
        std::thread::Builder::new()
            .name("shiso-image".to_owned())
            .spawn(move || worker::run_worker(&desc, rx))?;
        Ok(service)
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

    /// The one path both entry points take: validate, admit, await.
    async fn submit(
        &self,
        job: ImageGenerationJob,
        progress: Option<mpsc::UnboundedSender<ImageStepEvent>>,
    ) -> Result<GeneratedImage, AppError> {
        // Checked before the queue: a job the generator would refuse must not
        // cost a waiting slot.
        job.validate().map_err(refuse)?;
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
        answer.await.map_err(|_dropped| {
            AppError::internal("the image worker dropped the request", WorkerGone)
        })?
    }
}

/// The HTTP failure an [`ImageGenError`] answers with.
///
/// The generator's caller-side refusals (a blank prompt, dimensions off the
/// 16-grid, a format it does not encode, undecodable reference bytes) are the
/// request's fault and answer 400; its device-side failures (compute,
/// encoding, a checkpoint that went bad) are this server's fault and answer
/// 500 with the cause kept as a real source for the log.
#[must_use]
pub fn refuse(error: ImageGenError) -> AppError {
    let message = error.to_string();
    match &error {
        ImageGenError::EmptyPrompt => {
            AppError::bad_request(message, Some("prompt"), "invalid_request")
        }
        ImageGenError::InvalidDimensions { .. } => {
            AppError::bad_request(message, Some("size"), "invalid_request")
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
    use super::{DEFAULT_SIZE, DEFAULT_STEPS, ImageDefaults, size};

    /// The struct's hardcoded default and the published flag string are one
    /// fact; a change to one without the other would make a booted server and
    /// a test-built state answer a name-nothing request differently.
    #[test]
    fn the_default_defaults_are_the_published_ones() {
        let (width, height) = size::parse_exact_size(DEFAULT_SIZE)
            .expect("the published default size is servable by its own grammar");
        assert_eq!(
            ImageDefaults::default(),
            ImageDefaults {
                steps: DEFAULT_STEPS,
                width,
                height,
            }
        );
    }
}
