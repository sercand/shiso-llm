//! Embeddings: the service the HTTP handler submits to, and the thread that
//! runs the model.
//!
//! The work splits where the device does. Everything before the model -
//! choosing the task prompt, tokenizing, decoding and resizing pictures,
//! sampling video frames, framing audio - runs on a blocking thread of the
//! request's own, so a large picture is resized while the device is busy with
//! someone else's input. The model runs on one thread of its own, which is the
//! only thread that calls it, one request at a time in arrival order: a pass
//! over one input already fills the device. Callers beyond the running one
//! wait in a bounded queue and a full queue answers 503, the image service's
//! rule for the image service's reason.

mod prepare;
mod worker;

use std::sync::Arc;

use shiso_ir::{EmbedRequest, EmbedResponse, EmbedUsage};
use shiso_models::embedder::{EmbedError, EmbedFrontend, EmbedPart, Embedder, EmbedderInfo};
use shiso_vision::{FfmpegPolicy, ImageLimits};
use tokio::sync::{mpsc, oneshot};

use crate::error::AppError;
use crate::tokenizer::Vocabulary;

/// The most inputs one request may carry, `OpenAI`'s own limit: each is a
/// whole pass, and an unbounded list would let one request hold the queue.
pub const MAX_INPUTS: usize = 2048;

/// What the request side may decode, as the operator's flags set it.
#[derive(Debug, Clone)]
pub struct MediaRules {
    /// Decode limits for one picture or video frame.
    pub image_limits: ImageLimits,
    /// Whether an http or https address may be named. This build fetches
    /// nothing either way; the flag only changes which refusal is given.
    pub allow_remote: bool,
    /// The subprocess video decoder, or `None` when only GIF is read.
    pub video_ffmpeg: Option<FfmpegPolicy>,
    /// Decode limits for one audio clip.
    pub audio_limits: shiso_audio::DecodeLimits,
    /// The subprocess audio decoder, or `None` when only WAV is read.
    pub audio_ffmpeg: Option<FfmpegPolicy>,
}

/// What the worker thread is started with.
#[derive(Debug)]
pub struct EmbeddingServiceDesc {
    /// The model, loaded.
    pub embedder: Box<dyn Embedder>,
    /// The vocabulary text is tokenized with.
    pub vocabulary: Arc<Vocabulary>,
    /// The name the model answers to on the wire.
    pub model_id: String,
    /// The most requests that may wait beyond the running one.
    pub max_queue: usize,
    /// What may be decoded.
    pub media: MediaRules,
}

/// One queued request: every input's parts, and where the vectors go.
#[derive(Debug)]
struct Queued {
    inputs: Vec<Vec<EmbedPart>>,
    reply: oneshot::Sender<Result<Vec<shiso_models::embedder::Embedding>, AppError>>,
}

/// What the request side reads, shared by every handle.
#[derive(Debug)]
struct Shared {
    info: EmbedderInfo,
    frontend: EmbedFrontend,
    vocabulary: Arc<Vocabulary>,
    model_id: String,
    media: MediaRules,
}

/// The handle the HTTP handler holds.
#[derive(Debug, Clone)]
pub struct EmbeddingService {
    tx: mpsc::Sender<Queued>,
    shared: Arc<Shared>,
}

impl EmbeddingService {
    /// Starts the worker thread.
    ///
    /// Not joined: it ends when the last handle is dropped, which is when the
    /// process is done serving.
    ///
    /// # Errors
    ///
    /// The operating system's refusal to start a thread.
    pub fn start(desc: EmbeddingServiceDesc) -> std::io::Result<Self> {
        let EmbeddingServiceDesc {
            embedder,
            vocabulary,
            model_id,
            max_queue,
            media,
        } = desc;
        let shared = Arc::new(Shared {
            info: embedder.info().clone(),
            frontend: embedder.frontend(),
            vocabulary,
            model_id,
            media,
        });
        let (tx, rx) = mpsc::channel(max_queue.max(1));
        std::thread::Builder::new()
            .name("shiso-embed".to_owned())
            .spawn(move || worker::run(embedder, rx))?;
        Ok(Self { tx, shared })
    }

    /// The name the model answers to on the wire.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.shared.model_id
    }

    /// The model's fixed facts.
    #[must_use]
    pub fn info(&self) -> &EmbedderInfo {
        &self.shared.info
    }

    /// Embeds every input of `request`.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] for an input the model cannot read - an
    /// unknown task, a width it was not trained for, media it has no tower
    /// for, an input past its length - refused before it costs a queue slot
    /// where that is knowable; [`AppError::Overloaded`] when the queue is
    /// full; [`AppError::Internal`] for a device failure.
    pub async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, AppError> {
        let dim = self.width(request.dimensions)?;
        if request.inputs.is_empty() {
            return Err(AppError::bad_request(
                "input is empty",
                Some("input"),
                "invalid_request",
            ));
        }
        if request.inputs.len() > MAX_INPUTS {
            return Err(AppError::bad_request(
                format!(
                    "input holds {} items; at most {MAX_INPUTS} are embedded per request",
                    request.inputs.len()
                ),
                Some("input"),
                "invalid_request",
            ));
        }
        let prompt = prepare::prompt_for(&self.shared.info, request.task.as_deref())?;
        let shared = Arc::clone(&self.shared);
        let inputs = request.inputs;
        let began = std::time::Instant::now();
        // Tokenizing and decoding are CPU work measured in tens of
        // milliseconds for a picture; off the async runtime.
        let parts = tokio::task::spawn_blocking(move || {
            inputs
                .iter()
                .map(|input| prepare::parts(&shared, prompt.as_deref(), input))
                .collect::<Result<Vec<_>, AppError>>()
        })
        .await
        .map_err(|join| AppError::internal("preparing embedding inputs", join))??;
        tracing::debug!(
            inputs = parts.len(),
            prepare_ms = began.elapsed().as_secs_f64() * 1e3,
            "embedding inputs prepared"
        );
        let (reply, answer) = oneshot::channel();
        self.tx
            .try_send(Queued {
                inputs: parts,
                reply,
            })
            .map_err(|err| match err {
                mpsc::error::TrySendError::Full(_) => AppError::Overloaded {
                    what: "the embedding queue is full",
                },
                mpsc::error::TrySendError::Closed(_) => {
                    AppError::internal("the embedding queue is closed", WorkerGone)
                }
            })?;
        let embeddings = answer.await.map_err(|_dropped| {
            AppError::internal("the embedding worker dropped the request", WorkerGone)
        })??;
        let mut usage = EmbedUsage::default();
        let mut vectors = Vec::with_capacity(embeddings.len());
        for embedding in embeddings {
            let u = embedding.usage;
            usage.text = usage.text.saturating_add(u.text);
            usage.image = usage.image.saturating_add(u.image);
            usage.video = usage.video.saturating_add(u.video);
            usage.audio = usage.audio.saturating_add(u.audio);
            vectors
                .push(shiso_models::embedder::matryoshka(&embedding.vector, dim).map_err(refuse)?);
        }
        Ok(EmbedResponse {
            model: self.shared.model_id.clone(),
            vectors,
            encoding: request.encoding,
            usage,
        })
    }

    /// The width a request's vectors are cut to: the one it named, which
    /// must be a width the model was trained for, or the full width.
    fn width(&self, dimensions: Option<u32>) -> Result<usize, AppError> {
        let info = &self.shared.info;
        let Some(dimensions) = dimensions else {
            return Ok(info.dim);
        };
        let dim = usize::try_from(dimensions).unwrap_or(usize::MAX);
        if info.truncations.contains(&dim) {
            Ok(dim)
        } else {
            Err(AppError::bad_request(
                format!(
                    "dimensions {dimensions} is not a width this model was trained to be cut to; \
                     it takes {:?}",
                    info.truncations
                ),
                Some("dimensions"),
                "invalid_request",
            ))
        }
    }
}

/// The HTTP failure an [`EmbedError`] answers with: the caller's input is a
/// 400, anything else is this server's 500.
#[must_use]
pub fn refuse(error: EmbedError) -> AppError {
    if error.is_input() {
        AppError::bad_request(error.to_string(), Some("input"), "invalid_request")
    } else {
        AppError::internal("embedding failed", error)
    }
}

/// The worker thread is not there any more.
#[derive(Debug, thiserror::Error)]
#[error("the embedding worker thread is not running")]
struct WorkerGone;
