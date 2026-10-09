//! `POST /images`, `POST /v1/images`, `POST /v1/images/generations`.
//!
//! # One request, two readings
//!
//! The blocking answer waits for each finished image and encodes one JSON
//! body; the streamed one forwards the generator's step events as they happen
//! and closes with the finished image. Both readings come from the same
//! [`ImageGenerationJob`]s, so the `b64_json` of a streamed answer and of a
//! blocking one are the same bytes for the same request.
//!
//! # What the wire values become
//!
//! The size grammar lives in [`crate::image::size`]: `WxH` is rounded to the
//! generator's 16-grid and clamped to its bounds, the named aspect ratios
//! become the table's concrete pixels, and a named resolution (`2K`) becomes
//! its square. A size of `auto` is an omitted size: the caller gets the
//! server default. `steps` names the denoising steps per image; omitted, it
//! is the operator's `--image-default-steps`. `quality`, `user` and
//! `session_id` are accepted and unused, each for the reason spelled out on
//! the wire field.
//!
//! # Why a streamed request waits for admission
//!
//! A request the service refuses at admission - a job the generator would
//! reject, a queue at its bound - is answered with a status code, not with a
//! 200 whose first frame says it failed. That is the transcription route's
//! pattern for the same reason. The handler therefore waits only for the
//! admission answer, not for the first content frame: admission sends an SSE
//! comment (`: accepted`), which no client has to parse, and the 200 head
//! reaches the caller the moment the job is queued rather than when the
//! first partial is decoded. Past that point a failure arrives as an error
//! frame, because a stream that has sent its 200 cannot take it back.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse as _, Response};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use futures::StreamExt as _;
use shiso_core::ids::Seed;
use shiso_vision::generator::{
    GeneratedImage, ImageGenerationJob, ImageOutputFormat, ImageStepEvent,
};
use tokio::sync::mpsc;

use crate::api::sse::Frame;
use crate::error::{AppError, type_for_status};
use crate::image::ImageService;
use crate::image::{self, size};
use crate::openai::images_wire::{
    ImageData, ImageGenCompletedEvent, ImageGenPartialImageEvent, ImageGenerationRequest,
    ImageGenerationResponse, ImageGenerationUsage,
};
use crate::state::AppState;

/// The handler for all three spellings of the route.
///
/// The body is taken as bytes and parsed here rather than through a typed
/// extractor, for the chat route's reason: a malformed body must answer with
/// this surface's error shape, not the extractor's.
pub async fn generate_images(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    match answer(&state, &body).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

/// The handler, with its failures still typed.
async fn answer(state: &AppState, body: &Bytes) -> Result<Response, AppError> {
    let parsed: ImageGenerationRequest = serde_json::from_slice(body).map_err(|error| {
        AppError::bad_request(
            format!("the request body is not an image generation request: {error}"),
            None,
            "invalid_request",
        )
    })?;
    // Refused before the image-model question: a body this server will never
    // act on is the caller's fault whether or not a generator is loaded. The
    // plan is built in the same position for the same reason - a malformed
    // size or output format is a 400 with no generator loaded too, so the 503
    // is reserved for a well-formed request this server could have run.
    parsed.validate()?;
    let plan = Plan::from_wire(&parsed, state.image.as_ref())?;
    let service = state.image()?;
    if let Some(quality) = parsed.quality.as_deref() {
        tracing::debug!(
            quality,
            "quality is accepted and ignored: this server has one rendering path"
        );
    }
    if let Some(model) = parsed.model.as_deref() {
        tracing::debug!(
            model,
            served = service.model_id(),
            "the named model is answered by whichever generator is loaded"
        );
    }
    if plan.stream {
        stream(service.clone(), plan).await
    } else {
        buffered(service, &plan).await
    }
}

/// One validated request as the server will run it: the job template and how
/// many of it to make.
///
/// A named struct rather than a dozen arguments: the wire fields arrive
/// together, get checked together, and both answer paths need all of them.
#[derive(Debug)]
struct Plan {
    /// Prompt every image of the request carries.
    prompt: String,
    /// Output width, normalized to the grid.
    width: u32,
    /// Output height, normalized to the grid.
    height: u32,
    /// Denoising steps: the named count, or the server default.
    steps: u32,
    /// The seed of the first image: the named one, or entropy so the images
    /// of one request differ from each other.
    seed: u64,
    /// Encoding every image is returned in.
    output_format: ImageOutputFormat,
    /// Compression, when the caller named one.
    output_compression: Option<u32>,
    /// How many images to generate, `1..=10`.
    n: usize,
    /// Whether the caller asked for the SSE stream.
    stream: bool,
}

impl Plan {
    /// Checks the wire values the format alone cannot judge - the size
    /// grammar, the output format - and folds in the server defaults.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] for a size string neither shape, and for an
    /// output format the generator does not encode.
    fn from_wire(
        parsed: &ImageGenerationRequest,
        service: Option<&ImageService>,
    ) -> Result<Self, AppError> {
        // With no model loaded the request is still judged for shape, against
        // the published limits, so a malformed one stays a 400 ahead of the 503.
        let limits = service.map_or(image::UNLOADED_LIMITS, |s| *s.limits());
        let defaults = service.map_or_else(|| image::ImageDefaults::of(&limits), |s| *s.defaults());
        // `auto` is an omitted size: the caller is asking for this server's
        // default, which is exactly what an omitted size gets. It is read
        // out here rather than in the grammar because the grammar has no
        // business knowing what the default is.
        let size = parsed.size.as_deref().filter(|value| !size::is_auto(value));
        let (width, height) = match (size, parsed.aspect_ratio.as_deref()) {
            // `size` wins when both are named: it is the more specific of the
            // two, and a client that sends both wants the pixels it spelled.
            (Some(value), _) => {
                size::normalize_size(value, &limits).map_err(|error| size_error(&error, "size"))?
            }
            (None, Some(value)) => size::normalize_size(value, &limits)
                .map_err(|error| size_error(&error, "aspect_ratio"))?,
            (None, None) => (defaults.width, defaults.height),
        };
        // The generator's own format parser is the one authority on what it
        // encodes; its refusal is the request's fault and names the field.
        let output_format = parsed
            .output_format
            .as_deref()
            .map(str::parse::<ImageOutputFormat>)
            .transpose()
            .map_err(image::refuse)?;
        Ok(Self {
            prompt: parsed.prompt.clone(),
            width,
            height,
            steps: parsed.steps.unwrap_or(defaults.steps),
            // A named seed makes the request reproducible and image `i` of it
            // gets `seed + i`; an unnamed one draws a base from the process's
            // one entropy source so the `n` images of one request differ.
            seed: parsed.seed.unwrap_or_else(crate::ids::entropy_u64),
            output_format: output_format.unwrap_or_default(),
            output_compression: parsed.output_compression,
            n: parsed.n.unwrap_or(1),
            stream: parsed.stream.unwrap_or(false),
        })
    }

    /// The job for image `index` of the request.
    fn job(&self, index: usize) -> ImageGenerationJob {
        let offset = u64::try_from(index).unwrap_or(u64::MAX);
        ImageGenerationJob {
            prompt: self.prompt.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps,
            seed: Seed::new(self.seed.wrapping_add(offset)),
            output_format: self.output_format,
            output_compression: self.output_compression,
            input_references: Vec::new(),
        }
    }
}

/// A size refusal, naming the wire field that carried the value.
fn size_error(error: &size::SizeError, param: &str) -> AppError {
    AppError::bad_request(error.to_string(), Some(param), "invalid_request")
}

/// The token accounting across the images of one request: each image's
/// estimate, summed.
#[derive(Debug, Clone, Copy, Default)]
struct UsageTotals {
    prompt: u32,
    completion: u32,
}

impl UsageTotals {
    fn add(&mut self, image: &GeneratedImage) {
        self.prompt = self.prompt.saturating_add(image.prompt_tokens);
        self.completion = self.completion.saturating_add(image.completion_tokens);
    }

    fn wire(self) -> ImageGenerationUsage {
        ImageGenerationUsage {
            prompt_tokens: self.prompt,
            completion_tokens: self.completion,
            total_tokens: self.prompt.saturating_add(self.completion),
        }
    }
}

/// The blocking answer: every image generated in turn, then one JSON body.
///
/// In turn, not at once: the service runs one generation at a time anyway,
/// and a caller that asked for three gets them in request order.
///
/// All-or-nothing: an overload partway through an `n > 1` request answers
/// the whole request with the 503 and no image body. A JSON body carrying
/// fewer than the `n` images the caller asked for would be a silent
/// contract violation - the caller could not tell a partial success from a
/// truncated list - so the refusal is the whole answer and a retry redoes
/// the lot.
async fn buffered(service: &ImageService, plan: &Plan) -> Result<Response, AppError> {
    let mut data = Vec::with_capacity(plan.n);
    let mut usage = UsageTotals::default();
    for index in 0..plan.n {
        let generated = service.generate(plan.job(index)).await?;
        usage.add(&generated);
        data.push(ImageData {
            b64_json: BASE64_STANDARD.encode(&generated.bytes),
            media_type: Some(generated.media_type.to_owned()),
        });
    }
    let body = ImageGenerationResponse {
        created: crate::ids::unix_timestamp(),
        data,
        usage: Some(usage.wire()),
    };
    Ok(Json(body).into_response())
}

/// The streamed answer: frames from a task that runs the generations, after
/// the first one has been admitted.
///
/// The handler waits for the task's admission answer before committing to
/// the stream, so an admission refusal still answers with a status code;
/// the answer on success is a comment frame, so the head is flushed at
/// admission rather than at the first decoded partial.
async fn stream(service: ImageService, plan: Plan) -> Result<Response, AppError> {
    let (tx, mut rx) = mpsc::unbounded_channel::<Result<Event, AppError>>();
    tokio::spawn(run_stream(service, plan, tx));
    let first = rx.recv().await.ok_or_else(|| {
        AppError::internal("the image stream ended before it started", StreamGone)
    })?;
    let first = match first {
        Err(error) => return Err(error),
        Ok(event) => event,
    };
    let rest = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    });
    let events = futures::stream::once(async move { Ok(first) })
        .chain(rest)
        .map(|item| match item {
            Ok(event) => Ok::<_, std::convert::Infallible>(event),
            // Only the first item can carry an error and it was taken above;
            // a later one goes out as the error frame a client already knows
            // how to read rather than as a dropped connection.
            Err(error) => Ok(error_frame(&error).into_event()),
        });
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// Runs the plan's generations and sends the frames as they happen.
///
/// A task rather than a stream state machine because the generation and the
/// client's reading run at different speeds by design: a caller that stops
/// reading does not stop the generation (the service's own rule), and the
/// frames it missed are simply not sent.
async fn run_stream(
    service: ImageService,
    plan: Plan,
    tx: mpsc::UnboundedSender<Result<Event, AppError>>,
) {
    let created = crate::ids::unix_timestamp();
    let mut usage = UsageTotals::default();
    // The wire's `partial_image_index` counts across ALL images of one
    // request: with n>1 the counter keeps incrementing into the second image
    // rather than restarting, which is why it lives outside the loop.
    let mut partial_index = 0u32;
    // Whether any item has been promised to the client yet (the acceptance
    // comment counts): before the first one a failure is still answerable
    // as a status code; after it, the 200 is spent and the failure has to
    // arrive as content.
    let mut streaming = false;
    for index in 0..plan.n {
        let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<ImageStepEvent>();
        // Admission is decided synchronously - validate and the queue slot -
        // so a refusal is still answerable as a status code while nothing
        // has gone out, and an accepted job can flush the head before the
        // first step runs.
        let mut running = match service.admit(plan.job(index), Some(progress_tx)) {
            Ok(answer) => answer,
            Err(error) => {
                fail(&tx, error, streaming);
                return;
            }
        };
        // The acceptance comment: it carries no event and no data, so the
        // wire event shape is unchanged, but it is the first item out of
        // this task, which is the moment the handler commits to the 200.
        if !send(&tx, Ok(accepted_event()), &mut streaming) {
            return;
        }
        // The worker drops its progress sender before it sends the reply,
        // so `recv` starts answering `None` while the answer is still
        // pending; a bare `recv` branch would then hand back `None` forever
        // and spin beside the running generation, so the branch is disabled
        // once the channel says it is done.
        let mut progress_open = true;
        let answer = loop {
            tokio::select! {
                step = progress_rx.recv(), if progress_open => {
                    match step {
                        Some(event) => {
                            // A step event that carries no bytes produces no
                            // frame: this event's payload is the partial
                            // image, and a step tick has nothing to put in
                            // it. The keep-alive is what tells a waiting
                            // client the generation is still alive.
                            if !is_final_step(&event)
                                && let Some(bytes) = event.partial_image
                            {
                                let frame = partial_frame(bytes, partial_index);
                                partial_index += 1;
                                if !send(&tx, Ok(frame.into_event()), &mut streaming) {
                                    return;
                                }
                            }
                        }
                        None => progress_open = false,
                    }
                }
                answer = &mut running => break answer,
            }
        };
        // The worker sends every step event before the reply, so anything
        // still buffered belongs to this image and goes out before its
        // completed frame.
        while let Ok(event) = progress_rx.try_recv() {
            if !is_final_step(&event)
                && let Some(bytes) = event.partial_image
            {
                let frame = partial_frame(bytes, partial_index);
                partial_index += 1;
                if !send(&tx, Ok(frame.into_event()), &mut streaming) {
                    return;
                }
            }
        }
        match answer {
            Ok(generated) => {
                usage.add(&generated);
                let frame = completed_frame(created, &generated, usage.wire());
                if !send(&tx, Ok(frame.into_event()), &mut streaming) {
                    return;
                }
            }
            Err(error) => {
                fail(&tx, error, streaming);
                return;
            }
        }
    }
    let _ = tx.send(Ok(Frame::done().into_event()));
}

/// Reports a failure at the right altitude: while nothing has been promised
/// it is the first item and still answers as a status code; after the 200 is
/// spent it goes out as the error frame and the terminator, because a stream
/// cannot take its 200 back.
fn fail(tx: &mpsc::UnboundedSender<Result<Event, AppError>>, error: AppError, streaming: bool) {
    if !streaming {
        let _ = tx.send(Err(error));
        return;
    }
    let _ = tx.send(Ok(error_frame(&error).into_event()));
    let _ = tx.send(Ok(Frame::done().into_event()));
}

/// Sends one item and records that the stream has begun. `false` when the
/// receiver is gone: the client stopped reading, and a generation nobody is
/// waiting on does not owe the connection any more frames.
fn send(
    tx: &mpsc::UnboundedSender<Result<Event, AppError>>,
    event: Result<Event, AppError>,
    streaming: &mut bool,
) -> bool {
    *streaming = true;
    tx.send(event).is_ok()
}

/// The SSE comment that says the request was admitted. A comment carries no
/// event and no data - every SSE client ignores it - so the wire event shape
/// is unchanged; its whole job is to be the first item out, which is what
/// flushes the 200 head at admission rather than at the first partial.
fn accepted_event() -> Event {
    Event::default().comment("accepted")
}

/// Whether this step is the last one. Its partial decode is the finished
/// image byte for byte, so a partial frame for it would make every client
/// decode the same picture twice for no new information; the completed
/// frame carries it once.
fn is_final_step(event: &ImageStepEvent) -> bool {
    event.step.saturating_add(1) >= event.total_steps
}

/// One partial-image frame.
///
/// Partials are always PNG previews regardless of the request's
/// `output_format`: they are progress, not the product, and the completed
/// frame is the one that honors the format.
fn partial_frame(bytes: Vec<u8>, index: u32) -> Frame {
    let body = ImageGenPartialImageEvent {
        r#type: "image_generation.partial_image",
        partial_image_index: index,
        b64_json: BASE64_STANDARD.encode(bytes),
    };
    Frame::named(
        "image_generation.partial_image",
        serde_json::to_string(&body).expect("an event of strings and numbers always serializes"),
    )
}

/// The completed-image frame for one finished image. Its `usage` is the
/// request's running total, so the last frame of a multi-image stream carries
/// the whole request's accounting.
fn completed_frame(created: u64, image: &GeneratedImage, usage: ImageGenerationUsage) -> Frame {
    let body = ImageGenCompletedEvent {
        r#type: "image_generation.completed",
        created,
        b64_json: BASE64_STANDARD.encode(&image.bytes),
        media_type: Some(image.media_type.to_owned()),
        usage: Some(usage),
    };
    Frame::named(
        "image_generation.completed",
        serde_json::to_string(&body).expect("an event of strings and numbers always serializes"),
    )
}

/// A failure that arrives after the stream has spent its 200: the same error
/// object every other failure on this surface sends, as the frame's content,
/// because a client that parses every frame as JSON should not have to
/// special-case the one frame that says what went wrong (the chat surface's
/// reasoning, verbatim).
fn error_frame(error: &AppError) -> Frame {
    let message = match error {
        AppError::Internal { .. } => "internal server error".to_owned(),
        other => other.to_string(),
    };
    Frame::data(
        serde_json::json!({
            "error": {
                "message": message,
                "type": type_for_status(error.status()),
                "param": serde_json::Value::Null,
                "code": error.code(),
            }
        })
        .to_string(),
    )
}

/// The stream task ended before it said anything.
#[derive(Debug, thiserror::Error)]
#[error("the image stream task ended before its first frame")]
struct StreamGone;
