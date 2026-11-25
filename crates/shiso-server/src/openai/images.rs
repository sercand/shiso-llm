//! `POST /images`, `POST /v1/images`, `POST /v1/images/generations`.
//!
//! The edit routes ([`crate::openai::image_edits`]) enter at `run_edit` with
//! the same answer paths and a list of reference pictures; these three routes
//! are text-only.
//!
//! This module is the wire's half: it parses the body, picks the blocking or
//! the streaming core of [`shiso_runtime::image`], and spells the IR it gets
//! back as JSON or as SSE frames. The size grammar, the defaults, the plan and
//! the generation loop are the core's.
//!
//! # One request, two readings
//!
//! The blocking answer waits for each finished image and encodes one JSON
//! body; the streamed one forwards the generator's step events as they happen
//! and closes with the finished image. Both readings come from the same
//! image generation jobs, so the `b64_json` of a streamed answer and of a
//! blocking one are the same bytes for the same request.
//!
//! # What the wire values become
//!
//! See [`shiso_runtime::image`]: the size grammar, `auto`, and the default
//! step count. `quality` is echoed, `moderation`, `user` and `session_id` are
//! accepted and unused, each for the reason spelled out on the wire field.
//! `background` is the core's: see `shiso_models::matted` for what each value
//! answers.
//!
//! # Event names
//!
//! The generation routes stream `image_generation.*` events and the edit
//! route `image_edit.*`, as the published shape names them; the frames are
//! otherwise the same.
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

use crate::error::ErrorResponse as _;
use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse as _, Response};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use futures::StreamExt as _;
use shiso_ir::image::{Edit, Generate, Image, Images, ProducedBackground, Quality, Step, Usage};
use tokio::sync::mpsc;

use crate::api::sse::Frame;
use crate::error::{AppError, type_for_status};
use crate::openai::images_wire::{
    ImageData, ImageGenCompletedEvent, ImageGenPartialImageEvent, ImageGenerationRequest,
    ImageGenerationResponse, ImageGenerationUsage, InputTokensDetails, OutputTokensDetails,
    quality_name,
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
async fn answer(state: &Arc<AppState>, body: &Bytes) -> Result<Response, AppError> {
    let parsed: ImageGenerationRequest = serde_json::from_slice(body).map_err(|error| {
        AppError::bad_request(
            format!("the request body is not an image generation request: {error}"),
            None,
            "invalid_request",
        )
    })?;
    // Refused before the image-model question: a body this server will never
    // act on is the caller's fault whether or not a generator is loaded. The
    // core judges the size and the output format in the same position for the
    // same reason, so the 503 is reserved for a well-formed request this
    // server could have run.
    parsed.validate()?;
    let ir = parsed.to_ir()?;
    if parsed.stream.unwrap_or(false) {
        // Every refusal the core would make before it queues anything, ahead
        // of the task and of the debug lines, so a refused request spawns
        // nothing and the lines fire only for one that will run.
        shiso_runtime::image::preflight(state, &ir, &[])?;
        note_ignored(state, &parsed);
        stream(state, Job::Generate(ir)).await
    } else {
        note_ignored(state, &parsed);
        Ok(buffered(shiso_runtime::image::generate(state, ir).await?))
    }
}

/// Answers an edit: the generation route's own request plus the references,
/// through the same answer paths, so a streamed or buffered edit is the same
/// response a generation is.
pub(crate) async fn run_edit(
    state: &Arc<AppState>,
    parsed: ImageGenerationRequest,
    references: Vec<Vec<u8>>,
) -> Result<Response, AppError> {
    parsed.validate()?;
    let edit = Edit {
        generate: parsed.to_ir()?,
        references,
    };
    if parsed.stream.unwrap_or(false) {
        shiso_runtime::image::preflight(state, &edit.generate, &edit.references)?;
        note_ignored(state, &parsed);
        stream(state, Job::Edit(edit)).await
    } else {
        note_ignored(state, &parsed);
        Ok(buffered(shiso_runtime::image::edit(state, edit).await?))
    }
}

/// Says in the debug log what the request named that this server does not act
/// on.
fn note_ignored(state: &AppState, parsed: &ImageGenerationRequest) {
    if let Some(quality) = parsed.quality.as_deref() {
        tracing::debug!(
            quality,
            "quality is echoed and otherwise unused: this server has one rendering path"
        );
    }
    if let Some(moderation) = parsed.moderation.as_deref() {
        tracing::debug!(
            moderation,
            "moderation is accepted and unused: this server runs no content filter"
        );
    }
    if let Some(model) = parsed.model.as_deref()
        && let Some(service) = state.image.as_ref()
    {
        tracing::debug!(
            model,
            served = service.model_id(),
            "the named model is answered by whichever generator is loaded"
        );
    }
}

/// What a streamed request runs: a generation, or an edit with references.
enum Job {
    Generate(Generate),
    Edit(Edit),
}

impl Job {
    /// The prefix of the route's stream event names.
    const fn events(&self) -> &'static str {
        match self {
            Self::Generate(_) => "image_generation",
            Self::Edit(_) => "image_edit",
        }
    }
}

/// The blocking answer: one JSON body for every image.
fn buffered(images: Images) -> Response {
    let size = images.images.first().map(size_of).unwrap_or_default();
    let body = ImageGenerationResponse {
        created: images.created,
        background: background_name(images.background),
        output_format: images.output_format,
        quality: quality_echo(images.quality),
        size,
        data: images
            .images
            .into_iter()
            .map(|image| ImageData {
                b64_json: BASE64_STANDARD.encode(&image.bytes),
                media_type: Some(image.media_type),
            })
            .collect(),
        usage: Some(usage_wire(images.usage)),
    };
    Json(body).into_response()
}

fn usage_wire(usage: Usage) -> ImageGenerationUsage {
    ImageGenerationUsage {
        input_tokens: usage.input_tokens(),
        input_tokens_details: InputTokensDetails {
            text_tokens: usage.text_input_tokens,
            image_tokens: usage.image_input_tokens,
        },
        output_tokens: usage.image_output_tokens,
        output_tokens_details: OutputTokensDetails {
            image_tokens: usage.image_output_tokens,
            text_tokens: 0,
        },
        total_tokens: usage.total_tokens(),
    }
}

/// A picture's dimensions as the wire spells them.
fn size_of(image: &Image) -> String {
    format!("{}x{}", image.width, image.height)
}

const fn background_name(background: ProducedBackground) -> &'static str {
    match background {
        ProducedBackground::Opaque => "opaque",
        ProducedBackground::Transparent => "transparent",
    }
}

/// The quality echo: the value the caller named, except `auto`, which names
/// no quality to report.
fn quality_echo(quality: Option<Quality>) -> Option<&'static str> {
    quality
        .filter(|quality| *quality != Quality::Auto)
        .map(quality_name)
}

/// The streamed answer: frames from a task that runs the generations, after
/// the first one has been admitted.
///
/// The handler waits for the task's admission answer before committing to
/// the stream, so an admission refusal still answers with a status code;
/// the answer on success is a comment frame, so the head is flushed at
/// admission rather than at the first decoded partial.
async fn stream(state: &Arc<AppState>, job: Job) -> Result<Response, AppError> {
    let (tx, mut rx) = mpsc::unbounded_channel::<Result<Event, AppError>>();
    tokio::spawn(run_stream(Arc::clone(state), job, tx));
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

/// Runs the core and sends the frames as they happen.
///
/// A task rather than a stream state machine because the generation and the
/// client's reading run at different speeds by design: a caller that stops
/// reading does not stop the generation (the service's own rule), and the
/// frames it missed are simply not sent.
async fn run_stream(
    state: Arc<AppState>,
    job: Job,
    tx: mpsc::UnboundedSender<Result<Event, AppError>>,
) {
    // Whether any item has been promised to the client yet (the acceptance
    // comment counts): before the first one a failure is still answerable
    // as a status code; after it, the 200 is spent and the failure has to
    // arrive as content.
    let mut streaming = false;
    let events = job.events();
    let sink = |step: Step| {
        let event = match step {
            Step::Admitted => accepted_event(),
            Step::Partial { index, bytes } => partial_frame(events, &bytes, index).into_event(),
            Step::Completed {
                created,
                image,
                usage,
                output_format,
                quality,
            } => completed_frame(
                events,
                created,
                &image,
                Completion {
                    usage: usage_wire(usage),
                    output_format,
                    quality: quality_echo(quality),
                },
            )
            .into_event(),
        };
        streaming = true;
        tx.send(Ok(event)).is_ok()
    };
    let result = match job {
        Job::Generate(request) => {
            shiso_runtime::image::generate_stream(&state, request, sink).await
        }
        Job::Edit(request) => shiso_runtime::image::edit_stream(&state, request, sink).await,
    };
    match result {
        Ok(()) => {
            let _ = tx.send(Ok(Frame::done().into_event()));
        }
        Err(error) => fail(&tx, error, streaming),
    }
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

/// The SSE comment that says the request was admitted. A comment carries no
/// event and no data - every SSE client ignores it - so the wire event shape
/// is unchanged; its whole job is to be the first item out, which is what
/// flushes the 200 head at admission rather than at the first partial.
fn accepted_event() -> Event {
    Event::default().comment("accepted")
}

/// One partial-image frame.
///
/// Partials are always PNG previews regardless of the request's
/// `output_format`: they are progress, not the product, and the completed
/// frame is the one that honors the format.
fn partial_frame(events: &'static str, bytes: &[u8], index: u32) -> Frame {
    let name = if events == "image_edit" {
        "image_edit.partial_image"
    } else {
        "image_generation.partial_image"
    };
    let body = ImageGenPartialImageEvent {
        r#type: name,
        partial_image_index: index,
        b64_json: BASE64_STANDARD.encode(bytes),
    };
    Frame::named(
        name,
        serde_json::to_string(&body).expect("an event of strings and numbers always serializes"),
    )
}

/// What a completed frame says besides the picture.
struct Completion {
    usage: ImageGenerationUsage,
    output_format: String,
    quality: Option<&'static str>,
}

/// The completed-image frame for one finished image. Its `usage` is the
/// request's running total, so the last frame of a multi-image stream carries
/// the whole request's accounting.
fn completed_frame(
    events: &'static str,
    created: u64,
    image: &Image,
    completion: Completion,
) -> Frame {
    let name = if events == "image_edit" {
        "image_edit.completed"
    } else {
        "image_generation.completed"
    };
    let body = ImageGenCompletedEvent {
        r#type: name,
        created,
        b64_json: BASE64_STANDARD.encode(&image.bytes),
        media_type: Some(image.media_type.clone()),
        background: background_name(image.background),
        output_format: completion.output_format,
        quality: completion.quality,
        size: size_of(image),
        usage: Some(completion.usage),
    };
    Frame::named(
        name,
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

#[cfg(test)]
mod tests {
    use shiso_ir::image::{Image, Images, ProducedBackground, Quality, Usage};

    use super::{Completion, buffered, completed_frame, partial_frame, usage_wire};

    fn png(bytes: &[u8]) -> Image {
        Image {
            bytes: bytes.to_vec(),
            media_type: "image/png".to_owned(),
            width: 64,
            height: 32,
            background: ProducedBackground::Opaque,
        }
    }

    /// The IR a core returns is spelled as the image models' body.
    #[tokio::test]
    async fn images_are_spelled_as_the_buffered_body() {
        let response = buffered(Images {
            created: 5,
            images: vec![png(&[1, 2, 3]), png(&[4])],
            usage: Usage {
                text_input_tokens: 2,
                image_input_tokens: 3,
                image_output_tokens: u32::MAX,
            },
            background: ProducedBackground::Transparent,
            output_format: "png".to_owned(),
            quality: Some(Quality::Auto),
        });
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("reads");
        assert_eq!(
            std::str::from_utf8(&bytes).expect("utf8"),
            concat!(
                r#"{"created":5,"background":"transparent","#,
                r#""data":[{"b64_json":"AQID","media_type":"image/png"},"#,
                r#"{"b64_json":"BA==","media_type":"image/png"}],"#,
                r#""output_format":"png","size":"64x32","#,
                r#""usage":{"input_tokens":5,"input_tokens_details":{"text_tokens":2,"image_tokens":3},"#,
                r#""output_tokens":4294967295,"output_tokens_details":{"image_tokens":4294967295,"text_tokens":0},"#,
                r#""total_tokens":4294967295}}"#
            )
        );
    }

    #[test]
    fn steps_are_spelled_as_the_named_frames() {
        let partial = partial_frame("image_generation", &[1, 2, 3], 4);
        assert_eq!(
            partial.name.as_deref(),
            Some("image_generation.partial_image")
        );
        assert_eq!(
            partial.data,
            r#"{"type":"image_generation.partial_image","partial_image_index":4,"b64_json":"AQID"}"#
        );
        let usage = Usage {
            text_input_tokens: 1,
            image_input_tokens: 0,
            image_output_tokens: 10,
        };
        let completed = completed_frame(
            "image_edit",
            7,
            &png(&[4]),
            Completion {
                usage: usage_wire(usage),
                output_format: "png".to_owned(),
                quality: Some("high"),
            },
        );
        assert_eq!(completed.name.as_deref(), Some("image_edit.completed"));
        assert_eq!(
            completed.data,
            concat!(
                r#"{"type":"image_edit.completed","created":7,"b64_json":"BA==","#,
                r#""media_type":"image/png","background":"opaque","output_format":"png","#,
                r#""quality":"high","size":"64x32","#,
                r#""usage":{"input_tokens":1,"input_tokens_details":{"text_tokens":1,"image_tokens":0},"#,
                r#""output_tokens":10,"output_tokens_details":{"image_tokens":10,"text_tokens":0},"#,
                r#""total_tokens":11}}"#
            )
        );
    }
}
