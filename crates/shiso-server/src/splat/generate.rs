//! `POST /v1/3d/generations`: one picture in, one Gaussian-splat file out, a
//! `.ply` or a `.glb`.
//!
//! This module is the wire's half: it parses the body, picks the blocking or
//! the streaming core of [`shiso_runtime::splat`], and spells the IR it gets
//! back as JSON or as SSE frames. The defaults, the validation and the
//! generation loop are the core's.
//!
//! # One request, two readings
//!
//! The blocking answer waits for the finished splat and encodes one JSON
//! body; the streamed one forwards the generator's step events as they happen
//! and closes with the finished splat. Both readings come from the same job,
//! so a streamed request's `.ply` and a blocking one's are the same bytes.
//!
//! # Two ways to ask for a format, two shapes of answer
//!
//! `output_format` in the body picks the file and keeps the JSON answer:
//! `data[0].b64_ply` or `data[0].b64_glb`. An `Accept` header that names a
//! splat file's MIME type (`model/gltf-binary`, `model/ply`) picks the file
//! too, and asks for it as the response body itself, with that
//! `Content-Type` - what a `curl -o object.glb` or a browser download wants.
//! A streamed request cannot answer with a file, so there the header only
//! picks the format and the completed frame carries the base64.
//!
//! # Why a streamed request waits for admission
//!
//! A request the queue refuses must answer with a status code, not with a 200
//! and an error frame. The handler therefore awaits the first frame before it
//! commits to the stream, exactly as the image route does.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt as _;
use shiso_ir::splat::Splat;
use tokio::sync::mpsc;

use super::splat_wire::{
    SplatAdmittedEvent, SplatCompletedEvent, SplatData, SplatGenerationRequest,
    SplatGenerationResponse, SplatGenerationUsage, SplatProgressEvent,
};
use crate::api::sse::Frame;
use crate::error::{AppError, ErrorResponse as _};
use crate::state::AppState;

/// The handler for `POST /v1/3d/generations`.
///
/// The body is taken as bytes and parsed here rather than through a typed
/// extractor, for the chat route's reason: a malformed body must answer with
/// this surface's error shape, not the extractor's.
pub async fn generate_splats(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Every `Accept` line counts: a client may send the header more than once.
    let accepted = headers
        .get_all(header::ACCEPT)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .find_map(super::splat_wire::accepted_format);
    match answer(&state, accepted, &body).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

/// The handler, with its failures still typed.
async fn answer(
    state: &Arc<AppState>,
    accepted: Option<shiso_ir::splat::Format>,
    body: &Bytes,
) -> Result<Response, AppError> {
    let parsed: SplatGenerationRequest = serde_json::from_slice(body).map_err(|error| {
        AppError::bad_request(
            format!("the request body is not a 3D generation request: {error}"),
            None,
            "invalid_request",
        )
    })?;
    let stream = parsed.stream.unwrap_or(false);
    let ir = parsed.into_ir(accepted).map_err(|refusal| {
        AppError::bad_request(refusal.message, Some(refusal.field), "invalid_request")
    })?;
    // The knobs are judged before the model question, as the image route judges
    // them: a body this server will never act on is the caller's fault whether
    // or not a splat model is loaded, and answering 503 first would send a
    // caller to fix a flag only for the same request to start failing 400.
    shiso_runtime::splat::validate(&ir, shiso_vision::splat::SplatLimits::default())?;
    let service = state.splat()?;
    let seed = ir.seed.unwrap_or_else(shiso_runtime::ids::entropy_u64);
    let job =
        shiso_runtime::splat::build_job(ir, service.defaults(), shiso_core::ids::Seed::new(seed))?;
    if stream {
        stream_splat(state, job).await
    } else {
        let splat = service.generate(job).await?;
        Ok(if accepted.is_some() {
            raw(splat)
        } else {
            buffered(&splat)
        })
    }
}

/// The blocking answer to a request whose `Accept` named the file's type:
/// the file itself.
fn raw(splat: shiso_vision::splat::GeneratedSplat) -> Response {
    ([(header::CONTENT_TYPE, splat.media_type)], splat.bytes).into_response()
}

/// The blocking answer: one JSON body for the finished splat.
fn buffered(splat: &shiso_vision::splat::GeneratedSplat) -> Response {
    let body = SplatGenerationResponse {
        created: shiso_runtime::ids::unix_timestamp(),
        data: vec![SplatData::encode(&splat.bytes, splat.media_type)],
        usage: SplatGenerationUsage {
            prompt_tokens: 0,
            completion_tokens: splat.completion_tokens,
            total_tokens: splat.completion_tokens,
        },
    };
    Json(body).into_response()
}

/// The streamed answer: frames from a task that runs the generation, after
/// the first one has been admitted.
///
/// The handler waits for the task's admission answer before committing to the
/// stream, so an admission refusal still answers with a status code.
async fn stream_splat(
    state: &Arc<AppState>,
    job: shiso_vision::splat::SplatJob,
) -> Result<Response, AppError> {
    let (tx, mut rx) = mpsc::unbounded_channel::<Result<Event, AppError>>();
    tokio::spawn(run_stream(Arc::clone(state), job, tx));
    let first = rx.recv().await.ok_or_else(|| {
        AppError::internal("the splat stream ended before it started", StreamGone)
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

/// Runs the service and sends the frames as they happen.
///
/// A task rather than a stream state machine because the generation and the
/// client's reading run at different speeds by design: a caller that stops
/// reading does not stop the generation, and the frames it missed are simply
/// not sent.
///
/// Admission goes through [`SplatService::admit`] rather than
/// `generate_stream`, because that is what lets a refusal answer with a
/// status code: the admitted frame is sent only once the job holds a queue
/// slot.
async fn run_stream(
    state: Arc<AppState>,
    job: shiso_vision::splat::SplatJob,
    tx: mpsc::UnboundedSender<Result<Event, AppError>>,
) {
    let service = match state.splat() {
        Ok(service) => service.clone(),
        Err(error) => {
            let _ = tx.send(Err(error));
            return;
        }
    };
    let (progress, mut events) = mpsc::unbounded_channel::<shiso_vision::splat::SplatStepEvent>();
    // The generator emits on the worker thread; the frames are built here, so
    // the wire stays on this side of the service.
    let forward = tokio::spawn({
        let tx = tx.clone();
        async move {
            while let Some(event) = events.recv().await {
                let frame = progress_frame(event.step, event.total_steps);
                if tx.send(Ok(frame.into_event())).is_err() {
                    break;
                }
            }
        }
    });
    let answer = match service.admit(job, Some(progress)) {
        Ok(answer) => answer,
        Err(error) => {
            // Nothing has been promised yet, so the handler still answers
            // with a status code.
            let _ = tx.send(Err(error));
            return;
        }
    };
    let _admitted = tx.send(Ok(admitted_frame().into_event()));
    let outcome = answer.await;
    let _ = forward.await;
    match outcome {
        Ok(splat) => {
            let frame = completed_frame(
                shiso_runtime::ids::unix_timestamp(),
                &Splat {
                    bytes: splat.bytes,
                    media_type: splat.media_type.to_owned(),
                },
                shiso_ir::splat::Usage {
                    prompt_tokens: 0,
                    completion_tokens: splat.completion_tokens,
                },
            );
            let _ = tx.send(Ok(frame.into_event()));
            let _ = tx.send(Ok(Frame::done().into_event()));
        }
        Err(error) => fail(&tx, &error),
    }
}

/// Reports a failure at the right altitude: while nothing has been promised
/// it is the first item and still answers as a status code; after the 200 is
/// spent it goes out as the error frame and the terminator, because a stream
/// cannot take its 200 back.
fn fail(tx: &mpsc::UnboundedSender<Result<Event, AppError>>, error: &AppError) {
    let _ = tx.send(Ok(error_frame(error).into_event()));
    let _ = tx.send(Ok(Frame::done().into_event()));
}

/// The first frame: the job was admitted to the queue.
fn admitted_frame() -> Frame {
    let body = SplatAdmittedEvent {
        r#type: "splat_generation.admitted",
    };
    Frame::named(
        "splat_generation.admitted",
        serde_json::to_string(&body).expect("an event of one string always serializes"),
    )
}

/// One progress frame.
fn progress_frame(step: u32, total_steps: u32) -> Frame {
    let body = SplatProgressEvent {
        r#type: "splat_generation.progress",
        step,
        total_steps,
    };
    Frame::named(
        "splat_generation.progress",
        serde_json::to_string(&body).expect("an event of numbers always serializes"),
    )
}

/// The completed frame for the finished splat.
fn completed_frame(created: u64, splat: &Splat, usage: shiso_ir::splat::Usage) -> Frame {
    let body = SplatCompletedEvent {
        r#type: "splat_generation.completed",
        created,
        file: SplatData::encode(&splat.bytes, &splat.media_type),
        usage: SplatGenerationUsage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens(),
        },
    };
    Frame::named(
        "splat_generation.completed",
        serde_json::to_string(&body).expect("an event of strings and numbers always serializes"),
    )
}

/// The frame a failure after the 200 is spent goes out as.
///
/// The same error object every other failure on this server sends: a client
/// that parses every frame as JSON should not have to special-case the one
/// frame that says what went wrong.
fn error_frame(error: &AppError) -> Frame {
    let message = match error {
        AppError::Internal { .. } => "internal server error".to_owned(),
        other => other.to_string(),
    };
    Frame::data(
        serde_json::json!({
            "error": {
                "message": message,
                "type": crate::error::type_for_status(error.status()),
                "param": serde_json::Value::Null,
                "code": error.code(),
            }
        })
        .to_string(),
    )
}

/// The stream task is gone before its first frame.
#[derive(Debug, thiserror::Error)]
#[error("the splat stream task is not running")]
struct StreamGone;

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

    use crate::state::AppState;

    #[test]
    fn no_service_is_a_503_that_names_the_flag() {
        let state = AppState::without_language_model();
        let err = state.splat().expect_err("no splat model");
        assert_eq!(err.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(err.to_string().contains("no splat model"), "{err}");
    }
}
