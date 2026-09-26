//! `POST /v1/audio/transcriptions`: the form in, a transcript out.
//!
//! The handler reads and checks the form, decodes the audio off the runtime,
//! and hands the job to the audio worker. A blocking answer waits for the
//! finished transcript and encodes it in the requested format; a streamed one
//! waits only for the job's first event - so that a request the worker refuses
//! at admission is still answered with a status code - and then forwards the
//! events as they come.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse as _, Response};
use futures::StreamExt as _;
use shiso_audio::{AudioError, Language};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::audio::{JobFailure, TranscriptEvent, TranscriptionJob};
use crate::error::AppError;
use crate::openai::transcription_encode::{TranscriptionFrames, body};
use crate::openai::transcription_wire::{Field, ResponseFormat, ServedAudio, TranscriptionForm};
use crate::state::AppState;

/// The handler.
pub async fn transcriptions(State(state): State<Arc<AppState>>, multipart: Multipart) -> Response {
    match answer(&state, multipart).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

/// A failure of the job, as the status its cause deserves.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct JobError(String);

fn failure_to_error(failure: JobFailure) -> AppError {
    if failure.caller {
        AppError::bad_request(failure.message, Some("file"), "invalid_audio")
    } else {
        AppError::internal("the transcription failed", JobError(failure.message))
    }
}

/// A decode refusal: the request's audio is at fault, except when the decode
/// could not run at all.
fn decode_error(error: AudioError) -> AppError {
    match error {
        AudioError::Io { .. } | AudioError::Device(_) => {
            AppError::internal("decoding the audio", error)
        }
        other => AppError::bad_request(other.to_string(), Some("file"), "invalid_audio"),
    }
}

async fn read_fields(mut multipart: Multipart, limit: usize) -> Result<Vec<Field>, AppError> {
    // The body limit surfaces here, as a read error, so it is told apart from
    // a malformed form: one is fixed by a smaller file, the other by a client.
    let bad = |error: axum::extract::multipart::MultipartError| {
        if error.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
            return AppError::TooLarge {
                what: "a transcription upload (--audio-max-upload-mb)",
                limit,
            };
        }
        AppError::bad_request(
            format!("the body is not a readable multipart form: {error}"),
            None,
            "invalid_request",
        )
    };
    let mut fields = Vec::new();
    while let Some(field) = multipart.next_field().await.map_err(bad)? {
        let name = field.name().unwrap_or_default().to_owned();
        let bytes = field.bytes().await.map_err(bad)?;
        fields.push(Field { name, bytes });
    }
    Ok(fields)
}

async fn answer(state: &AppState, multipart: Multipart) -> Result<Response, AppError> {
    let audio = state.audio.as_ref().ok_or(AppError::NotReady {
        what: "a speech recognizer (start the server with --asr-model)",
    })?;
    let form =
        TranscriptionForm::from_fields(read_fields(multipart, audio.max_upload_bytes()).await?)?;
    form.validate(&ServedAudio {
        asr_id: audio.asr_id(),
        has_diarizer: audio.has_diarizer(),
    })?;
    let pcm = audio
        .decode(form.file.clone())
        .await
        .map_err(decode_error)?;
    let diarize = form.response_format == ResponseFormat::DiarizedJson;
    let mut events = audio.submit(TranscriptionJob {
        pcm,
        language: form.language.clone().map_or(Language::Auto, Language::Tag),
        diarize,
    })?;
    if form.stream {
        let first = events.recv().await.ok_or_else(worker_gone)?;
        if let TranscriptEvent::Failed(failure) = first {
            return Err(failure_to_error(failure));
        }
        return Ok(stream(first, events, diarize));
    }
    while let Some(event) = events.recv().await {
        match event {
            TranscriptEvent::Done(transcript) => {
                let (content_type, text) =
                    body(form.response_format, &transcript, &form.granularities);
                return Ok(([(header::CONTENT_TYPE, content_type)], text).into_response());
            }
            TranscriptEvent::Failed(failure) => return Err(failure_to_error(failure)),
            TranscriptEvent::Delta(_) | TranscriptEvent::Segment(_) => {}
        }
    }
    Err(worker_gone())
}

fn worker_gone() -> AppError {
    AppError::internal(
        "the transcription worker",
        JobError("the job ended without a transcript".into()),
    )
}

/// The events as server-sent frames, starting with the one already taken.
fn stream(
    first: TranscriptEvent,
    rest: UnboundedReceiver<TranscriptEvent>,
    diarized: bool,
) -> Response {
    let mut frames = TranscriptionFrames::new(diarized);
    let events = futures::stream::once(async move { first })
        .chain(futures::stream::unfold(rest, |mut rx| async move {
            rx.recv().await.map(|event| (event, rx))
        }));
    let out = events
        .scan(false, |ended, event| {
            // Nothing after the last event: a stream that went on past its
            // `done` or `error` frame would be a second answer.
            if *ended {
                return futures::future::ready(None);
            }
            *ended = matches!(event, TranscriptEvent::Done(_) | TranscriptEvent::Failed(_));
            futures::future::ready(Some(event))
        })
        .flat_map(move |event| futures::stream::iter(frames.on_event(&event)))
        .map(|frame| Ok::<Event, Infallible>(frame.into_event()));
    Sse::new(out)
        .keep_alive(KeepAlive::default())
        .into_response()
}
