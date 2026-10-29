//! `POST /v1/audio/transcriptions`: the form in, a transcript out.
//!
//! The handler reads and checks the form, decodes the audio off the runtime,
//! and hands the job to the audio worker. A blocking answer waits for the
//! finished transcript and encodes it in the requested format; a streamed one
//! waits only for the job's first event - so that a request the worker refuses
//! at admission is still answered with a status code - and then forwards the
//! events as they come.
//!
//! A server with no recognizer whose language model is a speech decoder
//! answers through that model instead ([`answer_speech`]): one greedy turn
//! asked with the `preset` instruction, encoded in the same formats.

use crate::error::ErrorResponse as _;
use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse as _, Response};
use futures::StreamExt as _;
use shiso_audio::{AudioError, Language};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::audio::{JobFailure, Transcript, TranscriptEvent, TranscriptionJob};
use crate::error::AppError;
use crate::openai::transcription_encode::{TranscriptionFrames, body};
use crate::openai::transcription_wire::Granularity;
use crate::openai::transcription_wire::{Field, ResponseFormat, ServedAudio, TranscriptionForm};
use crate::state::AppState;
use shiso_runtime::api::transcribe::{
    PresetError, SpeechDecoder, TranscribeRequest, parse_preset, transcribe,
};

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
    if state.audio.is_none()
        && let Some(speech) = state.speech.as_ref()
    {
        return answer_speech(state, speech, multipart).await;
    }
    let audio = state.audio.as_ref().ok_or(AppError::NotReady {
        what: "a speech recognizer (start the server with --asr-model)",
    })?;
    let form =
        TranscriptionForm::from_fields(read_fields(multipart, audio.max_upload_bytes()).await?)?;
    if form.preset.is_some() {
        return Err(AppError::bad_request(
            "preset is served by a speech decoder; this server's recognizer takes none",
            Some("preset"),
            "invalid_request",
        ));
    }
    form.validate(&ServedAudio {
        asr_id: audio.asr_id(),
        has_diarizer: audio.has_diarizer(),
    })?;
    // Refused before the audio is decoded: a context the recognizer cannot
    // carry is the request's fault whatever the audio holds.
    let context = audio.encode_context(&form.context)?;
    let pcm = audio
        .decode(form.file.clone())
        .await
        .map_err(decode_error)?;
    let diarize = form.response_format == ResponseFormat::DiarizedJson;
    let mut events = audio.submit(TranscriptionJob {
        pcm,
        language: form.language.clone().map_or(Language::Auto, Language::Tag),
        context,
        diarize,
        streaming: form.stream,
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

/// A preset the speech decoder will not be asked with.
fn preset_error(error: &PresetError) -> AppError {
    let param = match error {
        PresetError::Unknown { .. } => "preset",
        PresetError::LanguageRequired { .. } | PresetError::LanguageNotAccepted { .. } => {
            "language"
        }
        PresetError::PromptNotAccepted { .. } | PresetError::PromptRequired => "prompt",
    };
    AppError::bad_request(error.to_string(), Some(param), "invalid_request")
}

/// Refuses what a speech decoder's transcript cannot carry.
///
/// It has text and nothing else: no word or segment times (its timestamp head
/// is not wired yet), so the formats built from them are refused by name
/// rather than answered with empty timings.
fn check_speech_form(form: &TranscriptionForm, model_id: &str) -> Result<(), AppError> {
    let refuse = |param: &str, message: &str| {
        Err(AppError::bad_request(
            message,
            Some(param),
            "invalid_request",
        ))
    };
    if form.model != model_id {
        return Err(AppError::UnknownModel {
            model: form.model.clone(),
        });
    }
    if form.granularities.contains(&Granularity::Word) {
        return refuse(
            "timestamp_granularities",
            "word timestamps are not produced by this speech decoder yet",
        );
    }
    if !form.granularities.is_empty() {
        return refuse(
            "timestamp_granularities",
            "segment timestamps are not produced by this speech decoder",
        );
    }
    if !matches!(
        form.response_format,
        ResponseFormat::Json | ResponseFormat::Text
    ) {
        return refuse(
            "response_format",
            "this speech decoder answers json or text; the other formats need timestamps",
        );
    }
    if !form.context.keywords.is_empty() {
        return refuse(
            "keywords",
            "this speech decoder takes context terms as `prompt`, not keywords",
        );
    }
    Ok(())
}

/// The route over a speech decoder: the form checked against what it can
/// answer, the upload decoded, one greedy turn, the transcript encoded.
async fn answer_speech(
    state: &AppState,
    speech: &SpeechDecoder,
    multipart: Multipart,
) -> Result<Response, AppError> {
    let form =
        TranscriptionForm::from_fields(read_fields(multipart, speech.max_upload_bytes()).await?)?;
    check_speech_form(&form, &state.model_id)?;
    let preset = parse_preset(
        form.preset.as_deref().unwrap_or("verbatim"),
        form.language.as_deref(),
        form.context.prompt.as_deref(),
    )
    .map_err(|error| preset_error(&error))?;
    let pcm = speech
        .decode(form.file.clone())
        .await
        .map_err(decode_error)?;
    let duration = pcm.duration_secs();
    let out = transcribe(state, TranscribeRequest { pcm, preset }).await?;
    let transcript = Transcript {
        text: out.text,
        language: None,
        duration,
        words: Vec::new(),
        segments: Vec::new(),
        diarized: false,
    };
    if form.stream {
        // The whole text is one delta: the loop guard reads the finished
        // turn, so nothing is sent before it ends.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let _ = tx.send(TranscriptEvent::Done(transcript.clone()));
        drop(tx);
        return Ok(stream(TranscriptEvent::Delta(transcript.text), rx, false));
    }
    let (content_type, text) = body(form.response_format, &transcript, &[]);
    Ok(([(header::CONTENT_TYPE, content_type)], text).into_response())
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

#[cfg(test)]
mod speech_tests {
    use super::check_speech_form;
    use crate::openai::transcription_wire::{Granularity, ResponseFormat, TranscriptionForm};

    fn form() -> TranscriptionForm {
        TranscriptionForm {
            file: axum::body::Bytes::from_static(b"RIFF"),
            model: "speech".into(),
            language: None,
            response_format: ResponseFormat::Json,
            granularities: Vec::new(),
            stream: false,
            context: shiso_audio::Context::default(),
            preset: None,
        }
    }

    fn refused_param(form: &TranscriptionForm) -> String {
        let error = check_speech_form(form, "speech").expect_err("refused");
        format!("{error:?}")
    }

    #[test]
    fn a_plain_form_is_served() {
        assert!(check_speech_form(&form(), "speech").is_ok());
    }

    #[test]
    fn word_timestamps_are_refused_by_name() {
        let mut f = form();
        f.granularities.push(Granularity::Word);
        let error = refused_param(&f);
        assert!(error.contains("timestamp_granularities"), "{error}");
        assert!(error.contains("word timestamps"), "{error}");
    }

    #[test]
    fn formats_built_from_timestamps_are_refused() {
        for format in [
            ResponseFormat::VerboseJson,
            ResponseFormat::Srt,
            ResponseFormat::Vtt,
            ResponseFormat::DiarizedJson,
        ] {
            let mut f = form();
            f.response_format = format;
            assert!(refused_param(&f).contains("response_format"));
        }
    }

    #[test]
    fn another_model_is_unknown() {
        let mut f = form();
        f.model = "whisper".into();
        assert!(refused_param(&f).contains("UnknownModel"));
    }
}
