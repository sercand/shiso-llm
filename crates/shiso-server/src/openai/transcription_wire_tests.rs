//! The transcription form: what it accepts, what it refuses, and the field it
//! names when it refuses.

use axum::body::Bytes;
use axum::http::StatusCode;

use super::{Field, Granularity, ResponseFormat, ServedAudio, TranscriptionForm};

fn field(name: &str, value: &str) -> Field {
    Field {
        name: name.to_owned(),
        bytes: Bytes::from(value.to_owned()),
    }
}

fn file() -> Field {
    Field {
        name: "file".to_owned(),
        bytes: Bytes::from_static(b"RIFF....WAVE"),
    }
}

const SERVED: ServedAudio<'static> = ServedAudio {
    asr_id: "nemotron",
    has_diarizer: false,
};

fn parse(fields: Vec<Field>) -> Result<TranscriptionForm, crate::error::AppError> {
    TranscriptionForm::from_fields(fields).and_then(|form| form.validate(&SERVED).map(|()| form))
}

fn refused(fields: Vec<Field>, param: &str) {
    let error = parse(fields).expect_err("refused");
    assert_eq!(error.status(), StatusCode::BAD_REQUEST, "{error}");
    assert!(
        matches!(&error, crate::error::AppError::BadRequest { param: Some(p), .. } if p == param),
        "{error:?}"
    );
}

#[test]
fn a_bare_request_takes_the_documented_defaults() {
    let form = parse(vec![file(), field("model", "nemotron")]).expect("accepted");
    assert_eq!(form.response_format, ResponseFormat::Json);
    assert!(form.granularities.is_empty());
    assert!(!form.stream);
    assert!(form.language.is_none());
    assert_eq!(&form.file[..], b"RIFF....WAVE");
}

#[test]
fn every_documented_field_is_read() {
    let form = parse(vec![
        file(),
        field("model", "nemotron"),
        field("language", "en"),
        field("response_format", "verbose_json"),
        field("timestamp_granularities[]", "word"),
        field("timestamp_granularities[]", "segment"),
        field("temperature", "0"),
        field("chunking_strategy", "auto"),
        field("prompt", ""),
    ])
    .expect("accepted");
    assert_eq!(form.response_format, ResponseFormat::VerboseJson);
    assert_eq!(
        form.granularities,
        vec![Granularity::Word, Granularity::Segment]
    );
    assert_eq!(form.language.as_deref(), Some("en"));
}

#[test]
fn a_missing_file_is_refused() {
    refused(vec![field("model", "nemotron")], "file");
}

#[test]
fn another_model_is_not_found() {
    let error = parse(vec![file(), field("model", "whisper-1")]).expect_err("refused");
    assert_eq!(error.status(), StatusCode::NOT_FOUND, "{error}");
    assert!(error.to_string().contains("whisper-1"), "{error}");
}

#[test]
fn what_this_model_cannot_do_is_refused_by_field() {
    let base = || vec![file(), field("model", "nemotron")];
    let with = |extra: Field| {
        let mut fields = base();
        fields.push(extra);
        fields
    };
    refused(with(field("temperature", "0.2")), "temperature");
    refused(with(field("temperature", "warm")), "temperature");
    refused(with(field("include[]", "logprobs")), "include");
    refused(with(field("response_format", "xml")), "response_format");
    refused(
        with(field("timestamp_granularities[]", "char")),
        "timestamp_granularities",
    );
    refused(
        with(field("chunking_strategy", r#"{"type":"server_vad"}"#)),
        "chunking_strategy",
    );
    refused(with(field("stream", "maybe")), "stream");
    refused(
        with(field("response_format", "diarized_json")),
        "response_format",
    );
    for format in ["srt", "vtt", "verbose_json"] {
        let mut fields = with(field("response_format", format));
        fields.push(field("stream", "true"));
        refused(fields, "stream");
    }
}

#[test]
fn keywords_and_a_prompt_become_the_context() {
    let form = parse(vec![
        file(),
        field("model", "nemotron"),
        field("keywords[]", " premium plan "),
        field("keywords[]", "AC-42"),
        field("keywords", "billing"),
        field("prompt", " A support call. "),
    ])
    .expect("accepted");
    assert_eq!(
        form.context.keywords,
        vec!["premium plan", "AC-42", "billing"]
    );
    assert_eq!(form.context.prompt.as_deref(), Some("A support call."));
}

#[test]
fn an_empty_prompt_is_no_context() {
    let form = parse(vec![
        file(),
        field("model", "nemotron"),
        field("prompt", "  "),
    ])
    .expect("accepted");
    assert!(form.context.is_empty());
}

#[test]
fn an_empty_keyword_is_refused() {
    refused(
        vec![file(), field("model", "nemotron"), field("keywords[]", " ")],
        "keywords",
    );
}

#[test]
fn diarized_json_is_accepted_when_a_diarizer_is_loaded() {
    let form = TranscriptionForm::from_fields(vec![
        file(),
        field("model", "nemotron"),
        field("response_format", "diarized_json"),
        field("stream", "true"),
    ])
    .expect("parses");
    form.validate(&ServedAudio {
        asr_id: "nemotron",
        has_diarizer: true,
    })
    .expect("a diarizer is loaded");
    assert!(form.stream);
}
