//! The multipart form `POST /v1/audio/transcriptions` takes.
//!
//! Parsing and validation are two steps over plain `(name, bytes)` fields, so
//! the rules are testable without an HTTP body. Every field this server
//! cannot honour is refused by name rather than ignored: a caller who sent a
//! temperature expects it to change the answer, and a server that dropped it
//! would answer a question nobody asked.

use axum::body::Bytes;
use shiso_audio::Context;

use crate::error::AppError;

/// One multipart field.
#[derive(Debug, Clone)]
pub struct Field {
    /// Its name.
    pub name: String,
    /// Its value.
    pub bytes: Bytes,
}

/// The body shapes a transcription is answered in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseFormat {
    /// `{text, usage}`.
    Json,
    /// The text alone.
    Text,
    /// Text, language, duration, segments and words.
    VerboseJson,
    /// `SubRip` subtitles.
    Srt,
    /// `WebVTT` subtitles.
    Vtt,
    /// Speaker-labelled segments.
    DiarizedJson,
}

/// Timestamp detail a verbose body carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// Segment times.
    Segment,
    /// Word times.
    Word,
}

/// What this server serves, as the form is checked against it.
#[derive(Debug, Clone, Copy)]
pub struct ServedAudio<'a> {
    /// The id the `model` field must name.
    pub asr_id: &'a str,
    /// Whether `diarized_json` can be answered.
    pub has_diarizer: bool,
}

/// A parsed transcription request.
#[derive(Debug, Clone)]
pub struct TranscriptionForm {
    /// The audio file's bytes.
    pub file: Bytes,
    /// The model the caller named.
    pub model: String,
    /// The language the caller named; `None` lets the model decide.
    pub language: Option<String>,
    /// The body shape.
    pub response_format: ResponseFormat,
    /// Timestamp detail, in the order asked.
    pub granularities: Vec<Granularity>,
    /// Whether to answer as a stream of events.
    pub stream: bool,
    /// What the caller says the audio contains (`prompt`, `keywords`);
    /// empty for neither. Whether the recognizer can use it is checked
    /// when it is encoded, not here.
    pub context: Context,
}

fn refuse(param: &str, message: impl Into<String>) -> AppError {
    AppError::bad_request(message, Some(param), "invalid_request")
}

fn text(field: &Field) -> Result<&str, AppError> {
    std::str::from_utf8(&field.bytes)
        .map(str::trim)
        .map_err(|_| refuse(&field.name, format!("{} is not text", field.name)))
}

/// Refuses a field whose value this server cannot honour; every other field
/// passes.
fn refuse_unhonoured(field: &Field) -> Result<(), AppError> {
    match field.name.as_str() {
        "temperature" => {
            let value = text(field)?;
            let t: f64 = value.parse().map_err(|_| {
                refuse(
                    "temperature",
                    format!("temperature {value:?} is not a number"),
                )
            })?;
            if t.abs() > 0.0 {
                return Err(refuse(
                    "temperature",
                    "this model decodes greedily; temperature must be 0",
                ));
            }
            Ok(())
        }
        "include[]" | "include" if text(field)? == "logprobs" => {
            Err(refuse("include", "this model reports no token logprobs"))
        }
        "chunking_strategy" if text(field)? != "auto" => Err(refuse(
            "chunking_strategy",
            "only the auto chunking strategy is served",
        )),
        _ => Ok(()),
    }
}

impl TranscriptionForm {
    /// Reads the form's fields.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming the field for a missing file or model,
    /// a value that does not parse, or a field whose value this server cannot
    /// honour.
    pub fn from_fields(fields: Vec<Field>) -> Result<Self, AppError> {
        let mut file = None;
        let mut model = None;
        let mut language = None;
        let mut response_format = ResponseFormat::Json;
        let mut granularities = Vec::new();
        let mut stream = false;
        let mut context = Context::default();
        for field in fields {
            match field.name.as_str() {
                "file" => file = Some(field.bytes),
                "model" => model = Some(text(&field)?.to_owned()),
                "language" => {
                    let value = text(&field)?;
                    language = (!value.is_empty()).then(|| value.to_owned());
                }
                "response_format" => {
                    response_format = match text(&field)? {
                        "json" => ResponseFormat::Json,
                        "text" => ResponseFormat::Text,
                        "verbose_json" => ResponseFormat::VerboseJson,
                        "srt" => ResponseFormat::Srt,
                        "vtt" => ResponseFormat::Vtt,
                        "diarized_json" => ResponseFormat::DiarizedJson,
                        other => {
                            return Err(refuse(
                                "response_format",
                                format!(
                                    "response_format {other:?} is not one of json, text, \
                                     verbose_json, srt, vtt, diarized_json"
                                ),
                            ));
                        }
                    };
                }
                "timestamp_granularities[]" | "timestamp_granularities" => {
                    granularities.push(match text(&field)? {
                        "segment" => Granularity::Segment,
                        "word" => Granularity::Word,
                        other => {
                            return Err(refuse(
                                "timestamp_granularities",
                                format!("timestamp granularity {other:?} is not segment or word"),
                            ));
                        }
                    });
                }
                "stream" => {
                    stream = match text(&field)? {
                        "true" => true,
                        "false" => false,
                        other => {
                            return Err(refuse(
                                "stream",
                                format!("stream {other:?} is not a boolean"),
                            ));
                        }
                    };
                }
                "prompt" => {
                    let value = text(&field)?;
                    context.prompt = (!value.is_empty()).then(|| value.to_owned());
                }
                "keywords[]" | "keywords" => {
                    let value = text(&field)?;
                    if value.is_empty() {
                        return Err(refuse("keywords", "a keyword is empty"));
                    }
                    context.keywords.push(value.to_owned());
                }
                _ => refuse_unhonoured(&field)?,
            }
        }
        Ok(Self {
            file: file.ok_or_else(|| refuse("file", "the request carries no audio file"))?,
            model: model.ok_or_else(|| refuse("model", "the request names no model"))?,
            language,
            response_format,
            granularities,
            stream,
            context,
        })
    }

    /// Checks the form against what this server serves.
    ///
    /// # Errors
    ///
    /// [`AppError::UnknownModel`] for a model other than the recognizer, and
    /// [`AppError::BadRequest`] naming the field for diarized output with no
    /// diarizer or a stream of a format that cannot be streamed.
    pub fn validate(&self, served: &ServedAudio<'_>) -> Result<(), AppError> {
        if self.model != served.asr_id {
            return Err(AppError::UnknownModel {
                model: self.model.clone(),
            });
        }
        if self.response_format == ResponseFormat::DiarizedJson && !served.has_diarizer {
            return Err(refuse(
                "response_format",
                "diarized_json needs a diarizer; start the server with --diarization-model",
            ));
        }
        if self.stream
            && matches!(
                self.response_format,
                ResponseFormat::Srt | ResponseFormat::Vtt | ResponseFormat::VerboseJson
            )
        {
            return Err(refuse(
                "stream",
                "srt, vtt and verbose_json are whole-file formats and cannot be streamed",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "transcription_wire_tests.rs"]
mod tests;
