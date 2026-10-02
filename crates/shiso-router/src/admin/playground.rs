//! The admin chat console: send a hand-written message to one model on one
//! backend and see what comes back. It exists to answer "is this setup
//! wired up correctly", so it deliberately runs the request through
//! `proxy::proxy_request` — the same routing, header allowlist, credential
//! injection, model rename, usage teeing, pricing and logging a real client
//! request gets — rather than talking to the upstream itself.
//!
//! Two things differ from client traffic: the target is pinned to the
//! backend the operator picked (no failover masking a broken provider), and
//! the row is attributed to the synthetic `admin console` key.
//!
//! Streaming is out of scope: the console collects the whole response.
//!
//! Transcriptions are the one non-chat, non-JSON call: the UI sends the
//! audio base64-encoded in the same JSON request, and the console turns it
//! into the multipart form a client SDK would send.
//!
//! The scoring endpoints (`/v1/decisions`, `/v1/systemone`) are the other
//! non-chat calls: the console sends the newest user turn as `input` (or
//! the system prompt as `state`) plus the operator's `questions` JSON
//! array, and shows one line per scored answer.

use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::body::to_bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Method, Uri, header};
use axum::response::IntoResponse;
use base64::Engine;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::BackendKind;
use crate::error::AppError;
use crate::proxy::{self, Dispatch};
use crate::state::AppState;
use crate::tokens::KeyRecord;

/// The endpoints the console can call and the wire format each speaks —
/// the "API type" choice. One table so the UI's list and the validation
/// here cannot drift apart.
const ENDPOINTS: &[(&str, BackendKind)] = &[
    ("/v1/chat/completions", BackendKind::Openai),
    ("/v1/responses", BackendKind::Openai),
    ("/v1/embeddings", BackendKind::Openai),
    ("/v1/messages", BackendKind::Anthropic),
    ("/v1/audio/transcriptions", BackendKind::Openai),
    ("/v1/decisions", BackendKind::Openai),
    ("/v1/systemone", BackendKind::Openai),
];

const TRANSCRIPTIONS: &str = "/v1/audio/transcriptions";
const DECISIONS: &str = "/v1/decisions";
const SYSTEMONE: &str = "/v1/systemone";

/// The decision-scoring endpoints: the console sends an `input`/`state`
/// plus a `questions` JSON array instead of a transcript, and the reply is
/// scored answers, not generated text.
fn is_scoring(path: &str) -> bool {
    path == DECISIONS || path == SYSTEMONE
}

/// `response_format` values `/v1/audio/transcriptions` accepts. Which ones a
/// given model supports varies (`diarized_json` is only for the diarizing
/// models); that is the upstream's to reject, and seeing it do so is a
/// legitimate test result.
const RESPONSE_FORMATS: &[&str] = &[
    "json",
    "text",
    "srt",
    "verbose_json",
    "vtt",
    "diarized_json",
];

/// Anthropic requires `max_tokens`; the others are happy without one.
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// Console replies are read into memory, so they are capped. Generous
/// enough for an embedding vector, small enough not to be a lever.
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlaygroundRequest {
    /// Backend to pin the request to.
    pub backend: String,
    /// Inbound model name — a canonical name or one of its aliases, the
    /// same string a client would send.
    pub model: String,
    /// The API type: one of `ENDPOINTS`.
    pub path: String,
    #[serde(default)]
    pub system: Option<String>,
    /// The transcript so far, oldest first. Only `user` and `assistant`
    /// turns; the system prompt has its own field because each API spells
    /// it differently. Unused (and may be empty) for a transcription.
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    /// The audio to transcribe; required by, and only used for,
    /// `/v1/audio/transcriptions`.
    #[serde(default)]
    pub audio: Option<Audio>,
    /// ISO-639-1 language of the audio, for transcriptions. Unset lets the
    /// model detect it.
    #[serde(default)]
    pub language: Option<String>,
    /// One of `RESPONSE_FORMATS`, for transcriptions. Unset is the
    /// provider's default (`json`).
    #[serde(default)]
    pub response_format: Option<String>,
    /// Vocabulary-bias keywords for transcriptions. The OpenAI SDK has no
    /// `keywords` parameter; `extra_body={"keywords": [...]}` merges the
    /// array into the multipart body as a `keywords` form field whose
    /// value is the JSON encoding — that is the wire form providers
    /// document, so the console builds the same field.
    #[serde(default)]
    pub keywords: Option<Vec<String>>,
    /// The questions to score, as raw JSON-array text; required by, and
    /// only used for, the scoring endpoints. Kept as text so a malformed
    /// array surfaces as the console's own 400 about the field, not as a
    /// JSON error on the whole console request.
    #[serde(default)]
    pub questions: Option<String>,
    /// The pictures to score, for `/v1/decisions` only: with any, the
    /// `input` becomes a content-block array carrying the turn's text and
    /// these bytes; without any, the string form is sent unchanged. The
    /// systemone composer has no picker and ignores this.
    #[serde(default)]
    pub images: Vec<Image>,
}

/// An uploaded picture, as the browser read it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Image {
    /// Carried because the console sends it and `deny_unknown_fields` must
    /// accept it; the content block it becomes names only the media type and
    /// the bytes, so nothing here reads the name.
    #[expect(
        dead_code,
        reason = "accepted from the console's request shape; the content block it becomes carries only the media type and the bytes"
    )]
    pub filename: String,
    /// The file's MIME type; defaults to `application/octet-stream`, which
    /// the server accepts since it sniffs the bytes anyway.
    #[serde(default)]
    pub content_type: Option<String>,
    /// The file's bytes, standard base64.
    pub data: String,
}

/// An uploaded audio file, as the browser read it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Audio {
    pub filename: String,
    /// The file's MIME type; defaults to `application/octet-stream`, which
    /// providers accept since they sniff the container anyway.
    #[serde(default)]
    pub content_type: Option<String>,
    /// The file's bytes, standard base64.
    pub data: String,
}

#[derive(Deserialize, Serialize, Clone)]
pub(crate) struct Message {
    pub role: String,
    pub content: String,
    /// The reasoning an assistant turn came back with, exactly as the reply
    /// carried it, so the next request can hand it back. Models that think
    /// are trained on transcripts that keep their earlier reasoning, and a
    /// server with a prefix cache can only continue from the end of a turn
    /// it generated if the next prompt contains that turn as generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
}

/// One reply's reasoning, in the dialect it arrived in.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Reasoning {
    /// The endpoint the reply came from. The replay shapes are not
    /// interchangeable, so a transcript continued on another API type sends
    /// its earlier turns without it.
    pub path: String,
    /// The readable trace: shown in the console, and what Chat Completions
    /// sends back as `reasoning_content`.
    #[serde(default)]
    pub text: String,
    /// The reply's own reasoning items (Responses) or thinking blocks
    /// (Messages), replayed verbatim: their ids, signatures and encrypted
    /// payloads are what the upstream checks.
    #[serde(default)]
    pub items: Vec<Value>,
}

#[derive(Serialize)]
pub(crate) struct PlaygroundResponse {
    /// Id of the `request_logs` row for this call. The row is written
    /// asynchronously, so a reader may have to retry briefly.
    pub request_id: String,
    /// Status the router produced — the upstream's, or the router's own
    /// when it rejected the request before dispatch. Never an error of the
    /// console endpoint itself: a failed test is a result, not a fault.
    pub status: u16,
    /// Round trip as the console measured it, including reading the body.
    /// `request_logs` carries the precise upstream ttfb and duration.
    pub duration_ms: i64,
    /// The upstream body, parsed when it is JSON and a raw string when not.
    pub body: Value,
    /// The reply pulled out of `body` per dialect, or the error message
    /// when the call failed. Best-effort: an unrecognized shape leaves this
    /// null and the caller falls back to `body`.
    pub text: Option<String>,
    /// The reply's reasoning, when it carried any. The console keeps it on
    /// the assistant turn and sends it back with the next request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
}

pub(crate) async fn playground(
    State(state): State<AppState>,
    Json(req): Json<PlaygroundRequest>,
) -> Result<Json<PlaygroundResponse>, AppError> {
    let kind = endpoint_kind(&req.path)?;
    validate(&req)?;

    // Resolved up front so a misconfigured pairing is a clear 400 about the
    // form rather than a routing error from inside the proxy.
    let routing = state.routing();
    let route = routing
        .routes
        .get(&req.model)
        .ok_or_else(|| AppError::UnknownModel(req.model.clone()))?;
    if route.targets_on(&req.backend, kind).is_empty() {
        return Err(AppError::BadRequest(format!(
            "model {:?} has no target on backend {:?} serving {}",
            req.model, req.backend, req.path
        )));
    }

    let (body, content_type) = if req.path == TRANSCRIPTIONS {
        build_form(&req)?
    } else {
        let body = build_body(&req)?;
        let body = Bytes::from(serde_json::to_vec(&body).expect("serializing a Value cannot fail"));
        (body, HeaderValue::from_static("application/json"))
    };
    let uri: Uri = req
        .path
        .parse()
        .map_err(|_| AppError::BadRequest(format!("invalid path {:?}", req.path)))?;

    let dispatch = Dispatch {
        pin_backend: req.backend.clone(),
        row_id: uuid::Uuid::new_v4().to_string(),
    };
    let request_id = dispatch.row_id.clone();

    let started = Instant::now();
    let response = match proxy::proxy_request(proxy::ProxyRequestDesc {
        state,
        key: Arc::new(KeyRecord::admin_console()),
        method: Method::POST,
        uri,
        headers: console_headers(content_type),
        body,
        dispatch: Some(dispatch),
    })
    .await
    {
        Ok(resp) => resp,
        // A rejection is the test's answer, so it is rendered like any
        // other upstream reply instead of failing the console call.
        Err(err) => err.into_response(),
    };

    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), MAX_RESPONSE_BYTES)
        .await
        .map_err(|e| AppError::Internal(format!("reading upstream response: {e}")))?;
    let duration_ms = started.elapsed().as_millis() as i64;

    let body: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    let ok = (200..300).contains(&status);
    let text = if ok {
        extract_text(&req.path, &body)
    } else {
        error_message(&body)
    };
    let reasoning = if ok {
        extract_reasoning(&req.path, &body)
    } else {
        None
    };

    Ok(Json(PlaygroundResponse {
        request_id,
        status,
        duration_ms,
        body,
        text,
        reasoning,
    }))
}

fn endpoint_kind(path: &str) -> Result<BackendKind, AppError> {
    ENDPOINTS
        .iter()
        .find(|(p, _)| *p == path)
        .map(|(_, kind)| *kind)
        .ok_or_else(|| {
            let known: Vec<&str> = ENDPOINTS.iter().map(|(p, _)| *p).collect();
            AppError::BadRequest(format!(
                "path must be one of {} (got {path:?})",
                known.join(", ")
            ))
        })
}

fn validate(req: &PlaygroundRequest) -> Result<(), AppError> {
    if is_scoring(&req.path) && req.questions.as_deref().is_none_or(|q| q.trim().is_empty()) {
        return Err(AppError::BadRequest(
            "the scoring endpoints need a questions JSON array".into(),
        ));
    }
    if req.path == TRANSCRIPTIONS {
        if req.audio.is_none() {
            return Err(AppError::BadRequest(
                "attach an audio file to transcribe".into(),
            ));
        }
        if let Some(lang) = req
            .language
            .as_deref()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            && (lang.len() > 16 || !lang.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
        {
            return Err(AppError::BadRequest(format!(
                "invalid language code {lang:?}"
            )));
        }
        if let Some(f) = req.response_format.as_deref().filter(|f| !f.is_empty())
            && !RESPONSE_FORMATS.contains(&f)
        {
            return Err(AppError::BadRequest(format!(
                "response_format must be one of {} (got {f:?})",
                RESPONSE_FORMATS.join(", ")
            )));
        }
    } else if req.messages.is_empty() {
        return Err(AppError::BadRequest("send at least one message".into()));
    }
    for m in &req.messages {
        if !matches!(m.role.as_str(), "user" | "assistant") {
            return Err(AppError::BadRequest(format!(
                "message role must be user or assistant (got {:?})",
                m.role
            )));
        }
    }
    if let Some(t) = req.temperature
        && (!t.is_finite() || t < 0.0)
    {
        return Err(AppError::BadRequest(
            "temperature must be a non-negative number".into(),
        ));
    }
    Ok(())
}

/// Builds the upstream body in the dialect the chosen endpoint speaks. The
/// console keeps one transcript and one system prompt; each API spells them
/// differently, and that difference is exactly what this screen tests.
fn build_body(req: &PlaygroundRequest) -> Result<Value, AppError> {
    let turns: Vec<Value> = req
        .messages
        .iter()
        .flat_map(|m| turn(&req.path, m))
        .collect();
    let system = req
        .system
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let mut body = json!({ "model": req.model });
    let obj = body.as_object_mut().expect("object literal");
    match req.path.as_str() {
        "/v1/chat/completions" => {
            let mut messages = Vec::with_capacity(turns.len() + 1);
            if let Some(s) = system {
                messages.push(json!({"role": "system", "content": s}));
            }
            messages.extend(turns);
            obj.insert("messages".into(), Value::Array(messages));
            if let Some(n) = req.max_tokens {
                obj.insert("max_tokens".into(), json!(n));
            }
        }
        "/v1/responses" => {
            obj.insert("input".into(), Value::Array(turns));
            if let Some(s) = system {
                obj.insert("instructions".into(), json!(s));
            }
            if let Some(n) = req.max_tokens {
                obj.insert("max_output_tokens".into(), json!(n));
            }
        }
        "/v1/messages" => {
            // The only endpoint where max_tokens is mandatory.
            obj.insert(
                "max_tokens".into(),
                json!(req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
            );
            if let Some(s) = system {
                obj.insert("system".into(), json!(s));
            }
            obj.insert("messages".into(), Value::Array(turns));
        }
        "/v1/embeddings" => {
            // Not a conversation: the newest user turn is the input, and
            // the system prompt and token cap do not apply.
            let input = newest_user(req, "embeddings")?;
            obj.insert("input".into(), json!(input.content));
            return Ok(body);
        }
        DECISIONS => {
            // Not a conversation either: the newest user turn is the
            // input; the questions come from the console's own field.
            // Temperature is a decisions knob, so it is forwarded. With
            // picked images the input becomes a content-block array - the
            // turn's text first, then one base64 block per picture, since
            // a string input cannot carry one; without images the string
            // form goes out byte-identical to before the picker existed.
            let input = newest_user(req, "decisions")?;
            let input_value = if req.images.is_empty() {
                json!(input.content)
            } else {
                let mut blocks = vec![json!({"type": "text", "text": input.content})];
                blocks.extend(req.images.iter().map(|image| {
                    json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image
                                .content_type
                                .as_deref()
                                .unwrap_or("application/octet-stream"),
                            "data": image.data,
                        },
                    })
                }));
                Value::Array(blocks)
            };
            obj.insert("input".into(), input_value);
            obj.insert("questions".into(), parse_questions(req)?);
            if let Some(t) = req.temperature {
                obj.insert("temperature".into(), json!(t));
            }
            return Ok(body);
        }
        SYSTEMONE => {
            // The system prompt doubles as the System One `state`; a blank
            // one is legal there, so it is sent as the empty string.
            // Temperature is deliberately NOT forwarded: the systemone wire
            // refuses it with any value, scoring always runs at 1.0.
            obj.insert("state".into(), json!(system.unwrap_or("")));
            obj.insert("questions".into(), parse_questions(req)?);
            return Ok(body);
        }
        other => return Err(AppError::BadRequest(format!("unsupported path {other:?}"))),
    }
    if let Some(t) = req.temperature {
        obj.insert("temperature".into(), json!(t));
    }
    Ok(body)
}

/// The newest user turn — the one field the non-conversational endpoints
/// can read a console transcript into.
fn newest_user<'a>(req: &'a PlaygroundRequest, what: &str) -> Result<&'a Message, AppError> {
    req.messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .ok_or_else(|| AppError::BadRequest(format!("{what} need a user message")))
}

/// The console's questions field as the JSON array the scoring endpoints
/// expect. Refused loudly: a scoring request without questions has nothing
/// to score, and a malformed array is the console's error to name, not the
/// upstream's to guess at.
fn parse_questions(req: &PlaygroundRequest) -> Result<Value, AppError> {
    let raw = req
        .questions
        .as_deref()
        .map(str::trim)
        .filter(|q| !q.is_empty())
        .ok_or_else(|| AppError::BadRequest("send a questions JSON array".into()))?;
    let parsed: Value = serde_json::from_str(raw)
        .map_err(|e| AppError::BadRequest(format!("questions is not valid JSON: {e}")))?;
    if !parsed.is_array() {
        return Err(AppError::BadRequest(
            "questions must be a JSON array".into(),
        ));
    }
    Ok(parsed)
}

/// Builds a transcription request the way the OpenAI SDK does: a multipart
/// form with the file and the options as text fields. Returns the body and
/// its content type, which carries the boundary.
fn build_form(req: &PlaygroundRequest) -> Result<(Bytes, HeaderValue), AppError> {
    let audio = req
        .audio
        .as_ref()
        .ok_or_else(|| AppError::BadRequest("attach an audio file".into()))?;
    let data = base64::engine::general_purpose::STANDARD
        .decode(audio.data.trim())
        .map_err(|e| AppError::BadRequest(format!("audio is not valid base64: {e}")))?;
    if data.is_empty() {
        return Err(AppError::BadRequest("the audio file is empty".into()));
    }
    // A fresh random boundary can't occur in the audio in any way that
    // matters; the quotes and line breaks stripped from the filename are
    // the only characters that could break out of its header.
    let boundary = format!("llm-router-{}", uuid::Uuid::new_v4().simple());
    let filename: String = audio
        .filename
        .chars()
        .filter(|c| !matches!(c, '"' | '\\' | '\r' | '\n'))
        .collect();
    let file_type = audio
        .content_type
        .as_deref()
        .filter(|t| !t.is_empty() && !t.contains(['\r', '\n']))
        .unwrap_or("application/octet-stream");

    let mut body = Vec::with_capacity(data.len() + 1024);
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\
             Content-Type: {file_type}\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(&data);
    let mut field = |name: &str, value: &str| {
        body.extend_from_slice(
            format!(
                "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}"
            )
            .as_bytes(),
        );
    };
    field("model", &req.model);
    let nonblank = |v: &Option<String>| {
        v.as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    };
    if let Some(lang) = nonblank(&req.language) {
        field("language", &lang);
    }
    // Word bias the OpenAI SDK exposes only as `extra_body={"keywords":
    // [...]}`: on a multipart request the SDK merges that into the form,
    // JSON-encoding the array as the field value. Build the same field —
    // JSON escaping is also what keeps the value a single line.
    let keywords: Vec<&str> = req
        .keywords
        .iter()
        .flatten()
        .map(String::as_str)
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .collect();
    if !keywords.is_empty() {
        field(
            "keywords",
            &serde_json::to_string(&keywords).expect("a string array serializes"),
        );
    }
    // The console's one free-text option doubles as the transcription
    // prompt: vocabulary and spelling hints for the model.
    if let Some(prompt) = nonblank(&req.system) {
        field("prompt", &prompt);
    }
    if let Some(format) = nonblank(&req.response_format) {
        // Diarization needs a chunking strategy for anything over 30 s;
        // `auto` is what the SDK examples use.
        if format == "diarized_json" {
            field("chunking_strategy", "auto");
        }
        field("response_format", &format);
    }
    if let Some(t) = req.temperature {
        field("temperature", &t.to_string());
    }
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let content_type = HeaderValue::try_from(format!("multipart/form-data; boundary={boundary}"))
        .expect("boundary is ascii");
    Ok((Bytes::from(body), content_type))
}

/// One transcript turn as the endpoint's items: a plain turn, or an assistant
/// turn with its reasoning handed back the way that API carries it.
fn turn(path: &str, m: &Message) -> Vec<Value> {
    let plain = json!({"role": m.role, "content": m.content});
    let Some(r) = m
        .reasoning
        .as_ref()
        .filter(|r| m.role == "assistant" && r.path == path)
    else {
        return vec![plain];
    };
    match path {
        "/v1/chat/completions" if !r.text.is_empty() => vec![json!({
            "role": "assistant",
            "content": m.content,
            "reasoning_content": r.text,
        })],
        // Output items go back as input items, reasoning first, which is
        // the order the reply produced them in.
        "/v1/responses" if !r.items.is_empty() => {
            let mut out = r.items.clone();
            out.push(plain);
            out
        }
        "/v1/messages" if !r.items.is_empty() => {
            let mut blocks = r.items.clone();
            if !m.content.is_empty() {
                blocks.push(json!({"type": "text", "text": m.content}));
            }
            vec![json!({"role": "assistant", "content": blocks})]
        }
        _ => vec![plain],
    }
}

/// Pulls a successful reply's reasoning out per dialect, or `None` when it
/// carried none.
fn extract_reasoning(path: &str, body: &Value) -> Option<Reasoning> {
    let texts = |parts: &Value| -> String {
        parts
            .as_array()
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default()
    };
    let (text, items) = match path {
        // `reasoning_content` is the common spelling; some servers use
        // `reasoning`.
        "/v1/chat/completions" => {
            let msg = &body["choices"][0]["message"];
            let text = msg["reasoning_content"]
                .as_str()
                .or_else(|| msg["reasoning"].as_str())
                .unwrap_or_default()
                .to_string();
            (text, Vec::new())
        }
        "/v1/responses" => {
            let items: Vec<Value> = body["output"]
                .as_array()?
                .iter()
                .filter(|item| item["type"] == "reasoning")
                .cloned()
                .collect();
            // The summary is what most servers fill; raw reasoning text,
            // when a server sends it, sits in `content`.
            let text = items
                .iter()
                .map(|item| {
                    let summary = texts(&item["summary"]);
                    if summary.is_empty() {
                        texts(&item["content"])
                    } else {
                        summary
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            (text, items)
        }
        "/v1/messages" => {
            let items: Vec<Value> = body["content"]
                .as_array()?
                .iter()
                .filter(|b| b["type"] == "thinking" || b["type"] == "redacted_thinking")
                .cloned()
                .collect();
            let text = items
                .iter()
                .filter_map(|b| b["thinking"].as_str())
                .collect::<Vec<_>>()
                .join("\n");
            (text, items)
        }
        _ => return None,
    };
    (!text.is_empty() || !items.is_empty()).then(|| Reasoning {
        path: path.to_string(),
        text,
        items,
    })
}

/// Pulls the reply out of a successful response per dialect. Best-effort by
/// design — the raw body always rides along, so an unfamiliar shape degrades
/// to "no text extracted" instead of an error.
fn extract_text(path: &str, body: &Value) -> Option<String> {
    let joined = |parts: &Vec<Value>| {
        let s: String = parts
            .iter()
            .filter_map(|c| c.get("text").and_then(Value::as_str))
            .collect();
        (!s.is_empty()).then_some(s)
    };
    match path {
        "/v1/chat/completions" => body["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_string),
        // Responses returns a list of output items; the message item's
        // content holds the output_text parts. `output_text` itself is an
        // SDK convenience field and is not always on the wire.
        "/v1/responses" => {
            let s: String = body["output"]
                .as_array()?
                .iter()
                .filter_map(|item| item["content"].as_array())
                .filter_map(joined)
                .collect();
            (!s.is_empty()).then_some(s)
        }
        "/v1/messages" => joined(body["content"].as_array()?),
        // `text`/`srt`/`vtt` come back as plain text, which the caller
        // already turned into a JSON string. Diarized output is shown one
        // speaker turn per line, since the joined `text` loses who spoke.
        TRANSCRIPTIONS => {
            if let Some(s) = body.as_str() {
                return Some(s.to_string());
            }
            let segments = body["segments"]
                .as_array()
                .filter(|s| s.iter().any(|s| s["speaker"].is_string()));
            if let Some(segments) = segments {
                let lines: Vec<String> = segments
                    .iter()
                    .map(|s| {
                        let at = s["start"]
                            .as_f64()
                            .map(|t| format!("[{t:.1}s] "))
                            .unwrap_or_default();
                        let who = s["speaker"].as_str().unwrap_or("?");
                        format!("{at}{who}: {}", s["text"].as_str().unwrap_or("").trim())
                    })
                    .collect();
                return Some(lines.join("\n"));
            }
            body["text"].as_str().map(str::to_string)
        }
        "/v1/embeddings" => {
            let dims = body["data"][0]["embedding"].as_array()?.len();
            let count = body["data"].as_array().map_or(1, Vec::len);
            Some(format!("{count} embedding vector(s), {dims} dimensions"))
        }
        // One line per answer, in request order (the answers object is
        // insertion-ordered): the chosen label, the yes/no word, or the
        // expected score — whichever the answer carries. A System One
        // noul answer's number is shown as-is.
        DECISIONS | SYSTEMONE => {
            let answers = body["answers"].as_object()?;
            let lines: Vec<String> = answers
                .iter()
                .map(|(id, ans)| {
                    let shown = match ans.get("choice").or_else(|| ans.get("answer")) {
                        Some(Value::String(s)) => s.clone(),
                        Some(v @ Value::Number(_)) => v.to_string(),
                        _ => ans["score"]
                            .as_f64()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "?".to_string()),
                    };
                    format!("{id}: {shown}")
                })
                .collect();
            (!lines.is_empty()).then(|| lines.join("\n"))
        }
        _ => None,
    }
}

/// OpenAI and Anthropic both nest the message under `error`; the router's
/// own errors use the same shape.
fn error_message(body: &Value) -> Option<String> {
    body["error"]["message"]
        .as_str()
        .map(str::to_string)
        .or_else(|| body.as_str().map(str::to_string))
}

/// The console's own headers. `upstream_headers` builds the outbound set
/// from these, so the User-Agent identifies console traffic upstream.
fn console_headers(content_type: HeaderValue) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, content_type);
    headers.insert(
        header::USER_AGENT,
        HeaderValue::from_static(concat!(
            "llm-router-admin-console/",
            env!("CARGO_PKG_VERSION")
        )),
    );
    headers
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(path: &str) -> PlaygroundRequest {
        PlaygroundRequest {
            backend: "b1".into(),
            model: "m".into(),
            path: path.into(),
            system: Some("be brief".into()),
            messages: vec![
                Message {
                    role: "user".into(),
                    content: "hi".into(),
                    reasoning: None,
                },
                Message {
                    role: "assistant".into(),
                    content: "hello".into(),
                    reasoning: None,
                },
                Message {
                    role: "user".into(),
                    content: "again".into(),
                    reasoning: None,
                },
            ],
            max_tokens: Some(64),
            temperature: None,
            audio: None,
            language: None,
            response_format: None,
            keywords: None,
            questions: None,
            images: Vec::new(),
        }
    }

    fn transcription(format: Option<&str>) -> PlaygroundRequest {
        let mut r = req(TRANSCRIPTIONS);
        r.messages.clear();
        r.system = Some("Otsimo".into());
        r.language = Some("tr".into());
        r.response_format = format.map(str::to_string);
        r.keywords = Some(vec!["Otsimo".into(), "  ".into(), " Grok 3 ".into()]);
        r.audio = Some(Audio {
            filename: "a\"b.mp3".into(),
            content_type: Some("audio/mpeg".into()),
            data: base64::engine::general_purpose::STANDARD.encode(b"\x00\xffaudio"),
        });
        r
    }

    #[test]
    fn transcription_form_is_what_the_router_probes() {
        let r = transcription(Some("diarized_json"));
        validate(&r).unwrap();
        let (body, ct) = build_form(&r).unwrap();
        let boundary = crate::multipart::boundary(ct.to_str().unwrap()).unwrap();
        let probe = crate::multipart::probe(&body, boundary).expect("a well-formed form");
        assert_eq!(probe.model.as_deref(), Some("m"));

        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("filename=\"ab.mp3\"\r\nContent-Type: audio/mpeg\r\n\r\n"),
            "{text}"
        );
        assert!(
            body.windows(7).any(|w| w == b"\x00\xffaudio"),
            "audio bytes are raw, not base64"
        );
        for (name, value) in [
            ("language", "tr"),
            ("prompt", "Otsimo"),
            ("response_format", "diarized_json"),
            ("chunking_strategy", "auto"),
            ("keywords", r#"["Otsimo","Grok 3"]"#),
        ] {
            assert!(
                text.contains(&format!("name=\"{name}\"\r\n\r\n{value}\r\n")),
                "{name}: {text}"
            );
        }
    }

    #[test]
    fn transcription_form_omits_unset_options() {
        let mut r = transcription(None);
        r.language = Some(" ".into());
        r.system = None;
        r.keywords = Some(vec![" ".into()]);
        let (body, _) = build_form(&r).unwrap();
        let text = String::from_utf8_lossy(&body);
        for name in [
            "language",
            "prompt",
            "response_format",
            "chunking_strategy",
            "temperature",
            "keywords",
        ] {
            assert!(
                !text.contains(&format!("name=\"{name}\"")),
                "{name}: {text}"
            );
        }
    }

    #[test]
    fn transcription_validation() {
        let mut r = transcription(Some("json"));
        validate(&r).unwrap();
        r.response_format = Some("mp3".into());
        assert!(validate(&r).is_err());
        let mut r = transcription(None);
        r.language = Some("en\r\nX".into());
        assert!(validate(&r).is_err(), "the language becomes a form value");
        let mut r = transcription(None);
        r.audio = None;
        assert!(validate(&r).is_err());
        let mut r = transcription(None);
        r.audio.as_mut().unwrap().data = "not base64!".into();
        assert!(build_form(&r).is_err());
    }

    #[test]
    fn extracts_transcription_text_per_format() {
        assert_eq!(
            extract_text(TRANSCRIPTIONS, &json!({"text": "hello"})).as_deref(),
            Some("hello")
        );
        assert_eq!(
            extract_text(
                TRANSCRIPTIONS,
                &json!("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
            )
            .as_deref(),
            Some("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        );
        let diarized = json!({"text": "hi there", "segments": [
            {"speaker": "A", "start": 0.0, "end": 1.2, "text": " hi"},
            {"speaker": "B", "start": 1.25, "end": 2.0, "text": "there"}
        ]});
        assert_eq!(
            extract_text(TRANSCRIPTIONS, &diarized).as_deref(),
            Some("[0.0s] A: hi\n[1.2s] B: there")
        );
        // verbose_json segments carry no speaker: the joined text is shown.
        let verbose = json!({"text": "hi", "segments": [{"start": 0.0, "text": "hi"}]});
        assert_eq!(
            extract_text(TRANSCRIPTIONS, &verbose).as_deref(),
            Some("hi")
        );
    }

    #[test]
    fn chat_completions_body_puts_system_first() {
        let b = build_body(&req("/v1/chat/completions")).unwrap();
        assert_eq!(b["model"], "m");
        assert_eq!(b["messages"][0]["role"], "system");
        assert_eq!(b["messages"][0]["content"], "be brief");
        assert_eq!(b["messages"][1]["content"], "hi");
        assert_eq!(b["messages"][3]["content"], "again");
        assert_eq!(b["max_tokens"], 64);
        assert!(b.get("temperature").is_none(), "unset options are omitted");
    }

    #[test]
    fn responses_body_uses_input_and_instructions() {
        let b = build_body(&req("/v1/responses")).unwrap();
        assert_eq!(b["instructions"], "be brief");
        assert_eq!(b["input"][0]["role"], "user");
        assert_eq!(b["input"].as_array().unwrap().len(), 3);
        assert_eq!(b["max_output_tokens"], 64);
        assert!(b.get("messages").is_none());
    }

    #[test]
    fn anthropic_body_hoists_system_and_always_sets_max_tokens() {
        let b = build_body(&req("/v1/messages")).unwrap();
        assert_eq!(b["system"], "be brief");
        assert_eq!(b["messages"][0]["content"], "hi");
        assert_eq!(b["max_tokens"], 64);

        let mut r = req("/v1/messages");
        r.max_tokens = None;
        r.system = Some("   ".into());
        let b = build_body(&r).unwrap();
        assert_eq!(b["max_tokens"], DEFAULT_MAX_TOKENS);
        assert!(
            b.get("system").is_none(),
            "a blank system prompt is dropped"
        );
    }

    #[test]
    fn embeddings_body_takes_the_newest_user_turn() {
        let b = build_body(&req("/v1/embeddings")).unwrap();
        assert_eq!(b["input"], "again");
        assert!(b.get("messages").is_none());
        assert!(b.get("max_tokens").is_none());
        assert!(b.get("system").is_none());
    }

    /// `req(path)` with a two-question scoring array in its questions field.
    fn with_questions(path: &str) -> PlaygroundRequest {
        let mut r = req(path);
        r.questions = Some(
            r#"[{"id":"q1","type":"yes_no","question":"Is this about billing?"},
                {"id":"q2","type":"choice","question":"What is it?","options":["a bug","a feature"]}]"#
                .into(),
        );
        r
    }

    #[test]
    fn decisions_body_sends_input_and_questions_and_forwards_temperature() {
        let mut r = with_questions(DECISIONS);
        r.temperature = Some(0.5);
        validate(&r).unwrap();
        let b = build_body(&r).unwrap();
        assert_eq!(b["model"], "m");
        assert_eq!(b["input"], "again", "the newest user turn is the input");
        assert_eq!(b["questions"].as_array().unwrap().len(), 2);
        assert_eq!(b["questions"][0]["id"], "q1");
        assert_eq!(b["temperature"], 0.5);
        assert!(b.get("messages").is_none());
        assert!(b.get("max_tokens").is_none(), "scoring generates nothing");
    }

    #[test]
    fn systemone_body_uses_state_and_never_forwards_temperature() {
        let mut r = with_questions(SYSTEMONE);
        r.temperature = Some(0.5);
        validate(&r).unwrap();
        let b = build_body(&r).unwrap();
        assert_eq!(b["state"], "be brief", "the system prompt is the state");
        assert_eq!(b["questions"].as_array().unwrap().len(), 2);
        assert!(
            b.get("temperature").is_none(),
            "the systemone wire refuses temperature with any value"
        );

        let mut r = with_questions(SYSTEMONE);
        r.system = Some("   ".into());
        let b = build_body(&r).unwrap();
        assert_eq!(b["state"], "", "a blank state is legal on this wire");
    }

    /// Two images as the browser read them: the same base64 shape the audio
    /// picker's reader produces.
    fn two_images() -> Vec<Image> {
        [("a.png", Some("image/png")), ("b.jpg", Some("image/jpeg"))]
            .iter()
            .enumerate()
            .map(|(index, (name, mime))| Image {
                filename: (*name).to_string(),
                content_type: mime.map(str::to_string),
                data: base64::engine::general_purpose::STANDARD.encode(format!("bytes-{index}")),
            })
            .collect()
    }

    #[test]
    fn build_body_composes_text_and_image_blocks_for_decisions() {
        let mut r = with_questions(DECISIONS);
        r.images = two_images();
        validate(&r).unwrap();
        let b = build_body(&r).unwrap();
        let input = b["input"]
            .as_array()
            .expect("with images the input is blocks");
        assert_eq!(
            input.len(),
            3,
            "the turn's text leads, then one block per picture"
        );
        assert_eq!(input[0], json!({"type": "text", "text": "again"}));
        for (index, name) in (1..=2).zip(["a.png", "b.jpg"]) {
            let want_data =
                base64::engine::general_purpose::STANDARD.encode(format!("bytes-{}", index - 1));
            assert_eq!(
                input[index],
                json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": if name == "a.png" { "image/png" } else { "image/jpeg" },
                        "data": want_data,
                    },
                }),
                "block {index} is {name}"
            );
        }
        // A file the browser named no MIME type for still spells a
        // media_type; the server sniffs the bytes anyway.
        let mut r = with_questions(DECISIONS);
        r.images = vec![Image {
            filename: "x".into(),
            content_type: None,
            data: "aGk=".into(),
        }];
        let b = build_body(&r).unwrap();
        assert_eq!(
            b["input"][1]["source"]["media_type"],
            "application/octet-stream"
        );
    }

    #[test]
    fn build_body_without_images_sends_the_string_form() {
        // The golden the server's own container tests lean on: no images,
        // no change - `input` is the newest user turn's string, byte for
        // byte as before the picker existed.
        let r = with_questions(DECISIONS);
        let b = build_body(&r).unwrap();
        assert_eq!(b["input"], json!("again"));
    }

    #[test]
    fn build_body_systemone_is_unchanged() {
        // The systemone wire refuses block arrays outright; the console
        // gives its composer no picker and sends its state as the string it
        // always was, whatever `images` a stale request might carry.
        let mut r = with_questions(SYSTEMONE);
        r.images = two_images();
        let b = build_body(&r).unwrap();
        assert_eq!(b["state"], "be brief");
        assert!(
            b.get("images").is_none(),
            "images never ride the systemone body"
        );
    }

    #[test]
    fn questions_must_be_a_json_array() {
        // Absent or blank fails validation before any dispatch.
        let mut r = req(DECISIONS);
        assert!(validate(&r).is_err());
        r.questions = Some("  ".into());
        assert!(validate(&r).is_err());

        // Present-but-malformed fails at build with the field named.
        let mut r = with_questions(DECISIONS);
        r.questions = Some("[oops".into());
        let err = build_body(&r).unwrap_err().to_string();
        assert!(err.contains("questions"), "{err}");
        let mut r = with_questions(SYSTEMONE);
        r.questions = Some(r#"{"id":"q1"}"#.into());
        let err = build_body(&r).unwrap_err().to_string();
        assert!(err.contains("array"), "{err}");
    }

    #[test]
    fn extracts_scoring_answers_one_line_each_in_order() {
        let decisions = json!({"answers": {
            "q1": {"type": "yes_no", "answer": "no", "probabilities": [0.02, 0.98]},
            "q2": {"type": "choice", "choice": "a bug", "probabilities": [0.9, 0.1]},
            "q3": {"type": "score", "score": 1.75, "legend": ["0", "1", "2"]}
        }});
        assert_eq!(
            extract_text(DECISIONS, &decisions).as_deref(),
            Some("q1: no\nq2: a bug\nq3: 1.75")
        );

        // A System One noul answer carries the yes probability as a number.
        let systemone = json!({"answers": {
            "": {"type": "yes_no", "answer": "yes", "confidence": 0.8},
            "n1": {"type": "yes_no", "answer": 0.03, "confidence": 0.94}
        }});
        assert_eq!(
            extract_text(SYSTEMONE, &systemone).as_deref(),
            Some(": yes\nn1: 0.03")
        );

        // An error body or an answers-less reply degrades to no text.
        assert_eq!(extract_text(DECISIONS, &json!({"error": {}})), None);
        assert_eq!(extract_text(DECISIONS, &json!({"answers": {}})), None);
    }

    #[test]
    fn extracts_text_per_dialect() {
        let openai = json!({"choices": [{"message": {"content": "hi there"}}]});
        assert_eq!(
            extract_text("/v1/chat/completions", &openai).as_deref(),
            Some("hi there")
        );

        let responses = json!({"output": [
            {"type": "reasoning"},
            {"type": "message", "content": [{"type": "output_text", "text": "a"}, {"type": "output_text", "text": "b"}]}
        ]});
        assert_eq!(
            extract_text("/v1/responses", &responses).as_deref(),
            Some("ab")
        );

        let anthropic = json!({"content": [{"type": "text", "text": "yes"}]});
        assert_eq!(
            extract_text("/v1/messages", &anthropic).as_deref(),
            Some("yes")
        );

        let embeddings = json!({"data": [{"embedding": [0.1, 0.2, 0.3]}]});
        assert_eq!(
            extract_text("/v1/embeddings", &embeddings).as_deref(),
            Some("1 embedding vector(s), 3 dimensions")
        );

        // Unfamiliar shapes are not an error.
        assert_eq!(
            extract_text("/v1/chat/completions", &json!({"choices": []})),
            None
        );
        assert_eq!(
            extract_text("/v1/messages", &json!({"content": "plain"})),
            None
        );
    }

    #[test]
    fn error_messages_are_unwrapped() {
        let body = json!({"error": {"message": "bad key", "type": "authentication_error"}});
        assert_eq!(error_message(&body).as_deref(), Some("bad key"));
        assert_eq!(error_message(&json!({"nope": 1})), None);
    }

    #[test]
    fn endpoint_kinds_and_validation() {
        assert_eq!(
            endpoint_kind("/v1/messages").unwrap(),
            BackendKind::Anthropic
        );
        assert_eq!(endpoint_kind("/v1/responses").unwrap(), BackendKind::Openai);
        assert_eq!(endpoint_kind(DECISIONS).unwrap(), BackendKind::Openai);
        assert_eq!(endpoint_kind(SYSTEMONE).unwrap(), BackendKind::Openai);
        assert!(endpoint_kind("/v1/completions").is_err());

        let mut r = req("/v1/chat/completions");
        r.messages.clear();
        assert!(validate(&r).is_err());
        let mut r = req("/v1/chat/completions");
        r.messages[0].role = "system".into();
        assert!(validate(&r).is_err(), "system belongs in its own field");
    }

    /// `req(path)` with its assistant turn carrying the reasoning `reply`
    /// came back with.
    fn with_reasoning(path: &str, reply: &Value) -> PlaygroundRequest {
        let mut r = req(path);
        r.messages[1].reasoning = extract_reasoning(path, reply);
        r
    }

    #[test]
    fn chat_reasoning_is_handed_back_as_reasoning_content() {
        let reply =
            json!({"choices": [{"message": {"content": "hello", "reasoning_content": "think"}}]});
        let r = extract_reasoning("/v1/chat/completions", &reply).expect("a trace");
        assert_eq!(r.text, "think");
        let b = build_body(&with_reasoning("/v1/chat/completions", &reply)).unwrap();
        assert_eq!(b["messages"][2]["role"], "assistant");
        assert_eq!(b["messages"][2]["reasoning_content"], "think");
        assert_eq!(b["messages"][2]["content"], "hello");
    }

    #[test]
    fn responses_reasoning_items_are_replayed_verbatim_before_the_message() {
        let item = json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think"}], "content": []});
        let reply = json!({"output": [item, {"type": "message", "content": [{"type": "output_text", "text": "hello"}]}]});
        let r = extract_reasoning("/v1/responses", &reply).expect("a trace");
        assert_eq!(r.text, "think");
        let b = build_body(&with_reasoning("/v1/responses", &reply)).unwrap();
        let input = b["input"].as_array().unwrap();
        assert_eq!(input.len(), 4, "the reasoning item joins the three turns");
        assert_eq!(input[1], item, "ids and payloads go back untouched");
        assert_eq!(input[2]["role"], "assistant");
    }

    #[test]
    fn messages_thinking_blocks_lead_the_assistant_content() {
        let block = json!({"type": "thinking", "thinking": "think", "signature": "sig"});
        let reply = json!({"content": [block, {"type": "text", "text": "hello"}]});
        let b = build_body(&with_reasoning("/v1/messages", &reply)).unwrap();
        let content = b["messages"][1]["content"].as_array().unwrap();
        assert_eq!(content[0], block);
        assert_eq!(content[1], json!({"type": "text", "text": "hello"}));
    }

    #[test]
    fn reasoning_from_another_api_type_is_not_replayed() {
        let reply =
            json!({"choices": [{"message": {"content": "hello", "reasoning_content": "think"}}]});
        let mut r = with_reasoning("/v1/chat/completions", &reply);
        r.path = "/v1/responses".into();
        let b = build_body(&r).unwrap();
        assert_eq!(b["input"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn a_reply_without_reasoning_carries_none() {
        let reply = json!({"choices": [{"message": {"content": "hello"}}]});
        assert!(extract_reasoning("/v1/chat/completions", &reply).is_none());
        let reply = json!({"output": [{"type": "message", "content": []}]});
        assert!(extract_reasoning("/v1/responses", &reply).is_none());
    }
}
