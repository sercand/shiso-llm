//! `POST /v1/completions`: the text-completion surface, which has no
//! conversation in it.
//!
//! # Why this prompt does not go through the chat template
//!
//! A completion is raw continuation: the caller sends text and the model carries
//! on from it. Wrapping that in turn markers would put words in the prompt the
//! caller never wrote, and the answer would be a reply to a conversation rather
//! than a continuation of a document. So this surface tokenizes its prompt
//! directly and reaches the engine through the shared path that starts after
//! rendering.
//!
//! # What this surface does not carry
//!
//! No tools, no thinking, no media, no list of candidate answers. Those are all
//! things the format has no field for, and inventing one - a thinking trace
//! prefixed into the text, say - would corrupt the continuation it exists to
//! produce. A trace the model emits anyway is dropped here rather than mixed in.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse as _, Response};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use shiso_core::TokenId;
use shiso_ir::{
    BlockDelta, Request, Sampling, ServiceClass, StopReason, StreamEvent, ThinkingDirective, Usage,
};

use crate::api::pipeline::{self, PromptField};
use crate::api::service_tier::{self, ServiceTier};
use crate::api::sse::{Frame, FrameEncoder};
use crate::api::{blocking, sse};
use crate::error::AppError;
use crate::openai::chat_encode::{encode_usage, finish_reason};
use crate::openai::chat_wire::{StopField, StreamOptions};
use crate::state::AppState;

/// The prompt: text, ids, or a list of either.
///
/// Four variants because the field genuinely means four things on the wire. The
/// order matters: a list of numbers must be read as ids before it is read as
/// anything else, and a bare string before a list of them.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    /// One document.
    Text(String),
    /// One document, already tokenized.
    Ids(Vec<u32>),
    /// Several documents.
    Texts(Vec<String>),
    /// Several documents, already tokenized.
    IdLists(Vec<Vec<u32>>),
}

/// The completions request body.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CompletionBody {
    /// The model the caller named, echoed rather than honoured.
    #[serde(default)]
    pub model: Option<String>,
    /// What to continue.
    #[serde(default)]
    pub prompt: Option<PromptInput>,
    /// Softmax temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Cumulative-probability cutoff.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Candidates kept by the top-k stage.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Floor on a token's probability.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// How many answers the caller wants.
    #[serde(default)]
    pub n: Option<u32>,
    /// The token cap.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Sequences that end generation.
    #[serde(default)]
    pub stop: Option<StopField>,
    /// Stream the answer.
    #[serde(default)]
    pub stream: Option<bool>,
    /// What a streaming caller wants beyond the content.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Whether the prompt is repeated back before the continuation.
    #[serde(default)]
    pub echo: Option<bool>,
    /// Seed for reproducible sampling.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Report the sampled ids beside the text.
    #[serde(default)]
    pub return_token_ids: Option<bool>,
    /// The service class asked for. Read and never echoed: the legacy
    /// completion object has no field to echo it in.
    #[serde(default)]
    pub service_tier: Option<ServiceTier>,
}

impl CompletionBody {
    /// Whether the caller asked for the accounting on the stream.
    #[must_use]
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .and_then(|options| options.include_usage)
            .unwrap_or(false)
    }

    /// Checks what can be checked without a vocabulary.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming the field at fault.
    pub fn validate(&self) -> Result<(), AppError> {
        if self.prompt.is_none() {
            return Err(AppError::bad_request(
                "prompt must be a string, a list of token ids, or a list of either",
                Some("prompt"),
                "invalid_request",
            ));
        }
        if self.max_tokens == Some(0) {
            return Err(AppError::bad_request(
                "max_tokens must be at least 1",
                Some("max_tokens"),
                "invalid_request",
            ));
        }
        if let Some(n) = self.n
            && n != 1
        {
            return Err(AppError::bad_request(
                "this server answers one request with one continuation; send \
                 concurrent requests instead of n > 1",
                Some("n"),
                if n < 1 {
                    "invalid_request"
                } else {
                    "unsupported_value"
                },
            ));
        }
        Ok(())
    }
}

/// A prompt resolved against a vocabulary: the ids to run, and the text to echo.
#[derive(Debug, Clone)]
pub struct ResolvedPrompt {
    /// What the engine runs.
    pub tokens: Vec<TokenId>,
    /// What an echoing caller is shown, which is the caller's own text when they
    /// sent text and the decoded run when they sent ids.
    pub text: String,
}

/// Turns the prompt field into ids.
///
/// # Errors
///
/// [`AppError::BadRequest`] when the caller sent more than one document - this
/// server answers one request with one continuation - or an id the vocabulary
/// has no row for.
pub fn resolve_prompt(
    prompt: &PromptInput,
    vocabulary: &crate::tokenizer::Vocabulary,
) -> Result<ResolvedPrompt, AppError> {
    let too_many = |count: usize| {
        AppError::bad_request(
            format!(
                "prompt holds {count} documents, and this server answers one \
                 request with one continuation"
            ),
            Some("prompt"),
            "unsupported_value",
        )
    };
    match prompt {
        PromptInput::Text(text) => Ok(ResolvedPrompt {
            tokens: encode(vocabulary, text)?,
            text: text.clone(),
        }),
        PromptInput::Texts(texts) => match texts.as_slice() {
            [one] => Ok(ResolvedPrompt {
                tokens: encode(vocabulary, one)?,
                text: one.clone(),
            }),
            other => Err(too_many(other.len())),
        },
        PromptInput::Ids(ids) => from_ids(ids, vocabulary),
        PromptInput::IdLists(lists) => match lists.as_slice() {
            [one] => from_ids(one, vocabulary),
            other => Err(too_many(other.len())),
        },
    }
}

/// Encodes prompt text, with no template markers around it but with the
/// checkpoint's opening token in front of it.
///
/// No template markers is the point of this route: the caller's text is the
/// prompt. The opening token is not a marker in that sense - it is the token the
/// model was trained to see first, upstream llama.cpp puts it in front of a
/// `/v1/completions` prompt too (`handle_completions_impl` tokenizes with
/// `add_special = true`), and without it Gemma 4 answers `"The capital of
/// France is"` with `" France is France is France is"`.
fn encode(vocabulary: &crate::tokenizer::Vocabulary, text: &str) -> Result<Vec<TokenId>, AppError> {
    vocabulary
        .encode_prompt(text)
        .map_err(|error| AppError::internal("encoding a completion prompt", error))
}

/// Checks caller-supplied ids against the vocabulary and decodes them for echo.
///
/// The bound check is not ceremony: an id past the end of the embedding table is
/// a device-side out-of-bounds read, which reaches the caller as a backend
/// failure that says nothing about the id they sent.
fn from_ids(
    ids: &[u32],
    vocabulary: &crate::tokenizer::Vocabulary,
) -> Result<ResolvedPrompt, AppError> {
    let size = vocabulary.size();
    if let Some(bad) = ids.iter().find(|id| **id >= size) {
        return Err(AppError::bad_request(
            format!("the token id {bad} is outside this model's vocabulary of {size}"),
            Some("prompt"),
            "invalid_request",
        ));
    }
    let tokens: Vec<TokenId> = ids.iter().copied().map(TokenId::new).collect();
    let text = vocabulary
        .decode(&tokens)
        .map_err(|error| AppError::internal("decoding a completion prompt", error))?;
    Ok(ResolvedPrompt { tokens, text })
}

/// `POST /v1/completions`.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match answer(&state, &headers, &body).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

/// The handler, with its failures still typed.
async fn answer(state: &AppState, headers: &HeaderMap, body: &Bytes) -> Result<Response, AppError> {
    let mut parsed: CompletionBody = serde_json::from_slice(body).map_err(|error| {
        AppError::bad_request(
            format!("the request body is not a completion: {error}"),
            None,
            "invalid_request",
        )
    })?;
    parsed.service_tier = service_tier::resolve(parsed.service_tier, headers)?;
    parsed.validate()?;

    let renderer = state.renderer.as_ref().ok_or(AppError::NotReady {
        what: "a prompt renderer",
    })?;
    let prompt = resolve_prompt(
        parsed.prompt.as_ref().ok_or_else(|| {
            AppError::bad_request("prompt is required", Some("prompt"), "invalid_request")
        })?,
        renderer.vocabulary(),
    )?;

    let request = to_ir(&parsed, &state.model_id);
    let options = Options {
        id: crate::openai::new_completion_id(),
        model: state.model_id.clone(),
        echo: parsed.echo.unwrap_or(false).then(|| prompt.text.clone()),
        return_token_ids: parsed.return_token_ids.unwrap_or(false),
    };

    let started = pipeline::start_with_prompt(
        state,
        &request,
        prompt.tokens,
        Vec::new(),
        PromptField::Prompt,
    )
    .await?;
    let events = started.events;

    if parsed.stream.unwrap_or(false) {
        return Ok(sse::encode(
            events,
            CompletionChunkEncoder::new(options, parsed.include_usage()),
        ));
    }
    let response = blocking::collect_response(events).await?;
    Ok(axum::Json(encode_completion(&response, &options)).into_response())
}

/// The neutral request behind one completion.
///
/// Infallible, because this format carries nothing that can be spelled wrongly:
/// every field it has is either a number in a range the validator already
/// checked or a string with no vocabulary.
fn to_ir(body: &CompletionBody, model_id: &str) -> Request {
    let mut request = Request::new(model_id, Vec::new());
    request.sampling = Sampling {
        temperature: body.temperature,
        top_k: body.top_k,
        top_p: body.top_p,
        min_p: body.min_p,
        ..Sampling::default()
    };
    request.max_tokens = body.max_tokens.unwrap_or(0);
    request.stop = body
        .stop
        .clone()
        .map(StopField::into_vec)
        .unwrap_or_default();
    request.stream = body.stream.unwrap_or(false);
    // The format predates thinking and has no place to put a trace, so the
    // directive is explicit rather than left unspecified: a server default that
    // turned thinking on here would put reasoning into a document continuation.
    request.thinking = ThinkingDirective::Off;
    request.seed = body.seed;
    request.return_token_ids = body.return_token_ids.unwrap_or(false);
    request.service_class = body
        .service_tier
        .map_or(ServiceClass::Standard, ServiceTier::class);
    request
}

/// What this surface needs beyond the turn.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// The identifier this completion is known by.
    pub id: String,
    /// The model name to echo.
    pub model: String,
    /// The prompt to repeat back first, when the caller asked for it.
    pub echo: Option<String>,
    /// Whether the sampled ids ride along.
    pub return_token_ids: bool,
}

/// The blocking answer.
#[must_use]
pub fn encode_completion(response: &shiso_ir::Response, options: &Options) -> Value {
    let mut text = options.echo.clone().unwrap_or_default();
    text.push_str(&response.message.text());

    let mut choice = Map::new();
    choice.insert("text".to_owned(), json!(text));
    choice.insert("index".to_owned(), json!(0));
    choice.insert("logprobs".to_owned(), Value::Null);
    choice.insert(
        "finish_reason".to_owned(),
        json!(finish_reason(&response.stop)),
    );

    let mut out = Map::new();
    out.insert("id".to_owned(), json!(options.id));
    out.insert("object".to_owned(), json!("text_completion"));
    out.insert("created".to_owned(), json!(response.created));
    out.insert("model".to_owned(), json!(options.model));
    out.insert(
        "choices".to_owned(),
        Value::Array(vec![Value::Object(choice)]),
    );
    out.insert("usage".to_owned(), encode_usage(&response.usage));
    if options.return_token_ids {
        out.insert("token_ids".to_owned(), json!(response.token_ids));
    }
    Value::Object(out)
}

/// Turns the neutral stream into text-completion chunks.
#[derive(Debug)]
pub struct CompletionChunkEncoder {
    options: Options,
    include_usage: bool,
    created: u64,
    closed: bool,
}

impl CompletionChunkEncoder {
    /// An encoder for one request.
    #[must_use]
    pub fn new(options: Options, include_usage: bool) -> Self {
        Self {
            options,
            include_usage,
            created: 0,
            closed: false,
        }
    }

    /// One chunk carrying `text`.
    fn chunk(&self, text: &str, finish: Option<&str>, token_ids: &[u32]) -> Frame {
        let mut choice = Map::new();
        choice.insert("text".to_owned(), json!(text));
        choice.insert("index".to_owned(), json!(0));
        choice.insert("logprobs".to_owned(), Value::Null);
        choice.insert("finish_reason".to_owned(), json!(finish));
        if self.options.return_token_ids {
            choice.insert("token_ids".to_owned(), json!(token_ids));
        }
        self.frame(Value::Array(vec![Value::Object(choice)]), None)
    }

    /// One chunk with whatever choices and accounting it was given.
    fn frame(&self, choices: Value, usage: Option<Value>) -> Frame {
        let mut body = Map::new();
        body.insert("id".to_owned(), json!(self.options.id));
        body.insert("object".to_owned(), json!("text_completion"));
        body.insert("created".to_owned(), json!(self.created));
        body.insert("model".to_owned(), json!(self.options.model));
        body.insert("choices".to_owned(), choices);
        if let Some(usage) = usage {
            body.insert("usage".to_owned(), usage);
        }
        Frame::data(Value::Object(body).to_string())
    }

    /// The finish frame, the accounting frame when asked for, and the
    /// terminator.
    fn close(&mut self, stop: &StopReason, usage: &Usage, token_ids: &[u32]) -> Vec<Frame> {
        self.closed = true;
        let mut frames = vec![self.chunk("", Some(finish_reason(stop)), token_ids)];
        if self.include_usage {
            frames.push(self.frame(Value::Array(Vec::new()), Some(encode_usage(usage))));
        }
        frames.push(Frame::done());
        frames
    }
}

impl FrameEncoder for CompletionChunkEncoder {
    fn on_event(&mut self, event: StreamEvent) -> Vec<Frame> {
        match event {
            StreamEvent::Start { created, .. } => {
                self.created = created;
                // The echo is the first thing on the wire when it was asked for,
                // so a client concatenating `text` gets prompt-then-continuation
                // in the order it would have read them.
                match self.options.echo.clone() {
                    Some(prompt) => vec![self.chunk(&prompt, None, &[])],
                    None => Vec::new(),
                }
            }
            StreamEvent::BlockDelta {
                delta: BlockDelta::Text { text, token_ids },
                ..
            } => vec![self.chunk(&text, None, &token_ids)],
            StreamEvent::Stop {
                stop,
                usage,
                token_ids,
            } => self.close(&stop, &usage, &token_ids),
            StreamEvent::Failed { message } => {
                self.closed = true;
                let body = json!({"error": {
                    "message": message,
                    "type": "server_error",
                    "param": Value::Null,
                    "code": "internal_error",
                }});
                vec![Frame::data(body.to_string()), Frame::done()]
            }
            // Traces, tool calls and refusals have no field on this surface.
            // Dropping them is the only alternative to writing them into the
            // continuation, which would corrupt the document being continued.
            ref other => {
                tracing::debug!(?other, "no completion chunk for this event");
                Vec::new()
            }
        }
    }

    fn on_end(&mut self) -> Vec<Frame> {
        if self.closed {
            return Vec::new();
        }
        tracing::warn!("a stream ended with no terminal event; closing it anyway");
        vec![Frame::done()]
    }
}

#[cfg(test)]
mod tests {
    use super::{CompletionBody, CompletionChunkEncoder, Options, PromptInput, encode_completion};
    use crate::api::sse::{Frame, drain};
    use serde_json::Value;
    use shiso_ir::{BlockDelta, BlockStart, Message, Response, StopReason, StreamEvent, Usage};

    fn body(json: Value) -> CompletionBody {
        serde_json::from_value(json).expect("the body parses")
    }

    #[test]
    fn a_flex_tier_asks_the_scheduler_for_the_flex_class() {
        let flex = super::to_ir(
            &body(serde_json::json!({"prompt": "w1", "service_tier": "flex"})),
            "a-model",
        );
        assert_eq!(flex.service_class, shiso_ir::ServiceClass::Flex);
        let silent = super::to_ir(&body(serde_json::json!({"prompt": "w1"})), "a-model");
        assert_eq!(silent.service_class, shiso_ir::ServiceClass::Standard);
    }

    fn options() -> Options {
        Options {
            id: "cmpl-1".to_owned(),
            model: "a-model".to_owned(),
            echo: None,
            return_token_ids: false,
        }
    }

    fn response() -> Response {
        Response {
            id: "bare".to_owned(),
            model: "a-model".to_owned(),
            created: 1700,
            message: Message::assistant(" and then some"),
            stop: StopReason::EndTurn,
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 4,
                ..Usage::default()
            },
            logprobs: None,
            token_ids: vec![1, 2, 3, 4],
        }
    }

    /// The field means four things, and a list of numbers must not be read as
    /// anything but ids.
    #[test]
    fn a_prompt_parses_as_text_ids_or_a_list_of_either() {
        assert!(matches!(
            body(serde_json::json!({"prompt": "once upon"})).prompt,
            Some(PromptInput::Text(_))
        ));
        assert!(matches!(
            body(serde_json::json!({"prompt": [1, 2, 3]})).prompt,
            Some(PromptInput::Ids(_))
        ));
        assert!(matches!(
            body(serde_json::json!({"prompt": ["a", "b"]})).prompt,
            Some(PromptInput::Texts(_))
        ));
        assert!(matches!(
            body(serde_json::json!({"prompt": [[1, 2], [3]]})).prompt,
            Some(PromptInput::IdLists(_))
        ));
    }

    #[test]
    fn a_body_with_no_prompt_is_refused_before_anything_is_tokenized() {
        let error = body(serde_json::json!({}))
            .validate()
            .expect_err("no prompt");
        assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn more_than_one_continuation_is_refused_rather_than_clamped() {
        let error = body(serde_json::json!({"prompt": "x", "n": 3}))
            .validate()
            .expect_err("n is more than one");
        assert_eq!(error.code(), "unsupported_value");
    }

    #[test]
    fn a_blocking_completion_is_a_text_completion_with_one_choice() {
        let encoded = encode_completion(&response(), &options());
        assert_eq!(encoded["object"], "text_completion");
        assert_eq!(encoded["id"], "cmpl-1");
        assert_eq!(encoded["choices"][0]["text"], " and then some");
        assert_eq!(encoded["choices"][0]["index"], 0);
        assert_eq!(encoded["choices"][0]["finish_reason"], "stop");
        assert_eq!(encoded["usage"]["total_tokens"], 7);
    }

    /// Echo puts the prompt in front of the continuation, so a client reading
    /// only `text` gets the whole document.
    #[test]
    fn echoing_repeats_the_prompt_before_the_continuation() {
        let mut options = options();
        options.echo = Some("once upon a time".to_owned());
        let encoded = encode_completion(&response(), &options);
        assert_eq!(
            encoded["choices"][0]["text"],
            "once upon a time and then some"
        );
    }

    fn events() -> Vec<StreamEvent> {
        vec![
            StreamEvent::Start {
                id: "bare".to_owned(),
                model: "a-model".to_owned(),
                created: 1700,
            },
            StreamEvent::BlockStart {
                index: 0,
                block: BlockStart::Text,
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Text {
                    text: " and".to_owned(),
                    token_ids: vec![1],
                },
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Text {
                    text: " then".to_owned(),
                    token_ids: vec![2],
                },
            },
            StreamEvent::Stop {
                stop: StopReason::EndTurn,
                usage: Usage {
                    prompt_tokens: 3,
                    completion_tokens: 2,
                    ..Usage::default()
                },
                token_ids: Vec::new(),
            },
        ]
    }

    fn bodies(frames: &[Frame]) -> Vec<Value> {
        frames
            .iter()
            .filter(|frame| frame.data != "[DONE]")
            .map(|frame| serde_json::from_str(&frame.data).expect("every frame is JSON"))
            .collect()
    }

    #[test]
    fn a_streamed_completion_is_text_frames_then_a_finish_then_done() {
        let frames = drain(CompletionChunkEncoder::new(options(), false), events());
        assert_eq!(frames.last().expect("a terminator"), &Frame::done());
        let bodies = bodies(&frames);
        assert_eq!(bodies.len(), 3);
        assert_eq!(bodies[0]["choices"][0]["text"], " and");
        assert_eq!(bodies[1]["choices"][0]["text"], " then");
        assert_eq!(bodies[2]["choices"][0]["finish_reason"], "stop");
        for body in &bodies {
            assert_eq!(body["object"], "text_completion");
        }
    }

    /// The echo is the first frame, so concatenating `text` across the stream
    /// gives the same string a blocking echoing caller would have read.
    #[test]
    fn an_echoing_stream_sends_the_prompt_as_its_first_frame() {
        let mut options = options();
        options.echo = Some("once upon".to_owned());
        let frames = drain(CompletionChunkEncoder::new(options, false), events());
        let joined: String = bodies(&frames)
            .iter()
            .filter_map(|body| body["choices"][0]["text"].as_str())
            .collect();
        assert_eq!(joined, "once upon and then");
    }

    #[test]
    fn asking_for_the_accounting_adds_a_frame_with_no_choices() {
        let frames = drain(CompletionChunkEncoder::new(options(), true), events());
        let bodies = bodies(&frames);
        let totals = bodies.last().expect("the accounting frame");
        assert_eq!(totals["choices"].as_array().expect("an array").len(), 0);
        assert_eq!(totals["usage"]["total_tokens"], 5);
    }

    /// A trace has no field on this surface, and writing it into the text would
    /// corrupt the document being continued.
    #[test]
    fn a_trace_never_reaches_the_continuation() {
        let mut events = events();
        events.insert(
            2,
            StreamEvent::BlockDelta {
                index: 1,
                delta: BlockDelta::Thinking {
                    text: "hmm".to_owned(),
                    token_ids: vec![9],
                },
            },
        );
        let frames = drain(CompletionChunkEncoder::new(options(), false), events);
        let joined: String = bodies(&frames)
            .iter()
            .filter_map(|body| body["choices"][0]["text"].as_str())
            .collect();
        assert_eq!(joined, " and then");
    }
}
