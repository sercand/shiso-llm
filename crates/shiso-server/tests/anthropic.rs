//! The Messages surface, driven through the router.
//!
//! What the unit tests beside each module assert is the mapping - a body into
//! the neutral request, a turn into a wire shape, a stream into frames. What
//! this file asserts is that they are wired together: a real request over a
//! real router reaches a real engine and comes back in the shape a client of
//! this format reads, including the parts only the assembled path can get
//! wrong.
//!
//! The model behind it answers a scripted list of token ids, and the vocabulary
//! is built here so that an id is a readable word - which is what makes a
//! scripted answer something an assertion can name.

mod common;

use std::str::FromStr as _;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use shiso_server::tokenizer::Vocabulary;
use shiso_server::tokenizer::jinja_helpers::ToolJsonStyle;
use shiso_server::tokenizer::render_prompt::PromptRenderer;
use tower::ServiceExt as _;

/// Tokens the fake vocabulary defines, matching the fake model's own.
const VOCAB: u32 = 64;

/// The marker every unrecognised word encodes to.
const UNKNOWN: &str = "<unk>";

/// The context every state is built with, unless a test wants a smaller one.
const CTX: u32 = 256;

/// A template short enough that its whole output can be asserted on.
const TEMPLATE: &str = concat!(
    "{% if tools %}<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools>{% endif %}",
    "{% for m in messages %}<|{{ m.role }}|>",
    "{% for part in m.content %}{{ part.text }}{% endfor %}<|end|>{% endfor %}",
    "<|assistant|>",
);

/// A vocabulary of sixty-four one-word tokens, built in memory.
fn test_vocabulary() -> Arc<Vocabulary> {
    let mut vocab = serde_json::Map::new();
    vocab.insert(UNKNOWN.to_owned(), json!(0));
    for id in 1..VOCAB {
        vocab.insert(format!("w{id}"), json!(id));
    }
    let document = json!({
        "version": "1.0",
        "truncation": Value::Null,
        "padding": Value::Null,
        "added_tokens": [],
        "normalizer": Value::Null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": Value::Null,
        "decoder": Value::Null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": UNKNOWN},
    })
    .to_string();
    let inner = tokenizers::Tokenizer::from_str(&document).expect("the vocabulary loads");
    Arc::new(Vocabulary::from_tokenizer(inner))
}

/// A server whose model answers `script`, with `ctx` tokens of room.
fn state_with_ctx(script: Vec<u32>, ctx: u32) -> Arc<AppState> {
    let vocabulary = test_vocabulary();
    let (engine, _join, _teardowns) = common::fake_engine(script, Some(Arc::clone(&vocabulary)));
    let renderer = PromptRenderer::new(TEMPLATE, ToolJsonStyle::Compact, Arc::clone(&vocabulary))
        .expect("the test template compiles");
    Arc::new(
        AppState::new(engine, "test-model".to_owned(), ctx)
            .with_vocabulary(vocabulary)
            .with_renderer(Arc::new(renderer)),
    )
}

/// A server with the ordinary amount of room.
fn state(script: Vec<u32>) -> Arc<AppState> {
    state_with_ctx(script, CTX)
}

/// A minimal body, with greedy decoding asked for.
///
/// The scripted model writes a spike into the logits row and the real sampler
/// draws from it, so a test that left the temperature at the neutral preset
/// would be asserting on a random draw rather than on the scripted answer.
fn wire_body(extra: &Value) -> Value {
    let mut out = json!({
        "model": "claude-test",
        "max_tokens": 32,
        "temperature": 0,
        "messages": [{"role": "user", "content": "w1 w2"}],
    });
    if let (Some(fields), Some(extra)) = (out.as_object_mut(), extra.as_object()) {
        for (key, value) in extra {
            fields.insert(key.clone(), value.clone());
        }
    }
    out
}

/// One POST to the messages route.
fn post(body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("a POST request builds")
}

/// Sends one request and reads the whole answer as JSON.
async fn call(state: &Arc<AppState>, request: Request<Body>) -> (StatusCode, Value) {
    let response = build_router(Arc::clone(state))
        .oneshot(request)
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, body)
}

/// The `event:` / `data:` pairs of a streamed answer, in order.
async fn frames(state: &Arc<AppState>, request: Request<Body>) -> Vec<(String, Value)> {
    let response = build_router(Arc::clone(state))
        .oneshot(request)
        .await
        .expect("the router answers");
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "a stream starts with a 200"
    );
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default();
    assert!(
        content_type.starts_with("text/event-stream"),
        "a stream says what it is: {content_type}"
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let text = String::from_utf8(bytes.to_vec()).expect("a stream is text");
    // One frame is the lines up to a blank one, and the two lines inside it
    // arrive in whichever order the HTTP stack writes them - so a frame is read
    // as a unit rather than by remembering the last name seen.
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        let mut name = String::new();
        let mut data: Option<Value> = None;
        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("event: ") {
                name.clear();
                name.push_str(rest);
            } else if let Some(rest) = line.strip_prefix("data: ") {
                data = Some(serde_json::from_str(rest).expect("every frame is JSON"));
            }
        }
        if let Some(data) = data {
            out.push((name, data));
        }
    }
    out
}

#[tokio::test]
async fn a_message_answers_with_the_turn_the_model_produced() {
    let state = state(vec![10, 11]);
    let (status, body) = call(&state, post(&wire_body(&json!({})))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert!(
        body["id"].as_str().is_some_and(|id| id.starts_with("msg_")),
        "{body}"
    );
    assert_eq!(
        body["model"], "test-model",
        "the model echoed is the checkpoint this process serves"
    );
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], "w10 w11");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["stop_sequence"], json!(null));
    assert!(
        body["usage"]["input_tokens"]
            .as_u64()
            .is_some_and(|n| n > 0),
        "the accounting names the prompt it actually ran: {body}"
    );
    assert_eq!(
        body["usage"]["output_tokens"], 3,
        "the token that ends the turn is generated too: {body}"
    );
    assert!(body["usage"]["cache_read_input_tokens"].is_number());
}

/// The cap is mandatory in this format, and a body without one is refused in
/// this format's error shape rather than run with a cap the caller never chose.
#[tokio::test]
async fn a_request_without_a_token_cap_is_refused_in_this_surfaces_shape() {
    let state = state(vec![10]);
    let (status, body) = call(
        &state,
        post(&json!({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "w1"}],
        })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("max_tokens")),
        "the refusal names the field: {body}"
    );
}

#[tokio::test]
async fn the_validation_table_is_refused_before_anything_is_generated() {
    let state = state(vec![10]);
    for (name, extra) in [
        ("an empty conversation", json!({"messages": []})),
        ("a cap of zero", json!({"max_tokens": 0})),
        ("a temperature out of range", json!({"temperature": 1.5})),
        (
            "a forced call with nothing to call",
            json!({"tool_choice": {"type": "any"}}),
        ),
    ] {
        let (status, body) = call(&state, post(&wire_body(&extra))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{name}: {body}");
        assert_eq!(body["type"], "error", "{name}: {body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{name}");
    }
}

/// A prompt that leaves no room to answer is refused with the two numbers a
/// client needs to retry, in this surface's body shape.
#[tokio::test]
async fn a_prompt_that_does_not_fit_is_refused_with_the_numbers_to_retry_with() {
    let state = state_with_ctx(vec![10], 8);
    let long = (1..40)
        .map(|id| format!("w{id}"))
        .collect::<Vec<_>>()
        .join(" ");
    let (status, body) = call(
        &state,
        post(&wire_body(
            &json!({"messages": [{"role": "user", "content": long}]}),
        )),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["n_ctx"], 8);
    assert!(
        body["error"]["n_prompt_tokens"]
            .as_u64()
            .is_some_and(|n| n >= 8),
        "{body}"
    );
}

/// The sequence that ended the turn is echoed by the one reason that can have
/// one, and the whole chain - engine, accumulator, encoder - has to agree about
/// it for the body to say so.
#[tokio::test]
async fn a_stop_sequence_ends_the_turn_and_is_echoed_in_the_answer() {
    let state = state(vec![10, 11, 12]);
    let (status, body) = call(
        &state,
        post(&wire_body(&json!({"stop_sequences": ["w11"]}))),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["stop_reason"], "stop_sequence");
    assert_eq!(body["stop_sequence"], "w11");
    assert_eq!(
        body["content"][0]["text"], "w10 ",
        "everything before the sequence is kept, and the sequence is not: {body}"
    );
}

#[tokio::test]
async fn a_cap_the_caller_set_is_what_ends_the_turn() {
    let state = state(vec![10, 11, 12, 13, 14]);
    let (status, body) = call(&state, post(&wire_body(&json!({"max_tokens": 2})))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["stop_reason"], "max_tokens");
    assert_eq!(body["content"][0]["text"], "w10 w11");
}

/// A block kind this server cannot render is refused rather than dropped: an
/// answer about content the model never saw looks exactly like a good answer.
#[tokio::test]
async fn a_block_this_server_cannot_render_is_refused_and_says_which_one() {
    let state = state(vec![10]);
    let (status, body) = call(
        &state,
        post(&wire_body(&json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "w1"},
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "QQ=="}},
        ]}]}))),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("messages[0].content[1]")),
        "{body}"
    );
}

/// The instructions are rendered in the place the template keeps for them,
/// rather than as a turn of the conversation.
#[tokio::test]
async fn the_instructions_reach_the_prompt_without_becoming_a_turn() {
    let state = state(vec![10]);
    let (status, body) = call(
        &state,
        post(&wire_body(&json!({"system": "w3 w4", "max_tokens": 1}))),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // The system turn is rendered ahead of the caller's, so the prompt is
    // longer than the same request without one.
    let with_system = body["usage"]["input_tokens"]
        .as_u64()
        .expect("the accounting is counted");
    let (_status, plain) = call(&state, post(&wire_body(&json!({"max_tokens": 1})))).await;
    let without = plain["usage"]["input_tokens"]
        .as_u64()
        .expect("the accounting is counted");
    assert!(
        with_system > without,
        "instructions reach the prompt: {with_system} vs {without}"
    );
}

#[tokio::test]
async fn a_streamed_message_is_the_named_sequence_a_client_of_this_format_reads() {
    let state = state(vec![10, 11]);
    let streamed = frames(&state, post(&wire_body(&json!({"stream": true})))).await;
    let names: Vec<&str> = streamed.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ],
        "{streamed:?}"
    );
    let first = &streamed[0].1;
    assert_eq!(first["type"], "message_start");
    assert!(
        first["message"]["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("msg_")),
        "{first}"
    );
    assert_eq!(first["message"]["model"], "test-model");

    let text: String = streamed
        .iter()
        .filter(|(name, _)| name == "content_block_delta")
        .filter_map(|(_, body)| body["delta"]["text"].as_str())
        .collect::<Vec<&str>>()
        .join("");
    assert_eq!(
        text, "w10 w11",
        "the deltas concatenate to the blocking answer"
    );

    let (_name, closing) = &streamed[streamed.len() - 2];
    assert_eq!(closing["delta"]["stop_reason"], "end_turn");
    assert_eq!(closing["usage"]["output_tokens"], 3);
}

/// Both encodings read the same events, so the text a stream delivers and the
/// text a blocking answer carries are the same text.
#[tokio::test]
async fn a_streamed_message_concatenates_to_the_blocking_one() {
    let blocking = state(vec![10, 11, 12]);
    let (_status, body) = call(&blocking, post(&wire_body(&json!({})))).await;
    let whole = body["content"][0]["text"]
        .as_str()
        .expect("the answer is text")
        .to_owned();

    let streaming = state(vec![10, 11, 12]);
    let streamed = frames(&streaming, post(&wire_body(&json!({"stream": true})))).await;
    let joined: String = streamed
        .iter()
        .filter_map(|(_, frame)| frame["delta"]["text"].as_str())
        .collect();
    assert_eq!(joined, whole);
}

/// The credential header this format's clients send is the same credential, and
/// a client that sets it must not be refused before its request is read.
#[tokio::test]
async fn the_key_header_this_format_uses_opens_the_gate_and_a_wrong_one_does_not() {
    let vocabulary = test_vocabulary();
    let (engine, _join, _teardowns) = common::fake_engine(vec![10], Some(Arc::clone(&vocabulary)));
    let renderer = PromptRenderer::new(TEMPLATE, ToolJsonStyle::Compact, Arc::clone(&vocabulary))
        .expect("the test template compiles");
    let auth = shiso_server::auth::AuthConfig::new(vec!["sk-test".to_owned()])
        .expect("one token makes a gate");
    let state = Arc::new(
        AppState::new(engine, "test-model".to_owned(), CTX)
            .with_vocabulary(vocabulary)
            .with_renderer(Arc::new(renderer))
            .with_auth(auth),
    );

    let keyed = |key: &str| {
        Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .body(Body::from(wire_body(&json!({})).to_string()))
            .expect("a POST request builds")
    };

    let (status, answered) = call(&state, keyed("sk-test")).await;
    assert_eq!(status, StatusCode::OK, "{answered}");
    assert_eq!(answered["type"], "message");

    let (status, refused) = call(&state, keyed("sk-wrong")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{refused}");
}

/// The opening frame carries the input accounting, and it is sent before a
/// single token has been generated - so the number cannot come from the turn's
/// own usage, which does not exist yet. It comes from the pipeline, which
/// rendered the prompt and counted it.
///
/// A zero here is corrected by the terminal frame, which sounds harmless and is
/// not: a client that reads the opening frame to decide whether to keep going
/// would have been told the prompt was free, and the two frames of one message
/// would disagree about what it was charged for.
#[tokio::test]
async fn the_opening_frame_reports_the_prompt_that_was_actually_rendered() {
    let state = state(vec![10, 11]);
    let streamed = frames(&state, post(&wire_body(&json!({"stream": true})))).await;

    let opening = &streamed[0].1;
    let opened_with = opening["message"]["usage"]["input_tokens"]
        .as_u64()
        .expect("the opening frame carries the input accounting");
    assert!(
        opened_with > 0,
        "the prompt was rendered and counted before the stream opened: {opening}"
    );

    let (_name, closing) = &streamed[streamed.len() - 2];
    assert_eq!(
        closing["usage"]["input_tokens"].as_u64(),
        Some(opened_with),
        "the two frames of one message agree about what it cost: {closing}"
    );

    // And the same number the blocking body reports for the same request.
    let (status, body) = call(&state, post(&wire_body(&json!({})))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["usage"]["input_tokens"].as_u64(),
        Some(opened_with),
        "streaming and blocking agree about the prompt: {body}"
    );
}
