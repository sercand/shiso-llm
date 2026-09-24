//! The three `OpenAI` surfaces, driven through the router.
//!
//! What the unit tests beside each module assert is the mapping - a body into
//! the neutral request, a turn into a wire shape, a stream into frames. What
//! this file asserts is that the three are wired together: a real request over
//! a real router reaches a real engine and comes back in the shape a client
//! reads, including the parts only the assembled path can get wrong.
//!
//! The model behind it answers a scripted list of token ids, and the vocabulary
//! is built here so that an id is a readable word. A scripted answer is
//! therefore something an assertion can name.

mod common;

use std::str::FromStr as _;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use shiso_server::response_store::{ResponseStore, StoredKind};
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

/// A template that reads the same variables a checkpoint's does, written so its
/// output is short enough to assert on in full.
const TEMPLATE: &str = concat!(
    "{% if tools %}<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools>{% endif %}",
    "{% for m in messages %}<|{{ m.role }}|>",
    "{% for part in m.content %}{{ part.text }}{% endfor %}<|end|>{% endfor %}",
    "<|assistant|>",
);

/// A vocabulary of sixty-four one-word tokens, built in memory.
///
/// Word level and deterministic on purpose: an id is a word, so a scripted
/// answer is readable in an assertion. It is written as the document the
/// library loads rather than assembled through its builders, because the
/// document is a stable public interface and the builder types are not - and
/// because it is the same path a checkpoint's own vocabulary takes.
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
///
/// The join handle is dropped rather than joined: ending the engine loop is the
/// boundary test's subject, and doing it here would make every wire test depend
/// on a thread teardown.
fn state_with_ctx(script: Vec<u32>, ctx: u32) -> Arc<AppState> {
    let vocabulary = test_vocabulary();
    let (engine, _join, _teardowns) = common::fake_engine(script, Some(Arc::clone(&vocabulary)));
    let renderer = PromptRenderer::new(TEMPLATE, ToolJsonStyle::Compact, Arc::clone(&vocabulary))
        .expect("the test template compiles");
    Arc::new(
        AppState::new(engine, "test-model".to_owned(), ctx)
            .with_vocabulary(vocabulary)
            .with_renderer(Arc::new(renderer))
            .with_response_store(ResponseStore::new(16, Duration::from_secs(60))),
    )
}

/// A server with the ordinary amount of room.
fn state(script: Vec<u32>) -> Arc<AppState> {
    state_with_ctx(script, CTX)
}

/// The same body with greedy decoding asked for.
///
/// The scripted model writes a spike into the logits row and the real sampler
/// draws from it, so a test that left the temperature at the neutral preset
/// would be asserting on a random draw rather than on the scripted answer. Zero
/// temperature is what selects the argmax path, which means these tests still
/// go through the sampler rather than around it.
fn greedy(mut body: Value) -> Value {
    if let Some(fields) = body.as_object_mut() {
        fields.insert("temperature".to_owned(), json!(0));
    }
    body
}

/// One POST with a JSON body.
fn post(path: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("a POST request builds")
}

/// Sends one request and reads the whole answer as JSON.
async fn call(state: &Arc<AppState>, request: Request<Body>) -> (StatusCode, Value) {
    let (status, bytes) = raw(state, request).await;
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, body)
}

/// Sends one request and reads the whole answer as bytes.
async fn raw(state: &Arc<AppState>, request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = build_router(Arc::clone(state))
        .oneshot(request)
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    (status, bytes.to_vec())
}

/// The `data:` payloads of a streamed answer, in order.
async fn frames(state: &Arc<AppState>, request: Request<Body>) -> Vec<String> {
    let (status, bytes) = raw(state, request).await;
    assert_eq!(status, StatusCode::OK, "a stream starts with a 200");
    String::from_utf8(bytes)
        .expect("a stream is text")
        .lines()
        .filter_map(|line| line.strip_prefix("data: ").map(ToOwned::to_owned))
        .collect()
}

/// The JSON frames of a streamed answer, terminator excluded.
fn json_frames(frames: &[String]) -> Vec<Value> {
    frames
        .iter()
        .filter(|frame| frame.as_str() != "[DONE]")
        .map(|frame| serde_json::from_str(frame).expect("every frame is JSON"))
        .collect()
}

// --- chat completions ----------------------------------------------------

#[tokio::test]
async fn a_chat_completion_answers_with_the_turn_the_model_produced() {
    let state = state(vec![10, 11]);
    let (status, body) = call(
        &state,
        post(
            "/v1/chat/completions",
            &greedy(
                json!({"model": "test-model", "messages": [{"role": "user", "content": "w1 w2"}]}),
            ),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "chat.completion");
    assert!(
        body["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("chatcmpl-")),
        "{body}"
    );
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "w10 w11");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert!(
        body["usage"]["prompt_tokens"]
            .as_u64()
            .is_some_and(|n| n > 0),
        "the accounting names the prompt it actually ran: {body}"
    );
}

/// Every failure this surface can reach before generating, in the order a
/// caller would fix them.
#[tokio::test]
async fn the_validation_table_refuses_in_this_surfaces_own_error_shape() {
    let state = state(vec![10]);
    let cases: Vec<(Value, &str)> = vec![
        (json!({"messages": []}), "messages"),
        (
            json!({"messages": [{"role": "user", "content": "hi"}], "temperature": 9.0}),
            "temperature",
        ),
        (
            json!({"messages": [{"role": "user", "content": "hi"}], "top_p": 0.0}),
            "top_p",
        ),
        (
            json!({"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 0}),
            "max_completion_tokens",
        ),
        (
            json!({"messages": [{"role": "user", "content": "hi"}], "n": 4}),
            "n",
        ),
        (
            json!({"messages": [{"role": "user", "content": "hi"}], "tool_choice": "required"}),
            "tool_choice",
        ),
    ];
    for (body, param) in cases {
        let (status, answer) = call(&state, post("/v1/chat/completions", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(answer["error"]["type"], "invalid_request_error", "{answer}");
        assert_eq!(answer["error"]["param"], param, "{answer}");
    }
}

/// A role this server does not serve is refused at the edge, which is the only
/// place that still knows which field and which surface to name.
#[tokio::test]
async fn a_role_this_server_does_not_serve_is_refused_by_the_parser() {
    let state = state(vec![10]);
    let (status, body) = call(
        &state,
        post(
            "/v1/chat/completions",
            &json!({"messages": [{"role": "robot", "content": "hi"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("robot")),
        "{body}"
    );
}

/// The two numbers a client needs in order to retry travel as numbers, not as
/// words inside a sentence.
#[tokio::test]
async fn a_prompt_that_does_not_fit_is_refused_with_the_numbers_to_retry_with() {
    let state = state_with_ctx(vec![10], 2);
    let (status, body) = call(
        &state,
        post(
            "/v1/chat/completions",
            &greedy(json!({"messages": [{"role": "user", "content": "w1 w2 w3 w4 w5"}]})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "context_length_exceeded");
    assert_eq!(body["error"]["param"], "messages");
    assert_eq!(body["error"]["n_ctx"], 2);
    assert!(
        body["error"]["n_prompt_tokens"]
            .as_u64()
            .is_some_and(|n| n >= 2),
        "{body}"
    );
}

/// A stored chat completion keeps the conversation with the generated turn
/// appended, which is what a continuation would need.
#[tokio::test]
async fn a_stored_chat_completion_keeps_the_whole_conversation() {
    let state = state(vec![10, 11]);
    let (status, body) = call(
        &state,
        post(
            "/v1/chat/completions",
            &greedy(json!({
                "messages": [{"role": "user", "content": "w1"}],
                "store": true
            })),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let id = body["id"].as_str().expect("an identifier");
    let stored = state
        .response_store
        .get(id, StoredKind::ChatCompletion)
        .expect("the turn was stored");
    assert_eq!(stored.messages.len(), 2, "the caller's turn and the answer");
    assert_eq!(stored.messages[1].text(), "w10 w11");
    assert_eq!(stored.body["id"], body["id"]);
}

#[tokio::test]
async fn a_chat_completion_that_was_not_asked_to_be_stored_is_not() {
    let state = state(vec![10]);
    let (_status, body) = call(
        &state,
        post(
            "/v1/chat/completions",
            &greedy(json!({"messages": [{"role": "user", "content": "w1"}]})),
        ),
    )
    .await;
    let id = body["id"].as_str().expect("an identifier");
    assert!(
        state
            .response_store
            .get(id, StoredKind::ChatCompletion)
            .is_none()
    );
}

/// The concatenation of a stream has to equal the blocking answer, or the two
/// encodings are describing different turns.
#[tokio::test]
async fn a_streamed_chat_completion_concatenates_to_the_blocking_one() {
    let blocking = state(vec![10, 11]);
    let (_status, answer) = call(
        &blocking,
        post(
            "/v1/chat/completions",
            &greedy(json!({"messages": [{"role": "user", "content": "w1"}]})),
        ),
    )
    .await;

    let streaming = state(vec![10, 11]);
    let frames = frames(
        &streaming,
        post(
            "/v1/chat/completions",
            &greedy(json!({
                "messages": [{"role": "user", "content": "w1"}],
                "stream": true
            })),
        ),
    )
    .await;
    assert_eq!(frames.last().map(String::as_str), Some("[DONE]"));
    let bodies = json_frames(&frames);
    assert_eq!(bodies[0]["choices"][0]["delta"]["role"], "assistant");
    let joined: String = bodies
        .iter()
        .filter_map(|body| body["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(joined, answer["choices"][0]["message"]["content"]);
    assert_eq!(
        bodies.last().expect("a finish frame")["choices"][0]["finish_reason"],
        answer["choices"][0]["finish_reason"]
    );
}

#[tokio::test]
async fn a_streaming_caller_that_asks_for_the_accounting_gets_one_extra_frame() {
    let state = state(vec![10]);
    let frames = frames(
        &state,
        post(
            "/v1/chat/completions",
            &greedy(json!({
                "messages": [{"role": "user", "content": "w1"}],
                "stream": true,
                "stream_options": {"include_usage": true}
            })),
        ),
    )
    .await;
    let bodies = json_frames(&frames);
    let totals = bodies.last().expect("the accounting frame");
    assert_eq!(totals["choices"].as_array().expect("an array").len(), 0);
    assert!(
        totals["usage"]["total_tokens"]
            .as_u64()
            .is_some_and(|n| n > 0)
    );
}

/// A sequence the caller asked to stop on ends the turn, and the sequence is
/// not part of the answer. Both encodings have to agree about that, which they
/// do because there is one implementation and it sits where the text is made.
#[tokio::test]
async fn a_stop_sequence_ends_the_turn_and_is_not_in_the_answer() {
    let state = state(vec![10, 11, 12]);
    let (status, body) = call(
        &state,
        post(
            "/v1/chat/completions",
            &greedy(json!({
                "messages": [{"role": "user", "content": "w1"}],
                "stop": ["w11"]
            })),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .expect("some content");
    assert!(
        !content.contains("w11"),
        "the sequence is not returned: {content}"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

// --- legacy completions --------------------------------------------------

#[tokio::test]
async fn a_completion_continues_the_prompt_without_a_conversation_around_it() {
    let state = state(vec![10, 11]);
    let (status, body) = call(
        &state,
        post("/v1/completions", &greedy(json!({"prompt": "w1 w2"}))),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "text_completion");
    assert!(
        body["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("cmpl-"))
    );
    assert_eq!(body["choices"][0]["text"], "w10 w11");
    assert_eq!(
        body["usage"]["prompt_tokens"], 2,
        "no template markers: {body}"
    );
}

#[tokio::test]
async fn an_echoing_completion_repeats_the_prompt_before_the_continuation() {
    let state = state(vec![10]);
    let (_status, body) = call(
        &state,
        post(
            "/v1/completions",
            &greedy(json!({"prompt": "w1 w2", "echo": true})),
        ),
    )
    .await;
    assert_eq!(body["choices"][0]["text"], "w1 w2w10");
}

#[tokio::test]
async fn a_completion_prompt_of_token_ids_runs_those_ids() {
    let state = state(vec![10]);
    let (status, body) = call(
        &state,
        post("/v1/completions", &greedy(json!({"prompt": [1, 2, 3]}))),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["usage"]["prompt_tokens"], 3);
}

/// An id past the end of the embedding table is a device-side out-of-bounds
/// read, which would reach the caller as a backend failure saying nothing about
/// the id they sent.
#[tokio::test]
async fn a_completion_prompt_naming_an_id_the_model_has_no_row_for_is_refused() {
    let state = state(vec![10]);
    let (status, body) = call(
        &state,
        post("/v1/completions", &json!({"prompt": [1, 900_000]})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "prompt");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("900000")),
        "{body}"
    );
}

#[tokio::test]
async fn a_completion_asking_for_several_documents_is_refused_rather_than_clamped() {
    let state = state(vec![10]);
    let (status, body) = call(
        &state,
        post("/v1/completions", &json!({"prompt": ["a", "b"]})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "unsupported_value", "{body}");
}

#[tokio::test]
async fn a_streamed_completion_is_text_frames_and_then_the_terminator() {
    let state = state(vec![10, 11]);
    let frames = frames(
        &state,
        post(
            "/v1/completions",
            &greedy(json!({"prompt": "w1", "stream": true})),
        ),
    )
    .await;
    assert_eq!(frames.last().map(String::as_str), Some("[DONE]"));
    let bodies = json_frames(&frames);
    let joined: String = bodies
        .iter()
        .filter_map(|body| body["choices"][0]["text"].as_str())
        .collect();
    assert_eq!(joined, "w10 w11");
    assert!(
        bodies
            .iter()
            .all(|body| body["object"] == "text_completion")
    );
}

// --- responses -----------------------------------------------------------

#[tokio::test]
async fn a_response_answers_with_output_items_in_block_order() {
    let state = state(vec![10, 11]);
    let (status, body) = call(
        &state,
        post("/v1/responses", &greedy(json!({"input": "w1"}))),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "response");
    assert!(
        body["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("resp_"))
    );
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["content"][0]["text"], "w10 w11");
    assert!(
        body["usage"]["input_tokens"]
            .as_u64()
            .is_some_and(|n| n > 0)
    );
}

/// The round trip the whole store exists for: store one turn, name it in the
/// next request, and the conversation continues.
#[tokio::test]
async fn a_stored_response_can_be_continued_and_fetched_again() {
    let state = state(vec![10, 11]);
    let (_status, first) = call(
        &state,
        post(
            "/v1/responses",
            &greedy(json!({"input": "w1", "store": true})),
        ),
    )
    .await;
    let id = first["id"].as_str().expect("an identifier").to_owned();

    let (status, fetched) = call(
        &state,
        Request::builder()
            .uri(format!("/v1/responses/{id}"))
            .body(Body::empty())
            .expect("a GET request builds"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetched["id"], first["id"]);

    let (status, second) = call(
        &state,
        post(
            "/v1/responses",
            &greedy(json!({"input": "w2", "previous_response_id": id})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(second["previous_response_id"], first["id"]);
    assert!(
        second["usage"]["input_tokens"].as_u64().unwrap_or(0)
            > first["usage"]["input_tokens"].as_u64().unwrap_or(0),
        "continuing sends the model more prompt than starting fresh: {second}"
    );
}

/// A fresh answer to a continuation request is a server that says nothing went
/// wrong while showing the model none of the history.
#[tokio::test]
async fn continuing_a_conversation_the_server_forgot_is_refused() {
    let state = state(vec![10]);
    let (status, body) = call(
        &state,
        post(
            "/v1/responses",
            &json!({"input": "w1", "previous_response_id": "resp_missing"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "response_not_found");
    assert_eq!(body["error"]["param"], "previous_response_id");
}

#[tokio::test]
async fn a_hosted_tool_is_refused_by_name() {
    let state = state(vec![10]);
    let (status, body) = call(
        &state,
        post(
            "/v1/responses",
            &json!({"input": "w1", "tools": [{"type": "code_interpreter"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("code_interpreter")),
        "{body}"
    );
}

/// The named-event sequence, in order, with the numbering a client reorders on.
#[tokio::test]
async fn a_streamed_response_is_the_named_sequence_with_monotonic_numbering() {
    let state = state(vec![10, 11]);
    let (status, bytes) = raw(
        &state,
        post(
            "/v1/responses",
            &greedy(json!({"input": "w1", "stream": true})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(bytes).expect("a stream is text");
    let names: Vec<&str> = text
        .lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .collect();
    assert_eq!(names.first(), Some(&"response.created"));
    assert_eq!(names.last(), Some(&"response.completed"));
    assert!(names.contains(&"response.output_item.added"), "{names:?}");
    assert!(names.contains(&"response.output_text.delta"), "{names:?}");
    assert!(names.contains(&"response.output_item.done"), "{names:?}");

    let bodies: Vec<Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|frame| serde_json::from_str(frame).expect("every frame is JSON"))
        .collect();
    let numbers: Vec<u64> = bodies
        .iter()
        .filter_map(|body| body["sequence_number"].as_u64())
        .collect();
    assert_eq!(
        numbers,
        (0..u64::try_from(bodies.len()).unwrap_or(0)).collect::<Vec<u64>>()
    );
    let completed = bodies.last().expect("a terminal frame");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "w10 w11"
    );
}

/// A function call and its output are separate items on this format and one
/// turn in the conversation; the prompt has to show them as the model made them.
#[tokio::test]
async fn a_replayed_call_and_its_output_reach_the_prompt_as_one_exchange() {
    let state = state(vec![10]);
    let (status, body) = call(
        &state,
        post(
            "/v1/responses",
            &greedy(json!({"input": [
                {"role": "user", "content": "w1"},
                {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "w2"}
            ]})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

// --- rendering -----------------------------------------------------------

/// The prompt a request renders to, in full. This is the one golden that a
/// change to the mapping cannot pass by accident: every part of the conversation
/// appears exactly where the template put it, and the tool's keys are in the
/// order the caller wrote them.
#[tokio::test]
async fn a_request_renders_to_the_prompt_the_template_describes() {
    let state = state(vec![10]);
    let request = {
        let body: Value = json!({
            "messages": [
                {"role": "system", "content": "w1"},
                {"role": "user", "content": "w2"}
            ],
            "tools": [{"type": "function", "function": {
                "name": "lookup",
                "description": "looks up",
                "parameters": {"type": "object", "properties": {}}
            }}]
        });
        let parsed = serde_json::from_value(body).expect("the body parses");
        shiso_server::openai::chat_to_ir::to_ir(&parsed, "test-model").expect("it converts")
    };
    let prompt = state
        .renderer
        .as_ref()
        .expect("a renderer")
        .render(&request)
        .expect("it renders");
    assert_eq!(
        prompt.text,
        concat!(
            r#"<tools>{"type":"function","function":{"name":"lookup","#,
            r#""description":"looks up","parameters":{"type":"object","properties":{}}}}</tools>"#,
            "<|system|>w1<|end|><|user|>w2<|end|><|assistant|>",
        )
    );
    assert!(!prompt.tokens.is_empty());
    assert!(prompt.media.is_empty());
}

/// The count the accounting reports is the count of the prompt that was run,
/// not a second count taken somewhere else.
#[tokio::test]
async fn the_accounting_reports_the_prompt_that_was_actually_rendered() {
    let state = state(vec![10]);
    let body = greedy(json!({"messages": [{"role": "user", "content": "w1 w2 w3"}]}));
    let parsed = serde_json::from_value(body.clone()).expect("the body parses");
    let request =
        shiso_server::openai::chat_to_ir::to_ir(&parsed, "test-model").expect("it converts");
    let rendered = state
        .renderer
        .as_ref()
        .expect("a renderer")
        .render(&request)
        .expect("it renders");

    let (_status, answer) = call(&state, post("/v1/chat/completions", &body)).await;
    assert_eq!(
        answer["usage"]["prompt_tokens"].as_u64(),
        Some(rendered.token_count())
    );
}

/// A server with no chat template still serves everything that does not
/// generate, and says what is missing rather than inventing a prompt format.
#[tokio::test]
async fn a_server_with_no_chat_template_refuses_to_generate_and_says_why() {
    let vocabulary = test_vocabulary();
    let (engine, _join, _teardowns) = common::fake_engine(vec![10], Some(vocabulary));
    let state = Arc::new(AppState::new(engine, "test-model".to_owned(), CTX));
    let (status, body) = call(
        &state,
        post(
            "/v1/chat/completions",
            &greedy(json!({"messages": [{"role": "user", "content": "w1"}]})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "server_not_ready");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("renderer")),
        "{body}"
    );
}

// --- storing a streamed turn ---------------------------------------------

/// A streamed turn that asked to be stored is stored, and what is stored is the
/// same conversation a blocking one would have left behind.
///
/// The interesting part is *where* it comes from. The body a re-fetch returns is
/// built after the handler has already handed the response back, so there is no
/// point in the handler where the finished turn exists. It comes from folding
/// the same events the frames were built from, through the same accumulator the
/// blocking path uses - which is why the stored text below is identical to the
/// blocking test's, rather than nearly identical.
#[tokio::test]
async fn a_streamed_chat_completion_that_asked_to_be_stored_is_fetchable() {
    let state = state(vec![10, 11]);
    let streamed = frames(
        &state,
        post(
            "/v1/chat/completions",
            &greedy(json!({
                "messages": [{"role": "user", "content": "w1"}],
                "stream": true,
                "store": true
            })),
        ),
    )
    .await;
    let id = json_frames(&streamed)
        .first()
        .and_then(|frame| frame["id"].as_str().map(ToOwned::to_owned))
        .expect("the first chunk names the turn");

    let stored = state
        .response_store
        .get(&id, StoredKind::ChatCompletion)
        .expect("a streamed turn that asked to be stored was stored");
    assert_eq!(stored.messages.len(), 2, "the caller's turn and the answer");
    assert_eq!(
        stored.messages[1].text(),
        "w10 w11",
        "the stored turn is the one the frames described"
    );
    assert_eq!(stored.body["id"], json!(id));
    assert_eq!(stored.body["object"], "chat.completion");
}

#[tokio::test]
async fn a_streamed_chat_completion_that_did_not_ask_to_be_stored_is_not() {
    let state = state(vec![10]);
    let streamed = frames(
        &state,
        post(
            "/v1/chat/completions",
            &greedy(json!({
                "messages": [{"role": "user", "content": "w1"}],
                "stream": true
            })),
        ),
    )
    .await;
    let id = json_frames(&streamed)
        .first()
        .and_then(|frame| frame["id"].as_str().map(ToOwned::to_owned))
        .expect("the first chunk names the turn");
    assert!(
        state
            .response_store
            .get(&id, StoredKind::ChatCompletion)
            .is_none()
    );
}

/// The round trip the store exists for, over a streamed turn: stream one,
/// continue it by name, and find the first turn's answer in the second
/// request's history.
#[tokio::test]
async fn a_streamed_response_can_be_continued_by_name() {
    let state = state(vec![10, 11]);
    let streamed = frames(
        &state,
        post(
            "/v1/responses",
            &greedy(json!({"input": "w1", "stream": true, "store": true})),
        ),
    )
    .await;
    let id = json_frames(&streamed)
        .iter()
        .find_map(|frame| frame["response"]["id"].as_str().map(ToOwned::to_owned))
        .expect("a streamed response names itself");

    let stored = state
        .response_store
        .get(&id, StoredKind::Response)
        .expect("a streamed response that asked to be stored was stored");
    assert_eq!(stored.messages.len(), 2);
    assert_eq!(stored.messages[1].text(), "w10 w11");

    let (status, second) = call(
        &state,
        post(
            "/v1/responses",
            &greedy(json!({"input": "w2", "previous_response_id": id})),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(second["previous_response_id"], json!(id));
}

// --- service tier ---------------------------------------------------------

/// The same POST with an `X-Service-Tier` header.
fn post_tier(path: &str, body: &Value, tier: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header("x-service-tier", tier)
        .body(Body::from(body.to_string()))
        .expect("a POST request builds")
}

/// Every `OpenAI` surface reads the header, and a flex request on an otherwise
/// idle server is simply served.
#[tokio::test]
async fn every_openai_surface_serves_a_flex_request_named_by_the_header() {
    let state = state(vec![10, 11]);
    let (status, body) = call(
        &state,
        post_tier(
            "/v1/responses",
            &greedy(json!({"input": "w1 w2", "max_output_tokens": 4})),
            "flex",
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["service_tier"], "flex",
        "the header is echoed as the tier the body did not name"
    );

    let (status, body) = call(
        &state,
        post_tier(
            "/v1/completions",
            &greedy(json!({"prompt": "w1 w2"})),
            "flex",
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, body) = call(
        &state,
        post_tier(
            "/v1/chat/completions",
            &greedy(json!({"messages": [{"role": "user", "content": "w1"}]})),
            "flex",
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["service_tier"], "flex");
}

/// A tier this server does not have is refused on every surface, before
/// anything is queued.
#[tokio::test]
async fn an_unknown_tier_in_the_header_is_refused_on_every_openai_surface() {
    let state = state(vec![10]);
    for (path, body) in [
        ("/v1/responses", json!({"input": "w1"})),
        ("/v1/completions", json!({"prompt": "w1"})),
        (
            "/v1/chat/completions",
            json!({"messages": [{"role": "user", "content": "w1"}]}),
        ),
    ] {
        let (status, answer) = call(&state, post_tier(path, &body, "premium")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {answer}");
        assert_eq!(answer["error"]["param"], "service_tier", "{path}: {answer}");
    }
}
