//! A turn whose reasoning opener was spent by the chat template.
//!
//! Qwen 3.5's template closes the generation prompt with `<think>\n` when
//! thinking is on, and with `<think>\n\n</think>\n\n` when it is off. Either
//! way the model never emits an opener: with thinking on it emits the trace and
//! then a bare `</think>`, and with thinking off it emits prose. So the state a
//! turn starts in is a fact about the *rendered prompt*, and a parser that
//! always started in prose handed the whole trace to the client as text with
//! the closing tag still in it.
//!
//! This is the wire-level half of `shiso-parsers`' own `qwen35` cases: what
//! they pin is the parser, what this pins is that the scheduler builds it from
//! the prompt the renderer actually produced.

mod common;

use std::str::FromStr as _;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use shiso_parsers::{ReasoningScheme, ToolCallScheme};
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

/// The context every state is built with.
const CTX: u32 = 256;

/// The id the fake vocabulary gives the closing reasoning tag.
const CLOSE_ID: u32 = 40;

/// Qwen 3.5's template, cut down to the part this is about: the generation
/// prompt ends with the opener when thinking is on, and with an opener already
/// closed when it is off.
const TEMPLATE: &str = concat!(
    "{% for m in messages %}<|{{ m.role }}|>",
    "{% for part in m.content %}{{ part.text }}{% endfor %}<|end|>{% endfor %}",
    "<|assistant|>",
    "{% if enable_thinking %}<think>\n{% else %}<think>\n\n</think>\n\n{% endif %}",
);

/// A vocabulary of one-word tokens, plus the closing reasoning tag at
/// [`CLOSE_ID`] so that a scripted id can end the trace.
fn test_vocabulary() -> Arc<Vocabulary> {
    let mut vocab = serde_json::Map::new();
    vocab.insert(UNKNOWN.to_owned(), json!(0));
    for id in 1..VOCAB {
        if id == CLOSE_ID {
            vocab.insert("</think>".to_owned(), json!(id));
        } else {
            vocab.insert(format!("w{id}"), json!(id));
        }
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

/// A server whose model answers `script`, parsing Qwen 3.5 output.
fn state(script: Vec<u32>, thinking_default: bool) -> Arc<AppState> {
    let vocabulary = test_vocabulary();
    let (engine, _join) = common::parsed_engine(
        script,
        Some(Arc::clone(&vocabulary)),
        (ReasoningScheme::Qwen3, ToolCallScheme::Qwen35),
    );
    let renderer = PromptRenderer::new(TEMPLATE, ToolJsonStyle::Compact, Arc::clone(&vocabulary))
        .expect("the test template compiles")
        .with_thinking_default(thinking_default);
    Arc::new(
        AppState::new(engine, "test-model".to_owned(), CTX)
            .with_vocabulary(vocabulary)
            .with_renderer(Arc::new(renderer)),
    )
}

/// One POST to the messages route, decoded greedily.
async fn answer(state: &Arc<AppState>) -> Value {
    let body = json!({
        "model": "claude-test",
        "max_tokens": 32,
        "temperature": 0,
        "messages": [{"role": "user", "content": "w1 w2"}],
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("a POST request builds");
    let response = build_router(Arc::clone(state))
        .oneshot(request)
        .await
        .expect("the router answers");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    serde_json::from_slice(&bytes).expect("the answer is JSON")
}

/// The reported bug, in miniature: the trace goes to its own block and the
/// closing tag reaches nobody as text.
#[tokio::test]
async fn a_prefilled_opener_puts_the_trace_in_a_thinking_block() {
    let state = state(vec![10, CLOSE_ID, 11], true);
    let body = answer(&state).await;

    assert_eq!(body["content"][0]["type"], "thinking", "{body}");
    // The word-level fake vocabulary joins its tokens with a space, so the
    // space either side of the tag is the decoder's and not the parser's. The
    // one after it starts the answer's prose, and leading whitespace there is
    // framing a chat turn does not carry.
    assert_eq!(body["content"][0]["thinking"], "w10 ");
    assert_eq!(body["content"][1]["type"], "text", "{body}");
    assert_eq!(body["content"][1]["text"], "w11");
    assert!(
        !body.to_string().contains("</think>"),
        "the closing tag is structure, not text: {body}"
    );
}

/// The other half of the same template: with thinking off it closes the trace
/// itself, and a turn that resumed one would swallow the answer.
#[tokio::test]
async fn an_already_closed_trace_leaves_the_answer_as_text() {
    let state = state(vec![10, 11], false);
    let body = answer(&state).await;

    assert_eq!(body["content"][0]["type"], "text", "{body}");
    assert_eq!(body["content"][0]["text"], "w10 w11");
    assert_eq!(
        body["content"].as_array().map(Vec::len),
        Some(1),
        "nothing was taken for a trace: {body}"
    );
}
