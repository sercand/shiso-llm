//! A folded qwen4exp-shaped turn through every wire encoder.
//!
//! The slot-to-turn half of these cases is `shiso-runtime`'s
//! `a_thinking_turn_with_calls_encodes_trimmed_prose_and_counts_its_trace` and
//! `a_turn_of_only_calls_has_null_content`; the encoders live here, so this half
//! feeds them the response those tests assert the slot produces.

use serde_json::{Value, json};
use shiso_ir::{Block, Message, Response, Role, StopReason, Thinking, ToolUse, Usage};
use shiso_server::anthropic::messages_encode;
use shiso_server::openai::{chat_encode, responses_encode};

fn calls(n: usize, content: Vec<Block>, reasoning_tokens: u64) -> Response {
    let mut blocks = content;
    for i in 0..n {
        blocks.push(Block::ToolUse(ToolUse::new(
            format!("call_{i}"),
            "read_file",
            json!({ "path": "a" }),
        )));
    }
    Response {
        id: "turn".to_owned(),
        model: "m".to_owned(),
        created: 0,
        message: Message::new(Role::Assistant, blocks),
        stop: StopReason::ToolUse,
        usage: Usage {
            reasoning_tokens,
            ..Usage::default()
        },
        logprobs: None,
        token_ids: Vec::new(),
    }
}

/// The framing newlines the template puts around the calls are not content on
/// any surface, and the trace's tokens are counted as reasoning.
#[test]
fn a_thinking_turn_with_calls_encodes_trimmed_prose_and_counts_its_trace() {
    let response = calls(
        2,
        vec![
            Block::Thinking(Thinking::new("plan one plan two\n")),
            Block::Text("I'll read both.".to_owned()),
        ],
        3,
    );

    let chat = chat_encode::encode_response(&response, &chat_encode::EncodeOptions::default());
    let message = &chat["choices"][0]["message"];
    assert_eq!(message["content"], "I'll read both.");
    assert_eq!(message["reasoning_content"], "plan one plan two\n");
    assert_eq!(message["tool_calls"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        chat["usage"]["completion_tokens_details"]["reasoning_tokens"],
        3
    );

    let responses =
        responses_encode::encode_response(&response, &responses_encode::ResponseOptions::default());
    assert_eq!(
        responses["usage"]["output_tokens_details"]["reasoning_tokens"],
        3
    );

    let anthropic =
        messages_encode::encode_response(&response, &messages_encode::EncodeOptions::default());
    let texts: Vec<&str> = anthropic["content"]
        .as_array()
        .expect("content is a list")
        .iter()
        .filter(|block| block["type"] == "text")
        .filter_map(|block| block["text"].as_str())
        .collect();
    assert_eq!(texts, vec!["I'll read both."]);
}

/// A turn that is only calls has no prose at all, so its content is `null`
/// rather than the newlines between the calls.
#[test]
fn a_turn_of_only_calls_has_null_content() {
    // The same shape the slot folds to in `shiso-runtime`: a one-line trace
    // (two reasoning tokens) and three calls, no prose block.
    let response = calls(3, vec![Block::Thinking(Thinking::new("plan\n"))], 2);
    let chat = chat_encode::encode_response(&response, &chat_encode::EncodeOptions::default());
    let message = &chat["choices"][0]["message"];
    assert_eq!(message["content"], Value::Null);
    assert_eq!(message["reasoning_content"], "plan\n");
    assert_eq!(
        chat["usage"]["completion_tokens_details"]["reasoning_tokens"],
        2
    );
    assert_eq!(message["tool_calls"].as_array().map(Vec::len), Some(3));
}
