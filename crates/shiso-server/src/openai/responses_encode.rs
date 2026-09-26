//! The Responses answer, built from the neutral response.
//!
//! # Why the output is a list of items rather than one message
//!
//! This format models a turn as a sequence of *items* - a reasoning item, a
//! message, a function call - which is very nearly the neutral block list, and
//! is the reason the neutral shape was modelled after this family rather than
//! after chat completions. So the mapping is one item per block, in block order,
//! with no folding at all: an item's position in the output list is its block's
//! position in the turn, which is what lets the streaming encoder announce an
//! item under the same identifier and the same index that the finished body
//! gives it.
//!
//! # Why the status is derived from the stop reason
//!
//! `completed` and `incomplete` are the same turn with a different ending, and
//! this format puts the difference in a status plus a reason object rather than
//! in one word. Deriving both from the neutral reason is what keeps them
//! agreeing with each other.

use serde_json::{Map, Value, json};
use shiso_ir::{Block, Message, Response, StopReason, Usage};

use crate::openai::chat_wire::ServiceTier;

/// What a surface needs beyond the turn itself.
#[derive(Debug, Clone, Default)]
pub struct ResponseOptions {
    /// The identifier this response is known by.
    pub id: String,
    /// The model name to echo.
    pub model: String,
    /// The instructions the caller sent, echoed back.
    pub instructions: Option<String>,
    /// The response this one continues, echoed back.
    pub previous_response_id: Option<String>,
    /// The caller's opaque key-value pairs, echoed back untouched.
    pub metadata: Option<Value>,
    /// The token cap the caller named, echoed back.
    pub max_output_tokens: Option<u32>,
    /// The service class the caller asked for.
    pub service_tier: Option<ServiceTier>,
}

/// The status word for one neutral stop reason.
#[must_use]
pub const fn status_for(stop: &StopReason) -> &'static str {
    match stop {
        StopReason::MaxTokens | StopReason::Timeout => "incomplete",
        _ => "completed",
    }
}

/// Why an incomplete turn is incomplete, or `null` when it is not.
#[must_use]
pub fn incomplete_details(stop: &StopReason) -> Value {
    match stop {
        StopReason::MaxTokens => json!({"reason": "max_output_tokens"}),
        // Not one of the reasons this format defines, and said anyway: a caller
        // that cannot tell a deadline from an exhausted budget loses output and
        // has no way to learn that it did.
        StopReason::Timeout => json!({"reason": "timeout"}),
        _ => Value::Null,
    }
}

/// The accounting block, in this surface's own vocabulary.
#[must_use]
pub fn encode_usage(usage: &Usage) -> Value {
    json!({
        "input_tokens": usage.prompt_tokens,
        "input_tokens_details": {
            "cached_tokens": usage.cached_prompt_tokens,
            "audio_tokens": usage.audio_prompt_tokens,
        },
        "output_tokens": usage.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
        "total_tokens": usage.prompt_tokens.saturating_add(usage.completion_tokens),
    })
}

/// One generated turn as this format's output items.
///
/// `response_id` seeds the per-item identifiers, so that an item in a stream and
/// the same item in the stored body are the same object under the same name.
#[must_use]
pub fn encode_output(message: &Message, response_id: &str) -> Vec<Value> {
    let mut items = Vec::new();
    for (at, block) in message.content.iter().enumerate() {
        let id = |kind: &str| item_id(kind, response_id, at);
        let item = match block {
            Block::Text(text) => json!({
                "type": "message",
                "id": id("msg"),
                "status": "completed",
                "role": "assistant",
                // Present and empty annotations: a client that iterates them
                // should not have to check whether the key exists first.
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }),
            Block::Refusal(text) => json!({
                "type": "message",
                "id": id("msg"),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": text}],
            }),
            Block::Thinking(trace) => json!({
                "type": "reasoning",
                "id": id("rs"),
                "summary": [{"type": "summary_text", "text": trace.text}],
                "content": [],
            }),
            Block::ToolUse(call) => json!({
                "type": "function_call",
                "id": id("fc"),
                "call_id": call.id,
                "name": call.name,
                // Back to text, because that is what this format carries.
                "arguments": super::arguments_text(&call.input),
                "status": "completed",
            }),
            // Media and tool results are things a caller sends, not things a
            // model produces, so a generated turn never carries one. Skipping
            // them would shift every later item's index, so the position is
            // kept and the item is an empty message.
            _ => json!({
                "type": "message",
                "id": id("msg"),
                "status": "completed",
                "role": "assistant",
                "content": [],
            }),
        };
        items.push(item);
    }
    items
}

/// A stable per-item identifier.
///
/// Derived from the response's own identifier and the block's position, so the
/// item a stream announced and the item in the stored body are the same name.
#[must_use]
pub fn item_id(kind: &str, response_id: &str, at: usize) -> String {
    // The response identifier already carries the `resp_` prefix; the item's
    // own prefix replaces it so that the two are never confused for each other.
    let bare = response_id.strip_prefix("resp_").unwrap_or(response_id);
    format!("{kind}_{bare}_{at}")
}

/// The whole blocking answer.
#[must_use]
pub fn encode_response(response: &Response, options: &ResponseOptions) -> Value {
    encode_envelope(
        response.created,
        options,
        encode_output(&response.message, &options.id),
        status_for(&response.stop),
        incomplete_details(&response.stop),
        Some(encode_usage(&response.usage)),
    )
}

/// The envelope, with the output and the status supplied.
///
/// The streaming encoder needs one before any output exists - that is what
/// `response.created` carries - and it has to be the same envelope, or a client
/// that diffs the first frame against the last sees fields appear and disappear
/// for no reason. So the envelope takes its parts rather than a finished turn.
#[must_use]
pub fn encode_envelope(
    created: u64,
    options: &ResponseOptions,
    output: Vec<Value>,
    status: &str,
    incomplete: Value,
    usage: Option<Value>,
) -> Value {
    let mut out = Map::new();
    out.insert("id".to_owned(), json!(options.id));
    out.insert("object".to_owned(), json!("response"));
    out.insert("created_at".to_owned(), json!(created));
    out.insert("status".to_owned(), json!(status));
    out.insert("model".to_owned(), json!(options.model));
    out.insert("output".to_owned(), Value::Array(output));
    out.insert("error".to_owned(), Value::Null);
    out.insert("incomplete_details".to_owned(), incomplete);
    out.insert("instructions".to_owned(), json!(options.instructions));
    out.insert(
        "max_output_tokens".to_owned(),
        json!(options.max_output_tokens),
    );
    out.insert(
        "previous_response_id".to_owned(),
        json!(options.previous_response_id),
    );
    // This server answers one call at a time, and saying so is better than
    // omitting the field a client reads to decide how to batch results.
    out.insert("parallel_tool_calls".to_owned(), json!(false));
    out.insert(
        "metadata".to_owned(),
        options.metadata.clone().unwrap_or_else(|| json!({})),
    );
    out.insert("usage".to_owned(), usage.unwrap_or(Value::Null));
    if let Some(tier) = options.service_tier {
        out.insert("service_tier".to_owned(), json!(tier.as_str()));
    }
    Value::Object(out)
}

/// The envelope for a turn that failed part way.
#[must_use]
pub fn encode_failure(created: u64, options: &ResponseOptions, message: &str) -> Value {
    let mut body = encode_envelope(created, options, Vec::new(), "failed", Value::Null, None);
    if let Some(fields) = body.as_object_mut() {
        fields.insert(
            "error".to_owned(),
            json!({"code": "server_error", "message": message}),
        );
    }
    body
}

#[cfg(test)]
mod tests {
    use super::{ResponseOptions, encode_output, encode_response, encode_usage, status_for};
    use serde_json::json;
    use shiso_ir::{Block, Message, Response, Role, StopReason, Thinking, ToolUse, Usage};

    fn options() -> ResponseOptions {
        ResponseOptions {
            id: "resp_abc".to_owned(),
            model: "a-model".to_owned(),
            ..ResponseOptions::default()
        }
    }

    fn response(message: Message, stop: StopReason) -> Response {
        Response {
            id: "bare".to_owned(),
            model: "a-model".to_owned(),
            created: 1700,
            message,
            stop,
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 6,
                reasoning_tokens: 2,
                ..Usage::default()
            },
            logprobs: None,
            token_ids: Vec::new(),
        }
    }

    #[test]
    fn a_text_turn_is_one_message_item_with_one_output_text_part() {
        let body = encode_response(
            &response(Message::assistant("Paris"), StopReason::EndTurn),
            &options(),
        );
        assert_eq!(body["object"], "response");
        assert_eq!(body["id"], "resp_abc");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["incomplete_details"], json!(null));
        let output = body["output"].as_array().expect("an output list");
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "message");
        assert_eq!(output[0]["role"], "assistant");
        assert_eq!(output[0]["content"][0]["type"], "output_text");
        assert_eq!(output[0]["content"][0]["text"], "Paris");
        assert_eq!(output[0]["content"][0]["annotations"], json!([]));
    }

    /// The block order is the item order: a trace, then prose, then a call is
    /// three items in that order and not a message with two things attached.
    #[test]
    fn the_block_order_of_a_turn_is_the_item_order_of_the_output() {
        let message = Message::new(
            Role::Assistant,
            vec![
                Block::Thinking(Thinking::new("weighing it up")),
                Block::Text("checking".to_owned()),
                Block::ToolUse(ToolUse::new("call_1", "lookup", json!({"city": "Paris"}))),
            ],
        );
        let output = encode_output(&message, "resp_abc");
        let kinds: Vec<&str> = output
            .iter()
            .filter_map(|item| item["type"].as_str())
            .collect();
        assert_eq!(kinds, vec!["reasoning", "message", "function_call"]);
        assert_eq!(output[0]["summary"][0]["text"], "weighing it up");
        assert_eq!(output[2]["call_id"], "call_1");
        assert_eq!(output[2]["arguments"], "{\"city\":\"Paris\"}");
    }

    /// Every item needs an identifier a stream and a stored body agree on.
    #[test]
    fn every_item_carries_an_identifier_derived_from_the_responses_own() {
        let message = Message::new(
            Role::Assistant,
            vec![
                Block::Text("hi".to_owned()),
                Block::ToolUse(ToolUse::new("c", "f", json!({}))),
            ],
        );
        let output = encode_output(&message, "resp_abc");
        assert_eq!(output[0]["id"], "msg_abc_0");
        assert_eq!(output[1]["id"], "fc_abc_1");
        assert_eq!(output.len(), 2, "one item per block, in block order");
    }

    /// A truncated turn is the same turn with a different ending, and the two
    /// fields that say so are derived together so they cannot disagree.
    #[test]
    fn a_truncated_turn_is_incomplete_and_says_which_limit_stopped_it() {
        let body = encode_response(
            &response(Message::assistant("Par"), StopReason::MaxTokens),
            &options(),
        );
        assert_eq!(body["status"], "incomplete");
        assert_eq!(body["incomplete_details"]["reason"], "max_output_tokens");
    }

    #[test]
    fn every_neutral_reason_lands_on_a_status_this_format_defines() {
        assert_eq!(status_for(&StopReason::EndTurn), "completed");
        assert_eq!(status_for(&StopReason::ToolUse), "completed");
        assert_eq!(
            status_for(&StopReason::StopSequence("x".to_owned())),
            "completed"
        );
        assert_eq!(status_for(&StopReason::MaxTokens), "incomplete");
        assert_eq!(status_for(&StopReason::Timeout), "incomplete");
    }

    #[test]
    fn the_accounting_uses_this_surfaces_own_vocabulary() {
        let usage = encode_usage(&Usage {
            prompt_tokens: 5,
            completion_tokens: 6,
            reasoning_tokens: 2,
            ..Usage::default()
        });
        assert_eq!(usage["input_tokens"], 5);
        assert_eq!(usage["output_tokens"], 6);
        assert_eq!(usage["total_tokens"], 11);
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 2);
    }

    #[test]
    fn the_audio_share_of_the_prompt_is_reported_beside_the_cached_share() {
        let usage = encode_usage(&Usage {
            prompt_tokens: 774,
            audio_prompt_tokens: 750,
            cached_prompt_tokens: 3,
            ..Usage::default()
        });
        assert_eq!(usage["input_tokens_details"]["audio_tokens"], 750);
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 3);
    }

    #[test]
    fn what_the_caller_sent_is_echoed_back_untouched() {
        let mut options = options();
        options.instructions = Some("be brief".to_owned());
        options.previous_response_id = Some("resp_earlier".to_owned());
        options.metadata = Some(json!({"trace": "1"}));
        options.max_output_tokens = Some(64);
        let body = encode_response(
            &response(Message::assistant("hi"), StopReason::EndTurn),
            &options,
        );
        assert_eq!(body["instructions"], "be brief");
        assert_eq!(body["previous_response_id"], "resp_earlier");
        assert_eq!(body["metadata"]["trace"], "1");
        assert_eq!(body["max_output_tokens"], 64);
    }
}
