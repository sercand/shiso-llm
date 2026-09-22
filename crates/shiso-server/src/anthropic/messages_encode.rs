//! The Messages answer, built from the neutral response.
//!
//! # The content array is the block list
//!
//! A generated turn is an ordered list of blocks and so is this format's
//! `content`, so encoding one is a walk over the other. Nothing is grouped,
//! nothing is reordered, and a turn that thought, spoke, called a tool and
//! spoke again comes out in that order - which is the order it was produced in.
//! The shape this replaced had to sort the blocks into three buckets and emit
//! them bucket by bucket, because the response it was built from had three
//! parallel fields rather than one list.
//!
//! # Why the stop sequence needs no field of its own
//!
//! The neutral stop reason carries the sequence that matched, so
//! `stop_sequence` is filled in from the same value that decides `stop_reason`
//! and the two cannot disagree. A response that ended some other way has no
//! sequence to report, and there is no way to spell one.

use serde_json::{Map, Value, json};
use shiso_ir::{Block, Message, Response, StopReason, Usage};

/// What this surface needs to know beyond the turn itself.
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    /// The identifier this message is known by, prefix and all.
    pub id: String,
    /// The model name to echo.
    pub model: String,
}

/// The word this format uses for one neutral stop reason.
///
/// A deadline is reported as `max_tokens`, which is the closest true thing this
/// vocabulary can say: the turn was cut short by a limit the caller did not
/// choose. `end_turn` would claim the model finished, and a caller that
/// believes it stops asking for the rest of an answer that was never written.
#[must_use]
pub const fn stop_reason(stop: &StopReason) -> &'static str {
    match stop {
        StopReason::MaxTokens | StopReason::Timeout => "max_tokens",
        StopReason::StopSequence(_) => "stop_sequence",
        StopReason::ToolUse => "tool_use",
        StopReason::Refusal => "refusal",
        // A natural end, and any reason added after this was written: a turn
        // that ended is the one answer that cannot mislead a client into
        // retrying or into believing it was truncated.
        _ => "end_turn",
    }
}

/// The sequence that ended the turn, when one did.
///
/// Read from the same value as [`stop_reason`], so a non-null sequence beside a
/// reason that is not `stop_sequence` is not a state this can produce.
#[must_use]
pub fn stop_sequence(stop: &StopReason) -> Value {
    match stop {
        StopReason::StopSequence(sequence) => json!(sequence),
        _ => Value::Null,
    }
}

/// The accounting block.
///
/// `cache_creation_input_tokens` is present and zero: this server writes no
/// cache entries a caller is billed for, and a client that sums the two cache
/// counters expects both keys to exist.
#[must_use]
pub fn encode_usage(usage: &Usage) -> Value {
    json!({
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cache_read_input_tokens": usage.cached_prompt_tokens,
        "cache_creation_input_tokens": 0,
    })
}

/// One generated turn's blocks, in the order they were produced.
///
/// Blocks that only ever appear in a caller's turn - a tool result, a picture -
/// are skipped rather than encoded: nothing generates them, and this format has
/// no assistant-side spelling for them.
#[must_use]
pub fn encode_content(message: &Message) -> Value {
    let mut blocks = Vec::new();
    for block in &message.content {
        match block {
            // A refusal joins prose here: it is text the caller is meant to
            // read, this format's only carrier of prose is a text block, and
            // what makes it a refusal rather than an answer is `stop_reason`,
            // which says so.
            Block::Text(text) | Block::Refusal(text) => {
                blocks.push(json!({"type": "text", "text": text}));
            }
            Block::Thinking(trace) => {
                let mut out = Map::new();
                out.insert("type".to_owned(), json!("thinking"));
                out.insert("thinking".to_owned(), json!(trace.text));
                if let Some(signature) = trace.signature.as_ref() {
                    out.insert("signature".to_owned(), json!(signature));
                }
                blocks.push(Value::Object(out));
            }
            Block::ToolUse(call) => blocks.push(json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                // Already parsed, and sent as JSON rather than as a string
                // holding JSON: this format carries the arguments as a value,
                // which is why nothing on this route ever stringifies them.
                "input": call.input,
            })),
            ref other => tracing::debug!(?other, "no assistant-side block for this block"),
        }
    }
    Value::Array(blocks)
}

/// The whole blocking answer.
#[must_use]
pub fn encode_response(response: &Response, options: &EncodeOptions) -> Value {
    json!({
        "id": options.id,
        "type": "message",
        "role": "assistant",
        "model": options.model,
        "content": encode_content(&response.message),
        "stop_reason": stop_reason(&response.stop),
        "stop_sequence": stop_sequence(&response.stop),
        "usage": encode_usage(&response.usage),
    })
}

#[cfg(test)]
mod tests {
    use super::{EncodeOptions, encode_content, encode_response, stop_reason};
    use serde_json::json;
    use shiso_ir::{Block, Message, Response, Role, StopReason, Thinking, ToolUse, Usage};

    fn options() -> EncodeOptions {
        EncodeOptions {
            id: "msg_1".to_owned(),
            model: "a-model".to_owned(),
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
                prompt_tokens: 7,
                completion_tokens: 3,
                cached_prompt_tokens: 2,
                ..Usage::default()
            },
            logprobs: None,
            token_ids: Vec::new(),
        }
    }

    #[test]
    fn an_answer_has_every_key_a_client_of_this_format_reads() {
        let body = encode_response(
            &response(Message::assistant("Paris"), StopReason::EndTurn),
            &options(),
        );
        assert_eq!(body["id"], "msg_1");
        assert_eq!(body["type"], "message");
        assert_eq!(body["role"], "assistant");
        assert_eq!(body["model"], "a-model");
        assert_eq!(body["content"][0]["type"], "text");
        assert_eq!(body["content"][0]["text"], "Paris");
        assert_eq!(body["stop_reason"], "end_turn");
        assert_eq!(body["stop_sequence"], json!(null));
        assert_eq!(body["usage"]["input_tokens"], 7);
        assert_eq!(body["usage"]["output_tokens"], 3);
        assert_eq!(body["usage"]["cache_read_input_tokens"], 2);
        assert_eq!(body["usage"]["cache_creation_input_tokens"], 0);
    }

    /// The interleaving is the point: a turn that thought, spoke, called a tool
    /// and spoke again comes out in that order, because the content array is
    /// the block list rather than three buckets emitted in a fixed order.
    #[test]
    fn the_content_array_is_the_block_list_in_the_order_it_was_produced() {
        let turn = Message::new(
            Role::Assistant,
            vec![
                Block::Text("first".to_owned()),
                Block::ToolUse(ToolUse::new("call_1", "lookup", json!({"city": "Paris"}))),
                Block::Thinking(Thinking::new("hmm")),
                Block::Text("second".to_owned()),
            ],
        );
        let content = encode_content(&turn);
        let kinds: Vec<&str> = content
            .as_array()
            .expect("content is an array")
            .iter()
            .filter_map(|block| block["type"].as_str())
            .collect();
        assert_eq!(kinds, vec!["text", "tool_use", "thinking", "text"]);
        assert_eq!(content[1]["id"], "call_1");
        assert_eq!(content[1]["name"], "lookup");
        assert_eq!(
            content[1]["input"]["city"], "Paris",
            "the arguments travel as JSON, not as a string holding JSON"
        );
    }

    #[test]
    fn a_thinking_block_carries_its_signature_only_when_it_has_one() {
        let mut trace = Thinking::new("step");
        trace.signature = Some("sig".to_owned());
        let signed = Message::new(Role::Assistant, vec![Block::Thinking(trace)]);
        assert_eq!(encode_content(&signed)[0]["signature"], "sig");

        let unsigned = Message::new(
            Role::Assistant,
            vec![Block::Thinking(Thinking::new("step"))],
        );
        assert!(encode_content(&unsigned)[0].get("signature").is_none());
    }

    /// The sequence that matched travels inside the neutral reason, so the two
    /// fields are read from one value and cannot contradict each other.
    #[test]
    fn a_stop_sequence_is_echoed_by_the_one_reason_that_can_have_one() {
        let body = encode_response(
            &response(
                Message::assistant("up to here"),
                StopReason::StopSequence("END".to_owned()),
            ),
            &options(),
        );
        assert_eq!(body["stop_reason"], "stop_sequence");
        assert_eq!(body["stop_sequence"], "END");
    }

    /// A deadline says `max_tokens` because this vocabulary has no word for a
    /// deadline, and `end_turn` would claim the model finished.
    #[test]
    fn every_neutral_reason_has_a_word_in_this_vocabulary() {
        assert_eq!(stop_reason(&StopReason::EndTurn), "end_turn");
        assert_eq!(stop_reason(&StopReason::MaxTokens), "max_tokens");
        assert_eq!(stop_reason(&StopReason::Timeout), "max_tokens");
        assert_eq!(stop_reason(&StopReason::ToolUse), "tool_use");
        assert_eq!(stop_reason(&StopReason::Refusal), "refusal");
        assert_eq!(
            stop_reason(&StopReason::StopSequence("END".to_owned())),
            "stop_sequence"
        );
    }

    #[test]
    fn a_turn_with_nothing_in_it_answers_with_an_empty_content_array() {
        let body = encode_response(
            &response(
                Message::new(Role::Assistant, Vec::new()),
                StopReason::EndTurn,
            ),
            &options(),
        );
        assert_eq!(body["content"], json!([]));
    }
}
