//! The chat-completions answer, built from the neutral response.
//!
//! # Why the identifier is minted by the handler and not taken from the turn
//!
//! The neutral response carries a bare identifier, deliberately, because every
//! surface spells one differently and a value that already carried one prefix
//! would have to be un-prefixed by the next. So each surface stamps its own, and
//! the one it stamps is the one it stores and the one it streams - a client that
//! saw an identifier on a stream can fetch that identifier back.
//!
//! # Why the stop reason is mapped here
//!
//! The neutral reason has six cases and this format has four words. The mapping
//! is this surface's opinion and lives with this surface: the block-structured
//! family spells the same six differently, and a shared mapping would force one
//! of the two to adopt the other's vocabulary.

use serde_json::{Map, Value, json};
use shiso_ir::{Block, Message, Response, StopReason, Usage};

use crate::openai::SYSTEM_FINGERPRINT;
use crate::openai::chat_wire::ServiceTier;

/// What a surface needs to know beyond the turn itself.
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    /// The identifier this response is known by.
    pub id: String,
    /// The model name to echo.
    pub model: String,
    /// The service class the caller asked for, echoed when they named one.
    pub service_tier: Option<ServiceTier>,
    /// Whether the sampled ids ride alongside the text.
    pub return_token_ids: bool,
}

/// The word this format uses for one neutral stop reason.
///
/// `timeout` is not one of the four words the format defines, and it is used
/// anyway: a caller that cannot tell a deadline from an exhausted budget loses
/// output and has no way to learn that it did. Spelling it `length` would be a
/// compatible lie.
#[must_use]
pub const fn finish_reason(stop: &StopReason) -> &'static str {
    match stop {
        StopReason::MaxTokens => "length",
        StopReason::ToolUse => "tool_calls",
        StopReason::Refusal => "content_filter",
        StopReason::Timeout => "timeout",
        // A natural end and a stop sequence are both `stop` here, because this
        // format has no word for the difference and the sequence that matched
        // travels in the neutral reason for a surface that does. The wildcard
        // covers them and anything added later: `stop` is the one answer that
        // cannot mislead a client into retrying or into believing it was
        // truncated.
        _ => "stop",
    }
}

/// The accounting block.
///
/// `total_tokens` is derived rather than carried, because a total that disagreed
/// with its two parts is a number a client would have to decide which half to
/// believe.
#[must_use]
pub fn encode_usage(usage: &Usage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens.saturating_add(usage.completion_tokens),
        "prompt_tokens_details": {
            "cached_tokens": usage.cached_prompt_tokens,
            "audio_tokens": usage.audio_prompt_tokens,
        },
        "completion_tokens_details": {
            "reasoning_tokens": usage.reasoning_tokens,
            // Nothing this server generates is audio, and the field is present
            // rather than absent because a client that sums the details
            // expects it to be.
            "audio_tokens": 0,
            "accepted_prediction_tokens": usage.accepted_prediction_tokens,
            "rejected_prediction_tokens": 0,
        },
        // Not part of the format. A client that reads only the standard keys
        // ignores it; `shiso-registry bench` reads it, because timing the
        // response from outside would fold the HTTP round trip and the
        // rendering of the prompt into the prefill rate.
        "timings": {
            "time_to_first_token_ms": usage.time_to_first_token_ms,
            "decode_tokens_per_second": usage.tokens_per_second,
        },
    })
}

/// One generated turn, in the shape this format's `message` key wants.
///
/// The ordered block list is flattened into the parallel fields this format
/// has: prose joins into `content`, traces into `reasoning_content`, calls into
/// `tool_calls`. That flattening loses the interleaving - which is exactly why
/// the neutral shape does not do it - and it is the only shape this format can
/// express.
#[must_use]
pub fn encode_message(message: &Message) -> Value {
    let mut out = Map::new();
    out.insert("role".to_owned(), json!("assistant"));

    let text = message.text();
    let trace: String = message
        .content
        .iter()
        .filter_map(|block| match block {
            Block::Thinking(thinking) => Some(thinking.text.as_str()),
            _ => None,
        })
        .collect();
    let refusal: String = message
        .content
        .iter()
        .filter_map(|block| match block {
            Block::Refusal(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let calls: Vec<Value> = message
        .tool_uses()
        .enumerate()
        .map(|(at, call)| {
            json!({
                "index": at,
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    // Back to text, because that is what this format carries and
                    // what every client parses. The neutral shape keeps it
                    // parsed so that exactly one parse exists.
                    "arguments": super::arguments_text(&call.input),
                },
            })
        })
        .collect();

    // `null` rather than an empty string when a turn is only tool calls: an
    // empty string is content the model produced, and a client that appends it
    // to a transcript would add a blank turn.
    out.insert(
        "content".to_owned(),
        if text.is_empty() && !calls.is_empty() {
            Value::Null
        } else {
            json!(text)
        },
    );
    if !trace.is_empty() {
        out.insert("reasoning_content".to_owned(), json!(trace));
    }
    if !refusal.is_empty() {
        out.insert("refusal".to_owned(), json!(refusal));
    }
    if !calls.is_empty() {
        out.insert("tool_calls".to_owned(), Value::Array(calls));
    }
    Value::Object(out)
}

/// The whole blocking answer.
#[must_use]
pub fn encode_response(response: &Response, options: &EncodeOptions) -> Value {
    let mut choice = Map::new();
    choice.insert("index".to_owned(), json!(0));
    choice.insert("message".to_owned(), encode_message(&response.message));
    choice.insert(
        "finish_reason".to_owned(),
        json!(finish_reason(&response.stop)),
    );
    // Always present and always null: this server emits no per-token
    // probabilities, and a missing key makes a client that reads it raise a
    // different error than the one that says so.
    choice.insert("logprobs".to_owned(), Value::Null);

    let mut out = Map::new();
    out.insert("id".to_owned(), json!(options.id));
    out.insert("object".to_owned(), json!("chat.completion"));
    out.insert("created".to_owned(), json!(response.created));
    out.insert("model".to_owned(), json!(options.model));
    out.insert("system_fingerprint".to_owned(), json!(SYSTEM_FINGERPRINT));
    out.insert(
        "choices".to_owned(),
        Value::Array(vec![Value::Object(choice)]),
    );
    out.insert("usage".to_owned(), encode_usage(&response.usage));
    if let Some(tier) = options.service_tier {
        out.insert("service_tier".to_owned(), json!(tier.as_str()));
    }
    if options.return_token_ids {
        out.insert("token_ids".to_owned(), json!(response.token_ids));
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::{EncodeOptions, encode_message, encode_response, encode_usage, finish_reason};
    use serde_json::json;
    use shiso_ir::{Block, Message, Response, Role, StopReason, Thinking, ToolUse, Usage};

    fn options() -> EncodeOptions {
        EncodeOptions {
            id: "chatcmpl-1".to_owned(),
            model: "a-model".to_owned(),
            service_tier: None,
            return_token_ids: false,
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
                reasoning_tokens: 1,
                ..Usage::default()
            },
            logprobs: None,
            token_ids: vec![1, 2, 3],
        }
    }

    #[test]
    fn a_blocking_answer_has_every_key_a_client_of_this_format_reads() {
        let body = encode_response(
            &response(Message::assistant("Paris"), StopReason::EndTurn),
            &options(),
        );
        assert_eq!(body["id"], "chatcmpl-1");
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["created"], 1700);
        assert_eq!(body["model"], "a-model");
        assert_eq!(body["system_fingerprint"], "fp_shiso");
        assert_eq!(body["choices"][0]["index"], 0);
        assert_eq!(body["choices"][0]["message"]["content"], "Paris");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["choices"][0]["logprobs"], json!(null));
        assert_eq!(body["usage"]["prompt_tokens"], 7);
        assert_eq!(body["usage"]["total_tokens"], 10);
        assert_eq!(
            body["usage"]["completion_tokens_details"]["reasoning_tokens"],
            1
        );
        assert_eq!(body["usage"]["prompt_tokens_details"]["audio_tokens"], 0);
        assert!(
            body.get("token_ids").is_none(),
            "the ids ride only when they were asked for"
        );
    }

    #[test]
    fn the_audio_share_of_the_prompt_is_its_pad_run() {
        let usage = encode_usage(&Usage {
            prompt_tokens: 774,
            audio_prompt_tokens: 750,
            ..Usage::default()
        });
        assert_eq!(usage["prompt_tokens_details"]["audio_tokens"], 750);
        // Nothing this server generates is audio.
        assert_eq!(usage["completion_tokens_details"]["audio_tokens"], 0);
    }

    #[test]
    fn the_turn_timings_ride_beside_the_standard_keys() {
        let usage = encode_usage(&Usage {
            time_to_first_token_ms: 812.5,
            tokens_per_second: 41.25,
            ..Usage::default()
        });
        assert_eq!(usage["timings"]["time_to_first_token_ms"], 812.5);
        assert_eq!(usage["timings"]["decode_tokens_per_second"], 41.25);
    }

    /// One request, one turn: the list exists because the format demands one,
    /// and it is never longer than one.
    #[test]
    fn the_answer_holds_exactly_one_choice() {
        let body = encode_response(
            &response(Message::assistant("hi"), StopReason::EndTurn),
            &options(),
        );
        assert_eq!(body["choices"].as_array().expect("an array").len(), 1);
    }

    /// A stop sequence and a natural end are the same word on this surface, and
    /// a deadline is not a budget.
    #[test]
    fn every_neutral_reason_lands_on_a_word_this_format_defines_or_says_so() {
        assert_eq!(finish_reason(&StopReason::EndTurn), "stop");
        assert_eq!(
            finish_reason(&StopReason::StopSequence("END".to_owned())),
            "stop"
        );
        assert_eq!(finish_reason(&StopReason::MaxTokens), "length");
        assert_eq!(finish_reason(&StopReason::ToolUse), "tool_calls");
        assert_eq!(finish_reason(&StopReason::Refusal), "content_filter");
        assert_eq!(finish_reason(&StopReason::Timeout), "timeout");
    }

    /// The block-structured family's vocabulary must not leak onto this surface.
    #[test]
    fn no_neutral_variant_reaches_the_wire_under_its_own_name() {
        for stop in [
            StopReason::EndTurn,
            StopReason::MaxTokens,
            StopReason::StopSequence("x".to_owned()),
            StopReason::ToolUse,
            StopReason::Refusal,
            StopReason::Timeout,
        ] {
            let word = finish_reason(&stop);
            assert!(
                !matches!(
                    word,
                    "end_turn" | "max_tokens" | "stop_sequence" | "refusal"
                ),
                "{word} is the other family's spelling"
            );
        }
    }

    #[test]
    fn a_turn_of_only_tool_calls_has_no_content_rather_than_empty_content() {
        let message = Message::new(
            Role::Assistant,
            vec![Block::ToolUse(ToolUse::new(
                "call_1",
                "lookup",
                json!({"city": "Paris"}),
            ))],
        );
        let encoded = encode_message(&message);
        assert_eq!(encoded["content"], json!(null));
        assert_eq!(encoded["tool_calls"][0]["id"], "call_1");
        assert_eq!(encoded["tool_calls"][0]["type"], "function");
        assert_eq!(encoded["tool_calls"][0]["function"]["name"], "lookup");
        // Back to text, because that is what this format carries.
        assert_eq!(
            encoded["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
    }

    #[test]
    fn a_trace_travels_in_its_own_key_and_never_inside_the_content() {
        let message = Message::new(
            Role::Assistant,
            vec![
                Block::Thinking(Thinking::new("weighing it up")),
                Block::Text("Paris".to_owned()),
            ],
        );
        let encoded = encode_message(&message);
        assert_eq!(encoded["content"], "Paris");
        assert_eq!(encoded["reasoning_content"], "weighing it up");
    }

    #[test]
    fn a_withheld_answer_carries_the_refusal_the_model_showed_instead() {
        let message = Message::new(
            Role::Assistant,
            vec![Block::Refusal("I cannot help with that".to_owned())],
        );
        assert_eq!(
            encode_message(&message)["refusal"],
            "I cannot help with that"
        );
    }

    #[test]
    fn the_total_is_derived_so_it_cannot_disagree_with_its_parts() {
        let usage = encode_usage(&Usage {
            prompt_tokens: 11,
            completion_tokens: 22,
            ..Usage::default()
        });
        assert_eq!(usage["total_tokens"], 33);
    }

    #[test]
    fn the_service_class_is_echoed_only_when_the_caller_named_one() {
        let mut options = options();
        assert!(
            encode_response(
                &response(Message::assistant("hi"), StopReason::EndTurn),
                &options
            )
            .get("service_tier")
            .is_none()
        );
        options.service_tier = Some(crate::openai::chat_wire::ServiceTier::Flex);
        assert_eq!(
            encode_response(
                &response(Message::assistant("hi"), StopReason::EndTurn),
                &options
            )["service_tier"],
            "flex"
        );
    }

    #[test]
    fn the_sampled_ids_ride_along_when_the_caller_asked_for_them() {
        let mut options = options();
        options.return_token_ids = true;
        let body = encode_response(
            &response(Message::assistant("hi"), StopReason::EndTurn),
            &options,
        );
        assert_eq!(body["token_ids"], json!([1, 2, 3]));
    }
}
