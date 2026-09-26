//! The chat-completions body, turned into the neutral request.
//!
//! # The three places this format and the IR disagree
//!
//! **A system turn is not a turn.** Instructions are a field on the neutral
//! request because every template renders them in a place of their own, so the
//! system and developer turns are lifted out of the conversation in the order
//! they appeared. A client that puts one in the middle of a conversation gets it
//! rendered at the top, which is what every template would have done with it
//! anyway.
//!
//! **A tool result is not a speaker.** There is no tool role in the IR: a result
//! is a block, and it rides in the caller turn that carries it back. Consecutive
//! `tool` turns therefore collapse into one caller turn of results, which is
//! exactly the grouping the renderer splits back out.
//!
//! **A tool call is not a field beside the content.** `tool_calls` becomes
//! blocks in the turn's own block list, after whatever prose came with them, so
//! that replaying the turn into the next request keeps the order the model
//! produced.
//!
//! # Why the arguments are parsed here and kept when they do not parse
//!
//! A call's arguments arrive as JSON text and the IR carries them parsed, so the
//! parse happens once. A model that wrote something that is not JSON has still
//! said something, and dropping it would delete the only evidence of what went
//! wrong - so an unparsable argument string is kept as a string value rather
//! than refused.

use serde_json::Value;
use shiso_ir::{
    Block, Effort, MediaSource, Message, Request, Role, Sampling, ServiceClass, ThinkingDirective,
    ToolChoice, ToolDefinition, ToolResult, ToolUse,
};

use crate::api::service_tier::ServiceTier;
use crate::error::AppError;
use crate::openai::chat_wire::{
    ChatBody, Content, MediaRef, Part, WireMessage, WireRole, WireToolChoice,
};

/// Turns a validated body into the neutral request.
///
/// `model_id` is the checkpoint this process serves: the request's own `model`
/// field is a label a client chose and is echoed rather than honoured, because
/// one process serves one checkpoint and refusing a name mismatch would break
/// every client that routes by a gateway alias.
///
/// # Errors
///
/// [`AppError::BadRequest`] for a thinking level this server does not know, a
/// tool-choice mode it does not know, or a logit bias keyed on something that is
/// not a token id.
pub fn to_ir(body: &ChatBody, model_id: &str) -> Result<Request, AppError> {
    let mut request = Request::new(model_id, Vec::new());
    let (system, messages) = split_conversation(&body.messages);
    request.system = system;
    request.messages = messages;

    request.tools = body
        .tools
        .iter()
        .map(|tool| {
            // A tool with no schema takes no arguments, and an empty object is
            // how a template writes that. `null` would render as the word null
            // in the prompt.
            let schema = tool
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
            let mut definition = ToolDefinition::new(tool.function.name.clone(), schema);
            definition
                .description
                .clone_from(&tool.function.description);
            definition
        })
        .collect();
    request.tool_choice = tool_choice(body.tool_choice.as_ref())?;

    request.sampling = Sampling {
        temperature: body.temperature,
        top_k: body.top_k,
        top_p: body.top_p,
        top_n_sigma: body.top_n_sigma,
        min_p: body.min_p,
        repetition_penalty: body.repetition_penalty,
        presence_penalty: body.presence_penalty,
        frequency_penalty: body.frequency_penalty,
    };
    // Zero is the neutral request's spelling of "the server decides", because
    // the field is a resolved count rather than an option; the pipeline resolves
    // it against what is left of the context.
    request.max_tokens = body.token_cap().unwrap_or(0);
    request.min_tokens = body.min_tokens.unwrap_or(0);
    request.stop = body
        .stop
        .clone()
        .map(super::chat_wire::StopField::into_vec)
        .unwrap_or_default();
    request.stream = body.stream.unwrap_or(false);
    request.response_format = response_format(body.response_format.as_ref());
    request.thinking = thinking(body)?;
    // Either spelling reaches the template; clients that pass template
    // variables through send the knob, and ignoring it rendered every
    // historical trace the caller asked to have stripped. The body's own field
    // wins when both are sent, since it is the one this surface documents.
    request.preserve_thinking = body.preserve_thinking.or_else(|| {
        body.chat_template_kwargs
            .and_then(|kwargs| kwargs.preserve_thinking)
    });
    request.logit_bias = logit_bias(body.logit_bias.as_ref())?;
    request.top_logprobs = body
        .logprobs
        .unwrap_or(false)
        .then(|| body.top_logprobs.unwrap_or(0));
    request.seed = body.seed;
    request.return_token_ids = body.return_token_ids.unwrap_or(false);
    request.service_class = body
        .service_tier
        .map_or(ServiceClass::Standard, ServiceTier::class);
    Ok(request)
}

/// Lifts the instructions out and folds the rest into turns.
fn split_conversation(wire: &[WireMessage]) -> (Vec<Block>, Vec<Message>) {
    let mut system = Vec::new();
    let mut turns: Vec<Message> = Vec::new();
    for message in wire {
        match message.role {
            WireRole::System | WireRole::Developer => {
                system.extend(content_blocks(message.content.as_ref()));
            }
            WireRole::Tool | WireRole::Function => push_tool_result(&mut turns, message),
            WireRole::User => turns.push(Message::new(
                Role::User,
                content_blocks(message.content.as_ref()),
            )),
            WireRole::Assistant => turns.push(Message::new(Role::Assistant, assistant(message))),
        }
    }
    (system, turns)
}

/// Appends one tool result to the caller turn that carries it back.
///
/// A run of results joins one turn; a result after prose starts a new one. The
/// renderer splits results back out into their own template turns either way, so
/// what this grouping decides is only whether the prose and the results are one
/// turn - and they are not, because the caller sent them as separate messages.
fn push_tool_result(turns: &mut Vec<Message>, message: &WireMessage) {
    let result = Block::ToolResult(ToolResult::new(
        message.tool_call_id.clone().unwrap_or_default(),
        content_blocks(message.content.as_ref()),
    ));
    let joinable = turns.last().is_some_and(|turn| {
        turn.role == Role::User
            && turn
                .content
                .iter()
                .all(|block| matches!(block, Block::ToolResult(_)))
    });
    if joinable && let Some(turn) = turns.last_mut() {
        turn.content.push(result);
        return;
    }
    turns.push(Message::new(Role::User, vec![result]));
}

/// One assistant turn's blocks, in the order the model produced them.
fn assistant(message: &WireMessage) -> Vec<Block> {
    let mut blocks = Vec::new();
    // Thinking first: the trace is what the model wrote before the answer, and
    // a replayed turn that puts it after the prose is a turn the model never had.
    if let Some(trace) = message.trace()
        && !trace.is_empty()
    {
        blocks.push(Block::Thinking(shiso_ir::Thinking::new(trace)));
    }
    blocks.extend(content_blocks(message.content.as_ref()));
    if let Some(refusal) = message.refusal.as_ref() {
        blocks.push(Block::Refusal(refusal.clone()));
    }
    for (at, call) in message.tool_calls.iter().enumerate() {
        blocks.push(Block::ToolUse(ToolUse::new(
            call.id.clone().unwrap_or_else(|| format!("call_{at}")),
            call.function.name.clone().unwrap_or_default(),
            arguments(call.function.arguments.as_deref()),
        )));
    }
    blocks
}

/// A call's arguments, parsed, or kept verbatim when they are not JSON.
fn arguments(raw: Option<&str>) -> Value {
    let Some(raw) = raw else {
        return Value::Object(serde_json::Map::new());
    };
    if raw.trim().is_empty() {
        return Value::Object(serde_json::Map::new());
    }
    serde_json::from_str(raw).unwrap_or_else(|_unparsable| Value::String(raw.to_owned()))
}

/// One turn's content as blocks.
fn content_blocks(content: Option<&Content>) -> Vec<Block> {
    match content {
        None => Vec::new(),
        Some(Content::Text(text)) => {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![Block::Text(text.clone())]
            }
        }
        Some(Content::Parts(parts)) => parts.iter().map(part_block).collect(),
    }
}

/// One content part as a block.
fn part_block(part: &Part) -> Block {
    match part {
        Part::Text { text } => Block::Text(text.clone()),
        Part::ImageUrl { image_url } => Block::Image(media(image_url)),
        Part::VideoUrl { video_url } => Block::Video(media(video_url)),
        Part::InputAudio { input_audio } => {
            Block::Audio(MediaSource::Inline(input_audio.data.clone()))
        }
        Part::Refusal { refusal } => Block::Refusal(refusal.clone()),
    }
}

/// Where a media part's bytes are, classified.
///
/// Classification rather than a decode: whether a remote address may be fetched
/// is a policy the renderer holds, and a decoder that was handed a URL would
/// report a base64 failure about a request that never contained base64.
fn media(reference: &MediaRef) -> MediaSource {
    MediaSource::classify(reference.url())
}

/// The neutral tool choice, or the refusal for a mode this server does not know.
///
/// `Ok(None)` means the request named no tool choice at all, which is not a
/// refusal: the caller then applies its own default rather than being told the
/// body was wrong.
fn tool_choice(choice: Option<&WireToolChoice>) -> Result<Option<ToolChoice>, AppError> {
    let Some(choice) = choice else {
        return Ok(None);
    };
    Ok(Some(match choice {
        WireToolChoice::Named { function, .. } => ToolChoice::Named(function.name.clone()),
        WireToolChoice::Mode(mode) => match mode.as_str() {
            "none" => ToolChoice::None,
            "auto" => ToolChoice::Auto,
            "required" => ToolChoice::Any,
            other => {
                return Err(AppError::bad_request(
                    format!("tool_choice {other:?} is not a mode this server knows"),
                    Some("tool_choice"),
                    "invalid_request",
                ));
            }
        },
    }))
}

/// The neutral output constraint.
///
/// `text` is the absence of a constraint rather than a variant of one: a caller
/// that names it has asked for what an unconstrained request already gets.
fn response_format(
    format: Option<&super::chat_wire::WireResponseFormat>,
) -> Option<shiso_ir::ResponseFormat> {
    use super::chat_wire::WireResponseFormat;
    match format? {
        WireResponseFormat::Text => None,
        WireResponseFormat::JsonObject => Some(shiso_ir::ResponseFormat::JsonObject),
        WireResponseFormat::JsonSchema { json_schema } => {
            Some(shiso_ir::ResponseFormat::JsonSchema {
                name: json_schema
                    .name
                    .clone()
                    .unwrap_or_else(|| "schema".to_owned()),
                schema: json_schema.schema.clone().unwrap_or(Value::Null),
                strict: json_schema.strict.unwrap_or(false),
            })
        }
    }
}

/// What the caller said about thinking, from whichever field carried it.
///
/// The template knobs win over the effort word, because a caller that sets both
/// has said something specific with one and something general with the other.
/// An unknown effort word is refused rather than rounded to a rung: rounding up
/// is the most expensive guess available, and rounding down answers a different
/// question than the one asked.
fn thinking(body: &ChatBody) -> Result<ThinkingDirective, AppError> {
    let kwargs = body.chat_template_kwargs.unwrap_or_default();
    if let Some(enabled) = body.enable_thinking.or(kwargs.enable_thinking) {
        return Ok(if enabled {
            ThinkingDirective::On {
                budget: kwargs.thinking_budget,
            }
        } else {
            ThinkingDirective::Off
        });
    }
    if let Some(effort) = body.reasoning_effort.as_deref() {
        return Effort::parse(effort).ok_or_else(|| {
            AppError::bad_request(
                format!(
                    "reasoning_effort {effort:?} is not one of \"none\", \"minimal\", \
                     \"low\", \"medium\", \"high\" or \"xhigh\""
                ),
                Some("reasoning_effort"),
                "invalid_request",
            )
        });
    }
    if let Some(budget) = kwargs.thinking_budget {
        return Ok(ThinkingDirective::On {
            budget: Some(budget),
        });
    }
    Ok(ThinkingDirective::Unspecified)
}

/// The bias table, keyed on token ids rather than on their spelling.
fn logit_bias(
    bias: Option<&std::collections::HashMap<String, f32>>,
) -> Result<Vec<(u32, f32)>, AppError> {
    let Some(bias) = bias else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(bias.len());
    for (key, value) in bias {
        let id: u32 = key.parse().map_err(|_unparsable| {
            AppError::bad_request(
                format!("logit_bias is keyed on token ids, and {key:?} is not one"),
                Some("logit_bias"),
                "invalid_request",
            )
        })?;
        out.push((id, *value));
    }
    // A map has no order and a table that changed order between two identical
    // requests would make a reproducible run irreproducible.
    out.sort_by_key(|(id, _bias)| *id);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::to_ir;
    use crate::openai::chat_wire::ChatBody;
    use shiso_ir::{Block, Role, ThinkingDirective, ToolChoice};

    fn ir(json: serde_json::Value) -> shiso_ir::Request {
        let body: ChatBody = serde_json::from_value(json).expect("the body parses");
        to_ir(&body, "a-model").expect("the body converts")
    }

    #[test]
    fn a_flex_tier_asks_the_scheduler_for_the_flex_class() {
        let messages = serde_json::json!([{"role": "user", "content": "hi"}]);
        let flex = ir(serde_json::json!({"messages": messages, "service_tier": "flex"}));
        assert_eq!(flex.service_class, shiso_ir::ServiceClass::Flex);
        let named = ir(serde_json::json!({"messages": messages, "service_tier": "default"}));
        assert_eq!(named.service_class, shiso_ir::ServiceClass::Standard);
        let silent = ir(serde_json::json!({"messages": messages}));
        assert_eq!(silent.service_class, shiso_ir::ServiceClass::Standard);
    }

    #[test]
    fn instructions_leave_the_conversation_and_become_the_requests_own_field() {
        let request = ir(serde_json::json!({"messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"}
        ]}));
        assert_eq!(request.system, vec![Block::Text("be brief".to_owned())]);
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, Role::User);
    }

    /// The newer spelling reaches the same field, or a client that was upgraded
    /// silently loses its instructions.
    #[test]
    fn a_developer_turn_is_instructions_too() {
        let request = ir(serde_json::json!({"messages": [
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": "hi"}
        ]}));
        assert_eq!(request.system.len(), 1);
    }

    #[test]
    fn a_tool_call_becomes_a_block_of_the_turn_that_made_it() {
        let request = ir(serde_json::json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "checking", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "lookup", "arguments": "{\"city\":\"Paris\"}"}}
            ]}
        ]}));
        let turn = &request.messages[1];
        assert_eq!(turn.role, Role::Assistant);
        assert_eq!(turn.content.len(), 2);
        match &turn.content[1] {
            Block::ToolUse(call) => {
                assert_eq!(call.id, "call_1");
                assert_eq!(call.input["city"], "Paris");
            }
            other => panic!("the second block is the call, not {other:?}"),
        }
    }

    /// The model wrote something that is not JSON. Deleting it would delete the
    /// only evidence of what went wrong.
    #[test]
    fn arguments_that_are_not_json_survive_as_the_text_the_model_wrote() {
        let request = ir(serde_json::json!({"messages": [
            {"role": "assistant", "tool_calls": [
                {"id": "c", "function": {"name": "f", "arguments": "{not json"}}
            ]}
        ]}));
        match &request.messages[0].content[0] {
            Block::ToolUse(call) => assert_eq!(call.input, "{not json"),
            other => panic!("expected a call, got {other:?}"),
        }
    }

    /// There is no tool role, so a result is a block in the caller turn that
    /// carries it back, and a run of them is one turn.
    #[test]
    fn a_run_of_tool_results_becomes_one_caller_turn_of_blocks() {
        let request = ir(serde_json::json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "tool_calls": [
                {"id": "a", "function": {"name": "f", "arguments": "{}"}},
                {"id": "b", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "sunny"},
            {"role": "tool", "tool_call_id": "b", "content": "warm"}
        ]}));
        assert_eq!(request.messages.len(), 3);
        let results = &request.messages[2];
        assert_eq!(results.role, Role::User);
        assert_eq!(results.content.len(), 2);
        match (&results.content[0], &results.content[1]) {
            (Block::ToolResult(first), Block::ToolResult(second)) => {
                assert_eq!(first.tool_use_id, "a");
                assert_eq!(second.tool_use_id, "b");
            }
            other => panic!("two results, not {other:?}"),
        }
    }

    /// A result that follows prose starts its own turn: the caller sent two
    /// messages and the renderer has to see two.
    #[test]
    fn a_tool_result_after_prose_starts_its_own_turn() {
        let request = ir(serde_json::json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "a", "content": "sunny"}
        ]}));
        assert_eq!(request.messages.len(), 2);
    }

    #[test]
    fn a_thinking_trace_is_the_first_block_of_the_turn_that_produced_it() {
        let request = ir(serde_json::json!({"messages": [
            {"role": "assistant", "reasoning_content": "hmm", "content": "yes"}
        ]}));
        assert!(matches!(
            request.messages[0].content[0],
            Block::Thinking(ref trace) if trace.text == "hmm"
        ));
        assert!(matches!(request.messages[0].content[1], Block::Text(_)));
    }

    #[test]
    fn a_media_part_is_classified_rather_than_fetched_or_decoded() {
        let request = ir(
            serde_json::json!({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "video_url", "video_url": "https://example.test/clip.mp4"}
            ]}]}),
        );
        let blocks = &request.messages[0].content;
        assert!(matches!(
            blocks[0],
            Block::Image(shiso_ir::MediaSource::Inline(_))
        ));
        assert!(matches!(
            blocks[1],
            Block::Video(shiso_ir::MediaSource::Remote(_))
        ));
    }

    #[test]
    fn an_audio_part_carries_its_base64_inline_whatever_its_format_says() {
        let request = ir(
            serde_json::json!({"messages": [{"role": "user", "content": [
                {"type": "text", "text": "what was said?"},
                {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "mp3"}}
            ]}]}),
        );
        assert!(matches!(
            request.messages[0].content[1],
            Block::Audio(shiso_ir::MediaSource::Inline(ref data)) if data == "UklGRg=="
        ));
    }

    #[test]
    fn an_audio_part_without_data_is_refused() {
        let body = serde_json::json!({"messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"format": "wav"}}
        ]}]});
        assert!(serde_json::from_value::<ChatBody>(body).is_err());
    }

    #[test]
    fn the_tool_choice_vocabulary_lands_on_the_four_things_it_can_mean() {
        let mode = |value: serde_json::Value| {
            ir(serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_choice": value
            }))
            .tool_choice
        };
        assert_eq!(mode(serde_json::json!("none")), Some(ToolChoice::None));
        assert_eq!(mode(serde_json::json!("auto")), Some(ToolChoice::Auto));
        assert_eq!(mode(serde_json::json!("required")), Some(ToolChoice::Any));
        assert_eq!(
            mode(serde_json::json!({"type": "function", "function": {"name": "f"}})),
            Some(ToolChoice::Named("f".to_owned()))
        );
    }

    /// A tool with no schema takes no arguments, and an empty object is how a
    /// template writes that. `null` renders as the word null in the prompt.
    #[test]
    fn a_tool_with_no_schema_is_offered_as_one_that_takes_no_arguments() {
        let request = ir(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}]
        }));
        assert_eq!(request.tools[0].input_schema["type"], "object");
    }

    #[test]
    fn an_effort_word_this_server_does_not_know_is_refused_rather_than_rounded() {
        let body: ChatBody = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "enormous"
        }))
        .expect("the body parses");
        let error = to_ir(&body, "a-model").expect_err("an unknown rung");
        assert!(error.to_string().contains("enormous"), "{error}");
    }

    #[test]
    fn turning_thinking_off_through_the_template_knob_beats_an_effort_word() {
        let request = ir(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": false}
        }));
        assert_eq!(request.thinking, ThinkingDirective::Off);
    }

    /// `preserve_thinking` inside the template knobs reaches the template the
    /// same as the top-level field; the top-level one wins when both are sent.
    #[test]
    fn preserve_thinking_is_read_from_the_template_knobs_too() {
        let request = ir(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"preserve_thinking": false}
        }));
        assert_eq!(request.preserve_thinking, Some(false));
        let request = ir(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "preserve_thinking": true,
            "chat_template_kwargs": {"preserve_thinking": false}
        }));
        assert_eq!(request.preserve_thinking, Some(true));
    }

    #[test]
    fn a_bias_keyed_on_something_that_is_not_a_token_id_is_refused() {
        let body: ChatBody = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "logit_bias": {"hello": 1.0}
        }))
        .expect("the body parses");
        assert!(to_ir(&body, "a-model").is_err());
    }

    /// The label a client routes by is echoed, not enforced: one process serves
    /// one checkpoint, and a name mismatch would break every gateway alias.
    #[test]
    fn the_request_is_stamped_with_the_checkpoint_this_process_actually_serves() {
        let request = ir(serde_json::json!({
            "model": "something-else",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(request.model, "a-model");
    }
}
