//! The Messages body, turned into the neutral request.
//!
//! # This is the mapping the IR was rebuilt for
//!
//! Almost every block, role and field here has a counterpart of the same shape
//! on the other side, so the interesting thing about this module is how little
//! it does. A turn's blocks keep their order and their kind; a tool result
//! stays a block in the turn that carries it, with its failure flag still a
//! flag; a thinking trace stays a block with its signature beside it; the
//! instructions stay beside the conversation. The chat surface has to lift
//! system turns out of the history, collapse tool roles into blocks, reorder a
//! turn's parallel fields and parse a call's arguments out of a string. None of
//! that happens on this route, because the wire format and the IR agree about
//! what a turn is.
//!
//! # What is refused rather than dropped
//!
//! A content block of a kind this server cannot render is a 400 naming where it
//! was. Dropping it would leave the model answering about content it was never
//! shown, and the answer would look exactly like a good one.
//!
//! The one exception is a redacted thinking block, which is dropped: its
//! payload is ciphertext belonging to the service that issued it, so there is
//! no text a model could be shown and nothing a refusal would let the caller
//! fix.

use serde_json::{Value, json};
use shiso_ir::{
    Block, Effort, MediaSource, Message, Request, Role, Sampling, Thinking, ThinkingDirective,
    ToolChoice, ToolDefinition, ToolResult, ToolUse,
};

use crate::anthropic::messages_wire::{
    Content, MessagesBody, Source, SystemPrompt, ToolOutput, WireBlock, WireMessage,
    WireOutputConfig, WireRole, WireThinking, WireToolChoice,
};
use crate::error::AppError;

/// Turns a validated body into the neutral request.
///
/// `model_id` is the checkpoint this process serves; the body's own `model` is
/// a label a client chose and is echoed rather than honoured, because one
/// process serves one checkpoint and refusing a name mismatch would break every
/// client that routes through a gateway alias.
///
/// # Errors
///
/// [`AppError::BadRequest`] for a content block this server cannot render, a
/// media source it cannot reach, instructions that are not text, or an
/// `output_config.effort` word this server does not know.
pub fn to_ir(body: &MessagesBody, model_id: &str) -> Result<Request, AppError> {
    let mut request = Request::new(model_id, Vec::new());
    request.system = system_blocks(body.system.as_ref())?;
    request.messages = body
        .messages
        .iter()
        .enumerate()
        .map(|(at, message)| turn(at, message))
        .collect::<Result<Vec<Message>, AppError>>()?;

    request.tools = body
        .tools
        .iter()
        .map(|tool| {
            // A tool with no schema takes no arguments, and an empty object is
            // how a template writes that. `null` would render as the word null
            // in the prompt.
            let schema = tool
                .input_schema
                .clone()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            let mut definition = ToolDefinition::new(tool.name.clone(), schema);
            definition.description.clone_from(&tool.description);
            definition
        })
        .collect();
    request.tool_choice = body.tool_choice.as_ref().map(tool_choice);

    request.sampling = Sampling {
        temperature: body.temperature,
        top_k: body.top_k,
        top_p: body.top_p,
        top_n_sigma: None,
        min_p: None,
        repetition_penalty: None,
        presence_penalty: None,
        frequency_penalty: None,
    };
    request.max_tokens = body.max_tokens;
    request.stop.clone_from(&body.stop_sequences);
    request.stream = body.is_streaming();
    request.thinking = thinking(body.thinking.as_ref(), body.output_config.as_ref())?;
    Ok(request)
}

/// The standing instructions as blocks.
///
/// Text only: every other block kind in this position is a caching or citation
/// hint about text that is not here, and a server that accepted one would be
/// claiming to honour something it has no machinery for.
fn system_blocks(system: Option<&SystemPrompt>) -> Result<Vec<Block>, AppError> {
    match system {
        None => Ok(Vec::new()),
        Some(SystemPrompt::Text(text)) => Ok(if text.is_empty() {
            Vec::new()
        } else {
            vec![Block::Text(text.clone())]
        }),
        Some(SystemPrompt::Blocks(blocks)) => blocks
            .iter()
            .map(|block| match block {
                WireBlock::Text { text } => Ok(Block::Text(text.clone())),
                _ => Err(AppError::bad_request(
                    "system accepts text blocks only",
                    Some("system"),
                    "invalid_request",
                )),
            })
            .collect(),
    }
}

/// One turn, with its blocks in the order the caller sent them.
fn turn(at: usize, message: &WireMessage) -> Result<Message, AppError> {
    let role = match message.role {
        WireRole::User => Role::User,
        WireRole::Assistant => Role::Assistant,
    };
    Ok(Message::new(role, content_blocks(at, &message.content)?))
}

/// One turn's content as neutral blocks.
fn content_blocks(at: usize, content: &Content) -> Result<Vec<Block>, AppError> {
    match content {
        Content::Text(text) => Ok(if text.is_empty() {
            Vec::new()
        } else {
            vec![Block::Text(text.clone())]
        }),
        Content::Blocks(blocks) => {
            let mut out = Vec::with_capacity(blocks.len());
            for (index, block) in blocks.iter().enumerate() {
                if let Some(block) = map_block(&Where { at, index }, block)? {
                    out.push(block);
                }
            }
            Ok(out)
        }
    }
}

/// Where a block sat, so a refusal can point at it.
///
/// A caller whose thirtieth turn carries one block this server will not render
/// needs to know which one, and "there is a block I cannot render" is a
/// complaint they cannot act on.
#[derive(Debug, Clone, Copy)]
struct Where {
    at: usize,
    index: usize,
}

impl Where {
    /// The path a caller would use to find the block in their own body.
    fn path(self) -> String {
        format!("messages[{}].content[{}]", self.at, self.index)
    }
}

/// One wire block as a neutral one, or `None` for a block with nothing to
/// carry.
fn map_block(at: &Where, block: &WireBlock) -> Result<Option<Block>, AppError> {
    Ok(Some(match block {
        WireBlock::Text { text } => Block::Text(text.clone()),
        WireBlock::Image { source } => Block::Image(media(at, source)?),
        WireBlock::Thinking {
            thinking,
            signature,
        } => {
            let mut trace = Thinking::new(thinking.clone());
            trace.signature.clone_from(signature);
            Block::Thinking(trace)
        }
        WireBlock::RedactedThinking { .. } => {
            tracing::debug!(
                at = %at.path(),
                "a redacted thinking block carries no text this model could be \
                 shown, so it is not replayed"
            );
            return Ok(None);
        }
        WireBlock::ToolUse { id, name, input } => Block::ToolUse(ToolUse::new(
            id.clone(),
            name.clone(),
            normalize_input(input),
        )),
        WireBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            let blocks = tool_output(at, content.as_ref())?;
            Block::ToolResult(if *is_error {
                ToolResult::error(tool_use_id.clone(), blocks)
            } else {
                ToolResult::new(tool_use_id.clone(), blocks)
            })
        }
        WireBlock::Unsupported => {
            return Err(AppError::bad_request(
                format!(
                    "{} is a content block of a kind this server does not serve",
                    at.path()
                ),
                Some("messages"),
                "unsupported_value",
            ));
        }
    }))
}

/// A tool result's own content as blocks.
///
/// Absent content is an empty result rather than a refusal: a tool that
/// succeeded and had nothing to say is a thing that happens, and the block's
/// presence is what tells the model the call was answered at all.
fn tool_output(at: &Where, content: Option<&ToolOutput>) -> Result<Vec<Block>, AppError> {
    match content {
        None => Ok(Vec::new()),
        Some(ToolOutput::Text(text)) => Ok(vec![Block::Text(text.clone())]),
        Some(ToolOutput::Blocks(blocks)) => {
            let mut out = Vec::with_capacity(blocks.len());
            for block in blocks {
                if let Some(block) = map_block(at, block)? {
                    out.push(block);
                }
            }
            Ok(out)
        }
    }
}

/// A call's arguments, with the one spelling that is not an object repaired.
///
/// This format sends arguments as JSON rather than as a string holding JSON, so
/// there is no parse here at all - the one thing worth normalising is an absent
/// `input`, which serde fills in as null and which a template would render as
/// the word null.
fn normalize_input(input: &Value) -> Value {
    if input.is_null() {
        Value::Object(serde_json::Map::new())
    } else {
        input.clone()
    }
}

/// Where a media block's bytes are, classified.
///
/// The base64 payload travels as itself rather than wrapped in a data URI: the
/// decoder accepts either and reads the format from the bytes, so wrapping it
/// would build a string only to have it taken apart again. An address stays an
/// address, and whether this server may fetch one is the renderer's policy.
fn media(at: &Where, source: &Source) -> Result<MediaSource, AppError> {
    match source {
        Source::Base64 { data, .. } => Ok(MediaSource::Inline(data.clone())),
        Source::Url { url } => Ok(MediaSource::Remote(url.clone())),
        Source::Unsupported => Err(AppError::bad_request(
            format!(
                "{} names a media source this server cannot reach; send the \
                 bytes inline",
                at.path()
            ),
            Some("messages"),
            "unsupported_value",
        )),
    }
}

/// The neutral tool choice.
///
/// Total, because this format's four spellings are the IR's four variants - the
/// one place where a wire enum and a neutral one are the same enum.
fn tool_choice(choice: &WireToolChoice) -> ToolChoice {
    match choice {
        WireToolChoice::Auto => ToolChoice::Auto,
        WireToolChoice::Any => ToolChoice::Any,
        WireToolChoice::None => ToolChoice::None,
        WireToolChoice::Tool { name } => ToolChoice::Named(name.clone()),
    }
}

/// What the caller said about thinking.
///
/// `thinking.type` says whether; `output_config.effort` says how hard. Silence on
/// both stays silence, so the server's default applies later. `disabled` wins
/// over any effort, because off is the more specific statement. Adaptive with no
/// effort is thinking on with no rung named, so the template's own default
/// applies. A budget is accepted and not enforced (a decision, not an omission).
fn thinking(
    directive: Option<&WireThinking>,
    output: Option<&WireOutputConfig>,
) -> Result<ThinkingDirective, AppError> {
    let effort = match output.and_then(|config| config.effort.as_deref()) {
        None => None,
        Some(word) => Some(Effort::parse(word).ok_or_else(|| {
            AppError::bad_request(
                format!(
                    "output_config.effort {word:?} is not one of \"low\", \"medium\", \
                     \"high\", \"xhigh\" or \"max\""
                ),
                Some("output_config.effort"),
                "invalid_request",
            )
        })?),
    };
    Ok(match (directive, effort) {
        (Some(WireThinking::Disabled), _) => ThinkingDirective::Off,
        (_, Some(level)) => level,
        (Some(WireThinking::Enabled { budget_tokens }), None) => ThinkingDirective::On {
            budget: *budget_tokens,
        },
        (Some(WireThinking::Adaptive), None) => ThinkingDirective::On { budget: None },
        (None, None) => ThinkingDirective::Unspecified,
    })
}

#[cfg(test)]
mod tests {
    use super::to_ir;
    use crate::anthropic::messages_wire::MessagesBody;
    use serde_json::json;
    use shiso_ir::{Block, Effort, Role, ThinkingDirective, ToolChoice};

    fn body(json: serde_json::Value) -> MessagesBody {
        serde_json::from_value(json).expect("the body parses")
    }

    fn request(content: &serde_json::Value) -> shiso_ir::Request {
        let parsed = body(json!({
            "model": "a-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": content}],
        }));
        to_ir(&parsed, "test-model").expect("the body maps")
    }

    #[test]
    fn the_model_the_server_serves_is_the_model_the_request_carries() {
        let mapped = request(&json!("hi"));
        assert_eq!(mapped.model, "test-model");
        assert_eq!(mapped.max_tokens, 64);
    }

    /// The order of a turn's blocks is the order the caller sent them, with
    /// nothing lifted out into a field beside the list. That is the whole
    /// reason this mapping is short.
    #[test]
    fn a_turn_keeps_every_block_it_was_sent_in_the_order_it_was_sent() {
        let mapped = request(&json!([
            {"type": "text", "text": "before"},
            {"type": "tool_result", "tool_use_id": "call_1", "content": "42"},
            {"type": "text", "text": "after"},
        ]));
        let turn = &mapped.messages[0];
        assert_eq!(turn.role, Role::User);
        assert!(matches!(turn.content[0], Block::Text(ref t) if t == "before"));
        assert!(matches!(turn.content[1], Block::ToolResult(_)));
        assert!(matches!(turn.content[2], Block::Text(ref t) if t == "after"));
    }

    /// A failed tool result travels as a flag, not as a prefix on its text, so
    /// nothing downstream has to know a convention in order to see the failure.
    #[test]
    fn a_failed_tool_result_carries_its_failure_as_a_flag() {
        let mapped = request(&json!([
            {"type": "tool_result", "tool_use_id": "call_1", "content": "boom", "is_error": true},
        ]));
        let Block::ToolResult(ref result) = mapped.messages[0].content[0] else {
            panic!("a tool result maps to a tool-result block");
        };
        assert!(result.is_error);
        assert_eq!(result.tool_use_id, "call_1");
        assert!(matches!(result.content[0], Block::Text(ref t) if t == "boom"));

        let fine = request(&json!([
            {"type": "tool_result", "tool_use_id": "call_1", "content": "42"},
        ]));
        let Block::ToolResult(ref result) = fine.messages[0].content[0] else {
            panic!("a tool result maps to a tool-result block");
        };
        assert!(!result.is_error);
    }

    /// A trace replayed into a later turn keeps its signature: the signature is
    /// what lets the surface that issued it accept the trace back, and nothing
    /// between the two is entitled to drop it.
    #[test]
    fn a_thinking_block_keeps_its_text_and_its_signature() {
        let parsed = body(json!({
            "model": "a-model",
            "max_tokens": 64,
            "messages": [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "step one", "signature": "sig-abc"},
                {"type": "text", "text": "done"},
            ]}],
        }));
        let mapped = to_ir(&parsed, "test-model").expect("the body maps");
        let Block::Thinking(ref trace) = mapped.messages[0].content[0] else {
            panic!("a thinking block maps to a thinking block");
        };
        assert_eq!(trace.text, "step one");
        assert_eq!(trace.signature.as_deref(), Some("sig-abc"));
    }

    /// There is nothing in a redacted trace a model could be shown, so it is
    /// the one block that is dropped rather than refused.
    #[test]
    fn a_redacted_thinking_block_is_dropped_and_the_rest_of_the_turn_survives() {
        let parsed = body(json!({
            "model": "a-model",
            "max_tokens": 64,
            "messages": [{"role": "assistant", "content": [
                {"type": "redacted_thinking", "data": "AAAA"},
                {"type": "text", "text": "done"},
            ]}],
        }));
        let mapped = to_ir(&parsed, "test-model").expect("the body maps");
        assert_eq!(mapped.messages[0].content.len(), 1);
        assert!(matches!(mapped.messages[0].content[0], Block::Text(_)));
    }

    /// A block nobody can render must not be dropped: an answer about content
    /// the model never saw looks exactly like a good answer.
    #[test]
    fn a_block_this_server_cannot_render_is_refused_and_says_where_it_was() {
        let parsed = body(json!({
            "model": "a-model",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": [
                    {"type": "text", "text": "and"},
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "QQ=="}},
                ]},
            ],
        }));
        let error = to_ir(&parsed, "test-model").expect_err("a document block");
        assert_eq!(error.code(), "unsupported_value");
        assert!(
            error.to_string().contains("messages[1].content[1]"),
            "the refusal names the block: {error}"
        );
    }

    /// Inline bytes reach the decoder as the base64 the caller sent, and an
    /// address stays an address so the renderer's policy is what decides
    /// whether it may be fetched.
    #[test]
    fn a_media_block_carries_its_bytes_or_its_address_and_not_a_rebuilt_string() {
        let inline = request(&json!([
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QQ=="}},
        ]));
        assert!(matches!(
            inline.messages[0].content[0],
            Block::Image(shiso_ir::MediaSource::Inline(ref data)) if data == "QQ=="
        ));

        let remote = request(&json!([
            {"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}},
        ]));
        assert!(matches!(
            remote.messages[0].content[0],
            Block::Image(shiso_ir::MediaSource::Remote(ref url))
                if url == "https://example.test/a.png"
        ));
    }

    #[test]
    fn a_media_source_this_server_cannot_reach_is_refused() {
        let parsed = body(json!({
            "model": "a-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "file", "file_id": "file_1"}},
            ]}],
        }));
        let error = to_ir(&parsed, "test-model").expect_err("a file source");
        assert!(error.to_string().contains("inline"), "{error}");
    }

    #[test]
    fn the_instructions_are_a_field_and_blocks_join_it_in_order() {
        let parsed = body(json!({
            "model": "a-model",
            "max_tokens": 64,
            "system": [{"type": "text", "text": "be brief"}, {"type": "text", "text": "be kind"}],
            "messages": [{"role": "user", "content": "hi"}],
        }));
        let mapped = to_ir(&parsed, "test-model").expect("the body maps");
        assert_eq!(mapped.system.len(), 2);
        assert!(matches!(mapped.system[0], Block::Text(ref t) if t == "be brief"));
        assert_eq!(
            mapped.messages.len(),
            1,
            "instructions are not a turn of the conversation"
        );
    }

    #[test]
    fn a_tool_offer_carries_its_description_and_its_schema() {
        let parsed = body(json!({
            "model": "a-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "name": "lookup",
                "description": "look a city up",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
            "tool_choice": {"type": "tool", "name": "lookup"},
        }));
        let mapped = to_ir(&parsed, "test-model").expect("the body maps");
        assert_eq!(mapped.tools[0].name, "lookup");
        assert_eq!(
            mapped.tools[0].description.as_deref(),
            Some("look a city up")
        );
        assert_eq!(
            mapped.tools[0].input_schema["properties"]["city"]["type"],
            "string"
        );
        assert_eq!(
            mapped.tool_choice,
            Some(ToolChoice::Named("lookup".to_owned()))
        );
    }

    /// The arguments arrive as JSON and stay JSON: this surface never parses a
    /// string holding JSON, which is the one parse the chat surface cannot
    /// avoid.
    #[test]
    fn a_replayed_tool_call_keeps_its_arguments_parsed() {
        let parsed = body(json!({
            "model": "a-model",
            "max_tokens": 64,
            "messages": [{"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"city": "Paris"}},
            ]}],
        }));
        let mapped = to_ir(&parsed, "test-model").expect("the body maps");
        let Block::ToolUse(ref call) = mapped.messages[0].content[0] else {
            panic!("a tool_use block maps to a tool call");
        };
        assert_eq!(call.input["city"], "Paris");

        let no_input = body(json!({
            "model": "a-model",
            "max_tokens": 64,
            "messages": [{"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "ping"},
            ]}],
        }));
        let mapped = to_ir(&no_input, "test-model").expect("the body maps");
        let Block::ToolUse(ref call) = mapped.messages[0].content[0] else {
            panic!("a tool_use block maps to a tool call");
        };
        assert!(call.input.is_object(), "an absent input is an empty object");
    }

    #[test]
    fn the_thinking_ladder_keeps_off_on_and_silent_apart() {
        let with = |thinking: serde_json::Value| {
            let parsed = body(json!({
                "model": "a-model",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": thinking,
            }));
            to_ir(&parsed, "test-model")
                .expect("the body maps")
                .thinking
        };
        assert_eq!(with(json!({"type": "disabled"})), ThinkingDirective::Off);
        assert_eq!(
            with(json!({"type": "enabled", "budget_tokens": 2048})),
            ThinkingDirective::On { budget: Some(2048) }
        );
        assert_eq!(
            with(json!({"type": "enabled"})),
            ThinkingDirective::On { budget: None }
        );
        assert_eq!(
            request(&json!("hi")).thinking,
            ThinkingDirective::Unspecified,
            "silence is its own state until policy fills it in"
        );
    }

    fn base() -> serde_json::Value {
        json!({"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]})
    }

    #[test]
    fn adaptive_with_an_effort_is_that_effort() {
        let mut body = base();
        body["thinking"] = json!({"type": "adaptive"});
        body["output_config"] = json!({"effort": "low"});
        let request = to_ir(&serde_json::from_value(body).expect("parses"), "m").expect("converts");
        assert_eq!(request.thinking, ThinkingDirective::Effort(Effort::Low));
    }

    /// Adaptive with no effort is thinking on, and the template's own default
    /// effort applies - not the middle rung, and not a refusal.
    #[test]
    fn adaptive_without_an_effort_is_thinking_on_with_the_template_default() {
        let mut body = base();
        body["thinking"] = json!({"type": "adaptive"});
        let request = to_ir(&serde_json::from_value(body).expect("parses"), "m").expect("converts");
        assert_eq!(request.thinking, ThinkingDirective::On { budget: None });
    }

    #[test]
    fn an_effort_beside_enabled_is_honoured_and_disabled_wins_over_it() {
        let mut body = base();
        body["thinking"] = json!({"type": "enabled", "budget_tokens": 1024});
        body["output_config"] = json!({"effort": "max"});
        let request =
            to_ir(&serde_json::from_value(body.clone()).expect("parses"), "m").expect("converts");
        assert_eq!(request.thinking, ThinkingDirective::Effort(Effort::XHigh));
        body["thinking"] = json!({"type": "disabled"});
        let request = to_ir(&serde_json::from_value(body).expect("parses"), "m").expect("converts");
        assert_eq!(request.thinking, ThinkingDirective::Off);
    }

    #[test]
    fn an_unknown_effort_word_is_refused() {
        let mut body = base();
        body["thinking"] = json!({"type": "adaptive"});
        body["output_config"] = json!({"effort": "enormous"});
        let error =
            to_ir(&serde_json::from_value(body).expect("parses"), "m").expect_err("refused");
        assert!(
            error.to_string().contains("output_config.effort"),
            "{error}"
        );
    }

    #[test]
    fn the_stop_sequences_and_the_stream_flag_carry_through() {
        let parsed = body(json!({
            "model": "a-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["END", "STOP"],
            "stream": true,
            "temperature": 0.5,
            "top_k": 40,
        }));
        let mapped = to_ir(&parsed, "test-model").expect("the body maps");
        assert_eq!(mapped.stop, vec!["END".to_owned(), "STOP".to_owned()]);
        assert!(mapped.stream);
        assert_eq!(mapped.sampling.temperature, Some(0.5));
        assert_eq!(mapped.sampling.top_k, Some(40));
    }
}
