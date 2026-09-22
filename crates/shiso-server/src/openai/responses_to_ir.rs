//! The Responses body, mapped straight onto the neutral request.
//!
//! # There is no chat hop
//!
//! This surface is not lowered into a chat completion first. Routing one wire
//! format through another would make every field here a question about what the
//! chat format does with it - and the chat format has no place for half of them,
//! so the answer would be "it is lost". The narrow waist is the neutral request,
//! and every surface converts straight to it.
//!
//! # Continuing a conversation
//!
//! `previous_response_id` names a stored transcript, and the transcript is
//! prepended to whatever the caller sent. The stored form is the neutral turns,
//! so the prepend is a splice rather than a re-parse. An identifier the store
//! does not know is refused: continuing from a conversation the server has
//! forgotten would silently answer a different question than the caller asked.

use serde_json::Value;
use shiso_ir::{
    Block, Effort, MediaSource, Message, Request, Role, Sampling, ThinkingDirective, ToolChoice,
    ToolDefinition, ToolResult, ToolUse,
};

use crate::error::AppError;
use crate::openai::chat_wire::StopField;
use crate::openai::responses_wire::{
    BareMessage, ContentPart, InputContent, InputField, InputItem, InputRole, ResponsesBody,
    ResponsesTool, ResponsesToolChoice, TypedItem,
};

/// Turns a validated body into the neutral request.
///
/// `prior` is the transcript named by `previous_response_id`, already looked up
/// by the handler. It is passed in rather than fetched here so that this
/// function is a pure mapping and can be tested without a store.
///
/// # Errors
///
/// [`AppError::BadRequest`] for a hosted tool this server cannot run, a thinking
/// rung it does not know, or a tool-choice mode it does not know.
pub fn to_ir(
    body: &ResponsesBody,
    model_id: &str,
    prior: Vec<Message>,
) -> Result<Request, AppError> {
    let mut request = Request::new(model_id, prior);

    if let Some(instructions) = body.instructions.as_ref()
        && !instructions.is_empty()
    {
        request.system.push(Block::Text(instructions.clone()));
    }
    if let Some(input) = body.input.as_ref() {
        append_input(&mut request, input);
    }

    request.tools = tools(&body.tools)?;
    request.tool_choice = tool_choice(body.tool_choice.as_ref())?;
    request.sampling = Sampling {
        temperature: body.temperature,
        top_k: body.top_k,
        top_p: body.top_p,
        ..Sampling::default()
    };
    request.max_tokens = body.max_output_tokens.unwrap_or(0);
    request.stop = body
        .stop
        .clone()
        .map(StopField::into_vec)
        .unwrap_or_default();
    request.stream = body.stream.unwrap_or(false);
    request.thinking = thinking(body.reasoning.as_ref())?;
    request.response_format = response_format(body.text.as_ref());
    request.seed = body.seed;
    Ok(request)
}

/// Folds the input list into turns, merging adjacent items of one speaker.
///
/// Merging is what makes a function call and the prose beside it one turn again:
/// the caller sent them as separate items because that is the shape this format
/// has, and the model produced them as one turn.
fn append_input(request: &mut Request, input: &InputField) {
    match input {
        InputField::Text(text) => request.messages.push(Message::user(text.clone())),
        InputField::Items(items) => {
            for item in items {
                append_item(request, item);
            }
        }
    }
}

/// One input item, as blocks on the turn it belongs to.
fn append_item(request: &mut Request, item: &InputItem) {
    match item {
        InputItem::Bare(BareMessage { role, content })
        | InputItem::Typed(TypedItem::Message { role, content }) => {
            append_turn(request, *role, content_blocks(content));
        }
        InputItem::Typed(TypedItem::FunctionCall {
            call_id,
            name,
            arguments,
        }) => {
            let call = Block::ToolUse(ToolUse::new(
                call_id.clone().unwrap_or_default(),
                name.clone(),
                parse_arguments(arguments.as_deref()),
            ));
            push(request, Role::Assistant, vec![call]);
        }
        InputItem::Typed(TypedItem::FunctionCallOutput { call_id, output }) => {
            let result = Block::ToolResult(ToolResult::new(
                call_id.clone(),
                vec![Block::Text(output_text(output))],
            ));
            push(request, Role::User, vec![result]);
        }
        InputItem::Typed(TypedItem::Reasoning { summary }) => {
            let text: String = summary
                .iter()
                .filter_map(|part| part.text.as_deref())
                .collect();
            if !text.is_empty() {
                push(
                    request,
                    Role::Assistant,
                    vec![Block::Thinking(shiso_ir::Thinking::new(text))],
                );
            }
        }
    }
}

/// A turn whose role may be instructions rather than a speaker.
fn append_turn(request: &mut Request, role: InputRole, blocks: Vec<Block>) {
    match role {
        // Instructions are a field, not a turn, wherever they appear in the
        // list. Every template renders them in a place of its own.
        InputRole::System | InputRole::Developer => request.system.extend(blocks),
        InputRole::User => push(request, Role::User, blocks),
        InputRole::Assistant => push(request, Role::Assistant, blocks),
    }
}

/// Appends blocks to the last turn when it is the same speaker's, or opens one.
fn push(request: &mut Request, role: Role, blocks: Vec<Block>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = request.messages.last_mut()
        && last.role == role
    {
        last.content.extend(blocks);
        return;
    }
    request.messages.push(Message::new(role, blocks));
}

/// One item's content as blocks.
fn content_blocks(content: &InputContent) -> Vec<Block> {
    match content {
        InputContent::Text(text) => {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![Block::Text(text.clone())]
            }
        }
        InputContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::InputText { text } | ContentPart::OutputText { text } => {
                    Block::Text(text.clone())
                }
                ContentPart::InputImage { image_url } => {
                    Block::Image(MediaSource::classify(image_url.url()))
                }
                ContentPart::InputVideo { video_url } => {
                    Block::Video(MediaSource::classify(video_url.url()))
                }
                ContentPart::Refusal { refusal } => Block::Refusal(refusal.clone()),
            })
            .collect(),
    }
}

/// A tool result's text, whatever JSON shape it arrived as.
///
/// A string stays a string: quoting it would show the model an escaped value
/// where it expects the value.
fn output_text(output: &Value) -> String {
    match output {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// A replayed call's arguments, parsed, or kept verbatim when they are not JSON.
fn parse_arguments(raw: Option<&str>) -> Value {
    let Some(raw) = raw.filter(|text| !text.trim().is_empty()) else {
        return Value::Object(serde_json::Map::new());
    };
    serde_json::from_str(raw).unwrap_or_else(|_unparsable| Value::String(raw.to_owned()))
}

/// The tools this server can actually run, or a refusal naming the one it
/// cannot.
fn tools(offered: &[ResponsesTool]) -> Result<Vec<ToolDefinition>, AppError> {
    let mut out = Vec::with_capacity(offered.len());
    for tool in offered {
        if tool.tool_type != "function" {
            return Err(AppError::bad_request(
                format!(
                    "this server runs no hosted tools, and {:?} is one; offer a \
                     function tool and run it yourself",
                    tool.tool_type
                ),
                Some("tools"),
                "unsupported_value",
            ));
        }
        let nested = tool.function.as_ref();
        let name = tool
            .name
            .clone()
            .or_else(|| nested.map(|function| function.name.clone()))
            .ok_or_else(|| {
                AppError::bad_request(
                    "a function tool must have a name",
                    Some("tools"),
                    "invalid_request",
                )
            })?;
        let schema = tool
            .parameters
            .clone()
            .or_else(|| nested.and_then(|function| function.parameters.clone()))
            .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
        let mut definition = ToolDefinition::new(name, schema);
        definition.description = tool
            .description
            .clone()
            .or_else(|| nested.and_then(|function| function.description.clone()));
        out.push(definition);
    }
    Ok(out)
}

/// The neutral tool choice.
///
/// `Ok(None)` means the request named no tool choice at all, which is not a
/// refusal: the caller then applies its own default rather than being told the
/// body was wrong.
fn tool_choice(choice: Option<&ResponsesToolChoice>) -> Result<Option<ToolChoice>, AppError> {
    let Some(choice) = choice else {
        return Ok(None);
    };
    Ok(Some(match choice {
        ResponsesToolChoice::Named { name, .. } => ToolChoice::Named(name.clone()),
        ResponsesToolChoice::Mode(mode) => match mode.as_str() {
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

/// What the caller said about thinking.
fn thinking(
    reasoning: Option<&crate::openai::responses_wire::ReasoningConfig>,
) -> Result<ThinkingDirective, AppError> {
    let Some(effort) = reasoning.and_then(|config| config.effort.as_deref()) else {
        return Ok(ThinkingDirective::Unspecified);
    };
    Effort::parse(effort).ok_or_else(|| {
        AppError::bad_request(
            format!(
                "reasoning.effort {effort:?} is not one of \"none\", \"minimal\", \
                 \"low\", \"medium\", \"high\" or \"xhigh\""
            ),
            Some("reasoning.effort"),
            "invalid_request",
        )
    })
}

/// The neutral output constraint, from this surface's nested spelling.
fn response_format(
    text: Option<&crate::openai::responses_wire::TextConfig>,
) -> Option<shiso_ir::ResponseFormat> {
    let format = text?.format.as_ref()?;
    match format.get("type").and_then(Value::as_str)? {
        "json_object" => Some(shiso_ir::ResponseFormat::JsonObject),
        "json_schema" => Some(shiso_ir::ResponseFormat::JsonSchema {
            name: format
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("schema")
                .to_owned(),
            schema: format.get("schema").cloned().unwrap_or(Value::Null),
            strict: format
                .get("strict")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }),
        // `text` is the absence of a constraint, and an unknown spelling is not
        // something to guess at: an unconstrained answer is what a caller that
        // named nothing would have got.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::to_ir;
    use crate::openai::responses_wire::ResponsesBody;
    use shiso_ir::{Block, Message, Request, Role, ThinkingDirective, ToolChoice};

    fn ir(json: serde_json::Value) -> Request {
        let body: ResponsesBody = serde_json::from_value(json).expect("the body parses");
        to_ir(&body, "a-model", Vec::new()).expect("the body converts")
    }

    #[test]
    fn a_bare_string_input_is_one_caller_turn() {
        let request = ir(serde_json::json!({"input": "hello"}));
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, Role::User);
        assert_eq!(request.messages[0].text(), "hello");
    }

    #[test]
    fn instructions_become_the_requests_own_field_rather_than_a_turn() {
        let request = ir(serde_json::json!({"input": "hi", "instructions": "be brief"}));
        assert_eq!(request.system, vec![Block::Text("be brief".to_owned())]);
        assert_eq!(request.messages.len(), 1);
    }

    /// A system item in the middle of the list is still instructions, because
    /// every template renders them in a place of their own.
    #[test]
    fn a_system_item_anywhere_in_the_list_is_still_instructions() {
        let request = ir(serde_json::json!({"input": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "be brief"}
        ]}));
        assert_eq!(request.system.len(), 1);
        assert_eq!(request.messages.len(), 1);
    }

    /// A call and the prose beside it were sent as two items and were produced
    /// as one turn; merging is what puts them back.
    #[test]
    fn a_call_and_the_prose_beside_it_become_one_assistant_turn() {
        let request = ir(serde_json::json!({"input": [
            {"role": "user", "content": "weather?"},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "checking"}]},
            {"type": "function_call", "call_id": "c1", "name": "lookup",
             "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"}
        ]}));
        assert_eq!(request.messages.len(), 3);
        let turn = &request.messages[1];
        assert_eq!(turn.role, Role::Assistant);
        assert_eq!(turn.content.len(), 2);
        match &turn.content[1] {
            Block::ToolUse(call) => {
                assert_eq!(call.id, "c1");
                assert_eq!(call.input["city"], "Paris");
            }
            other => panic!("the second block is the call, not {other:?}"),
        }
        match &request.messages[2].content[0] {
            Block::ToolResult(result) => {
                assert_eq!(result.tool_use_id, "c1");
                assert_eq!(result.content[0], Block::Text("sunny".to_owned()));
            }
            other => panic!("the result, not {other:?}"),
        }
    }

    /// A stored transcript is spliced in front of what the caller sent, and the
    /// splice is of neutral turns rather than of a re-parsed body.
    #[test]
    fn a_prior_transcript_is_prepended_to_what_the_caller_sent() {
        let body: ResponsesBody = serde_json::from_value(serde_json::json!({"input": "and then?"}))
            .expect("the body parses");
        let request = to_ir(
            &body,
            "a-model",
            vec![Message::user("hello"), Message::assistant("hi")],
        )
        .expect("it converts");
        assert_eq!(request.messages.len(), 3);
        assert_eq!(request.messages[0].text(), "hello");
        assert_eq!(request.messages[2].text(), "and then?");
    }

    #[test]
    fn a_hosted_tool_is_refused_by_name_so_the_caller_knows_which_one() {
        let body: ResponsesBody = serde_json::from_value(serde_json::json!({
            "input": "hi",
            "tools": [{"type": "function", "name": "f"},
                      {"type": "web_search_preview"}]
        }))
        .expect("the body parses");
        let error = to_ir(&body, "a-model", Vec::new()).expect_err("a hosted tool");
        assert!(error.to_string().contains("web_search_preview"), "{error}");
        assert_eq!(error.code(), "unsupported_value");
    }

    #[test]
    fn a_function_tool_parses_flat_or_nested() {
        let flat = ir(serde_json::json!({
            "input": "hi",
            "tools": [{"type": "function", "name": "f", "description": "does f",
                       "parameters": {"type": "object"}}]
        }));
        assert_eq!(flat.tools[0].name, "f");
        assert_eq!(flat.tools[0].description.as_deref(), Some("does f"));

        let nested = ir(serde_json::json!({
            "input": "hi",
            "tools": [{"type": "function", "function": {"name": "g"}}]
        }));
        assert_eq!(nested.tools[0].name, "g");
    }

    #[test]
    fn the_tool_choice_vocabulary_lands_on_the_four_things_it_can_mean() {
        let choice = |value: serde_json::Value| {
            ir(serde_json::json!({"input": "hi", "tool_choice": value})).tool_choice
        };
        assert_eq!(choice(serde_json::json!("none")), Some(ToolChoice::None));
        assert_eq!(choice(serde_json::json!("required")), Some(ToolChoice::Any));
        assert_eq!(
            choice(serde_json::json!({"type": "function", "name": "f"})),
            Some(ToolChoice::Named("f".to_owned()))
        );
    }

    #[test]
    fn an_effort_rung_this_server_does_not_know_is_refused_rather_than_rounded() {
        let body: ResponsesBody = serde_json::from_value(serde_json::json!({
            "input": "hi", "reasoning": {"effort": "enormous"}
        }))
        .expect("the body parses");
        let error = to_ir(&body, "a-model", Vec::new()).expect_err("an unknown rung");
        assert!(error.to_string().contains("enormous"), "{error}");
    }

    #[test]
    fn turning_thinking_off_is_spelled_as_a_rung_that_is_not_a_rung() {
        let request = ir(serde_json::json!({
            "input": "hi", "reasoning": {"effort": "none"}
        }));
        assert_eq!(request.thinking, ThinkingDirective::Off);
    }

    #[test]
    fn an_inline_image_part_is_classified_and_a_remote_one_is_kept_apart() {
        let request = ir(serde_json::json!({"input": [{"role": "user", "content": [
            {"type": "input_text", "text": "what is this?"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ]}]}));
        assert!(matches!(
            request.messages[0].content[1],
            Block::Image(shiso_ir::MediaSource::Inline(_))
        ));
    }

    #[test]
    fn the_token_cap_comes_from_this_surfaces_own_spelling() {
        let request = ir(serde_json::json!({"input": "hi", "max_output_tokens": 64}));
        assert_eq!(request.max_tokens, 64);
    }
}
