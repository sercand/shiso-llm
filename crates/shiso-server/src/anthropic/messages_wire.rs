//! The Messages request, exactly as a client spells it.
//!
//! # Why there is so little to translate
//!
//! The neutral request was shaped after this format, so most of what would
//! normally be an adapter's work has already been done: a turn is an ordered
//! list of content blocks, a tool result is a block rather than a speaker,
//! instructions sit beside the conversation rather than inside it, and the only
//! two roles are the two the IR has. What remains here is the envelope - the
//! source object around an image, the discriminated union around a tool choice,
//! the field that is either a string or a list of blocks - and none of those are
//! facts about a conversation.
//!
//! # Why `max_tokens` has no default
//!
//! This format makes the cap mandatory, and it is the one field where being
//! forgiving would change what the server does: a missing cap that silently
//! became "whatever is left of the context" is a request that can run for
//! thousands of tokens the caller never budgeted for. Leaving the field without
//! `#[serde(default)]` is what turns its absence into a parse failure, and a
//! parse failure on this route is a 400 that names the field.

use serde::Deserialize;
use serde_json::Value;

use crate::error::AppError;

/// The most messages one request may carry.
///
/// The same bound the chat surface uses, and for the same reason: the body
/// limit caps the bytes, and a body of ten thousand one-character turns is
/// cheap to send and expensive to render.
pub const MAX_MESSAGES: usize = 2048;

/// Who sent one turn.
///
/// Two variants, because this format has two and so does the conversation. An
/// unknown role is refused by the parser rather than folded into `user`, which
/// is what the shape this replaced did - and folding an unknown role into the
/// caller's own turn puts words in the caller's mouth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WireRole {
    /// The caller. Tool results ride here.
    User,
    /// The model.
    Assistant,
}

/// Where a media block's bytes are.
///
/// `base64` carries them inline; `url` names an address the server would have
/// to fetch, and whether it may is the renderer's policy rather than this
/// parser's. A source this server cannot reach at all parses into
/// [`Self::Unsupported`] so the refusal can be a 400 about media instead of a
/// parse error about a discriminant.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Source {
    /// Bytes in the request.
    Base64 {
        /// What the bytes are, as the caller declared them.
        media_type: String,
        /// The bytes, base64 encoded.
        data: String,
    },
    /// An address.
    Url {
        /// Where the bytes are.
        url: String,
    },
    /// A source this server does not serve - the file store, for one, which
    /// this server has no equivalent of.
    #[serde(other)]
    Unsupported,
}

/// What a tool result says: one string, or blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolOutput {
    /// The whole result as one string.
    Text(String),
    /// The result as blocks, which is how a tool answers with a picture.
    Blocks(Vec<WireBlock>),
}

/// One piece of a turn.
///
/// Every variant becomes one [`shiso_ir::Block`], in place, with no reordering
/// and nothing lifted out into a field beside the list.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireBlock {
    /// Prose.
    Text {
        /// The words.
        text: String,
    },
    /// A still image.
    Image {
        /// Where the bytes are.
        source: Source,
    },
    /// A reasoning trace being replayed into a later turn.
    Thinking {
        /// The trace.
        thinking: String,
        /// The signature that was issued with it, when one was.
        #[serde(default)]
        signature: Option<String>,
    },
    /// A reasoning trace whose text is withheld.
    ///
    /// The payload is opaque to everything but the service that encrypted it,
    /// so there is nothing here that could be shown to a model.
    RedactedThinking {
        /// The opaque payload.
        #[serde(default)]
        data: String,
    },
    /// A call the model asked for, replayed into a later turn.
    ToolUse {
        /// The identifier the matching result quotes back.
        id: String,
        /// The tool that was called.
        name: String,
        /// The arguments, already parsed - this format sends them as JSON
        /// rather than as a string holding JSON, which is one parse this
        /// surface does not have to do.
        #[serde(default)]
        input: Value,
    },
    /// The answer to one call.
    ToolResult {
        /// The call this answers.
        tool_use_id: String,
        /// What the tool produced.
        #[serde(default)]
        content: Option<ToolOutput>,
        /// Whether the tool failed. A field rather than a marker inside the
        /// text, here and in the IR both, so no reader has to know a
        /// convention in order to see a failure.
        #[serde(default)]
        is_error: bool,
    },
    /// A block of a kind this server does not serve.
    ///
    /// Present so the body still parses and the refusal can say which block was
    /// at fault. Dropping it instead would answer confidently about content the
    /// model was never shown.
    #[serde(other)]
    Unsupported,
}

/// A turn's content: one string, or blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// The whole turn as one string.
    Text(String),
    /// The turn as an ordered list of blocks.
    Blocks(Vec<WireBlock>),
}

/// One turn of the conversation.
#[derive(Debug, Clone, Deserialize)]
pub struct WireMessage {
    /// Who sent it.
    pub role: WireRole,
    /// What it says.
    pub content: Content,
}

/// The standing instructions: one string, or blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    /// The instructions as one string.
    Text(String),
    /// The instructions as blocks, which is how a client that caches part of
    /// them spells it.
    Blocks(Vec<WireBlock>),
}

/// One tool on offer.
///
/// `input_schema` is optional here although the format documents it as
/// required, because this server also has to accept the server-tool spellings
/// that carry a `type` and no schema. A tool with no schema takes no arguments,
/// which is a thing a template can render.
#[derive(Debug, Clone, Deserialize)]
pub struct WireTool {
    /// The discriminant, which a custom tool omits. Ignored.
    #[serde(rename = "type", default)]
    pub tool_type: Option<String>,
    /// The name the model emits to call it.
    pub name: String,
    /// What it does, shown to the model.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the arguments.
    #[serde(default)]
    pub input_schema: Option<Value>,
}

/// How the caller wants the offered tools used.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireToolChoice {
    /// The model decides.
    Auto,
    /// The model must call at least one, its pick.
    Any,
    /// The model must call this one.
    Tool {
        /// The tool that must be called.
        name: String,
    },
    /// The model must not call a tool.
    None,
}

/// What the caller said about thinking.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireThinking {
    /// Thinking is on, with a budget when the caller gave one.
    Enabled {
        /// The budget in tokens.
        #[serde(default)]
        budget_tokens: Option<u32>,
    },
    /// Thinking is off.
    Disabled,
}

/// The Messages request body.
///
/// Unknown keys are accepted and ignored, for the same reason the chat surface
/// accepts them: a client library that sends a field this server has never
/// heard of is not making a mistake a 400 would help with. That tolerance is
/// for *keys*, not for content - a content block of an unknown kind is refused,
/// because a key nobody reads changes nothing and a block nobody reads changes
/// what the model was asked.
#[derive(Debug, Clone, Deserialize)]
pub struct MessagesBody {
    /// The model the caller named. Echoed rather than honoured: one process
    /// serves one checkpoint.
    #[serde(default)]
    pub model: Option<String>,
    /// The conversation.
    #[serde(default)]
    pub messages: Vec<WireMessage>,
    /// Cap on generated tokens. Mandatory in this format; see the module docs.
    pub max_tokens: u32,
    /// Standing instructions.
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    /// Sequences that end generation.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Stream the answer.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Softmax temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Cumulative-probability cutoff.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Candidates kept by the top-k stage.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Tools on offer.
    #[serde(default)]
    pub tools: Vec<WireTool>,
    /// How the caller wants them used.
    #[serde(default)]
    pub tool_choice: Option<WireToolChoice>,
    /// What the caller said about thinking.
    #[serde(default)]
    pub thinking: Option<WireThinking>,
}

impl MessagesBody {
    /// Whether the caller asked for a stream.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    /// Checks everything that can be checked without rendering a prompt.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming the field at fault, in the order a
    /// caller would fix them: the conversation first, then the cap, then the
    /// sampling knobs, then the tools.
    pub fn validate(&self) -> Result<(), AppError> {
        if self.messages.is_empty() {
            return Err(AppError::bad_request(
                "messages must not be empty",
                Some("messages"),
                "invalid_request",
            ));
        }
        if self.messages.len() > MAX_MESSAGES {
            return Err(AppError::bad_request(
                format!("messages must not hold more than {MAX_MESSAGES} turns"),
                Some("messages"),
                "invalid_request",
            ));
        }
        if self.max_tokens == 0 {
            return Err(AppError::bad_request(
                "max_tokens must be at least 1",
                Some("max_tokens"),
                "invalid_request",
            ));
        }
        // Zero to one, which is the range this format defines, rather than the
        // zero to two the chat family allows. A caller that sends 1.5 here has
        // sent a value this format has no meaning for, and clamping it would
        // serve a different distribution than the one asked for without saying
        // so.
        validate_range("temperature", self.temperature, 0.0, 1.0)?;
        if let Some(top_p) = self.top_p
            && !(top_p > 0.0 && top_p <= 1.0)
        {
            return Err(AppError::bad_request(
                "top_p must be greater than 0 and at most 1",
                Some("top_p"),
                "invalid_request",
            ));
        }
        self.validate_tool_choice()
    }

    /// The one tool-choice combination that means nothing: a forced call with
    /// nothing on offer to call.
    fn validate_tool_choice(&self) -> Result<(), AppError> {
        let forced = match self.tool_choice.as_ref() {
            Some(WireToolChoice::Any) => true,
            Some(WireToolChoice::Tool { name }) => {
                if !self.tools.iter().any(|tool| tool.name == *name) {
                    return Err(AppError::bad_request(
                        format!(
                            "tool_choice names {name:?}, which is not one of the tools offered"
                        ),
                        Some("tool_choice"),
                        "invalid_request",
                    ));
                }
                true
            }
            _ => false,
        };
        if forced && self.tools.is_empty() {
            return Err(AppError::bad_request(
                "tool_choice requires a tool call, and no tools were offered",
                Some("tool_choice"),
                "invalid_request",
            ));
        }
        Ok(())
    }
}

/// Refuses a knob that is outside the range its stage is defined on.
fn validate_range(
    field: &'static str,
    value: Option<f32>,
    low: f32,
    high: f32,
) -> Result<(), AppError> {
    let Some(value) = value else { return Ok(()) };
    if value.is_nan() || value < low || value > high {
        return Err(AppError::bad_request(
            format!("{field} must be between {low} and {high}"),
            Some(field),
            "invalid_request",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Content, MessagesBody, Source, SystemPrompt, WireBlock, WireRole, WireThinking};
    use serde_json::json;

    fn body(json: serde_json::Value) -> MessagesBody {
        serde_json::from_value(json).expect("the body parses")
    }

    fn minimal() -> serde_json::Value {
        json!({
            "model": "a-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        })
    }

    /// The cap is what stands between a request and a run the caller never
    /// budgeted for, so its absence is a parse failure rather than a default.
    #[test]
    fn a_request_without_a_token_cap_does_not_parse_at_all() {
        let error = serde_json::from_value::<MessagesBody>(json!({
            "model": "a-model",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .expect_err("max_tokens is mandatory");
        assert!(error.to_string().contains("max_tokens"), "{error}");
    }

    #[test]
    fn a_turn_is_either_a_string_or_a_list_of_blocks_and_both_parse() {
        assert!(matches!(
            body(minimal()).messages[0].content,
            Content::Text(ref text) if text == "hi"
        ));
        let mut json = minimal();
        json["messages"][0]["content"] = json!([
            {"type": "text", "text": "look"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QQ=="}},
        ]);
        let parsed = body(json);
        let Content::Blocks(ref blocks) = parsed.messages[0].content else {
            panic!("a list of blocks parses as blocks");
        };
        assert!(matches!(blocks[0], WireBlock::Text { .. }));
        assert!(matches!(
            blocks[1],
            WireBlock::Image {
                source: Source::Base64 { .. }
            }
        ));
    }

    /// A role this format does not have must not be folded into the caller's
    /// own turn, which is what putting words in someone's mouth looks like in a
    /// parser.
    #[test]
    fn a_role_this_format_does_not_have_is_refused_by_the_parser() {
        let mut json = minimal();
        json["messages"][0]["role"] = json!("system");
        let error = serde_json::from_value::<MessagesBody>(json).expect_err("no system role");
        assert!(error.to_string().contains("system"), "{error}");
    }

    #[test]
    fn instructions_parse_from_one_string_or_from_blocks() {
        let mut json = minimal();
        json["system"] = json!("be brief");
        assert!(matches!(body(json).system, Some(SystemPrompt::Text(_))));

        let mut json = minimal();
        json["system"] = json!([{"type": "text", "text": "be brief"}]);
        assert!(matches!(body(json).system, Some(SystemPrompt::Blocks(_))));
    }

    /// A block kind this server has no rendering for parses into its own
    /// variant so the refusal can be about the block. The alternative - a parse
    /// failure - would blame the whole body for one block.
    #[test]
    fn a_block_kind_this_server_does_not_serve_parses_as_unsupported() {
        let mut json = minimal();
        json["messages"][0]["content"] = json!([{"type": "container_upload", "file_id": "f_1"}]);
        let parsed = body(json);
        let Content::Blocks(ref blocks) = parsed.messages[0].content else {
            panic!("a list of blocks parses as blocks");
        };
        assert!(matches!(blocks[0], WireBlock::Unsupported));
    }

    #[test]
    fn a_thinking_directive_parses_with_and_without_a_budget() {
        let mut json = minimal();
        json["thinking"] = json!({"type": "enabled", "budget_tokens": 2048});
        assert!(matches!(
            body(json).thinking,
            Some(WireThinking::Enabled {
                budget_tokens: Some(2048)
            })
        ));

        let mut json = minimal();
        json["thinking"] = json!({"type": "disabled"});
        assert!(matches!(body(json).thinking, Some(WireThinking::Disabled)));
    }

    #[test]
    fn an_empty_conversation_is_refused_before_anything_is_rendered() {
        let mut json = minimal();
        json["messages"] = json!([]);
        let error = body(json).validate().expect_err("an empty conversation");
        assert_eq!(error.code(), "invalid_request");
        assert!(error.to_string().contains("must not be empty"), "{error}");
    }

    #[test]
    fn a_cap_of_zero_is_refused_rather_than_run_as_no_cap() {
        let mut json = minimal();
        json["max_tokens"] = json!(0);
        let error = body(json).validate().expect_err("a cap of zero");
        assert!(error.to_string().contains("max_tokens"), "{error}");
    }

    #[test]
    fn the_sampling_ranges_are_the_ones_this_format_defines() {
        for (field, value) in [("temperature", 1.5_f32), ("top_p", 0.0)] {
            let mut json = minimal();
            json[field] = json!(value);
            let error = body(json).validate().expect_err("out of range");
            assert!(error.to_string().contains(field), "{error}");
        }
        let mut json = minimal();
        json["temperature"] = json!(1.0);
        assert!(
            body(json).validate().is_ok(),
            "the top of the range is in it"
        );
    }

    #[test]
    fn a_forced_tool_choice_with_nothing_to_call_is_refused() {
        let mut json = minimal();
        json["tool_choice"] = json!({"type": "any"});
        assert!(body(json).validate().is_err());

        let mut json = minimal();
        json["tool_choice"] = json!({"type": "tool", "name": "lookup"});
        json["tools"] = json!([{"name": "other", "input_schema": {"type": "object"}}]);
        let error = body(json).validate().expect_err("a name nothing offers");
        assert!(error.to_string().contains("lookup"), "{error}");

        let mut json = minimal();
        json["tool_choice"] = json!({"type": "tool", "name": "lookup"});
        json["tools"] = json!([{"name": "lookup", "input_schema": {"type": "object"}}]);
        assert!(body(json).validate().is_ok());
    }

    #[test]
    fn a_field_this_server_does_not_know_is_ignored_rather_than_refused() {
        let mut json = minimal();
        json["metadata"] = json!({"user_id": "u_1"});
        json["service_tier"] = json!("auto");
        assert_eq!(body(json).messages[0].role, WireRole::User);
    }
}
