//! The Responses request, exactly as a client spells it.
//!
//! # Why `input` is four shapes and not one
//!
//! A caller may send a bare string, a list of turns, a list of typed items, or a
//! mixture - and a function call and its output are items in that same list
//! rather than turns with a special role. Every one of those is a different
//! thing to map, so every one of them is a variant here and the ambiguity is
//! gone before anything downstream reads it.
//!
//! # Why the tool type is a string
//!
//! This surface carries hosted tools this server cannot run - a web search, a
//! file search, a sandboxed interpreter. Refusing them has to name the one that
//! was asked for, because a caller offering four tools needs to know which of
//! them is the problem. A closed enum would either have to list every hosted
//! tool that will ever exist or lose the name on the way to the refusal.

use serde::Deserialize;
use serde_json::Value;

use crate::error::AppError;
use crate::openai::chat_wire::{MediaRef, ServiceTier, StopField, StreamOptions};

/// Who sent one input turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputRole {
    /// Standing instructions.
    System,
    /// Standing instructions, under the newer spelling.
    Developer,
    /// The caller.
    User,
    /// The model.
    Assistant,
}

/// One piece of a turn's content.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Prose the caller sent.
    InputText {
        /// The words.
        text: String,
    },
    /// Prose the model produced, replayed.
    OutputText {
        /// The words.
        text: String,
    },
    /// A still image.
    InputImage {
        /// Where the bytes are.
        image_url: MediaRef,
    },
    /// A video.
    InputVideo {
        /// Where the bytes are.
        video_url: MediaRef,
    },
    /// Output a safety decision withheld.
    Refusal {
        /// What is shown instead.
        refusal: String,
    },
}

/// A turn's content: one string, or parts.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum InputContent {
    /// The whole turn as one string.
    Text(String),
    /// The turn as an ordered list of parts.
    Parts(Vec<ContentPart>),
}

/// A turn with no `type` key, which is the shortest thing a client can send.
#[derive(Debug, Clone, Deserialize)]
pub struct BareMessage {
    /// Who sent it.
    pub role: InputRole,
    /// What it says.
    pub content: InputContent,
}

/// One item of a typed input list.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TypedItem {
    /// A turn.
    Message {
        /// Who sent it.
        role: InputRole,
        /// What it says.
        content: InputContent,
    },
    /// A call the model made in an earlier turn, replayed.
    FunctionCall {
        /// The identifier the output quotes back.
        #[serde(default)]
        call_id: Option<String>,
        /// The tool that was called.
        name: String,
        /// The arguments, as the JSON text the model wrote.
        #[serde(default)]
        arguments: Option<String>,
    },
    /// The answer to one call.
    FunctionCallOutput {
        /// The call this answers.
        call_id: String,
        /// What the tool produced.
        output: Value,
    },
    /// A reasoning item replayed from an earlier turn.
    Reasoning {
        /// The summary pieces, when the caller kept them.
        #[serde(default)]
        summary: Vec<SummaryPart>,
    },
}

/// One piece of a replayed reasoning summary.
#[derive(Debug, Clone, Deserialize)]
pub struct SummaryPart {
    /// The words.
    #[serde(default)]
    pub text: Option<String>,
}

/// One item of the input list, typed or bare.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    /// An item that named its own kind.
    Typed(TypedItem),
    /// A turn that did not.
    Bare(BareMessage),
}

/// The `input` field.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum InputField {
    /// A single caller turn, spelled as one string.
    Text(String),
    /// An ordered list of items.
    Items(Vec<InputItem>),
}

/// One tool on offer.
///
/// Flat rather than an enum over the hosted kinds, so that a refusal can name
/// the kind the caller actually asked for.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesTool {
    /// The kind. Only `function` is one this server can run.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The tool's name, when it is a function declared flat.
    #[serde(default)]
    pub name: Option<String>,
    /// What it does.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the arguments.
    #[serde(default)]
    pub parameters: Option<Value>,
    /// The nested spelling some clients still send.
    #[serde(default)]
    pub function: Option<crate::openai::chat_wire::WireFunction>,
}

/// How the caller wants tools used.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    /// One of `none`, `auto` or `required`.
    Mode(String),
    /// A specific tool, which this surface names flat.
    Named {
        /// The discriminant, ignored.
        #[serde(rename = "type", default)]
        choice_type: Option<String>,
        /// The tool that must be called.
        name: String,
    },
}

/// How hard the model should think.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ReasoningConfig {
    /// The rung the caller asked for.
    #[serde(default)]
    pub effort: Option<String>,
    /// Whether a summary is wanted, and how detailed. Recorded and not acted
    /// on: this server streams the trace itself rather than a summary of it.
    #[serde(default)]
    pub summary: Option<String>,
}

/// The output-shape block.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TextConfig {
    /// The format the answer has to be in.
    #[serde(default)]
    pub format: Option<Value>,
}

/// The Responses request body.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ResponsesBody {
    /// The model the caller named, echoed rather than honoured.
    #[serde(default)]
    pub model: Option<String>,
    /// What to answer.
    #[serde(default)]
    pub input: Option<InputField>,
    /// Standing instructions for this request.
    #[serde(default)]
    pub instructions: Option<String>,
    /// The response whose transcript this one continues.
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// Whether this response can be fetched again later.
    #[serde(default)]
    pub store: Option<bool>,
    /// Stream the answer.
    #[serde(default)]
    pub stream: Option<bool>,
    /// What a streaming caller wants beyond the content.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// The token cap for the generated turn.
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Softmax temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Cumulative-probability cutoff.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Candidates kept by the top-k stage.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Sequences that end generation.
    #[serde(default)]
    pub stop: Option<StopField>,
    /// Tools on offer.
    #[serde(default)]
    pub tools: Vec<ResponsesTool>,
    /// How the caller wants them used.
    #[serde(default)]
    pub tool_choice: Option<ResponsesToolChoice>,
    /// How hard the model should think.
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    /// What shape the answer has to be.
    #[serde(default)]
    pub text: Option<TextConfig>,
    /// Seed for reproducible sampling.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Opaque key-value pairs, echoed back untouched.
    #[serde(default)]
    pub metadata: Option<Value>,
    /// The service class asked for.
    #[serde(default)]
    pub service_tier: Option<ServiceTier>,
}

impl ResponsesBody {
    /// Whether the caller asked for the accounting on the stream.
    #[must_use]
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .and_then(|options| options.include_usage)
            .unwrap_or(false)
    }

    /// Checks what can be checked before anything is resolved.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming the field at fault. An empty request -
    /// no input and nothing to continue - is refused here rather than reaching
    /// a template that would raise about an empty conversation.
    pub fn validate(&self) -> Result<(), AppError> {
        if self.input.is_none() && self.previous_response_id.is_none() {
            return Err(AppError::bad_request(
                "input is required unless previous_response_id names a conversation \
                 to continue",
                Some("input"),
                "invalid_request",
            ));
        }
        if self.max_output_tokens == Some(0) {
            return Err(AppError::bad_request(
                "max_output_tokens must be at least 1",
                Some("max_output_tokens"),
                "invalid_request",
            ));
        }
        if let Some(temperature) = self.temperature
            && (temperature.is_nan() || !(0.0..=2.0).contains(&temperature))
        {
            return Err(AppError::bad_request(
                "temperature must be between 0 and 2",
                Some("temperature"),
                "invalid_request",
            ));
        }
        if let Some(top_p) = self.top_p
            && !(top_p > 0.0 && top_p <= 1.0)
        {
            return Err(AppError::bad_request(
                "top_p must be greater than 0 and at most 1",
                Some("top_p"),
                "invalid_request",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{InputField, InputItem, ResponsesBody, TypedItem};

    fn body(json: serde_json::Value) -> ResponsesBody {
        serde_json::from_value(json).expect("the body parses")
    }

    #[test]
    fn a_bare_string_input_is_the_shortest_request_a_client_can_send() {
        let parsed = body(serde_json::json!({"input": "hello"}));
        assert!(matches!(parsed.input, Some(InputField::Text(ref text)) if text == "hello"));
    }

    /// A turn may omit `type`, and a typed item may not - so both shapes have to
    /// parse out of the same list.
    #[test]
    fn an_item_list_parses_whether_or_not_each_item_names_its_kind() {
        let parsed = body(serde_json::json!({"input": [
            {"role": "user", "content": "hi"},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "hello"}]},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"}
        ]}));
        let Some(InputField::Items(items)) = parsed.input else {
            panic!("an item list");
        };
        assert_eq!(items.len(), 4);
        assert!(matches!(items[0], InputItem::Bare(_)));
        assert!(matches!(
            items[1],
            InputItem::Typed(TypedItem::Message { .. })
        ));
        assert!(matches!(
            items[2],
            InputItem::Typed(TypedItem::FunctionCall { .. })
        ));
        assert!(matches!(
            items[3],
            InputItem::Typed(TypedItem::FunctionCallOutput { .. })
        ));
    }

    #[test]
    fn a_request_with_nothing_to_answer_is_refused_before_anything_is_rendered() {
        let error = body(serde_json::json!({}))
            .validate()
            .expect_err("nothing to answer");
        assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    /// Continuing a stored conversation is a complete request on its own: the
    /// transcript is the input.
    #[test]
    fn naming_a_conversation_to_continue_is_enough_on_its_own() {
        let parsed = body(serde_json::json!({"previous_response_id": "resp_1"}));
        assert!(parsed.validate().is_ok());
    }

    #[test]
    fn the_sampling_ranges_are_checked_before_a_prompt_is_built() {
        let error = body(serde_json::json!({"input": "hi", "temperature": 9.0}))
            .validate()
            .expect_err("out of range");
        assert!(error.to_string().contains("temperature"), "{error}");
    }
}
