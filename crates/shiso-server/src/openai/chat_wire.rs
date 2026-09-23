//! The chat-completions request, exactly as a client spells it.
//!
//! # Why the wire types are here and not in the IR
//!
//! Nothing in this file is a fact about a conversation. The nested `function`
//! wrapper around a tool, the two spellings of a token cap, the sentinel that
//! turns off the top-k stage, the field that is either a string or a list of
//! them - each is a fact about this one format, and the neutral request is
//! better for not carrying any of them. The translation happens once, in
//! [`super::chat_to_ir`], and everything downstream reads the neutral shape.
//!
//! # Why a field that means two things becomes two variants
//!
//! `stop` is a string or a list. `content` is a string or a list of parts.
//! `tool_choice` is a mode word or an object naming a function. Parsing each
//! into an enum makes the ambiguity disappear at the edge; keeping them as raw
//! values would push "which is it this time" into every reader.

use serde::Deserialize;
use serde_json::Value;

use crate::error::AppError;

/// The most messages one request may carry.
///
/// A bound rather than no bound: the body limit already caps the bytes, but a
/// body of ten thousand one-character turns is cheap to send and expensive to
/// render, and a template that loops over all of them is the expensive part.
pub const MAX_MESSAGES: usize = 2048;

/// The most alternatives a caller may ask for.
pub const MAX_N: u32 = 128;

/// Who sent one turn, as this format spells it.
///
/// Closed, so an unknown role is refused at the edge by the parser rather than
/// reaching a renderer that would have to guess. `function` is the retired
/// spelling of `tool` and is accepted because clients still send it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WireRole {
    /// Standing instructions.
    System,
    /// Standing instructions, under the newer spelling.
    Developer,
    /// The caller.
    User,
    /// The model.
    Assistant,
    /// The answer to a tool call.
    Tool,
    /// The retired spelling of [`Self::Tool`].
    Function,
}

/// Where a media part's bytes are.
///
/// Either an object with a `url` key or the bare string, because both are sent
/// in the wild and the difference means nothing.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MediaRef {
    /// `{"url": "..."}`.
    Wrapped {
        /// The address or the inline payload.
        url: String,
    },
    /// The address or the inline payload, unwrapped.
    Bare(String),
}

impl MediaRef {
    /// The address or payload, however it was spelled.
    #[must_use]
    pub fn url(&self) -> &str {
        match self {
            Self::Wrapped { url } | Self::Bare(url) => url,
        }
    }
}

/// One piece of a structured turn.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    /// Prose.
    Text {
        /// The words.
        text: String,
    },
    /// A still image.
    ImageUrl {
        /// Where the bytes are.
        image_url: MediaRef,
    },
    /// A video.
    VideoUrl {
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
pub enum Content {
    /// The whole turn as one string.
    Text(String),
    /// The turn as an ordered list of parts.
    Parts(Vec<Part>),
}

/// The function half of a tool call.
#[derive(Debug, Clone, Deserialize)]
pub struct WireFunctionCall {
    /// The tool being called.
    #[serde(default)]
    pub name: Option<String>,
    /// The arguments, as the JSON text the model wrote.
    #[serde(default)]
    pub arguments: Option<String>,
}

/// One call an assistant turn made.
#[derive(Debug, Clone, Deserialize)]
pub struct WireToolCall {
    /// The identifier the matching result quotes back.
    #[serde(default)]
    pub id: Option<String>,
    /// The discriminant, which is always `function` today and is ignored.
    #[serde(rename = "type", default)]
    pub call_type: Option<String>,
    /// What was called, and with what.
    pub function: WireFunctionCall,
}

/// One turn of the conversation.
#[derive(Debug, Clone, Deserialize)]
pub struct WireMessage {
    /// Who sent it.
    pub role: WireRole,
    /// What it says. Absent is legal on an assistant turn that only calls tools.
    #[serde(default)]
    pub content: Option<Content>,
    /// A name for the speaker, which this server does not render.
    #[serde(default)]
    pub name: Option<String>,
    /// Calls this assistant turn made.
    #[serde(default)]
    pub tool_calls: Vec<WireToolCall>,
    /// The call a tool turn answers.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// A reasoning trace, under the spelling this family uses.
    #[serde(default)]
    pub reasoning_content: Option<String>,
    /// A reasoning trace, under the older spelling.
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Output a safety decision withheld.
    #[serde(default)]
    pub refusal: Option<String>,
}

impl WireMessage {
    /// The reasoning trace, whichever key carried it.
    #[must_use]
    pub fn trace(&self) -> Option<&str> {
        self.reasoning_content
            .as_deref()
            .or(self.reasoning.as_deref())
    }
}

/// The function half of a tool offer.
#[derive(Debug, Clone, Deserialize)]
pub struct WireFunction {
    /// The name the model emits to call it.
    pub name: String,
    /// What it does, shown to the model.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the arguments.
    #[serde(default)]
    pub parameters: Option<Value>,
}

/// One tool on offer.
#[derive(Debug, Clone, Deserialize)]
pub struct WireTool {
    /// The discriminant, which is always `function` on this surface.
    #[serde(rename = "type", default)]
    pub tool_type: Option<String>,
    /// The offer itself.
    pub function: WireFunction,
}

/// The function a forced tool choice names.
#[derive(Debug, Clone, Deserialize)]
pub struct NamedFunction {
    /// The tool that must be called.
    pub name: String,
}

/// How the caller wants tools used.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum WireToolChoice {
    /// One of `none`, `auto` or `required`.
    Mode(String),
    /// A specific tool.
    Named {
        /// The discriminant, ignored.
        #[serde(rename = "type", default)]
        choice_type: Option<String>,
        /// The tool that must be called.
        function: NamedFunction,
    },
}

/// The schema half of a structured-output request.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonSchemaSpec {
    /// The schema's name, for logs and errors.
    #[serde(default)]
    pub name: Option<String>,
    /// The schema itself.
    #[serde(default)]
    pub schema: Option<Value>,
    /// Whether anything the schema does not allow must be refused.
    #[serde(default)]
    pub strict: Option<bool>,
}

/// What shape the answer has to be.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireResponseFormat {
    /// Unconstrained prose.
    Text,
    /// Valid JSON, any shape.
    JsonObject,
    /// JSON matching a schema.
    JsonSchema {
        /// The schema and its options.
        json_schema: JsonSchemaSpec,
    },
}

/// What a streaming caller wants beyond the content.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct StreamOptions {
    /// Whether the stream ends with the accounting.
    #[serde(default)]
    pub include_usage: Option<bool>,
}

/// Knobs a chat template reads that have no field of their own.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct ChatTemplateKwargs {
    /// Whether the model should open a thinking block.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// How many tokens the thinking block may use.
    #[serde(default)]
    pub thinking_budget: Option<u32>,
    /// Whether historical thinking blocks survive a re-render. The same knob as
    /// the body's own `preserve_thinking`, spelled where clients that pass
    /// template variables through put it.
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
}

/// A stop field: one sequence, or several.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    /// One sequence.
    One(String),
    /// Several.
    Many(Vec<String>),
}

impl StopField {
    /// The sequences, however they were spelled.
    #[must_use]
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(one) => vec![one],
            Self::Many(many) => many,
        }
    }
}

/// The service class a caller asked to be served in.
///
/// Parsed rather than echoed as an opaque value, because a mistyped tier is a
/// caller that thinks it bought something it did not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceTier {
    /// Whatever the server would do anyway.
    Auto,
    /// The ordinary class, named explicitly.
    Default,
    /// The class that trades latency for capacity.
    Flex,
}

impl ServiceTier {
    /// Parses the wire vocabulary, case insensitively.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "default" => Some(Self::Default),
            "flex" => Some(Self::Flex),
            _ => None,
        }
    }

    /// The spelling a response echoes.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Default => "default",
            Self::Flex => "flex",
        }
    }
}

impl<'de> Deserialize<'de> for ServiceTier {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "service_tier must be one of \"auto\", \"default\" or \"flex\", not {raw:?}"
            ))
        })
    }
}

/// The chat-completions request body.
///
/// Unknown keys are accepted and ignored. A client library that sends a field
/// this server has never heard of is not making a mistake a 400 would help with,
/// and refusing them would make every new field in the format a breaking change
/// for every client at once.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatBody {
    /// The model the caller named. Ignored beyond being echoed: one process
    /// serves one checkpoint.
    #[serde(default)]
    pub model: Option<String>,
    /// The conversation.
    #[serde(default)]
    pub messages: Vec<WireMessage>,
    /// Softmax temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Cumulative-probability cutoff.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Candidates kept by the top-k stage. Zero turns the stage off.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Floor on a token's probability.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Cutoff in standard deviations of the logit distribution.
    #[serde(default)]
    pub top_n_sigma: Option<f32>,
    /// Additive penalty for a token having appeared at all.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Additive penalty scaling with how often a token has appeared.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Multiplicative penalty on tokens already in the context.
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// How many answers the caller wants.
    #[serde(default)]
    pub n: Option<u32>,
    /// The retired spelling of the token cap.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// The current spelling of the token cap.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    /// Floor on generated tokens.
    #[serde(default)]
    pub min_tokens: Option<u32>,
    /// Sequences that end generation.
    #[serde(default)]
    pub stop: Option<StopField>,
    /// Stream the answer.
    #[serde(default)]
    pub stream: Option<bool>,
    /// What a streaming caller wants beyond the content.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Tools on offer.
    #[serde(default)]
    pub tools: Vec<WireTool>,
    /// How the caller wants them used.
    #[serde(default)]
    pub tool_choice: Option<WireToolChoice>,
    /// What shape the answer has to be.
    #[serde(default)]
    pub response_format: Option<WireResponseFormat>,
    /// How hard the model should think.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Whether thinking is enabled.
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    /// Template knobs with no field of their own.
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Whether historical thinking blocks survive a re-render.
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
    /// Seed for reproducible sampling.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Additive bias per token id.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// Whether per-token probabilities are wanted.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// How many alternatives per token.
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    /// Report the sampled ids beside the text.
    #[serde(default)]
    pub return_token_ids: Option<bool>,
    /// The service class asked for.
    #[serde(default)]
    pub service_tier: Option<ServiceTier>,
    /// Whether the answer can be fetched again later.
    #[serde(default)]
    pub store: Option<bool>,
}

impl ChatBody {
    /// The token cap, under either spelling.
    ///
    /// The newer field wins when both are present, because a client that sends
    /// both is a client that was upgraded and still carries the old default.
    #[must_use]
    pub fn token_cap(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    /// Whether the caller asked for the accounting on the stream.
    #[must_use]
    pub fn include_usage(&self) -> bool {
        self.stream_options
            .and_then(|options| options.include_usage)
            .unwrap_or(false)
    }

    /// Checks everything that can be checked without rendering a prompt.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming the field at fault. The order is the
    /// order a caller would fix them in - the conversation first, then the
    /// sampling knobs, then the tools - so that a request with two mistakes
    /// reports the one nearer the top of the body.
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
        validate_range("temperature", self.temperature, 0.0, 2.0)?;
        if let Some(top_p) = self.top_p
            && !(top_p > 0.0 && top_p <= 1.0)
        {
            return Err(AppError::bad_request(
                "top_p must be greater than 0 and at most 1",
                Some("top_p"),
                "invalid_request",
            ));
        }
        validate_range("min_p", self.min_p, 0.0, 1.0)?;
        if self.token_cap() == Some(0) {
            return Err(AppError::bad_request(
                "max_completion_tokens must be at least 1",
                Some("max_completion_tokens"),
                "invalid_request",
            ));
        }
        self.validate_alternatives()?;
        self.validate_tool_choice()?;
        if let Some(top_logprobs) = self.top_logprobs
            && top_logprobs > 20
        {
            return Err(AppError::bad_request(
                "top_logprobs must be between 0 and 20",
                Some("top_logprobs"),
                "invalid_request",
            ));
        }
        Ok(())
    }

    /// The `n` field, which this server can only answer with one.
    ///
    /// # Why more than one is refused rather than clamped
    ///
    /// A neutral response carries one generated turn and there is no list of
    /// candidates to index, because a list is a question - "which one?" - that
    /// every consumer answers with "the first". Clamping would hand a caller
    /// that asked for five one answer in a five-shaped envelope and say nothing,
    /// and a caller sampling five and picking the best would silently get the
    /// only one it was given. The out-of-range check comes first so that a
    /// caller who sent nonsense is told it is nonsense rather than that it is
    /// unsupported.
    fn validate_alternatives(&self) -> Result<(), AppError> {
        let Some(n) = self.n else { return Ok(()) };
        if !(1..=MAX_N).contains(&n) {
            return Err(AppError::bad_request(
                format!("n must be between 1 and {MAX_N}"),
                Some("n"),
                "invalid_request",
            ));
        }
        if n > 1 {
            return Err(AppError::bad_request(
                "this server answers one request with one turn; send n concurrent \
                 requests instead of n > 1",
                Some("n"),
                "unsupported_value",
            ));
        }
        Ok(())
    }

    /// The tool-choice vocabulary, and the one combination that means nothing.
    fn validate_tool_choice(&self) -> Result<(), AppError> {
        let Some(choice) = self.tool_choice.as_ref() else {
            return Ok(());
        };
        if let WireToolChoice::Mode(mode) = choice {
            if !matches!(mode.as_str(), "none" | "auto" | "required") {
                return Err(AppError::bad_request(
                    "tool_choice must be \"none\", \"auto\", \"required\", or an \
                     object naming a function",
                    Some("tool_choice"),
                    "invalid_request",
                ));
            }
            if mode == "required" && self.tools.is_empty() {
                return Err(AppError::bad_request(
                    "tool_choice \"required\" needs at least one tool",
                    Some("tool_choice"),
                    "invalid_request",
                ));
            }
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
    use super::{ChatBody, Content, ServiceTier, StopField, WireRole, WireToolChoice};

    fn body(json: serde_json::Value) -> ChatBody {
        serde_json::from_value(json).expect("the body parses")
    }

    fn minimal() -> serde_json::Value {
        serde_json::json!({"messages": [{"role": "user", "content": "hi"}]})
    }

    #[test]
    fn a_turn_is_either_a_string_or_a_list_of_parts_and_both_parse() {
        let flat = body(minimal());
        assert!(matches!(
            flat.messages[0].content,
            Some(Content::Text(ref text)) if text == "hi"
        ));
        let structured = body(serde_json::json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        }));
        assert!(matches!(
            structured.messages[0].content,
            Some(Content::Parts(ref parts)) if parts.len() == 1
        ));
    }

    #[test]
    fn a_role_this_server_does_not_serve_is_refused_by_the_parser() {
        let error = serde_json::from_value::<ChatBody>(serde_json::json!({
            "messages": [{"role": "robot", "content": "hi"}]
        }))
        .expect_err("an unknown role does not parse");
        assert!(error.to_string().contains("robot"), "{error}");
    }

    #[test]
    fn a_stop_field_parses_from_one_sequence_or_from_several() {
        let one = body(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}], "stop": "END"
        }));
        assert_eq!(
            one.stop.expect("a stop field").into_vec(),
            vec!["END".to_owned()]
        );
        let many = body(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}], "stop": ["A", "B"]
        }));
        assert_eq!(many.stop.expect("a stop field").into_vec().len(), 2);
    }

    #[test]
    fn the_newer_token_cap_wins_when_a_client_sends_both() {
        let both = body(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10, "max_completion_tokens": 20
        }));
        assert_eq!(both.token_cap(), Some(20));
    }

    #[test]
    fn an_empty_conversation_is_refused_before_anything_is_rendered() {
        let empty = body(serde_json::json!({"messages": []}));
        let error = empty.validate().expect_err("an empty conversation");
        assert_eq!(error.code(), "invalid_request");
        assert!(error.to_string().contains("must not be empty"));
    }

    /// Out of range first, unsupported second: a caller who sent nonsense is
    /// told it is nonsense rather than that it is unsupported.
    #[test]
    fn more_than_one_answer_is_refused_and_nonsense_is_refused_differently() {
        let zero = body(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}], "n": 0
        }));
        assert_eq!(
            zero.validate().expect_err("n is zero").code(),
            "invalid_request"
        );

        let two = body(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}], "n": 2
        }));
        let error = two.validate().expect_err("n is more than one");
        assert_eq!(error.code(), "unsupported_value");

        let one = body(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}], "n": 1
        }));
        assert!(one.validate().is_ok());
    }

    #[test]
    fn the_sampling_ranges_are_the_ones_the_stages_are_defined_on() {
        for (field, value) in [("temperature", 2.5_f32), ("top_p", 0.0), ("min_p", 1.5)] {
            let mut json = minimal();
            json[field] = serde_json::json!(value);
            let error = body(json).validate().expect_err("out of range");
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn a_forced_tool_choice_with_no_tools_on_offer_is_refused() {
        let mut json = minimal();
        json["tool_choice"] = serde_json::json!("required");
        let error = body(json).validate().expect_err("nothing to require");
        assert!(error.to_string().contains("at least one tool"), "{error}");
    }

    #[test]
    fn an_unknown_tool_choice_mode_is_refused_and_a_named_one_is_not() {
        let mut json = minimal();
        json["tool_choice"] = serde_json::json!("whenever");
        assert!(body(json).validate().is_err());

        let mut json = minimal();
        json["tool_choice"] = serde_json::json!({"type": "function", "function": {"name": "f"}});
        let parsed = body(json);
        assert!(matches!(
            parsed.tool_choice,
            Some(WireToolChoice::Named { .. })
        ));
        assert!(parsed.validate().is_ok());
    }

    #[test]
    fn a_service_tier_parses_case_insensitively_and_refuses_anything_else() {
        let mut json = minimal();
        json["service_tier"] = serde_json::json!("FLEX");
        assert_eq!(body(json).service_tier, Some(ServiceTier::Flex));

        let mut json = minimal();
        json["service_tier"] = serde_json::json!("premium");
        assert!(serde_json::from_value::<ChatBody>(json).is_err());

        let mut json = minimal();
        json["service_tier"] = serde_json::json!(3);
        assert!(serde_json::from_value::<ChatBody>(json).is_err());
    }

    /// A field this server has never heard of is not a client mistake, and a
    /// 400 for one would make every addition to the format a breaking change.
    #[test]
    fn a_field_this_server_does_not_know_is_ignored_rather_than_refused() {
        let mut json = minimal();
        json["parallel_tool_calls"] = serde_json::json!(true);
        json["metadata"] = serde_json::json!({"anything": "at all"});
        let parsed = body(json);
        assert_eq!(parsed.messages[0].role, WireRole::User);
    }

    #[test]
    fn the_accounting_rides_the_stream_only_when_the_caller_asked_for_it() {
        let silent = body(minimal());
        assert!(!silent.include_usage());
        let mut json = minimal();
        json["stream_options"] = serde_json::json!({"include_usage": true});
        assert!(body(json).include_usage());
    }

    #[test]
    fn a_stop_field_of_one_sequence_and_of_many_reach_the_same_shape() {
        assert_eq!(StopField::One("x".to_owned()).into_vec(), vec!["x"]);
        assert_eq!(
            StopField::Many(vec!["x".to_owned(), "y".to_owned()]).into_vec(),
            vec!["x", "y"]
        );
    }
}
