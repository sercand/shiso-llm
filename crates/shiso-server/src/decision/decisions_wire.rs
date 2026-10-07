//! The `POST /v1/decisions` request body, exactly as the official `OpenAI`
//! Decisions API spells it: a `model`, an `input` that is a string or an
//! array of user messages carrying text and inline images, and `questions`
//! of type `predicate`, `choice` or `score`.
//!
//! Every struct refuses unknown fields. This wire is a scoring contract, not
//! a chat conversation: a caller that sends `choices` to a `score` question,
//! or a field this server has never heard of, has a misconception about what
//! will be scored, and answering anyway would be answering a different
//! question than the one it asked. (The chat surface ignores unknown fields
//! for the opposite reason - a stray client-library field there changes no
//! token.)
//!
//! The wire types carry no logic beyond parsing and field-shape checks; every
//! semantic rule - the bounds, the name collisions, the model form - lives in
//! [`super::core`] so both surfaces of the decision family refuse the same bad
//! question for the same reason.
//!
//! Two pieces outlive the old block grammar and stay exported for
//! `/v1/systemone`: [`is_content_block_array`], the one rule that decides
//! what a block array is whichever field it arrives in, and the
//! [`InputBlock`]/[`InputImageSource`] types the systemone handler builds
//! from its own `images` extension.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::decision::core::{self, ChoiceOption, ChoiceValue, Kind, RawQuestion, ScoreLevel};
use crate::error::AppError;

/// How a decision question is answered, as the wire spells the discriminant.
const TYPE_PREDICATE: &str = "predicate";
/// The choice discriminant.
const TYPE_CHOICE: &str = "choice";
/// The score discriminant.
const TYPE_SCORE: &str = "score";

/// The most images one request may carry, across every message (the
/// official cap).
const MAX_INPUT_IMAGES: usize = 128;

/// The accepted `detail` spellings of an input image. The level is validated
/// and then not used downstream: this server resizes by its own media knobs,
/// so the caller's claim is accepted for wire compatibility and dropped.
const IMAGE_DETAILS: [&str; 4] = ["low", "high", "auto", "original"];

/// The decisions `input`, split per the string-or-messages rule.
#[derive(Debug)]
pub(crate) enum DecisionInput {
    /// A string: the text to score, today's path.
    Text(String),
    /// The parts of the caller's user messages, flattened in order.
    Blocks(Vec<InputBlock>),
}

/// One part of a message `content` array.
#[derive(Debug)]
pub(crate) enum InputBlock {
    /// Verbatim text: the template concatenates parts with no delimiter, so
    /// the caller's own spacing is the spacing.
    Text {
        /// The text, untrimmed.
        text: String,
    },
    /// An image and where its bytes come from.
    Image(InputImageSource),
}

/// Where an input image's bytes come from.
#[derive(Debug)]
pub(crate) enum InputImageSource {
    /// Inline base64 bytes and the media type the caller claimed. The media
    /// type is validated and then not used downstream: the IR carries only
    /// the bytes, and the decoder sniffs them.
    Base64 {
        /// The caller's claimed media type; present and non-blank.
        #[cfg_attr(
            not(test),
            expect(
                dead_code,
                reason = "validated at the wire and deliberately not carried further: the IR holds only the bytes, and the decoder sniffs them; the unit tests read it to pin the parse"
            )
        )]
        media_type: String,
        /// The base64 payload, undecoded.
        data: String,
    },
    /// A remote url (shiso's extension beside the official data URLs). It
    /// parses here; the `--vision-allow-remote-images` policy refusal already
    /// lives in the renderer and is not duplicated.
    Url {
        /// The url as the caller spelled it.
        url: String,
    },
}

/// True when `value` is an array in which *any* element is an object with a
/// `"type"` key whose value is one of `"text"`, `"image"`, `"audio"`, `"video"`.
///
/// The rule is deliberately "any", not "all": an array in which *no* element
/// has a known `type` is a JSON container by design, so arbitrary datasets
/// that legitimately carry `"type"` keys (`GeoJSON` features, form schemas)
/// keep the compact-JSON reading wherever containers are still read. This is
/// no longer the decisions route's own input rule - that route only takes
/// strings and messages now - but `/v1/systemone` refuses a block array
/// pasted into its `state` with exactly this predicate, so one rule decides
/// what a block array is, whichever field it arrives in.
pub(crate) fn is_content_block_array(value: &Value) -> bool {
    value.as_array().is_some_and(|items| {
        items.iter().any(|item| {
            item.get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| matches!(kind, "text" | "image" | "audio" | "video"))
        })
    })
}

/// Parses the data-URL spelling of an inline image:
/// `data:image/<subtype>;base64,<payload>`.
///
/// The structural parts of the head fold case (the pattern this repo's
/// systemone `images` extension already reads by); the payload is the
/// caller's bytes, taken verbatim after the comma. The media type is carried
/// in its folded spelling and dropped downstream like every other claimed
/// media type.
pub(crate) fn parse_image_data_url(text: &str) -> Result<InputImageSource, String> {
    let (head, payload) = text.split_once(',').ok_or_else(|| {
        "an inline image must be a data URL spelled `data:image/<subtype>;base64,<payload>`"
            .to_owned()
    })?;
    let head = head.to_ascii_lowercase();
    let spec = head.strip_prefix("data:").ok_or_else(|| {
        "an inline image must be a data URL spelled `data:image/<subtype>;base64,<payload>`"
            .to_owned()
    })?;
    let Some((media_type, params)) = spec.split_once(';') else {
        return Err("a data URL must carry the `;base64` parameter".to_owned());
    };
    if params != "base64" {
        return Err(format!(
            "a data URL's parameter must be `base64`, got `{params}`"
        ));
    }
    if !media_type.starts_with("image/") || media_type == "image/" {
        return Err(format!(
            "a data URL must name an `image/<subtype>` media type, got `{media_type}`"
        ));
    }
    if payload.is_empty() {
        return Err("a data URL must carry a non-empty base64 payload".to_owned());
    }
    Ok(InputImageSource::Base64 {
        media_type: media_type.to_owned(),
        data: payload.to_owned(),
    })
}

/// One option of a choice question: a bare value, or a value with a
/// description.
///
/// Unknown keys inside the object spelling are refused by hand, naming the
/// key: serde's derive forbids `deny_unknown_fields` on an untagged variant,
/// so this struct implements [`Deserialize`] against a buffered [`Value`] and
/// enforces the same rule the derived structs get for free. A silently
/// dropped key here would mean the server scored an option its caller never
/// described. Values are typed - a string and a boolean with the same text
/// are distinct options - and a bare string or boolean is the lenient
/// spelling of `{value}`.
#[derive(Debug)]
pub(crate) struct ChoiceWire {
    /// The option's value, kept typed for the answer's echo.
    value: ChoiceValue,
    /// Extra wording, shown when non-blank.
    description: Option<String>,
}

impl<'de> Deserialize<'de> for ChoiceWire {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match Value::deserialize(deserializer)? {
            Value::String(text) => Ok(Self {
                value: ChoiceValue::Str(text),
                description: None,
            }),
            Value::Bool(value) => Ok(Self {
                value: ChoiceValue::Bool(value),
                description: None,
            }),
            Value::Object(fields) => {
                let mut value: Option<Value> = None;
                let mut description: Option<Value> = None;
                for (key, value_in) in fields {
                    match key.as_str() {
                        "value" => value = Some(value_in),
                        "description" => description = Some(value_in),
                        other => {
                            return Err(serde::de::Error::custom(format!(
                                "unknown field `{other}`, expected `value` or `description`"
                            )));
                        }
                    }
                }
                let value = match value {
                    Some(Value::String(text)) => ChoiceValue::Str(text),
                    Some(Value::Bool(flag)) => ChoiceValue::Bool(flag),
                    Some(other) => {
                        return Err(serde::de::Error::custom(format!(
                            "`value` must be a string or a boolean, got {other}"
                        )));
                    }
                    None => return Err(serde::de::Error::missing_field("value")),
                };
                let description = match description {
                    None | Some(Value::Null) => None,
                    Some(Value::String(text)) => Some(text),
                    Some(other) => {
                        return Err(serde::de::Error::custom(format!(
                            "`description` must be a string or null, got {other}"
                        )));
                    }
                };
                Ok(Self { value, description })
            }
            other => Err(serde::de::Error::custom(format!(
                "a choice must be a string, a boolean, or a {{value, description}} object, got {other}"
            ))),
        }
    }
}

impl ChoiceWire {
    /// The option as the core renders and validates it.
    fn into_option(self) -> ChoiceOption {
        ChoiceOption {
            value: self.value,
            description: self.description,
        }
    }
}

/// One level of a score question, in the official object spelling.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LevelWire {
    /// The level's label: what follows the digit on its line.
    label: String,
    /// Extra wording, shown when non-blank.
    #[serde(default)]
    description: Option<String>,
}

impl LevelWire {
    /// The level as the core renders and validates it.
    fn into_level(self) -> ScoreLevel {
        ScoreLevel {
            label: self.label,
            description: self.description,
        }
    }
}

/// One question of the request.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QuestionWire {
    /// The name the answer comes back under. Absent or null answers by
    /// position and echoes `name: null`; blank and duplicates are refused by
    /// validation.
    #[serde(default)]
    pub name: Option<String>,
    /// One of `predicate`, `choice`, `score`.
    #[serde(rename = "type")]
    pub kind: String,
    /// What to evaluate; required non-blank text on this surface.
    pub instructions: String,
    /// The options of a choice question.
    #[serde(default)]
    pub choices: Option<Vec<ChoiceWire>>,
    /// The levels of a score question, indexed from zero.
    #[serde(default)]
    pub levels: Option<Vec<LevelWire>>,
}

/// The decisions request body.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionsBody {
    /// The model the caller named. Echoed back; refused outright when it uses
    /// the adapter colon form, which this server cannot load.
    #[serde(default)]
    pub model: Option<String>,
    /// The evidence every question judges: a text string or an array of user
    /// messages carrying text and inline images (see [`decision_input`]).
    pub input: Value,
    /// The questions, in the order the answers come back.
    pub questions: Vec<QuestionWire>,
    /// Softmax temperature over the labels. Absent is 1.0. (shiso extension:
    /// the official route has no knob here; a strict client never sends one.)
    #[serde(default)]
    pub temperature: Option<f64>,
    /// The prompt format the caller believes it is speaking. Only 1 exists.
    /// (shiso extension.)
    #[serde(default)]
    pub prompt_format_version: Option<u32>,
    /// Whether each answer carries its prompt and label token ids. When the
    /// prompt carries media, the returned ids are the *expanded* ones the
    /// model read - pad runs included - and their count is the prompt side
    /// of `usage.input_tokens`. (shiso extension.)
    #[serde(default)]
    pub return_prompt_token_ids: Option<bool>,
    /// Chat-template knobs; the accepted keys and their rules are
    /// [`super::core::TemplateKwargs`]'s. (shiso extension.)
    #[serde(default)]
    pub chat_template_kwargs: Option<Map<String, Value>>,
    /// The official opaque end-user id. Accepted for wire compatibility and
    /// deliberately not read: this server scores without end-user identity,
    /// so there is nothing to scope it by.
    #[serde(default)]
    #[expect(
        dead_code,
        reason = "accepted for official wire compatibility and never read: the server keeps no per-user state; serde's typing (string or null, anything else a parse refusal) is the whole handling"
    )]
    pub safety_identifier: Option<String>,
}

impl DecisionsBody {
    /// Turns a well-shaped body into the questions the core scores.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming the field at fault, in the order a
    /// caller would fix them: the format version first, then the model, then
    /// the input, then each question in request order - its name, its type,
    /// the fields that type does not have, the instructions, the choice or
    /// level list - and last the names, once every question can be named.
    pub(crate) fn validate(self) -> Result<Vec<RawQuestion>, AppError> {
        if self
            .prompt_format_version
            .is_some_and(|version| version != 1)
        {
            return Err(AppError::bad_request(
                "this server speaks prompt format version 1; prompt_format_version must be 1 or absent",
                Some("prompt_format_version"),
                "invalid_request",
            ));
        }
        core::check_model(self.model.as_deref())?;
        match decision_input(&self.input)? {
            DecisionInput::Text(text) => core::check_input(&text)?,
            DecisionInput::Blocks(blocks) => {
                // "Blank" means no text *and* no images: an image-only input
                // scores, an array of only-blank text parts does not. An
                // empty message array cannot occur - it is refused earlier.
                if blocks.iter().all(
                    |block| matches!(block, InputBlock::Text { text } if text.trim().is_empty()),
                ) {
                    return Err(AppError::bad_request(
                        "input must not be blank",
                        Some("input"),
                        "invalid_request",
                    ));
                }
            }
        }

        let mut questions = Vec::with_capacity(self.questions.len());
        for (index, question) in self.questions.into_iter().enumerate() {
            if question
                .name
                .as_deref()
                .is_some_and(|name| name.trim().is_empty())
            {
                return Err(AppError::bad_request(
                    format!(
                        "questions[{index}].name must not be blank (omit it to answer by position)"
                    ),
                    Some("questions"),
                    "invalid_request",
                ));
            }
            let id = question.name.clone().unwrap_or_default();
            let kind = match question.kind.as_str() {
                TYPE_PREDICATE => {
                    refuse_fields_other_than(&id, &question.kind, &question, &[])?;
                    core::check_question_text(&question.instructions, &id, false)?;
                    Kind::YesNo {
                        question: question.instructions,
                        // A predicate spells no label wordings: the answer is
                        // the yes probability over the version-1 `yes`/`no`
                        // labels.
                        yes: None,
                        no: None,
                    }
                }
                TYPE_CHOICE => {
                    refuse_fields_other_than(&id, &question.kind, &question, &["choices"])?;
                    core::check_question_text(&question.instructions, &id, false)?;
                    let options = question
                        .choices
                        .unwrap_or_default()
                        .into_iter()
                        .map(ChoiceWire::into_option)
                        .collect::<Vec<_>>();
                    core::check_choice_options(&options, &id, 2)?;
                    Kind::Choice {
                        question: question.instructions,
                        options,
                    }
                }
                TYPE_SCORE => {
                    refuse_fields_other_than(&id, &question.kind, &question, &["levels"])?;
                    core::check_question_text(&question.instructions, &id, false)?;
                    let levels = question
                        .levels
                        .unwrap_or_default()
                        .into_iter()
                        .map(LevelWire::into_level)
                        .collect::<Vec<_>>();
                    core::check_score_levels(&levels, &id, 2)?;
                    Kind::Score {
                        question: question.instructions,
                        levels,
                    }
                }
                other => {
                    return Err(AppError::bad_request(
                        format!(
                            "question {:?}: type {other:?} is not one of \"{TYPE_PREDICATE}\", \"{TYPE_CHOICE}\", \"{TYPE_SCORE}\"",
                            question_name_or_position(&id, index)
                        ),
                        Some("questions"),
                        "invalid_request",
                    ));
                }
            };
            questions.push(RawQuestion {
                id,
                kind,
                // The decisions wire has no noul block: every answer keeps
                // its kind's own answer shape.
                probability_answer: false,
            });
        }
        core::check_ids(&questions, true)?;
        Ok(questions)
    }
}

/// How a refusal names a question that may not have arrived with a name.
fn question_name_or_position(id: &str, index: usize) -> String {
    if id.is_empty() {
        format!("questions[{index}]")
    } else {
        id.to_owned()
    }
}

/// Refuses a field the question's type has no place for, projecting this
/// struct's field list into the shared refusal. The presence table belongs to
/// the wire struct, so that much stays here; the sentence, the param and the
/// code are [`core::refuse_fields_other_than`]'s, shared with the other
/// surface so both refuse the same misplaced field in the same words.
fn refuse_fields_other_than(
    id: &str,
    kind: &str,
    question: &QuestionWire,
    allowed: &[&str],
) -> Result<(), AppError> {
    core::refuse_fields_other_than(
        id,
        kind,
        &[
            ("choices", question.choices.is_some()),
            ("levels", question.levels.is_some()),
        ],
        allowed,
    )
}

/// One message of the `input` array, strictly. `content` stays a [`Value`]
/// because its two arms (a string, or an array of typed parts) are read by
/// hand so each part's refusal can name the part's index.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MessageWire {
    /// The only supported role; anything else is refused by name.
    role: String,
    /// The official optional discriminator; `message` or absent.
    #[serde(default)]
    r#type: Option<String>,
    /// Text evidence, or an ordered list of text and inline image parts.
    content: Value,
}

/// One `input_text` part, strictly. The `type` discriminant is read from the
/// element before dispatch, so this struct carries only what follows it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputTextWire {
    // The discriminant is read before dispatch; the field exists so
    // `deny_unknown_fields` accepts the `type` key rather than refusing it.
    #[expect(dead_code, reason = "carried only so `type` is a known field")]
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

/// One `input_image` part, strictly.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputImageWire {
    #[expect(dead_code, reason = "carried only so `type` is a known field")]
    #[serde(rename = "type")]
    kind: String,
    image_url: String,
    /// The official detail level; validated against the enum and dropped -
    /// see [`IMAGE_DETAILS`].
    #[serde(default)]
    detail: Option<String>,
}

/// The decisions `input`, split per the string-or-messages rule. Messages
/// are validated strictly and their parts flatten, in order, into one block
/// list; every refusal is a 400 naming `input` (and the offending message's
/// and part's 0-based index where one is known).
///
/// # Errors
///
/// [`AppError::BadRequest`] when the input is neither a string nor a
/// non-empty message array, when a message is not a `user` message or its
/// content holds a part the grammar refuses, when an `image_url` spells
/// neither a base64 data URL nor an http(s) url, or when the request carries
/// more than [`MAX_INPUT_IMAGES`] images.
pub(crate) fn decision_input(value: &Value) -> Result<DecisionInput, AppError> {
    match value {
        Value::String(text) => Ok(DecisionInput::Text(text.clone())),
        Value::Array(messages) => {
            if messages.is_empty() {
                return Err(input_error(None, &"input must carry at least one message"));
            }
            let mut blocks = Vec::new();
            let mut images = 0usize;
            for (index, message) in messages.iter().enumerate() {
                // The content arms are dispatched by hand because serde
                // forbids `deny_unknown_fields` on the variants of an
                // internally-tagged enum: each part's grammar gets its own
                // strict struct instead.
                let wire = MessageWire::deserialize(message)
                    .map_err(|error| input_error(Some(index), &error))?;
                if wire.role != "user" {
                    return Err(input_error(
                        Some(index),
                        &format!("only `user` messages are scored, got role {:?}", wire.role),
                    ));
                }
                if wire.r#type.as_deref().is_some_and(|kind| kind != "message") {
                    return Err(input_error(
                        Some(index),
                        &format!(
                            "a message's `type` must be `message`, got {:?}",
                            wire.r#type
                        ),
                    ));
                }
                match &wire.content {
                    Value::String(text) => blocks.push(InputBlock::Text { text: text.clone() }),
                    Value::Array(parts) => {
                        for (part_index, part) in parts.iter().enumerate() {
                            let block = input_part(part)
                                .map_err(|cause| part_error(index, part_index, &cause))?;
                            if matches!(block, InputBlock::Image(_)) {
                                images += 1;
                            }
                            blocks.push(block);
                        }
                    }
                    other => {
                        return Err(input_error(
                            Some(index),
                            &format!("content must be a string or an array of parts, got {other}"),
                        ));
                    }
                }
            }
            if images > MAX_INPUT_IMAGES {
                return Err(input_error(
                    None,
                    &format!(
                        "input carries {images} images; at most {MAX_INPUT_IMAGES} are allowed"
                    ),
                ));
            }
            Ok(DecisionInput::Blocks(blocks))
        }
        other => Err(input_error(
            None,
            &format!(
                "input must be a string or an array of user messages, got a {}",
                value_kind(other)
            ),
        )),
    }
}

/// One `content` part: `input_text` or `input_image`, strictly. The error
/// arm is a `String` cause so the caller can prefix it with the part's
/// position.
fn input_part(part: &Value) -> Result<InputBlock, String> {
    match part.get("type").and_then(Value::as_str) {
        Some("input_text") => Ok(InputBlock::Text {
            text: InputTextWire::deserialize(part)
                .map_err(|error| error.to_string())?
                .text,
        }),
        Some("input_image") => {
            let wire = InputImageWire::deserialize(part).map_err(|error| error.to_string())?;
            if let Some(detail) = &wire.detail
                && !IMAGE_DETAILS.contains(&detail.as_str())
            {
                return Err(format!(
                    "`detail` must be one of low, high, auto, original or null, got {detail:?}"
                ));
            }
            image_source(&wire.image_url)
        }
        // The scoring endpoint takes text and images only: the other
        // official input-item types parse far enough to name themselves, and
        // are refused.
        Some(kind @ ("input_audio" | "input_file" | "file" | "item_reference")) => Err(format!(
            "a `{kind}` part is not scored: this endpoint takes text and images only"
        )),
        other => {
            let got = match other {
                Some(kind) => format!("`{kind:?}`"),
                // A non-object element is not "an object without a `type`";
                // the refusal names what the caller actually sent.
                None if part.is_object() => "an object without a string `type`".to_owned(),
                None => format!("a {} element", value_kind(part)),
            };
            Err(format!(
                "expected a part with a `type` of `input_text` or `input_image`, got {got}"
            ))
        }
    }
}

/// The `image_url` of an input image: a base64 data URL, or - shiso's
/// extension beside the official data-URL-only rule - an http(s) url whose
/// policy refusal already lives in the renderer.
fn image_source(url: &str) -> Result<InputBlock, String> {
    if url.starts_with("data:") {
        return parse_image_data_url(url).map(InputBlock::Image);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Ok(InputBlock::Image(InputImageSource::Url {
            url: url.to_owned(),
        }));
    }
    Err(format!(
        "`image_url` must be a base64 data URL or an http(s) url, got {url:?}"
    ))
}

/// The name of a JSON value's kind, for refusals that say what arrived.
fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// A message-shape refusal: the same 400 naming `input`, prefixed with the
/// message's 0-based index so a caller can find the element it must fix.
fn input_error(index: Option<usize>, cause: &dyn std::fmt::Display) -> AppError {
    let at = index.map_or_else(
        || "input".to_owned(),
        |index| format!("input message[{index}]"),
    );
    AppError::bad_request(format!("{at}: {cause}"), Some("input"), "invalid_request")
}

/// A content-part refusal, prefixed with the message's and the part's
/// 0-based indices.
fn part_error(message: usize, part: usize, cause: &dyn std::fmt::Display) -> AppError {
    AppError::bad_request(
        format!("input message[{message}] part[{part}]: {cause}"),
        Some("input"),
        "invalid_request",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The known block discriminants, spelled once so the predicate and the
    /// parser cannot drift apart.
    const KNOWN_TYPES: [&str; 4] = ["text", "image", "audio", "video"];

    /// Runs a body down the handler's own path: parse as the wire type, then
    /// `validate`. The refusals under test are `validate`'s, so a body that
    /// cannot even parse fails as a bug in the test, not as a pass.
    fn validated(body: Value) -> Result<Vec<RawQuestion>, AppError> {
        serde_json::from_value::<DecisionsBody>(body)
            .expect("the body parses: shape refusals are validate's job")
            .validate()
    }

    /// Asserts `body` is refused by `validate` as a 400 and returns the
    /// message so the caller can pin what the refusal names.
    fn refused_message(body: Value) -> String {
        match validated(body) {
            Err(AppError::BadRequest { message, .. }) => message,
            Err(other) => panic!("expected a BadRequest, got {other:?}"),
            Ok(questions) => panic!("expected a refusal, got {questions:?}"),
        }
    }

    /// Asserts the body does not parse at all and that serde's own sentence
    /// names the field at fault.
    fn parse_refused(body: Value, needle: &str) {
        let spelling = body.to_string();
        let error =
            serde_json::from_value::<DecisionsBody>(body).expect_err("the body must not parse");
        assert!(
            error.to_string().contains(needle),
            "refusal of {spelling} must name {needle}: {error}"
        );
    }

    /// Asserts `input` is refused as a 400 naming `input` and returns the
    /// message.
    fn refused_input(value: &Value) -> String {
        match decision_input(value) {
            Err(AppError::BadRequest {
                message,
                param,
                code: _,
            }) => {
                assert_eq!(param.as_deref(), Some("input"), "{message}");
                message
            }
            Err(other) => panic!("expected a BadRequest naming input, got {other:?}"),
            Ok(split) => panic!("expected a refusal, got {split:?}"),
        }
    }

    #[test]
    fn the_three_official_question_types_validate() {
        let questions = validated(json!({
            "input": "the input",
            "questions": [
                {"type": "predicate", "name": "damaged", "instructions": "is it damaged"},
                {"type": "choice", "name": "dept", "instructions": "which team",
                 "choices": [{"value": "billing"}, {"value": true, "description": "urgent"}]},
                {"type": "score", "name": "sev", "instructions": "how bad",
                 "levels": [{"label": "Low"}, {"label": "High", "description": "blocked"}]},
            ],
        }))
        .expect("the official shapes validate");
        assert_eq!(questions.len(), 3);
        assert_eq!(questions[0].id, "damaged");
        match &questions[0].kind {
            Kind::YesNo {
                question, yes, no, ..
            } => {
                assert_eq!(question, "is it damaged");
                assert!(
                    yes.is_none() && no.is_none(),
                    "a predicate spells no wordings"
                );
            }
            other => panic!("expected a predicate, got {other:?}"),
        }
        match &questions[1].kind {
            Kind::Choice { options, .. } => {
                assert_eq!(options[0].value, ChoiceValue::Str("billing".to_owned()));
                assert_eq!(options[1].value, ChoiceValue::Bool(true));
                assert_eq!(options[1].description.as_deref(), Some("urgent"));
            }
            other => panic!("expected a choice, got {other:?}"),
        }
        match &questions[2].kind {
            Kind::Score { levels, .. } => {
                assert_eq!(levels[0].label, "Low");
                assert_eq!(levels[1].description.as_deref(), Some("blocked"));
            }
            other => panic!("expected a score, got {other:?}"),
        }
    }

    #[test]
    fn the_docs_examples_validate() {
        validated(json!({
            "model": "gpt-6-luna",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Inspect the product in this photo."},
                {"type": "input_image", "image_url": "data:image/png;base64,aGk="}]}],
            "questions": [{"type": "predicate", "name": "visible_damage",
                           "instructions": "Does the product have visible damage?"}],
        }))
        .expect("the damage example validates");
        validated(json!({
            "model": "gpt-6-luna",
            "input": "I was charged twice for my order.",
            "questions": [{"type": "choice", "name": "department",
                           "instructions": "Which department should handle this complaint?",
                           "choices": [{"value": "billing", "description": "Payments, invoices, and refunds."},
                                       {"value": "technical"}, {"value": "shipping"}, {"value": "other"}]}],
        }))
        .expect("the routing example validates");
        validated(json!({
            "model": "gpt-6-luna",
            "input": "Export fails in Safari but works in Chrome.",
            "questions": [{"type": "score", "name": "severity",
                           "instructions": "How severe is this issue?",
                           "levels": [{"label": "Cosmetic", "description": "Appearance only."},
                                      {"label": "Workaround available"},
                                      {"label": "Fully blocked"}]}],
        }))
        .expect("the severity example validates");
    }

    #[test]
    fn absent_and_null_names_answer_unnamed_without_colliding() {
        let questions = validated(json!({"input": "x", "questions": [
            {"type": "predicate", "instructions": "a"},
            {"type": "predicate", "name": null, "instructions": "b"},
        ]}))
        .expect("unnamed questions validate");
        assert!(
            questions.iter().all(|question| question.id.is_empty()),
            "{questions:?}"
        );
    }

    #[test]
    fn blank_and_duplicate_names_are_refused() {
        let message = refused_message(json!({"input": "x", "questions": [
            {"type": "predicate", "name": "  ", "instructions": "a"}]}));
        assert!(message.contains("questions[0].name"), "{message}");
        let message = refused_message(json!({"input": "x", "questions": [
            {"type": "predicate", "name": "a", "instructions": "x"},
            {"type": "predicate", "name": "a", "instructions": "y"}]}));
        assert!(
            message.contains("questions[0] and questions[1]"),
            "{message}"
        );
    }

    #[test]
    fn blank_instructions_are_refused_naming_the_name() {
        let message = refused_message(json!({"input": "x", "questions": [
            {"type": "predicate", "name": "q", "instructions": "  "}]}));
        assert!(message.contains('q'), "{message}");
    }

    #[test]
    fn missing_instructions_is_a_parse_refusal() {
        parse_refused(
            json!({"input": "x", "questions": [{"type": "predicate"}]}),
            "instructions",
        );
    }

    #[test]
    fn fields_do_not_cross_question_types() {
        for (kind, field) in [
            (
                "predicate",
                json!({"choices": [{"value": "a"}, {"value": "b"}]}),
            ),
            (
                "predicate",
                json!({"levels": [{"label": "a"}, {"label": "b"}]}),
            ),
            (
                "choice",
                json!({"levels": [{"label": "a"}, {"label": "b"}]}),
            ),
            (
                "score",
                json!({"choices": [{"value": "a"}, {"value": "b"}]}),
            ),
        ] {
            let key = field
                .as_object()
                .expect("the extra field is one key")
                .keys()
                .next()
                .expect("one key")
                .clone();
            let mut question = json!({"type": kind, "name": "q", "instructions": "i"});
            question
                .as_object_mut()
                .expect("an object")
                .insert(key.clone(), field[&key].clone());
            let message = refused_message(json!({"input": "x", "questions": [question]}));
            assert!(
                message.contains(&key) && message.contains("does not belong"),
                "{message} must name the misplaced {key}"
            );
        }
    }

    #[test]
    fn an_unknown_question_type_names_the_official_three() {
        let message = refused_message(json!({"input": "x", "questions": [
            {"type": "yes_no", "instructions": "i"}]}));
        assert!(
            message.contains("yes_no") && message.contains("predicate"),
            "{message}"
        );
    }

    #[test]
    fn a_bare_string_choice_is_a_value_and_objects_keep_descriptions() {
        let questions = validated(json!({"input": "x", "questions": [
            {"type": "choice", "name": "q", "instructions": "i",
             "choices": ["billing", {"value": false, "description": "not urgent"}]}]}))
        .expect("bare values and objects both parse");
        match &questions[0].kind {
            Kind::Choice { options, .. } => {
                assert_eq!(options[0].value, ChoiceValue::Str("billing".to_owned()));
                assert!(options[0].description.is_none());
                assert_eq!(options[1].value, ChoiceValue::Bool(false));
                assert_eq!(options[1].description.as_deref(), Some("not urgent"));
            }
            other => panic!("expected a choice, got {other:?}"),
        }
    }

    #[test]
    fn a_choice_object_refuses_unknown_keys_and_untyped_values() {
        parse_refused(
            json!({"input": "x", "questions": [{"type": "choice", "instructions": "i",
                   "choices": [{"value": "a", "bogus": 1}]}]}),
            "bogus",
        );
        parse_refused(
            json!({"input": "x", "questions": [{"type": "choice", "instructions": "i",
                   "choices": [{"value": 5}]}]}),
            "value",
        );
        parse_refused(
            json!({"input": "x", "questions": [{"type": "choice", "instructions": "i",
                   "choices": [5]}]}),
            "choice",
        );
    }

    #[test]
    fn score_levels_are_label_objects_only() {
        parse_refused(
            json!({"input": "x", "questions": [{"type": "score", "instructions": "i",
                   "levels": ["low"]}]}),
            "LevelWire",
        );
        parse_refused(
            json!({"input": "x", "questions": [{"type": "score", "instructions": "i",
                   "levels": [{"label": "low", "bogus": 1}]}]}),
            "bogus",
        );
    }

    #[test]
    fn safety_identifier_accepts_string_and_null_only() {
        for value in [json!("user-1"), json!(null)] {
            validated(json!({"input": "x", "safety_identifier": value,
                "questions": [{"type": "predicate", "instructions": "i"}]}))
            .expect("the official field is accepted and ignored");
        }
        parse_refused(
            json!({"input": "x", "safety_identifier": 5,
                   "questions": [{"type": "predicate", "instructions": "i"}]}),
            "expected a string",
        );
    }

    #[test]
    fn unknown_top_level_fields_are_refused() {
        parse_refused(
            json!({"input": "x", "questions": [], "options": 1}),
            "options",
        );
    }

    #[test]
    fn prompt_format_version_must_say_one() {
        let message =
            refused_message(json!({"input": "x", "prompt_format_version": 2, "questions": []}));
        assert!(message.contains("prompt_format_version"), "{message}");
    }

    #[test]
    fn input_string_is_the_text_arm_verbatim() {
        match decision_input(&json!("  keep me  ")).expect("a string is text") {
            DecisionInput::Text(text) => assert_eq!(text, "  keep me  "),
            other @ DecisionInput::Blocks(_) => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn message_content_strings_and_parts_flatten_in_order() {
        let split = decision_input(&json!([
            {"role": "user", "type": "message", "content": "first "},
            {"role": "user", "content": [
                {"type": "input_text", "text": "second"},
                {"type": "input_image", "image_url": "data:image/png;base64,aGk=", "detail": "high"},
                {"type": "input_image", "image_url": "https://example.com/x.png"},
            ]},
        ]))
        .expect("the doc's message shape parses");
        match split {
            DecisionInput::Blocks(blocks) => {
                assert_eq!(blocks.len(), 4);
                assert!(
                    matches!(&blocks[0], InputBlock::Text { text } if text == "first "),
                    "{blocks:?}"
                );
                assert!(
                    matches!(&blocks[1], InputBlock::Text { text } if text == "second"),
                    "{blocks:?}"
                );
                assert!(
                    matches!(&blocks[2], InputBlock::Image(InputImageSource::Base64 { media_type, data })
                        if media_type == "image/png" && data == "aGk="),
                    "{blocks:?}"
                );
                assert!(
                    matches!(&blocks[3], InputBlock::Image(InputImageSource::Url { url })
                        if url == "https://example.com/x.png"),
                    "{blocks:?}"
                );
            }
            other @ DecisionInput::Text(_) => panic!("expected blocks, got {other:?}"),
        }
    }

    #[test]
    fn image_detail_levels_are_accepted_and_bounded() {
        for detail in [
            json!("low"),
            json!("high"),
            json!("auto"),
            json!("original"),
            json!(null),
        ] {
            decision_input(&json!([{"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,aGk=", "detail": detail}]}]))
                .expect("the official detail levels parse");
        }
        let message = refused_input(&json!([{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,aGk=", "detail": "max"}]}]));
        assert!(message.contains("detail"), "{message}");
    }

    #[test]
    fn bad_image_urls_are_refused_naming_the_part() {
        for (url, needle) in [
            ("data:image/png;base64,", "payload"),
            ("data:application/pdf;base64,aGk=", "image/"),
            ("data:image/png,aGk=", "base64"),
            ("ftp://example.com/x.png", "image_url"),
            ("file_id=abc", "image_url"),
        ] {
            let message = refused_input(&json!([{"role": "user", "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": url}]}]));
            assert!(
                message.contains("message[0] part[1]"),
                "{message} must name the part"
            );
            assert!(message.contains(needle), "{message} must name {needle}");
        }
    }

    #[test]
    fn only_user_messages_and_text_image_parts_are_supported() {
        let message = refused_input(&json!([{"role": "assistant", "content": "hi"}]));
        assert!(message.contains("user"), "{message}");
        for part_type in ["input_audio", "input_file", "file", "item_reference"] {
            let message =
                refused_input(&json!([{"role": "user", "content": [{"type": part_type}]}]));
            assert!(message.contains(part_type), "{message}");
        }
        let message = refused_input(&json!([{"role": "user", "content": ["bare string"]}]));
        assert!(message.contains("input_text"), "{message}");
    }

    #[test]
    fn containers_flat_block_arrays_and_empty_inputs_are_refused() {
        for (value, needle) in [
            (json!({"a": 1}), "string or an array"),
            (json!([]), "at least one message"),
            // Yesterday's flat block array is not a message: no role.
            (json!([{"type": "text", "text": "a"}]), "role"),
            (json!(42), "string or an array"),
        ] {
            let message = refused_input(&value);
            assert!(message.contains(needle), "{message} must name {needle}");
        }
    }

    #[test]
    fn more_than_128_images_are_refused() {
        let parts: Vec<Value> = (0..129)
            .map(|_| json!({"type": "input_image", "image_url": "data:image/png;base64,aGk="}))
            .collect();
        let message = refused_input(&json!([{"role": "user", "content": parts}]));
        assert!(message.contains("128"), "{message}");
    }

    #[test]
    fn blank_input_is_refused_on_both_arms() {
        let message = refused_message(json!({"input": "   ",
            "questions": [{"type": "predicate", "instructions": "i"}]}));
        assert!(message.contains("input must not be blank"), "{message}");
        let message = refused_message(json!({
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "  "}]}],
            "questions": [{"type": "predicate", "instructions": "i"}]}));
        assert!(message.contains("input must not be blank"), "{message}");
    }

    #[test]
    fn is_content_block_array_detects_any_typed_element() {
        // The predicate outlives the decisions block grammar: systemone still
        // refuses a block array pasted into its `state` with this one rule.
        for value in [
            json!([{"type": "text", "text": "x"}]),
            json!([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}]),
            // "any", not "all": one typed element makes the array blocks even
            // beside a bare string.
            json!([{"type": "audio"}, "bare"]),
        ] {
            assert!(is_content_block_array(&value), "{value}");
        }
        for value in [
            json!([]),
            // All-unknown types stay a container by design: arbitrary JSON
            // datasets legitimately carry other "type" values.
            json!([{"type": "photo"}]),
            json!([{"type": "img", "source": {}}]),
            json!([1, 2]),
            json!([{"a": 1}]),
            // A top-level object is never blocks.
            json!({"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}),
        ] {
            assert!(!is_content_block_array(&value), "{value}");
        }
        // The known set is exactly the four discriminants.
        for kind in KNOWN_TYPES {
            assert!(is_content_block_array(&json!([{"type": kind}])));
        }
    }
}
