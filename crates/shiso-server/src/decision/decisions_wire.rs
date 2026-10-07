//! The `POST /v1/decisions` request body, exactly as a caller spells it.
//!
//! Every struct refuses unknown fields. This wire is a scoring contract, not a
//! chat conversation: a caller that sends `options` to a `score` question, or a
//! field this server has never heard of, has a misconception about what will
//! be scored, and answering anyway would be answering a different question
//! than the one it asked. (The chat surface ignores unknown fields for the
//! opposite reason - a stray client-library field there changes no token.)
//!
//! The wire types carry no logic beyond parsing and field-shape checks; every
//! semantic rule - the bounds, the name collisions, the model form - lives in
//! [`super::core`] so both surfaces of the decision family refuse the same bad
//! question for the same reason.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::decision::core::{self, ChoiceOption, Kind, RawQuestion};
use crate::error::AppError;

/// How a decision question is answered, as the wire spells the discriminant.
const TYPE_CHOICE: &str = "choice";
/// The score discriminant.
const TYPE_SCORE: &str = "score";
/// The yes/no discriminant.
const TYPE_YES_NO: &str = "yes_no";

/// One option of a choice question: a bare name, or a name with a description.
///
/// Unknown keys inside the object spelling are refused by hand, naming the
/// key: serde's derive forbids `deny_unknown_fields` on an untagged variant,
/// so this enum implements [`Deserialize`] against a buffered [`Value`] and
/// enforces the same rule the derived structs get for free. A silently
/// dropped key here would mean the server scored an option its caller never
/// described.
#[derive(Debug)]
pub(crate) enum OptionOrChoice {
    /// Just the name.
    Bare(String),
    /// The name and the wording shown after `" - "` on its line.
    Detailed {
        /// What follows the letter on the option's line.
        name: String,
        /// Extra wording, shown when non-blank.
        description: Option<String>,
    },
}

impl<'de> Deserialize<'de> for OptionOrChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match Value::deserialize(deserializer)? {
            Value::String(name) => Ok(Self::Bare(name)),
            Value::Object(fields) => {
                let mut name: Option<Value> = None;
                let mut description: Option<Value> = None;
                for (key, value) in fields {
                    match key.as_str() {
                        "name" => name = Some(value),
                        "description" => description = Some(value),
                        other => {
                            return Err(serde::de::Error::custom(format!(
                                "unknown field `{other}`, expected `name` or `description`"
                            )));
                        }
                    }
                }
                let name = name.ok_or_else(|| serde::de::Error::missing_field("name"))?;
                Ok(Self::Detailed {
                    name: String::deserialize(name).map_err(serde::de::Error::custom)?,
                    description: description
                        .map(String::deserialize)
                        .transpose()
                        .map_err(serde::de::Error::custom)?,
                })
            }
            other => Err(serde::de::Error::custom(format!(
                "an option must be a name or a {{name, description}} object, got {other}"
            ))),
        }
    }
}

impl OptionOrChoice {
    /// The option as the core renders and validates it.
    pub(crate) fn into_option(self) -> ChoiceOption {
        let (name, description) = match self {
            Self::Bare(name) => (name, None),
            Self::Detailed { name, description } => (name, description),
        };
        ChoiceOption {
            value: core::ChoiceValue::Str(name),
            description,
        }
    }
}

/// One question of the request.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QuestionWire {
    /// The id the answer comes back under. Blank is refused by validation;
    /// the field itself is required so a caller cannot answer by position
    /// without saying so.
    pub id: String,
    /// One of `choice`, `score`, `yes_no`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The line above the block; required non-blank text on this surface.
    #[serde(default)]
    pub question: Option<Value>,
    /// The options of a choice question.
    #[serde(default)]
    pub options: Option<Vec<OptionOrChoice>>,
    /// The levels of a score question, indexed from zero.
    #[serde(default)]
    pub levels: Option<Vec<String>>,
    /// The wording of the `yes` label.
    #[serde(default)]
    pub yes: Option<Value>,
    /// The wording of the `no` label.
    #[serde(default)]
    pub no: Option<Value>,
}

/// The decisions request body.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecisionsBody {
    /// The model the caller named. Echoed back; refused outright when it uses
    /// the adapter colon form, which this server cannot load.
    #[serde(default)]
    pub model: Option<String>,
    /// The text every question judges. A string, any JSON container rendered
    /// to text by the shared rule, or a content-block array of text and
    /// images (see [`decision_input`]).
    pub input: Value,
    /// The questions, in the order the answers come back.
    pub questions: Vec<QuestionWire>,
    /// Softmax temperature over the labels. Absent is 1.0.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// The prompt format the caller believes it is speaking. Only 1 exists.
    #[serde(default)]
    pub prompt_format_version: Option<u32>,
    /// Whether each answer carries its prompt and label token ids. When the
    /// prompt carries media, the returned ids are the *expanded* ones the
    /// model read - pad runs included - and their count is the prompt side
    /// of `usage.prompt_tokens`.
    #[serde(default)]
    pub return_prompt_token_ids: Option<bool>,
    /// Chat-template knobs; the accepted keys and their rules are
    /// [`super::core::TemplateKwargs`]'s.
    #[serde(default)]
    pub chat_template_kwargs: Option<Map<String, Value>>,
}

impl DecisionsBody {
    /// Turns a well-shaped body into the questions the core scores.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming the field at fault, in the order a
    /// caller would fix them: the format version first, then the model, then
    /// the input, then each question in request order - its type, the fields
    /// that type does not have, the question text, the option or level list -
    /// and last the ids, once every question can be named.
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
                // "Blank" means no text *and* no images (spec ruling 2): an
                // image-only input scores, an array of only-blank text
                // blocks does not. An empty blocks array cannot occur -
                // detection requires a typed element.
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
        for question in self.questions {
            let kind = match question.kind.as_str() {
                TYPE_CHOICE => {
                    refuse_fields_other_than(&question, &["question", "options"])?;
                    let text = core::question_text(question.question.as_ref());
                    core::check_question_text(&text, &question.id, false)?;
                    let options = question
                        .options
                        .unwrap_or_default()
                        .into_iter()
                        .map(OptionOrChoice::into_option)
                        .collect::<Vec<_>>();
                    core::check_choice_options(&options, &question.id, 2)?;
                    Kind::Choice {
                        question: text,
                        options,
                    }
                }
                TYPE_SCORE => {
                    refuse_fields_other_than(&question, &["question", "levels"])?;
                    let text = core::question_text(question.question.as_ref());
                    core::check_question_text(&text, &question.id, false)?;
                    let levels = question
                        .levels
                        .unwrap_or_default()
                        .into_iter()
                        .map(|label| core::ScoreLevel {
                            label,
                            description: None,
                        })
                        .collect::<Vec<_>>();
                    core::check_score_levels(&levels, &question.id, 2)?;
                    Kind::Score {
                        question: text,
                        levels,
                    }
                }
                TYPE_YES_NO => {
                    refuse_fields_other_than(&question, &["question", "yes", "no"])?;
                    let text = core::question_text(question.question.as_ref());
                    core::check_question_text(&text, &question.id, false)?;
                    Kind::YesNo {
                        question: text,
                        yes: core::wording(question.yes.as_ref()),
                        no: core::wording(question.no.as_ref()),
                    }
                }
                other => {
                    return Err(AppError::bad_request(
                        format!(
                            "question \"{}\": type {other:?} is not one of \"{TYPE_CHOICE}\", \"{TYPE_SCORE}\", \"{TYPE_YES_NO}\"",
                            question.id
                        ),
                        Some("questions"),
                        "invalid_request",
                    ));
                }
            };
            questions.push(RawQuestion {
                id: question.id,
                kind,
                // The decisions wire has no noul block: every answer keeps
                // its kind's own answer shape.
                probability_answer: false,
            });
        }
        core::check_ids(&questions, false)?;
        Ok(questions)
    }
}

/// Refuses a field the question's type has no place for, projecting this
/// struct's field list into the shared refusal. The presence table belongs to
/// the wire struct, so that much stays here; the sentence, the param and the
/// code are [`core::refuse_fields_other_than`]'s, shared with the other
/// surface so both refuse the same misplaced field in the same words.
fn refuse_fields_other_than(question: &QuestionWire, allowed: &[&str]) -> Result<(), AppError> {
    core::refuse_fields_other_than(
        &question.id,
        &question.kind,
        &[
            ("options", question.options.is_some()),
            ("levels", question.levels.is_some()),
            ("yes", question.yes.is_some()),
            ("no", question.no.is_some()),
        ],
        allowed,
    )
}

/// True when `value` is an array in which *any* element is an object with a
/// `"type"` key whose value is one of `"text"`, `"image"`, `"audio"`, `"video"`.
///
/// The rule is deliberately "any", not "all" (spec ruling 1): an array in
/// which *no* element has a known `type` is a JSON container by design, so
/// arbitrary datasets that legitimately carry `"type"` keys (`GeoJSON`
/// features, form schemas) keep today's compact-JSON reading. Once one
/// element looks like a block, the array is blocks and every element is held
/// to the block grammar - a typo'd type in a mixed array is refused loudly,
/// never silently rendered as JSON. A top-level object is never blocks.
pub(crate) fn is_content_block_array(value: &Value) -> bool {
    value.as_array().is_some_and(|items| {
        items.iter().any(|item| {
            item.get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| matches!(kind, "text" | "image" | "audio" | "video"))
        })
    })
}

/// The decisions `input`, split per the block/container rule.
#[derive(Debug)]
pub(crate) enum DecisionInput {
    /// A string or a non-block container: the compact text to score, today's path.
    Text(String),
    /// A strictly-validated content-block array, in the caller's order.
    Blocks(Vec<InputBlock>),
}

/// One element of a content-block `input`.
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

/// Where a content-block image's bytes come from.
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
    /// A remote url. It parses here; the `--vision-allow-remote-images`
    /// policy refusal already lives in the renderer and is not duplicated.
    Url {
        /// The url as the caller spelled it.
        url: String,
    },
}

/// One block's `text` arm, strictly. The `type` discriminant is read from
/// the element before dispatch, so this struct carries only what follows it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TextBlockWire {
    // The discriminant is read before dispatch; the field exists so
    // `deny_unknown_fields` accepts the `type` key rather than refusing it.
    #[expect(dead_code, reason = "carried only so `type` is a known field")]
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

/// One block's `image` arm, strictly.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageBlockWire {
    #[expect(dead_code, reason = "carried only so `type` is a known field")]
    #[serde(rename = "type")]
    kind: String,
    source: SourceWire,
}

/// An image block's source, strictly: `deny_unknown_fields` covers both arms.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
enum SourceWire {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

/// The decisions `input`, split per the block/container rule. Blocks are
/// validated strictly; every refusal is a 400 naming `input` (and the
/// offending element's 0-based index where the spec asks for one).
///
/// # Errors
///
/// [`AppError::BadRequest`] when an array detected as blocks holds an
/// element the block grammar refuses, or an `audio`/`video` block.
pub(crate) fn decision_input(value: &Value) -> Result<DecisionInput, AppError> {
    if !is_content_block_array(value) {
        return Ok(DecisionInput::Text(core::render_text(value)));
    }
    let items = value
        .as_array()
        .expect("is_content_block_array answers true only for arrays");
    let mut blocks = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        // The discriminant is read by hand because serde forbids
        // `deny_unknown_fields` on the variants of an internally-tagged
        // enum: each arm's grammar gets its own strict struct instead.
        let block = match item.get("type").and_then(Value::as_str) {
            Some("text") => InputBlock::Text {
                text: TextBlockWire::deserialize(item)
                    .map_err(|error| block_error(index, &error))?
                    .text,
            },
            Some("image") => {
                let source = ImageBlockWire::deserialize(item)
                    .map_err(|error| block_error(index, &error))?
                    .source;
                InputBlock::Image(match source {
                    SourceWire::Base64 { media_type, data } => {
                        if media_type.trim().is_empty() {
                            return Err(block_error(index, &"`media_type` must not be blank"));
                        }
                        InputImageSource::Base64 { media_type, data }
                    }
                    SourceWire::Url { url } => InputImageSource::Url { url },
                })
            }
            // The scoring endpoints take images only in v1: audio and video
            // parse far enough to name their type, and are refused.
            Some(kind @ ("audio" | "video")) => {
                return Err(block_error(
                    index,
                    &format!("a `{kind}` block is not scored: this endpoint takes images only"),
                ));
            }
            other => {
                let got = match other {
                    Some(kind) => format!("`{kind:?}`"),
                    // A non-object element is not "an object without a
                    // `type`"; the refusal names what the caller actually
                    // sent, beside the index the message already carries.
                    None if item.is_object() => "an object without a string `type`".to_owned(),
                    None => format!(
                        "a {} element",
                        match item {
                            Value::Null => "null",
                            Value::Bool(_) => "boolean",
                            Value::Number(_) => "number",
                            Value::String(_) => "string",
                            Value::Array(_) => "array",
                            // Reached only through `is_object` above.
                            Value::Object(_) => "object",
                        }
                    ),
                };
                return Err(block_error(
                    index,
                    &format!(
                        "expected a block with a `type` of `text`, `image`, `audio` or `video`, got {got}"
                    ),
                ));
            }
        };
        blocks.push(block);
    }
    Ok(DecisionInput::Blocks(blocks))
}

/// A block-shape refusal: the same 400 naming `input`, prefixed with the
/// element's 0-based index so a caller can find the element it must fix.
fn block_error(index: usize, cause: &dyn std::fmt::Display) -> AppError {
    AppError::bad_request(
        format!("input block[{index}]: {cause}"),
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

    /// Asserts `value` is refused as a 400 naming `input` and returns the
    /// message so the caller can pin what the refusal names.
    fn refused_message(value: &Value) -> String {
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
    fn is_content_block_array_detects_any_typed_element() {
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
            // All-unknown types stay a container by design (spec ruling 1):
            // arbitrary JSON datasets legitimately carry other "type" values.
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

    #[test]
    fn decision_input_passes_strings_and_containers_through_as_text() {
        let split = decision_input(&json!("  keep me verbatim  ")).expect("a string is text");
        assert!(
            matches!(&split, DecisionInput::Text(text) if text == "  keep me verbatim  "),
            "{split:?}"
        );
        // A container is the same key-sorted compact bytes render_text gives.
        let container = json!({"b": 1, "a": [1, 2]});
        let split = decision_input(&container).expect("a container is text");
        assert!(
            matches!(&split, DecisionInput::Text(text) if *text == core::render_text(&container)),
            "{split:?}"
        );
    }

    #[test]
    fn decision_input_parses_valid_block_arrays_in_order() {
        let split = decision_input(&json!([
            {"type": "text", "text": "a"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
        ]))
        .expect("a valid block array parses");
        match split {
            DecisionInput::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(matches!(&blocks[0], InputBlock::Text { text } if text == "a"));
                match &blocks[1] {
                    InputBlock::Image(InputImageSource::Base64 { media_type, data }) => {
                        assert_eq!(media_type, "image/png");
                        assert_eq!(data, "aGk=");
                    }
                    other => panic!("expected a base64 image, got {other:?}"),
                }
            }
            other @ DecisionInput::Text(_) => panic!("expected blocks, got {other:?}"),
        }
        let split = decision_input(&json!([
            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
        ]))
        .expect("a url source parses here; the policy refusal lives in the renderer");
        assert!(
            matches!(&split, DecisionInput::Blocks(blocks)
                if matches!(&blocks[0], InputBlock::Image(InputImageSource::Url { url })
                    if url == "https://x/y.png")),
            "{split:?}"
        );
    }

    #[test]
    fn malformed_blocks_are_refused_naming_the_element_index() {
        for (value, index) in [
            // Unknown key on a text block.
            (json!([{"type": "text", "text": "a", "extra": 1}]), 0),
            // A typo'd type in a mixed array: element 0 is a block, so the
            // array is blocks and element 1 is held to the grammar (Review
            // Focus 1) - it must not fall back to compact JSON.
            (json!([{"type": "text", "text": "a"}, {"type": "img"}]), 1),
            // A missing required field on the image source.
            (
                json!([{"type": "image", "source": {"type": "base64", "data": "aGk="}}]),
                0,
            ),
            // An unknown key on the image source.
            (
                json!([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk=", "extra": 1}}]),
                0,
            ),
            // A bare string element in a block array.
            (json!(["just a string", {"type": "text", "text": "a"}]), 0),
            // A blank media_type is present but says nothing.
            (
                json!([{"type": "image", "source": {"type": "base64", "media_type": "  ", "data": "aGk="}}]),
                0,
            ),
        ] {
            let message = refused_message(&value);
            assert!(
                message.contains(&index.to_string()),
                "refusal of {value} must name index {index}: {message}"
            );
        }
    }

    #[test]
    fn audio_and_video_blocks_are_refused_naming_the_type() {
        for kind in ["audio", "video"] {
            let message = refused_message(&json!([{"type": kind, "source": {}}]));
            assert!(
                message.contains(kind),
                "refusal must name {kind}: {message}"
            );
        }
    }
}
