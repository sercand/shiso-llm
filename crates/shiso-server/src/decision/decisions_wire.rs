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
        match self {
            Self::Bare(name) => ChoiceOption {
                name,
                description: None,
            },
            Self::Detailed { name, description } => ChoiceOption { name, description },
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
    /// The text every question judges. A string, or any JSON value rendered
    /// to text by the shared rule.
    pub input: Value,
    /// The questions, in the order the answers come back.
    pub questions: Vec<QuestionWire>,
    /// Softmax temperature over the labels. Absent is 1.0.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// The prompt format the caller believes it is speaking. Only 1 exists.
    #[serde(default)]
    pub prompt_format_version: Option<u32>,
    /// Whether each answer carries its prompt and label token ids.
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
        core::check_input(&core::render_text(&self.input))?;

        let mut questions = Vec::with_capacity(self.questions.len());
        for question in self.questions {
            let kind = match question.kind.as_str() {
                TYPE_CHOICE => {
                    refuse_fields_other_than(&question, &["question", "options"])?;
                    let text = question_text(&question);
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
                    let text = question_text(&question);
                    core::check_question_text(&text, &question.id, false)?;
                    let levels = question.levels.unwrap_or_default();
                    core::check_score_levels(&levels, &question.id, 2)?;
                    Kind::Score {
                        question: text,
                        levels,
                    }
                }
                TYPE_YES_NO => {
                    refuse_fields_other_than(&question, &["question", "yes", "no"])?;
                    let text = question_text(&question);
                    core::check_question_text(&text, &question.id, false)?;
                    Kind::YesNo {
                        question: text,
                        yes: wording(question.yes.as_ref()),
                        no: wording(question.no.as_ref()),
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
            });
        }
        core::check_ids(&questions, false)?;
        Ok(questions)
    }
}

/// The question line as text: absent is blank, which the caller is refused for.
fn question_text(question: &QuestionWire) -> String {
    question
        .question
        .as_ref()
        .map_or_else(String::new, core::render_text)
}

/// The `yes`/`no` wording: absent or blank is no wording at all, which drops
/// that line from the block.
fn wording(value: Option<&Value>) -> Option<String> {
    let text = value.map(core::render_text)?;
    (!text.trim().is_empty()).then_some(text)
}

/// Refuses a field the question's type has no place for - `options` on a
/// score question is a caller expecting a different question than the one its
/// `type` says will be scored.
fn refuse_fields_other_than(question: &QuestionWire, allowed: &[&str]) -> Result<(), AppError> {
    let present: [(&str, bool); 4] = [
        ("options", question.options.is_some()),
        ("levels", question.levels.is_some()),
        ("yes", question.yes.is_some()),
        ("no", question.no.is_some()),
    ];
    for (field, present) in present {
        if present && !allowed.contains(&field) {
            return Err(AppError::bad_request(
                format!(
                    "question \"{}\": {field} does not belong on a \"{}\" question",
                    question.id, question.kind
                ),
                Some("questions"),
                "invalid_request",
            ));
        }
    }
    Ok(())
}
