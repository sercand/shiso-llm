//! The `POST /v1/systemone` request body, exactly as a caller spells it.
//!
//! This schema is deliberately looser than the decisions one at its edges and
//! exactly as strict where an answer could go wrong. Unknown *top-level*
//! fields are ignored: a System One client stamps its whole document into the
//! call, and a field the server has never heard of changes no scored byte.
//! Nested objects — a question, the `noul` block, one criterion — refuse
//! unknown keys, because inside one of those a key the server does not read
//! would be a key the caller believes is scored. The decisions-only fields
//! `temperature`, `prompt_format_version` and `return_prompt_token_ids` are
//! accepted as `null` and refused with any other value: this surface scores
//! at temperature 1, speaks prompt format 1, and never returns token ids, so
//! a caller setting one has the wrong route, and a refusal beats quietly
//! ignoring the knob it asked for.
//!
//! How a failure is *classified* is this wire's business; what is refused is
//! the shared validators' ([`super::core`]), run here with the relaxed
//! arguments the schema allows — blank ids and question text, lists of one —
//! and the strict ones it does not relax: the bounds, the name collisions,
//! the model form. Parsing is done by the handler so a `serde_json` failure
//! can be mapped to a schema violation with its location; everything below
//! just says what a well-shaped body means.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::decision::core::{self, Kind, RawQuestion};
use crate::decision::decisions_wire::OptionOrChoice;
use crate::error::AppError;

/// How a decision question is answered, as the wire spells the discriminant.
const TYPE_CHOICE: &str = "choice";
/// The score discriminant.
const TYPE_SCORE: &str = "score";
/// The yes/no discriminant.
const TYPE_YES_NO: &str = "yes_no";

/// The systemone request body.
///
/// No `deny_unknown_fields`, and that absence is the whole top-level schema
/// rule — see the module docs for why the top ignores what it does not read.
/// Every field is typed so that a known field of the wrong type is a parse
/// failure (a 422 with a location) rather than a value that survives to be
/// quietly reinterpreted.
#[derive(Debug, Deserialize)]
pub(crate) struct SystemOneBody {
    /// The model the caller named. Never echoed back: this surface answers
    /// under the model the server actually serves. Only the adapter colon
    /// form is refused.
    #[serde(default)]
    pub model: Option<String>,
    /// The text every question judges: a string, or any JSON value rendered
    /// to text by the shared rule. Absent and blank answer here, unlike the
    /// decisions route's `input`.
    #[serde(default)]
    pub state: Option<Value>,
    /// The questions, in the order the answers come back; noul criteria
    /// answer after them, in the order the `noul` block lists them.
    pub questions: Vec<QuestionWireS1>,
    /// The noul block: standalone binary criteria over the state.
    #[serde(default)]
    pub noul: Option<Noul>,
    /// Decisions-only: null or absent, anything else is a refusal.
    #[serde(default)]
    pub temperature: Option<Value>,
    /// Decisions-only: null or absent, anything else is a refusal.
    #[serde(default)]
    pub prompt_format_version: Option<Value>,
    /// Decisions-only: null or absent, anything else is a refusal.
    #[serde(default)]
    pub return_prompt_token_ids: Option<Value>,
    /// Chat-template knobs; the accepted keys and their rules are
    /// [`super::core::TemplateKwargs`]'s.
    #[serde(default)]
    pub chat_template_kwargs: Option<Map<String, Value>>,
}

/// One question of the request. The keys are the decisions question's keys;
/// what differs is which values validation lets through — blanks and lists
/// of one — not which keys exist, so a nested unknown key is refused here
/// exactly as there.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QuestionWireS1 {
    /// The id the answer comes back under. Blank and absent are answers on
    /// this surface; uniqueness still holds among the ids that name anything.
    #[serde(default)]
    pub id: String,
    /// One of `choice`, `score`, `yes_no`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The line above the block; blank drops the line here instead of being
    /// refused.
    #[serde(default)]
    pub question: Option<Value>,
    /// The options of a choice question, as bare names or detailed objects —
    /// the same spelling (and the same nested unknown-key refusal) the
    /// decisions wire defines.
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

/// The noul block: a list of binary criteria, each answered `yes` or `no`
/// over the state.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Noul {
    /// The criteria, answered in this order after every question.
    #[serde(default)]
    pub criteria: Vec<Criterion>,
}

/// One noul criterion: a yes/no question over the state, spelled shorter than
/// a full question object because it has no options, levels or label wording.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Criterion {
    /// The id the answer comes back under; blank and absent are answers.
    #[serde(default)]
    pub id: String,
    /// What is being asserted; blank asks the bare lead question.
    #[serde(default)]
    pub question: Option<Value>,
}

impl SystemOneBody {
    /// Turns a well-shaped body into the questions the core scores: the
    /// request's questions in order, then every noul criterion as its own
    /// yes/no question with no label wording.
    ///
    /// Every refusal is the shared validator's, run with `blanks_allowed`
    /// true and `min` 1 — the two relaxations this schema states. A refusal
    /// names the field at fault, in the order a caller would fix them: the
    /// model form, the state, then each question in request order, then the
    /// ids once every question can be named (a question and a criterion that
    /// share an id answer under one name, so they collide like any two
    /// questions do).
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming the field at fault.
    pub(crate) fn validate(self) -> Result<Vec<RawQuestion>, AppError> {
        core::check_model(self.model.as_deref())?;
        let state = self.state_text();
        // A blank state answers here — the schema allows it — so the input
        // check runs only over a state that says something, which is the
        // decisions route's rule for everything this schema does not relax.
        if !state.trim().is_empty() {
            core::check_input(&state)?;
        }

        let mut questions = Vec::with_capacity(self.questions.len());
        for question in self.questions {
            let kind = match question.kind.as_str() {
                TYPE_CHOICE => {
                    refuse_fields_other_than(&question, &["question", "options"])?;
                    let text = core::question_text(question.question.as_ref());
                    // The blank branch exists here because the schema allows
                    // blank and the wording builder renders it.
                    core::check_question_text(&text, &question.id, true)?;
                    let options = question
                        .options
                        .unwrap_or_default()
                        .into_iter()
                        .map(OptionOrChoice::into_option)
                        .collect::<Vec<_>>();
                    core::check_choice_options(&options, &question.id, 1)?;
                    Kind::Choice {
                        question: text,
                        options,
                    }
                }
                TYPE_SCORE => {
                    refuse_fields_other_than(&question, &["question", "levels"])?;
                    let text = core::question_text(question.question.as_ref());
                    core::check_question_text(&text, &question.id, true)?;
                    let levels = question.levels.unwrap_or_default();
                    core::check_score_levels(&levels, &question.id, 1)?;
                    Kind::Score {
                        question: text,
                        levels,
                    }
                }
                TYPE_YES_NO => {
                    refuse_fields_other_than(&question, &["question", "yes", "no"])?;
                    let text = core::question_text(question.question.as_ref());
                    core::check_question_text(&text, &question.id, true)?;
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
                probability_answer: false,
            });
        }
        if let Some(noul) = self.noul {
            for criterion in noul.criteria {
                let text = core::question_text(criterion.question.as_ref());
                questions.push(RawQuestion {
                    id: criterion.id,
                    kind: Kind::YesNo {
                        question: text,
                        yes: None,
                        no: None,
                    },
                    // The design: "the `noul` answer is the yes
                    // probability". The criterion is scored as a yes/no
                    // question; only its answer shape differs, and the mark
                    // says so at the one place the wire knows it.
                    probability_answer: true,
                });
            }
        }
        core::check_ids(&questions, true)?;
        Ok(questions)
    }

    /// The state as the prompt and the response echo see it: absent is blank.
    pub(crate) fn state_text(&self) -> String {
        self.state
            .as_ref()
            .map(core::render_text)
            .unwrap_or_default()
    }
}

/// Refuses a decisions-only field that arrived with any value other than
/// null.
///
/// A caller that stamps `null` on every knob it does not use costs this route
/// nothing; a caller that sets one is asking this route to rescale the
/// softmax, change the wording or hand back token ids — behaviour that lives
/// on `/v1/decisions` — so the refusal names the field whose route was meant
/// rather than answering a different request than the one it got.
pub(crate) fn refuse_decisions_only(field: &str, value: Option<&Value>) -> Result<(), AppError> {
    if matches!(value, None | Some(Value::Null)) {
        return Ok(());
    }
    Err(AppError::bad_request(
        format!(
            "{field} is a decisions-route knob; this surface scores at temperature 1, \
             speaks prompt format version 1 and never returns token ids, so {field} \
             must be null or absent"
        ),
        Some(field),
        "invalid_request",
    ))
}

/// Refuses a field the question's type has no place for, projecting this
/// struct's field list into the shared refusal. The presence table belongs to
/// the wire struct, so that much stays here; the sentence, the param and the
/// code are [`core::refuse_fields_other_than`]'s — the same refusal, in the
/// same words, on both surfaces.
fn refuse_fields_other_than(question: &QuestionWireS1, allowed: &[&str]) -> Result<(), AppError> {
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
