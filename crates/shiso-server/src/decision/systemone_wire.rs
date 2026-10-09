//! The `POST /v1/systemone` request body, exactly as a caller spells it.
//!
//! This schema is the official System One wire (docs.typesafe.ai/api.md):
//! `questions` is a JSON object keyed by the caller's question ids, each
//! question is `{"type": …, "instructions": …, "criteria": …}`, and the types
//! are `noul`, `choice` and `score`. The port's older array spelling — an
//! array of questions carrying `id`/`question`/`options`/`levels`, the
//! `yes_no` type, the top-level `noul` block — is removed, not aliased: the
//! surface speaks one dialect, the upstream one. `serde_json`'s
//! `preserve_order` carries the map's insertion order into the answers
//! exactly as array order did.
//!
//! The kept deviations from the official doc (spec §1.3): `model` and `state`
//! may be absent (official: both required); choice lists of one answer
//! (official: two), and a list past 26 options is labelled `AA`, `AB`, ..
//! with the pairs the vocabulary spells as one token, so a vocabulary short
//! of 255 such pairs refuses the longest lists; a schema violation is a 422 and a bound or policy refusal
//! a 400 (official: all validation is 422); structured `instructions` render
//! as key-sorted compact JSON rather than being restructured into the prompt.
//!
//! Unknown *top-level* fields stay ignored: a System One client stamps its
//! whole document into the call, and a field the server has never heard of
//! changes no scored byte. A question object and a noul `criteria` object
//! refuse unknown keys, because inside one of those a key the server does not
//! read would be a key the caller believes is scored. The decisions-only
//! fields `temperature`, `prompt_format_version` and `return_prompt_token_ids`
//! are accepted as `null` and refused with any other value: this surface
//! scores at temperature 1, speaks prompt format 1, and never returns token
//! ids, so a caller setting one has the wrong route, and a refusal beats
//! quietly ignoring the knob it asked for.
//!
//! The clef `images` extension (spec §3) is a top-level array of embedded
//! images scored before the state. Each item is a data URL string, a
//! `{"content_type", "base64"}` object, or a `{"url"}` object; a malformed
//! item is a 422 naming `images[i]`, while every media *policy* refusal —
//! bad base64, pixels, remote off, no tower, over budget — stays the 400
//! the renderer and scheduler already raise for `/v1/decisions`. The parsed
//! items are the decisions wire's own image sources: no new media type
//! enters the codebase.
//!
//! How a failure is *classified* is this wire's business; what is refused is
//! the shared validators' ([`super::core`]), run here with the relaxed
//! arguments the schema allows — blank ids and question text, lists of one —
//! and the strict ones it does not relax: the bounds, the name collisions,
//! the model form. Everything below the wire — `RawQuestion`, the version-1
//! wording, the label proofs, the scoring — is the shared core's: the dialect
//! dissolves at this boundary. Parsing is done by the handler so a
//! `serde_json` failure can be mapped to a schema violation with its
//! location; everything below just says what a well-shaped body means.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::decision::core::{self, ChoiceOption, Kind, RawQuestion};
use crate::decision::decisions_wire::{self, InputImageSource};
use crate::error::AppError;

/// How a systemone question is answered, as the official wire spells the
/// discriminant: the binary criterion whose answer is the yes probability.
const TYPE_NOUL: &str = "noul";
/// The choice discriminant.
const TYPE_CHOICE: &str = "choice";
/// The score discriminant.
const TYPE_SCORE: &str = "score";

/// The systemone request body.
///
/// No `deny_unknown_fields`, and that absence is the whole top-level schema
/// rule — see the module docs for why the top ignores what it does not read.
/// Every field is typed so that a known field of the wrong type is a parse
/// failure (a 422 with a location) rather than a value that survives to be
/// quietly reinterpreted. `questions` stays a raw [`Value`] because the map's
/// *keys* are the question ids: the entries are read by hand in `validate`,
/// which is what lets each question's grammar keep `deny_unknown_fields`
/// (serde forbids it on internally-tagged enums), and an absent `questions`
/// stays the body-parse 422 its un-defaulted type already gives it.
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
    /// The questions, keyed by the caller's ids; the map's insertion order is
    /// the order the answers come back in.
    pub questions: Value,
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
    /// The clef extension: embedded images scored before the state. The
    /// items stay raw `Value`s because their grammar is three shapes with
    /// per-shape strictness, read by [`images_sources`] with the item's
    /// 0-based index as the refusal location; a non-array `images` is the
    /// body-parse 422 its typed field already gives it.
    #[serde(default)]
    pub images: Option<Vec<Value>>,
}

/// The `noul` arm: a binary criterion over the state whose answer is the yes
/// probability. `criteria` carries the wording of the two labels under the
/// official keys `true` and `false`; absent criteria ask the bare lead
/// question.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NoulWire {
    // The discriminant is read before dispatch; the field exists so
    // `deny_unknown_fields` accepts the `type` key rather than refusing it.
    #[expect(dead_code, reason = "carried only so `type` is a known field")]
    #[serde(rename = "type")]
    kind: String,
    /// What is being asserted; blank drops the question line.
    instructions: Value,
    /// The label wording, optional as a whole and key by key.
    #[serde(default)]
    criteria: Option<NoulCriteria>,
}

/// A noul question's label wording: what `yes` and `no` mean here, spelled
/// as the official doc spells them.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct NoulCriteria {
    /// The wording of the `yes` label, under the official key `true`.
    r#true: Option<Value>,
    /// The wording of the `no` label, under the official key `false`.
    r#false: Option<Value>,
}

/// The `choice` arm: `criteria` is a map of option name to description — any
/// JSON the shared render rule turns to text, `null` meaning no detail — and
/// the map's order is the order the `A..` labels follow.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChoiceWire {
    #[expect(dead_code, reason = "carried only so `type` is a known field")]
    #[serde(rename = "type")]
    kind: String,
    /// The line above the options; blank drops the line here.
    instructions: Value,
    /// The options, keyed by name.
    criteria: Map<String, Value>,
}

/// The `score` arm: `criteria` is the array of levels, indexed from zero.
/// A non-string item is the shared compact rendering; a `null` item is the
/// one value the grammar does not enumerate and is refused.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScoreWire {
    #[expect(dead_code, reason = "carried only so `type` is a known field")]
    #[serde(rename = "type")]
    kind: String,
    /// The line above the levels; blank drops the line here.
    instructions: Value,
    /// The levels, in the caller's order.
    criteria: Vec<Value>,
}

impl SystemOneBody {
    /// Turns a well-shaped body into the questions the core scores: the map's
    /// entries in insertion order, which `preserve_order` keeps as the
    /// caller's key order and the encoder's answer order.
    ///
    /// Every refusal is the shared validator's, run with `blanks_allowed`
    /// true and `min` 1 — the two relaxations this schema states — or this
    /// wire's own schema refusal naming the question's key path. Refusals
    /// come in the order a caller would fix them: the model form, the state,
    /// then each question in request order, then the ids once every question
    /// can be named (the map makes a duplicate id unreachable, but the
    /// shared check stays as the guard, and a blank key is still a blank id
    /// this surface allows).
    ///
    /// # Errors
    ///
    /// [`AppError::SchemaViolation`] for a body the schema cannot express,
    /// [`AppError::BadRequest`] naming the field at fault for a bound.
    pub(crate) fn validate(self) -> Result<Vec<RawQuestion>, AppError> {
        core::check_model(self.model.as_deref())?;
        // A content-block array as `state` is a caller error, not a document
        // to score: pasting a base64 blob into a scored prompt is refused
        // loudly with this surface's schema class (spec ruling 6). The
        // predicate is the decisions wire's - one rule decides what a block
        // array is, whichever field it arrives in.
        if self
            .state
            .as_ref()
            .is_some_and(decisions_wire::is_content_block_array)
        {
            return Err(AppError::SchemaViolation {
                message: "state must be a string or a JSON container: systemone takes text only"
                    .to_owned(),
                at: "state".to_owned(),
            });
        }
        let state = self.state_text();
        // A blank state answers here — the schema allows it — so the input
        // check runs only over a state that says something, which is the
        // decisions route's rule for everything this schema does not relax.
        if !state.trim().is_empty() {
            core::check_input(&state)?;
        }

        // The map's keys are the ids and its order is the answer order;
        // `preserve_order` is what makes both statements true at once. An
        // empty map scores nothing, which is not an answer anyone asked for.
        let entries = match self.questions {
            Value::Object(entries) if !entries.is_empty() => entries,
            Value::Object(_) => {
                return Err(AppError::SchemaViolation {
                    message: "systemone needs at least one question".to_owned(),
                    at: "questions".to_owned(),
                });
            }
            other => {
                return Err(AppError::SchemaViolation {
                    message: format!(
                        "questions must be a JSON object keyed by question id, got {other}"
                    ),
                    at: "questions".to_owned(),
                });
            }
        };

        let mut questions = Vec::with_capacity(entries.len());
        for (id, entry) in entries {
            questions.push(question_from(&id, &entry)?);
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

/// One entry of the questions map: the key is the id, the value the question
/// object. The `type` discriminant is read by hand because serde forbids
/// `deny_unknown_fields` on the variants of an internally-tagged enum: each
/// arm's grammar gets its own strict struct instead, exactly as the decisions
/// wire's block parser does.
fn question_from(id: &str, entry: &Value) -> Result<RawQuestion, AppError> {
    let at = format!("questions.{id}");
    let Some(fields) = entry.as_object() else {
        return Err(AppError::SchemaViolation {
            message: format!("question \"{id}\": a question must be an object, got {entry}"),
            at,
        });
    };
    let discriminant = match fields.get("type") {
        Some(Value::String(kind)) => kind.as_str(),
        Some(other) => {
            return Err(AppError::SchemaViolation {
                message: format!("question \"{id}\": `type` must be a string, got {other}"),
                at,
            });
        }
        None => {
            return Err(AppError::SchemaViolation {
                message: format!(
                    "question \"{id}\": `type` is required, one of \"{TYPE_NOUL}\", \
                     \"{TYPE_CHOICE}\", \"{TYPE_SCORE}\""
                ),
                at,
            });
        }
    };
    let kind = match discriminant {
        TYPE_NOUL => {
            let wire = NoulWire::deserialize(entry).map_err(|error| question_error(id, &error))?;
            let text = instructions_text(id, &wire.instructions)?;
            core::check_question_text(&text, id, true)?;
            let criteria = wire.criteria.unwrap_or_default();
            Kind::YesNo {
                question: text,
                yes: core::wording(criteria.r#true.as_ref()),
                no: core::wording(criteria.r#false.as_ref()),
            }
        }
        TYPE_CHOICE => {
            let wire =
                ChoiceWire::deserialize(entry).map_err(|error| question_error(id, &error))?;
            let text = instructions_text(id, &wire.instructions)?;
            core::check_question_text(&text, id, true)?;
            let options: Vec<ChoiceOption> = wire
                .criteria
                .into_iter()
                .map(|(name, description)| ChoiceOption {
                    value: core::ChoiceValue::Str(name),
                    // `null` says "no detail"; any other value is the shared
                    // rendering, so a container description is compact JSON
                    // text, never a refusal.
                    description: (description != Value::Null)
                        .then(|| core::render_text(&description)),
                })
                .collect();
            core::check_choice_options(&options, id, 1)?;
            Kind::Choice {
                question: text,
                options,
            }
        }
        TYPE_SCORE => {
            let wire = ScoreWire::deserialize(entry).map_err(|error| question_error(id, &error))?;
            let text = instructions_text(id, &wire.instructions)?;
            core::check_question_text(&text, id, true)?;
            let levels = score_levels(id, &wire.criteria)?;
            core::check_score_levels(&levels, id, 1)?;
            Kind::Score {
                question: text,
                levels,
            }
        }
        other => {
            return Err(AppError::SchemaViolation {
                message: format!(
                    "question \"{id}\": type {other:?} is not one of \"{TYPE_NOUL}\", \
                     \"{TYPE_CHOICE}\", \"{TYPE_SCORE}\""
                ),
                at,
            });
        }
    };
    // The noul arm is the only one that renders a yes/no kind, and the
    // design pins "the `noul` answer is the yes probability" - the mark
    // says so at the one place the wire knows it.
    let probability_answer = matches!(kind, Kind::YesNo { .. });
    Ok(RawQuestion {
        id: id.to_owned(),
        kind,
        probability_answer,
    })
}

/// A question-shape refusal: serde's own cause, prefixed with the question's
/// id and located at its key, so the caller sees the field to fix rather than
/// a bare parse sentence.
fn question_error(id: &str, cause: &serde_json::Error) -> AppError {
    AppError::SchemaViolation {
        message: format!("question \"{id}\": {cause}"),
        at: format!("questions.{id}"),
    }
}

/// The `instructions` text: the shared rendering, minus the one value the
/// wire grammar does not enumerate. The grammar (spec §1.2) spells
/// `instructions` as string/object/array, never null, and the shared rule
/// would render a null as the literal word `"null"` — a question line no
/// caller wrote, scored like any other — so the null is refused as the
/// schema violation it already is, naming the question's key.
fn instructions_text(id: &str, instructions: &Value) -> Result<String, AppError> {
    if matches!(instructions, Value::Null) {
        return Err(AppError::SchemaViolation {
            message: format!(
                "question \"{id}\": `instructions` must be a string, object or array, \
                 got null"
            ),
            at: format!("questions.{id}"),
        });
    }
    Ok(core::question_text(Some(instructions)))
}

/// The score levels as text: a string is itself, any other non-null value is
/// the shared compact rendering. A null item is refused like `instructions`
/// — the same grammar enumerates the items as string/object/array, and a
/// scored `"null"` level would sit among the caller's real levels as a level
/// nobody wrote — and the refusal names the item's index so the caller edits
/// the right element.
fn score_levels(id: &str, criteria: &[Value]) -> Result<Vec<core::ScoreLevel>, AppError> {
    criteria
        .iter()
        .enumerate()
        .map(|(index, level)| match level {
            Value::String(text) => Ok(core::ScoreLevel {
                label: text.clone(),
                description: None,
            }),
            Value::Null => Err(AppError::SchemaViolation {
                message: format!(
                    "question \"{id}\": a score level must be a string, object or array, \
                     got null (level {index})"
                ),
                at: format!("questions.{id}"),
            }),
            other => Ok(core::ScoreLevel {
                label: core::render_text(other),
                description: None,
            }),
        })
        .collect()
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

/// One `images` item in clef's base64-object form, strictly: `content_type`
/// is validated non-blank and then dropped - the decoder sniffs the bytes,
/// the decisions rule - and an unknown key is refused because inside one of
/// these objects a key the server does not read is a claim the caller
/// believes the server checked.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Base64ImageItem {
    /// The claimed media type; present and non-blank.
    content_type: String,
    /// The base64 payload, undecoded.
    base64: String,
}

/// One `images` item in the url form (shiso's extra beside clef's two),
/// strictly. It parses here; the `--vision-allow-remote-images` policy
/// refusal already lives in the renderer - the same arm decisions' url
/// block rides - and is not duplicated at this wire.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UrlImageItem {
    /// The url as the caller spelled it.
    url: String,
}

/// Turns the clef `images` array into the decisions wire's image sources.
/// Each item is one of three forms:
///
/// - a **data URL string**: `data:image/<subtype>;base64,<payload>`, the
///   prefix read case-insensitively (clef's pattern). The media type is read
///   from the prefix, validated non-blank, and then dropped downstream - the
///   decoder sniffs the bytes.
/// - a **base64 object**: `{"content_type": …, "base64": …}`.
/// - a **url object**: `{"url": …}` - it parses, and the existing
///   `--vision-allow-remote-images` refusal in the renderer rejects it at
///   request time when the operator's flag is off.
///
/// Absent or empty yields no sources, which is the handler's signal to take
/// the text arm byte-for-byte.
///
/// # Errors
///
/// [`AppError::SchemaViolation`] naming `images[<0-based index>]` for an
/// item none of the three forms can express. Media *policy* refusals are
/// not this function's: they stay the 400s the renderer and scheduler
/// already raise.
pub(crate) fn images_sources(
    images: Option<&Vec<Value>>,
) -> Result<Vec<InputImageSource>, AppError> {
    let Some(items) = images else {
        return Ok(Vec::new());
    };
    let mut sources = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        // The object arms are dispatched on which keys are present because
        // serde forbids `deny_unknown_fields` on the variants of an
        // internally-tagged enum: each form gets its own strict struct,
        // exactly as the question grammar above does.
        let source = match item {
            Value::String(text) => match parse_image_data_url(text) {
                Ok(source) => source,
                Err(cause) => return Err(image_error(index, &cause)),
            },
            Value::Object(fields) if fields.contains_key("url") => {
                let wire =
                    UrlImageItem::deserialize(item).map_err(|error| image_error(index, &error))?;
                InputImageSource::Url { url: wire.url }
            }
            Value::Object(fields)
                if fields.contains_key("content_type") || fields.contains_key("base64") =>
            {
                let wire = Base64ImageItem::deserialize(item)
                    .map_err(|error| image_error(index, &error))?;
                if wire.content_type.trim().is_empty() {
                    return Err(image_error(index, &"`content_type` must not be blank"));
                }
                InputImageSource::Base64 {
                    media_type: wire.content_type,
                    data: wire.base64,
                }
            }
            other => {
                return Err(image_error(
                    index,
                    &format!(
                        "an image must be a data URL string, a \
                         `{{\"content_type\", \"base64\"}}` object or a `{{\"url\"}}` \
                         object, got {other}"
                    ),
                ));
            }
        };
        sources.push(source);
    }
    Ok(sources)
}

/// Parses clef's data URL form: `data:image/<subtype>;base64,<payload>`.
/// The rule is the decisions wire's own - one parser reads a data URL,
/// whichever field it arrives in - and the folded spelling of the prefix
/// (clef's pattern) lives there.
fn parse_image_data_url(text: &str) -> Result<InputImageSource, String> {
    decisions_wire::parse_image_data_url(text)
}

/// An `images` item-shape refusal: this surface's schema class, located at
/// the item's 0-based index so the caller edits the element it must fix.
fn image_error(index: usize, cause: &dyn std::fmt::Display) -> AppError {
    AppError::SchemaViolation {
        message: format!("images[{index}]: {cause}"),
        at: format!("images[{index}]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::core::ChoiceOption;
    use serde_json::json;

    /// Runs a body down the handler's own path: parse as the wire type, then
    /// `validate`. The refusals under test are `validate`'s, so a body that
    /// cannot even parse fails as a bug in the test, not as a pass.
    fn validated(body: Value) -> Result<Vec<RawQuestion>, AppError> {
        serde_json::from_value::<SystemOneBody>(body)
            .expect("the body parses: shape refusals are validate's job")
            .validate()
    }

    /// The questions of a body that must pass validation.
    fn questions_of(body: Value) -> Vec<RawQuestion> {
        validated(body).expect("the body is valid")
    }

    /// Asserts `body` is refused as a schema violation located at `at` whose
    /// message contains `needle`.
    fn assert_refused_at(body: Value, at: &str, needle: &str) {
        let error = validated(body).expect_err("the body must be refused");
        match error {
            AppError::SchemaViolation { message, at: got } => {
                assert_eq!(got, at, "{message}");
                assert!(
                    message.contains(needle),
                    "the refusal must name {needle}: {message}"
                );
            }
            other => panic!("expected a SchemaViolation at {at}, got {other:?}"),
        }
    }

    #[test]
    fn the_three_official_types_parse() {
        let questions = questions_of(json!({
            "state": "the input",
            "questions": {
                "c1": {"type": "choice", "instructions": "pick one",
                       "criteria": {"red": null, "blue": null}},
                "s1": {"type": "score", "instructions": "how bad",
                       "criteria": ["none", "some"]},
                "n1": {"type": "noul", "instructions": "ready"},
            },
        }));
        assert_eq!(questions.len(), 3);
        let ids: Vec<&str> = questions.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, ["c1", "s1", "n1"]);
        assert!(!questions[0].probability_answer);
        match &questions[0].kind {
            Kind::Choice { question, options } => {
                assert_eq!(question, "pick one");
                let names: Vec<String> = options.iter().map(|o| o.value.text()).collect();
                assert_eq!(names, ["red", "blue"]);
                assert!(
                    options.iter().all(|o| o.description.is_none()),
                    "a null description is no description: {options:?}"
                );
            }
            other => panic!("expected a choice, got {other:?}"),
        }
        assert!(!questions[1].probability_answer);
        match &questions[1].kind {
            Kind::Score { question, levels } => {
                assert_eq!(question, "how bad");
                let names: Vec<&str> = levels.iter().map(|l| l.label.as_str()).collect();
                assert_eq!(names, ["none", "some"]);
            }
            other => panic!("expected a score, got {other:?}"),
        }
        match &questions[2].kind {
            Kind::YesNo { question, yes, no } => {
                assert_eq!(question, "ready");
                assert!(yes.is_none() && no.is_none(), "absent criteria say nothing");
            }
            other => panic!("expected a noul, got {other:?}"),
        }
        assert!(
            questions[2].probability_answer,
            "the noul answer is the yes probability"
        );
    }

    #[test]
    fn array_form_questions_are_refused_naming_questions() {
        // Review Focus 1: yesterday's body after the cutover gets the
        // document-fix class naming the field, not a 500 or an empty answer.
        assert_refused_at(
            json!({
                "state": "the input",
                "questions": [{"id": "q1", "type": "noul", "instructions": "ready"}],
            }),
            "questions",
            "object",
        );
    }

    #[test]
    fn an_empty_question_map_is_refused() {
        assert_refused_at(
            json!({"state": "the input", "questions": {}}),
            "questions",
            "systemone needs at least one question",
        );
    }

    #[test]
    fn a_non_object_question_entry_is_refused_naming_its_id() {
        assert_refused_at(
            json!({"state": "the input", "questions": {"q1": "ready"}}),
            "questions.q1",
            "q1",
        );
    }

    #[test]
    fn an_unknown_or_missing_type_is_refused_naming_the_question() {
        // The old `yes_no` discriminant is gone with the array spelling, and
        // a `type` that is absent or not a string is the schema's business.
        for (question, needle) in [
            (json!({"type": "yes_no", "instructions": "ready"}), "yes_no"),
            (json!({"instructions": "ready"}), "type"),
            (json!({"type": 5, "instructions": "ready"}), "type"),
        ] {
            let mut body = json!({"state": "the input", "questions": {"q1": {}}});
            body["questions"]["q1"] = question;
            assert_refused_at(body, "questions.q1", needle);
        }
    }

    #[test]
    fn an_unknown_key_inside_a_question_is_refused() {
        assert_refused_at(
            json!({
                "state": "the input",
                "questions": {"q1": {"type": "noul", "instructions": "ready",
                                     "bogus": 1}},
            }),
            "questions.q1",
            "bogus",
        );
    }

    #[test]
    fn a_question_without_instructions_is_refused() {
        assert_refused_at(
            json!({
                "state": "the input",
                "questions": {"q1": {"type": "choice", "criteria": {"red": null}}},
            }),
            "questions.q1",
            "instructions",
        );
    }

    #[test]
    fn structured_instructions_render_as_compact_json() {
        // Review Focus 2: a container-valued `instructions` is the shared
        // key-sorted compact rendering, never a refusal and never a panic.
        let questions = questions_of(json!({
            "state": "the input",
            "questions": {"q1": {"type": "noul", "instructions": {"b": 1, "a": [1, 2]}}},
        }));
        match &questions[0].kind {
            Kind::YesNo { question, .. } => assert_eq!(question, "{\"a\":[1,2],\"b\":1}"),
            other => panic!("expected a noul, got {other:?}"),
        }
    }

    #[test]
    fn null_criteria_descriptions_become_bare_options() {
        // Review Focus 3: a null description is "no detail", not an empty
        // one - the option line carries no `" - "` tail.
        let questions = questions_of(json!({
            "state": "the input",
            "questions": {"q1": {"type": "choice", "instructions": "pick one",
                                 "criteria": {"red": null, "blue": "the sky one"}}},
        }));
        match &questions[0].kind {
            Kind::Choice { options, .. } => {
                assert!(
                    matches!(&options[0], ChoiceOption { value, description: None }
                        if value.text() == "red"),
                    "{options:?}"
                );
                assert!(
                    matches!(&options[1], ChoiceOption { value, description }
                        if value.text() == "blue" && description.as_deref() == Some("the sky one")),
                    "{options:?}"
                );
            }
            other => panic!("expected a choice, got {other:?}"),
        }
    }

    #[test]
    fn null_instructions_are_refused_for_every_type() {
        // The wire grammar (spec §1.2) enumerates `instructions` as
        // string/object/array, never null: rendering a null would score the
        // literal word "null", so it is refused as the schema violation it
        // is, naming the question's key.
        for question in [
            json!({"type": "noul", "instructions": null}),
            json!({"type": "choice", "instructions": null, "criteria": {"red": null}}),
            json!({"type": "score", "instructions": null, "criteria": ["none"]}),
        ] {
            assert_refused_at(
                json!({"state": "the input", "questions": {"q1": question}}),
                "questions.q1",
                "instructions",
            );
        }
    }

    #[test]
    fn a_null_score_level_is_refused() {
        // The same grammar enumerates score `criteria` items as
        // string/object/array: a null level would score as the literal word
        // "null" beside the caller's real levels.
        assert_refused_at(
            json!({
                "state": "the input",
                "questions": {"q1": {"type": "score", "instructions": "how bad",
                                     "criteria": ["none", null, "some"]}},
            }),
            "questions.q1",
            "level",
        );
    }

    #[test]
    fn null_choice_descriptions_and_noul_wordings_keep_their_handling() {
        // The refusal above is the grammar's business only where the grammar
        // enumerates it: a null choice description is still "no description"
        // and a null noul wording is still absent wording — serde reads a
        // null into an `Option` field as absent — never a scored "null".
        let questions = questions_of(json!({
            "state": "the input",
            "questions": {
                "c1": {"type": "choice", "instructions": "pick one",
                       "criteria": {"red": null}},
                "n1": {"type": "noul", "instructions": "ready",
                       "criteria": {"true": null, "false": null}},
            },
        }));
        match &questions[0].kind {
            Kind::Choice { options, .. } => assert!(
                matches!(&options[0], ChoiceOption { value, description: None }
                    if value.text() == "red"),
                "a null description is no description: {options:?}"
            ),
            other => panic!("expected a choice, got {other:?}"),
        }
        match &questions[1].kind {
            Kind::YesNo { yes, no, .. } => {
                assert!(yes.is_none(), "a null wording is absent wording: {yes:?}");
                assert!(no.is_none(), "a null wording is absent wording: {no:?}");
            }
            other => panic!("expected a noul, got {other:?}"),
        }
    }

    #[test]
    fn noul_criteria_with_an_unknown_key_are_refused() {
        assert_refused_at(
            json!({
                "state": "the input",
                "questions": {"q1": {"type": "noul", "instructions": "ready",
                                     "criteria": {"true": "confirmed", "yes": "old"}}},
            }),
            "questions.q1",
            "yes",
        );
    }

    #[test]
    fn noul_true_and_false_wordings_land_on_yes_and_no() {
        let questions = questions_of(json!({
            "state": "the input",
            "questions": {"q1": {"type": "noul", "instructions": "ready",
                                 "criteria": {"true": "confirmed", "false": "denied"}}},
        }));
        match &questions[0].kind {
            Kind::YesNo { yes, no, .. } => {
                assert_eq!(yes.as_deref(), Some("confirmed"));
                assert_eq!(no.as_deref(), Some("denied"));
            }
            other => panic!("expected a noul, got {other:?}"),
        }
        // A blank wording drops its line - the shared wording rule, the same
        // one the old `yes`/`no` keys rode.
        let questions = questions_of(json!({
            "state": "the input",
            "questions": {"q1": {"type": "noul", "instructions": "ready",
                                 "criteria": {"true": "  "}}},
        }));
        match &questions[0].kind {
            Kind::YesNo { yes, no, .. } => assert!(yes.is_none() && no.is_none()),
            other => panic!("expected a noul, got {other:?}"),
        }
    }

    #[test]
    fn a_blank_map_key_is_a_blank_id() {
        // A blank key is legal JSON and this surface allows blank ids; the
        // map wire simply cannot repeat one.
        let questions = questions_of(json!({
            "state": "the input",
            "questions": {"": {"type": "noul", "instructions": "ready"}},
        }));
        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].id, "");
    }

    #[test]
    fn duplicate_json_keys_collapse_to_one_question() {
        // The wire cannot deliver two questions under one id: JSON's
        // duplicate keys collapse in the parse, so `check_ids`' duplicate
        // branch is unreachable from this surface by construction. Pinned
        // because that unreachability is a property of the map wire, not of
        // the shared validator, and the shared call stays as the guard.
        let body = r#"{"questions": {"q1": {"type": "noul", "instructions": "ready"},
                                        "q1": {"type": "noul", "instructions": "or"}}}"#;
        let questions = serde_json::from_str::<SystemOneBody>(body)
            .expect("duplicate keys parse")
            .validate()
            .expect("one question, no duplicate to refuse");
        assert_eq!(questions.len(), 1);
        assert_eq!(questions[0].id, "q1");
    }

    #[test]
    fn map_insertion_order_survives_into_the_answer_order() {
        // `preserve_order` carries the caller's key order into the Vec the
        // encoder answers from - the map form's replacement for array order.
        let questions = questions_of(json!({
            "state": "the input",
            "questions": {
                "zeta": {"type": "noul", "instructions": "ready"},
                "alpha": {"type": "noul", "instructions": "ready"},
                "mid": {"type": "noul", "instructions": "ready"},
            },
        }));
        let ids: Vec<&str> = questions.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, ["zeta", "alpha", "mid"]);
    }

    // --- the clef `images` extension -----------------------------------------

    /// The sources a body's `images` array parses to. The questions ride
    /// along because the body parse needs them; only `images_sources` is
    /// under test here, so an item that cannot even parse as the body type
    /// fails as a bug in the test, not as a pass.
    fn sources_of(images: &Value) -> Vec<InputImageSource> {
        let parsed: SystemOneBody = serde_json::from_value(json!({
            "questions": {"q1": {"type": "noul", "instructions": "ready"}},
            "images": images,
        }))
        .expect("the body parses: item refusals are images_sources'");
        images_sources(parsed.images.as_ref()).expect("the items parse")
    }

    /// Asserts an `images` array is refused as a schema violation located at
    /// `at` whose message contains `needle`.
    fn assert_images_refused_at(images: &Value, at: &str, needle: &str) {
        let parsed: SystemOneBody = serde_json::from_value(json!({
            "questions": {"q1": {"type": "noul", "instructions": "ready"}},
            "images": images,
        }))
        .expect("the body parses: item refusals are images_sources'");
        let error = images_sources(parsed.images.as_ref()).expect_err("the item must be refused");
        match error {
            AppError::SchemaViolation { message, at: got } => {
                assert_eq!(got, at, "{message}");
                assert!(
                    message.contains(needle),
                    "the refusal must name {needle}: {message}"
                );
            }
            other => panic!("expected a SchemaViolation at {at}, got {other:?}"),
        }
    }

    #[test]
    fn a_data_url_string_parses_to_a_base64_source() {
        let sources = sources_of(&json!(["data:image/png;base64,aGk="]));
        assert_eq!(sources.len(), 1);
        assert!(
            matches!(&sources[0], InputImageSource::Base64 { media_type, data }
                if media_type == "image/png" && data == "aGk="),
            "{sources:?}"
        );
    }

    #[test]
    fn data_urls_are_read_case_insensitively() {
        // clef's pattern: the structural parts of the prefix fold, so a
        // caller that spells `DaTa` or `IMAGE/Png` sends the same bytes.
        let sources = sources_of(&json!(["DaTa:IMAGE/Png;base64,aGk="]));
        assert!(
            matches!(&sources[0], InputImageSource::Base64 { media_type, data }
                if media_type == "image/png" && data == "aGk="),
            "{sources:?}"
        );
    }

    #[test]
    fn malformed_data_urls_are_refused_naming_the_index() {
        for (item, needle) in [
            (json!("data:image/png;base64,"), "payload"),
            (json!("data:application/pdf;base64,aGk="), "image/"),
            (json!("data:image/png,aGk="), "base64"),
            (json!("a bare string"), "data URL"),
        ] {
            assert_images_refused_at(&json!([item]), "images[0]", needle);
        }
    }

    #[test]
    fn a_base64_object_parses() {
        let sources = sources_of(&json!([{"content_type": "image/png", "base64": "aGk="}]));
        assert!(
            matches!(&sources[0], InputImageSource::Base64 { media_type, data }
                if media_type == "image/png" && data == "aGk="),
            "{sources:?}"
        );
    }

    #[test]
    fn a_blank_content_type_is_refused() {
        // Validated non-blank and then dropped - the decoder sniffs the
        // bytes - but a blank claim is still the schema's business.
        assert_images_refused_at(
            &json!([{"content_type": "  ", "base64": "aGk="}]),
            "images[0]",
            "content_type",
        );
    }

    #[test]
    fn an_unknown_key_inside_an_image_object_is_refused() {
        assert_images_refused_at(
            &json!([{"content_type": "image/png", "base64": "aGk=", "bogus": 1}]),
            "images[0]",
            "bogus",
        );
        // The two object arms are exclusive: a url item carrying a claimed
        // media type is not half of each.
        assert_images_refused_at(
            &json!([{"url": "https://example.com/x.png", "content_type": "image/png"}]),
            "images[0]",
            "content_type",
        );
    }

    #[test]
    fn an_image_object_missing_a_key_is_refused() {
        assert_images_refused_at(
            &json!([{"content_type": "image/png"}]),
            "images[0]",
            "base64",
        );
        assert_images_refused_at(&json!([{"base64": "aGk="}]), "images[0]", "content_type");
    }

    #[test]
    fn a_url_object_parses() {
        // It parses here; the `--vision-allow-remote-images` refusal already
        // lives in the renderer and is not duplicated at the wire.
        let sources = sources_of(&json!([{"url": "https://example.com/x.png"}]));
        assert!(
            matches!(&sources[0], InputImageSource::Url { url }
                if url == "https://example.com/x.png"),
            "{sources:?}"
        );
    }

    #[test]
    fn an_item_of_another_json_type_is_refused() {
        for item in [json!(5), json!(true), json!(null), json!(["a"]), json!({})] {
            assert_images_refused_at(&json!([item]), "images[0]", "images[0]");
        }
    }

    #[test]
    fn the_refusal_names_the_offending_index() {
        // The first item is fine; the location is the second one's index.
        assert_images_refused_at(&json!(["data:image/png;base64,aGk=", 5]), "images[1]", "5");
    }

    #[test]
    fn absent_or_empty_images_yield_no_sources() {
        assert!(
            images_sources(None)
                .expect("absent is not a refusal")
                .is_empty()
        );
        assert!(sources_of(&json!([])).is_empty());
    }
}
