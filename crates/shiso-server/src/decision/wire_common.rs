//! The wire-side helpers both decision surfaces share: how a JSON value
//! becomes prompt text, which template knobs a body may carry, and how a
//! structured state is spelled for a readout layout.
//!
//! Everything here reads or writes `serde_json`, which the typed core in
//! `shiso_runtime::decision` deliberately does not know. What it hands the core
//! is IR: a string, a `preserve_thinking` flag, a
//! [`shiso_ir::decision::ReadoutState`].

use serde_json::{Map, Value};
use shiso_ir::decision::{AnswerKind, ChoiceValue, ReadoutState};
use shiso_runtime::decision;

use crate::error::AppError;

/// The text a JSON input value contributes to a prompt.
///
/// A string is itself; objects and arrays render as compact JSON and other
/// scalars as their JSON form. Object keys are sorted because these bytes are
/// scored: the same logical object must render the same bytes however the
/// client happened to order its keys, and the shared parser preserves the
/// order it was sent in, so sorting is done here rather than assumed.
pub(crate) fn render_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(&canonical(other)).unwrap_or_else(|_| other.to_string()),
    }
}

/// A copy of `value` with every object's keys in sorted order, so serialising
/// it is byte-deterministic.
///
/// Recursion depth is bounded by the parser's own nesting limit: a value this
/// walks was parsed, and parsing is where runaway nesting is refused.
fn canonical(value: &Value) -> Value {
    use Value;
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut sorted = Map::with_capacity(keys.len());
            for key in keys {
                sorted.insert(key.clone(), canonical(&map[key]));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// The question line as text: absent is blank.
///
/// Both surfaces spell the question key the same way and render it with the
/// shared rule, so the reading lives here; whether a blank is a refusal or a
/// dropped line is each surface's [`check_question_text`] call, not this
/// function's business.
pub(crate) fn question_text(value: Option<&Value>) -> String {
    value.map_or_else(String::new, render_text)
}

/// The `yes`/`no` wording: absent or blank is no wording at all, which drops
/// that line from the block.
pub(crate) fn wording(value: Option<&Value>) -> Option<String> {
    let text = value.map(render_text)?;
    (!text.trim().is_empty()).then_some(text)
}

/// Refuses a field the question's type has no place for - `options` on a
/// score question is a caller expecting a different question than the one its
/// `type` says will be scored.
///
/// Each wire calls this with its own struct in hand - the question's id, the
/// type word and the field-presence table are arguments, because the fields
/// belong to the wire - while the refusal itself is shared: the same sentence
/// naming the same field, the same `questions` param, a 400 on both routes.
/// A schema difference between the surfaces (400-versus-422) is about what
/// the *body* says, not about what a misplaced field means to the scored
/// bytes, so this refusal does not split.
///
/// # Errors
///
/// [`AppError::BadRequest`] naming the misplaced field and the question.
pub(crate) fn refuse_fields_other_than(
    id: &str,
    kind: &str,
    present: &[(&str, bool)],
    allowed: &[&str],
) -> Result<(), AppError> {
    for (field, present) in present {
        if *present && !allowed.contains(field) {
            return Err(AppError::bad_request(
                format!(
                    "{}: {field} does not belong on a \"{kind}\" question",
                    decision::core::question_name(id)
                ),
                Some("questions"),
                "invalid_request",
            ));
        }
    }
    Ok(())
}

/// Chat-template knobs a decision request sent, validated and kept raw.
///
/// Every key rule fires once, in [`Self::from_wire`], so the refusal names
/// the offending key while the request is still being read; [`Self::apply`]
/// only lowers what survived.
#[derive(Debug)]
pub(crate) struct TemplateKwargs(Map<String, Value>);

impl TemplateKwargs {
    /// Accepts exactly three keys: `enable_thinking` (which must say false or
    /// nothing, because scoring never opens a thinking block),
    /// `reasoning_effort` (a spelling this server knows, or nothing), and
    /// `preserve_thinking` (a boolean, or nothing). Every other key is a 400.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming the key or the spelling at fault.
    pub(crate) fn from_wire(map: Option<Map<String, Value>>) -> Result<Self, AppError> {
        let mut checked = Map::new();
        for (key, value) in map.unwrap_or_default() {
            match key.as_str() {
                // The answer is the token right after the question block;
                // a thinking block would open at that position and score a
                // reasoning token instead of the label. `false` is accepted
                // because callers that send the key everywhere should not
                // have to special-case this surface.
                "enable_thinking" if matches!(value, Value::Bool(false) | Value::Null) => {}
                "enable_thinking" => {
                    return Err(AppError::bad_request(
                        "a decision is scored from one prefill with thinking off; \
                         enable_thinking must be false",
                        Some("chat_template_kwargs"),
                        "invalid_request",
                    ));
                }
                "reasoning_effort" => match &value {
                    Value::Null => {}
                    Value::String(spelling) => {
                        if shiso_ir::Effort::parse(spelling).is_none() {
                            return Err(AppError::bad_request(
                                format!(
                                    "reasoning_effort {spelling:?} is not a level this server knows"
                                ),
                                Some("chat_template_kwargs"),
                                "invalid_request",
                            ));
                        }
                    }
                    other => {
                        return Err(AppError::bad_request(
                            format!("reasoning_effort must be a string or null, got {other}"),
                            Some("chat_template_kwargs"),
                            "invalid_request",
                        ));
                    }
                },
                "preserve_thinking" if matches!(value, Value::Bool(_) | Value::Null) => {}
                "preserve_thinking" => {
                    return Err(AppError::bad_request(
                        "preserve_thinking must be a boolean or null",
                        Some("chat_template_kwargs"),
                        "invalid_request",
                    ));
                }
                other => {
                    return Err(AppError::bad_request(
                        format!(
                            "chat_template_kwargs key {other:?} is not accepted on this surface"
                        ),
                        Some("chat_template_kwargs"),
                        "invalid_request",
                    ));
                }
            }
            checked.insert(key, value);
        }
        Ok(Self(checked))
    }

    /// The one knob that reaches the render: whether earlier thinking is kept.
    /// The surviving knobs are lowered onto the request's own fields.
    pub(crate) fn preserve_thinking(&self) -> Option<bool> {
        // `enable_thinking` can only have said false or nothing, which is
        // what every scoring render already does, so it lowers to no change.
        // `reasoning_effort` was parsed and checked and is deliberately not
        // lowered: scoring renders with `ThinkingDirective::Off`, so no rung
        // would ever reach a template. A caller that named one gets the same
        // answer as one that did not, and the check exists to refuse a typo,
        // not to promise a thinking budget this surface cannot spend.
        match self.0.get("preserve_thinking") {
            Some(Value::Bool(keep)) => Some(*keep),
            _ => None,
        }
    }
}

/// The value as JSON, for an answer that echoes it back with its type.
pub(crate) fn choice_value_json(value: &ChoiceValue) -> Value {
    match value {
        ChoiceValue::Str(text) => Value::String(text.clone()),
        ChoiceValue::Bool(value) => Value::Bool(*value),
    }
}

/// The wire word for a kind of answer, used in `/v1/systemone` answers.
pub(crate) fn type_name(kind: AnswerKind) -> &'static str {
    match kind {
        AnswerKind::Choice => "choice",
        AnswerKind::Score => "score",
        AnswerKind::YesNo => "yes_no",
    }
}

/// The state a readout reads, as the layouts that spell it differently get it.
///
/// A missing or `null` state is no state, and so is a blank string: the
/// caller who sent `""` beside a picture meant the picture to be the whole
/// state, and the layout writes no state block for it. Any other string is
/// the caller's text verbatim, untrimmed, because these bytes are scored. A
/// structured value is spelled both ways; which one a layout reads is the
/// core's call, made against the layout it has.
pub(crate) fn readout_state(state: Option<&Value>) -> ReadoutState {
    match state {
        None | Some(Value::Null) => ReadoutState::Absent,
        Some(Value::String(_)) => ReadoutState::Input,
        Some(structured) => ReadoutState::Structured {
            compact: render_text(structured),
            // Serialising a parsed value cannot fail; the compact form is the
            // fallback only so the type says so.
            indented: serde_json::to_string_pretty(structured)
                .unwrap_or_else(|_| structured.to_string()),
        },
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use shiso_ir::decision::ReadoutState;

    use super::{readout_state, render_text};

    #[test]
    fn render_text_passes_strings_and_compacts_containers() {
        assert_eq!(render_text(&json!("plain")), "plain");
        assert_eq!(
            render_text(&json!({"b": 1, "a": [1, 2]})),
            "{\"a\":[1,2],\"b\":1}"
        );
        assert_eq!(render_text(&json!(42)), "42");
    }

    #[test]
    fn a_structured_state_is_spelled_both_ways_for_the_family_to_pick() {
        let state = json!({"b": 1, "a": ["é", true]});
        assert_eq!(
            readout_state(Some(&state)),
            ReadoutState::Structured {
                compact: "{\"a\":[\"é\",true],\"b\":1}".to_owned(),
                indented: "{\n  \"b\": 1,\n  \"a\": [\n    \"é\",\n    true\n  ]\n}".to_owned(),
            }
        );
    }

    #[test]
    fn a_missing_or_null_state_is_no_state_and_a_string_is_the_input_text() {
        assert_eq!(readout_state(None), ReadoutState::Absent);
        assert_eq!(readout_state(Some(&json!(null))), ReadoutState::Absent);
        assert_eq!(readout_state(Some(&json!("  "))), ReadoutState::Input);
        assert_eq!(readout_state(Some(&json!(" hi\n"))), ReadoutState::Input);
    }
}
