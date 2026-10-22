//! The `POST /v1/decisions` answer, assembled as JSON in the official
//! `OpenAI` Decisions shape: an `answers` array in question order, each answer
//! typed `predicate`, `choice` or `score` and echoing its `name` (or null),
//! and a `usage` object that counts only input tokens.
//!
//! The body is a [`serde_json::Map`] rather than a serde struct because its
//! shape is positional in a way a struct cannot express: the answers keep the
//! request's question order, and the workspace builds `serde_json` with
//! `preserve_order` so the map keeps insertion order into the response bytes.
//! Nothing here computes anything - every number was decided in
//! [`super::core`] and [`shiso_decision::math`] - this only spells them out.
//!
//! Two spellings are shiso's beside the official shape, both documented
//! deviations: `confidence` is the label mass (the official API does not
//! define its formula, and this server's label mass is the quantity it
//! actually measures), and a local scoring model never refuses, so the
//! official `refusal` arm is never sent.

use serde_json::{Map, Value, json};

use crate::decision::core::{Answered, InputTally, Kind, choice_name};
use shiso_decision::math::expected_score;

/// The one prompt format this server answers in, echoed so a caller knows
/// which bytes produced the answer it is reading. (shiso extension.)
const PROMPT_FORMAT_VERSION: u32 = 1;

/// The `name` field: the caller's own name, or null for a question that
/// arrived unnamed and answers by position.
fn name_value(id: &str) -> Value {
    if id.is_empty() {
        Value::Null
    } else {
        Value::String(id.to_owned())
    }
}

/// Encodes the scored answers under `model` (the request's own name, or the
/// server's when the caller named none). `tally` is the prompt side of
/// `usage`: nothing is generated on this surface, so the output count is zero
/// and the total is the input.
pub(crate) fn encode(
    model: String,
    answers: &[Answered],
    tally: InputTally,
    want_token_ids: bool,
) -> Value {
    let answered: Vec<Value> = answers
        .iter()
        .map(|answer| {
            let mut entry = Map::new();
            match &answer.kind {
                Kind::Choice { options, .. } => {
                    entry.insert("type".to_owned(), json!("choice"));
                    entry.insert("name".to_owned(), name_value(&answer.id));
                    // The answer echoes the option's value with the type the
                    // caller sent it under.
                    entry.insert(
                        "choice".to_owned(),
                        choice_name(&answer.kind, answer.top).to_json(),
                    );
                    let probabilities: Vec<Value> = options
                        .iter()
                        .zip(&answer.probabilities)
                        .map(|(option, probability)| {
                            json!({"value": option.value.to_json(), "probability": probability})
                        })
                        .collect();
                    entry.insert(
                        "probabilities".to_owned(),
                        Value::Array(probabilities),
                    );
                    entry.insert("confidence".to_owned(), json!(answer.label_mass));
                    entry.insert("label_mass".to_owned(), json!(answer.label_mass));
                }
                Kind::Score { levels, .. } => {
                    entry.insert("type".to_owned(), json!("score"));
                    entry.insert("name".to_owned(), name_value(&answer.id));
                    entry.insert(
                        "score".to_owned(),
                        json!(expected_score(&answer.probabilities)),
                    );
                    let probabilities: Vec<Value> = levels
                        .iter()
                        .zip(&answer.probabilities)
                        .enumerate()
                        .map(|(index, (level, probability))| {
                            json!({"value": index, "label": level.label, "probability": probability})
                        })
                        .collect();
                    entry.insert(
                        "probabilities".to_owned(),
                        Value::Array(probabilities),
                    );
                    entry.insert("confidence".to_owned(), json!(answer.label_mass));
                    entry.insert("label_mass".to_owned(), json!(answer.label_mass));
                }
                Kind::YesNo { .. } => {
                    // The official `predicate`: the probability that the
                    // condition is true. Entry zero of the probabilities is
                    // `yes` by construction, the labels being exactly `yes`
                    // then `no`.
                    entry.insert("type".to_owned(), json!("predicate"));
                    entry.insert("name".to_owned(), name_value(&answer.id));
                    entry.insert(
                        "probability".to_owned(),
                        json!(answer.probabilities.first().copied().unwrap_or(0.0)),
                    );
                    entry.insert("label_mass".to_owned(), json!(answer.label_mass));
                }
            }
            if want_token_ids {
                // The ids are the ones the model read: with media they include
                // the expanded pad runs, and their count is the prompt side
                // of `usage.input_tokens` (spec §5).
                entry.insert(
                    "prompt_token_ids".to_owned(),
                    json!(answer.prompt_token_ids),
                );
                entry.insert("label_token_ids".to_owned(), json!(answer.label_token_ids));
            }
            Value::Object(entry)
        })
        .collect();

    let mut body = Map::new();
    body.insert("object".to_owned(), json!("decisions"));
    body.insert("model".to_owned(), Value::String(model));
    body.insert(
        "prompt_format_version".to_owned(),
        json!(PROMPT_FORMAT_VERSION),
    );
    body.insert("answers".to_owned(), Value::Array(answered));
    // Nothing is generated on this surface, so the output count is zero and
    // the total is the input - the same accounting a caller would compute
    // from the answers, stated rather than implied. `cached_tokens` is the
    // part of the input a stored prefix covered; nothing is billed for
    // writing a checkpoint, so `cache_write_tokens` stays zero, and scoring
    // runs with reasoning off.
    body.insert(
        "usage".to_owned(),
        json!({
            "input_tokens": tally.input,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": tally.cached},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": tally.input,
        }),
    );
    Value::Object(body)
}

#[cfg(test)]
mod tests {
    use super::encode;
    use crate::decision::core::{
        Answered, ChoiceOption, ChoiceValue, InputTally, Kind, ScoreLevel,
    };
    use serde_json::{Value, json};

    /// One scored answer with every field the encoder reads set by hand, and
    /// the token-id fields at their no-request values. The probabilities are
    /// exact binary fractions so the assertions below can be JSON equality
    /// rather than closeness.
    fn answered(
        id: &str,
        kind: Kind,
        probabilities: &[f64],
        label_mass: f64,
        top: usize,
    ) -> Answered {
        Answered {
            id: id.to_owned(),
            kind,
            probability_answer: false,
            probabilities: probabilities.to_vec(),
            label_mass,
            top,
            prompt_tokens: 0,
            cached_tokens: 0,
            prompt_token_ids: None,
            label_token_ids: None,
        }
    }

    fn body(answers: &[Answered], want_ids: bool) -> Value {
        encode(
            "test-model".to_owned(),
            answers,
            InputTally {
                input: 42,
                cached: 0,
            },
            want_ids,
        )
    }

    /// The entry one answer renders to, on its own.
    fn only_answer(answer: &Answered) -> Value {
        body(std::slice::from_ref(answer), false)["answers"][0].clone()
    }

    /// The key order of one JSON object, for pinning the shape's order.
    fn keys(value: &Value) -> Vec<&str> {
        value
            .as_object()
            .expect("the value is an object")
            .keys()
            .map(String::as_str)
            .collect()
    }

    fn yes_no(id: &str, probabilities: &[f64], label_mass: f64, top: usize) -> Answered {
        answered(
            id,
            Kind::YesNo {
                question: "q".to_owned(),
                yes: None,
                no: None,
            },
            probabilities,
            label_mass,
            top,
        )
    }

    #[test]
    fn a_predicate_answer_carries_the_yes_probability() {
        let answer = yes_no("damaged", &[0.95, 0.05], 0.98, 0);
        assert_eq!(
            only_answer(&answer),
            json!({
                "type": "predicate",
                "name": "damaged",
                "probability": 0.95,
                "label_mass": 0.98,
            })
        );
    }

    #[test]
    fn an_unnamed_answer_echoes_a_null_name() {
        let answer = yes_no("", &[1.0, 0.0], 1.0, 0);
        assert_eq!(only_answer(&answer)["name"], Value::Null);
    }

    #[test]
    fn a_choice_answer_echoes_the_typed_value_and_pairs() {
        let answer = answered(
            "dept",
            Kind::Choice {
                question: "q".to_owned(),
                options: vec![
                    ChoiceOption {
                        value: ChoiceValue::Str("billing".to_owned()),
                        description: None,
                    },
                    ChoiceOption {
                        value: ChoiceValue::Bool(true),
                        description: None,
                    },
                ],
            },
            &[0.95, 0.05],
            0.97,
            0,
        );
        assert_eq!(
            only_answer(&answer),
            json!({
                "type": "choice",
                "name": "dept",
                "choice": "billing",
                "probabilities": [
                    {"value": "billing", "probability": 0.95},
                    {"value": true, "probability": 0.05},
                ],
                "confidence": 0.97,
                "label_mass": 0.97,
            })
        );
    }

    #[test]
    fn a_boolean_choice_wins_by_type_and_echoes_as_a_boolean() {
        let answer = answered(
            "q",
            Kind::Choice {
                question: "q".to_owned(),
                options: vec![
                    ChoiceOption {
                        value: ChoiceValue::Str("true".to_owned()),
                        description: None,
                    },
                    ChoiceOption {
                        value: ChoiceValue::Bool(true),
                        description: None,
                    },
                ],
            },
            &[0.25, 0.75],
            1.0,
            1,
        );
        assert_eq!(only_answer(&answer)["choice"], json!(true));
    }

    #[test]
    fn a_score_answer_weights_the_level_indices() {
        let answer = answered(
            "sev",
            Kind::Score {
                question: "q".to_owned(),
                levels: vec![
                    ScoreLevel {
                        label: "Cosmetic".to_owned(),
                        description: None,
                    },
                    ScoreLevel {
                        label: "Workaround".to_owned(),
                        description: Some("another way".to_owned()),
                    },
                    ScoreLevel {
                        label: "Blocked".to_owned(),
                        description: None,
                    },
                ],
            },
            &[0.25, 0.5, 0.25],
            0.75,
            1,
        );
        assert_eq!(
            only_answer(&answer),
            json!({
                "type": "score",
                "name": "sev",
                "score": 1.0,
                "probabilities": [
                    {"value": 0, "label": "Cosmetic", "probability": 0.25},
                    {"value": 1, "label": "Workaround", "probability": 0.5},
                    {"value": 2, "label": "Blocked", "probability": 0.25},
                ],
                "confidence": 0.75,
                "label_mass": 0.75,
            })
        );
    }

    #[test]
    fn answers_keep_the_request_order() {
        let answers = vec![
            yes_no("b", &[1.0, 0.0], 1.0, 0),
            yes_no("a", &[0.0, 1.0], 1.0, 1),
        ];
        let body = body(&answers, false);
        let names: Vec<&str> = body["answers"]
            .as_array()
            .expect("answers is an array")
            .iter()
            .map(|entry| entry["name"].as_str().expect("a name"))
            .collect();
        assert_eq!(names, ["b", "a"]);
    }

    #[test]
    fn usage_reports_the_official_input_token_shape() {
        let answers = [yes_no("n", &[1.0, 0.0], 1.0, 0)];
        assert_eq!(
            body(&answers, false)["usage"],
            json!({
                "input_tokens": 42,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 42,
            })
        );
    }

    #[test]
    fn the_top_level_keeps_the_shiso_extras_in_order() {
        let answers = [yes_no("n", &[1.0, 0.0], 1.0, 0)];
        let body = body(&answers, false);
        assert_eq!(
            keys(&body),
            [
                "object",
                "model",
                "prompt_format_version",
                "answers",
                "usage"
            ]
        );
        assert_eq!(body["object"], json!("decisions"));
        assert_eq!(body["model"], json!("test-model"));
        assert_eq!(body["prompt_format_version"], json!(1));
    }

    #[test]
    fn a_choice_answer_spells_its_fields_in_the_official_order() {
        let answer = answered(
            "q",
            Kind::Choice {
                question: "q".to_owned(),
                options: vec![
                    ChoiceOption {
                        value: ChoiceValue::Str("a".to_owned()),
                        description: None,
                    },
                    ChoiceOption {
                        value: ChoiceValue::Str("b".to_owned()),
                        description: None,
                    },
                ],
            },
            &[0.5, 0.5],
            1.0,
            0,
        );
        assert_eq!(
            keys(&only_answer(&answer)),
            [
                "type",
                "name",
                "choice",
                "probabilities",
                "confidence",
                "label_mass"
            ]
        );
    }

    #[test]
    fn token_ids_are_appended_only_when_requested() {
        let mut answer = yes_no("n", &[1.0, 0.0], 1.0, 0);
        answer.prompt_token_ids = Some(vec![1, 2]);
        answer.label_token_ids = Some(vec![3, 4]);
        let entry = body(std::slice::from_ref(&answer), true)["answers"][0].clone();
        assert_eq!(entry["prompt_token_ids"], json!([1, 2]));
        assert_eq!(entry["label_token_ids"], json!([3, 4]));
        let entry = only_answer(&answer);
        assert!(
            entry.get("prompt_token_ids").is_none(),
            "no request, no ids: {entry}"
        );
    }
}
