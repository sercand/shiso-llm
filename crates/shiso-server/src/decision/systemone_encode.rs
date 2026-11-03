//! The `POST /v1/systemone` answer, assembled as JSON.
//!
//! The answer shapes are the official System One ones: a noul criterion
//! answers `{"type": "noul", "noul": <yes probability>}`, a choice carries
//! its `probabilities` as a map keyed by option name, and a score carries
//! `legend` and `probabilities` as string-indexed maps. The official fields
//! travel with this surface's kept extras — a documented superset, not a
//! byte-identical upstream response: every answer's `confidence` and
//! `x_label_mass`, a noul's `probabilities` pair in label order, the top
//! level's `state` echo, and `usage.total_tokens`.
//!
//! A [`serde_json::Map`] rather than a serde struct for the decisions
//! encoder's reason: the answers keep the request's order — every question
//! id in the order the request's map carried them — and the workspace builds
//! `serde_json` with `preserve_order` so the map keeps that insertion order
//! into the response bytes. Nothing here computes anything: every number was
//! decided in `shiso_runtime::decision`, and the mass this wire
//! calls `x_label_mass` is the same quantity the decisions wire calls
//! `label_mass`, renamed at the edge.

use serde_json::{Map, Value, json};

use shiso_ir::decision::{AnswerKind, Response};

use crate::decision::wire_common::type_name;

/// Encodes the scored answers. `model` is the server's served model id —
/// this surface never echoes the request's name — and `state` is the
/// request's state rendered to text, blank when it said nothing.
pub(crate) fn encode(model: String, state: String, response: &Response) -> Value {
    let mut answered = Map::new();
    for answer in &response.answers {
        let mut entry = Map::new();
        entry.insert("type".to_owned(), json!(type_name(answer.kind)));
        // Confidence is per kind and comes with the answer: a choice or
        // binary answer is how far the top stands above a uniform draw, a
        // score answer is how concentrated it is around its top level. A
        // noul criterion answers as a yes/no question, so its confidence is
        // the binary one over [yes, no]; the official "the `noul` answer is
        // the yes probability" is the `noul` field itself, spelled below.
        match answer.kind {
            AnswerKind::Choice => {
                entry.insert("choice".to_owned(), json!(answer.choice.text()));
                // The official shape keys the probabilities by option name;
                // the map order is the option order, which is the order the
                // caller's `criteria` map carried.
                let mut probabilities = Map::new();
                for (value, probability) in &answer.probabilities {
                    probabilities.insert(value.text(), json!(probability));
                }
                entry.insert("probabilities".to_owned(), Value::Object(probabilities));
            }
            AnswerKind::Score => {
                entry.insert(
                    "score".to_owned(),
                    json!(answer.expected_score.unwrap_or_default()),
                );
                // The official shape indexes the legend and the
                // probabilities by level number spelled as a string.
                let mut legend = Map::new();
                let mut probabilities = Map::new();
                for (index, (level, probability)) in answer.probabilities.iter().enumerate() {
                    legend.insert(index.to_string(), json!(level.text()));
                    probabilities.insert(index.to_string(), json!(probability));
                }
                entry.insert("legend".to_owned(), Value::Object(legend));
                entry.insert("probabilities".to_owned(), Value::Object(probabilities));
            }
            AnswerKind::YesNo => {
                let pair: Vec<f64> = answer.probabilities.iter().map(|(_, p)| *p).collect();
                if answer.probability_answer {
                    // A noul criterion: the official shape names the type
                    // `noul` and carries the yes probability under `noul` -
                    // entry zero of the probabilities is yes by
                    // construction, the labels being exactly `yes` then
                    // `no`. Re-inserting an existing key replaces its value
                    // in place, so the type keeps the front slot.
                    entry.insert("type".to_owned(), json!("noul"));
                    entry.insert(
                        "noul".to_owned(),
                        json!(
                            pair.first()
                                .expect("a scored yes/no answer carries two label probabilities")
                        ),
                    );
                } else {
                    // A regular yes/no answers with its label word. The
                    // official wire has no spelling that reaches this arm -
                    // every binary question arrives as a noul - but the
                    // flag is data rather than a type, so the arm answers
                    // rather than panicking on a scored response.
                    entry.insert("answer".to_owned(), json!(answer.choice.text()));
                }
                // The kept extra: the pair in label order.
                entry.insert("probabilities".to_owned(), json!(pair));
            }
        }
        entry.insert("confidence".to_owned(), json!(answer.confidence));
        // The mass comes from the scoring pass, where it is independent of
        // the temperature; recomputing it from the probabilities would not
        // recover it, because the softmax over the labels has already thrown
        // away the mass the rest of the distribution carries.
        entry.insert("x_label_mass".to_owned(), json!(answer.label_mass));
        answered.insert(answer.name.clone(), Value::Object(entry));
    }

    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(model));
    body.insert("state".to_owned(), Value::String(state));
    body.insert("answers".to_owned(), Value::Object(answered));
    // Nothing is generated on this surface either, and every input token is
    // one question's prompt counted once - the same accounting the decisions
    // route reports, under this wire's own key names.
    let total_input_tokens = response.usage.input_tokens;
    body.insert(
        "usage".to_owned(),
        json!({
            "input_tokens": total_input_tokens,
            "output_tokens": 0,
            "total_tokens": total_input_tokens,
        }),
    );
    Value::Object(body)
}

#[cfg(test)]
mod tests {
    use super::encode;
    use serde_json::{Value, json};
    use shiso_ir::decision::{
        Answer, Choice, ChoiceValue, Level, Question, QuestionKind, Response, Usage,
    };
    use shiso_runtime::decision::scored_answer;

    /// One scored answer with every field the encoder reads set by hand, and
    /// the token-id fields at their no-request values. The probabilities are
    /// exact binary fractions so the assertions below can be JSON equality
    /// rather than closeness: every derived number (a confidence, an
    /// expected score) lands on an exactly representable value too.
    fn answered(
        id: &str,
        kind: QuestionKind,
        probability_answer: bool,
        probabilities: &[f64],
        label_mass: f64,
        top: usize,
    ) -> Answer {
        scored_answer(
            Question {
                name: id.to_owned(),
                instructions: String::new(),
                kind,
                probability_answer,
            },
            probabilities.to_vec(),
            label_mass,
            top,
        )
    }

    fn response(answers: &[Answer], input_tokens: u64) -> Response {
        Response {
            answers: answers.to_vec(),
            usage: Usage {
                input_tokens,
                cached_tokens: 0,
            },
        }
    }

    /// The entry one answer renders to, on its own.
    fn only_answer(answer: &Answer) -> Value {
        let body = encode(
            "test-model".to_owned(),
            String::new(),
            &response(std::slice::from_ref(answer), 1),
        );
        body["answers"][answer.name.as_str()].clone()
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

    #[test]
    fn a_probability_answer_yes_no_renders_the_official_noul_shape() {
        let answer = answered(
            "n1",
            QuestionKind::YesNo {
                yes: None,
                no: None,
            },
            true,
            &[0.75, 0.25],
            0.5,
            0,
        );
        let entry = only_answer(&answer);
        assert_eq!(
            entry,
            json!({
                "type": "noul",
                "noul": 0.75,
                // The kept extra: the pair in label order.
                "probabilities": [0.75, 0.25],
                "confidence": 0.5,
                "x_label_mass": 0.5,
            })
        );
        assert_eq!(
            keys(&entry),
            [
                "type",
                "noul",
                "probabilities",
                "confidence",
                "x_label_mass"
            ],
            "the type override keeps the front slot and the number is `noul`, not `answer`"
        );
    }

    #[test]
    fn a_choice_answer_keys_its_probabilities_by_option_name() {
        let answer = answered(
            "q1",
            QuestionKind::Choice(vec![
                Choice {
                    value: ChoiceValue::Str("red".to_owned()),
                    description: None,
                },
                Choice {
                    value: ChoiceValue::Str("blue".to_owned()),
                    description: Some("the sky".to_owned()),
                },
                Choice {
                    value: ChoiceValue::Str("green".to_owned()),
                    description: None,
                },
            ]),
            false,
            &[0.125, 0.75, 0.125],
            0.25,
            1,
        );
        let entry = only_answer(&answer);
        assert_eq!(
            entry,
            json!({
                "type": "choice",
                "choice": "blue",
                "probabilities": {"red": 0.125, "blue": 0.75, "green": 0.125},
                "confidence": 0.625,
                "x_label_mass": 0.25,
            })
        );
        assert_eq!(
            keys(&entry["probabilities"]),
            ["red", "blue", "green"],
            "the map is in option order, not sorted: the caller's criteria order is the answer order"
        );
    }

    #[test]
    fn a_score_answer_indexes_its_legend_and_probabilities_by_level() {
        let answer = answered(
            "s1",
            QuestionKind::Score(
                ["none", "some", "lots", "max"]
                    .into_iter()
                    .map(|label| Level {
                        label: label.to_owned(),
                        description: None,
                    })
                    .collect(),
            ),
            false,
            &[0.0, 0.25, 0.5, 0.25],
            0.75,
            2,
        );
        let entry = only_answer(&answer);
        assert_eq!(
            entry,
            json!({
                "type": "score",
                "score": 2.0,
                "legend": {"0": "none", "1": "some", "2": "lots", "3": "max"},
                "probabilities": {"0": 0.0, "1": 0.25, "2": 0.5, "3": 0.25},
                "confidence": 0.5,
                "x_label_mass": 0.75,
            })
        );
        assert_eq!(
            keys(&entry["legend"]),
            ["0", "1", "2", "3"],
            "the legend is a string-indexed map in level order"
        );
        assert_eq!(
            keys(&entry["probabilities"]),
            ["0", "1", "2", "3"],
            "the probabilities are string-indexed too"
        );
    }

    #[test]
    fn a_label_word_yes_no_keeps_the_answer_word_arm() {
        // Unreachable from the official wire - every binary question arrives
        // as a noul - but `probability_answer` is data rather than a type,
        // so the arm must answer with the word rather than panic.
        let answer = answered(
            "q2",
            QuestionKind::YesNo {
                yes: None,
                no: None,
            },
            false,
            &[0.25, 0.75],
            0.5,
            1,
        );
        assert_eq!(
            only_answer(&answer),
            json!({
                "type": "yes_no",
                "answer": "no",
                "probabilities": [0.25, 0.75],
                "confidence": 0.5,
                "x_label_mass": 0.5,
            })
        );
    }

    #[test]
    fn the_top_level_echoes_model_and_state_and_reports_usage() {
        let answers = vec![
            answered(
                "n1",
                QuestionKind::YesNo {
                    yes: None,
                    no: None,
                },
                true,
                &[0.75, 0.25],
                0.5,
                0,
            ),
            answered(
                "q1",
                QuestionKind::Choice(vec![Choice {
                    value: ChoiceValue::Str("red".to_owned()),
                    description: None,
                }]),
                false,
                &[1.0],
                1.0,
                0,
            ),
        ];
        let body = encode(
            "served-model".to_owned(),
            "the input".to_owned(),
            &response(&answers, 42),
        );
        assert_eq!(keys(&body), ["model", "state", "answers", "usage"]);
        assert_eq!(body["model"], "served-model");
        assert_eq!(body["state"], "the input");
        assert_eq!(
            keys(&body["answers"]),
            ["n1", "q1"],
            "the answers keep the request's order"
        );
        assert_eq!(
            body["usage"],
            json!({"input_tokens": 42u64, "output_tokens": 0u64, "total_tokens": 42u64})
        );
        assert_eq!(
            keys(&body["usage"]),
            ["input_tokens", "output_tokens", "total_tokens"],
            "exactly the three usage keys, in order"
        );
    }
}
