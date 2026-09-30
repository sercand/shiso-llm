//! The `POST /v1/systemone` answer, assembled as JSON.
//!
//! A [`serde_json::Map`] rather than a serde struct for the decisions
//! encoder's reason: the answers keep the request's order — every question id
//! in request order, then every noul criterion id — and the workspace builds
//! `serde_json` with `preserve_order` so the map keeps that insertion order
//! into the response bytes. Nothing here computes anything: every number was
//! decided in [`super::core`] and [`super::math`], and the mass this wire
//! calls `x_label_mass` is the same quantity the decisions wire calls
//! `label_mass`, renamed at the edge.

use serde_json::{Map, Value, json};

use crate::decision::core::{Answered, Kind, answer_label, choice_name, type_name};
use crate::decision::math::{choice_confidence, expected_score, score_confidence};

/// Encodes the scored answers. `model` is the server's served model id —
/// this surface never echoes the request's name — and `state` is the
/// request's state rendered to text, blank when it said nothing.
pub(crate) fn encode(
    model: String,
    state: String,
    answers: &[Answered],
    total_input_tokens: u64,
) -> Value {
    let mut answered = Map::new();
    for answer in answers {
        let mut entry = Map::new();
        entry.insert("type".to_owned(), json!(type_name(&answer.kind)));
        // Confidence is per kind: a choice or binary answer is how far the
        // top stands above a uniform draw, a score answer is how
        // concentrated it is around its top level. A noul criterion answers
        // as a yes/no question, so its confidence is the binary one over
        // [yes, no] - and its probabilities array is where the design's
        // "the noul answer is the yes probability" lives: entry zero is yes.
        let confidence = match &answer.kind {
            Kind::Choice { .. } => {
                entry.insert(
                    "choice".to_owned(),
                    json!(choice_name(&answer.kind, answer.top)),
                );
                choice_confidence(&answer.probabilities)
            }
            Kind::Score { levels, .. } => {
                entry.insert(
                    "score".to_owned(),
                    json!(expected_score(&answer.probabilities)),
                );
                entry.insert("legend".to_owned(), json!(levels));
                score_confidence(&answer.probabilities)
            }
            Kind::YesNo { .. } => {
                entry.insert(
                    "answer".to_owned(),
                    json!(answer_label(&answer.kind, answer.top)),
                );
                choice_confidence(&answer.probabilities)
            }
        };
        entry.insert("probabilities".to_owned(), json!(answer.probabilities));
        entry.insert("confidence".to_owned(), json!(confidence));
        // The mass comes from the scoring pass, where it is independent of
        // the temperature; recomputing it from the probabilities would not
        // recover it, because the softmax over the labels has already thrown
        // away the mass the rest of the distribution carries.
        entry.insert("x_label_mass".to_owned(), json!(answer.label_mass));
        answered.insert(answer.id.clone(), Value::Object(entry));
    }

    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(model));
    body.insert("state".to_owned(), Value::String(state));
    body.insert("answers".to_owned(), Value::Object(answered));
    // Nothing is generated on this surface either, and every input token is
    // one question's prompt counted once - the same accounting the decisions
    // route reports, under this wire's own key names.
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
