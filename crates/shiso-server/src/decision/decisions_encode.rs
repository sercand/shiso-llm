//! The `POST /v1/decisions` answer, assembled as JSON.
//!
//! The body is a [`serde_json::Map`] rather than a serde struct because its
//! shape is positional in a way a struct cannot express: the answers keep the
//! request's question order, and the workspace builds `serde_json` with
//! `preserve_order` so the map keeps insertion order into the response bytes.
//! Nothing here computes anything - every number was decided in
//! [`super::core`] and [`super::math`] - this only spells them out.

use serde_json::{Map, Value, json};

use crate::decision::core::{Answered, Kind, answer_label, choice_name, type_name};
use crate::decision::math::expected_score;

/// The one prompt format this server answers in, echoed so a caller knows
/// which bytes produced the answer it is reading.
const PROMPT_FORMAT_VERSION: u32 = 1;

/// Encodes the scored answers under `model` (the request's own name, or the
/// server's when the caller named none).
pub(crate) fn encode(
    model: String,
    answers: &[Answered],
    total_prompt_tokens: u64,
    want_token_ids: bool,
) -> Value {
    let mut answered = Map::new();
    for answer in answers {
        let mut entry = Map::new();
        entry.insert("type".to_owned(), json!(type_name(&answer.kind)));
        match &answer.kind {
            Kind::Choice { .. } => {
                entry.insert(
                    "choice".to_owned(),
                    json!(choice_name(&answer.kind, answer.top)),
                );
            }
            Kind::Score { .. } => {
                entry.insert(
                    "score".to_owned(),
                    json!(expected_score(&answer.probabilities)),
                );
            }
            Kind::YesNo { .. } => {
                entry.insert(
                    "answer".to_owned(),
                    json!(answer_label(&answer.kind, answer.top)),
                );
            }
        }
        entry.insert("probabilities".to_owned(), json!(answer.probabilities));
        entry.insert("label_mass".to_owned(), json!(answer.label_mass));
        if want_token_ids {
            // The ids are the ones the model read: with media they include
            // the expanded pad runs, and their count is the prompt side of
            // `usage.prompt_tokens` (spec §5).
            entry.insert(
                "prompt_token_ids".to_owned(),
                json!(answer.prompt_token_ids),
            );
            entry.insert("label_token_ids".to_owned(), json!(answer.label_token_ids));
        }
        answered.insert(answer.id.clone(), Value::Object(entry));
    }

    let mut body = Map::new();
    body.insert("object".to_owned(), json!("decisions"));
    body.insert("model".to_owned(), Value::String(model));
    body.insert(
        "prompt_format_version".to_owned(),
        json!(PROMPT_FORMAT_VERSION),
    );
    body.insert("answers".to_owned(), Value::Object(answered));
    // Nothing is generated on this surface, so the completion count is zero
    // and the total is the prompts - the same accounting a caller would
    // compute from the answers, stated rather than implied.
    body.insert(
        "usage".to_owned(),
        json!({
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": total_prompt_tokens,
        }),
    );
    Value::Object(body)
}
