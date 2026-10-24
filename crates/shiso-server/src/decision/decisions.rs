//! `POST /v1/decisions`.
//!
//! The handler is the same sequence of refusals the chat surface runs, just
//! pointed at scoring: the body has to parse, the questions have to be
//! well-shaped, the template knobs have to be ones this surface accepts, and
//! every question has to render and survive its label checks — all before a
//! token is read. Then every question is scored from its own prefill at once,
//! and the answers come back in the request's own order.
//!
//! The body is taken as bytes for the same reason `openai::chat` takes them:
//! a typed extractor would answer a malformed body with its own error shape,
//! and every failure on this route should be rendered by the one renderer.

use crate::error::ErrorResponse as _;
use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse as _, Response};
use shiso_ir::decision::{ReadoutState, Request};
use shiso_runtime::decision::{self, core};

use crate::decision::decisions_encode::encode;
use crate::decision::decisions_wire::{self, DecisionsBody};
use crate::decision::wire_common::TemplateKwargs;
use crate::error::AppError;
use crate::state::AppState;

/// `POST /v1/decisions`.
pub(crate) async fn decisions_endpoint(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Response {
    match answer(&state, &body).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

/// The handler, with its failures still typed.
async fn answer(state: &Arc<AppState>, body: &Bytes) -> Result<Response, AppError> {
    let parsed: DecisionsBody = serde_json::from_slice(body).map_err(|error| {
        AppError::bad_request(
            format!("the request body is not a decisions request: {error}"),
            None,
            "invalid_request",
        )
    })?;
    // What the answer carries but `validate` does not hand back, taken out
    // while the body is still whole: `validate` consumes it to move the
    // questions into the scoring types.
    let model = parsed
        .model
        .clone()
        .unwrap_or_else(|| state.model_id.clone());
    let want_token_ids = parsed.return_prompt_token_ids.unwrap_or(false);
    let kwargs_wire = parsed.chat_template_kwargs.clone();
    let temperature = core::check_temperature(parsed.temperature)?;
    // The same split `validate` applies, taken out while the body is still
    // whole: the text arm is a string `input`, the blocks arm carries the
    // flattened parts of the caller's user messages - and their pictures -
    // to the renderer.
    let input = decisions_wire::decision_input(&parsed.input)?;

    let questions = parsed.validate()?;
    let kwargs = TemplateKwargs::from_wire(kwargs_wire)?;

    // The core answers from whichever way the model reads a decision: a head,
    // a readout, or the label route's own prompts. The input is text here,
    // never a structured value, so a readout reads it verbatim.
    let response = decision::decide(
        state,
        Request {
            input: input.into_ir(),
            readout_state: ReadoutState::Input,
            questions,
            temperature: Some(temperature),
            preserve_thinking: kwargs.preserve_thinking(),
            token_ids: want_token_ids,
        },
    )
    .await?;
    Ok(Json(encode(model, &response, want_token_ids)).into_response())
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use shiso_ir::decision::{Response, Usage};
    use shiso_runtime::decision::scored_answer;

    use crate::decision::decisions_encode::encode;
    use crate::decision::decisions_wire::DecisionsBody;

    /// The wire body's questions go to the IR and the IR's answers come back
    /// out in the shape the encoder has always spelled, for answers fixed by
    /// hand. The probabilities are exact binary fractions, so equality is
    /// exact.
    #[test]
    fn a_body_round_trips_through_the_ir_to_the_same_answer_bytes() {
        let body: DecisionsBody = serde_json::from_value(json!({
            "input": "the ticket",
            "questions": [
                {"type": "predicate", "name": "damaged", "instructions": "is it damaged"},
                {"type": "choice", "name": "dept", "instructions": "which team",
                 "choices": ["billing", true]},
                {"type": "score", "name": "sev", "instructions": "how bad",
                 "levels": [{"label": "Low"}, {"label": "High"}]},
            ],
        }))
        .expect("the body parses");
        let questions = body.validate().expect("the body validates");
        let probabilities = [[0.75, 0.25], [0.25, 0.75], [0.25, 0.75]];
        let tops = [0, 1, 1];
        let answers = questions
            .into_iter()
            .zip(probabilities)
            .zip(tops)
            .map(|((question, probabilities), top)| {
                scored_answer(question, probabilities.to_vec(), 0.5, top)
            })
            .collect();
        let response = Response {
            answers,
            usage: Usage {
                input_tokens: 10,
                cached_tokens: 2,
            },
        };
        let encoded = encode("m".to_owned(), &response, false);
        let want: Value = json!({
            "object": "decisions",
            "model": "m",
            "prompt_format_version": 1,
            "answers": [
                {"type": "predicate", "name": "damaged", "probability": 0.75, "label_mass": 0.5},
                {"type": "choice", "name": "dept", "choice": true,
                 "probabilities": [
                     {"value": "billing", "probability": 0.25},
                     {"value": true, "probability": 0.75},
                 ],
                 "confidence": 0.5, "label_mass": 0.5},
                {"type": "score", "name": "sev", "score": 0.75,
                 "probabilities": [
                     {"value": 0, "label": "Low", "probability": 0.25},
                     {"value": 1, "label": "High", "probability": 0.75},
                 ],
                 "confidence": 0.5, "label_mass": 0.5},
            ],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 2},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 10,
            },
        });
        assert_eq!(encoded, want);
        assert_eq!(
            encoded.to_string(),
            want.to_string(),
            "the answers keep the encoder's key order"
        );
    }
}
