//! `POST /v1/systemone`.
//!
//! The same scoring engine `/v1/decisions` runs, behind the System One wire:
//! the body has to parse into this schema - map-form questions keyed by id,
//! `noul`/`choice`/`score` - the decisions-only knobs have to say nothing,
//! the questions have to be well-shaped under the relaxed validation, the
//! template knobs have to be ones this surface accepts, and every question
//! has to render and survive its label checks — all before a token is read.
//! Then everything is scored from its own prefill at once, and the answers
//! come back keyed by the caller's ids, questions first and criteria after.
//!
//! Media arrives in the clef `images` extension: a top-level array of
//! embedded images scored *before* the state, composed here into the
//! decisions route's media pipeline as image blocks ahead of the state's
//! text block — the scheduler, renderer and media cache are not systemone's
//! business at all. `state` itself stays text-only.
//!
//! Two refusals differ from the decisions route, and only two. A body that
//! does not match the schema answers 422 with a location, because these
//! clients fix whole documents rather than one parameter; every other
//! refusal — bounds, blanks, duplicates, the model form — is the same 400
//! the shared validators raise. And the response model is the model this
//! server serves: the request's name is read only to refuse the adapter form
//! and then dropped, so no answer can claim to come from a model the caller
//! asked for but this server did not load.
//!
//! The body is taken as bytes for the same reason the decisions handler takes
//! them: a typed extractor would answer a malformed body with its own error
//! shape, and every failure on this route — 400 or 422 — should be rendered
//! by the one renderer.

use crate::error::ErrorResponse as _;
use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse as _, Response};
use shiso_ir::decision::{Input, Request};
use shiso_runtime::decision;

use crate::decision::decisions_wire::InputImageSource;
use crate::decision::systemone_encode::encode;
use crate::decision::systemone_wire::{self, SystemOneBody};
use crate::decision::wire_common::{self, TemplateKwargs};
use crate::error::AppError;
use crate::state::AppState;

/// `POST /v1/systemone`.
pub(crate) async fn systemone_endpoint(
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
    // A parse failure here is a schema violation, not a bad request: serde's
    // line and column are rendered into the `at` field because the fix is an
    // edit at a place in the document, not a different value of a parameter.
    let mut parsed: SystemOneBody =
        serde_json::from_slice(body).map_err(|error| AppError::SchemaViolation {
            message: format!("the request body is not a systemone request: {error}"),
            at: format!("line {} column {}", error.line(), error.column()),
        })?;
    // The decisions-only knobs are refused before anything is validated: a
    // caller that set one asked the other route, and saying so beats
    // quietly scoring a request that expected different numbers back.
    systemone_wire::refuse_decisions_only("temperature", parsed.temperature.as_ref())?;
    systemone_wire::refuse_decisions_only(
        "prompt_format_version",
        parsed.prompt_format_version.as_ref(),
    )?;
    systemone_wire::refuse_decisions_only(
        "return_prompt_token_ids",
        parsed.return_prompt_token_ids.as_ref(),
    )?;

    // What `validate` consumes on its way to the scoring types, taken out
    // while the body is still whole. The request's model name is deliberately
    // not among them: the response names the served model.
    let kwargs_wire = parsed.chat_template_kwargs.clone();
    let state_text = parsed.state_text();
    // A readout family spells a structured state its own way, from the value
    // the caller sent rather than from the wire's compact rendering of it.
    let readout_state = wire_common::readout_state(parsed.state.as_ref());
    // `images` is taken out the same way: it is not a scored text, it is the
    // media the blocks arm renders, and its item refusals are the wire's.
    let images_wire = parsed.images.take();
    let questions = parsed.validate()?;
    let sources = systemone_wire::images_sources(images_wire.as_ref())?;
    let kwargs = TemplateKwargs::from_wire(kwargs_wire)?;

    // The state, with the pictures scored before it (spec §3, clef's rule):
    // with no images the text arm is taken byte-for-byte as it always was.
    // Where the pictures go in a decision checkpoint's prompt is its layout's
    // call, and the state travels as the caller sent it.
    let input = if sources.is_empty() {
        Input::Text(state_text.clone())
    } else {
        Input::Pictured {
            media: sources
                .into_iter()
                .map(InputImageSource::into_source)
                .collect(),
            text: state_text.clone(),
        }
    };

    // The scoring knobs this wire refused are gone, so the temperature is the
    // default and no caller asked for token ids - nothing can have said
    // otherwise.
    let response = decision::decide(
        state,
        Request {
            input,
            readout_state,
            questions,
            temperature: None,
            preserve_thinking: kwargs.preserve_thinking(),
            token_ids: false,
        },
    )
    .await?;
    Ok(Json(encode(state.model_id.clone(), state_text, &response)).into_response())
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use shiso_ir::decision::{Response, Usage};
    use shiso_runtime::decision::scored_answer;

    use crate::decision::systemone_encode::encode;
    use crate::decision::systemone_wire::SystemOneBody;

    /// The body's questions go to the IR and the IR's answers come back out
    /// in the shape the encoder has always spelled: the `noul` is the yes
    /// probability, and the derived confidences are the core's.
    #[test]
    fn a_body_round_trips_through_the_ir_to_the_same_answer_bytes() {
        let body: SystemOneBody = serde_json::from_value(json!({
            "state": "the input",
            "questions": {
                "c1": {"type": "choice", "instructions": "pick one",
                       "criteria": {"red": null, "blue": null}},
                "s1": {"type": "score", "instructions": "how bad",
                       "criteria": ["none", "some"]},
                "n1": {"type": "noul", "instructions": "ready"},
            },
        }))
        .expect("the body parses");
        let questions = body.validate().expect("the body validates");
        // The noul's yes is entry zero and wins; the others put 0.75 on
        // their second option.
        let answers = questions
            .into_iter()
            .map(|question| {
                if question.name == "n1" {
                    scored_answer(question, vec![0.75, 0.25], 0.5, 0)
                } else {
                    scored_answer(question, vec![0.25, 0.75], 0.5, 1)
                }
            })
            .collect();
        let response = Response {
            answers,
            usage: Usage {
                input_tokens: 10,
                cached_tokens: 0,
            },
        };
        let encoded = encode("m".to_owned(), "the input".to_owned(), &response);
        let want: Value = json!({
            "model": "m",
            "state": "the input",
            "answers": {
                "c1": {"type": "choice", "choice": "blue",
                       "probabilities": {"red": 0.25, "blue": 0.75},
                       "confidence": 0.5, "x_label_mass": 0.5},
                "s1": {"type": "score", "score": 0.75,
                       "legend": {"0": "none", "1": "some"},
                       "probabilities": {"0": 0.25, "1": 0.75},
                       "confidence": 0.5, "x_label_mass": 0.5},
                "n1": {"type": "noul", "noul": 0.75, "probabilities": [0.75, 0.25],
                       "confidence": 0.5, "x_label_mass": 0.5},
            },
            "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
        });
        assert_eq!(encoded, want);
        assert_eq!(encoded.to_string(), want.to_string());
    }
}
