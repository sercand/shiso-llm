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

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse as _, Response};

use crate::decision::core::{TemplateKwargs, prepare_question, run_questions};
use crate::decision::decisions_wire;
use crate::decision::systemone_encode::encode;
use crate::decision::systemone_wire::{self, SystemOneBody};
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
    // `images` is taken out the same way: it is not a scored text, it is the
    // media the blocks arm renders, and its item refusals are the wire's.
    let images_wire = parsed.images.take();
    let questions = parsed.validate()?;
    let sources = systemone_wire::images_sources(images_wire.as_ref())?;
    let kwargs = TemplateKwargs::from_wire(kwargs_wire)?;

    // Every question is prepared before any is scored, as on the decisions
    // route, so a refusal to score the last criterion is a refusal rather
    // than a partial answer already paid for. The scoring knobs this wire
    // refused are gone, so temperature is 1 and no caller asked for token
    // ids — `false` here because nothing can have said otherwise.
    let mut prepared = Vec::with_capacity(questions.len());
    // Composition (spec §3): images before the state, clef's rule. With no
    // images the text arm is taken byte-for-byte as it always was; with
    // them the state enters as a *trimmed* text block, which is what keeps
    // the question-wording seam (`"\n\n" + wording` after a text block)
    // byte-identical to the string path's `trim() + "\n\n" + wording`. A
    // blank state sends no text block at all: image-only is valid.
    let input = if sources.is_empty() {
        decisions_wire::DecisionInput::Text(state_text.clone())
    } else {
        let mut blocks = Vec::with_capacity(sources.len() + 1);
        blocks.extend(sources.into_iter().map(decisions_wire::InputBlock::Image));
        let trimmed = state_text.trim();
        if !trimmed.is_empty() {
            blocks.push(decisions_wire::InputBlock::Text {
                text: trimmed.to_owned(),
            });
        }
        decisions_wire::DecisionInput::Blocks(blocks)
    };
    for question in &questions {
        prepared.push(prepare_question(state, &input, question, &kwargs, false)?);
    }
    let answers = run_questions(state, prepared, 1.0, false).await?;
    let total_input_tokens = answers.iter().map(|answer| answer.prompt_tokens).sum();
    Ok(Json(encode(
        state.model_id.clone(),
        state_text,
        &answers,
        total_input_tokens,
    ))
    .into_response())
}
