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

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse as _, Response};

use crate::decision::core::{self, TemplateKwargs, prepare_question, run_questions};
use crate::decision::decisions_encode::encode;
use crate::decision::decisions_wire::{self, DecisionsBody};
use crate::decision::head;
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

    // A decision checkpoint answers every question from one prompt through
    // its head, at the caller's temperature, laid out the way its family
    // lays it out.
    if let Some(layout) = state.decision_layout.as_deref() {
        let (text, media) = head::split_input(&input);
        let (answers, prompt_tokens) = head::run(
            state,
            layout,
            &text,
            media,
            questions,
            temperature,
            want_token_ids,
        )
        .await?;
        return Ok(Json(encode(model, &answers, prompt_tokens, want_token_ids)).into_response());
    }

    // Every question is prepared before any is scored: a refusal to score the
    // third question should reach the caller as a refusal, not as a partial
    // answer with the first two questions already paid for.
    let mut prepared = Vec::with_capacity(questions.len());
    for question in &questions {
        prepared.push(prepare_question(
            state,
            &input,
            question,
            &kwargs,
            want_token_ids,
        )?);
    }
    let answers = run_questions(state, prepared, temperature, want_token_ids).await?;
    let total_prompt_tokens = answers.iter().map(|answer| answer.prompt_tokens).sum();
    Ok(Json(encode(model, &answers, total_prompt_tokens, want_token_ids)).into_response())
}
