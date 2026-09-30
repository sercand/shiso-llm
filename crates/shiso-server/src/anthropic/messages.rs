//! `POST /v1/messages`.
//!
//! # The handler is a sequence of refusals and then one pipeline call
//!
//! The same shape as the chat surface, and deliberately so: the body has to
//! parse, the fields have to be in range, the conversation has to map to turns
//! this server can render, and only then does anything reach the queue. What
//! differs is the wire format on either end, which is the whole of what an
//! adapter is.
//!
//! # Why the body is taken as bytes
//!
//! A typed extractor would answer a malformed body in its own error shape,
//! which is not this surface's shape, and a client of this format would see a
//! failure it cannot parse instead of the server's actual complaint. Taking the
//! bytes and parsing them here means every failure on this route is rendered by
//! the one renderer, in this surface's shape.
//!
//! # The headers this format sends and this server does not read
//!
//! `anthropic-version` and `anthropic-beta` are accepted and not enforced.
//! Nothing here varies by version, so validating one would refuse requests this
//! server can serve in order to enforce a distinction it does not make.
//!
//! # The service class is read from the header only
//!
//! This format's body has a `service_tier` of its own, and it means something
//! else - whether a request may fall back to a different capacity pool - so it
//! is left unread. `X-Service-Tier: flex` asks for the flex class here as on
//! every other surface.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse as _, Response};

use crate::anthropic::messages_encode::{EncodeOptions, encode_response};
use crate::anthropic::messages_sse::MessagesEventEncoder;
use crate::anthropic::messages_to_ir::to_ir;
use crate::anthropic::messages_wire::MessagesBody;
use crate::api::pipeline::{self, PromptField};
use crate::api::{blocking, service_tier, sse};
use crate::error::{ApiSurface, AppError};
use crate::state::AppState;

/// `POST /v1/messages`.
pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match answer(&state, &headers, &body).await {
        Ok(response) => response,
        // Every failure on this route is rendered in this format's error shape,
        // whichever layer raised it: a client of this family parses one body
        // shape, and a 400 it cannot read is a 400 it reports as a transport
        // failure.
        Err(error) => error.on(ApiSurface::Anthropic).into_response(),
    }
}

/// The handler, with its failures still typed.
async fn answer(state: &AppState, headers: &HeaderMap, body: &Bytes) -> Result<Response, AppError> {
    let parsed: MessagesBody = serde_json::from_slice(body).map_err(|error| {
        AppError::bad_request(
            format!("the request body is not a message request: {error}"),
            None,
            "invalid_request",
        )
    })?;
    parsed.validate()?;

    let streaming = parsed.is_streaming();
    let options = EncodeOptions {
        id: super::new_message_id(),
        model: state.model_id.clone(),
    };

    let mut request = to_ir(&parsed, &state.model_id)?;
    if let Some(tier) = service_tier::header_tier(headers)? {
        request.service_class = tier.class();
    }
    let started = pipeline::start(state, &request, PromptField::Messages).await?;
    if streaming {
        return Ok(sse::encode(
            started.events,
            MessagesEventEncoder::new(options).with_input_tokens(started.prompt_tokens),
        ));
    }

    let response = blocking::collect_response(started.events).await?;
    Ok(axum::Json(encode_response(&response, &options)).into_response())
}
