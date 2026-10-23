//! `POST /v1/chat/completions`.
//!
//! # The handler is a sequence of refusals and then one pipeline call
//!
//! Everything that can be refused is refused before a token is generated, in the
//! order a caller would fix it: the body has to parse, the fields have to be in
//! range, the conversation has to be renderable, and the prompt has to leave
//! room to answer. Only then does the request reach the queue, and from that
//! point the two encodings - one response, or a stream of frames - are two
//! readings of the same events.
//!
//! # Why the body is taken as bytes
//!
//! A typed extractor would answer a malformed body with its own error shape,
//! which is not this surface's shape, and a client would see a failure it cannot
//! parse instead of the server's actual complaint. Taking the bytes and parsing
//! them here means every failure on this route is rendered by the one renderer.

use crate::error::ErrorResponse as _;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse as _, Response};

use crate::api::pipeline::{self, PromptField};
use crate::api::{blocking, service_tier, sse};
use crate::error::AppError;
use crate::openai::chat_encode::{EncodeOptions, encode_response};
use crate::openai::chat_sse::ChatChunkEncoder;
use crate::openai::chat_to_ir::to_ir;
use crate::openai::chat_wire::ChatBody;
use crate::response_store::{StoredEntry, StoredKind};
use crate::state::AppState;

/// `POST /v1/chat/completions`.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match answer(&state, &headers, &body).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

/// The handler, with its failures still typed.
async fn answer(state: &AppState, headers: &HeaderMap, body: &Bytes) -> Result<Response, AppError> {
    let mut parsed: ChatBody = serde_json::from_slice(body).map_err(|error| {
        AppError::bad_request(
            format!("the request body is not a chat completion: {error}"),
            None,
            "invalid_request",
        )
    })?;
    parsed.service_tier = service_tier::resolve(parsed.service_tier, headers)?;
    parsed.validate()?;

    let streaming = parsed.stream.unwrap_or(false);
    let include_usage = parsed.include_usage();
    let store = parsed.store.unwrap_or(false);
    let service_tier = parsed.service_tier;
    let return_token_ids = parsed.return_token_ids.unwrap_or(false);

    let request = to_ir(&parsed, &state.model_id)?;
    let options = EncodeOptions {
        id: crate::openai::new_chunk_id(),
        model: state.model_id.clone(),
        service_tier,
        return_token_ids,
    };

    let started = pipeline::start(state, &request, PromptField::Messages).await?;
    if streaming {
        // A streamed turn is stored the same way a blocking one is: the adapter
        // folds the events it is already passing through into the same response
        // the blocking path builds, and this hook encodes it with the same
        // encoder. Nothing here reassembles a body out of frames.
        let hook = store.then(|| store_hook(state, &options, request.messages.clone()));
        return Ok(sse::encode_storing(
            started.events,
            ChatChunkEncoder::new(options, include_usage),
            hook,
        ));
    }

    let response = blocking::collect_response(started.events).await?;
    let body = encode_response(&response, &options);
    if store {
        // The transcript is the conversation the caller sent with the generated
        // turn appended, which is exactly what a continuation would need. It is
        // stored as the neutral turns rather than as the encoded body, because
        // reparsing the body would be a second, weaker copy of the mapping that
        // produced it.
        let mut messages = request.messages;
        messages.push(response.message);
        state.response_store.insert(StoredEntry {
            id: options.id.clone(),
            kind: StoredKind::ChatCompletion,
            model: state.model_id.clone(),
            created_at: response.created,
            messages,
            body: body.clone(),
        });
    }
    Ok(axum::Json(body).into_response())
}

/// What a streamed turn does when it finishes, for a caller that asked to store
/// it.
///
/// Built before the stream starts, because everything it needs - the
/// identifier, the encoding options, the conversation the turn continues - is
/// known then and none of it is reachable from inside the adapter.
fn store_hook(
    state: &AppState,
    options: &EncodeOptions,
    mut messages: Vec<shiso_ir::Message>,
) -> sse::StoreHook {
    let store = Arc::clone(&state.response_store);
    let options = options.clone();
    let model = state.model_id.clone();
    Box::new(move |response: shiso_ir::Response| {
        let body = encode_response(&response, &options);
        let created_at = response.created;
        messages.push(response.message);
        store.insert(StoredEntry {
            id: options.id.clone(),
            kind: StoredKind::ChatCompletion,
            model,
            created_at,
            messages,
            body,
        });
    })
}
