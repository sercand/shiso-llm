//! `POST /v1/responses` and `GET /v1/responses/{id}`.
//!
//! # Continuing a conversation is a store lookup and a splice
//!
//! `previous_response_id` names a transcript the server kept. It is looked up
//! here, refused here if it is unknown, and handed to the mapper as neutral
//! turns - so the mapper stays a pure function of its inputs and can be tested
//! without a store, and so continuing a conversation never re-parses a wire
//! body.
//!
//! An identifier the store does not know is a 400 rather than a silent fresh
//! conversation. A caller that asked to continue and got a fresh answer has been
//! told nothing went wrong while the model was shown none of the history.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse as _, Response};
use axum::{Json, body::Bytes};

use crate::api::pipeline::{self, PromptField};
use crate::api::{blocking, service_tier, sse};
use crate::error::AppError;
use crate::openai::responses_encode::{ResponseOptions, encode_response};
use crate::openai::responses_sse::ResponsesEventEncoder;
use crate::openai::responses_to_ir::to_ir;
use crate::openai::responses_wire::ResponsesBody;
use crate::response_store::{StoredEntry, StoredKind};
use crate::state::AppState;

/// `POST /v1/responses`.
pub async fn responses_endpoint(
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
    let mut parsed: ResponsesBody = serde_json::from_slice(body).map_err(|error| {
        AppError::bad_request(
            format!("the request body is not a response request: {error}"),
            None,
            "invalid_request",
        )
    })?;
    parsed.service_tier = service_tier::resolve(parsed.service_tier, headers)?;
    parsed.validate()?;

    let prior = resolve_prior(state, parsed.previous_response_id.as_deref())?;
    let streaming = parsed.stream.unwrap_or(false);
    let store = parsed.store.unwrap_or(false);
    let options = ResponseOptions {
        id: crate::openai::new_response_id(),
        model: state.model_id.clone(),
        instructions: parsed.instructions.clone(),
        previous_response_id: parsed.previous_response_id.clone(),
        metadata: parsed.metadata.clone(),
        max_output_tokens: parsed.max_output_tokens,
        service_tier: parsed.service_tier,
    };

    let request = to_ir(&parsed, &state.model_id, prior)?;
    let started = pipeline::start(state, &request, PromptField::Input).await?;

    if streaming {
        // A streamed turn is stored the same way a blocking one is: the adapter
        // folds the events it is already passing through into the same response
        // the blocking path builds, and this hook encodes it with the same
        // encoder. The alternative - reassembling the body out of the terminal
        // frame - would be a second assembly of the same body, which is the
        // drift the one-collector rule exists to prevent.
        let hook = store.then(|| store_hook(state, &options, request.messages.clone()));
        return Ok(sse::encode_storing(
            started.events,
            ResponsesEventEncoder::new(options),
            hook,
        ));
    }

    let response = blocking::collect_response(started.events).await?;
    let body = encode_response(&response, &options);
    if store {
        let mut messages = request.messages;
        messages.push(response.message);
        state.response_store.insert(StoredEntry {
            id: options.id.clone(),
            kind: StoredKind::Response,
            model: state.model_id.clone(),
            created_at: response.created,
            messages,
            body: body.clone(),
        });
    }
    Ok(Json(body).into_response())
}

/// What a streamed turn does when it finishes, for a caller that asked to store
/// it.
///
/// Built before the stream starts, because everything it needs - the
/// identifier, the encoding options, the conversation the turn continues - is
/// known then and none of it is reachable from inside the adapter.
fn store_hook(
    state: &AppState,
    options: &ResponseOptions,
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
            kind: StoredKind::Response,
            model,
            created_at,
            messages,
            body,
        });
    })
}

/// The transcript `previous_response_id` names, or the refusal for one the store
/// does not know.
fn resolve_prior(
    state: &AppState,
    previous: Option<&str>,
) -> Result<Vec<shiso_ir::Message>, AppError> {
    let Some(id) = previous else {
        return Ok(Vec::new());
    };
    state
        .response_store
        .get(id, StoredKind::Response)
        .map(|found| found.messages)
        .ok_or_else(|| {
            AppError::bad_request(
                format!(
                    "no stored response {id:?}; it may have expired, or it was never \
                     stored - send store: true on the response you want to continue"
                ),
                Some("previous_response_id"),
                "response_not_found",
            )
        })
}

/// `GET /v1/responses/{id}`.
pub async fn get_stored_response(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    match state.response_store.get(&id, StoredKind::Response) {
        Some(found) => Json(found.body).into_response(),
        None => AppError::NotFound {
            path: format!(
                "/v1/responses/{id}; it may have expired, or it was never stored \
                 - send store: true to keep it"
            ),
        }
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_prior;
    use crate::response_store::{ResponseStore, StoredEntry, StoredKind};
    use crate::scheduler::EngineHandle;
    use crate::state::AppState;
    use serde_json::json;
    use shiso_ir::Message;
    use std::sync::Arc;
    use std::time::Duration;

    /// A state with a store and a queue that goes nowhere: these tests are about
    /// what happens before anything is generated.
    fn state() -> AppState {
        let (handle, _rx) = queue();
        AppState::new(handle, "a-model".to_owned(), 4096)
            .with_response_store(ResponseStore::new(4, Duration::from_secs(60)))
    }

    /// A queue whose receiver is kept alive by the caller's binding.
    fn queue() -> (
        EngineHandle,
        tokio::sync::mpsc::Receiver<crate::scheduler::EngineRequest>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        (
            EngineHandle::new(
                tx,
                Arc::from("a-model"),
                4096,
                1,
                Duration::from_millis(1),
                Arc::new(crate::scheduler::SchedulerStats::new()),
            ),
            rx,
        )
    }

    #[test]
    fn continuing_nothing_starts_from_an_empty_transcript() {
        let state = state();
        assert!(resolve_prior(&state, None).expect("no prior").is_empty());
    }

    #[test]
    fn continuing_a_stored_response_hands_back_its_turns() {
        let state = state();
        state.response_store.insert(StoredEntry {
            id: "resp_1".to_owned(),
            kind: StoredKind::Response,
            model: "a-model".to_owned(),
            created_at: 0,
            messages: vec![Message::user("hello"), Message::assistant("hi")],
            body: json!({}),
        });
        let prior = resolve_prior(&state, Some("resp_1")).expect("a stored transcript");
        assert_eq!(prior.len(), 2);
    }

    /// A fresh answer to a continuation request is a server that says nothing
    /// went wrong while showing the model none of the history.
    #[test]
    fn continuing_a_conversation_the_server_forgot_is_refused_rather_than_restarted() {
        let state = state();
        let error = resolve_prior(&state, Some("resp_missing")).expect_err("not stored");
        assert_eq!(error.code(), "response_not_found");
        assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
    }

    /// One store, two surfaces: a chat identifier must not answer this question.
    #[test]
    fn a_chat_identifier_cannot_be_continued_as_a_response() {
        let state = state();
        state.response_store.insert(StoredEntry {
            id: "chatcmpl-1".to_owned(),
            kind: StoredKind::ChatCompletion,
            model: "a-model".to_owned(),
            created_at: 0,
            messages: vec![Message::user("hello")],
            body: json!({}),
        });
        assert!(resolve_prior(&state, Some("chatcmpl-1")).is_err());
    }
}
