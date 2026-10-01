use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::{Extension, Json};
use serde_json::json;

use crate::state::AppState;
use crate::tokens::KeyRecord;

/// `GET /v1/models`: the models the calling key may use, in OpenAI's list
/// shape by default and Anthropic's when the request carries an
/// `anthropic-version` header (the Anthropic SDK always sends one, OpenAI
/// SDKs never do). Canonical names only — an alias resolves to the same
/// route, so listing it would show one model twice.
pub(crate) async fn list_models(
    State(state): State<AppState>,
    Extension(key): Extension<Arc<KeyRecord>>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let routing = state.routing();
    let mut names: Vec<&str> = routing
        .models
        .iter()
        .map(|m| m.name.as_str())
        .filter(|name| key.allows_model(name))
        .collect();
    names.sort_unstable();
    Json(models_response(
        &names,
        headers.contains_key("anthropic-version"),
    ))
}

/// The catalog stores no creation time, so both shapes advertise the epoch
/// rather than a fake timestamp that changes between calls.
fn models_response(names: &[&str], anthropic: bool) -> serde_json::Value {
    if anthropic {
        let data: Vec<_> = names
            .iter()
            .map(|name| {
                json!({
                    "type": "model",
                    "id": name,
                    "display_name": name,
                    "created_at": "1970-01-01T00:00:00Z",
                })
            })
            .collect();
        json!({
            "data": data,
            "first_id": names.first(),
            "last_id": names.last(),
            "has_more": false,
        })
    } else {
        let data: Vec<_> = names
            .iter()
            .map(|name| {
                json!({
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "llm-router",
                })
            })
            .collect();
        json!({ "object": "list", "data": data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_shape() {
        let body = models_response(&["a-model", "b-model"], false);
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["id"], "a-model");
        assert_eq!(body["data"][0]["object"], "model");
        assert_eq!(body["data"][1]["id"], "b-model");
    }

    #[test]
    fn anthropic_shape() {
        let body = models_response(&["a-model", "b-model"], true);
        assert_eq!(body["data"][0]["type"], "model");
        assert_eq!(body["data"][0]["id"], "a-model");
        assert_eq!(body["data"][0]["display_name"], "a-model");
        assert_eq!(body["first_id"], "a-model");
        assert_eq!(body["last_id"], "b-model");
        assert_eq!(body["has_more"], false);
    }

    #[test]
    fn empty_list_has_null_cursors() {
        let body = models_response(&[], true);
        assert_eq!(body["data"].as_array().unwrap().len(), 0);
        assert!(body["first_id"].is_null());
        assert!(body["last_id"].is_null());
    }
}
