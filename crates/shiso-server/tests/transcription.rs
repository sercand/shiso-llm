//! The transcription surface and the audio-only server, without a device.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::Value;
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use tower::ServiceExt as _;

async fn call(state: Arc<AppState>, request: Request<Body>) -> (StatusCode, Value) {
    let response = build_router(state)
        .oneshot(request)
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn a_generation_route_on_a_server_without_a_language_model_says_so() {
    let state = Arc::new(AppState::without_language_model());
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .expect("builds");
    let (status, body) = call(state, request).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "model_not_loaded", "{body}");
}

#[tokio::test]
async fn a_server_without_a_language_model_lists_none() {
    let state = Arc::new(AppState::without_language_model());
    let request = Request::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .expect("builds");
    let (status, body) = call(state, request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"].as_array().map(Vec::len), Some(0), "{body}");
}
