//! The auth gate, through the router.
//!
//! The constant-time compare has its own unit tests where it is defined; what
//! is checked here is the part only the router can answer: which paths are
//! behind the gate, what a refusal looks like, and that the scrape targets stay
//! open.

mod common;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::Value;
use shiso_server::auth::AuthConfig;
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use tower::ServiceExt as _;

/// The token the gated server accepts.
const TOKEN: &str = "sk-test";

/// A server with one configured token.
fn guarded() -> Arc<AppState> {
    let (engine, _join, _teardowns) = common::fake_engine(vec![10], None);
    let auth = AuthConfig::new(vec![TOKEN.to_owned()]).expect("one token makes a gate");
    Arc::new(AppState::new(engine, "test-model".to_owned(), 4096).with_auth(auth))
}

/// Sends one request and reads the whole answer.
async fn call(state: Arc<AppState>, request: Request<Body>) -> (StatusCode, Value) {
    let response = build_router(state)
        .oneshot(request)
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, body)
}

/// A GET, optionally bearing a token.
fn get(path: &str, token: Option<&str>) -> Request<Body> {
    let mut builder = Request::builder().uri(path);
    if let Some(token) = token {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    builder.body(Body::empty()).expect("a GET request builds")
}

#[tokio::test]
async fn a_gated_route_without_a_token_is_refused_as_a_missing_key() {
    let (status, body) = call(guarded(), get("/v1/models", None)).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "missing_api_key");
    assert_eq!(body["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn a_gated_route_with_the_wrong_token_is_refused_as_an_invalid_key() {
    let (status, body) = call(guarded(), get("/v1/models", Some("sk-wrong"))).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "invalid_api_key");
}

#[tokio::test]
async fn a_gated_route_with_the_configured_token_is_served() {
    let (status, body) = call(guarded(), get("/v1/models", Some(TOKEN))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
}

/// The store directory is in the cache report, so the report is behind the
/// gate even though the path is not under `/v1/`.
#[tokio::test]
async fn the_cache_report_is_behind_the_gate() {
    let (status, _body) = call(guarded(), get("/cache", None)).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _body) = call(guarded(), get("/cache", Some(TOKEN))).await;
    assert_eq!(status, StatusCode::OK);
}

/// A monitoring system that has to be handed an inference token in order to
/// check liveness is a monitoring system that holds an inference token.
#[tokio::test]
async fn the_scrape_targets_answer_without_a_token() {
    let (status, body) = call(guarded(), get("/health", None)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ready");

    let response = build_router(guarded())
        .oneshot(get("/metrics", None))
        .await
        .expect("the router answers");
    assert_eq!(response.status(), StatusCode::OK);
}

/// A server with no tokens configured serves everything, which is the
/// documented default and not an accident of the middleware.
#[tokio::test]
async fn a_server_with_no_tokens_serves_every_route() {
    let (engine, _join, _teardowns) = common::fake_engine(vec![10], None);
    let open = Arc::new(AppState::new(engine, "test-model".to_owned(), 4096));
    let (status, _body) = call(open, get("/v1/models", None)).await;
    assert_eq!(status, StatusCode::OK);
}

/// A header that is present but malformed is a missing token, not an invalid
/// one: nothing was presented for the compare to reject.
#[tokio::test]
async fn an_authorization_header_that_is_not_a_bearer_token_is_a_missing_key() {
    let request = Request::builder()
        .uri("/v1/models")
        .header("authorization", "Basic c2VjcmV0")
        .body(Body::empty())
        .expect("a GET request builds");
    let (status, body) = call(guarded(), request).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "missing_api_key");
}

/// One family of clients sends the credential in a header of its own, and it is
/// the same credential: a correctly configured client must not be refused
/// before its request is read.
#[tokio::test]
async fn the_key_header_carries_the_same_credential_as_the_bearer_one() {
    let keyed = |value: &str| {
        Request::builder()
            .uri("/v1/models")
            .header("x-api-key", value)
            .body(Body::empty())
            .expect("a GET request builds")
    };

    let (status, body) = call(guarded(), keyed(TOKEN)).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, body) = call(guarded(), keyed("sk-wrong")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["code"], "invalid_api_key");
}

/// A malformed authorization header does not end the search: the credential may
/// be sitting in the other header, and refusing without looking is a 401 that
/// says a key is missing while it was presented.
#[tokio::test]
async fn a_malformed_bearer_header_falls_through_to_the_key_header() {
    let request = Request::builder()
        .uri("/v1/models")
        .header("authorization", "Basic c2VjcmV0")
        .header("x-api-key", TOKEN)
        .body(Body::empty())
        .expect("a GET request builds");
    let (status, body) = call(guarded(), request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}
