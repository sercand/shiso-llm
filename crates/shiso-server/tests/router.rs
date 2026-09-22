//! The routes, driven through the router rather than through a socket.
//!
//! `tower::ServiceExt::oneshot` calls the built router directly, so these tests
//! bind no port, need no runtime beyond the test's own, and cannot flake on a
//! busy address. What they assert is the thing a client actually sees: the
//! status, and the shape of the body.

mod common;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::Value;
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use tower::ServiceExt as _;

/// The context every test state is built with.
const CTX: u32 = 4096;

/// A server with a scripted model behind it.
///
/// The join handle is dropped rather than joined: these tests never end the
/// engine loop, because ending it is [`engine_thread`](engine_thread)'s subject
/// and doing it here would make every route test depend on a thread teardown.
fn state() -> Arc<AppState> {
    let (engine, _join, _teardowns) = common::fake_engine(vec![10, 11], None);
    Arc::new(AppState::new(engine, "test-model".to_owned(), CTX))
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

/// A bare GET.
fn get(path: &str) -> Request<Body> {
    Request::builder()
        .uri(path)
        .body(Body::empty())
        .expect("a GET request builds")
}

#[tokio::test]
async fn health_reports_the_model_it_is_serving() {
    let (status, body) = call(state(), get("/health")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ready");
    assert_eq!(body["model"], "test-model");
}

#[tokio::test]
async fn the_model_listing_holds_exactly_the_model_that_is_loaded() {
    let (status, body) = call(state(), get("/v1/models")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("the listing is an array");
    assert_eq!(data.len(), 1, "one process serves one checkpoint");
    assert_eq!(data[0]["id"], "test-model");
    assert_eq!(data[0]["object"], "model");
    assert_eq!(data[0]["context_length"], CTX);
    assert_eq!(data[0]["top_provider"]["context_length"], CTX);
    assert_eq!(data[0]["top_provider"]["is_moderated"], false);
    assert!(
        data[0]["supported_parameters"]
            .as_array()
            .is_some_and(|params| params.iter().any(|p| p == "tools")),
        "the listing names the parameters a client may send: {body}"
    );
}

#[tokio::test]
async fn metrics_are_scraped_as_text_with_the_counters_that_exist() {
    let response = build_router(state())
        .oneshot(get("/metrics"))
        .await
        .expect("the router answers");
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    assert!(
        content_type.starts_with("text/plain"),
        "a scrape expects text, got {content_type}"
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let text = String::from_utf8(bytes.to_vec()).expect("the metrics are UTF-8");
    for counter in [
        "shiso_up",
        "shiso_uptime_seconds",
        "shiso_model_loaded",
        "shiso_kvstore_enabled",
        "shiso_kvstore_entries",
        "shiso_kvstore_bytes",
    ] {
        assert!(text.contains(counter), "{counter} is missing from {text}");
    }
}

/// With no disk tier, the report says so rather than omitting the section: a
/// client that has to tell "disabled" from "the key is not in this version"
/// cannot.
#[tokio::test]
async fn the_cache_report_says_the_disk_tier_is_off_when_it_is() {
    let (status, body) = call(state(), get("/cache")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["disk"]["enabled"], false);
    assert_eq!(body["disk"]["dir"], Value::Null);
    assert_eq!(body["disk"]["total_bytes"], 0);
    assert_eq!(body["disk"]["entries"], 0);
}

/// A backing that moves no bytes, because this test opens a cache to read its
/// report and never stores anything in it.
#[derive(Debug)]
struct NullBacking;

impl shiso_kvstore::Backing for NullBacking {
    fn copy_page(
        &mut self,
        _dst: shiso_kvstore::PageId,
        _src: shiso_kvstore::PageId,
    ) -> Result<(), shiso_kvstore::KvError> {
        Ok(())
    }

    fn page_to_host(
        &self,
        _page: shiso_kvstore::PageId,
        _buf: &mut [u8],
    ) -> Result<(), shiso_kvstore::KvError> {
        Ok(())
    }

    fn page_from_host(
        &mut self,
        _page: shiso_kvstore::PageId,
        _buf: &[u8],
    ) -> Result<(), shiso_kvstore::KvError> {
        Ok(())
    }
}

/// With a disk tier attached, the same report carries the directory and the
/// budget it was opened with.
#[tokio::test]
async fn the_cache_report_describes_a_disk_tier_that_is_on() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let geometry = shiso_kvstore::CacheGeometry::new(16, 4096, 4096, 0.001, 600.0);
    let cache = shiso_kvstore::KvCache::open(
        &geometry,
        Box::new(NullBacking),
        4,
        2,
        Some(shiso_kvstore::SpillConfig::new(dir.path(), 8)),
    )
    .expect("the cache opens over an empty directory");

    let (engine, _join, _teardowns) = common::fake_engine(vec![10], None);
    let state =
        Arc::new(AppState::new(engine, "test-model".to_owned(), CTX).with_cache(Arc::new(cache)));

    let (status, body) = call(state, get("/cache")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["disk"]["enabled"], true);
    assert_eq!(
        body["disk"]["dir"],
        Value::String(dir.path().display().to_string())
    );
    assert_eq!(body["disk"]["total_bytes"], 0);
    assert_eq!(body["disk"]["entries"], 0);
    assert!(
        body["disk"]["budget_bytes"].as_u64().is_some_and(|b| b > 0),
        "an opened tier reports the budget it was opened with: {body}"
    );
}

/// Each generation route answers in the shape its own clients can parse. These
/// three are live handlers now, and the cheapest thing that proves the route is
/// wired to one is an empty body: a stub would have refused it with the same
/// 501 for every route, and a handler complains about the body.
#[tokio::test]
async fn the_openai_generation_routes_answer_in_an_openai_shaped_error() {
    for path in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
        let request = Request::builder()
            .method("POST")
            .uri(path)
            .body(Body::empty())
            .expect("a POST request builds");
        let (status, body) = call(state(), request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{path}");
        assert_eq!(body["error"]["code"], "invalid_request", "{path}");
        assert!(
            body["error"]["message"].as_str().is_some(),
            "the refusal says what is wrong with the body: {body}"
        );
        assert!(
            body.get("type").is_none(),
            "the two surfaces do not share a body shape: {body}"
        );
    }
}

/// A response nobody stored cannot be fetched, and the refusal says why rather
/// than leaving a client to guess that the identifier was wrong.
#[tokio::test]
async fn fetching_a_response_that_was_never_stored_is_a_json_404() {
    let (status, body) = call(state(), get("/v1/responses/resp_missing")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "not_found_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("store: true")),
        "the refusal says how to make it fetchable: {body}"
    );
}

/// The messages route is served by the Anthropic adapter, and everything it
/// refuses is refused in that family's body shape rather than the other one's.
#[tokio::test]
async fn the_messages_route_answers_in_its_own_surfaces_error_shape() {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .body(Body::empty())
        .expect("a POST request builds");
    let (status, body) = call(state(), request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"].is_string());
    assert!(
        body.get("choices").is_none(),
        "the two surfaces do not share a body shape: {body}"
    );
}

/// A count of the prompt a request would run has no route here, and the point
/// of this row is that it is genuinely absent rather than stubbed: a 501 would
/// be a promise, and a 404 is the truth about what this server serves.
#[tokio::test]
async fn the_token_counting_route_is_absent_rather_than_stubbed() {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/messages/count_tokens")
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .expect("a POST request builds");
    let (status, body) = call(state(), request).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(
        body["type"], "error",
        "a path under the messages route answers that route's client: {body}"
    );
    assert_eq!(body["error"]["type"], "not_found_error");
}

/// An unknown path answers with a JSON body, not with axum's empty one: a
/// client that gets an empty 404 reports a transport failure.
#[tokio::test]
async fn an_unknown_path_is_a_json_404() {
    let (status, body) = call(state(), get("/v1/nope")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "not_found_error");
    assert_eq!(body["error"]["code"], "unknown_endpoint");
}

#[tokio::test]
async fn a_generation_route_asked_with_the_wrong_method_is_a_json_405() {
    let (status, body) = call(state(), get("/v1/chat/completions")).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(body["error"]["code"], "method_not_allowed");
}

/// The preflight has to be answered by the CORS layer before the router's own
/// 405 handling sees it, or a browser cannot call this server at all.
#[tokio::test]
async fn a_preflight_is_answered_with_the_methods_that_are_allowed() {
    let request = Request::builder()
        .method("OPTIONS")
        .uri("/v1/models")
        .header("origin", "https://example.test")
        .header("access-control-request-method", "GET")
        .body(Body::empty())
        .expect("a preflight builds");
    let response = build_router(state())
        .oneshot(request)
        .await
        .expect("the router answers");
    assert!(
        response.status().is_success(),
        "a preflight is not a failure: {}",
        response.status()
    );
    assert!(
        response
            .headers()
            .contains_key("access-control-allow-methods"),
        "the preflight answer names the methods"
    );
}

/// A tier the engine opened but nobody here can measure reports the two facts
/// this process does know - where it is and what it was allowed to use - and
/// says plainly that the rest was not read. A body that reported zeroes without
/// saying so would make "nothing is stored" indistinguishable from "nobody here
/// can count what is stored".
#[tokio::test]
async fn a_tier_this_process_cannot_measure_says_so_rather_than_reporting_zero() {
    let (engine, _join, _teardowns) = common::fake_engine(vec![10], None);
    let state = Arc::new(
        AppState::new(engine, "test-model".to_owned(), CTX).with_checkpoint_tier(
            shiso_server::state::CheckpointTier {
                dir: std::path::PathBuf::from("/var/cache/shiso"),
                budget_bytes: 4096,
                min_tokens: 512,
            },
        ),
    );

    let (status, body) = call(state, get("/cache")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["disk"]["enabled"], true);
    assert_eq!(body["disk"]["dir"], "/var/cache/shiso");
    assert_eq!(body["disk"]["budget_bytes"], 4096);
    assert_eq!(body["disk"]["min_tokens"], 512);
    assert_eq!(body["disk"]["measured"], false, "{body}");
}

/// The scheduler's own account of the cache: what it asked for, what it got,
/// and how much prompt it had to run because it did not get it.
#[tokio::test]
async fn the_cache_report_carries_the_schedulers_own_counters() {
    let (status, body) = call(state(), get("/cache")).await;
    assert_eq!(status, StatusCode::OK);
    for key in [
        "requests",
        "hits",
        "hit_rate",
        "tokens_reused",
        "tokens_prefilled",
        "writes",
        "resident",
        "capacity",
    ] {
        assert!(
            body["engine"].get(key).is_some(),
            "{key} is missing from {body}"
        );
    }
    assert!(
        body["engine"]["capacity"].as_u64().is_some_and(|n| n > 0),
        "the resident bound is a real number: {body}"
    );
}

/// The counters `/metrics` promised once there was a scheduler to fill them in.
#[tokio::test]
async fn the_metrics_body_carries_the_scheduler_counters() {
    let response = build_router(state())
        .oneshot(get("/metrics"))
        .await
        .expect("the router answers");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let text = String::from_utf8(bytes.to_vec()).expect("the metrics are UTF-8");
    for counter in [
        "shiso_scheduler_requests_total",
        "shiso_scheduler_admitted_total",
        "shiso_scheduler_rejected_total",
        "shiso_scheduler_active",
        "shiso_scheduler_session_opens_total",
        "shiso_scheduler_session_closes_total",
        "shiso_checkpoint_hits_total",
        "shiso_checkpoint_tokens_reused_total",
    ] {
        assert!(text.contains(counter), "{counter} is missing from {text}");
    }
}
