//! The routes that answer without generating anything.
//!
//! Every one of them is a thin wrapper over a pure function that takes what it
//! needs and returns what it produces. That split is what makes them testable
//! without a router: the interesting question about `/health` is what it says
//! when the model is missing, not whether axum can call it.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::api::inference_types::{
    CacheReport, DiskReport, EngineCacheReport, ModelInfo, ModelListResponse, SUPPORTED_PARAMETERS,
    TopProvider,
};
use crate::error::{ApiSurface, AppError};
use crate::state::AppState;

/// The content type a scrape expects.
const METRICS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// What `/health` says.
///
/// A pure function of the one fact it reports. There is no "degraded" arm: this
/// server has exactly one model and either loaded it before binding the
/// listener or exited, so the loading state is reachable only by a caller that
/// built the state by hand - which is precisely the test that wants to see it.
#[must_use]
pub fn readiness(model_id: Option<&str>) -> (StatusCode, Value) {
    match model_id {
        Some(id) => (StatusCode::OK, json!({"status": "ready", "model": id})),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"status": "loading"}),
        ),
    }
}

/// `GET /health`.
pub async fn health(State(state): State<Arc<AppState>>) -> Response {
    let (status, body) = readiness(Some(state.model_id.as_str()));
    (status, Json(body)).into_response()
}

/// `GET /metrics`.
///
/// Hand-written rather than produced by a metrics library. Every counter is read
/// from state that already exists; a registry would add a dependency, a second
/// source of truth for what a counter is called, and a way for a counter to be
/// registered and never updated.
pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let body = render_metrics(&state);
    ([(header::CONTENT_TYPE, METRICS_CONTENT_TYPE)], body).into_response()
}

/// The metrics body.
///
/// Only counters that are read from something real. A counter that is always
/// zero because nothing updates it is worse than an absent one: a dashboard
/// built on it looks healthy for the wrong reason.
#[must_use]
pub fn render_metrics(state: &AppState) -> String {
    let spill = state.kv.as_ref().and_then(|kv| kv.stats().spill);
    let enabled = u8::from(spill.is_some());
    let entries = spill.as_ref().map_or(0, |stats| u64::from(stats.entries));
    let bytes = spill.as_ref().map_or(0, |stats| stats.total_bytes);
    let mut out = String::new();
    push_gauge(&mut out, "shiso_up", "1 when the process is serving", 1);
    push_gauge(
        &mut out,
        "shiso_uptime_seconds",
        "Seconds since the server started",
        state.uptime_seconds(),
    );
    push_gauge(
        &mut out,
        "shiso_model_loaded",
        "1 when a checkpoint is loaded",
        1,
    );
    push_gauge(
        &mut out,
        "shiso_kvstore_enabled",
        "1 when the cache has a disk tier",
        u64::from(enabled),
    );
    push_gauge(
        &mut out,
        "shiso_kvstore_entries",
        "Entries the disk tier holds",
        entries,
    );
    push_gauge(
        &mut out,
        "shiso_kvstore_bytes",
        "Bytes the disk tier holds",
        bytes,
    );
    // Writing into a String cannot fail; the result is discarded rather than
    // unwrapped so that a formatting change can never introduce a panic here.
    let _ = state.sched.render(&mut out);
    out
}

/// Appends one gauge with its help and type lines.
fn push_gauge(out: &mut String, name: &str, help: &str, value: u64) {
    use std::fmt::Write as _;
    // Writing into a String cannot fail; the result is discarded rather than
    // unwrapped so that a formatting change can never introduce a panic here.
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    let _ = writeln!(out, "{name} {value}");
}

/// `GET /v1/models`.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    Json(model_listing(&state))
}

/// The model listing.
///
/// Always answers, never 503. A client polls this to discover what to send, and
/// a server that refuses the question while it has an answer makes the client
/// retry for information it could have had.
#[must_use]
pub fn model_listing(state: &AppState) -> ModelListResponse {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
        .saturating_sub(state.uptime_seconds());
    ModelListResponse {
        object: "list",
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model",
            created,
            owned_by: "shiso",
            name: state.model_id.clone(),
            context_length: state.ctx,
            top_provider: TopProvider {
                context_length: state.ctx,
                max_completion_tokens: state.ctx,
                is_moderated: false,
            },
            supported_parameters: SUPPORTED_PARAMETERS.to_vec(),
        }],
    }
}

/// `GET /cache`.
pub async fn cache_report(State(state): State<Arc<AppState>>) -> Json<CacheReport> {
    Json(cache_body(&state))
}

/// The cache report.
///
/// The disk section prefers measured numbers and falls back to the tier the
/// engine was told to open. Those are two different kinds of fact - what is
/// there, and what was asked for - so the body says which one it is carrying
/// rather than letting an unmeasurable tier look like an empty one.
#[must_use]
pub fn cache_body(state: &AppState) -> CacheReport {
    let measured = state.kv.as_ref().and_then(|kv| kv.stats().spill);
    let disk = match (measured, state.ckpt.as_ref()) {
        (Some(spill), tier) => DiskReport {
            enabled: true,
            dir: Some(spill.dir.display().to_string()),
            budget_bytes: spill.budget_bytes,
            min_tokens: tier.map_or(0, |tier| tier.min_tokens),
            measured: true,
            total_bytes: spill.total_bytes,
            entries: spill.entries,
            evictions: spill.evictions,
        },
        (None, Some(tier)) => DiskReport {
            enabled: true,
            dir: Some(tier.dir.display().to_string()),
            budget_bytes: tier.budget_bytes,
            min_tokens: tier.min_tokens,
            measured: false,
            total_bytes: 0,
            entries: 0,
            evictions: 0,
        },
        (None, None) => DiskReport {
            enabled: false,
            dir: None,
            budget_bytes: 0,
            min_tokens: 0,
            measured: false,
            total_bytes: 0,
            entries: 0,
            evictions: 0,
        },
    };
    let sched = &state.sched;
    CacheReport {
        disk,
        engine: EngineCacheReport {
            requests: sched.admitted(),
            hits: sched.checkpoint_hits(),
            hit_rate: (sched.hit_rate() * 100.0).round() / 100.0,
            tokens_reused: sched.reused_tokens(),
            tokens_prefilled: sched.prefill_tokens(),
            writes: sched.checkpoint_writes(),
            resident: sched.active(),
            capacity: u64::try_from(state.engine.max_num_seqs()).unwrap_or(u64::MAX),
        },
    }
}

/// Anything that matched no route.
///
/// Rendered through [`AppError`] so that an unknown path answers with the same
/// JSON envelope as every other failure, in the shape the surface that owns
/// that path speaks. Axum's own fallback sends an empty body, which a client
/// reports as a transport error rather than as a 404.
pub async fn not_found(request: axum::extract::Request) -> Response {
    let path = request.uri().path().to_owned();
    let surface = ApiSurface::for_path(&path);
    AppError::NotFound { path }.on(surface).into_response()
}

/// A path that exists, tried with a method it does not serve.
///
/// Also rendered through [`AppError`], for the same reason: a bare 405 with no
/// body tells a client the request failed and nothing about why.
pub async fn method_not_allowed(request: axum::extract::Request) -> Response {
    let path = request.uri().path().to_owned();
    let surface = ApiSurface::for_path(&path);
    AppError::MethodNotAllowed {
        method: request.method().to_string(),
        path,
    }
    .on(surface)
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::readiness;
    use axum::http::StatusCode;

    #[test]
    fn a_server_with_a_model_reports_ready_and_names_it() {
        let (status, body) = readiness(Some("a-model"));
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ready");
        assert_eq!(body["model"], "a-model");
    }

    /// The loading arm answers 503, because a load balancer reads the status
    /// and not the body: a 200 that says "loading" in its text is a server that
    /// gets sent traffic it cannot serve.
    #[test]
    fn a_server_with_no_model_reports_loading_with_a_status_that_says_so() {
        let (status, body) = readiness(None);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "loading");
        assert!(body.get("model").is_none());
    }
}
