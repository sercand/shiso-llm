//! Access log: one tracing line per inbound request, on every listener.
//!
//! The request-log table only holds rows for requests that named a model,
//! so a client hitting a path the router does not serve (a misconfigured
//! base URL, an endpoint not yet proxied) left no trace anywhere. This
//! layer sits outside routing and auth, so matched, rejected and unmatched
//! requests all produce a line.

use std::time::Instant;

use axum::extract::Request;
use axum::http::{Method, Uri};
use axum::middleware::Next;
use axum::response::Response;

use crate::error::AppError;

/// Logs method, path, status and time-to-response-headers for every
/// request. Only the path is logged, never the query string, which some
/// clients use to carry credentials. For a streamed response the elapsed
/// time ends when headers are sent; the request-log row has the full
/// duration.
pub(crate) async fn log_request(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let started = Instant::now();
    let resp = next.run(req).await;
    let status = resp.status().as_u16();
    let elapsed_ms = started.elapsed().as_millis();
    if resp.status().is_server_error() {
        tracing::warn!(%method, path, status, elapsed_ms, "request");
    } else {
        tracing::info!(%method, path, status, elapsed_ms, "request");
    }
    resp
}

/// Fallback for paths no route matches: a provider-style 404 body, so an
/// SDK pointed at the wrong path gets an error it can parse, and a warning
/// naming the path so the operator can see what clients are asking for.
pub(crate) async fn unknown_endpoint(method: Method, uri: Uri) -> AppError {
    tracing::warn!(%method, path = uri.path(), "unknown endpoint");
    AppError::NotFound
}
