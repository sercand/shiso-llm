//! The HTTP rendering of [`AppError`].
//!
//! The error type and its bodies live in `shiso_runtime::error`, which has no
//! HTTP framework in it. A handler turns one into an axum response with
//! [`ErrorResponse::into_response`]: a trait rather than a `From` newtype
//! because every handler already folds its failure into a [`Response`] at the
//! edge, so nothing needs the error to be an axum `IntoResponse` itself.

pub use shiso_runtime::error::*;

use axum::Json;
use axum::response::IntoResponse as _;
use axum::response::Response;

/// A failure that can answer an HTTP request.
pub trait ErrorResponse {
    /// The response this failure renders as: its status and its JSON body in
    /// the shape of the surface it was bound to.
    fn into_response(self) -> Response;
}

impl ErrorResponse for SurfacedError {
    fn into_response(self) -> Response {
        render(self.error(), self.surface())
    }
}

/// An unbound failure renders in the broader family's shape.
impl ErrorResponse for AppError {
    fn into_response(self) -> Response {
        render(&self, ApiSurface::OpenAi)
    }
}

/// Turns one failure into the response one surface reads.
fn render(error: &AppError, surface: ApiSurface) -> Response {
    let status = error.status();
    // The cause is logged exactly once, here, rather than at every raise site:
    // a handler that logs and then returns produces two lines for one failure,
    // and the second one is the only one with the status in it.
    if let AppError::Internal { context, source } = error {
        tracing::error!(context, error = %source, "request failed");
    }
    (status, Json(body_for(error, surface))).into_response()
}
