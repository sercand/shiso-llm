use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Errors surfaced to API clients. Rendered as a provider-style JSON body
/// (`{"error": {"message", "type"}}`) that both OpenAI and Anthropic SDKs
/// know how to parse.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    BadRequest(String),
    #[error("invalid or missing API token")]
    Unauthorized,
    #[error("{0}")]
    Forbidden(String),
    #[error("{0}")]
    QuotaExceeded(String),
    #[error("model {0:?} is not configured")]
    UnknownModel(String),
    #[error("{0}")]
    NoBackendForEndpoint(String),
    /// The request would have queued past its QoS deadline and the route
    /// has nowhere to shed it to. 503 (rather than 429) because the
    /// capacity is busy, not the caller over quota, and it is retryable.
    #[error("{0}")]
    DeadlineExceeded(String),
    #[error("not found")]
    NotFound,
    #[error("{0}")]
    Conflict(String),
    #[error("all backends failed: {0}")]
    BadGateway(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl AppError {
    pub fn status(&self) -> StatusCode {
        match self {
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::Unauthorized => StatusCode::UNAUTHORIZED,
            AppError::Forbidden(_) => StatusCode::FORBIDDEN,
            AppError::QuotaExceeded(_) => StatusCode::TOO_MANY_REQUESTS,
            AppError::UnknownModel(_) | AppError::NoBackendForEndpoint(_) | AppError::NotFound => {
                StatusCode::NOT_FOUND
            }
            AppError::Conflict(_) => StatusCode::CONFLICT,
            AppError::DeadlineExceeded(_) => StatusCode::SERVICE_UNAVAILABLE,
            AppError::BadGateway(_) => StatusCode::BAD_GATEWAY,
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            AppError::BadRequest(_) => "invalid_request_error",
            AppError::Unauthorized => "authentication_error",
            AppError::Forbidden(_) => "permission_error",
            AppError::QuotaExceeded(_) => "insufficient_quota",
            AppError::UnknownModel(_) | AppError::NoBackendForEndpoint(_) | AppError::NotFound => {
                "not_found_error"
            }
            AppError::Conflict(_) => "invalid_request_error",
            AppError::DeadlineExceeded(_) => "overloaded_error",
            AppError::BadGateway(_) => "api_error",
            AppError::Internal(_) => "api_error",
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "message": self.to_string(),
                "type": self.error_type(),
            }
        });
        (self.status(), Json(body)).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError::Internal(e.to_string())
    }
}
