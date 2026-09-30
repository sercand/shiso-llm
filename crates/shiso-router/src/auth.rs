use axum::extract::{Request, State};
use axum::http::HeaderMap;
use axum::middleware::Next;
use axum::response::Response;
use subtle::ConstantTimeEq;

use crate::error::AppError;
use crate::state::AppState;
use crate::tokens::hash_token;

/// Pulls the client credential from `Authorization: Bearer <tok>` or
/// `x-api-key: <tok>`. OpenAI SDKs send the former, the Anthropic SDK the
/// latter; we accept both on every route.
fn extract_token(headers: &HeaderMap) -> Option<&str> {
    if let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok())
        && let Some(tok) = auth
            .strip_prefix("Bearer ")
            .or_else(|| auth.strip_prefix("bearer "))
    {
        let tok = tok.trim();
        if !tok.is_empty() {
            return Some(tok);
        }
    }
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|t| !t.is_empty())
}

/// Resolves the client token to its key record and stashes it in request
/// extensions so downstream handlers can attribute the request. Disabled
/// keys are rejected with 403 (distinguishable from an unknown token's 401).
pub(crate) async fn require_client_token(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Result<Response, AppError> {
    let token = extract_token(request.headers()).ok_or(AppError::Unauthorized)?;
    let record = state
        .token_cache
        .get(&hash_token(token))
        .ok_or(AppError::Unauthorized)?;
    if record.disabled {
        return Err(AppError::Forbidden("API key is disabled".into()));
    }
    request.extensions_mut().insert(record);
    Ok(next.run(request).await)
}

pub(crate) async fn require_admin_token(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, AppError> {
    let token = extract_token(request.headers()).ok_or(AppError::Unauthorized)?;
    if token
        .as_bytes()
        .ct_eq(state.admin_token.as_bytes())
        .unwrap_u8()
        != 1
    {
        return Err(AppError::Unauthorized);
    }
    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn extracts_bearer() {
        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer sk-abc"));
        assert_eq!(extract_token(&h), Some("sk-abc"));
    }

    #[test]
    fn extracts_x_api_key() {
        let mut h = HeaderMap::new();
        h.insert("x-api-key", HeaderValue::from_static("sk-xyz"));
        assert_eq!(extract_token(&h), Some("sk-xyz"));
    }

    #[test]
    fn bearer_wins_over_x_api_key() {
        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer a"));
        h.insert("x-api-key", HeaderValue::from_static("b"));
        assert_eq!(extract_token(&h), Some("a"));
    }

    #[test]
    fn missing_or_malformed() {
        assert_eq!(extract_token(&HeaderMap::new()), None);
        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Basic dXNlcg=="));
        assert_eq!(extract_token(&h), None);
        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer "));
        assert_eq!(extract_token(&h), None);
    }
}
