//! Bearer-token authentication, and the one comparison that has to be careful.
//!
//! # Why the compare is constant time
//!
//! A byte-by-byte equality check that returns on the first mismatch leaks how
//! many leading bytes of a guess were right, and a caller who can time a few
//! thousand requests can recover a token one byte at a time. [`ct_eq`] always
//! looks at every byte and folds the differences together, so the time it takes
//! depends on the length of the input and on nothing else.
//!
//! The scan over the configured tokens is linear and deliberately does not stop
//! at the first match: `|=` over every candidate keeps the loop's shape
//! independent of which one matched, or whether any did. The *number* of
//! candidates is controlled by the operator and is small, and an attacker
//! cannot add to it, so its being linear leaks nothing usable.
//!
//! Length is still observable, and that is accepted: token generators emit
//! fixed-length tokens, so a token's length is not the secret.
//!
//! # Why two headers carry the same credential
//!
//! One family of clients sends `Authorization: Bearer`, the other sends
//! `x-api-key`, and both are the same secret being presented to the same gate.
//! Reading only the first would refuse a correctly configured client of the
//! second family before its request ever reached a handler, with a 401 that
//! says its key is missing while it is sitting in a header nobody looked at.
//! The bearer header wins when both are present: a proxy that adds one must not
//! be able to override a credential the caller chose.
//!
//! # What is gated
//!
//! Everything under `/v1/` plus `/cache`, which names the directory the disk
//! tier writes to. `/health` and `/metrics` stay open because they are scrape
//! targets, and a monitoring system that has to be given an inference token in
//! order to check liveness is a monitoring system that holds an inference
//! token.

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::error::{ApiSurface, AppError, AuthReject};
use crate::state::AppState;

/// The tokens this server accepts.
///
/// Holding an empty set is not representable through [`AuthConfig::new`]: an
/// empty set would accept nothing at all while looking like a configured gate,
/// and "no gate" is spelled by not having one of these at all - it is
/// `AppState::auth == None`.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    tokens: Vec<String>,
}

impl AuthConfig {
    /// A gate over `tokens`, or `None` when there are none.
    #[must_use]
    pub fn new(tokens: Vec<String>) -> Option<Self> {
        let tokens: Vec<String> = tokens.into_iter().filter(|t| !t.is_empty()).collect();
        (!tokens.is_empty()).then_some(Self { tokens })
    }

    /// How many tokens are configured, for a startup log that must not print
    /// them.
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Whether `presented` is one of the configured tokens.
    ///
    /// Constant time per candidate; see the module docs for what that does and
    /// does not protect.
    #[must_use]
    pub fn validate(&self, presented: &[u8]) -> bool {
        let mut any_match = 0u8;
        for valid in &self.tokens {
            any_match |= ct_eq(presented, valid.as_bytes());
        }
        any_match == 1
    }
}

/// Constant-time byte-slice equality, as `1` or `0`.
///
/// Always processes every byte of an equal-length pair: no early exit, and the
/// final conversion to `{0, 1}` is arithmetic rather than a branch.
#[must_use]
pub fn ct_eq(a: &[u8], b: &[u8]) -> u8 {
    if a.len() != b.len() {
        return 0;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    // diff == 0 means equal. Subtracting one from a zero wraps to all ones, so
    // the sign bit of the widened difference is the answer.
    1u8 & u8::try_from((u32::from(diff).wrapping_sub(1) >> 31) & 1).unwrap_or(0)
}

/// Whether a path is behind the gate.
///
/// A free function so the rule can be tested without building a request: the
/// interesting failure here is a route that quietly falls outside the gate, and
/// that is a property of this predicate alone.
#[must_use]
pub fn is_protected(path: &str) -> bool {
    path.starts_with("/v1/") || path == "/cache"
}

/// The header the Messages family sends its credential in.
const API_KEY_HEADER: &str = "x-api-key";

/// The credential a request presents, if it presents one.
///
/// A malformed `Authorization` header falls through to the other spelling
/// rather than ending the search: `Authorization: Basic ...` is not a bearer
/// token, and a client that sent one alongside a key header presented a
/// credential either way.
fn presented_token(request: &Request) -> Option<&str> {
    bearer_token(request).or_else(|| api_key(request))
}

/// The bearer token, when the authorization header carries one.
fn bearer_token(request: &Request) -> Option<&str> {
    request
        .headers()
        .get(axum::http::header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(str::trim)
        .filter(|token| !token.is_empty())
}

/// The key the Messages family's own header carries.
fn api_key(request: &Request) -> Option<&str> {
    request
        .headers()
        .get(API_KEY_HEADER)?
        .to_str()
        .ok()
        .map(str::trim)
        .filter(|token| !token.is_empty())
}

/// Refuses a request that does not present a configured token.
pub async fn require_auth_middleware(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let Some(auth) = state.auth.as_ref() else {
        return next.run(request).await;
    };
    if !is_protected(request.uri().path()) {
        return next.run(request).await;
    }
    // The gate runs before any handler, so the path is all there is to say
    // which format the caller reads. A refusal in the other family's shape is
    // one a client reports as a transport failure.
    let surface = ApiSurface::for_path(request.uri().path());
    let Some(token) = presented_token(&request) else {
        return AppError::Unauthorized(AuthReject::Missing)
            .on(surface)
            .into_response();
    };
    if !auth.validate(token.as_bytes()) {
        return AppError::Unauthorized(AuthReject::Invalid)
            .on(surface)
            .into_response();
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::{AuthConfig, ct_eq, is_protected};

    #[test]
    fn equal_slices_compare_equal_and_unequal_ones_do_not() {
        assert_eq!(ct_eq(b"sk-test", b"sk-test"), 1);
        assert_eq!(ct_eq(b"sk-test", b"sk-tesX"), 0);
        assert_eq!(ct_eq(b"sk-test", b"sk-tes"), 0);
        assert_eq!(ct_eq(b"", b""), 1);
        assert_eq!(ct_eq(b"", b"x"), 0);
    }

    /// Only the first byte differs, and only the last byte differs: both are
    /// mismatches, and the point of the compare is that neither is cheaper to
    /// discover than the other.
    #[test]
    fn a_mismatch_in_the_first_byte_and_in_the_last_are_both_refused() {
        assert_eq!(ct_eq(b"abcdef", b"Xbcdef"), 0);
        assert_eq!(ct_eq(b"abcdef", b"abcdeX"), 0);
    }

    #[test]
    fn a_gate_with_no_tokens_is_no_gate_at_all() {
        assert!(AuthConfig::new(Vec::new()).is_none());
        assert!(AuthConfig::new(vec![String::new()]).is_none());
    }

    #[test]
    fn any_configured_token_is_accepted_and_nothing_else_is() {
        let auth = AuthConfig::new(vec!["one".to_owned(), "two".to_owned()])
            .expect("two tokens make a gate");
        assert_eq!(auth.token_count(), 2);
        assert!(auth.validate(b"one"));
        assert!(auth.validate(b"two"));
        assert!(!auth.validate(b"three"));
        assert!(!auth.validate(b""));
    }

    #[test]
    fn the_scrape_targets_are_open_and_everything_else_under_v1_is_not() {
        assert!(is_protected("/v1/models"));
        assert!(is_protected("/v1/chat/completions"));
        assert!(is_protected("/cache"));
        assert!(!is_protected("/health"));
        assert!(!is_protected("/metrics"));
        assert!(!is_protected("/"));
    }
}
