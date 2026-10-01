//! One error type, rendered in whichever shape the surface that was asked for
//! speaks.
//!
//! # Why the surface is chosen at the edge and not at the raise site
//!
//! Two wire formats are served from one router, and they disagree about what a
//! failure looks like: one nests everything under `error` with a `type`, a
//! `param` and a `code`, the other puts a discriminant at the top level and a
//! shorter object beneath it. A handler that returned a pre-rendered body would
//! force every helper it calls to already know which surface it is serving, and
//! the fallback handlers - 404 and 405, which run before any handler at all -
//! have no surface to ask.
//!
//! So every failure is raised as an [`AppError`] that names no format, and the
//! rendering happens once. A handler on a surface that is not the `OpenAI` family
//! binds one with [`AppError::on`] as it hands the error back; everything else -
//! the fallbacks included - renders in the broader family's shape, which is the
//! safer default because a client of the other one still sees a JSON body with a
//! message in it.
//!
//! # Why [`AppError::Internal`] keeps its cause
//!
//! An `Internal(String)` throws away the engine or IO error that produced it,
//! which is precisely the thing an operator needs to diagnose a 500. The cause
//! is a real `#[source]`, so `{:#}` and every error-chain walker reach it, while
//! the body sent to the client says only that something failed on this side.

use std::error::Error;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

/// Which wire format an error is being rendered for.
///
/// Two variants because two surfaces are served. A third would be a third
/// variant and a third arm in `body_for`, which is the whole extent of what
/// adding a surface costs here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiSurface {
    /// The chat-completions and responses family.
    OpenAi,
    /// The messages family.
    Anthropic,
}

impl ApiSurface {
    /// The surface's name, for logs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }

    /// Which format a request for `path` is answered in.
    ///
    /// The path is the only thing the layers that run before a handler - the
    /// 404 and the 405 - know about which client is asking, and
    /// it is enough: a route belongs to exactly one surface.
    #[must_use]
    pub fn for_path(path: &str) -> Self {
        if path == "/v1/messages" || path.starts_with("/v1/messages/") {
            Self::Anthropic
        } else {
            Self::OpenAi
        }
    }
}

/// Everything a handler in this crate can fail with.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AppError {
    /// The server is running with no model loaded.
    #[error("no model is loaded")]
    NoModel,
    /// A request named a model this server does not serve.
    #[error("the model {model:?} is not served here")]
    UnknownModel {
        /// The name the request gave.
        model: String,
    },
    /// No route matched the path.
    #[error("unknown endpoint {path}")]
    NotFound {
        /// The path that matched nothing.
        path: String,
    },
    /// The path exists but not for this method.
    #[error("{method} is not allowed on {path}")]
    MethodNotAllowed {
        /// The method that was tried.
        method: String,
        /// The path it was tried on.
        path: String,
    },
    /// The server is up but one of the things a generation needs was never
    /// built.
    ///
    /// Distinct from [`Self::NoModel`] because the two are fixed differently: a
    /// missing model is a checkpoint that did not load, and a missing renderer
    /// or vocabulary is a file the operator did not point the server at. Both
    /// answer 503, and only one of them is about the weights.
    #[error("this server cannot generate: {what} is not loaded")]
    NotReady {
        /// What is missing, named as an operator would have to fix it.
        what: &'static str,
    },
    /// The rendered prompt does not leave room to generate anything.
    ///
    /// A variant of its own rather than a [`Self::BadRequest`] with a longer
    /// message, because the body carries two numbers a client needs in order to
    /// retry - how long the prompt was and how long it may be - and numbers
    /// formatted into prose are numbers every client has to scrape back out.
    #[error(
        "this request needs {prompt_tokens} tokens of context, and this server \
         serves {ctx}"
    )]
    ContextOverflow {
        /// The wire field that carried the prompt.
        param: &'static str,
        /// Tokens the rendered prompt occupies.
        prompt_tokens: u64,
        /// Tokens one session may hold.
        ctx: u32,
    },
    /// The request itself is wrong, and the client can fix it.
    #[error("{message}")]
    BadRequest {
        /// What is wrong, in words a caller can act on.
        message: String,
        /// The request field at fault, when one field is.
        param: Option<String>,
        /// The stable machine-readable code.
        code: &'static str,
    },
    /// The request body does not match the schema this route speaks.
    ///
    /// The systemone route's schema refusal: bytes that do not parse, a known
    /// field of the wrong type, or an unknown key inside a nested object. It
    /// answers 422 rather than 400 because its clients fix whole documents
    /// rather than one parameter, and `at` names where in the document to
    /// look for the same reason. This is the only 422 in the crate - no other
    /// route raises this variant - and a *refusal* on systemone (a bound, a
    /// blank, a duplicate) stays a [`Self::BadRequest`]: the split is between
    /// what the schema cannot express and what this server will not score.
    #[error("{message}")]
    SchemaViolation {
        /// What is wrong with the document, in words a caller can act on.
        message: String,
        /// Where it is, as `line L column C` in the request body.
        at: String,
    },
    /// The request body is larger than the route accepts.
    #[error("the request body is larger than the {limit} bytes {what} accepts")]
    TooLarge {
        /// Which route's limit was reached, named as an operator would tune it.
        what: &'static str,
        /// The limit, in bytes.
        limit: usize,
    },
    /// The server has more work in flight than it can hold.
    ///
    /// A variant of its own rather than an internal failure, because the two
    /// call for opposite responses: an internal failure will fail the same way
    /// again, and this one will not - a client that backs off and retries is
    /// served. It is also the only 503 here that is about load rather than about
    /// something the operator did not point the server at.
    #[error("this server is overloaded: {what}")]
    Overloaded {
        /// Which limit was reached, named as an operator would tune it.
        what: &'static str,
    },
    /// Something on this side failed.
    #[error("{context}")]
    Internal {
        /// What was being attempted, as a fixed string so it can be grepped.
        context: &'static str,
        /// The failure that caused it.
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
}

impl AppError {
    /// A field-level complaint the client can act on.
    ///
    /// A constructor rather than a literal at every raise site: the three fields
    /// always travel together, and a raise site that builds the struct by hand
    /// is one that can leave `param` unset on a complaint that is about exactly
    /// one field.
    #[must_use]
    pub fn bad_request(
        message: impl Into<String>,
        param: Option<&str>,
        code: &'static str,
    ) -> Self {
        Self::BadRequest {
            message: message.into(),
            param: param.map(ToOwned::to_owned),
            code,
        }
    }

    /// A complaint about the request as a whole, with no one field at fault.
    #[must_use]
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::bad_request(message, None, "invalid_request")
    }

    /// Wraps a cause as an internal failure.
    ///
    /// A constructor rather than a `From` impl: `context` is the half a `From`
    /// cannot supply, and it is the half that makes the log line useful.
    pub fn internal(
        context: &'static str,
        source: impl Into<Box<dyn Error + Send + Sync>>,
    ) -> Self {
        Self::Internal {
            context,
            source: source.into(),
        }
    }

    /// The status this error answers with.
    #[must_use]
    pub const fn status(&self) -> StatusCode {
        match self {
            Self::NoModel | Self::NotReady { .. } | Self::Overloaded { .. } => {
                StatusCode::SERVICE_UNAVAILABLE
            }

            Self::NotFound { .. } | Self::UnknownModel { .. } => StatusCode::NOT_FOUND,
            Self::MethodNotAllowed { .. } => StatusCode::METHOD_NOT_ALLOWED,
            Self::TooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            Self::SchemaViolation { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            Self::BadRequest { .. } | Self::ContextOverflow { .. } => StatusCode::BAD_REQUEST,
            Self::Internal { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// The same failure, bound to the format it must be rendered in.
    ///
    /// A wrapper rather than a field on every variant: which format a failure is
    /// read in is a fact about the route it happened on, and every raise site
    /// inside a request is several layers below the one place that knows.
    #[must_use]
    pub const fn on(self, surface: ApiSurface) -> SurfacedError {
        SurfacedError {
            surface,
            error: self,
        }
    }

    /// The stable machine-readable code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::NoModel => "model_not_loaded",
            Self::NotReady { .. } => "server_not_ready",
            Self::Overloaded { .. } => "server_overloaded",
            Self::ContextOverflow { .. } => "context_length_exceeded",

            Self::NotFound { .. } => "unknown_endpoint",
            Self::UnknownModel { .. } => "model_not_found",
            Self::MethodNotAllowed { .. } => "method_not_allowed",
            Self::TooLarge { .. } => "request_too_large",
            Self::SchemaViolation { .. } => "invalid_schema",
            Self::BadRequest { code, .. } => code,
            Self::Internal { .. } => "internal_error",
        }
    }

    /// The `param` field, which only a field-level complaint fills in.
    fn param(&self) -> Option<&str> {
        match self {
            Self::BadRequest { param, .. } => param.as_deref(),
            Self::ContextOverflow { param, .. } => Some(param),
            _ => None,
        }
    }

    /// What the client is told.
    ///
    /// An internal failure says only that it failed: the cause is a real
    /// `#[source]` and goes to the log, because it names files, devices and
    /// sometimes prompt content, none of which belongs in a 500 body.
    fn client_message(&self) -> String {
        match self {
            Self::Internal { .. } => "internal server error".to_owned(),
            other => other.to_string(),
        }
    }
}

/// The `type` an OpenAI-shaped error body carries for a status.
///
/// A function of the status alone, so a new variant of [`AppError`] cannot
/// forget to pick one.
#[must_use]
pub const fn type_for_status(status: StatusCode) -> &'static str {
    match status {
        // A schema violation answers 422 but is still the client's bad
        // request in every client's eyes; only the code and the location
        // tell it apart.
        StatusCode::BAD_REQUEST
        | StatusCode::PAYLOAD_TOO_LARGE
        | StatusCode::UNPROCESSABLE_ENTITY => "invalid_request_error",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_exceeded",
        _ => "server_error",
    }
}

/// Renders one error in one surface's shape.
fn body_for(error: &AppError, surface: ApiSurface) -> Value {
    let status = error.status();
    let message = error.client_message();
    match surface {
        ApiSurface::OpenAi => {
            let mut body = json!({
                "message": message,
                "type": type_for_status(status),
                "param": error.param(),
                "code": error.code(),
            });
            // The two context numbers ride inside the error object rather than
            // inside the message, because a client that wants to retry with a
            // shorter prompt needs them as numbers. They are added here rather
            // than in a second body shape so that every OpenAI-surface failure
            // still has exactly one renderer.
            if let AppError::ContextOverflow {
                prompt_tokens, ctx, ..
            } = error
                && let Some(fields) = body.as_object_mut()
            {
                fields.insert("n_prompt_tokens".to_owned(), json!(prompt_tokens));
                fields.insert("n_ctx".to_owned(), json!(ctx));
            }
            // A schema violation names its location for the same reason the
            // context numbers ride here rather than in the prose: a client
            // that wants to fix the document needs it as a field, not as
            // something scraped out of the message.
            if let AppError::SchemaViolation { at, .. } = error
                && let Some(fields) = body.as_object_mut()
            {
                fields.insert("at".to_owned(), json!(at));
            }
            json!({ "error": body })
        }
        ApiSurface::Anthropic => {
            let mut body = json!({
                "type": anthropic_type_for_status(status),
                "message": message,
            });
            // The two context numbers ride here for the same reason they ride
            // in the other shape: a client that wants to retry with a shorter
            // prompt needs them as numbers, and numbers formatted into prose
            // are numbers every client has to scrape back out.
            if let AppError::ContextOverflow {
                prompt_tokens, ctx, ..
            } = error
                && let Some(fields) = body.as_object_mut()
            {
                fields.insert("n_prompt_tokens".to_owned(), json!(prompt_tokens));
                fields.insert("n_ctx".to_owned(), json!(ctx));
            }
            // The location rides here too, for the same reason and with the
            // same note as above: 422 arises only on the systemone route in
            // this crate, but the renderer stays one renderer for every
            // failure that reaches it.
            if let AppError::SchemaViolation { at, .. } = error
                && let Some(fields) = body.as_object_mut()
            {
                fields.insert("at".to_owned(), json!(at));
            }
            json!({"type": "error", "error": body})
        }
    }
}

/// The `error.type` an Anthropic-shaped body carries for a status.
const fn anthropic_type_for_status(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
        _ => "api_error",
    }
}

/// One failure, bound to the wire format it will be rendered in.
///
/// Built by [`AppError::on`] and turned into a response by the same renderer
/// every other failure goes through, so a surface that needs its own shape gets
/// it without gaining its own error body builder.
#[derive(Debug)]
pub struct SurfacedError {
    surface: ApiSurface,
    error: AppError,
}

impl SurfacedError {
    /// The failure itself.
    #[must_use]
    pub const fn error(&self) -> &AppError {
        &self.error
    }

    /// The format it will be rendered in.
    #[must_use]
    pub const fn surface(&self) -> ApiSurface {
        self.surface
    }
}

impl IntoResponse for SurfacedError {
    fn into_response(self) -> Response {
        render(&self.error, self.surface)
    }
}

impl IntoResponse for AppError {
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

#[cfg(test)]
mod tests {
    use super::{ApiSurface, AppError};

    /// The layers that run before a handler have only the path to go on, and a
    /// refusal in the wrong family's shape is one a client reports as a
    /// transport failure rather than as the server's answer.
    #[test]
    fn a_path_names_the_family_that_will_read_the_answer() {
        assert_eq!(ApiSurface::for_path("/v1/messages"), ApiSurface::Anthropic);
        assert_eq!(
            ApiSurface::for_path("/v1/messages/count_tokens"),
            ApiSurface::Anthropic
        );
        assert_eq!(
            ApiSurface::for_path("/v1/chat/completions"),
            ApiSurface::OpenAi
        );
        assert_eq!(ApiSurface::for_path("/v1/models"), ApiSurface::OpenAi);
        assert_eq!(
            ApiSurface::for_path("/v1/messages-of-some-other-kind"),
            ApiSurface::OpenAi,
            "the messages route is a route, not a prefix anything may extend"
        );
    }

    /// The same failure, two shapes: one nests everything under `error` with a
    /// `param` and a `code`, the other puts a discriminant at the top level.
    #[test]
    fn one_failure_renders_in_whichever_shape_it_was_bound_to() {
        let overflow = || AppError::ContextOverflow {
            param: "messages",
            prompt_tokens: 99,
            ctx: 64,
        };
        let openai = super::body_for(&overflow(), ApiSurface::OpenAi);
        assert_eq!(openai["error"]["code"], "context_length_exceeded");
        assert_eq!(openai["error"]["param"], "messages");
        assert_eq!(openai["error"]["n_ctx"], 64);

        let anthropic = super::body_for(&overflow(), ApiSurface::Anthropic);
        assert_eq!(anthropic["type"], "error");
        assert_eq!(anthropic["error"]["type"], "invalid_request_error");
        assert_eq!(
            anthropic["error"]["n_prompt_tokens"], 99,
            "the numbers a client retries with reach both families"
        );
        assert!(
            anthropic.get("choices").is_none() && anthropic["error"].get("param").is_none(),
            "the two shapes are not the same shape: {anthropic}"
        );
    }

    /// An error that was never bound to a surface renders in the broader
    /// family's shape, which is what the fallback handlers rely on.
    #[test]
    fn an_unbound_failure_renders_in_the_broader_familys_shape() {
        let body = super::body_for(
            &AppError::NotFound {
                path: "/v1/nope".to_owned(),
            },
            ApiSurface::OpenAi,
        );
        assert_eq!(body["error"]["code"], "unknown_endpoint");
    }
}
