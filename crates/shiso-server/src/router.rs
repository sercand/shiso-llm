//! The routes, the layers around them, and the accept loop that drives them.
//!
//! # Bind first, announce second
//!
//! [`serve`] binds the listener before it logs anything about being ready. An
//! operator reading "listening on 127.0.0.1:8000" has to be able to believe it,
//! and a server that prints the line and *then* discovers the port is taken has
//! told them something false. A busy port therefore fails with nothing having
//! been announced.
//!
//! # Why the accept loop is written out rather than `axum::serve`
//!
//! Two things the convenience wrapper does not offer are wanted here: a header
//! read timeout, so a connection that opens and says nothing cannot hold a slot
//! indefinitely, and an explicit in-flight set that a drain can wait on. Both
//! are properties of the connection, so both live where connections are
//! accepted.
//!
//! Accept failures log and continue. A transient accept error - a file
//! descriptor limit, a connection reset between the SYN and the accept - is not
//! a reason to stop serving every other client.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use tower::Service as _;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::{Any, CorsLayer};

use crate::api::misc_handlers;
use crate::state::AppState;

/// The body ceiling, when nothing else is configured: 32 MiB.
///
/// Generous because an inline image is base64 and base64 is a third larger than
/// the bytes it carries, and finite because a body limit is the only thing
/// between an open port and a memory exhaustion.
pub const DEFAULT_MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

/// How long a connection has to send its request head.
const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a drain waits for in-flight connections before giving up.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(15);

/// How long the accept loop pauses after an accept error.
const ACCEPT_BACKOFF: Duration = Duration::from_millis(10);

/// Builds the router.
///
/// The layer order is the one a request travels through from the outside in:
/// the panic catcher is outermost so it also covers the layers beneath it, CORS
/// answers preflight before anything else has to, then the body limit closest to
/// the handlers.
pub fn build_router(state: Arc<AppState>) -> Router {
    // An audio upload is a file, not a JSON body, so the transcription routes
    // take their own, larger limit; the route's layer is the inner one and wins.
    let upload_limit = state.audio.as_ref().map_or(
        DEFAULT_MAX_BODY_BYTES,
        crate::audio::AudioService::max_upload_bytes,
    );
    let routes = Router::new()
        .route("/health", get(misc_handlers::health))
        .route("/metrics", get(misc_handlers::metrics))
        .route("/debug/memory", get(misc_handlers::memory_report))
        .route("/v1/models", get(misc_handlers::list_models))
        .route("/cache", get(misc_handlers::cache_report))
        .route(
            "/v1/chat/completions",
            post(crate::openai::chat::chat_completions),
        )
        .route(
            "/v1/completions",
            post(crate::openai::completions::completions),
        )
        .route(
            "/v1/responses",
            post(crate::openai::responses::responses_endpoint),
        )
        .route(
            "/v1/responses/{id}",
            get(crate::openai::responses::get_stored_response),
        )
        .route("/v1/messages", post(crate::anthropic::messages::messages))
        // Auth covers this route for free: the gate's `is_protected` takes
        // every `/v1/` path, and this is one of them.
        .route(
            "/v1/decisions",
            post(crate::decision::decisions::decisions_endpoint),
        )
        // Auth covers this route the same way, and the schema-shaped 422 the
        // error type carries has no other sender in this crate.
        .route(
            "/v1/systemone",
            post(crate::decision::systemone::systemone_endpoint),
        )
        .route(
            "/v1/audio/transcriptions",
            post(crate::openai::transcription::transcriptions)
                .layer(DefaultBodyLimit::max(upload_limit)),
        )
        .route(
            "/audio/transcriptions",
            post(crate::openai::transcription::transcriptions)
                .layer(DefaultBodyLimit::max(upload_limit)),
        )
        // The three spellings of the image route are one handler: `/images`
        // is the OpenRouter shape, `/v1/images/generations` the OpenAI one,
        // and `/v1/images` the shorthand both clients reach for.
        .route("/images", post(crate::openai::images::generate_images))
        .route("/v1/images", post(crate::openai::images::generate_images))
        .route(
            "/v1/images/generations",
            post(crate::openai::images::generate_images),
        )
        .method_not_allowed_fallback(misc_handlers::method_not_allowed)
        .fallback(misc_handlers::not_found)
        .layer(DefaultBodyLimit::max(max_body_bytes()));

    routes
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        // The default panic body is kept deliberately: overriding it is how a
        // backtrace ends up in a response.
        .layer(CatchPanicLayer::new())
        .with_state(state)
}

/// The configured body ceiling.
fn max_body_bytes() -> usize {
    std::env::var("SHISO_MAX_BODY_BYTES")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(DEFAULT_MAX_BODY_BYTES)
}

/// The address an operator should actually connect to.
///
/// A wildcard bind is not an address a client can use, so it is rendered as
/// loopback; an IPv6 address is bracketed, because `::1:8000` is not a thing
/// anyone can paste into a client.
#[must_use]
pub fn ready_line(addr: SocketAddr) -> String {
    let port = addr.port();
    match addr.ip() {
        std::net::IpAddr::V4(ip) if ip.is_unspecified() => format!("http://127.0.0.1:{port}"),
        std::net::IpAddr::V4(ip) => format!("http://{ip}:{port}"),
        std::net::IpAddr::V6(ip) if ip.is_unspecified() => format!("http://127.0.0.1:{port}"),
        std::net::IpAddr::V6(ip) => format!("http://[{ip}]:{port}"),
    }
}

/// Binds, serves, and drains on the shutdown signal.
///
/// # Errors
///
/// Whatever binding the address reports. Once bound, the loop does not fail:
/// an accept error is logged and the loop continues, because one refused
/// connection is not a reason to stop serving the rest.
pub async fn serve(state: Arc<AppState>, bind: &str, port: u16) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind((bind, port)).await?;
    let local = listener.local_addr()?;
    if is_exposed(&local) {
        tracing::warn!(
            "listening on all interfaces; anyone who can reach this host can reach the model"
        );
    }
    tracing::info!(address = %ready_line(local), "shiso is listening");

    accept_loop(listener, build_router(state)).await;
    Ok(())
}

/// Whether the bind address is reachable from off the host.
fn is_exposed(addr: &SocketAddr) -> bool {
    addr.ip().is_unspecified()
}

/// Accepts until shutdown, then drains.
///
/// The router is cloned per connection rather than wrapped in a make-service:
/// the peer address is the only thing a make-service would add, and inserting
/// it as an extension here is both shorter and the same thing axum's own
/// connect-info wrapper does.
async fn accept_loop(listener: tokio::net::TcpListener, app: Router) {
    let mut in_flight = tokio::task::JoinSet::new();
    loop {
        let accepted = tokio::select! {
            () = crate::shutdown::wait() => break,
            result = listener.accept() => result,
        };
        let (stream, peer) = match accepted {
            Ok(pair) => pair,
            Err(error) => {
                tracing::warn!(%error, "accept failed");
                tokio::time::sleep(ACCEPT_BACKOFF).await;
                continue;
            }
        };
        let app = app.clone();
        in_flight.spawn(async move {
            let io = TokioIo::new(stream);
            let service = hyper::service::service_fn(
                move |mut request: axum::http::Request<hyper::body::Incoming>| {
                    request
                        .extensions_mut()
                        .insert(axum::extract::ConnectInfo(peer));
                    app.clone().call(request)
                },
            );
            let mut builder = hyper_util::server::conn::auto::Builder::new(TokioExecutor::new());
            builder
                .http1()
                .timer(TokioTimer::new())
                .header_read_timeout(HEADER_READ_TIMEOUT);
            if let Err(error) = builder.serve_connection_with_upgrades(io, service).await {
                tracing::debug!(%peer, error = %error, "connection ended");
            }
        });
    }

    tracing::info!("no longer accepting; draining in-flight connections");
    let drained = tokio::time::timeout(DRAIN_TIMEOUT, async {
        while in_flight.join_next().await.is_some() {}
    })
    .await;
    if drained.is_err() {
        tracing::warn!(
            seconds = DRAIN_TIMEOUT.as_secs(),
            "connections were still in flight at the drain deadline"
        );
    }
    tracing::info!("every in-flight connection is finished");
}

#[cfg(test)]
mod tests {
    use super::{is_exposed, ready_line};

    /// A wildcard bind is not an address anyone can connect to, so the line an
    /// operator is shown names one that is. Printing `0.0.0.0:8000` and letting
    /// them work out that it means loopback is how a ready line becomes noise.
    #[test]
    fn a_wildcard_bind_is_announced_as_loopback() {
        assert_eq!(
            ready_line("0.0.0.0:8000".parse().expect("an address")),
            "http://127.0.0.1:8000"
        );
        assert_eq!(
            ready_line("[::]:8000".parse().expect("an address")),
            "http://127.0.0.1:8000"
        );
    }

    #[test]
    fn a_concrete_address_is_announced_as_itself() {
        assert_eq!(
            ready_line("192.0.2.7:9000".parse().expect("an address")),
            "http://192.0.2.7:9000"
        );
    }

    /// An IPv6 address without brackets is not a URL anyone can paste, because
    /// the colons of the address and the colon of the port are the same
    /// character.
    #[test]
    fn an_ipv6_address_is_announced_bracketed() {
        assert_eq!(
            ready_line("[::1]:8000".parse().expect("an address")),
            "http://[::1]:8000"
        );
    }

    /// The warning is about reachability from off the host, which is exactly
    /// the unspecified addresses and nothing else.
    #[test]
    fn only_a_wildcard_bind_counts_as_exposed() {
        assert!(is_exposed(&"0.0.0.0:1".parse().expect("an address")));
        assert!(is_exposed(&"[::]:1".parse().expect("an address")));
        assert!(!is_exposed(&"127.0.0.1:1".parse().expect("an address")));
        assert!(!is_exposed(&"192.0.2.7:1".parse().expect("an address")));
    }
}
