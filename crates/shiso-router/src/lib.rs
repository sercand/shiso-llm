//! HTTP router that proxies LLM API requests to third-party or local backends.

#![expect(
    clippy::pedantic,
    clippy::unwrap_used,
    reason = "imported router crate uses standard idioms and internal lock unwraps"
)]

/// Active in-flight requests.
pub(crate) mod active;
/// Admin API and UI endpoints.
pub(crate) mod admin;
/// Client authentication middleware.
pub(crate) mod auth;
/// Smooth weighted round-robin load balancing.
pub(crate) mod balancer;
/// Catalog definitions and SQLite persistence.
pub(crate) mod catalog;
/// Command line interface.
pub mod cli;
/// Configuration file format and loading.
pub mod config;
/// Application error types.
pub(crate) mod error;
/// Rate and usage limits.
pub(crate) mod limits;
/// Request logging and analytics sink.
pub mod logging;
/// Model listing endpoint.
pub(crate) mod models_endpoint;
/// Multipart body parser for transcription requests.
pub(crate) mod multipart;
/// Request proxy handler and upstream dispatch.
pub(crate) mod proxy;
/// Fair queuing and gang scheduling.
pub(crate) mod sched;
/// Shared application state.
pub mod state;
/// Tee stream observing upstream response bytes.
pub(crate) mod tee;
/// API key database and token cache.
pub mod tokens;
/// Request tracing.
pub(crate) mod trace;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};

use crate::state::AppState;

/// The main listener: proxy routes, plus the admin surface when no
/// separate `admin_listen` is configured.
pub fn build_router(state: AppState) -> Router {
    // route_layer (not layer) keeps auth off the router's fallback, so
    // unmatched paths are a plain 404 rather than 401.
    // Each endpoint is also served without the `/v1` prefix, for clients
    // configured with a base URL that already ends in `/v1`; the handler
    // canonicalizes the path back to the `/v1` form.
    let proxy_routes = Router::new()
        .route("/v1/chat/completions", post(proxy::proxy_handler))
        .route("/v1/responses", post(proxy::proxy_handler))
        .route("/v1/v1/responses", post(proxy::proxy_handler))
        .route("/v1/embeddings", post(proxy::proxy_handler))
        .route("/v1/audio/transcriptions", post(proxy::proxy_handler))
        .route("/v1/messages", post(proxy::proxy_handler))
        .route("/v1/v1/messages", post(proxy::proxy_handler))
        .route("/v1/models", get(models_endpoint::list_models))
        .route("/v1/v1/models", get(models_endpoint::list_models))
        .route("/chat/completions", post(proxy::proxy_handler))
        .route("/responses", post(proxy::proxy_handler))
        .route("/embeddings", post(proxy::proxy_handler))
        .route("/audio/transcriptions", post(proxy::proxy_handler))
        .route("/messages", post(proxy::proxy_handler))
        .route("/models", get(models_endpoint::list_models))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth::require_client_token,
        ));

    let mut app = Router::new().merge(proxy_routes);
    if state.admin_on_main {
        app = app
            .merge(admin_routes(state.clone()))
            .merge(admin_ui_routes());
    }
    app.route("/healthz", get(|| async { "ok" }))
        .layer(DefaultBodyLimit::max(state.max_body_bytes))
        .with_state(state)
}

/// The dedicated admin listener used when `admin_listen` is configured;
/// serves only the admin surface (plus its own healthz). The merge into a
/// fresh Router keeps the auth layer off the fallback, so unknown paths
/// get a plain 404 instead of 401.
pub fn build_admin_router(state: AppState) -> Router {
    Router::new()
        .merge(admin_routes(state.clone()))
        .merge(admin_ui_routes())
        .route("/healthz", get(|| async { "ok" }))
        .layer(DefaultBodyLimit::max(state.max_body_bytes))
        .with_state(state)
}

/// The UI page itself is unauthenticated (it contains no data); everything
/// it renders comes from the Bearer-authenticated API.
fn admin_ui_routes() -> Router<AppState> {
    Router::new()
        .route("/admin", get(admin::ui_redirect))
        .route("/admin/ui", get(admin::ui_page))
}

/// Runs the router with the given command-line arguments.
///
/// # Errors
///
/// Fails if the configuration cannot be loaded, the database cannot be opened,
/// or a network listener cannot be bound.
pub async fn run(cli: cli::Cli) -> anyhow::Result<()> {
    let cfg = config::Config::load_with(&cli.config, &cli.overrides())
        .map_err(|e| anyhow::anyhow!("loading config from {}: {e}", cli.config.display()))?;

    let db = tokens::Db::open(&cfg.sqlite_path)?;
    let (log_sink, log_writer) = logging::start_writer(db.clone());
    let state = AppState::from_config(&cfg, db, log_sink)?;
    let app = build_router(state.clone());

    let listener = tokio::net::TcpListener::bind(cfg.listen)
        .await
        .map_err(|e| anyhow::anyhow!("failed to bind {}: {e}", cfg.listen))?;
    let routing = state.routing();
    tracing::info!(
        listen = %cfg.listen,
        models = routing.models.len(),
        backends = routing.backends.len(),
        admin = %cfg.admin_listen.map_or_else(|| "shared".to_string(), |a| a.to_string()),
        "shiso-router listening"
    );

    let main_srv = axum::serve(listener, app).with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutting down");
    });

    if let Some(admin_addr) = cfg.admin_listen {
        let admin_app = build_admin_router(state.clone());
        let admin_listener = tokio::net::TcpListener::bind(admin_addr)
            .await
            .map_err(|e| anyhow::anyhow!("failed to bind admin listener {admin_addr}: {e}"))?;
        let admin_srv = axum::serve(admin_listener, admin_app).with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        });
        tokio::try_join!(main_srv, admin_srv).map_err(|e| anyhow::anyhow!("server error: {e}"))?;
    } else {
        main_srv
            .await
            .map_err(|e| anyhow::anyhow!("server error: {e}"))?;
    }

    drop(state);
    let _ = log_writer.await;

    Ok(())
}
fn admin_routes(state: AppState) -> Router<AppState> {
    Router::new()
        .route(
            "/admin/tokens",
            post(admin::create_token).get(admin::list_tokens),
        )
        .route(
            "/admin/tokens/{id}",
            get(admin::get_token)
                .patch(admin::update_token)
                .delete(admin::delete_token),
        )
        .route(
            "/admin/models",
            get(admin::list_models).post(admin::create_model),
        )
        .route(
            "/admin/models/{name}",
            get(admin::get_model)
                .put(admin::replace_model)
                .delete(admin::delete_model),
        )
        .route("/admin/models/{name}/usage", get(admin::model_usage))
        .route(
            "/admin/backends",
            get(admin::list_backends).post(admin::create_backend),
        )
        .route(
            "/admin/backends/{name}",
            get(admin::get_backend)
                .patch(admin::update_backend)
                .delete(admin::delete_backend),
        )
        .route("/admin/backends/{name}/usage", get(admin::backend_usage))
        .route(
            "/admin/groups",
            get(admin::list_groups).post(admin::create_group),
        )
        .route(
            "/admin/groups/{id}",
            get(admin::get_group)
                .put(admin::replace_group)
                .delete(admin::delete_group),
        )
        .route("/admin/playground", post(admin::playground))
        .route("/admin/active", get(admin::list_active))
        .route("/admin/logs", get(admin::list_logs))
        .route("/admin/logs/{id}", get(admin::get_log))
        .route("/admin/usage", get(admin::usage_summary))
        .route_layer(middleware::from_fn_with_state(
            state,
            auth::require_admin_token,
        ))
}
