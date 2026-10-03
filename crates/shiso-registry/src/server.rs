//! Hardened loopback HTTP server and embedded web UI (§7.2, §7.3, §7.6).

#![expect(
    clippy::print_stdout,
    reason = "startup URL with session token printed to console"
)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, Method, Request, Response, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Json};
use axum::routing::{get, post};
use rand::RngCore as _;
use serde::Deserialize;

use crate::editor::{self, BuildResult, GitStatusResult, HfInspectResult, RecipeValidationReport};

const UI_HTML: &str = include_str!("ui.html");

/// Shared state for the registry editor server.
#[derive(Debug, Clone)]
pub(crate) struct ServerState {
    /// Path to repository root containing `registry/`.
    pub(crate) repo: PathBuf,
    /// Random 256-bit session token required on all `/api` calls.
    pub(crate) session_token: String,
    /// Bound port.
    pub(crate) port: u16,
}

/// Runs the registry editor web UI and API on 127.0.0.1 (§7.1, §7.6).
///
/// # Errors
/// Returns an error if binding or serving fails.
pub(crate) async fn run_server(repo: PathBuf, port: u16, open_browser: bool) -> anyhow::Result<()> {
    // Generate 256-bit (32-byte) random session token
    let mut token_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut token_bytes);
    let session_token = hex::encode(token_bytes);

    let state = Arc::new(ServerState {
        repo,
        session_token: session_token.clone(),
        port,
    });

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound_port = listener.local_addr()?.port();

    let server_url = format!("http://127.0.0.1:{bound_port}/#token={session_token}");
    println!("\n  Shiso Registry Editor running at:\n  {server_url}\n");

    if open_browser {
        let _ = open_url_in_browser(&server_url);
    }

    let app = build_router(state);
    axum::serve(listener, app).await?;
    Ok(())
}

/// Builds the Axum router with security middleware applied (§7.6).
pub(crate) fn build_router(state: Arc<ServerState>) -> Router {
    Router::new()
        .route("/", get(serve_ui))
        .route("/api/families", get(get_families))
        .route("/api/recipes", get(list_recipes))
        .route(
            "/api/recipes/{name}",
            get(get_recipe).put(put_recipe).delete(delete_recipe),
        )
        .route("/api/hf/inspect", post(inspect_hf_handler))
        .route("/api/validate", post(validate_handler))
        .route("/api/build", post(build_handler))
        .route("/api/git/status", get(git_status_handler))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            security_middleware,
        ))
        .with_state(state)
}

/// Security middleware enforcing Host, Origin, and Token requirements (§7.6).
pub(crate) async fn security_middleware(
    State(state): State<Arc<ServerState>>,
    req: Request<Body>,
    next: Next,
) -> Result<Response<Body>, StatusCode> {
    let headers = req.headers();

    // 1. DNS Rebinding check: Host must be 127.0.0.1:<port> or localhost:<port>
    if let Some(host) = headers.get(header::HOST).and_then(|h| h.to_str().ok()) {
        let expected_ip = format!("127.0.0.1:{}", state.port);
        let expected_localhost = format!("localhost:{}", state.port);
        let host_clean = host.trim();
        if host_clean != expected_ip
            && host_clean != expected_localhost
            && host_clean != "127.0.0.1"
            && host_clean != "localhost"
        {
            return Err(StatusCode::BAD_REQUEST);
        }
    } else {
        return Err(StatusCode::BAD_REQUEST);
    }

    // 2. CSRF check: For mutating requests, Origin must match same-origin if present
    let method = req.method();
    if (method == Method::POST || method == Method::PUT || method == Method::DELETE)
        && let Some(origin) = headers.get(header::ORIGIN).and_then(|o| o.to_str().ok())
    {
        let expected_origin_ip = format!("http://127.0.0.1:{}", state.port);
        let expected_origin_localhost = format!("http://localhost:{}", state.port);
        if origin != expected_origin_ip && origin != expected_origin_localhost {
            return Err(StatusCode::FORBIDDEN);
        }
    }

    // 3. Authorization check for all /api routes
    if req.uri().path().starts_with("/api") {
        let auth_header = headers
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok());
        let expected = format!("Bearer {}", state.session_token);
        if auth_header != Some(&expected) {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }

    let response = next.run(req).await;
    Ok(response)
}

async fn serve_ui() -> impl IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        "default-src 'self' 'unsafe-inline';"
            .parse()
            .expect("valid CSP header"),
    );
    (headers, Html(UI_HTML))
}

async fn get_families() -> Json<&'static [shiso_models::FamilyFacts]> {
    Json(shiso_models::family_catalog())
}

async fn list_recipes(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<Vec<RecipeValidationReport>>, (StatusCode, String)> {
    let reports = editor::validate_all(&state.repo).map_err(internal_error)?;
    Ok(Json(reports))
}

async fn get_recipe(
    State(state): State<Arc<ServerState>>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<shiso_hub::recipe::Recipe>, (StatusCode, String)> {
    let recipe = editor::get_recipe(&state.repo, &name).map_err(bad_request_or_not_found)?;
    Ok(Json(recipe))
}

async fn put_recipe(
    State(state): State<Arc<ServerState>>,
    AxumPath(name): AxumPath<String>,
    body: String,
) -> Result<StatusCode, (StatusCode, String)> {
    editor::save_recipe(&state.repo, &name, &body).map_err(bad_request)?;
    Ok(StatusCode::OK)
}

async fn delete_recipe(
    State(state): State<Arc<ServerState>>,
    AxumPath(name): AxumPath<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    editor::delete_recipe(&state.repo, &name).map_err(bad_request)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct HfInspectReq {
    repo: String,
    #[serde(default = "default_revision")]
    revision: String,
}

fn default_revision() -> String {
    "main".to_string()
}

async fn inspect_hf_handler(
    Json(payload): Json<HfInspectReq>,
) -> Result<Json<HfInspectResult>, (StatusCode, String)> {
    let result = editor::inspect_hf(&payload.repo, &payload.revision, None)
        .await
        .map_err(internal_error)?;
    Ok(Json(result))
}

async fn validate_handler(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<Vec<RecipeValidationReport>>, (StatusCode, String)> {
    let reports = editor::validate_all(&state.repo).map_err(internal_error)?;
    Ok(Json(reports))
}

#[derive(Deserialize)]
struct BuildReq {
    out: Option<String>,
    #[serde(default)]
    allow_dirty: bool,
}

async fn build_handler(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<BuildReq>,
) -> Result<Json<BuildResult>, (StatusCode, String)> {
    let out_dir = payload
        .out
        .map_or_else(|| state.repo.join("registry/v1"), PathBuf::from);

    let res =
        editor::build_registry(&state.repo, &out_dir, payload.allow_dirty).map_err(bad_request)?;
    Ok(Json(res))
}

async fn git_status_handler(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<GitStatusResult>, (StatusCode, String)> {
    let status = editor::git_status(&state.repo);
    Ok(Json(status))
}

fn bad_request(e: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, e.to_string())
}

fn bad_request_or_not_found(e: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::NOT_FOUND, e.to_string())
}

fn internal_error(e: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

fn open_url_in_browser(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).spawn()?;
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", url])
            .spawn();
    }
    Ok(())
}
