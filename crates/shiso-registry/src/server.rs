//! Hardened loopback HTTP server and embedded web UI (§7.2, §7.3, §7.6).

#![expect(
    clippy::print_stdout,
    reason = "startup URL with session token printed to console"
)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, Method, Request, Response, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Json, Redirect};
use axum::routing::{get, post};
use rand::RngCore as _;
use serde::{Deserialize, Serialize};
use tower_http::services::ServeDir;

use crate::bench::{self, BenchLog};

use crate::editor::{self, BuildResult, GitStatusResult, HfInspectResult, RecipeValidationReport};

const UI_HTML: &str = include_str!("ui.html");

/// Shared state for the registry editor server.
#[derive(Debug, Clone)]
pub(crate) struct ServerState {
    /// Path to repository root containing `registry/`.
    pub(crate) repo: PathBuf,
    /// Random 256-bit session token required on all `/api` calls, when the
    /// editor was started with `--require-token`; `None` requires none.
    pub(crate) session_token: Option<String>,
    /// Bound port.
    pub(crate) port: u16,
    /// `shiso` home whose installed variants the benchmark runner measures.
    pub(crate) home: PathBuf,
    /// The benchmark started from the editor, if any. One at a time: two
    /// measurements against the same server would each slow the other down.
    pub(crate) bench: Arc<Mutex<Option<BenchJob>>>,
}

impl ServerState {
    /// State for a repository with no benchmark running.
    pub(crate) fn new(
        repo: PathBuf,
        session_token: Option<String>,
        port: u16,
        home: PathBuf,
    ) -> Self {
        Self {
            repo,
            session_token,
            port,
            home,
            bench: Arc::default(),
        }
    }
}

/// A benchmark started from the editor.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct BenchJob {
    /// What is being measured (`name[:variant]`).
    pub(crate) model: String,
    /// Still measuring.
    pub(crate) running: bool,
    /// Progress so far.
    #[serde(skip)]
    pub(crate) log: BenchLog,
    /// Path of the written record once it succeeded.
    pub(crate) record: Option<String>,
    /// Why it failed, once it did.
    pub(crate) error: Option<String>,
}

/// How to run the editor.
#[derive(Debug, Clone)]
pub(crate) struct ServeDesc {
    /// Registry repository.
    pub(crate) repo: PathBuf,
    /// Loopback port.
    pub(crate) port: u16,
    /// Open a browser at startup.
    pub(crate) open: bool,
    /// Require a session token on `/api`.
    pub(crate) require_token: bool,
    /// `shiso` home for the benchmark runner.
    pub(crate) home: PathBuf,
}

/// Runs the registry editor web UI and API on 127.0.0.1 (§7.1, §7.6).
///
/// # Errors
/// Returns an error if binding or serving fails.
pub(crate) async fn run_server(desc: ServeDesc) -> anyhow::Result<()> {
    // A 256-bit (32-byte) random session token, only when asked for.
    let session_token = desc.require_token.then(|| {
        let mut token_bytes = [0u8; 32];
        rand::rng().fill_bytes(&mut token_bytes);
        hex::encode(token_bytes)
    });

    let state = Arc::new(ServerState::new(
        desc.repo,
        session_token.clone(),
        desc.port,
        desc.home,
    ));

    let addr = SocketAddr::from(([127, 0, 0, 1], desc.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound_port = listener.local_addr()?.port();

    let server_url = match &session_token {
        Some(token) => format!("http://127.0.0.1:{bound_port}/#token={token}"),
        None => format!("http://127.0.0.1:{bound_port}/"),
    };
    println!("\n  Shiso Registry Editor running at:\n  {server_url}");
    println!("  Hub preview (after a build): http://127.0.0.1:{bound_port}/hub/\n");

    if desc.open {
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
        .route(
            "/api/models/{model}/files/{file}",
            axum::routing::put(put_hosted_file_handler)
                .layer(axum::extract::DefaultBodyLimit::max(HOSTED_FILE_BODY_LIMIT)),
        )
        .route("/api/validate", post(validate_handler))
        .route("/api/build", post(build_handler))
        .route("/api/git/status", get(git_status_handler))
        .route("/api/installed", get(installed_handler))
        .route("/api/bench", post(start_bench_handler))
        .route("/api/bench/status", get(bench_status_handler))
        .route("/api/bench/records", get(bench_records_handler))
        // The built site, as the host would serve it, for review before a
        // push. It serves the default build output, the repository itself.
        .nest_service(
            "/hub",
            ServeDir::new(editor::resolve_out(&state.repo, None))
                .append_index_html_on_directories(true),
        )
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

    // 3. The preview serves the repository directory, which also holds
    // `.git` and editor state; nothing whose path has a dot segment is part
    // of the site.
    let path = req.uri().path();
    if path.starts_with("/hub/") && path.split('/').any(|seg| seg.starts_with('.')) {
        return Err(StatusCode::NOT_FOUND);
    }
    // The site links relative to each page, so its root needs the slash.
    if path == "/hub" {
        return Ok(Redirect::permanent("/hub/").into_response());
    }

    // 4. Authorization check for all /api routes, when a token is required
    if let Some(token) = &state.session_token
        && req.uri().path().starts_with("/api")
    {
        let auth_header = headers
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok());
        let expected = format!("Bearer {token}");
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

/// The upload route's body limit: the hub's per-file limit, so an oversized
/// file is refused before it is buffered.
#[expect(
    clippy::cast_possible_truncation,
    reason = "one mebibyte fits every usize this builds for"
)]
const HOSTED_FILE_BODY_LIMIT: usize = shiso_hub::recipe::HUB_FILE_MAX_BYTES as usize;

async fn put_hosted_file_handler(
    State(state): State<Arc<ServerState>>,
    AxumPath((model, file)): AxumPath<(String, String)>,
    body: axum::body::Bytes,
) -> Result<Json<editor::HostedFile>, (StatusCode, String)> {
    editor::write_hosted_file(&state.repo, &model, &file, &body)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))
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
    #[serde(default)]
    offline: bool,
}

async fn build_handler(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<BuildReq>,
) -> Result<Json<BuildResult>, (StatusCode, String)> {
    let out_dir = editor::resolve_out(&state.repo, payload.out.as_deref().map(Path::new));
    if !payload.offline {
        let recipes: Vec<_> = editor::load_recipes(&state.repo)
            .map_err(bad_request)?
            .into_iter()
            .map(|(_, r)| r)
            .collect();
        crate::card::refresh_cache(&state.repo, &recipes)
            .await
            .map_err(bad_request)?;
    }

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

#[derive(Serialize)]
struct InstalledRow {
    model: String,
    variant: String,
    quant: String,
    backends: Vec<&'static str>,
}

async fn installed_handler(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<Vec<InstalledRow>>, (StatusCode, String)> {
    let store = shiso_hub::BlobStore::new(shiso_hub::HubConfig::with_defaults(state.home.clone()));
    let mut rows: Vec<InstalledRow> = store
        .list_installed()
        .map_err(internal_error)?
        .into_iter()
        .map(|v| InstalledRow {
            model: v.model.to_string(),
            variant: v.variant.to_string(),
            quant: v.quant,
            backends: v.backends.iter().map(|b| b.as_str()).collect(),
        })
        .collect();
    rows.sort_by(|a, b| (&a.model, &a.variant).cmp(&(&b.model, &b.variant)));
    Ok(Json(rows))
}

#[derive(Deserialize)]
struct BenchReq {
    model: String,
    server_url: String,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    prompt_tokens: Vec<u64>,
    #[serde(default = "default_decode_tokens")]
    decode_tokens: u64,
    #[serde(default)]
    image_sizes: Vec<String>,
    #[serde(default)]
    audio: Option<String>,
    runs: u32,
    #[serde(default)]
    shiso_version: Option<String>,
    #[serde(default)]
    notes: Option<String>,
}

const fn default_decode_tokens() -> u64 {
    128
}

fn non_empty(s: Option<String>) -> Option<String> {
    s.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

async fn start_bench_handler(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<BenchReq>,
) -> Result<Json<BenchJob>, (StatusCode, String)> {
    let model_ref = shiso_hub::ModelRef::parse(req.model.trim()).map_err(bad_request)?;
    let device = non_empty(req.device)
        .as_deref()
        .map(bench::DeviceSlug::parse)
        .transpose()
        .map_err(bad_request)?;
    let log = BenchLog::default();
    let job = BenchJob {
        model: model_ref.to_string(),
        running: true,
        log: log.clone(),
        record: None,
        error: None,
    };
    {
        let mut slot = state.bench.lock().map_err(internal_error)?;
        if slot.as_ref().is_some_and(|j| j.running) {
            return Err((
                StatusCode::CONFLICT,
                "a benchmark is already running".to_string(),
            ));
        }
        *slot = Some(job.clone());
    }
    let desc = bench::BenchDesc {
        repo: state.repo.clone(),
        model: model_ref.model,
        variant: model_ref.variant,
        server_url: req.server_url,
        device,
        task: non_empty(req.task)
            .map(|t| t.parse::<shiso_hub::Task>())
            .transpose()
            .map_err(bad_request)?,
        prompt_tokens: req.prompt_tokens,
        decode_tokens: req.decode_tokens,
        image_sizes: req.image_sizes,
        audio: non_empty(req.audio).map(PathBuf::from),
        runs: req.runs,
        home: state.home.clone(),
        shiso_version: non_empty(req.shiso_version),
        notes: non_empty(req.notes),
        log: Some(log),
    };
    let slot = state.bench.clone();
    tokio::spawn(async move {
        let outcome = bench::run(desc).await;
        if let Ok(mut slot) = slot.lock()
            && let Some(job) = slot.as_mut()
        {
            job.running = false;
            match outcome {
                Ok(path) => job.record = Some(path.display().to_string()),
                Err(e) => job.error = Some(format!("{e:#}")),
            }
        }
    });
    Ok(Json(job))
}

#[derive(Serialize)]
struct BenchStatus {
    #[serde(flatten)]
    job: Option<BenchJob>,
    log: Vec<String>,
}

async fn bench_status_handler(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<BenchStatus>, (StatusCode, String)> {
    let job = state.bench.lock().map_err(internal_error)?.clone();
    let log = job.as_ref().map(|j| j.log.lines()).unwrap_or_default();
    Ok(Json(BenchStatus { job, log }))
}

async fn bench_records_handler(
    State(state): State<Arc<ServerState>>,
) -> Result<Json<Vec<bench::PublishedRecord>>, (StatusCode, String)> {
    let recipes: Vec<_> = editor::load_recipes(&state.repo)
        .map_err(internal_error)?
        .into_iter()
        .map(|(_, r)| r)
        .collect();
    let records = bench::load_all(&state.repo, &recipes).map_err(bad_request)?;
    Ok(Json(records.into_values().flatten().collect()))
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
        std::process::Command::new("xdg-open").arg(url).spawn()?;
    }
    #[cfg(windows)]
    {
        std::process::Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", url])
            .spawn()?;
    }
    Ok(())
}
