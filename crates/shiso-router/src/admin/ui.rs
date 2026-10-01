use axum::http::header;
use axum::response::{Html, IntoResponse, Redirect, Response};

/// The whole admin UI is one self-contained page, embedded at build time.
/// It is served without auth — every piece of data it shows comes from the
/// Bearer-authenticated JSON API.
const ADMIN_HTML: &str = include_str!("../../assets/admin.html");

/// Served with `no-store`, which matters because the page is embedded at
/// build time: the response otherwise carries no `Cache-Control`, `ETag`
/// or `Last-Modified`, so a browser applies its heuristic cache and can
/// keep serving the previous deploy's page against the new binary's API.
/// That failure mode reads as a missing feature rather than a stale page —
/// the JSON carries a field the cached HTML has no code to render — so it
/// is worth spending a re-fetch to make impossible. The page is ~90 KB
/// once per load, against a router whose day job is proxying LLM streams.
pub(crate) async fn ui_page() -> Response {
    ([(header::CACHE_CONTROL, "no-store")], Html(ADMIN_HTML)).into_response()
}

pub(crate) async fn ui_redirect() -> Redirect {
    Redirect::temporary("/admin/ui")
}
