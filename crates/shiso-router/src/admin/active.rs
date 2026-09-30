use axum::Json;
use axum::extract::State;
use serde::Serialize;

use crate::active::ApiActiveRow;
use crate::error::AppError;
use crate::logging::now_ms;
use crate::state::AppState;

/// Requests in flight right now, longest-running first.
///
/// Deliberately unfiltered: the set is bounded by the router's own
/// concurrency limits, so it is small enough for the UI to filter
/// client-side — which keeps one copy of the filter logic (`logs.rs`'s)
/// rather than two that can disagree. Purely an in-memory read; unlike
/// every other admin endpoint it touches no SQLite at all, which is what
/// makes it safe to poll every couple of seconds.
#[derive(Serialize)]
pub(crate) struct ActivePage {
    pub items: Vec<ApiActiveRow>,
    pub count: usize,
}

pub(crate) async fn list_active(
    State(state): State<AppState>,
) -> Result<Json<ActivePage>, AppError> {
    // One `now` for the whole page, so two rows that started together
    // report the same age.
    let now = now_ms();
    let items: Vec<ApiActiveRow> = state
        .active
        .snapshot()
        .into_iter()
        .map(|e| ApiActiveRow::from_entry(e, now))
        .collect();
    Ok(Json(ActivePage {
        count: items.len(),
        items,
    }))
}
