use axum::Json;
use axum::extract::{Path, Query, State};
use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::state::AppState;
use crate::tokens::Db;

/// A request-log row as served by the API: `created_at` is RFC3339 (the
/// table stores epoch milliseconds). `token_name` is joined from `tokens`
/// and is null once the key has been deleted (logs outlive keys).
#[derive(Debug, Serialize)]
pub(crate) struct ApiLogRow {
    pub id: String,
    pub created_at: String,
    pub token_id: String,
    pub token_name: Option<String>,
    pub model: String,
    pub backend: Option<String>,
    pub upstream_model: Option<String>,
    pub path: String,
    pub status: Option<u16>,
    pub streamed: bool,
    pub ttfb_ms: Option<i64>,
    pub duration_ms: Option<i64>,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
    pub cache_write_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub cost: Option<f64>,
    pub finish_reason: Option<String>,
    pub error: Option<String>,
    /// The client's `x-service-tier` hint; null for ordinary traffic.
    pub service_tier: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct LogsQuery {
    pub token_id: Option<String>,
    pub model: Option<String>,
    pub backend: Option<String>,
    pub status: Option<u16>,
    /// RFC3339 inclusive lower bound on created_at.
    pub since: Option<String>,
    /// RFC3339 exclusive upper bound on created_at.
    pub until: Option<String>,
    pub limit: Option<usize>,
    /// Opaque keyset cursor from the previous page's `next_cursor`.
    pub cursor: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct LogsPage {
    pub items: Vec<ApiLogRow>,
    pub next_cursor: Option<String>,
}

const DEFAULT_LIMIT: usize = 50;
const MAX_LIMIT: usize = 500;

pub(crate) async fn list_logs(
    State(state): State<AppState>,
    Query(q): Query<LogsQuery>,
) -> Result<Json<LogsPage>, AppError> {
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let since = q.since.as_deref().map(parse_rfc3339_ms).transpose()?;
    let until = q.until.as_deref().map(parse_rfc3339_ms).transpose()?;
    let cursor = q.cursor.as_deref().map(parse_cursor).transpose()?;

    let db = state.db.clone();
    let page =
        super::keys::blocking(move || query_logs(&db, &q, since, until, cursor, limit)).await?;
    Ok(Json(page))
}

pub(crate) async fn get_log(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ApiLogRow>, AppError> {
    let db = state.db.clone();
    let row = super::keys::blocking(move || {
        db.with_read(|conn| {
            let row = conn
                .query_row(
                    &format!("{LOG_SELECT} WHERE l.id = ?1"),
                    [&id],
                    api_row_from_row,
                )
                .map(Some)
                .or_else(|e| match e {
                    rusqlite::Error::QueryReturnedNoRows => Ok(None),
                    e => Err(e),
                })?;
            Ok(row)
        })
    })
    .await?
    .ok_or(AppError::NotFound)?;
    Ok(Json(row))
}

/// Every filter/cursor clause qualifies its columns with `l.` — `tokens`
/// carries `id`, `name` and `created_at` too, so bare names are ambiguous.
const LOG_SELECT: &str = "SELECT l.id, l.created_at, l.token_id, t.name, l.model, l.backend, \
     l.upstream_model, l.path, l.status, l.streamed, l.ttfb_ms, l.duration_ms, \
     l.prompt_tokens, l.completion_tokens, l.cached_tokens, l.cache_write_tokens, \
     l.total_tokens, l.cost, l.finish_reason, l.error, l.service_tier \
     FROM request_logs l LEFT JOIN tokens t ON t.id = l.token_id";

fn query_logs(
    db: &Db,
    q: &LogsQuery,
    since: Option<i64>,
    until: Option<i64>,
    cursor: Option<(i64, String)>,
    limit: usize,
) -> anyhow::Result<LogsPage> {
    use rusqlite::types::Value;

    let mut sql = format!("{LOG_SELECT} WHERE 1=1");
    let mut params: Vec<Value> = Vec::new();
    if let Some(t) = &q.token_id {
        sql.push_str(" AND l.token_id = ?");
        params.push(t.clone().into());
    }
    if let Some(m) = &q.model {
        sql.push_str(" AND l.model = ?");
        params.push(m.clone().into());
    }
    if let Some(b) = &q.backend {
        sql.push_str(" AND l.backend = ?");
        params.push(b.clone().into());
    }
    if let Some(s) = q.status {
        sql.push_str(" AND l.status = ?");
        params.push(Value::Integer(s as i64));
    }
    if let Some(s) = since {
        sql.push_str(" AND l.created_at >= ?");
        params.push(Value::Integer(s));
    }
    if let Some(u) = until {
        sql.push_str(" AND l.created_at < ?");
        params.push(Value::Integer(u));
    }
    if let Some((ms, id)) = cursor {
        sql.push_str(" AND (l.created_at < ? OR (l.created_at = ? AND l.id < ?))");
        params.push(Value::Integer(ms));
        params.push(Value::Integer(ms));
        params.push(id.into());
    }
    sql.push_str(" ORDER BY l.created_at DESC, l.id DESC LIMIT ?");
    params.push(Value::Integer(limit as i64));

    db.with_read(|conn| {
        let mut stmt = conn.prepare(&sql)?;
        let mut raw_rows: Vec<(i64, ApiLogRow)> = stmt
            .query_map(rusqlite::params_from_iter(params), |row| {
                Ok((row.get::<_, i64>(1)?, api_row_from_row(row)?))
            })?
            .collect::<Result<_, _>>()?;
        let next_cursor = if raw_rows.len() == limit {
            raw_rows.last().map(|(ms, row)| format!("{ms}:{}", row.id))
        } else {
            None
        };
        let items = raw_rows.drain(..).map(|(_, row)| row).collect();
        Ok(LogsPage { items, next_cursor })
    })
}

fn api_row_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ApiLogRow> {
    let token_id: String = row.get(2)?;
    // The admin console has no `tokens` row to join, so its name is filled
    // in here; a genuinely deleted key stays null.
    let token_name: Option<String> = row
        .get::<_, Option<String>>(3)?
        .or_else(|| crate::tokens::synthetic_key_name(&token_id).map(str::to_string));
    Ok(ApiLogRow {
        id: row.get(0)?,
        created_at: ms_to_rfc3339(row.get(1)?),
        token_id,
        token_name,
        model: row.get(4)?,
        backend: row.get(5)?,
        upstream_model: row.get(6)?,
        path: row.get(7)?,
        status: row.get(8)?,
        streamed: row.get(9)?,
        ttfb_ms: row.get(10)?,
        duration_ms: row.get(11)?,
        prompt_tokens: row.get(12)?,
        completion_tokens: row.get(13)?,
        cached_tokens: row.get(14)?,
        cache_write_tokens: row.get(15)?,
        total_tokens: row.get(16)?,
        cost: row.get(17)?,
        finish_reason: row.get(18)?,
        error: row.get(19)?,
        service_tier: row.get(20)?,
    })
}

pub(crate) fn ms_to_rfc3339(ms: i64) -> String {
    time::OffsetDateTime::from_unix_timestamp_nanos(ms as i128 * 1_000_000)
        .expect("valid stored timestamp")
        .format(&time::format_description::well_known::Rfc3339)
        .expect("RFC3339 formatting cannot fail for UTC")
}

pub(crate) fn parse_rfc3339_ms(s: &str) -> Result<i64, AppError> {
    time::OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339)
        .map(|t| (t.unix_timestamp_nanos() / 1_000_000) as i64)
        .map_err(|_| AppError::BadRequest(format!("invalid RFC3339 timestamp {s:?}")))
}

fn parse_cursor(s: &str) -> Result<(i64, String), AppError> {
    let invalid = || AppError::BadRequest("invalid cursor".into());
    let (ms, id) = s.split_once(':').ok_or_else(invalid)?;
    let ms: i64 = ms.parse().map_err(|_| invalid())?;
    if id.is_empty() {
        return Err(invalid());
    }
    Ok((ms, id.to_string()))
}
