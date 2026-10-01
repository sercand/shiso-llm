use axum::Json;
use axum::extract::{Path, Query, State};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::AppError;
use crate::limits::{ResetInterval, window_start_ms};
use crate::state::AppState;
use crate::tokens::Db;

use super::logs::parse_rfc3339_ms;

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct UsageQuery {
    /// `day` (default) | `model` | `token` | `backend`.
    pub group_by: Option<String>,
    pub token_id: Option<String>,
    pub model: Option<String>,
    pub backend: Option<String>,
    pub since: Option<String>,
    pub until: Option<String>,
}

/// Time range for the model/provider analytics endpoints. It scopes the
/// daily series and the breakdowns; the four aggregate cards always cover
/// their own fixed UTC windows.
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct RangeQuery {
    pub since: Option<String>,
    pub until: Option<String>,
}

impl RangeQuery {
    fn parse(&self) -> Result<(Option<i64>, Option<i64>), AppError> {
        Ok((
            self.since.as_deref().map(parse_rfc3339_ms).transpose()?,
            self.until.as_deref().map(parse_rfc3339_ms).transpose()?,
        ))
    }
}

/// The slice of `request_logs` an aggregate covers. Every clause qualifies
/// its column with `l.` because the token-grouped query joins `tokens`,
/// which carries `id` and `name` of its own.
#[derive(Default, Clone)]
pub(crate) struct Filters {
    pub token_id: Option<String>,
    pub model: Option<String>,
    pub backend: Option<String>,
    pub since: Option<i64>,
    pub until: Option<i64>,
}

impl Filters {
    pub(crate) fn for_token(id: &str) -> Self {
        Filters {
            token_id: Some(id.to_string()),
            ..Filters::default()
        }
    }

    fn for_model(name: &str) -> Self {
        Filters {
            model: Some(name.to_string()),
            ..Filters::default()
        }
    }

    fn for_backend(name: &str) -> Self {
        Filters {
            backend: Some(name.to_string()),
            ..Filters::default()
        }
    }

    fn push_clauses(&self, sql: &mut String, params: &mut Vec<rusqlite::types::Value>) {
        use rusqlite::types::Value;
        if let Some(t) = &self.token_id {
            sql.push_str(" AND l.token_id = ?");
            params.push(t.clone().into());
        }
        if let Some(m) = &self.model {
            sql.push_str(" AND l.model = ?");
            params.push(m.clone().into());
        }
        if let Some(b) = &self.backend {
            sql.push_str(" AND l.backend = ?");
            params.push(b.clone().into());
        }
        if let Some(s) = self.since {
            sql.push_str(" AND l.created_at >= ?");
            params.push(Value::Integer(s));
        }
        if let Some(u) = self.until {
            sql.push_str(" AND l.created_at < ?");
            params.push(Value::Integer(u));
        }
    }
}

/// One `group_by` choice: the SQL bucket expression, an optional expression
/// for a human-readable label (only opaque ids need one), and whether the
/// natural order is chronological rather than biggest-spender-first.
#[derive(Clone, Copy)]
struct Grouping {
    expr: &'static str,
    label: Option<&'static str>,
    chronological: bool,
}

impl Grouping {
    const DAY: Grouping = Grouping {
        expr: "date(l.created_at/1000, 'unixepoch')",
        label: None,
        chronological: true,
    };
    const MODEL: Grouping = Grouping {
        expr: "l.model",
        label: None,
        chronological: false,
    };
    const BACKEND: Grouping = Grouping {
        expr: "l.backend",
        label: None,
        chronological: false,
    };
    /// Token ids are opaque, so the key name is joined in as the label. It
    /// is null once the key is deleted — logs outlive keys.
    const TOKEN: Grouping = Grouping {
        expr: "l.token_id",
        label: Some("MAX(t.name)"),
        chronological: false,
    };

    fn parse(name: &str) -> Result<Grouping, AppError> {
        match name {
            "day" => Ok(Self::DAY),
            "model" => Ok(Self::MODEL),
            "token" => Ok(Self::TOKEN),
            "backend" => Ok(Self::BACKEND),
            other => Err(AppError::BadRequest(format!(
                "group_by must be day, model, token, or backend (got {other:?})"
            ))),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct UsageBucket {
    /// The group key: a `YYYY-MM-DD` date, model name, token id or backend
    /// name. Null only when grouping by backend over requests that never
    /// reached one (unknown model, quota rejection, exhausted failover).
    pub bucket: Option<String>,
    /// Display name for an opaque bucket — the key's name when grouping by
    /// token, null otherwise (and for keys that have since been deleted).
    pub label: Option<String>,
    pub requests: i64,
    /// Requests that failed: a 4xx/5xx status, or none at all.
    pub errors: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub cached_tokens: i64,
    pub total_tokens: i64,
    pub cost: f64,
    pub avg_ttfb_ms: Option<f64>,
    pub avg_duration_ms: Option<f64>,
}

pub(crate) async fn usage_summary(
    State(state): State<AppState>,
    Query(q): Query<UsageQuery>,
) -> Result<Json<Vec<UsageBucket>>, AppError> {
    let group = Grouping::parse(q.group_by.as_deref().unwrap_or("day"))?;
    let filters = Filters {
        token_id: q.token_id,
        model: q.model,
        backend: q.backend,
        since: q.since.as_deref().map(parse_rfc3339_ms).transpose()?,
        until: q.until.as_deref().map(parse_rfc3339_ms).transpose()?,
    };

    let db = state.db.clone();
    let buckets = super::keys::blocking(move || query_usage(&db, group, &filters)).await?;
    Ok(Json(buckets))
}

/// Analytics for one model: the four aggregate cards, a daily series, and
/// which providers served it and which keys called it. Names are not checked
/// against the catalog — logs outlive the models that produced them, so a
/// deleted model still answers with its history.
pub(crate) async fn model_usage(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<RangeQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let (since, until) = q.parse()?;
    let db = state.db.clone();
    let body = super::keys::blocking(move || {
        let scope = Filters {
            since,
            until,
            ..Filters::for_model(&name)
        };
        Ok(json!({
            "model": name,
            "usage": usage_windows(&db, &Filters::for_model(&name))?,
            "daily": query_usage(&db, Grouping::DAY, &scope)?,
            "by_backend": query_usage(&db, Grouping::BACKEND, &scope)?,
            "by_token": query_usage(&db, Grouping::TOKEN, &scope)?,
        }))
    })
    .await?;
    Ok(Json(body))
}

/// Analytics for one backend, mirroring `model_usage`: totals plus the
/// models it served and the keys whose traffic it carried.
pub(crate) async fn backend_usage(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<RangeQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let (since, until) = q.parse()?;
    let db = state.db.clone();
    let body = super::keys::blocking(move || {
        let scope = Filters {
            since,
            until,
            ..Filters::for_backend(&name)
        };
        Ok(json!({
            "backend": name,
            "usage": usage_windows(&db, &Filters::for_backend(&name))?,
            "daily": query_usage(&db, Grouping::DAY, &scope)?,
            "by_model": query_usage(&db, Grouping::MODEL, &scope)?,
            "by_token": query_usage(&db, Grouping::TOKEN, &scope)?,
        }))
    })
    .await?;
    Ok(Json(body))
}

/// The aggregate columns shared by the grouped and the single-window
/// queries, in the order both readers expect.
const SUM_COLUMNS: &str = "COUNT(*), \
     COALESCE(SUM(l.status IS NULL OR l.status >= 400), 0), \
     COALESCE(SUM(l.prompt_tokens), 0), COALESCE(SUM(l.completion_tokens), 0), \
     COALESCE(SUM(l.cached_tokens), 0), COALESCE(SUM(l.total_tokens), 0), \
     COALESCE(SUM(l.cost), 0.0), AVG(l.ttfb_ms), AVG(l.duration_ms)";

fn query_usage(db: &Db, group: Grouping, f: &Filters) -> anyhow::Result<Vec<UsageBucket>> {
    let join = if group.label.is_some() {
        " LEFT JOIN tokens t ON t.id = l.token_id"
    } else {
        ""
    };
    let mut sql = format!(
        "SELECT {} AS bucket, {}, {SUM_COLUMNS} FROM request_logs l{join} WHERE 1=1",
        group.expr,
        group.label.unwrap_or("NULL"),
    );
    let mut params = Vec::new();
    f.push_clauses(&mut sql, &mut params);
    sql.push_str(" GROUP BY bucket");
    sql.push_str(if group.chronological {
        " ORDER BY bucket"
    } else {
        // Biggest spender first, falling back to volume when nothing is
        // priced (SQLite sorts NULL sums last under DESC).
        " ORDER BY SUM(l.cost) DESC, SUM(l.total_tokens) DESC"
    });

    db.with_read(|conn| {
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params), |row| {
                let bucket: Option<String> = row.get(0)?;
                // Only the token grouping has labels to fall back on, and
                // only the admin console lacks a `tokens` row to join.
                let label: Option<String> = row.get::<_, Option<String>>(1)?.or_else(|| {
                    group.label?;
                    crate::tokens::synthetic_key_name(bucket.as_deref()?).map(str::to_string)
                });
                Ok(UsageBucket {
                    bucket,
                    label,
                    requests: row.get(2)?,
                    errors: row.get(3)?,
                    prompt_tokens: row.get(4)?,
                    completion_tokens: row.get(5)?,
                    cached_tokens: row.get(6)?,
                    total_tokens: row.get(7)?,
                    cost: row.get(8)?,
                    avg_ttfb_ms: row.get(9)?,
                    avg_duration_ms: row.get(10)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    })
}

/// Aggregate totals for one slice of the log — the today/this-week/
/// this-month/all-time cards on the key, model and provider detail pages.
#[derive(Debug, Serialize)]
pub(super) struct WindowSummary {
    pub requests: i64,
    pub errors: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub cached_tokens: i64,
    pub total_tokens: i64,
    pub cost: f64,
    pub avg_ttfb_ms: Option<f64>,
    pub avg_duration_ms: Option<f64>,
}

/// The four cards for `scope`. Windows are UTC and start where the matching
/// limit interval in `limits.rs` starts, so a card lines up with the quota
/// it is read against.
pub(crate) fn usage_windows(db: &Db, scope: &Filters) -> anyhow::Result<serde_json::Value> {
    let now = time::OffsetDateTime::now_utc();
    let window = |since: Option<i64>| {
        window_summary(
            db,
            &Filters {
                since,
                until: None,
                ..scope.clone()
            },
        )
    };
    Ok(json!({
        "today": window(Some(window_start_ms(ResetInterval::Daily, now)))?,
        "this_week": window(Some(window_start_ms(ResetInterval::Weekly, now)))?,
        "this_month": window(Some(window_start_ms(ResetInterval::Monthly, now)))?,
        "all_time": window(None)?,
    }))
}

fn window_summary(db: &Db, f: &Filters) -> anyhow::Result<WindowSummary> {
    let mut sql = format!("SELECT {SUM_COLUMNS} FROM request_logs l WHERE 1=1");
    let mut params = Vec::new();
    f.push_clauses(&mut sql, &mut params);

    db.with_read(|conn| {
        let row = conn.query_row(&sql, rusqlite::params_from_iter(params), |row| {
            Ok(WindowSummary {
                requests: row.get(0)?,
                errors: row.get(1)?,
                prompt_tokens: row.get(2)?,
                completion_tokens: row.get(3)?,
                cached_tokens: row.get(4)?,
                total_tokens: row.get(5)?,
                cost: row.get(6)?,
                avg_ttfb_ms: row.get(7)?,
                avg_duration_ms: row.get(8)?,
            })
        })?;
        Ok(row)
    })
}
