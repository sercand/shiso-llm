use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Deserializer};
use serde_json::json;

use crate::error::AppError;
use crate::limits::{ResetInterval, UsageSnapshot};
use crate::state::AppState;
use crate::tokens::{KeyRecord, generate_token, hash_token};

use super::usage::{Filters, usage_windows};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CreateTokenRequest {
    pub name: String,
    #[serde(default)]
    pub credit_limit: Option<f64>,
    #[serde(default)]
    pub credit_limit_reset: Option<ResetInterval>,
    #[serde(default)]
    pub token_limit: Option<i64>,
    #[serde(default)]
    pub token_limit_reset: Option<ResetInterval>,
    #[serde(default)]
    pub allowed_models: Option<Vec<String>>,
    #[serde(default)]
    pub flex_models: Option<Vec<String>>,
    #[serde(default)]
    pub can_create_children: bool,
}

/// PATCH body: absent field = leave unchanged, `null` = clear (for the
/// nullable fields), value = set.
#[expect(
    clippy::option_option,
    reason = "outer None is absent field, inner None is explicit null"
)]
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct UpdateTokenRequest {
    pub name: Option<String>,
    pub disabled: Option<bool>,
    #[serde(deserialize_with = "some_option")]
    pub credit_limit: Option<Option<f64>>,
    pub credit_limit_reset: Option<ResetInterval>,
    #[serde(deserialize_with = "some_option")]
    pub token_limit: Option<Option<i64>>,
    pub token_limit_reset: Option<ResetInterval>,
    #[serde(deserialize_with = "some_option")]
    pub allowed_models: Option<Option<Vec<String>>>,
    #[serde(deserialize_with = "some_option")]
    pub flex_models: Option<Option<Vec<String>>>,
    pub can_create_children: Option<bool>,
}

#[expect(
    clippy::option_option,
    reason = "outer None is absent field, inner None is explicit null"
)]
/// Distinguishes an absent field (outer None) from an explicit null (inner None).
pub(crate) fn some_option<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Ok(Some(Option::<T>::deserialize(d)?))
}

pub(crate) fn usage_snapshot_json(w: UsageSnapshot) -> serde_json::Value {
    json!({
        "window_cost": w.cost,
        "window_tokens": w.tokens,
        "window_prompt_tokens": w.prompt_tokens,
        "window_completion_tokens": w.completion_tokens,
    })
}

pub(crate) async fn create_token(
    State(state): State<AppState>,
    Json(req): Json<CreateTokenRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), AppError> {
    let name = validate_name(&req.name)?;
    validate_credit_limit(req.credit_limit)?;
    validate_token_limit(req.token_limit)?;
    if let Some(models) = &req.allowed_models {
        validate_model_list(&state, models, "allowed_models")?;
    }
    if let Some(models) = &req.flex_models {
        validate_model_list(&state, models, "flex_models")?;
    }

    let token = generate_token();
    let hash = hash_token(&token);
    let mut record = KeyRecord::new(name);
    record.credit_limit = req.credit_limit;
    record.credit_limit_reset = req.credit_limit_reset.unwrap_or_default();
    record.token_limit = req.token_limit;
    record.token_limit_reset = req.token_limit_reset.unwrap_or_default();
    record.allowed_models = req.allowed_models;
    record.flex_models = req.flex_models;
    record.can_create_children = req.can_create_children;

    let db = state.db.clone();
    let record_clone = record.clone();
    blocking(move || db.insert(&record_clone, &hash)).await?;
    state.token_cache.insert(hash, Arc::new(record.clone()));

    // The plaintext token is returned exactly once; only its hash is stored.
    let mut body = serde_json::to_value(&record).expect("KeyRecord serializes");
    body["token"] = json!(token);
    Ok((StatusCode::CREATED, Json(body)))
}

/// Key list with each key's current-window usage (from the in-memory
/// counters — no SQL on this path) for limit progress bars.
pub(crate) async fn list_tokens(
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, AppError> {
    let db = state.db.clone();
    let tokens = blocking(move || db.list()).await?;
    let mut child_counts: HashMap<String, usize> = HashMap::new();
    for t in &tokens {
        if let Some(pid) = &t.parent_id {
            *child_counts.entry(pid.clone()).or_default() += 1;
        }
    }
    let now = time::OffsetDateTime::now_utc();
    let items = tokens
        .into_iter()
        .map(|record| {
            let w = state.usage_tracker.snapshot(&record, now);
            let mut body = serde_json::to_value(&record).expect("KeyRecord serializes");
            body["usage"] = usage_snapshot_json(w);
            if record.parent_id.is_none() {
                let cw = state.usage_tracker.children_snapshot(&record, now);
                body["children_usage"] = usage_snapshot_json(cw);
                body["children_count"] =
                    json!(child_counts.get(&record.id).copied().unwrap_or(0));
            }
            body
        })
        .collect();
    Ok(Json(items))
}

/// Key detail: the record plus usage aggregates for the OpenRouter-style
/// stat cards (today / this week / this month / all time, all UTC).
pub(crate) async fn get_token(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let db = state.db.clone();
    let tracker = Arc::clone(&state.usage_tracker);
    let detail = blocking(move || {
        let Some(record) = db.get(&id)? else {
            return Ok(None);
        };
        let mut body = serde_json::to_value(&record).expect("KeyRecord serializes");
        body["usage"] = usage_windows(&db, &Filters::for_token(&id))?;
        if record.parent_id.is_none() {
            body["children_usage"] = usage_windows(&db, &Filters::for_parent(&id))?;
            let now = time::OffsetDateTime::now_utc();
            let children: Vec<serde_json::Value> = db
                .list_children(&id)?
                .into_iter()
                .map(|c| {
                    let w = tracker.snapshot(&c, now);
                    let mut cbody = serde_json::to_value(&c).expect("KeyRecord serializes");
                    cbody["usage"] = usage_snapshot_json(w);
                    cbody
                })
                .collect();
            body["children"] = json!(children);
        }
        Ok(Some(body))
    })
    .await?
    .ok_or(AppError::NotFound)?;
    Ok(Json(detail))
}

pub(crate) async fn update_token(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<UpdateTokenRequest>,
) -> Result<Json<KeyRecord>, AppError> {
    let db = state.db.clone();
    let id_clone = id.clone();
    let before = blocking(move || db.get(&id_clone))
        .await?
        .ok_or(AppError::NotFound)?;
    let mut record = before.clone();

    if let Some(name) = &req.name {
        record.name = validate_name(name)?;
    }
    if let Some(disabled) = req.disabled {
        record.disabled = disabled;
    }
    if let Some(limit) = req.credit_limit {
        validate_credit_limit(limit)?;
        record.credit_limit = limit;
    }
    if let Some(reset) = req.credit_limit_reset {
        record.credit_limit_reset = reset;
    }
    if let Some(limit) = req.token_limit {
        validate_token_limit(limit)?;
        record.token_limit = limit;
    }
    if let Some(reset) = req.token_limit_reset {
        record.token_limit_reset = reset;
    }
    if let Some(models) = req.allowed_models {
        if let Some(models) = &models {
            validate_model_list(&state, models, "allowed_models")?;
        }
        record.allowed_models = models;
    }
    if let Some(models) = req.flex_models {
        if let Some(models) = &models {
            validate_model_list(&state, models, "flex_models")?;
        }
        record.flex_models = models;
    }
    if let Some(can_create) = req.can_create_children {
        if can_create && record.parent_id.is_some() {
            return Err(AppError::BadRequest(
                "child keys cannot create child keys".into(),
            ));
        }
        record.can_create_children = can_create;
    }

    let limits_changed = record.credit_limit != before.credit_limit
        || record.credit_limit_reset != before.credit_limit_reset
        || record.token_limit != before.token_limit
        || record.token_limit_reset != before.token_limit_reset;

    let db = state.db.clone();
    let record_clone = record.clone();
    let hash = blocking(move || db.update(&record_clone))
        .await?
        .ok_or(AppError::NotFound)?;
    state.token_cache.insert(hash, Arc::new(record.clone()));

    if limits_changed {
        // Interval changes move the window boundary; recompute the sums
        // from request_logs so enforcement matches the new window.
        let db = state.db.clone();
        let tracker = Arc::clone(&state.usage_tracker);
        let record_clone = record.clone();
        blocking(move || tracker.warm_key(&db, &record_clone, time::OffsetDateTime::now_utc()))
            .await?;
    }

    Ok(Json(record))
}

pub(crate) async fn delete_token(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    let db = state.db.clone();
    let id_clone = id.clone();
    let removed = blocking(move || db.delete_cascade(&id_clone)).await?;

    if removed.is_empty() {
        return Err(AppError::NotFound);
    }
    for (del_id, hash) in removed {
        state.token_cache.remove(&hash);
        state.usage_tracker.remove(&del_id);
    }
    Ok(StatusCode::NO_CONTENT)
}

pub(crate) async fn blocking<T, F>(f: F) -> Result<T, AppError>
where
    F: FnOnce() -> anyhow::Result<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| AppError::Internal(e.to_string()))?
        .map_err(|e| AppError::Internal(e.to_string()))
}

pub(crate) fn validate_name(name: &str) -> Result<String, AppError> {
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err(AppError::BadRequest("token name must not be empty".into()));
    }
    Ok(name)
}

pub(crate) fn validate_credit_limit(limit: Option<f64>) -> Result<(), AppError> {
    if let Some(v) = limit
        && (!v.is_finite() || v < 0.0)
    {
        return Err(AppError::BadRequest(
            "credit_limit must be a non-negative number".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_token_limit(limit: Option<i64>) -> Result<(), AppError> {
    if let Some(v) = limit
        && v < 0
    {
        return Err(AppError::BadRequest(
            "token_limit must be non-negative".into(),
        ));
    }
    Ok(())
}

/// Allowlist/flex-list entries must name configured models — a typo here
/// would silently lock the key out of the model it was meant to reach (or
/// silently never force flex). Aliases are rejected for the same reason:
/// the proxy matches these lists against the canonical name an alias
/// resolves to, so an aliased entry would never match anything.
pub(crate) fn validate_model_list(
    state: &AppState,
    models: &[String],
    field: &str,
) -> Result<(), AppError> {
    let routing = state.routing();
    for m in models {
        match routing.model_named(m) {
            Some(def) if def.name == *m => {}
            Some(def) => {
                return Err(AppError::BadRequest(format!(
                    "{field} entry {m:?} is an alias of model {:?} — list the model name",
                    def.name
                )));
            }
            None => {
                return Err(AppError::BadRequest(format!(
                    "{field} entry {m:?} is not a configured model"
                )));
            }
        }
    }
    Ok(())
}
