//! Client-authenticated key management (`/v1/keys/*`).
//!
//! Top-level API keys with `can_create_children = true` may mint child keys
//! under themselves without holding the router's master admin token. Child
//! keys are strictly 1 level deep, may only be granted models allowed to
//! their parent, and are isolated to their owning parent on every read,
//! update, delete, and usage query.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::json;

use crate::admin::keys::{
    blocking, some_option, usage_snapshot_json, validate_credit_limit, validate_name,
    validate_token_limit,
};
use crate::admin::logs::parse_rfc3339_ms;
use crate::admin::usage::{Filters, Grouping, UsageBucket, query_usage, usage_windows};
use crate::error::AppError;
use crate::limits::ResetInterval;
use crate::state::AppState;
use crate::tokens::{KeyRecord, generate_token, hash_token};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CreateChildKeyRequest {
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
}

#[expect(
    clippy::option_option,
    reason = "outer None is absent field, inner None is explicit null"
)]
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct UpdateChildKeyRequest {
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
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ClientUsageQuery {
    /// `day` (default) | `model` | `token`.
    pub group_by: Option<String>,
    pub token_id: Option<String>,
    pub model: Option<String>,
    pub since: Option<String>,
    pub until: Option<String>,
}

fn require_top_level(caller: &KeyRecord) -> Result<(), AppError> {
    if caller.parent_id.is_some() {
        return Err(AppError::Forbidden(
            "child API keys cannot manage child keys".into(),
        ));
    }
    Ok(())
}

/// Validates that every model in `models` is a canonical configured model AND
/// is permitted for `caller`. For private models `caller` cannot access (even
/// when referenced by alias), returns `400 BadRequest` matching the
/// unknown-model wording so private model names cannot be probed.
fn validate_child_models(
    state: &AppState,
    caller: &KeyRecord,
    models: &[String],
    field: &str,
) -> Result<(), AppError> {
    let routing = state.routing();
    for m in models {
        match routing.model_named(m) {
            Some(def) if def.private && !caller.allows_model(&def.name, true) => {
                return Err(AppError::BadRequest(format!(
                    "{field} entry {m:?} is not a configured model"
                )));
            }
            Some(def) if def.name != *m => {
                return Err(AppError::BadRequest(format!(
                    "{field} entry {m:?} is an alias of model {:?} — list the model name",
                    def.name
                )));
            }
            Some(def) if !caller.allows_model(&def.name, def.private) => {
                return Err(AppError::Forbidden(format!(
                    "{field} entry {m:?} is not allowed for the parent API key"
                )));
            }
            Some(_) => {}
            None => {
                return Err(AppError::BadRequest(format!(
                    "{field} entry {m:?} is not a configured model"
                )));
            }
        }
    }
    Ok(())
}

/// `POST /v1/keys`: create a child key owned by `caller`.
pub(crate) async fn create_child_key(
    State(state): State<AppState>,
    Extension(caller): Extension<Arc<KeyRecord>>,
    Json(req): Json<CreateChildKeyRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), AppError> {
    require_top_level(&caller)?;
    if !caller.can_create_children {
        return Err(AppError::Forbidden(
            "API key is not allowed to create child keys".into(),
        ));
    }

    let name = validate_name(&req.name)?;
    validate_credit_limit(req.credit_limit)?;
    validate_token_limit(req.token_limit)?;

    let allowed_models = match req.allowed_models {
        Some(models) => {
            validate_child_models(&state, &caller, &models, "allowed_models")?;
            Some(models)
        }
        None => caller.allowed_models.clone(),
    };
    let flex_models = match req.flex_models {
        Some(models) => {
            validate_child_models(&state, &caller, &models, "flex_models")?;
            Some(models)
        }
        None => caller.flex_models.clone(),
    };

    let token = generate_token();
    let hash = hash_token(&token);
    let mut record = KeyRecord::new(name);
    record.credit_limit = req.credit_limit;
    record.credit_limit_reset = req.credit_limit_reset.unwrap_or_default();
    record.token_limit = req.token_limit;
    record.token_limit_reset = req.token_limit_reset.unwrap_or_default();
    record.allowed_models = allowed_models;
    record.flex_models = flex_models;
    record.parent_id = Some(caller.id.clone());
    record.can_create_children = false;

    let db = state.db.clone();
    let record_clone = record.clone();
    blocking(move || db.insert(&record_clone, &hash)).await?;
    state.token_cache.insert(hash, Arc::new(record.clone()));

    let mut body = serde_json::to_value(&record).expect("KeyRecord serializes");
    body["token"] = json!(token);
    Ok((StatusCode::CREATED, Json(body)))
}

/// `GET /v1/keys`: list `caller`'s own child keys with current-window usage.
pub(crate) async fn list_child_keys(
    State(state): State<AppState>,
    Extension(caller): Extension<Arc<KeyRecord>>,
) -> Result<Json<Vec<serde_json::Value>>, AppError> {
    require_top_level(&caller)?;
    let db = state.db.clone();
    let parent_id = caller.id.clone();
    let children = blocking(move || db.list_children(&parent_id)).await?;
    let now = time::OffsetDateTime::now_utc();
    let items = children
        .into_iter()
        .map(|record| {
            let w = state.usage_tracker.snapshot(&record, now);
            let mut body = serde_json::to_value(&record).expect("KeyRecord serializes");
            body["usage"] = usage_snapshot_json(w);
            body
        })
        .collect();
    Ok(Json(items))
}

/// `GET /v1/keys/{id}`: fetch one child key owned by `caller` plus its
/// four-window usage aggregates.
pub(crate) async fn get_child_key(
    State(state): State<AppState>,
    Extension(caller): Extension<Arc<KeyRecord>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    require_top_level(&caller)?;
    let db = state.db.clone();
    let caller_id = caller.id.clone();
    let detail = blocking(move || {
        let Some(record) = db.get(&id)? else {
            return Ok(None);
        };
        if record.parent_id.as_deref() != Some(caller_id.as_str()) {
            return Ok(None);
        }
        let mut body = serde_json::to_value(&record).expect("KeyRecord serializes");
        body["usage"] = usage_windows(&db, &Filters::for_token(&id))?;
        Ok(Some(body))
    })
    .await?
    .ok_or(AppError::NotFound)?;
    Ok(Json(detail))
}

/// `PATCH /v1/keys/{id}`: update a child key owned by `caller`. Clearing
/// `allowed_models` with explicit `null` resets it to `caller.allowed_models`,
/// never widening past the parent.
pub(crate) async fn update_child_key(
    State(state): State<AppState>,
    Extension(caller): Extension<Arc<KeyRecord>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateChildKeyRequest>,
) -> Result<Json<KeyRecord>, AppError> {
    require_top_level(&caller)?;
    let db = state.db.clone();
    let id_clone = id.clone();
    let before = blocking(move || db.get(&id_clone))
        .await?
        .filter(|r| r.parent_id.as_deref() == Some(caller.id.as_str()))
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
        record.allowed_models = match models {
            Some(list) => {
                validate_child_models(&state, &caller, &list, "allowed_models")?;
                Some(list)
            }
            None => caller.allowed_models.clone(),
        };
    }
    if let Some(models) = req.flex_models {
        record.flex_models = match models {
            Some(list) => {
                validate_child_models(&state, &caller, &list, "flex_models")?;
                Some(list)
            }
            None => caller.flex_models.clone(),
        };
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
        let db = state.db.clone();
        let tracker = Arc::clone(&state.usage_tracker);
        let record_clone = record.clone();
        blocking(move || tracker.warm_key(&db, &record_clone, time::OffsetDateTime::now_utc()))
            .await?;
    }

    Ok(Json(record))
}

/// `DELETE /v1/keys/{id}`: delete a child key owned by `caller`.
pub(crate) async fn delete_child_key(
    State(state): State<AppState>,
    Extension(caller): Extension<Arc<KeyRecord>>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    require_top_level(&caller)?;
    let db = state.db.clone();
    let id_clone = id.clone();
    let caller_id = caller.id.clone();
    let removed = blocking(move || {
        let Some(rec) = db.get(&id_clone)? else {
            return Ok(None);
        };
        if rec.parent_id.as_deref() != Some(caller_id.as_str()) {
            return Ok(None);
        }
        db.delete(&id_clone)
    })
    .await?;

    match removed {
        Some(hash) => {
            state.token_cache.remove(&hash);
            state.usage_tracker.remove(&id);
            Ok(StatusCode::NO_CONTENT)
        }
        None => Err(AppError::NotFound),
    }
}

/// `GET /v1/keys/usage`: usage aggregates strictly scoped to `caller` and its
/// children (`l.token_id = caller.id OR l.parent_token_id = caller.id`).
pub(crate) async fn keys_usage(
    State(state): State<AppState>,
    Extension(caller): Extension<Arc<KeyRecord>>,
    Query(q): Query<ClientUsageQuery>,
) -> Result<Json<Vec<UsageBucket>>, AppError> {
    let group_name = q.group_by.as_deref().unwrap_or("day");
    if group_name == "backend" {
        return Err(AppError::BadRequest(
            "group_by must be day, model, or token".into(),
        ));
    }
    let group = Grouping::parse(group_name)?;
    let filters = Filters {
        token_id: q.token_id,
        include_children: false,
        parent_token_id: None,
        scope_parent_id: Some(caller.id.clone()),
        model: q.model,
        backend: None,
        since: q.since.as_deref().map(parse_rfc3339_ms).transpose()?,
        until: q.until.as_deref().map(parse_rfc3339_ms).transpose()?,
    };
    let db = state.db.clone();
    let buckets = blocking(move || query_usage(&db, group, &filters)).await?;
    Ok(Json(buckets))
}

/// `GET /v1/keys/models`: read-only list of models `caller` is allowed to use,
/// with their aliases and pricing (no backend or target internals).
pub(crate) async fn keys_models(
    State(state): State<AppState>,
    Extension(caller): Extension<Arc<KeyRecord>>,
) -> Result<Json<Vec<serde_json::Value>>, AppError> {
    let routing = state.routing();
    let parent = match caller.parent_id.as_deref() {
        Some(pid) => match state.token_cache.get_by_id(pid) {
            Some(p) if !p.disabled => Some(p),
            _ => return Err(AppError::Forbidden("parent API key is disabled".into())),
        },
        None => None,
    };
    let mut allowed: Vec<_> = routing
        .models
        .iter()
        .filter(|m| {
            caller.allows_model(&m.name, m.private)
                && parent
                    .as_ref()
                    .is_none_or(|p| p.allows_model(&m.name, m.private))
        })
        .collect();
    allowed.sort_by(|a, b| a.name.cmp(&b.name));
    let items = allowed
        .into_iter()
        .map(|m| {
            json!({
                "name": m.name,
                "aliases": m.aliases,
                "pricing": m.pricing,
            })
        })
        .collect();
    Ok(Json(items))
}
