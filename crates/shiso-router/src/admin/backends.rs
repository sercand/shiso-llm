//! Backend CRUD. Every mutation writes the database and then rebuilds the
//! routing snapshot, so changes take effect without a restart. Stored API
//! keys are never echoed back — responses carry `has_api_key` instead.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Deserializer};

use crate::catalog::{
    self, BackendDef, BackendView, DeleteBackend, SchedulingClass, default_inject_usage,
};
use crate::config::BackendKind;
use crate::error::AppError;
use crate::state::AppState;

use super::keys::blocking;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CreateBackendRequest {
    pub name: String,
    pub kinds: Vec<BackendKind>,
    pub base_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub anthropic_version: Option<String>,
    #[serde(default = "default_inject_usage")]
    pub inject_usage: bool,
    #[serde(default)]
    pub scheduling_class: SchedulingClass,
    #[serde(default)]
    pub exclusive_group_id: Option<String>,
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    #[serde(default)]
    pub max_tokens_cap: Option<i64>,
    #[serde(default)]
    pub tok_s_estimate: Option<f64>,
}

/// PATCH body: absent field = leave unchanged, `null` = clear (for the
/// nullable fields), value = set. Setting one key source clears the other,
/// keeping the at-most-one-of invariant.
#[expect(
    clippy::option_option,
    reason = "outer None is absent field, inner None is explicit null"
)]
#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct UpdateBackendRequest {
    pub kinds: Option<Vec<BackendKind>>,
    pub base_url: Option<String>,
    #[serde(deserialize_with = "some_option")]
    pub api_key: Option<Option<String>>,
    #[serde(deserialize_with = "some_option")]
    pub api_key_env: Option<Option<String>>,
    #[serde(deserialize_with = "some_option")]
    pub anthropic_version: Option<Option<String>>,
    pub inject_usage: Option<bool>,
    pub scheduling_class: Option<SchedulingClass>,
    #[serde(deserialize_with = "some_option")]
    pub exclusive_group_id: Option<Option<String>>,
    #[serde(deserialize_with = "some_option")]
    pub max_concurrency: Option<Option<u32>>,
    #[serde(deserialize_with = "some_option")]
    pub max_tokens_cap: Option<Option<i64>>,
    #[serde(deserialize_with = "some_option")]
    pub tok_s_estimate: Option<Option<f64>>,
}

#[expect(
    clippy::option_option,
    reason = "outer None is absent field, inner None is explicit null"
)]
/// Distinguishes an absent field (outer None) from an explicit null (inner None).
fn some_option<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Ok(Some(Option::<T>::deserialize(d)?))
}

pub(crate) async fn list_backends(State(state): State<AppState>) -> Json<Vec<BackendView>> {
    Json(
        state
            .routing()
            .backends
            .iter()
            .map(BackendDef::view)
            .collect(),
    )
}

pub(crate) async fn get_backend(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<BackendView>, AppError> {
    state
        .routing()
        .backends
        .iter()
        .find(|b| b.name == name)
        .map(|b| Json(b.view()))
        .ok_or(AppError::NotFound)
}

pub(crate) async fn create_backend(
    State(state): State<AppState>,
    Json(req): Json<CreateBackendRequest>,
) -> Result<(StatusCode, Json<BackendView>), AppError> {
    let def = BackendDef {
        name: req.name.trim().to_string(),
        kinds: req.kinds,
        base_url: req.base_url.trim().to_string(),
        api_key: blank_to_none(req.api_key),
        api_key_env: blank_to_none(req.api_key_env),
        anthropic_version: blank_to_none(req.anthropic_version),
        inject_usage: req.inject_usage,
        scheduling_class: req.scheduling_class,
        exclusive_group_id: blank_to_none(req.exclusive_group_id),
        max_concurrency: req.max_concurrency,
        max_tokens_cap: req.max_tokens_cap,
        tok_s_estimate: req.tok_s_estimate,
    };
    validate(&state, &def)?;

    let db = state.db.clone();
    let def_clone = def.clone();
    if !blocking(move || catalog::insert_backend(&db, &def_clone)).await? {
        return Err(AppError::Conflict(format!(
            "backend {:?} already exists",
            def.name
        )));
    }
    reload(&state).await?;
    Ok((StatusCode::CREATED, Json(def.view())))
}

pub(crate) async fn update_backend(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<UpdateBackendRequest>,
) -> Result<Json<BackendView>, AppError> {
    let mut def = state
        .routing()
        .backends
        .iter()
        .find(|b| b.name == name)
        .cloned()
        .ok_or(AppError::NotFound)?;

    if let Some(kinds) = req.kinds {
        def.kinds = kinds;
    }
    if let Some(base_url) = req.base_url {
        def.base_url = base_url.trim().to_string();
    }
    if let Some(api_key) = req.api_key {
        def.api_key = blank_to_none(api_key);
        if def.api_key.is_some() {
            def.api_key_env = None;
        }
    }
    if let Some(api_key_env) = req.api_key_env {
        def.api_key_env = blank_to_none(api_key_env);
        if def.api_key_env.is_some() {
            def.api_key = None;
        }
    }
    if let Some(version) = req.anthropic_version {
        def.anthropic_version = blank_to_none(version);
    }
    if let Some(inject) = req.inject_usage {
        def.inject_usage = inject;
    }
    if let Some(class) = req.scheduling_class {
        def.scheduling_class = class;
    }
    if let Some(group) = req.exclusive_group_id {
        def.exclusive_group_id = blank_to_none(group);
    }
    if let Some(v) = req.max_concurrency {
        def.max_concurrency = v;
    }
    if let Some(v) = req.max_tokens_cap {
        def.max_tokens_cap = v;
    }
    if let Some(v) = req.tok_s_estimate {
        def.tok_s_estimate = v;
    }
    validate(&state, &def)?;

    let db = state.db.clone();
    let def_clone = def.clone();
    if !blocking(move || catalog::update_backend(&db, &def_clone)).await? {
        return Err(AppError::NotFound);
    }
    reload(&state).await?;
    Ok(Json(def.view()))
}

pub(crate) async fn delete_backend(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, AppError> {
    let db = state.db.clone();
    let name_clone = name.clone();
    match blocking(move || catalog::delete_backend(&db, &name_clone)).await? {
        DeleteBackend::Deleted => {
            reload(&state).await?;
            Ok(StatusCode::NO_CONTENT)
        }
        DeleteBackend::Missing => Err(AppError::NotFound),
        DeleteBackend::InUse(models) => Err(AppError::Conflict(format!(
            "backend {name:?} is still used by model(s): {}",
            models.join(", ")
        ))),
    }
}

/// Structural validation plus an eager check that a referenced environment
/// variable actually exists — a typo there would only surface as a broken
/// route later.
fn validate(state: &AppState, def: &BackendDef) -> Result<(), AppError> {
    catalog::validate_backend(def, &state.routing().group_ids()).map_err(AppError::BadRequest)?;
    def.resolved_api_key()
        .map_err(|e| AppError::BadRequest(e.to_string()))?;
    Ok(())
}

fn blank_to_none(v: Option<String>) -> Option<String> {
    v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

pub(crate) async fn reload(state: &AppState) -> Result<(), AppError> {
    let state = state.clone();
    blocking(move || state.reload_routing()).await
}
