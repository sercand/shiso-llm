//! Model-route CRUD. A model is edited as a whole (pricing + ordered
//! target list) — `PUT` replaces the definition — because targets carry no
//! stable identity of their own. Every mutation rebuilds the routing
//! snapshot, so changes take effect without a restart.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::Deserialize;

use crate::catalog::{self, ModelDef, OnShedWithoutFallback, TargetDef};
use crate::config::PricingCfg;
use crate::error::AppError;
use crate::state::AppState;

use super::backends::reload;
use super::keys::blocking;

/// Model definitions with their targets. Also serves as the admin UI's
/// cheap login-check call.
pub(crate) async fn list_models(State(state): State<AppState>) -> Json<Vec<ModelDef>> {
    Json(state.routing().models.clone())
}

pub(crate) async fn get_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<ModelDef>, AppError> {
    state
        .routing()
        .models
        .iter()
        .find(|m| m.name == name)
        .map(|m| Json(m.clone()))
        .ok_or(AppError::NotFound)
}

/// Body of a `PUT` — the name comes from the path (models are not
/// renameable; keys' `allowed_models` and log rows reference the name).
/// `aliases`, by contrast, are freely editable: they carry no identity, a
/// request arriving under one is logged under the canonical name.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelBody {
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub pricing: Option<PricingCfg>,
    #[serde(default)]
    pub qos_max_wait_ms: Option<i64>,
    #[serde(default)]
    pub qos_background_max_wait_ms: Option<i64>,
    #[serde(default)]
    pub on_shed_without_fallback: OnShedWithoutFallback,
    pub targets: Vec<TargetDef>,
}

pub(crate) async fn create_model(
    State(state): State<AppState>,
    Json(def): Json<ModelDef>,
) -> Result<(StatusCode, Json<ModelDef>), AppError> {
    let def = ModelDef {
        name: def.name.trim().to_string(),
        aliases: trimmed(def.aliases),
        ..def
    };
    validate(&state, &def)?;

    let db = state.db.clone();
    let def_clone = def.clone();
    if !blocking(move || catalog::insert_model(&db, &def_clone)).await? {
        return Err(AppError::Conflict(format!(
            "model {:?} already exists",
            def.name
        )));
    }
    reload(&state).await?;
    Ok((StatusCode::CREATED, Json(def)))
}

pub(crate) async fn replace_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(body): Json<ModelBody>,
) -> Result<Json<ModelDef>, AppError> {
    let def = ModelDef {
        name,
        aliases: trimmed(body.aliases),
        pricing: body.pricing,
        qos_max_wait_ms: body.qos_max_wait_ms,
        qos_background_max_wait_ms: body.qos_background_max_wait_ms,
        on_shed_without_fallback: body.on_shed_without_fallback,
        targets: body.targets,
    };
    validate(&state, &def)?;

    let db = state.db.clone();
    let def_clone = def.clone();
    if !blocking(move || catalog::replace_model(&db, &def_clone)).await? {
        return Err(AppError::NotFound);
    }
    reload(&state).await?;
    Ok(Json(def))
}

pub(crate) async fn delete_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, AppError> {
    let db = state.db.clone();
    if !blocking(move || catalog::delete_model(&db, &name)).await? {
        return Err(AppError::NotFound);
    }
    reload(&state).await?;
    Ok(StatusCode::NO_CONTENT)
}

fn trimmed(values: Vec<String>) -> Vec<String> {
    values.into_iter().map(|v| v.trim().to_string()).collect()
}

fn validate(state: &AppState, def: &ModelDef) -> Result<(), AppError> {
    let routing = state.routing();
    catalog::validate_model(def, &routing.backend_names()).map_err(AppError::BadRequest)?;
    // Aliases share one namespace with model names: whatever this
    // definition answers to must not already be served by another route.
    // (Its own row, if any, is excluded — a PUT keeps its name.)
    let others = routing.models.iter().filter(|m| m.name != def.name);
    catalog::validate_no_name_conflict(def, others).map_err(AppError::BadRequest)
}
