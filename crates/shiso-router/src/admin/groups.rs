//! Exclusive-group CRUD. A group names a device that several backends
//! share, and carries the timings that govern how its lease moves between
//! them. Like the backend and model handlers, every mutation writes SQLite
//! and then rebuilds the routing snapshot, so changes take effect without a
//! restart — `Scheduler::reconcile` picks the new timings up from there.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde::Deserialize;

use crate::catalog::{self, DeleteGroup, GroupDef};
use crate::error::AppError;
use crate::state::AppState;

use super::backends::reload;
use super::keys::blocking;

/// Body of a `PUT` — the id comes from the path. Groups are not renameable:
/// backends reference them by id.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GroupBody {
    pub min_hold_ms: i64,
    pub max_hold_ms: i64,
    pub drain_timeout_ms: i64,
}

pub(crate) async fn list_groups(State(state): State<AppState>) -> Json<Vec<GroupDef>> {
    Json(state.routing().groups.clone())
}

pub(crate) async fn get_group(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<GroupDef>, AppError> {
    state
        .routing()
        .groups
        .iter()
        .find(|g| g.id == id)
        .map(|g| Json(g.clone()))
        .ok_or(AppError::NotFound)
}

pub(crate) async fn create_group(
    State(state): State<AppState>,
    Json(def): Json<GroupDef>,
) -> Result<(StatusCode, Json<GroupDef>), AppError> {
    let def = GroupDef {
        id: def.id.trim().to_string(),
        ..def
    };
    catalog::validate_group(&def).map_err(AppError::BadRequest)?;

    let db = state.db.clone();
    let def_clone = def.clone();
    if !blocking(move || catalog::insert_group(&db, &def_clone)).await? {
        return Err(AppError::Conflict(format!(
            "exclusive group {:?} already exists",
            def.id
        )));
    }
    reload(&state).await?;
    Ok((StatusCode::CREATED, Json(def)))
}

pub(crate) async fn replace_group(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<GroupBody>,
) -> Result<Json<GroupDef>, AppError> {
    let def = GroupDef {
        id,
        min_hold_ms: body.min_hold_ms,
        max_hold_ms: body.max_hold_ms,
        drain_timeout_ms: body.drain_timeout_ms,
    };
    catalog::validate_group(&def).map_err(AppError::BadRequest)?;

    let db = state.db.clone();
    let def_clone = def.clone();
    if !blocking(move || catalog::update_group(&db, &def_clone)).await? {
        return Err(AppError::NotFound);
    }
    reload(&state).await?;
    Ok(Json(def))
}

pub(crate) async fn delete_group(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    let db = state.db.clone();
    let id_clone = id.clone();
    match blocking(move || catalog::delete_group(&db, &id_clone)).await? {
        DeleteGroup::Deleted => {
            reload(&state).await?;
            Ok(StatusCode::NO_CONTENT)
        }
        DeleteGroup::Missing => Err(AppError::NotFound),
        DeleteGroup::InUse(backends) => Err(AppError::Conflict(format!(
            "exclusive group {id:?} is still used by backend(s): {}",
            backends.join(", ")
        ))),
    }
}
