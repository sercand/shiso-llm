//! Admin API handlers. Everything here sits behind
//! `auth::require_admin_token`; the JSON surface is complete enough to
//! power an external UI (see also `assets/admin.html`).

mod active;
mod backends;
mod groups;
mod keys;
mod logs;
mod models;
mod playground;
mod ui;
mod usage;

pub(crate) use active::list_active;
pub(crate) use backends::{
    create_backend, delete_backend, get_backend, list_backends, update_backend,
};
pub(crate) use groups::{create_group, delete_group, get_group, list_groups, replace_group};
pub(crate) use keys::{create_token, delete_token, get_token, list_tokens, update_token};
/// Shared with `active`, which formats the same epoch-millis clock.
pub(crate) use logs::ms_to_rfc3339;
pub(crate) use logs::{get_log, list_logs};
pub(crate) use models::{create_model, delete_model, get_model, list_models, replace_model};
pub(crate) use playground::playground;
pub(crate) use ui::{ui_page, ui_redirect};
pub(crate) use usage::{backend_usage, model_usage, usage_summary};
