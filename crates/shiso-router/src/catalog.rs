//! The routing catalog — backends and model routes — as stored in SQLite
//! and edited through the admin API.
//!
//! Definitions live in the `backends` / `models` / `model_targets` tables;
//! `state::Routing` is the compiled, immutable form the proxy path reads.
//! A config file's `[[backends]]` / `[[models]]` sections only seed an
//! empty catalog once (`seed_from_config`) and are ignored afterwards.

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::config::{ConfigError, resolve_secret};
use crate::tokens::Db;

/// API wire format kinds supported by backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// OpenAI Chat Completions, Responses, and Embeddings APIs.
    Openai,
    /// Anthropic Messages API.
    Anthropic,
}

/// USD per 1M tokens.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PricingCfg {
    /// USD per 1M prompt tokens.
    pub input_per_mtok: f64,
    /// USD per 1M completion tokens.
    pub output_per_mtok: f64,
    /// USD per 1M cached prompt tokens.
    #[serde(default)]
    pub cached_input_per_mtok: Option<f64>,
    /// USD per 1M Anthropic cache write tokens.
    #[serde(default)]
    pub cache_write_per_mtok: Option<f64>,
}

pub(crate) const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

pub(crate) fn default_inject_usage() -> bool {
    true
}

pub(crate) fn default_weight() -> u32 {
    1
}

/// How a backend is scheduled against the device it runs on.
///
/// `Unrestricted` and `Exempt` behave identically at runtime — neither is
/// subject to a group lease, both still honor their own `max_concurrency`.
/// They stay distinct so the admin UI can say *why* a backend sitting on a
/// contended device is unscheduled: `Exempt` means "on the device, but its
/// work is cheap enough to interleave" (the small embedding model),
/// `Unrestricted` means "not on a contended device at all".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchedulingClass {
    /// Not on a contended accelerator device.
    #[default]
    Unrestricted,
    /// Sits on an accelerator device but generates without waiting for the group lease.
    Exempt,
    /// Gang-scheduled under the accelerator group lease.
    Grouped,
}

impl SchedulingClass {
    /// Returns the static string representation of the scheduling class.
    pub fn as_str(self) -> &'static str {
        match self {
            SchedulingClass::Unrestricted => "unrestricted",
            SchedulingClass::Exempt => "exempt",
            SchedulingClass::Grouped => "grouped",
        }
    }

    /// Parses a scheduling class from a string slice.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "unrestricted" => Some(SchedulingClass::Unrestricted),
            "exempt" => Some(SchedulingClass::Exempt),
            "grouped" => Some(SchedulingClass::Grouped),
            _ => None,
        }
    }
}

/// What to do when a request blows its QoS deadline on a route that has no
/// fallback target to shed to. Silently dropping the deadline would make
/// `max_wait_ms` a lie, so the operator picks explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnShedWithoutFallback {
    /// Queue anyway and wait however long it takes.
    #[default]
    Wait,
    /// Fail the request rather than exceed the deadline.
    Error,
}

impl OnShedWithoutFallback {
    /// Returns the static string representation of the shed behavior.
    pub fn as_str(self) -> &'static str {
        match self {
            OnShedWithoutFallback::Wait => "wait",
            OnShedWithoutFallback::Error => "error",
        }
    }

    /// Parses shed behavior from a string slice.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "wait" => Some(OnShedWithoutFallback::Wait),
            "error" => Some(OnShedWithoutFallback::Error),
            _ => None,
        }
    }
}

/// A set of backends sharing one accelerator. At most one member holds the
/// group *lease* — and so may generate — at a time; the timings below bound
/// how the lease moves between them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupDef {
    /// Exclusive group identifier.
    pub id: String,
    /// Anti-thrash floor: how long a holder keeps the lease before a
    /// waiting peer can take it.
    pub min_hold_ms: i64,
    /// Anti-starvation ceiling: once the holder has held this long and
    /// someone is waiting, the lease moves regardless of `min_hold_ms`.
    pub max_hold_ms: i64,
    /// Watchdog: if draining takes longer than this the lease is
    /// force-flipped, because a wedged group takes the whole device down.
    pub drain_timeout_ms: i64,
}

/// A backend as stored: everything needed to reach an upstream, with the
/// secret either inline (`api_key`) or as the name of an environment
/// variable (`api_key_env`) resolved at routing-build time.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendDef {
    /// Unique backend name.
    pub name: String,
    /// API wire formats this base_url serves.
    pub kinds: Vec<BackendKind>,
    /// Base URL of upstream backend.
    pub base_url: String,
    /// Literal API key for outbound authentication.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Environment variable name holding the API key.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Anthropic version header value.
    #[serde(default)]
    pub anthropic_version: Option<String>,
    /// Whether to inject `stream_options.include_usage` for OpenAI streams.
    #[serde(default = "default_inject_usage")]
    pub inject_usage: bool,
    /// Scheduling class for device contention.
    #[serde(default)]
    pub scheduling_class: SchedulingClass,
    /// The device this backend shares, for `grouped` and `exempt` classes.
    #[serde(default)]
    pub exclusive_group_id: Option<String>,
    /// Concurrent in-flight requests allowed on this backend; `None` =
    /// unlimited (no semaphore at all).
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    /// Ceiling clamped onto every request's generated-token limit. Required
    /// for `grouped` backends: an unbounded generation makes drain time —
    /// and therefore every wait estimate — unbounded too.
    #[serde(default)]
    pub max_tokens_cap: Option<i64>,
    /// Seed for the observed-throughput estimator, output tokens/second.
    #[serde(default)]
    pub tok_s_estimate: Option<f64>,
}

/// The admin-API view of a backend: the stored secret is never echoed
/// back, only whether one is set.
#[derive(Debug, Clone, Serialize)]
pub struct BackendView {
    /// Backend name.
    pub name: String,
    /// Supported API wire format kinds.
    pub kinds: Vec<BackendKind>,
    /// Base URL.
    pub base_url: String,
    /// Whether backend has an API key configured.
    pub has_api_key: bool,
    /// Environment variable name holding the API key.
    pub api_key_env: Option<String>,
    /// Anthropic version header value.
    pub anthropic_version: Option<String>,
    /// Whether usage injection is enabled.
    pub inject_usage: bool,
    /// Scheduling class for device contention.
    pub scheduling_class: SchedulingClass,
    /// Associated exclusive group id.
    pub exclusive_group_id: Option<String>,
    /// Maximum concurrency limit.
    pub max_concurrency: Option<u32>,
    /// Generated token limit clamp.
    pub max_tokens_cap: Option<i64>,
    /// Output throughput estimate in tokens per second.
    pub tok_s_estimate: Option<f64>,
}

impl BackendDef {
    /// Returns the sanitized API view of the backend definition.
    pub fn view(&self) -> BackendView {
        BackendView {
            name: self.name.clone(),
            kinds: self.kinds.clone(),
            base_url: self.base_url.clone(),
            has_api_key: self.api_key.is_some(),
            api_key_env: self.api_key_env.clone(),
            anthropic_version: self.anthropic_version.clone(),
            inject_usage: self.inject_usage,
            scheduling_class: self.scheduling_class,
            exclusive_group_id: self.exclusive_group_id.clone(),
            max_concurrency: self.max_concurrency,
            max_tokens_cap: self.max_tokens_cap,
            tok_s_estimate: self.tok_s_estimate,
        }
    }

    /// `None` when the backend needs no key (e.g. a local vllm instance).
    pub fn resolved_api_key(&self) -> Result<Option<String>, ConfigError> {
        if self.api_key.is_none() && self.api_key_env.is_none() {
            return Ok(None);
        }
        resolve_secret(self.api_key.as_deref(), self.api_key_env.as_deref()).map(Some)
    }
}

/// One inbound model name and the weighted targets it fans out to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDef {
    /// Canonical model name.
    pub name: String,
    /// Extra inbound names that resolve to this same route (shared load
    /// balancer state included). A request naming an alias is billed and
    /// logged under `name`, and — absent a target rename — forwarded
    /// upstream as `name`, so an alias is purely a router-side synonym.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Default pricing for every target; targets may override.
    #[serde(default)]
    pub pricing: Option<PricingCfg>,
    /// Deadline for ordinary traffic on this route, in milliseconds of
    /// predicted queue wait. `None` = wait indefinitely (the pre-scheduling
    /// behavior).
    #[serde(default)]
    pub qos_max_wait_ms: Option<i64>,
    /// Deadline selected by `x-router-qos: background` — batch work is
    /// usually happy to wait far longer for local hardware.
    #[serde(default)]
    pub qos_background_max_wait_ms: Option<i64>,
    /// Action when QoS deadline is missed and route has no fallback.
    #[serde(default)]
    pub on_shed_without_fallback: OnShedWithoutFallback,
    /// Upstream targets for this model route.
    pub targets: Vec<TargetDef>,
}

impl ModelDef {
    /// Every inbound name this route answers to, canonical name first.
    pub fn call_names(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.name.as_str()).chain(self.aliases.iter().map(String::as_str))
    }
}

/// Target definition for a model route.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetDef {
    /// Backend name to route to.
    pub backend: String,
    /// Model name to substitute when forwarding to this backend.
    #[serde(default)]
    pub model: Option<String>,
    /// Selection weight for load balancing.
    #[serde(default = "default_weight")]
    pub weight: u32,
    /// Pricing override for this target.
    #[serde(default)]
    pub pricing: Option<PricingCfg>,
    /// Excluded from normal weighted selection; reachable only when a
    /// request is shed off a local backend for missing its QoS deadline.
    #[serde(default)]
    pub fallback_only: bool,
}

// ---- validation ---------------------------------------------------------

pub(crate) fn validate_group(g: &GroupDef) -> Result<(), String> {
    if g.id.trim().is_empty() {
        return Err("exclusive group id must not be empty".into());
    }
    for (field, value) in [
        ("min_hold_ms", g.min_hold_ms),
        ("max_hold_ms", g.max_hold_ms),
        ("drain_timeout_ms", g.drain_timeout_ms),
    ] {
        if value <= 0 {
            return Err(format!("exclusive group {:?}: {field} must be > 0", g.id));
        }
    }
    // The ordering is what makes the timings mean anything: a max hold
    // below the min hold could never fire, and a drain timeout below the
    // max hold would fire the watchdog on healthy switches.
    if g.min_hold_ms > g.max_hold_ms {
        return Err(format!(
            "exclusive group {:?}: min_hold_ms must be <= max_hold_ms",
            g.id
        ));
    }
    if g.max_hold_ms > g.drain_timeout_ms {
        return Err(format!(
            "exclusive group {:?}: max_hold_ms must be <= drain_timeout_ms",
            g.id
        ));
    }
    Ok(())
}

/// Shared by the config-file loader and the admin API so both reject the
/// same things; the message is surfaced verbatim in either context.
/// `known_groups` is the exclusive-group id list a member backend may name.
pub(crate) fn validate_backend(b: &BackendDef, known_groups: &[String]) -> Result<(), String> {
    if b.name.trim().is_empty() {
        return Err("backend name must not be empty".into());
    }
    if b.kinds.is_empty() {
        return Err(format!(
            "backend {:?}: kinds must list at least one API kind",
            b.name
        ));
    }
    let mut seen = Vec::new();
    for k in &b.kinds {
        if seen.contains(&k) {
            return Err(format!("backend {:?}: duplicate kind entry", b.name));
        }
        seen.push(k);
    }
    Url::parse(&b.base_url).map_err(|e| format!("backend {:?}: invalid base_url: {e}", b.name))?;
    if b.api_key.is_some() && b.api_key_env.is_some() {
        return Err(format!(
            "backend {:?}: set at most one of api_key / api_key_env, not both",
            b.name
        ));
    }

    for (field, value) in [
        ("max_concurrency", b.max_concurrency.map(|v| v as i64)),
        ("max_tokens_cap", b.max_tokens_cap),
    ] {
        if let Some(v) = value
            && v <= 0
        {
            return Err(format!("backend {:?}: {field} must be > 0", b.name));
        }
    }
    if let Some(t) = b.tok_s_estimate
        && (!t.is_finite() || t <= 0.0)
    {
        return Err(format!(
            "backend {:?}: tok_s_estimate must be a positive number",
            b.name
        ));
    }

    match b.scheduling_class {
        SchedulingClass::Grouped => {
            if b.exclusive_group_id.is_none() {
                return Err(format!(
                    "backend {:?}: scheduling_class \"grouped\" requires an exclusive group",
                    b.name
                ));
            }
            // Without a cap, generation length — and so the time the group
            // needs to drain before the lease can move — is unbounded, and
            // every wait estimate built on it is meaningless.
            if b.max_tokens_cap.is_none() {
                return Err(format!(
                    "backend {:?}: scheduling_class \"grouped\" requires max_tokens_cap \
                     (it bounds drain time, and with it every other member's queue wait)",
                    b.name
                ));
            }
        }
        SchedulingClass::Unrestricted if b.exclusive_group_id.is_some() => {
            return Err(format!(
                "backend {:?}: exclusive group needs scheduling_class \"grouped\" or \"exempt\"",
                b.name
            ));
        }
        _ => {}
    }
    if let Some(group) = &b.exclusive_group_id
        && !known_groups.iter().any(|g| g == group)
    {
        return Err(format!(
            "backend {:?}: unknown exclusive group {group:?}",
            b.name
        ));
    }
    Ok(())
}

pub(crate) fn validate_model(m: &ModelDef, known_backends: &[String]) -> Result<(), String> {
    if m.name.trim().is_empty() {
        return Err("model name must not be empty".into());
    }
    let mut seen: Vec<&str> = vec![m.name.as_str()];
    for a in &m.aliases {
        if a.trim().is_empty() {
            return Err(format!("model {:?}: alias must not be empty", m.name));
        }
        if a == &m.name {
            return Err(format!(
                "model {:?}: alias {a:?} duplicates the model name",
                m.name
            ));
        }
        if seen.contains(&a.as_str()) {
            return Err(format!("model {:?}: duplicate alias {a:?}", m.name));
        }
        seen.push(a);
    }
    if m.targets.is_empty() {
        return Err(format!(
            "model {:?}: at least one target is required",
            m.name
        ));
    }
    if let Some(p) = &m.pricing {
        validate_pricing(p, &m.name)?;
    }
    for t in &m.targets {
        if !known_backends.iter().any(|b| b == &t.backend) {
            return Err(format!(
                "model {:?}: target references unknown backend {:?}",
                m.name, t.backend
            ));
        }
        if t.weight == 0 {
            return Err(format!(
                "model {:?}: target {:?} weight must be > 0",
                m.name, t.backend
            ));
        }
        if let Some(p) = &t.pricing {
            validate_pricing(p, &m.name)?;
        }
    }

    for (field, value) in [
        ("qos.max_wait_ms", m.qos_max_wait_ms),
        ("qos.background_max_wait_ms", m.qos_background_max_wait_ms),
    ] {
        if let Some(v) = value
            && v <= 0
        {
            return Err(format!("model {:?}: {field} must be > 0", m.name));
        }
    }

    if m.targets.iter().all(|t| t.fallback_only) {
        return Err(format!(
            "model {:?}: every target is fallback_only, so nothing would serve it normally",
            m.name
        ));
    }
    // Not fatal: the route works, the fallback is simply dead weight until
    // a deadline exists to shed against. Worth saying out loud, though —
    // this is exactly the misconfiguration that looks like "my fallback
    // never fires".
    if m.targets.iter().any(|t| t.fallback_only)
        && m.qos_max_wait_ms.is_none()
        && m.qos_background_max_wait_ms.is_none()
    {
        tracing::warn!(
            model = %m.name,
            "model has a fallback_only target but no QoS deadline — nothing can shed to it"
        );
    }
    Ok(())
}

/// Rejects a definition whose name or any of its aliases is already
/// answered by one of `others` — one inbound name must map to exactly one
/// route. Callers choose what `others` contains: the config loader passes
/// the models accepted so far (so a duplicate name is caught too), the
/// admin API passes every *other* model in the catalog.
pub(crate) fn validate_no_name_conflict<'a>(
    def: &ModelDef,
    others: impl IntoIterator<Item = &'a ModelDef>,
) -> Result<(), String> {
    for other in others {
        for taken in other.call_names() {
            if let Some(dup) = def.call_names().find(|n| *n == taken) {
                if dup == def.name && dup == other.name {
                    return Err(format!("duplicate model name {dup:?}"));
                }
                return Err(format!(
                    "model {:?}: name/alias {dup:?} is already used by model {:?}",
                    def.name, other.name
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_pricing(p: &PricingCfg, model: &str) -> Result<(), String> {
    for (field, value) in [
        ("input_per_mtok", Some(p.input_per_mtok)),
        ("output_per_mtok", Some(p.output_per_mtok)),
        ("cached_input_per_mtok", p.cached_input_per_mtok),
        ("cache_write_per_mtok", p.cache_write_per_mtok),
    ] {
        if let Some(v) = value
            && (!v.is_finite() || v < 0.0)
        {
            return Err(format!(
                "model {model:?}: pricing.{field} must be a non-negative number"
            ));
        }
    }
    Ok(())
}

// ---- persistence --------------------------------------------------------

/// The whole stored catalog, as one snapshot. Groups come first because
/// backends reference them and models reference backends.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    /// Stored exclusive groups.
    pub groups: Vec<GroupDef>,
    /// Stored backends.
    pub backends: Vec<BackendDef>,
    /// Stored models.
    pub models: Vec<ModelDef>,
}

#[cfg(test)]
pub(crate) fn load_backends(db: &Db) -> Result<Vec<BackendDef>> {
    db.with_read(read_backends)
}

#[cfg(test)]
pub(crate) fn load_models(db: &Db) -> Result<Vec<ModelDef>> {
    db.with_read(read_models)
}

#[cfg(test)]
pub(crate) fn load_groups(db: &Db) -> Result<Vec<GroupDef>> {
    db.with_read(read_groups)
}

pub(crate) fn load_all(db: &Db) -> Result<Catalog> {
    db.with_read(|conn| {
        Ok(Catalog {
            groups: read_groups(conn)?,
            backends: read_backends(conn)?,
            models: read_models(conn)?,
        })
    })
}

/// Inserts an exclusive group; `false` means the id is already taken.
///
/// # Errors
///
/// Returns an error if the database write fails.
pub fn insert_group(db: &Db, def: &GroupDef) -> Result<bool> {
    db.with_write(|conn| {
        if group_exists(conn, &def.id)? {
            return Ok(false);
        }
        write_group(conn, def)?;
        Ok(true)
    })
}

/// Overwrites an existing group's timings in place; `false` means it is gone.
pub(crate) fn update_group(db: &Db, def: &GroupDef) -> Result<bool> {
    db.with_write(|conn| {
        if !group_exists(conn, &def.id)? {
            return Ok(false);
        }
        conn.execute(
            "UPDATE exclusive_groups
             SET min_hold_ms = ?2, max_hold_ms = ?3, drain_timeout_ms = ?4
             WHERE id = ?1",
            rusqlite::params![
                def.id,
                def.min_hold_ms,
                def.max_hold_ms,
                def.drain_timeout_ms
            ],
        )?;
        Ok(true)
    })
}

pub(crate) enum DeleteGroup {
    Deleted,
    Missing,
    /// Still named by these backends.
    InUse(Vec<String>),
}

pub(crate) fn delete_group(db: &Db, id: &str) -> Result<DeleteGroup> {
    db.with_write(|conn| {
        if !group_exists(conn, id)? {
            return Ok(DeleteGroup::Missing);
        }
        let mut stmt =
            conn.prepare("SELECT name FROM backends WHERE exclusive_group_id = ?1 ORDER BY name")?;
        let users: Vec<String> = stmt
            .query_map([id], |row| row.get(0))?
            .collect::<std::result::Result<_, _>>()?;
        if !users.is_empty() {
            return Ok(DeleteGroup::InUse(users));
        }
        drop(stmt);
        conn.execute("DELETE FROM exclusive_groups WHERE id = ?1", [id])?;
        Ok(DeleteGroup::Deleted)
    })
}

/// Inserts a backend; `false` means the name is already taken.
///
/// # Errors
///
/// Returns an error if the database write fails.
pub fn insert_backend(db: &Db, def: &BackendDef) -> Result<bool> {
    db.with_write(|conn| {
        if backend_exists(conn, &def.name)? {
            return Ok(false);
        }
        write_backend(conn, def)?;
        Ok(true)
    })
}

/// Overwrites an existing backend in place; `false` means it is gone.
pub(crate) fn update_backend(db: &Db, def: &BackendDef) -> Result<bool> {
    db.with_write(|conn| {
        if !backend_exists(conn, &def.name)? {
            return Ok(false);
        }
        conn.execute("DELETE FROM backends WHERE name = ?1", [&def.name])?;
        write_backend(conn, def)?;
        Ok(true)
    })
}

pub(crate) enum DeleteBackend {
    Deleted,
    Missing,
    /// Still referenced by these model routes.
    InUse(Vec<String>),
}

pub(crate) fn delete_backend(db: &Db, name: &str) -> Result<DeleteBackend> {
    db.with_write(|conn| {
        if !backend_exists(conn, name)? {
            return Ok(DeleteBackend::Missing);
        }
        let mut stmt = conn.prepare(
            "SELECT DISTINCT model FROM model_targets WHERE backend = ?1 ORDER BY model",
        )?;
        let users: Vec<String> = stmt
            .query_map([name], |row| row.get(0))?
            .collect::<std::result::Result<_, _>>()?;
        if !users.is_empty() {
            return Ok(DeleteBackend::InUse(users));
        }
        drop(stmt);
        conn.execute("DELETE FROM backends WHERE name = ?1", [name])?;
        Ok(DeleteBackend::Deleted)
    })
}

/// Creates a model route; `false` means the name is already taken.
///
/// # Errors
///
/// Returns an error if the database write fails.
pub fn insert_model(db: &Db, def: &ModelDef) -> Result<bool> {
    db.with_write(|conn| {
        if model_exists(conn, &def.name)? {
            return Ok(false);
        }
        write_model(conn, def)?;
        Ok(true)
    })
}

/// Replaces an existing model route wholesale (pricing + target list);
/// `false` means it is gone.
pub(crate) fn replace_model(db: &Db, def: &ModelDef) -> Result<bool> {
    db.with_write(|conn| {
        if !model_exists(conn, &def.name)? {
            return Ok(false);
        }
        delete_model_rows(conn, &def.name)?;
        write_model(conn, def)?;
        Ok(true)
    })
}

pub(crate) fn delete_model(db: &Db, name: &str) -> Result<bool> {
    db.with_write(|conn| {
        if !model_exists(conn, name)? {
            return Ok(false);
        }
        delete_model_rows(conn, name)?;
        Ok(true)
    })
}

fn group_exists(conn: &Connection, id: &str) -> Result<bool> {
    let n: i64 = conn.query_row(
        "SELECT COUNT(*) FROM exclusive_groups WHERE id = ?1",
        [id],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}

fn backend_exists(conn: &Connection, name: &str) -> Result<bool> {
    let n: i64 = conn.query_row(
        "SELECT COUNT(*) FROM backends WHERE name = ?1",
        [name],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}

fn model_exists(conn: &Connection, name: &str) -> Result<bool> {
    let n: i64 = conn.query_row("SELECT COUNT(*) FROM models WHERE name = ?1", [name], |r| {
        r.get(0)
    })?;
    Ok(n > 0)
}

fn delete_model_rows(conn: &Connection, name: &str) -> Result<()> {
    conn.execute("DELETE FROM model_targets WHERE model = ?1", [name])?;
    conn.execute("DELETE FROM models WHERE name = ?1", [name])?;
    Ok(())
}

fn write_group(conn: &Connection, def: &GroupDef) -> Result<()> {
    conn.execute(
        "INSERT INTO exclusive_groups (id, min_hold_ms, max_hold_ms, drain_timeout_ms)
         VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![
            def.id,
            def.min_hold_ms,
            def.max_hold_ms,
            def.drain_timeout_ms
        ],
    )?;
    Ok(())
}

fn write_backend(conn: &Connection, def: &BackendDef) -> Result<()> {
    conn.execute(
        "INSERT INTO backends (name, kinds, base_url, api_key, api_key_env,
                               anthropic_version, inject_usage, scheduling_class,
                               exclusive_group_id, max_concurrency, max_tokens_cap,
                               tok_s_estimate)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        rusqlite::params![
            def.name,
            serde_json::to_string(&def.kinds).context("serializing backend kinds")?,
            def.base_url,
            def.api_key,
            def.api_key_env,
            def.anthropic_version,
            def.inject_usage,
            def.scheduling_class.as_str(),
            def.exclusive_group_id,
            def.max_concurrency,
            def.max_tokens_cap,
            def.tok_s_estimate,
        ],
    )?;
    Ok(())
}

fn write_model(conn: &Connection, def: &ModelDef) -> Result<()> {
    conn.execute(
        "INSERT INTO models (name, aliases, pricing, qos_max_wait_ms,
                             qos_background_max_wait_ms, on_shed_without_fallback)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            def.name,
            aliases_to_sql(&def.aliases)?,
            pricing_to_sql(&def.pricing)?,
            def.qos_max_wait_ms,
            def.qos_background_max_wait_ms,
            def.on_shed_without_fallback.as_str(),
        ],
    )?;
    for (i, t) in def.targets.iter().enumerate() {
        conn.execute(
            "INSERT INTO model_targets (model, position, backend, rename, weight, pricing,
                                        fallback_only)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                def.name,
                i as i64,
                t.backend,
                t.model,
                t.weight,
                pricing_to_sql(&t.pricing)?,
                t.fallback_only,
            ],
        )?;
    }
    Ok(())
}

fn read_groups(conn: &Connection) -> Result<Vec<GroupDef>> {
    let mut stmt = conn.prepare(
        "SELECT id, min_hold_ms, max_hold_ms, drain_timeout_ms
         FROM exclusive_groups ORDER BY id",
    )?;
    let rows = stmt
        .query_map([], |row| {
            Ok(GroupDef {
                id: row.get(0)?,
                min_hold_ms: row.get(1)?,
                max_hold_ms: row.get(2)?,
                drain_timeout_ms: row.get(3)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn read_backends(conn: &Connection) -> Result<Vec<BackendDef>> {
    let mut stmt = conn.prepare(
        "SELECT name, kinds, base_url, api_key, api_key_env, anthropic_version, inject_usage,
                scheduling_class, exclusive_group_id, max_concurrency, max_tokens_cap,
                tok_s_estimate
         FROM backends ORDER BY name",
    )?;
    let rows = stmt
        .query_map([], |row| {
            let kinds: String = row.get(1)?;
            let kinds: Vec<BackendKind> = serde_json::from_str(&kinds).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Text,
                    format!("invalid backend kinds JSON: {e}").into(),
                )
            })?;
            let class: String = row.get(7)?;
            let scheduling_class = SchedulingClass::parse(&class).ok_or_else(|| {
                rusqlite::Error::FromSqlConversionFailure(
                    7,
                    rusqlite::types::Type::Text,
                    format!("invalid scheduling class {class:?}").into(),
                )
            })?;
            Ok(BackendDef {
                name: row.get(0)?,
                kinds,
                base_url: row.get(2)?,
                api_key: row.get(3)?,
                api_key_env: row.get(4)?,
                anthropic_version: row.get(5)?,
                inject_usage: row.get(6)?,
                scheduling_class,
                exclusive_group_id: row.get(8)?,
                max_concurrency: row.get(9)?,
                max_tokens_cap: row.get(10)?,
                tok_s_estimate: row.get(11)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn read_models(conn: &Connection) -> Result<Vec<ModelDef>> {
    let mut stmt = conn.prepare(
        "SELECT name, aliases, pricing, qos_max_wait_ms, qos_background_max_wait_ms,
                on_shed_without_fallback
         FROM models ORDER BY name",
    )?;
    let mut models: Vec<ModelDef> = stmt
        .query_map([], |row| {
            let on_shed: String = row.get(5)?;
            let on_shed_without_fallback =
                OnShedWithoutFallback::parse(&on_shed).ok_or_else(|| {
                    rusqlite::Error::FromSqlConversionFailure(
                        5,
                        rusqlite::types::Type::Text,
                        format!("invalid on_shed_without_fallback {on_shed:?}").into(),
                    )
                })?;
            Ok(ModelDef {
                name: row.get(0)?,
                aliases: aliases_from_sql(row.get::<_, Option<String>>(1)?, 1)?,
                pricing: pricing_from_sql(row.get::<_, Option<String>>(2)?, 2)?,
                qos_max_wait_ms: row.get(3)?,
                qos_background_max_wait_ms: row.get(4)?,
                on_shed_without_fallback,
                targets: Vec::new(),
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    drop(stmt);

    let mut stmt = conn.prepare(
        "SELECT model, backend, rename, weight, pricing, fallback_only
         FROM model_targets ORDER BY model, position, id",
    )?;
    let targets = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                TargetDef {
                    backend: row.get(1)?,
                    model: row.get(2)?,
                    weight: row.get(3)?,
                    pricing: pricing_from_sql(row.get::<_, Option<String>>(4)?, 4)?,
                    fallback_only: row.get(5)?,
                },
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for (model, target) in targets {
        if let Some(m) = models.iter_mut().find(|m| m.name == model) {
            m.targets.push(target);
        }
    }
    Ok(models)
}

/// Stored as a JSON array; returns None for the common no-aliases case.
fn aliases_to_sql(aliases: &[String]) -> Result<Option<String>> {
    if aliases.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        serde_json::to_string(aliases).context("serializing model aliases")?,
    ))
}

fn aliases_from_sql(raw: Option<String>, idx: usize) -> rusqlite::Result<Vec<String>> {
    match raw {
        None => Ok(Vec::new()),
        Some(json) => serde_json::from_str(&json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(
                idx,
                rusqlite::types::Type::Text,
                format!("invalid model aliases JSON: {e}").into(),
            )
        }),
    }
}

/// Serializes pricing to JSON, returning None when pricing is unset.
fn pricing_to_sql(p: &Option<PricingCfg>) -> Result<Option<String>> {
    p.as_ref()
        .map(|p| serde_json::to_string(p).context("serializing pricing"))
        .transpose()
}

/// Deserializes pricing from JSON, returning None when raw is null.
fn pricing_from_sql(raw: Option<String>, idx: usize) -> rusqlite::Result<Option<PricingCfg>> {
    raw.map(|json| {
        serde_json::from_str::<PricingCfg>(&json).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(
                idx,
                rusqlite::types::Type::Text,
                format!("invalid pricing JSON: {e}").into(),
            )
        })
    })
    .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (tempfile::TempDir, Db) {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("t.db")).unwrap();
        (dir, db)
    }

    fn backend(name: &str) -> BackendDef {
        BackendDef {
            name: name.into(),
            kinds: vec![BackendKind::Openai],
            base_url: "https://api.openai.com".into(),
            api_key: Some("sk-secret".into()),
            api_key_env: None,
            anthropic_version: None,
            inject_usage: true,
            scheduling_class: SchedulingClass::Unrestricted,
            exclusive_group_id: None,
            max_concurrency: None,
            max_tokens_cap: None,
            tok_s_estimate: None,
        }
    }

    fn group(id: &str) -> GroupDef {
        GroupDef {
            id: id.into(),
            min_hold_ms: 1_000,
            max_hold_ms: 30_000,
            drain_timeout_ms: 45_000,
        }
    }

    /// A fully-scheduled backend: on `group`, capped, with a throughput seed.
    fn grouped_backend(name: &str, group: &str) -> BackendDef {
        BackendDef {
            scheduling_class: SchedulingClass::Grouped,
            exclusive_group_id: Some(group.into()),
            max_concurrency: Some(4),
            max_tokens_cap: Some(1024),
            tok_s_estimate: Some(90.0),
            ..backend(name)
        }
    }

    fn model(name: &str, backend: &str) -> ModelDef {
        ModelDef {
            name: name.into(),
            aliases: Vec::new(),
            pricing: Some(PricingCfg {
                input_per_mtok: 1.0,
                output_per_mtok: 2.0,
                cached_input_per_mtok: Some(0.5),
                cache_write_per_mtok: None,
            }),
            qos_max_wait_ms: None,
            qos_background_max_wait_ms: None,
            on_shed_without_fallback: OnShedWithoutFallback::Wait,
            targets: vec![TargetDef {
                backend: backend.into(),
                model: Some("upstream-name".into()),
                weight: 3,
                pricing: None,
                fallback_only: false,
            }],
        }
    }

    #[test]
    fn backends_round_trip() {
        let (_dir, db) = temp_db();
        assert!(insert_backend(&db, &backend("openai")).unwrap());
        // Duplicate names are rejected without clobbering the original.
        assert!(!insert_backend(&db, &backend("openai")).unwrap());

        let got = load_backends(&db).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], backend("openai"));

        let mut updated = backend("openai");
        updated.api_key = None;
        updated.api_key_env = Some("SOME_VAR".into());
        updated.kinds = vec![BackendKind::Openai, BackendKind::Anthropic];
        updated.anthropic_version = Some("2023-06-01".into());
        updated.inject_usage = false;
        assert!(update_backend(&db, &updated).unwrap());
        assert_eq!(load_backends(&db).unwrap()[0], updated);

        let mut missing = backend("nope");
        missing.name = "nope".into();
        assert!(!update_backend(&db, &missing).unwrap());
    }

    #[test]
    fn view_masks_the_secret() {
        let v = backend("openai").view();
        assert!(v.has_api_key);
        assert_eq!(v.api_key_env, None);
        let json = serde_json::to_string(&v).unwrap();
        assert!(!json.contains("sk-secret"), "{json}");
    }

    #[test]
    fn models_round_trip_with_target_order() {
        let (_dir, db) = temp_db();
        insert_backend(&db, &backend("a")).unwrap();
        insert_backend(&db, &backend("b")).unwrap();
        let mut m = model("gpt-4o", "a");
        m.targets.push(TargetDef {
            backend: "b".into(),
            model: None,
            weight: 1,
            pricing: Some(PricingCfg {
                input_per_mtok: 0.0,
                output_per_mtok: 0.0,
                cached_input_per_mtok: None,
                cache_write_per_mtok: None,
            }),
            fallback_only: false,
        });
        assert!(insert_model(&db, &m).unwrap());
        assert!(!insert_model(&db, &m).unwrap());

        let got = load_models(&db).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], m);

        // Replace swaps the whole target list, keeping no stale rows.
        let mut replaced = m.clone();
        replaced.targets.truncate(1);
        replaced.pricing = None;
        assert!(replace_model(&db, &replaced).unwrap());
        assert_eq!(load_models(&db).unwrap()[0], replaced);

        assert!(delete_model(&db, "gpt-4o").unwrap());
        assert!(!delete_model(&db, "gpt-4o").unwrap());
        assert!(load_models(&db).unwrap().is_empty());
        let orphans: i64 = db
            .with_read(|c| Ok(c.query_row("SELECT COUNT(*) FROM model_targets", [], |r| r.get(0))?))
            .unwrap();
        assert_eq!(orphans, 0);
    }

    #[test]
    fn backend_delete_is_blocked_while_referenced() {
        let (_dir, db) = temp_db();
        insert_backend(&db, &backend("a")).unwrap();
        insert_model(&db, &model("gpt-4o", "a")).unwrap();
        match delete_backend(&db, "a").unwrap() {
            DeleteBackend::InUse(users) => assert_eq!(users, vec!["gpt-4o".to_string()]),
            _ => panic!("expected InUse"),
        }
        delete_model(&db, "gpt-4o").unwrap();
        assert!(matches!(
            delete_backend(&db, "a").unwrap(),
            DeleteBackend::Deleted
        ));
        assert!(matches!(
            delete_backend(&db, "a").unwrap(),
            DeleteBackend::Missing
        ));
    }

    #[test]
    fn groups_round_trip_and_block_delete_while_referenced() {
        let (_dir, db) = temp_db();
        assert!(insert_group(&db, &group("dgx")).unwrap());
        assert!(!insert_group(&db, &group("dgx")).unwrap(), "duplicate id");
        assert_eq!(load_groups(&db).unwrap(), vec![group("dgx")]);

        let retuned = GroupDef {
            min_hold_ms: 2_000,
            ..group("dgx")
        };
        assert!(update_group(&db, &retuned).unwrap());
        assert_eq!(load_groups(&db).unwrap(), vec![retuned]);
        assert!(!update_group(&db, &group("nope")).unwrap());

        insert_backend(&db, &grouped_backend("moe", "dgx")).unwrap();
        match delete_group(&db, "dgx").unwrap() {
            DeleteGroup::InUse(users) => assert_eq!(users, vec!["moe".to_string()]),
            _ => panic!("expected InUse"),
        }
        delete_backend(&db, "moe").unwrap();
        assert!(matches!(
            delete_group(&db, "dgx").unwrap(),
            DeleteGroup::Deleted
        ));
        assert!(matches!(
            delete_group(&db, "dgx").unwrap(),
            DeleteGroup::Missing
        ));
    }

    #[test]
    fn scheduling_fields_round_trip() {
        let (_dir, db) = temp_db();
        insert_group(&db, &group("dgx")).unwrap();
        let b = grouped_backend("moe", "dgx");
        insert_backend(&db, &b).unwrap();
        assert_eq!(load_backends(&db).unwrap()[0], b);

        // The exempt class stores its group but takes no lease.
        let mut embed = backend("embed");
        embed.scheduling_class = SchedulingClass::Exempt;
        embed.exclusive_group_id = Some("dgx".into());
        insert_backend(&db, &embed).unwrap();
        let got = load_backends(&db).unwrap();
        assert_eq!(got.iter().find(|x| x.name == "embed").unwrap(), &embed);

        let mut m = model("qwen", "moe");
        m.qos_max_wait_ms = Some(2_000);
        m.qos_background_max_wait_ms = Some(60_000);
        m.on_shed_without_fallback = OnShedWithoutFallback::Error;
        m.targets.push(TargetDef {
            backend: "embed".into(),
            model: None,
            weight: 1,
            pricing: None,
            fallback_only: true,
        });
        insert_model(&db, &m).unwrap();
        assert_eq!(load_models(&db).unwrap()[0], m);
    }

    #[test]
    fn aliases_round_trip() {
        let (_dir, db) = temp_db();
        insert_backend(&db, &backend("a")).unwrap();
        let mut m = model("gpt-4o", "a");
        m.aliases = vec!["gpt-4o-latest".into(), "gpt4".into()];
        assert!(insert_model(&db, &m).unwrap());
        let got = load_models(&db).unwrap();
        assert_eq!(got[0].aliases, m.aliases);
        assert_eq!(
            got[0].call_names().collect::<Vec<_>>(),
            vec!["gpt-4o", "gpt-4o-latest", "gpt4"]
        );

        // Replacing clears them; no-aliases rows read back as an empty list.
        let mut replaced = m.clone();
        replaced.aliases.clear();
        assert!(replace_model(&db, &replaced).unwrap());
        assert!(load_models(&db).unwrap()[0].aliases.is_empty());
    }

    #[test]
    fn validation_rejects_bad_aliases() {
        let known = vec!["a".to_string()];
        let with = |aliases: &[&str]| {
            let mut m = model("m", "a");
            m.aliases = aliases.iter().map(|s| s.to_string()).collect();
            m
        };
        assert!(validate_model(&with(&["m2", "m3"]), &known).is_ok());
        assert!(validate_model(&with(&[""]), &known).is_err());
        assert!(validate_model(&with(&["  "]), &known).is_err());
        assert!(validate_model(&with(&["dup", "dup"]), &known).is_err());
        assert!(
            validate_model(&with(&["m"]), &known).is_err(),
            "alias == name"
        );
    }

    #[test]
    fn name_conflicts_span_names_and_aliases() {
        let mut existing = model("gpt-4o", "a");
        existing.aliases = vec!["gpt4".into()];
        let others = [existing];

        let ok = model("claude", "a");
        assert!(validate_no_name_conflict(&ok, &others).is_ok());
        // Same name as an existing model.
        let err = validate_no_name_conflict(&model("gpt-4o", "a"), &others).unwrap_err();
        assert!(err.contains("duplicate model name"), "{err}");
        // A new name that shadows an existing alias, and vice versa.
        assert!(validate_no_name_conflict(&model("gpt4", "a"), &others).is_err());
        let mut clashing = model("claude", "a");
        clashing.aliases = vec!["gpt4".into()];
        assert!(validate_no_name_conflict(&clashing, &others).is_err());
        let mut clashing = model("claude", "a");
        clashing.aliases = vec!["gpt-4o".into()];
        assert!(validate_no_name_conflict(&clashing, &others).is_err());
        assert!(validate_no_name_conflict(&ok, []).is_ok());
    }

    #[test]
    fn validation_rejects_bad_definitions() {
        let no_groups: &[String] = &[];
        assert!(validate_backend(&backend("ok"), no_groups).is_ok());
        let mut b = backend("ok");
        b.kinds.clear();
        assert!(validate_backend(&b, no_groups).is_err());
        let mut b = backend("ok");
        b.kinds = vec![BackendKind::Openai, BackendKind::Openai];
        assert!(validate_backend(&b, no_groups).is_err());
        let mut b = backend("ok");
        b.base_url = "not a url".into();
        assert!(validate_backend(&b, no_groups).is_err());
        let mut b = backend("ok");
        b.api_key_env = Some("X".into());
        assert!(validate_backend(&b, no_groups).is_err());

        let known = vec!["a".to_string()];
        assert!(validate_model(&model("m", "a"), &known).is_ok());
        assert!(validate_model(&model("m", "missing"), &known).is_err());
        let mut m = model("m", "a");
        m.targets[0].weight = 0;
        assert!(validate_model(&m, &known).is_err());
        let mut m = model("m", "a");
        m.targets.clear();
        assert!(validate_model(&m, &known).is_err());
        let mut m = model("m", "a");
        m.pricing.as_mut().unwrap().input_per_mtok = -1.0;
        assert!(validate_model(&m, &known).is_err());
    }

    #[test]
    fn validation_rejects_bad_groups() {
        assert!(validate_group(&group("dgx")).is_ok());
        assert!(
            validate_group(&GroupDef {
                id: "  ".into(),
                ..group("dgx")
            })
            .is_err()
        );
        for bad in [
            GroupDef {
                min_hold_ms: 0,
                ..group("dgx")
            },
            GroupDef {
                max_hold_ms: -1,
                ..group("dgx")
            },
            GroupDef {
                drain_timeout_ms: 0,
                ..group("dgx")
            },
            // min > max: the ceiling could never fire.
            GroupDef {
                min_hold_ms: 40_000,
                ..group("dgx")
            },
            // max > drain: the watchdog would trip on healthy switches.
            GroupDef {
                drain_timeout_ms: 10_000,
                ..group("dgx")
            },
        ] {
            assert!(validate_group(&bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn validation_rejects_bad_scheduling_setups() {
        let groups = vec!["dgx".to_string()];
        assert!(validate_backend(&grouped_backend("moe", "dgx"), &groups).is_ok());

        // A grouped backend must name a group, and one that exists.
        let mut b = grouped_backend("moe", "dgx");
        b.exclusive_group_id = None;
        assert!(validate_backend(&b, &groups).is_err());
        assert!(validate_backend(&grouped_backend("moe", "other"), &groups).is_err());

        // …and must be token-capped: an unbounded generation makes drain
        // time, and so every peer's wait estimate, unbounded.
        let mut b = grouped_backend("moe", "dgx");
        b.max_tokens_cap = None;
        let err = validate_backend(&b, &groups).unwrap_err();
        assert!(err.contains("max_tokens_cap"), "{err}");

        // A group on an unrestricted backend is a contradiction; exempt is
        // the way to say "on the device but not leased".
        let mut b = backend("embed");
        b.exclusive_group_id = Some("dgx".into());
        assert!(validate_backend(&b, &groups).is_err());
        b.scheduling_class = SchedulingClass::Exempt;
        assert!(validate_backend(&b, &groups).is_ok());

        for bad in [
            BackendDef {
                max_concurrency: Some(0),
                ..backend("ok")
            },
            BackendDef {
                max_tokens_cap: Some(-5),
                ..backend("ok")
            },
            BackendDef {
                tok_s_estimate: Some(0.0),
                ..backend("ok")
            },
            BackendDef {
                tok_s_estimate: Some(f64::NAN),
                ..backend("ok")
            },
        ] {
            assert!(validate_backend(&bad, &groups).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn validation_rejects_bad_qos_and_fallback_setups() {
        let known = vec!["a".to_string(), "remote".to_string()];
        let fallback = |only: bool| TargetDef {
            backend: "remote".into(),
            model: None,
            weight: 1,
            pricing: None,
            fallback_only: only,
        };

        // A fallback alongside a normal target is the intended shape.
        let mut m = model("m", "a");
        m.qos_max_wait_ms = Some(2_000);
        m.targets.push(fallback(true));
        assert!(validate_model(&m, &known).is_ok());

        // Nothing left to serve the route normally.
        let mut m = model("m", "a");
        m.targets[0].fallback_only = true;
        let err = validate_model(&m, &known).unwrap_err();
        assert!(err.contains("fallback_only"), "{err}");

        // A fallback with no deadline is only a warning — the route works.
        let mut m = model("m", "a");
        m.targets.push(fallback(true));
        assert!(validate_model(&m, &known).is_ok());

        for bad in [Some(0), Some(-1)] {
            let mut m = model("m", "a");
            m.qos_max_wait_ms = bad;
            assert!(validate_model(&m, &known).is_err());
            let mut m = model("m", "a");
            m.qos_background_max_wait_ms = bad;
            assert!(validate_model(&m, &known).is_err());
        }
    }
}
