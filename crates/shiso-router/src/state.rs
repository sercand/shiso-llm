use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::{Context, Result};
use url::Url;

use crate::active::ActiveRegistry;
use crate::balancer::{Backend, ModelRoute, Pricing, RouteQos, Target};
use crate::catalog::{
    self, BackendDef, Catalog, DEFAULT_ANTHROPIC_VERSION, GroupDef, ModelDef, SchedulingClass,
};
use crate::config::Config;
use crate::limits::UsageTracker;
use crate::logging::LogSink;
use crate::sched::Scheduler;
use crate::tokens::{Db, TokenCache};

/// The compiled catalog the proxy path reads: inbound model name -> route,
/// plus the definitions it was built from (what the admin API serves and
/// edits). Aliases are extra keys pointing at the same `Arc<ModelRoute>`,
/// so an alias shares the route's load-balancer state and its canonical
/// name. Immutable — an admin write rebuilds a fresh `Routing` and swaps it
/// in, so in-flight requests keep using the snapshot they started with.
#[derive(Debug)]
pub struct Routing {
    /// Route mapping keyed by canonical model name and aliases.
    pub routes: HashMap<String, Arc<ModelRoute>>,
    /// Defined exclusive groups.
    pub groups: Vec<GroupDef>,
    /// Defined backends.
    pub backends: Vec<BackendDef>,
    /// Defined model configurations.
    pub models: Vec<ModelDef>,
}

impl Routing {
    /// Resolves secrets (env vars are read here) and compiles the routes.
    /// Fails when a model names a backend that does not exist, a grouped
    /// backend names a group that does not exist, or an `api_key_env` is
    /// unset — the previous snapshot then stays live.
    pub fn build(catalog: Catalog) -> Result<Self> {
        let Catalog {
            groups,
            backends,
            models,
        } = catalog;
        let mut resolved: HashMap<&str, Arc<Backend>> = HashMap::new();
        for b in &backends {
            let api_key = b.resolved_api_key()?;
            let base_url = Url::parse(&b.base_url)
                .with_context(|| format!("backend {:?}: invalid base_url", b.name))?;
            // A dangling group would leave the backend un-schedulable in a
            // way nothing downstream could detect, so fail the whole
            // rebuild here — the previous snapshot keeps serving.
            if let Some(group) = &b.exclusive_group_id
                && !groups.iter().any(|g| &g.id == group)
            {
                anyhow::bail!(
                    "backend {:?}: references unknown exclusive group {group:?}",
                    b.name
                );
            }
            if b.scheduling_class == SchedulingClass::Grouped && b.max_tokens_cap.is_none() {
                anyhow::bail!(
                    "backend {:?}: a grouped backend requires max_tokens_cap",
                    b.name
                );
            }
            resolved.insert(
                b.name.as_str(),
                Arc::new(Backend {
                    name: b.name.clone(),
                    kinds: b.kinds.clone(),
                    base_url,
                    api_key,
                    anthropic_version: b
                        .anthropic_version
                        .clone()
                        .unwrap_or_else(|| DEFAULT_ANTHROPIC_VERSION.to_string()),
                    inject_usage: b.inject_usage,
                    class: b.scheduling_class,
                    group: b.exclusive_group_id.clone(),
                    max_concurrency: b.max_concurrency,
                    max_tokens_cap: b.max_tokens_cap,
                    tok_s_estimate: b.tok_s_estimate,
                }),
            );
        }

        let mut routes = HashMap::new();
        for m in &models {
            let model_pricing = m.pricing.as_ref().map(Pricing::from_cfg);
            let mut targets = Vec::with_capacity(m.targets.len());
            for t in &m.targets {
                let backend = resolved.get(t.backend.as_str()).ok_or_else(|| {
                    anyhow::anyhow!(
                        "model {:?}: target references unknown backend {:?}",
                        m.name,
                        t.backend
                    )
                })?;
                targets.push(Target {
                    backend: Arc::clone(backend),
                    rename: t.model.clone(),
                    weight: t.weight,
                    pricing: t.pricing.as_ref().map(Pricing::from_cfg).or(model_pricing),
                    fallback_only: t.fallback_only,
                });
            }
            if targets.is_empty() {
                anyhow::bail!("model {:?}: at least one target is required", m.name);
            }
            if targets.iter().all(|t| t.fallback_only) {
                anyhow::bail!("model {:?}: every target is fallback_only", m.name);
            }
            let qos = RouteQos {
                max_wait_ms: m.qos_max_wait_ms,
                background_max_wait_ms: m.qos_background_max_wait_ms,
                on_shed_without_fallback: m.on_shed_without_fallback,
            };
            let route = Arc::new(ModelRoute::new(m.name.clone(), targets, qos));
            for name in m.call_names() {
                // The admin API and the config loader both reject conflicts
                // before a write, so this only fires on a hand-edited DB.
                if let Some(prev) = routes.insert(name.to_string(), Arc::clone(&route)) {
                    anyhow::bail!(
                        "inbound model name {name:?} is claimed by both model {:?} and {:?}",
                        prev.name,
                        m.name
                    );
                }
            }
        }

        Ok(Routing {
            routes,
            groups,
            backends,
            models,
        })
    }

    /// Backend names, for validating model targets on the admin path.
    pub fn backend_names(&self) -> Vec<String> {
        self.backends.iter().map(|b| b.name.clone()).collect()
    }

    /// Exclusive group ids, for validating a backend's group on the admin
    /// path — the same role `backend_names` plays for model targets.
    pub fn group_ids(&self) -> Vec<String> {
        self.groups.iter().map(|g| g.id.clone()).collect()
    }

    /// The model a name resolves to, canonical name or alias alike.
    pub fn model_named(&self, name: &str) -> Option<&ModelDef> {
        let canonical = &self.routes.get(name)?.name;
        self.models.iter().find(|m| &m.name == canonical)
    }
}

/// Shared application state wrapped in an `Arc`.
#[derive(Debug, Clone)]
pub struct AppState(Arc<Inner>);

/// Inner shared application state.
#[derive(Debug)]
pub struct Inner {
    /// Reusable outbound HTTP client.
    pub http: reqwest::Client,
    /// Swapped wholesale by `reload_routing` after an admin catalog edit;
    /// readers clone the `Arc` and never hold the lock across an await.
    routing: RwLock<Arc<Routing>>,
    /// In-memory API key token cache.
    pub token_cache: TokenCache,
    /// SQLite token and catalog database.
    pub db: Db,
    /// Asynchronous request log sink.
    pub log_sink: LogSink,
    /// In-memory key usage limit tracker.
    pub usage_tracker: Arc<UsageTracker>,
    /// Admission control. Deliberately *not* part of `routing`: it must
    /// survive the snapshot swap a catalog edit performs, or an unrelated
    /// edit would drop held leases and zero in-flight counters.
    pub scheduler: Arc<Scheduler>,
    /// In-flight requests, for the logs UI's active panel. Outside
    /// `routing` for the same reason the scheduler is: a catalog edit must
    /// not empty it under requests that are still running.
    pub active: Arc<ActiveRegistry>,
    /// Admin API token.
    pub admin_token: String,
    /// Request body size ceiling in bytes.
    pub max_body_bytes: usize,
    /// False when a separate admin listener is configured — the admin
    /// surface then disappears from the main router.
    pub admin_on_main: bool,
}

impl std::ops::Deref for AppState {
    type Target = Inner;
    fn deref(&self) -> &Inner {
        &self.0
    }
}

impl AppState {
    /// Constructs application state from configuration, database, and log sink.
    ///
    /// # Errors
    ///
    /// Returns an error if the catalog fails to seed or load.
    pub fn from_config(cfg: &Config, db: Db, log_sink: LogSink) -> Result<Self> {
        let admin_token = cfg.resolved_admin_token()?;

        let catalog = catalog::load_all(&db)?;
        if catalog.models.is_empty() {
            tracing::warn!("no models configured — add one via POST /admin/models or the admin UI");
        }
        let routing = Routing::build(catalog)?;

        // No total timeout: SSE responses legitimately run for minutes.
        // read_timeout guards against a stalled upstream mid-stream, but it
        // is an inter-read timeout that is also running while we wait for the
        // response head — so it has to cover the worst-case time to first
        // token, which on a big prompt against a slow-prefill backend is
        // minutes, not seconds.
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(600))
            .pool_idle_timeout(Duration::from_secs(90))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .context("failed to build HTTP client")?;

        let entries = db.load_all()?;
        // Warm current-window usage sums from request_logs before serving
        // so limits survive restarts. Startup-only blocking DB work.
        let usage_tracker = log_sink.usage_tracker();
        usage_tracker.warm_all(&db, entries.iter().map(|(_, r)| r))?;
        let token_cache = TokenCache::new(entries);
        let scheduler = Arc::new(Scheduler::from_routing(&routing));

        Ok(AppState(Arc::new(Inner {
            http,
            routing: RwLock::new(Arc::new(routing)),
            token_cache,
            db,
            log_sink,
            usage_tracker,
            scheduler,
            active: ActiveRegistry::new(),
            admin_token,
            max_body_bytes: cfg.max_body_bytes,
            admin_on_main: cfg.admin_listen.is_none(),
        })))
    }

    /// Current routing snapshot. Cheap (one `Arc` clone) — the proxy path
    /// takes one per request and works from it consistently.
    pub fn routing(&self) -> Arc<Routing> {
        Arc::clone(&self.0.routing.read().unwrap())
    }

    /// Rebuilds the routing snapshot from the database after a catalog
    /// edit. Blocking (reads SQLite); call it from `spawn_blocking`. On
    /// failure the live snapshot is left untouched. Load-balancer state is
    /// reset for every route, which only re-phases the round robin.
    pub fn reload_routing(&self) -> Result<()> {
        let routing = Routing::build(catalog::load_all(&self.db)?)?;
        // Reconcile before publishing: a request picking up the new
        // snapshot must find the scheduler already aware of its backends.
        self.0.scheduler.reconcile(&routing);
        *self.0.routing.write().unwrap() = Arc::new(routing);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::TargetDef;
    use crate::config::BackendKind;

    fn backends() -> Vec<BackendDef> {
        vec![BackendDef {
            name: "b".into(),
            kinds: vec![BackendKind::Openai],
            base_url: "http://127.0.0.1:1".into(),
            api_key: None,
            api_key_env: None,
            anthropic_version: None,
            inject_usage: true,
            scheduling_class: SchedulingClass::Unrestricted,
            exclusive_group_id: None,
            max_concurrency: None,
            max_tokens_cap: None,
            tok_s_estimate: None,
        }]
    }

    fn model(name: &str, aliases: &[&str]) -> ModelDef {
        ModelDef {
            name: name.into(),
            aliases: aliases.iter().map(|a| a.to_string()).collect(),
            pricing: None,
            qos_max_wait_ms: None,
            qos_background_max_wait_ms: None,
            on_shed_without_fallback: Default::default(),
            targets: vec![TargetDef {
                backend: "b".into(),
                model: None,
                weight: 1,
                pricing: None,
                fallback_only: false,
            }],
        }
    }

    fn catalog(groups: Vec<GroupDef>, backends: Vec<BackendDef>, models: Vec<ModelDef>) -> Catalog {
        Catalog {
            groups,
            backends,
            models,
        }
    }

    #[test]
    fn aliases_share_one_route_instance() {
        let routing = Routing::build(catalog(
            vec![],
            backends(),
            vec![model("gpt-4o", &["gpt4", "latest"])],
        ))
        .unwrap();
        let canonical = routing.routes.get("gpt-4o").unwrap();
        assert_eq!(routing.routes.len(), 3);
        for alias in ["gpt4", "latest"] {
            let route = routing.routes.get(alias).unwrap();
            // Same allocation: an alias shares the balancer state, and its
            // route reports the canonical name for logging and billing.
            assert!(Arc::ptr_eq(canonical, route));
            assert_eq!(route.name, "gpt-4o");
        }
        assert_eq!(routing.model_named("gpt4").unwrap().name, "gpt-4o");
        assert_eq!(routing.model_named("gpt-4o").unwrap().name, "gpt-4o");
        assert!(routing.model_named("nope").is_none());
    }

    #[test]
    fn colliding_inbound_names_fail_the_build() {
        // Only reachable from a hand-edited database — the API and the
        // config loader both reject this before writing.
        let err = Routing::build(catalog(
            vec![],
            backends(),
            vec![model("gpt-4o", &["shared"]), model("claude", &["shared"])],
        ))
        .expect_err("a duplicated inbound name must not build");
        assert!(err.to_string().contains("shared"), "{err}");
    }

    #[test]
    fn grouped_backend_needs_an_existing_group() {
        let mut b = backends();
        b[0].scheduling_class = SchedulingClass::Grouped;
        b[0].exclusive_group_id = Some("dgx".into());
        b[0].max_tokens_cap = Some(1024);

        // No such group: the whole rebuild fails, leaving the previous
        // snapshot serving.
        let err = Routing::build(catalog(vec![], b.clone(), vec![model("m", &[])]))
            .expect_err("a dangling group must not build");
        assert!(err.to_string().contains("dgx"), "{err}");

        let group = GroupDef {
            id: "dgx".into(),
            min_hold_ms: 1000,
            max_hold_ms: 5000,
            drain_timeout_ms: 10_000,
        };
        let routing = Routing::build(catalog(
            vec![group.clone()],
            b.clone(),
            vec![model("m", &[])],
        ))
        .unwrap();
        assert_eq!(routing.group_ids(), vec!["dgx".to_string()]);
        let target = &routing.routes.get("m").unwrap().targets[0];
        assert_eq!(target.backend.lease_group(), Some("dgx"));

        // A grouped backend without a token cap leaves drain time — and so
        // every wait estimate — unbounded.
        b[0].max_tokens_cap = None;
        assert!(Routing::build(catalog(vec![group], b, vec![model("m", &[])])).is_err());
    }

    #[test]
    fn a_route_of_only_fallback_targets_does_not_build() {
        let mut m = model("m", &[]);
        m.targets[0].fallback_only = true;
        let err = Routing::build(catalog(vec![], backends(), vec![m]))
            .expect_err("a route with nothing to serve it normally must not build");
        assert!(err.to_string().contains("fallback_only"), "{err}");
    }
}
