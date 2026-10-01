//! The in-flight request registry: what the logs UI shows above the
//! finished rows.
//!
//! A `request_logs` row only exists once a request has finalized, so a
//! request that is still queued behind a device lease or still streaming is
//! invisible to SQLite. This module keeps the live set in memory instead.
//!
//! Like [`crate::sched::Scheduler`], the registry lives in
//! `state::Inner` and never in `Routing`: `Routing` is rebuilt wholesale
//! after every catalog write, so entries kept there would vanish whenever
//! an admin edited an unrelated backend. And like `SlotGuard`,
//! [`ActiveGuard`]'s `Drop` is the *only* removal path — it rides in
//! `LogContext` for the life of the response, so end of stream, upstream
//! error and client disconnect all deregister without an explicit call.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::Serialize;

use crate::logging::LogRow;

/// How far along an in-flight request is. The distinction operators
/// actually need is "waiting for the accelerator" versus "waiting for the
/// provider" — a request parked in `Queued` on a grouped backend is held by
/// our own admission control, not by anything upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Routed and admitted-pending: waiting on a group lease or a
    /// concurrency slot.
    Queued,
    /// Dispatched upstream, waiting for the response head.
    Upstream,
    /// Response head relayed; body bytes are flowing to the client.
    Streaming,
}

/// One in-flight request. Mirrors the subset of [`LogRow`] that is known
/// before finalization; token counts and cost are deliberately absent
/// (usage is only parsed as the body streams, and for OpenAI SSE only
/// arrives in the final chunk).
#[derive(Debug, Clone)]
pub struct ActiveEntry {
    pub id: String,
    /// Unix epoch milliseconds UTC, request start — the same clock and the
    /// same instant as the eventual log row's `created_at`.
    pub created_at: i64,
    pub token_id: String,
    pub token_name: String,
    pub model: String,
    pub path: String,
    /// The backend currently being attempted. `None` only in the sliver
    /// between routing and the first attempt.
    pub backend: Option<String>,
    pub upstream_model: Option<String>,
    pub phase: Phase,
    /// Whether the upstream response is an SSE stream. Only known once the
    /// head has been relayed, so false until `Streaming`.
    pub streamed: bool,
    /// The client's `x-service-tier` hint; `None` for ordinary traffic.
    pub service_tier: Option<String>,
}

/// The live set, keyed by request id (the same id the finished log row
/// carries, so a trace or a console dispatch lines up across both).
#[derive(Debug, Default)]
pub struct ActiveRegistry {
    inner: RwLock<HashMap<String, ActiveEntry>>,
}

impl ActiveRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(ActiveRegistry::default())
    }

    /// Enters `row`'s request into the live set and returns the guard whose
    /// drop removes it. Called once the route is resolved, so `row.model`
    /// is already the canonical name an alias resolved to.
    ///
    /// `token_name` is passed in rather than looked up: the caller has the
    /// `KeyRecord` in hand, and the admin console's synthetic key has no
    /// `tokens` row to join against anyway.
    pub fn register(self: &Arc<Self>, row: &LogRow, token_name: &str) -> ActiveGuard {
        let entry = ActiveEntry {
            id: row.id.clone(),
            created_at: row.created_at,
            token_id: row.token_id.clone(),
            token_name: token_name.to_string(),
            model: row.model.clone(),
            path: row.path.clone(),
            backend: None,
            upstream_model: None,
            phase: Phase::Queued,
            streamed: false,
            service_tier: row.service_tier.clone(),
        };
        let id = entry.id.clone();
        self.inner.write().unwrap().insert(id.clone(), entry);
        ActiveGuard {
            registry: Arc::clone(self),
            id,
        }
    }

    /// Every in-flight request, longest-running first: the request an
    /// operator opens this panel to look at is the one that has been stuck
    /// the longest.
    pub fn snapshot(&self) -> Vec<ActiveEntry> {
        let mut out: Vec<ActiveEntry> = self.inner.read().unwrap().values().cloned().collect();
        out.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        out
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn edit(&self, id: &str, f: impl FnOnce(&mut ActiveEntry)) {
        if let Some(entry) = self.inner.write().unwrap().get_mut(id) {
            f(entry);
        }
    }
}

/// Removal-on-drop handle for one registry entry. Moved into `LogContext`
/// when a response is relayed, so the entry lives exactly as long as the
/// request does; on every path that never relays (unknown model, quota,
/// shed, 502) it drops with the handler's stack frame.
#[derive(Debug)]
pub struct ActiveGuard {
    registry: Arc<ActiveRegistry>,
    id: String,
}

impl ActiveGuard {
    /// Records which backend this attempt is against, before admission —
    /// so a request parked in `Queued` names the device it is waiting for.
    /// Called again per failover attempt.
    pub fn set_attempt(&self, backend: &str) {
        self.registry.edit(&self.id, |e| {
            e.backend = Some(backend.to_string());
            e.phase = Phase::Queued;
        });
    }

    /// Admitted; the request is on the wire and we are waiting for a head.
    pub fn set_upstream(&self) {
        self.registry.edit(&self.id, |e| e.phase = Phase::Upstream);
    }

    /// The head has been relayed and bytes are flowing. Attribution comes
    /// straight from the log row, hence the `Option`s: the row's `backend`
    /// and `upstream_model` are filled in by `relay`, and a `None` leaves
    /// whatever `set_attempt` recorded rather than blanking it.
    pub fn set_streaming(
        &self,
        backend: Option<&str>,
        upstream_model: Option<&str>,
        streamed: bool,
    ) {
        self.registry.edit(&self.id, |e| {
            if let Some(b) = backend {
                e.backend = Some(b.to_string());
            }
            if let Some(m) = upstream_model {
                e.upstream_model = Some(m.to_string());
            }
            e.phase = Phase::Streaming;
            e.streamed = streamed;
        });
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.registry.inner.write().unwrap().remove(&self.id);
    }
}

/// An entry as the admin API serves it: RFC3339 timestamp and an elapsed
/// time computed at read time, since the whole point is watching it grow.
#[derive(Debug, Serialize)]
pub(crate) struct ApiActiveRow {
    pub id: String,
    pub created_at: String,
    pub token_id: String,
    pub token_name: String,
    pub model: String,
    pub path: String,
    pub backend: Option<String>,
    pub upstream_model: Option<String>,
    pub phase: Phase,
    pub streamed: bool,
    pub service_tier: Option<String>,
    pub elapsed_ms: i64,
}

impl ApiActiveRow {
    pub(crate) fn from_entry(entry: ActiveEntry, now: i64) -> Self {
        ApiActiveRow {
            created_at: crate::admin::ms_to_rfc3339(entry.created_at),
            // A request that started before a clock adjustment must not
            // report a negative age.
            elapsed_ms: (now - entry.created_at).max(0),
            id: entry.id,
            token_id: entry.token_id,
            token_name: entry.token_name,
            model: entry.model,
            path: entry.path,
            backend: entry.backend,
            upstream_model: entry.upstream_model,
            phase: entry.phase,
            streamed: entry.streamed,
            service_tier: entry.service_tier,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, model: &str) -> LogRow {
        let mut row = LogRow::new("tok-1".into(), model.into(), "/v1/chat/completions".into());
        row.id = id.into();
        row
    }

    #[test]
    fn a_registered_request_is_listed_until_its_guard_drops() {
        let reg = ActiveRegistry::new();
        assert!(reg.is_empty());

        let mut r = row("r1", "gpt-4o");
        r.service_tier = Some("flex".into());
        let guard = reg.register(&r, "primary");
        let listed = reg.snapshot();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "r1");
        assert_eq!(listed[0].model, "gpt-4o");
        assert_eq!(listed[0].token_name, "primary");
        assert_eq!(listed[0].phase, Phase::Queued);
        assert_eq!(listed[0].backend, None);
        assert!(!listed[0].streamed);
        // Taken from the row, so one header extraction serves both the
        // active panel and the eventual log entry.
        assert_eq!(listed[0].service_tier.as_deref(), Some("flex"));

        drop(guard);
        assert!(reg.is_empty());
        assert!(reg.snapshot().is_empty());
    }

    #[test]
    fn phases_advance_through_the_guard() {
        let reg = ActiveRegistry::new();
        let guard = reg.register(&row("r1", "local-qwen"), "primary");

        guard.set_attempt("vllm-a");
        let e = reg.snapshot().remove(0);
        assert_eq!(e.phase, Phase::Queued);
        assert_eq!(e.backend.as_deref(), Some("vllm-a"));

        guard.set_upstream();
        let e = reg.snapshot().remove(0);
        assert_eq!(e.phase, Phase::Upstream);
        // The backend survives the transition: it is what the request is
        // waiting on, not a property of the queue.
        assert_eq!(e.backend.as_deref(), Some("vllm-a"));

        guard.set_streaming(Some("vllm-a"), Some("Qwen/Qwen3-32B"), true);
        let e = reg.snapshot().remove(0);
        assert_eq!(e.phase, Phase::Streaming);
        assert_eq!(e.upstream_model.as_deref(), Some("Qwen/Qwen3-32B"));
        assert!(e.streamed);
    }

    #[test]
    fn streaming_without_attribution_keeps_the_attempted_backend() {
        let reg = ActiveRegistry::new();
        let guard = reg.register(&row("r1", "m"), "k");
        guard.set_attempt("vllm-a");
        guard.set_streaming(None, None, false);

        let e = reg.snapshot().remove(0);
        assert_eq!(e.phase, Phase::Streaming);
        assert_eq!(e.backend.as_deref(), Some("vllm-a"));
        assert_eq!(e.upstream_model, None);
    }

    /// A failover attempt re-queues against the next backend, and the
    /// panel must show the new one rather than the failed one.
    #[test]
    fn a_second_attempt_requeues_against_its_own_backend() {
        let reg = ActiveRegistry::new();
        let guard = reg.register(&row("r1", "m"), "primary");
        guard.set_attempt("vllm-a");
        guard.set_upstream();
        guard.set_attempt("vllm-b");

        let e = reg.snapshot().remove(0);
        assert_eq!(e.backend.as_deref(), Some("vllm-b"));
        assert_eq!(e.phase, Phase::Queued);
    }

    #[test]
    fn snapshot_is_longest_running_first() {
        let reg = ActiveRegistry::new();
        let mut newest = row("newer", "m");
        newest.created_at += 1_000;
        let mut oldest = row("older", "m");
        oldest.created_at -= 1_000;

        let _a = reg.register(&newest, "k");
        let _b = reg.register(&row("middle", "m"), "k");
        let _c = reg.register(&oldest, "k");

        let ids: Vec<String> = reg.snapshot().into_iter().map(|e| e.id).collect();
        assert_eq!(ids, ["older", "middle", "newer"]);
    }

    /// Guards are independent: one finishing must not disturb the others.
    #[test]
    fn dropping_one_guard_leaves_the_rest_listed() {
        let reg = ActiveRegistry::new();
        let a = reg.register(&row("r1", "m"), "k");
        let _b = reg.register(&row("r2", "m"), "k");
        assert_eq!(reg.len(), 2);
        drop(a);
        let ids: Vec<String> = reg.snapshot().into_iter().map(|e| e.id).collect();
        assert_eq!(ids, ["r2"]);
    }

    #[test]
    fn edits_after_the_entry_is_gone_are_no_ops() {
        // Not reachable from the proxy path (the guard owns the entry), but
        // the write lock must not panic or resurrect a finished request.
        let reg = ActiveRegistry::new();
        let guard = reg.register(&row("r1", "m"), "k");
        reg.inner.write().unwrap().remove("r1");
        guard.set_attempt("vllm-a");
        guard.set_streaming(Some("vllm-a"), Some("m"), true);
        assert!(reg.is_empty());
    }

    #[test]
    fn api_row_reports_elapsed_and_never_goes_negative() {
        let reg = ActiveRegistry::new();
        let mut r = row("r1", "m");
        r.created_at = 10_000;
        let _guard = reg.register(&r, "k");
        let entry = reg.snapshot().remove(0);

        let api = ApiActiveRow::from_entry(entry.clone(), 12_500);
        assert_eq!(api.elapsed_ms, 2_500);
        assert_eq!(api.service_tier, None);
        assert!(
            api.created_at.starts_with("1970-01-01T00:00:10"),
            "{}",
            api.created_at
        );
        assert_eq!(serde_json::to_value(api.phase).unwrap(), "queued");

        // A clock that jumped backwards must not produce a negative age.
        assert_eq!(ApiActiveRow::from_entry(entry, 9_000).elapsed_ms, 0);
    }
}
