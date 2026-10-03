//! The routes that answer without generating anything.
//!
//! Every one of them is a thin wrapper over a pure function that takes what it
//! needs and returns what it produces. That split is what makes them testable
//! without a router: the interesting question about `/health` is what it says
//! when the model is missing, not whether axum can call it.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::api::inference_types::{
    CacheReport, DiskReport, EngineCacheReport, ModelInfo, ModelListResponse, SUPPORTED_PARAMETERS,
    TopProvider,
};
use crate::error::{ApiSurface, AppError};
use crate::state::AppState;

/// The content type a scrape expects.
const METRICS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// What `/health` says.
///
/// A pure function of the one fact it reports. There is no "degraded" arm: this
/// server has exactly one model and either loaded it before binding the
/// listener or exited, so the loading state is reachable only by a caller that
/// built the state by hand - which is precisely the test that wants to see it.
#[must_use]
pub fn readiness(model_id: Option<&str>) -> (StatusCode, Value) {
    match model_id {
        Some(id) => (StatusCode::OK, json!({"status": "ready", "model": id})),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"status": "loading"}),
        ),
    }
}

/// `GET /health`.
pub async fn health(State(state): State<Arc<AppState>>) -> Response {
    // A server without a language model reports the recognizer it serves.
    let model = match (&state.engine, &state.audio) {
        (None, Some(audio)) => audio.asr_id(),
        _ => state.model_id.as_str(),
    };
    let (status, body) = readiness(Some(model));
    (status, Json(body)).into_response()
}

/// `GET /metrics`.
///
/// Hand-written rather than produced by a metrics library. Every counter is read
/// from state that already exists; a registry would add a dependency, a second
/// source of truth for what a counter is called, and a way for a counter to be
/// registered and never updated.
pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let body = render_metrics(&state);
    ([(header::CONTENT_TYPE, METRICS_CONTENT_TYPE)], body).into_response()
}

/// The metrics body.
///
/// Only counters that are read from something real. A counter that is always
/// zero because nothing updates it is worse than an absent one: a dashboard
/// built on it looks healthy for the wrong reason.
#[must_use]
pub fn render_metrics(state: &AppState) -> String {
    let disk = state.kv.as_ref().and_then(|kv| kv.disk_stats());
    let enabled = u8::from(disk.is_some());
    let entries = disk.as_ref().map_or(0, |stats| u64::from(stats.entries));
    let bytes = disk.as_ref().map_or(0, |stats| stats.durable_bytes);
    let mut out = String::new();
    push_gauge(&mut out, "shiso_up", "1 when the process is serving", 1);
    push_gauge(
        &mut out,
        "shiso_uptime_seconds",
        "Seconds since the server started",
        state.uptime_seconds(),
    );
    push_gauge(
        &mut out,
        "shiso_model_loaded",
        "1 when a checkpoint is loaded",
        1,
    );
    push_gauge(
        &mut out,
        "shiso_kvstore_enabled",
        "1 when the cache has a disk tier",
        u64::from(enabled),
    );
    push_gauge(
        &mut out,
        "shiso_kvstore_entries",
        "Entries the disk tier holds",
        entries,
    );
    push_gauge(
        &mut out,
        "shiso_kvstore_bytes",
        "Bytes the disk tier holds",
        bytes,
    );
    if let Some(disk) = disk.as_ref() {
        render_disk(&mut out, disk);
    }
    if let Some(report) = state
        .memory
        .as_ref()
        .and_then(crate::state::MemorySource::report)
    {
        render_memory(&mut out, &report);
    }
    // Writing into a String cannot fail; the result is discarded rather than
    // unwrapped so that a formatting change can never introduce a panic here.
    let _ = state.sched.render(&mut out);
    out
}

/// The memory ledger's gauges: one series per consumer, labelled by component
/// and residence, and the device's own figures beside them.
pub fn render_memory(out: &mut String, report: &shiso_core::LedgerReport) {
    use std::fmt::Write as _;
    type Read = fn(&shiso_core::ledger::Entry) -> Option<u64>;
    let series: [(&str, &str, Read); 3] = [
        (
            "shiso_memory_bytes",
            "Bytes a memory consumer holds now",
            |entry| Some(entry.bytes),
        ),
        (
            "shiso_memory_peak_bytes",
            "The most a memory consumer has held at once",
            |entry| Some(entry.peak),
        ),
        (
            "shiso_memory_limit_bytes",
            "The ceiling a memory consumer may grow to",
            |entry| entry.limit,
        ),
    ];
    // Writing into a String cannot fail; see `push_gauge`.
    for (name, help, read) in &series {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} gauge");
        for entry in &report.entries {
            if let Some(value) = read(entry) {
                let _ = writeln!(
                    out,
                    "{name}{{component=\"{}\",residence=\"{}\"}} {value}",
                    entry.tag.name(),
                    entry.tag.residence().name()
                );
            }
        }
    }
    if let Some(device) = report.now.and_then(|now| now.device) {
        push_gauge(
            out,
            "shiso_memory_device_total_bytes",
            "Device memory the driver reports",
            device.total,
        );
        push_gauge(
            out,
            "shiso_memory_device_free_bytes",
            "Device memory the backend counts as free",
            device.free,
        );
    }
    if let Some(available) = report.now.and_then(|now| now.host_available) {
        push_gauge(
            out,
            "shiso_memory_host_available_bytes",
            "The host kernel's MemAvailable",
            available,
        );
    }
    if let Some(unaccounted) = report.unaccounted() {
        let name = "shiso_memory_unaccounted_bytes";
        let _ = writeln!(
            out,
            "# HELP {name} Device memory in use since boot that no tag covers"
        );
        let _ = writeln!(out, "# TYPE {name} gauge");
        let _ = writeln!(out, "{name} {unaccounted}");
    }
}

/// `GET /debug/memory`: the memory ledger as JSON, or 404 when the backend
/// keeps none.
pub async fn memory_report(State(state): State<Arc<AppState>>) -> Response {
    match state
        .memory
        .as_ref()
        .and_then(crate::state::MemorySource::report)
    {
        Some(report) => Json(memory_body(&report)).into_response(),
        None => AppError::NotFound {
            path: "/debug/memory".to_owned(),
        }
        .on(ApiSurface::for_path("/debug/memory"))
        .into_response(),
    }
}

/// The JSON body of `/debug/memory`.
#[must_use]
pub fn memory_body(report: &shiso_core::LedgerReport) -> Value {
    let mark = |mark: &shiso_core::ledger::Mark| {
        json!({
            "label": mark.label,
            "device_total_bytes": mark.device.map(|info| info.total),
            "device_free_bytes": mark.device.map(|info| info.free),
            "host_available_bytes": mark.host_available,
        })
    };
    json!({
        "unified": report.unified,
        "components": report.entries.iter().map(|entry| json!({
            "component": entry.tag.name(),
            "residence": entry.tag.residence().name(),
            "bytes": entry.bytes,
            "peak_bytes": entry.peak,
            "allocations": entry.count,
            "limit_bytes": entry.limit,
        })).collect::<Vec<_>>(),
        "totals": {
            "device_bytes": report.total(shiso_core::Residence::Device),
            "host_bytes": report.total(shiso_core::Residence::Host),
            "disk_bytes": report.total(shiso_core::Residence::Disk),
            "unaccounted_bytes": report.unaccounted(),
        },
        "marks": report.marks.iter().map(mark).collect::<Vec<_>>(),
        "now": report.now.as_ref().map(mark),
        "table": report.to_string(),
    })
}

/// Appends one gauge with its help and type lines.
fn push_gauge(out: &mut String, name: &str, help: &str, value: u64) {
    use std::fmt::Write as _;
    // Writing into a String cannot fail; the result is discarded rather than
    // unwrapped so that a formatting change can never introduce a panic here.
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    let _ = writeln!(out, "{name} {value}");
}

/// Appends one counter series.
fn push_total(out: &mut String, name: &str, help: &str, value: u64) {
    use std::fmt::Write as _;
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} counter");
    let _ = writeln!(out, "{name} {value}");
}

/// The disk tier's traffic, queue and restore figures.
fn render_disk(out: &mut String, disk: &shiso_kvstore::DiskStats) {
    let gauges: [(&str, &str, u64); 5] = [
        (
            "shiso_kvstore_budget_bytes",
            "Bytes the disk tier may hold",
            disk.budget_bytes,
        ),
        (
            "shiso_kvstore_reserved_bytes",
            "Bytes promised to disk writes still queued",
            disk.reserved_bytes,
        ),
        (
            "shiso_kvstore_pages",
            "Page files the disk tier holds",
            u64::from(disk.pages),
        ),
        (
            "shiso_kvstore_queued_jobs",
            "Checkpoints waiting for their pages to be copied off the device",
            u64::from(disk.queued_jobs),
        ),
        (
            "shiso_kvstore_staged_bytes",
            "Host bytes copied off the device and not yet written",
            disk.staged_bytes,
        ),
    ];
    for (name, help, value) in gauges {
        push_gauge(out, name, help, value);
    }
    let totals: [(&str, &str, u64); 7] = [
        (
            "shiso_kvstore_written_bytes_total",
            "Bytes the disk tier has written",
            disk.bytes_written,
        ),
        (
            "shiso_kvstore_read_bytes_total",
            "Bytes restores have read back",
            disk.bytes_read,
        ),
        (
            "shiso_kvstore_write_failures_total",
            "Disk-tier writes that failed",
            disk.write_failures,
        ),
        (
            "shiso_kvstore_jobs_dropped_total",
            "Checkpoints dropped because the write queue was full",
            disk.jobs_dropped,
        ),
        (
            "shiso_kvstore_evictions_total",
            "Entries the disk budget released",
            disk.entries_evicted,
        ),
        (
            "shiso_kvstore_restores_total",
            "Sessions stood up from the disk tier",
            disk.restores,
        ),
        (
            "shiso_kvstore_restore_failures_total",
            "Restores that found a file unusable",
            disk.restore_failures,
        ),
    ];
    for (name, help, value) in totals {
        push_total(out, name, help, value);
    }
    render_disk_history(out, disk);
}

/// The disk tier's per-reason write counts and restore seconds.
fn render_disk_history(out: &mut String, disk: &shiso_kvstore::DiskStats) {
    use std::fmt::Write as _;
    let _ = writeln!(
        out,
        "# HELP shiso_kvstore_entries_written_total Entry files written, by reason"
    );
    let _ = writeln!(out, "# TYPE shiso_kvstore_entries_written_total counter");
    for (reason, count) in shiso_kvstore::Reason::ALL
        .iter()
        .zip(disk.entries_by_reason.iter())
    {
        let _ = writeln!(
            out,
            "shiso_kvstore_entries_written_total{{reason=\"{}\"}} {count}",
            reason.label()
        );
    }
    for (name, help, value) in [
        (
            "shiso_kvstore_restore_seconds_total",
            "Seconds spent restoring from disk",
            disk.restore_seconds,
        ),
        (
            "shiso_kvstore_restore_prefill_seconds_total",
            "Seconds of prefill restores replaced, estimated",
            disk.restore_prefill_seconds,
        ),
    ] {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} counter");
        let _ = writeln!(out, "{name} {value:.6}");
    }
}

/// `GET /v1/models`.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    Json(model_listing(&state))
}

/// The model listing.
///
/// Always answers, never 503. A client polls this to discover what to send, and
/// a server that refuses the question while it has an answer makes the client
/// retry for information it could have had.
#[must_use]
pub fn model_listing(state: &AppState) -> ModelListResponse {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
        .saturating_sub(state.uptime_seconds());
    let mut data = Vec::new();
    if state.engine.is_some() {
        data.push(ModelInfo {
            id: state.model_id.clone(),
            object: "model",
            created,
            owned_by: "shiso",
            name: state.model_id.clone(),
            context_length: state.ctx,
            top_provider: TopProvider {
                context_length: state.ctx,
                max_completion_tokens: state.ctx,
                is_moderated: false,
            },
            supported_parameters: SUPPORTED_PARAMETERS.to_vec(),
        });
    }
    // The recognizer answers to its own id, which is what a transcription's
    // `model` field must name.
    if let Some(audio) = &state.audio {
        data.push(ModelInfo {
            id: audio.asr_id().to_owned(),
            object: "model",
            created,
            owned_by: "shiso",
            name: audio.asr_id().to_owned(),
            context_length: 0,
            top_provider: TopProvider {
                context_length: 0,
                max_completion_tokens: 0,
                is_moderated: false,
            },
            supported_parameters: Vec::new(),
        });
    }
    // The generator answers to its own id the same way, for the same reason:
    // a client discovers what to name in `model` by reading this list.
    if let Some(image) = &state.image {
        data.push(ModelInfo {
            id: image.model_id().to_owned(),
            object: "model",
            created,
            owned_by: "shiso",
            name: image.model_id().to_owned(),
            context_length: 0,
            top_provider: TopProvider {
                context_length: 0,
                max_completion_tokens: 0,
                is_moderated: false,
            },
            supported_parameters: Vec::new(),
        });
    }
    ModelListResponse {
        object: "list",
        data,
    }
}

/// `GET /cache`.
pub async fn cache_report(State(state): State<Arc<AppState>>) -> Json<CacheReport> {
    Json(cache_body(&state))
}

/// The cache report.
///
/// The disk section prefers measured numbers and falls back to the tier the
/// engine was told to open. Those are two different kinds of fact - what is
/// there, and what was asked for - so the body says which one it is carrying
/// rather than letting an unmeasurable tier look like an empty one.
#[must_use]
pub fn cache_body(state: &AppState) -> CacheReport {
    let cache_stats = state.kv.as_ref().map(|kv| kv.stats());
    let min_tokens = state.ckpt.as_ref().map_or(0, |tier| tier.min_tokens);
    let disk = match (
        cache_stats.as_ref().and_then(|cache| cache.disk.clone()),
        state.ckpt.as_ref(),
    ) {
        (Some(disk), _) => DiskReport {
            enabled: true,
            dir: Some(disk.dir.display().to_string()),
            budget_bytes: disk.budget_bytes,
            min_tokens,
            measured: true,
            total_bytes: disk.durable_bytes,
            reserved_bytes: disk.reserved_bytes,
            entries: disk.entries,
            pending_entries: disk.pending_entries,
            pages: disk.pages,
            queued_jobs: disk.queued_jobs,
            staged_bytes: disk.staged_bytes,
            evictions: disk.entries_evicted,
            bytes_written: disk.bytes_written,
            bytes_read: disk.bytes_read,
            entries_written: shiso_kvstore::Reason::ALL
                .iter()
                .zip(disk.entries_by_reason.iter())
                .filter(|(_, count)| **count > 0)
                .map(|(reason, count)| (reason.label(), *count))
                .collect(),
            write_failures: disk.write_failures,
            jobs_dropped: disk.jobs_dropped,
            restores: disk.restores,
            restore_failures: disk.restore_failures,
            restores_declined: disk.restores_declined,
            restore_seconds: disk.restore_seconds,
            restore_prefill_seconds: disk.restore_prefill_seconds,
            read_bps: disk.read_bps,
            prefetches: disk.prefetches,
        },
        (None, Some(tier)) => DiskReport {
            enabled: true,
            dir: Some(tier.dir.display().to_string()),
            budget_bytes: tier.budget_bytes,
            min_tokens,
            ..DiskReport::default()
        },
        (None, None) => DiskReport::default(),
    };
    let sched = &state.sched;
    CacheReport {
        disk,
        engine: EngineCacheReport {
            requests: sched.admitted(),
            hits: sched.checkpoint_hits(),
            hit_rate: (sched.hit_rate() * 100.0).round() / 100.0,
            tokens_reused: sched.reused_tokens(),
            tokens_prefilled: sched.prefill_tokens(),
            writes: sched.checkpoint_writes(),
            resumes_memory: cache_stats.as_ref().map_or(0, |cache| cache.resumes_tree),
            resumes_disk: cache_stats.as_ref().map_or(0, |cache| cache.resumes_disk),
            prefill_s_per_token: cache_stats
                .as_ref()
                .map_or(0.0, |cache| cache.prefill_s_per_token),
            resident: sched.active(),
            capacity: state.engine.as_ref().map_or(0, |engine| {
                u64::try_from(engine.max_num_seqs()).unwrap_or(u64::MAX)
            }),
        },
    }
}

/// Anything that matched no route.
///
/// Rendered through [`AppError`] so that an unknown path answers with the same
/// JSON envelope as every other failure, in the shape the surface that owns
/// that path speaks. Axum's own fallback sends an empty body, which a client
/// reports as a transport error rather than as a 404.
pub async fn not_found(request: axum::extract::Request) -> Response {
    let path = request.uri().path().to_owned();
    let surface = ApiSurface::for_path(&path);
    AppError::NotFound { path }.on(surface).into_response()
}

/// A path that exists, tried with a method it does not serve.
///
/// Also rendered through [`AppError`], for the same reason: a bare 405 with no
/// body tells a client the request failed and nothing about why.
pub async fn method_not_allowed(request: axum::extract::Request) -> Response {
    let path = request.uri().path().to_owned();
    let surface = ApiSurface::for_path(&path);
    AppError::MethodNotAllowed {
        method: request.method().to_string(),
        path,
    }
    .on(surface)
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::readiness;
    use axum::http::StatusCode;

    #[test]
    fn a_server_with_a_model_reports_ready_and_names_it() {
        let (status, body) = readiness(Some("a-model"));
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ready");
        assert_eq!(body["model"], "a-model");
    }

    /// The loading arm answers 503, because a load balancer reads the status
    /// and not the body: a 200 that says "loading" in its text is a server that
    /// gets sent traffic it cannot serve.
    #[test]
    fn a_server_with_no_model_reports_loading_with_a_status_that_says_so() {
        let (status, body) = readiness(None);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "loading");
        assert!(body.get("model").is_none());
    }
}
