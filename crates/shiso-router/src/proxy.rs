use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use axum::Extension;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::Response;
use bytes::Bytes;
use serde::Deserialize;

use crate::active::ActiveGuard;
use crate::balancer::{Backend, RouteQos, Target};
use crate::catalog::OnShedWithoutFallback;
use crate::config::BackendKind;
use crate::error::AppError;
use crate::logging::{LogRow, LogSink};
use crate::multipart;
use crate::sched::SlotGuard;
use crate::state::AppState;
use crate::tee::{Dialect, LogContext, TeeStream, UsageAccumulator};
use crate::tokens::KeyRecord;
use crate::trace::Tracer;

/// Minimal probe: routing needs `model` (and `stream` for usage injection),
/// and the scheduler needs whatever generated-token limit the client asked
/// for. Every other field is skipped by serde without being materialized,
/// so the probe stays cheap.
#[derive(Deserialize)]
struct ModelProbe {
    model: String,
    #[serde(default)]
    stream: Option<bool>,
    /// `/v1/chat/completions` (deprecated spelling) and `/v1/messages`.
    #[serde(default)]
    max_tokens: Option<i64>,
    /// `/v1/chat/completions`, current spelling.
    #[serde(default)]
    max_completion_tokens: Option<i64>,
    /// `/v1/responses`.
    #[serde(default)]
    max_output_tokens: Option<i64>,
}

impl ModelProbe {
    /// The client's own ceiling on generated tokens, whichever endpoint it
    /// spelled it for. `None` means it asked for no limit.
    fn requested_tokens(&self) -> Option<i64> {
        self.max_completion_tokens
            .or(self.max_output_tokens)
            .or(self.max_tokens)
            .filter(|n| *n > 0)
    }
}

/// Reads the routing fields out of the request body. Every endpoint takes
/// JSON except `/v1/audio/transcriptions`, which is a multipart form with
/// the audio file beside `model`/`stream`; for a form, the byte range of
/// the `model` value comes back too, so a rename can be spliced in without
/// re-encoding the audio.
fn probe_body(
    path: &str,
    headers: &HeaderMap,
    body: &[u8],
) -> Result<(ModelProbe, Option<Range<usize>>), AppError> {
    if Dialect::for_path(path) != Dialect::OpenAiTranscriptions {
        let probe = serde_json::from_slice(body).map_err(|_| {
            AppError::BadRequest("request body must be JSON with a `model` field".into())
        })?;
        return Ok((probe, None));
    }
    let bad = || {
        AppError::BadRequest("request body must be multipart/form-data with a `model` field".into())
    };
    let boundary = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(multipart::boundary)
        .ok_or_else(bad)?;
    let form = multipart::probe(body, boundary).ok_or_else(bad)?;
    let (Some(model), Some(span)) = (form.model, form.model_span) else {
        return Err(bad());
    };
    let probe = ModelProbe {
        model,
        stream: form.stream,
        max_tokens: None,
        max_completion_tokens: None,
        max_output_tokens: None,
    };
    Ok((probe, Some(span)))
}

/// Overrides for a proxy call made in-process rather than over the wire —
/// today only the admin chat console's. Requests that arrive through the
/// router's own routes never carry one.
#[derive(Clone, Debug)]
pub(crate) struct Dispatch {
    /// Send only to this backend's targets, with no failover to the rest:
    /// a console test of one provider has to surface that provider's
    /// failure rather than quietly succeeding somewhere else.
    pub pin_backend: String,
    /// Id to give the request-log row, so the caller can read the row back
    /// once the async writer has stored it.
    pub row_id: String,
}

pub(crate) async fn proxy_handler(
    State(state): State<AppState>,
    Extension(key): Extension<Arc<KeyRecord>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, AppError> {
    proxy_request(ProxyRequestDesc {
        state,
        key,
        method,
        uri,
        headers,
        body,
        dispatch: None,
    })
    .await
}

/// Description of a request to be proxied.
pub(crate) struct ProxyRequestDesc {
    pub(crate) state: AppState,
    pub(crate) key: Arc<KeyRecord>,
    pub(crate) method: Method,
    pub(crate) uri: Uri,
    pub(crate) headers: HeaderMap,
    pub(crate) body: Bytes,
    pub(crate) dispatch: Option<Dispatch>,
}

/// The proxy path itself. `proxy_handler` is the axum entry point; the
/// admin console calls this directly with a `Dispatch` so its test traffic
/// goes through exactly the same routing, failover, teeing and logging as
/// a real client request.
pub(crate) async fn proxy_request(desc: ProxyRequestDesc) -> Result<Response, AppError> {
    let ProxyRequestDesc {
        state,
        key,
        method,
        uri,
        headers,
        body,
        dispatch,
    } = desc;
    // into their canonical `/v1` form here, so everything downstream —
    // dialect detection, `required_kind`, the logged path — sees one
    // spelling per endpoint.
    let path = canonical_path(uri.path()).to_string();
    let started = Instant::now();

    // `X-Debug: trace` prints the whole exchange to stdout. Set up before
    // the body probe so a malformed body is traceable too; the tracer's id
    // becomes the log row id so a trace maps to a stored log entry.
    let tracer = Tracer::from_headers(&headers);
    if let Some(t) = &tracer {
        t.client_request(method.as_str(), uri.path(), &path, &headers, &body);
    }

    // Body-parse failures are not logged: there is no model to attribute
    // the request to. Everything after this point produces a log row.
    let (probe, form_model) = probe_body(&path, &headers, &body).inspect_err(|err| {
        if let Some(t) = &tracer {
            t.event("rejected", &err.to_string());
        }
    })?;

    let mut row = LogRow::new(key.id.clone(), probe.model.clone(), path.clone());
    row.service_tier = service_tier(&headers);
    if let Some(t) = &tracer {
        row.id = t.id().to_string();
    }
    // A caller-supplied id wins over the tracer's: it is what the console
    // hands back to the UI to look the finished row up with.
    if let Some(d) = &dispatch {
        row.id = d.row_id.clone();
    }

    // One snapshot for the whole request: a concurrent admin catalog edit
    // cannot change the route out from under a failover loop.
    let routing = state.routing();
    let Some(route) = routing.routes.get(&probe.model) else {
        let err = AppError::UnknownModel(probe.model.clone());
        return Err(reject(&state, row, started, err, tracer.as_deref()));
    };

    // An alias resolves to its model's canonical name here, and everything
    // downstream — the allowlist, the log row, the name sent upstream —
    // works from that, so aliases are synonyms and never a second identity.
    let model = route.name.as_str();
    // Absent a target rename, a request that came in under an alias is
    // still forwarded as the canonical model name: the backend has never
    // heard of the alias.
    let alias_rename = (model != probe.model).then_some(model);
    if alias_rename.is_some() {
        row.model = model.to_string();
    }

    // Enter the in-flight registry here rather than at dispatch: the case
    // the logs page's active panel exists for is a request parked waiting
    // on a device lease, which happens below and would otherwise be
    // invisible. Registering before the pre-dispatch checks costs one map
    // insert on a request that is about to be rejected, and in exchange the
    // guard covers the request's whole post-routing life. Every path from
    // here drops it on the way out; only the relaying path hands it to the
    // tee, which holds it for the life of the response.
    let active = state.active.register(&row, &key.name);

    if !key.allows_model(model) {
        let err = AppError::Forbidden(format!(
            "model {:?} is not allowed for this API key",
            probe.model
        ));
        return Err(reject(&state, row, started, err, tracer.as_deref()));
    }

    if let Err(err) = state
        .usage_tracker
        .check(&key, time::OffsetDateTime::now_utc())
    {
        return Err(reject(&state, row, started, err, tracer.as_deref()));
    }

    // The endpoint fixes the wire format; only targets whose backend
    // speaks it are eligible (a route may mix kinds to serve the same
    // model name on both APIs).
    let kind = required_kind(&path);
    let candidates = match &dispatch {
        Some(d) => route.targets_on(&d.pin_backend, kind),
        None => route.candidates(kind),
    };
    // A route can be fallback-only for *this endpoint* even though
    // validation guarantees it is not fallback-only overall (its normal
    // target may speak the other wire format). Serving from the fallback
    // beats a 404.
    let fallbacks = match &dispatch {
        Some(_) => Vec::new(),
        None => route.fallback_candidates(kind),
    };
    if candidates.is_empty() && fallbacks.is_empty() {
        let err = AppError::NoBackendForEndpoint(match &dispatch {
            Some(d) => format!(
                "model {:?} has no target on backend {:?} serving {path}",
                probe.model, d.pin_backend
            ),
            None => format!("model {:?} has no backend serving {path}", probe.model),
        });
        return Err(reject(&state, row, started, err, tracer.as_deref()));
    }

    let streaming = probe.stream == Some(true);
    // One dialect for the whole request: it decides which field carries the
    // generated-token limit as well as how the response is parsed.
    let dialect = Dialect::for_path(&path);
    let requested_tokens = probe.requested_tokens();
    // The slot rides along with the held response: relaying its body still
    // reads from the upstream connection, so the backend is not free until
    // that is done.
    let mut last_5xx: Option<(reqwest::Response, usize, SlotGuard)> = None;
    let mut last_err: Option<String> = None;
    let mut attempt_notes: Vec<String> = Vec::new();

    // Admission control, before anything is enqueued. The console's pinned
    // dispatch deliberately opts out: a setup test waits however long it
    // has to rather than quietly reporting the fallback's behavior.
    let deadline = dispatch
        .is_none()
        .then(|| deadline_ms(&headers, &route.qos))
        .flatten();

    let mut attempts: Vec<usize> = Vec::new();
    let mut shed: Vec<usize> = Vec::new();
    // Once a device has told us it is too busy, its other members will say
    // the same thing — they contend for the same lease, so shedding from
    // moe-local to dense-local would be pointless.
    let mut shed_groups: Vec<String> = Vec::new();
    for idx in candidates {
        let backend = &route.targets[idx].backend;
        if let (Some(deadline), Some(group)) = (deadline, backend.lease_group()) {
            if shed_groups.iter().any(|g| g == group) {
                shed.push(idx);
                continue;
            }
            let est = state.scheduler.est_drain_ms(backend, requested_tokens);
            let wait = state.scheduler.predicted_wait_ms(backend, est).await;
            if wait > deadline {
                tracing::debug!(
                    backend = %backend.name, model = %model, wait, deadline,
                    "shedding: predicted wait exceeds the deadline"
                );
                if let Some(t) = &tracer {
                    t.event(
                        "shedding",
                        &format!("{}: predicted wait {wait}ms > {deadline}ms", backend.name),
                    );
                }
                // Recorded through the same attempt_notes -> row.error
                // channel failover reasons already use, so it shows up in
                // the existing logs UI with no schema change.
                attempt_notes.push(format!(
                    "{}: shed, predicted wait {wait}ms > {deadline}ms deadline",
                    backend.name
                ));
                shed_groups.push(group.to_string());
                shed.push(idx);
                continue;
            }
        }
        attempts.push(idx);
    }

    if !shed.is_empty() {
        // Reachable only from here: a fallback is what a shed sheds *to*.
        // Appending rather than replacing keeps ordinary failover working
        // when some candidate did pass admission.
        attempts.extend(fallbacks);
    }
    if attempts.is_empty() {
        match route.qos.on_shed_without_fallback {
            // Silently dropping the deadline would make it a lie, so the
            // operator chose which way to break the promise.
            OnShedWithoutFallback::Wait => {
                attempt_notes.push("no fallback target; waiting past the deadline".into());
                attempts = shed;
            }
            OnShedWithoutFallback::Error => {
                let err = AppError::DeadlineExceeded(format!(
                    "model {:?} would exceed its {}ms wait deadline and has no fallback target",
                    probe.model,
                    deadline.unwrap_or(0)
                ));
                if !attempt_notes.is_empty() {
                    row.error = Some(attempt_notes.join("; "));
                }
                return Err(reject(&state, row, started, err, tracer.as_deref()));
            }
        }
    }

    // Failover is safe: nothing is sent to the client until an upstream
    // responds, and the status line arrives before any body bytes. Once a
    // response head has been relayed the stream is committed — a mid-stream
    // upstream failure terminates the client stream (and is logged), it
    // can never retry.
    for idx in attempts {
        let target = &route.targets[idx];
        let backend = &target.backend;
        // Re-recorded per attempt: after a failover the panel must name the
        // backend now being waited on, not the one that failed.
        active.set_attempt(&backend.name);

        // Streaming OpenAI-style requests get stream_options.include_usage
        // injected (unless opted out per backend) so usage is observable;
        // the resulting extra final usage chunk passes through to the
        // client. Anthropic and Responses streams always carry usage.
        let inject = streaming && backend.inject_usage && path.ends_with("/v1/chat/completions");

        let rename = target.rename.as_deref().or(alias_rename);
        // Only a target on a grouped backend is clamped: a remote fallback
        // has no device to protect and gets the client's own value.
        let clamp_tokens = backend
            .is_grouped()
            .then_some(backend.max_tokens_cap)
            .flatten();
        let edits = BodyEdits {
            rename,
            inject_usage: inject,
            clamp_tokens,
            dialect,
            form_model: form_model.clone(),
        };
        let out_body = match prepare_body(&body, &edits) {
            Ok(b) => b,
            Err(e) => return Err(reject(&state, row, started, e, tracer.as_deref())),
        };

        let url = match join_url(&backend.base_url, &path) {
            Ok(u) => u,
            Err(e) => return Err(reject(&state, row, started, e, tracer.as_deref())),
        };
        let out_headers = upstream_headers(&headers, backend, kind);
        if let Some(t) = &tracer {
            t.upstream_request(
                &backend.name,
                url.as_str(),
                &out_headers,
                &out_body,
                out_body != body,
            );
        }
        let request = state.http.post(url).headers(out_headers).body(out_body);

        // Admission: the group's lease (for a grouped backend) and then a
        // free concurrency slot. The guard is dropped by every path out of
        // this iteration except the one that relays a response, which moves
        // it into the tee.
        // The estimate is deliberately an upper bound: the clamp above
        // guarantees the request cannot generate past the cap, and most
        // requests stop well short of it.
        let est_drain_ms = state.scheduler.est_drain_ms(backend, requested_tokens);
        let slot = state.scheduler.admit(backend, est_drain_ms).await;
        // Admitted: from here the wait is the provider's, not ours.
        active.set_upstream();

        let result = request.send().await;
        if let (Some(t), Ok(resp)) = (&tracer, &result) {
            t.upstream_response(
                &backend.name,
                resp.status().as_u16(),
                resp.headers(),
                started.elapsed().as_millis(),
            );
        }

        match result {
            Ok(resp) if resp.status().is_server_error() => {
                tracing::warn!(
                    backend = %backend.name,
                    model = %probe.model,
                    status = %resp.status(),
                    "upstream returned 5xx, failing over"
                );
                if let Some(t) = &tracer {
                    t.event(
                        "failing over",
                        &format!("{} returned {}", backend.name, resp.status()),
                    );
                }
                attempt_notes.push(format!("{}: HTTP {}", backend.name, resp.status().as_u16()));
                last_5xx = Some((resp, idx, slot));
            }
            // Any non-5xx response (including 4xx) is relayed immediately:
            // client errors are deterministic and not worth retrying.
            Ok(resp) => {
                tracing::debug!(backend = %backend.name, model = %probe.model, status = %resp.status(), "proxied");
                return Ok(relay(RelayDesc {
                    state: &state,
                    row,
                    started,
                    upstream: resp,
                    target,
                    model,
                    attempt_notes,
                    key,
                    trace: tracer,
                    slot,
                    active,
                }));
            }
            Err(e) => {
                tracing::warn!(
                    backend = %backend.name,
                    model = %probe.model,
                    error = %e,
                    "upstream request failed, failing over"
                );
                if let Some(t) = &tracer {
                    t.event("failing over", &format!("{}: {e}", backend.name));
                }
                attempt_notes.push(format!("{}: {e}", backend.name));
                last_err = Some(e.to_string());
            }
        }
    }

    if let Some((resp, idx, slot)) = last_5xx {
        // The 5xx response that gets relayed already appears in
        // attempt_notes; relay() records the notes on the log row.
        let target = &route.targets[idx];
        return Ok(relay(RelayDesc {
            state: &state,
            row,
            started,
            upstream: resp,
            target,
            model,
            attempt_notes,
            key,
            trace: tracer,
            slot,
            active,
        }));
    }
    if !attempt_notes.is_empty() {
        row.error = Some(format!("failover: {}", attempt_notes.join("; ")));
    }
    let err = AppError::BadGateway(last_err.unwrap_or_else(|| "no backend available".into()));
    Err(reject(&state, row, started, err, tracer.as_deref()))
}

/// Finalizes and emits the log row for a request rejected before any
/// upstream response was relayed, passing the error through.
fn reject(
    state: &AppState,
    mut row: LogRow,
    started: Instant,
    err: AppError,
    trace: Option<&Tracer>,
) -> AppError {
    row.status = Some(err.status().as_u16());
    row.duration_ms = Some(started.elapsed().as_millis() as i64);
    let msg = err.to_string();
    row.error = Some(match row.error.take() {
        Some(prev) => format!("{prev}; {msg}"),
        None => msg,
    });
    state.log_sink.send(row);
    if let Some(t) = trace {
        t.event(
            &format!("rejected  status: {}", err.status().as_u16()),
            &err.to_string(),
        );
    }
    err
}

struct RelayDesc<'a> {
    state: &'a AppState,
    row: LogRow,
    started: Instant,
    upstream: reqwest::Response,
    target: &'a Target,
    model: &'a str,
    attempt_notes: Vec<String>,
    key: Arc<KeyRecord>,
    trace: Option<Arc<Tracer>>,
    slot: SlotGuard,
    active: ActiveGuard,
}

/// Fills in the target attribution and hands the response to the teeing
/// relay; from here on the log row is finalized by the stream itself.
fn relay(desc: RelayDesc<'_>) -> Response {
    let RelayDesc {
        state,
        mut row,
        started,
        upstream,
        target,
        model,
        attempt_notes,
        key,
        trace,
        slot,
        active,
    } = desc;
    row.backend = Some(target.backend.name.clone());
    row.upstream_model = Some(target.rename.clone().unwrap_or_else(|| model.to_string()));
    if !attempt_notes.is_empty() {
        row.error = Some(format!("failover: {}", attempt_notes.join("; ")));
    }
    let dialect = Dialect::for_path(&row.path);
    let ctx = LogContext {
        row,
        started,
        pricing: target.pricing,
        dialect,
        key,
        trace,
        // From here the slot's lifetime is the stream's: the tee releases
        // it on completion, error, or client disconnect alike. The registry
        // entry rides along for exactly the same reason.
        slot: Some(slot),
        active: Some(active),
    };
    stream_through(upstream, ctx, state.log_sink.clone())
}

/// The client's service tier as it is worth *recording*. The header itself
/// is forwarded verbatim by `upstream_headers` — this is only for the log
/// row and the admin UI, so it normalizes case and whitespace and folds the
/// two unremarkable tiers away: `auto` and `default` are what a request
/// gets by saying nothing, so storing them would put a value on almost
/// every row and tell the reader nothing. A null tier means "ordinary".
///
/// The value is client-controlled and ends up in the database, so it is
/// length-capped; a longer one is dropped rather than truncated, since half
/// a tier name is worse than no tier name.
fn service_tier(headers: &HeaderMap) -> Option<String> {
    const MAX_TIER_LEN: usize = 32;
    let raw = headers.get(SERVICE_TIER_HEADER)?.to_str().ok()?.trim();
    if raw.is_empty() || raw.len() > MAX_TIER_LEN {
        return None;
    }
    let tier = raw.to_ascii_lowercase();
    (tier != "auto" && tier != "default").then_some(tier)
}

/// Forwarded upstream *and* recorded; unlike the two headers below it is in
/// `upstream_headers`' allowlist, because the tier is the client's own
/// instruction to the provider.
const SERVICE_TIER_HEADER: &str = "x-service-tier";

/// Header naming follows the project's existing `x-…` convention (see
/// `X-Debug: trace`). Neither is in `upstream_headers`' allowlist, so
/// neither ever reaches a backend.
const QOS_CLASS_HEADER: &str = "x-router-qos";
const QOS_MAX_WAIT_HEADER: &str = "x-router-max-wait-ms";

/// How long this request may be predicted to queue before it is shed to a
/// fallback. `None` means wait indefinitely, which is what every route did
/// before scheduling existed.
///
/// The route's `qos_max_wait_ms` is the default; `x-router-qos: background`
/// selects the (usually far longer) background deadline instead, because
/// one deadline for all traffic serves nobody — interactive chat wants a
/// couple of seconds and a batch summarization job is happy to wait a
/// minute for local hardware. `x-router-max-wait-ms` lets a client name its
/// own, but only *downward*: a client can make itself shed sooner, never
/// hold a slot longer than the operator sanctioned.
fn deadline_ms(headers: &HeaderMap, qos: &RouteQos) -> Option<i64> {
    let background = headers
        .get(QOS_CLASS_HEADER)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("background"));
    let class = if background {
        qos.background_max_wait_ms
    } else {
        qos.max_wait_ms
    };
    let requested = headers
        .get(QOS_MAX_WAIT_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<i64>().ok())
        .filter(|n| *n > 0);
    match (requested, class) {
        (Some(client), Some(ceiling)) => Some(client.min(ceiling)),
        (Some(client), None) => Some(client),
        (None, class) => class,
    }
}

/// The API kind an inbound endpoint speaks (same mapping as
/// `Dialect::for_path`): `/v1/messages` is Anthropic wire format,
/// `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`,
/// `/v1/audio/transcriptions`, `/v1/decisions`, `/v1/systemone` and
/// `/v1/images/generations` are OpenAI.
fn required_kind(path: &str) -> BackendKind {
    if path.ends_with("/v1/messages") {
        BackendKind::Anthropic
    } else {
        BackendKind::Openai
    }
}

/// Maps an inbound endpoint path to its canonical `/v1`-prefixed form.
/// Anything unrecognized is returned unchanged (the router only matches
/// the endpoints listed here, so that branch is defensive).
fn canonical_path(path: &str) -> &str {
    match path {
        "/chat/completions" => "/v1/chat/completions",
        "/responses" => "/v1/responses",
        "/v1/v1/responses" => "/v1/responses",
        "/embeddings" => "/v1/embeddings",
        "/audio/transcriptions" => "/v1/audio/transcriptions",
        "/messages" => "/v1/messages",
        "/v1/v1/messages" => "/v1/messages",
        "/decisions" => "/v1/decisions",
        "/systemone" => "/v1/systemone",
        // `/v1/images/generations` is the OpenAI spelling; `/images` (the
        // OpenRouter one) and `/v1/images` are the shorthands clients reach
        // for. All of them go upstream as the OpenAI path.
        "/images" | "/v1/images" | "/images/generations" => "/v1/images/generations",
        other => other,
    }
}

fn join_url(base: &url::Url, path: &str) -> Result<url::Url, AppError> {
    // Preserve any path prefix on the base URL (e.g. http://host/proxy).
    let base = base.as_str().trim_end_matches('/');
    let mut path = path.trim_start_matches('/');
    // A base URL that already ends in /v1 (as most provider URLs are
    // written) must not produce /v1/v1/responses.
    if base.ends_with("/v1") {
        path = path.strip_prefix("v1/").unwrap_or(path);
    }
    let joined = format!("{base}/{path}");
    url::Url::parse(&joined).map_err(|e| AppError::Internal(format!("bad upstream url: {e}")))
}

/// Everything one candidate needs rewritten into the outbound body. All of
/// it happens in the single JSON round-trip below, so a clamp costs nothing
/// beyond what a rename already did.
struct BodyEdits<'a> {
    /// Model name to substitute (a target rename, or an alias resolved to
    /// its canonical name).
    rename: Option<&'a str>,
    inject_usage: bool,
    /// Ceiling to clamp the generated-token limit to — set only for a
    /// target on a `grouped` backend, so a remote fallback still gets the
    /// client's own value.
    clamp_tokens: Option<i64>,
    /// Which field carries that limit on this endpoint.
    dialect: Dialect,
    /// Set when the body is a multipart form (transcriptions): where the
    /// `model` value sits. A form has no usage option to inject and no
    /// generated-token limit to clamp, so only a rename applies to it.
    form_model: Option<Range<usize>>,
}

impl BodyEdits<'_> {
    fn is_noop(&self) -> bool {
        self.rename.is_none()
            && (self.form_model.is_some() || (!self.inject_usage && self.clamp_tokens.is_none()))
    }
}

/// Prepares the outbound body: applies a model rename, injects
/// `stream_options.include_usage`, and/or clamps the generated-token limit,
/// in one JSON round-trip. When none is needed the original bytes are
/// forwarded untouched (refcounted clone). Full JSON round-trip on the
/// rewrite path — correctness (escaping, `"model"` appearing inside message
/// content) beats a byte-splice micro-optimization.
fn prepare_body(body: &Bytes, edits: &BodyEdits<'_>) -> Result<Bytes, AppError> {
    if edits.is_noop() {
        return Ok(body.clone());
    }
    if let Some(span) = &edits.form_model {
        let new_model = edits.rename.expect("a form edit is always a rename");
        return Ok(multipart::splice(body, span.clone(), new_model));
    }
    let mut value: serde_json::Value = serde_json::from_slice(body)
        .map_err(|_| AppError::BadRequest("request body must be valid JSON".into()))?;
    let obj = value
        .as_object_mut()
        .ok_or_else(|| AppError::BadRequest("request body must be a JSON object".into()))?;
    if let Some(new_model) = edits.rename {
        obj.insert(
            "model".to_string(),
            serde_json::Value::String(new_model.to_string()),
        );
    }
    if edits.inject_usage {
        let so = obj
            .entry("stream_options")
            .or_insert_with(|| serde_json::Value::Object(Default::default()));
        // A client-set include_usage (true or false) is respected; sibling
        // keys in an existing stream_options are preserved. A non-object
        // stream_options is left alone for the upstream to reject.
        if let Some(map) = so.as_object_mut() {
            map.entry("include_usage")
                .or_insert(serde_json::Value::Bool(true));
        }
    }
    if let Some(cap) = edits.clamp_tokens {
        clamp_token_limit(obj, edits.dialect, cap);
    }
    Ok(Bytes::from(
        serde_json::to_vec(&value).expect("serializing a Value cannot fail"),
    ))
}

/// The request field that caps generated tokens on each endpoint. `None`
/// for embeddings, which generate nothing, and transcriptions, whose output
/// is bounded by the audio rather than by a request field.
fn token_limit_field(
    dialect: Dialect,
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Option<&'static str> {
    match dialect {
        // `max_tokens` is deprecated in favor of `max_completion_tokens`,
        // but both are still accepted and clients send either — clamp
        // whichever one is actually present, and inject the widely
        // supported `max_tokens` when neither is (grouped backends are
        // local OpenAI-compatible servers, which all take it).
        Dialect::OpenAiChat => Some(if obj.contains_key("max_completion_tokens") {
            "max_completion_tokens"
        } else {
            "max_tokens"
        }),
        Dialect::OpenAiResponses => Some("max_output_tokens"),
        // Required on this endpoint, so it is always a clamp, never an
        // injection.
        Dialect::Anthropic => Some("max_tokens"),
        Dialect::OpenAiEmbeddings | Dialect::OpenAiTranscriptions => None,
        // Scoring endpoints generate no tokens, so there is no limit to
        // clamp — and injecting one would be a refusal: the decisions wire
        // refuses unknown fields.
        Dialect::Scoring => None,
        // An image's cost is its step count and size, not a token budget.
        Dialect::Images => None,
    }
}

/// Holds the generated-token limit at or below `cap`, inserting it when the
/// client sent none at all. This is what bounds how long the request can
/// occupy a shared device — and so what makes every wait estimate in the
/// scheduler mean anything. The client sees the effect as a normal
/// `finish_reason: "length"`.
fn clamp_token_limit(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    dialect: Dialect,
    cap: i64,
) {
    let Some(field) = token_limit_field(dialect, obj) else {
        return;
    };
    // A present-but-not-a-number value (`null`, most often) counts as
    // absent: the client expressed no limit.
    let requested = obj.get(field).and_then(serde_json::Value::as_i64);
    match requested {
        Some(n) if n <= cap => return,
        Some(n) => tracing::info!(
            field,
            requested = n,
            cap,
            "clamped the generated-token limit"
        ),
        None => tracing::info!(field, cap, "capped an unlimited generation"),
    }
    obj.insert(field.to_string(), serde_json::Value::from(cap));
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    name == header::CONNECTION
        || name.as_str() == "keep-alive"
        || name == header::PROXY_AUTHENTICATE
        || name == header::PROXY_AUTHORIZATION
        || name == header::TE
        || name == header::TRAILER
        || name == header::TRANSFER_ENCODING
        || name == header::UPGRADE
}

/// Builds the outbound header set from scratch (allowlist) — the client's
/// credentials are never forwarded, and the backend's key is injected in
/// the header the wire format being spoken (`kind`) expects. A dual-kind
/// backend gets Bearer auth on OpenAI endpoints and x-api-key on
/// `/v1/messages`.
fn upstream_headers(inbound: &HeaderMap, backend: &Backend, kind: BackendKind) -> HeaderMap {
    let mut out = HeaderMap::new();

    let content_type = inbound
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or_else(|| HeaderValue::from_static("application/json"));
    out.insert(header::CONTENT_TYPE, content_type);
    if let Some(accept) = inbound.get(header::ACCEPT) {
        out.insert(header::ACCEPT, accept.clone());
    }
    if let Some(beta) = inbound.get("anthropic-beta") {
        out.insert(HeaderName::from_static("anthropic-beta"), beta.clone());
    }
    // Service tier is a provider-side routing/priority hint the client owns
    // (OpenAI and Anthropic both read it); it carries no credential, so a
    // transparent proxy forwards it rather than silently downgrading the
    // caller's tier.
    if let Some(tier) = inbound.get(SERVICE_TIER_HEADER) {
        out.insert(HeaderName::from_static(SERVICE_TIER_HEADER), tier.clone());
    }
    // The client's User-Agent is forwarded verbatim: providers key
    // compatibility behavior off the SDK that's calling them. OpenRouter,
    // for one, omits the leading `: ` SSE keepalive comment for the OpenAI
    // SDK user agents and sends it for everyone else — so stripping the UA
    // silently changes the shape of the stream the client receives.
    if let Some(ua) = inbound.get(header::USER_AGENT) {
        out.insert(header::USER_AGENT, ua.clone());
    }
    // Same reasoning for the SDK telemetry headers the OpenAI/Anthropic
    // clients send (`x-stainless-lang`, `-runtime`, `-package-version`, …):
    // providers read them alongside the UA to identify the caller, so a
    // transparent proxy passes them through.
    for (name, value) in inbound {
        if name.as_str().starts_with("x-stainless-") {
            out.insert(name.clone(), value.clone());
        }
    }

    match kind {
        BackendKind::Openai => {
            if let Some(key) = &backend.api_key {
                let mut value = HeaderValue::try_from(format!("Bearer {key}")).expect("valid key");
                value.set_sensitive(true);
                out.insert(header::AUTHORIZATION, value);
            }
        }
        BackendKind::Anthropic => {
            if let Some(key) = &backend.api_key {
                let mut value = HeaderValue::try_from(key.as_str()).expect("valid key");
                value.set_sensitive(true);
                out.insert(HeaderName::from_static("x-api-key"), value);
            }
            let version = inbound
                .get("anthropic-version")
                .cloned()
                .unwrap_or_else(|| {
                    HeaderValue::try_from(backend.anthropic_version.as_str())
                        .expect("valid version string")
                });
            out.insert(HeaderName::from_static("anthropic-version"), version);
        }
    }

    out
}

/// Relays an upstream response as-is, streaming body bytes straight
/// through without parsing or re-serializing — SSE chunks are flushed as
/// they arrive. The stream is teed through a `UsageAccumulator` that
/// observes (never modifies) the bytes and emits the log row when the
/// stream completes, errors, or the client disconnects.
fn stream_through(upstream: reqwest::Response, mut ctx: LogContext, sink: LogSink) -> Response {
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).expect("status already validated");
    ctx.row.status = Some(status.as_u16());

    let content_type = upstream
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let is_sse = content_type.starts_with("text/event-stream");
    ctx.row.streamed = is_sse;
    // The active panel's last transition: `relay` filled the row's
    // attribution in, and whether the body is an SSE stream is only known
    // here, from the response head.
    if let Some(a) = &ctx.active {
        a.set_streaming(
            ctx.row.backend.as_deref(),
            ctx.row.upstream_model.as_deref(),
            is_sse,
        );
    }
    // upstream_headers never asks for compression, but guard anyway: an
    // encoded body cannot be parsed, so usage stays unknown rather than
    // silently wrong.
    let content_encoded = upstream
        .headers()
        .get(header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"));
    let acc = UsageAccumulator::new(ctx.dialect, is_sse, content_encoded);

    let mut builder = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        if is_hop_by_hop(name) {
            continue;
        }
        builder = builder.header(name, value);
    }

    let expected_len = upstream.content_length();
    let trace = ctx.trace.clone();
    let stream = TeeStream::new(upstream.bytes_stream(), acc, ctx, sink, expected_len);
    let response = builder
        .body(Body::from_stream(stream))
        .expect("response builder with valid upstream parts");
    // The head as the client sees it: same status, minus hop-by-hop
    // headers. The body is relayed byte-for-byte, so the chunks the tee
    // traces are exactly what goes out.
    if let Some(t) = trace {
        t.client_response(response.status().as_u16(), response.headers());
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The common case: a rename and/or usage injection on a chat request,
    /// with no token clamp.
    fn edits(rename: Option<&str>, inject_usage: bool) -> BodyEdits<'_> {
        BodyEdits {
            rename,
            inject_usage,
            clamp_tokens: None,
            dialect: Dialect::OpenAiChat,
            form_model: None,
        }
    }

    fn clamp(dialect: Dialect, cap: i64) -> BodyEdits<'static> {
        BodyEdits {
            rename: None,
            inject_usage: false,
            clamp_tokens: Some(cap),
            dialect,
            form_model: None,
        }
    }

    #[test]
    fn rewrite_replaces_only_top_level_model() {
        let body = Bytes::from_static(
            br#"{"model":"gpt-4o","messages":[{"role":"user","content":"the model is \"gpt-4o\""}],"stream":true}"#,
        );
        let out = prepare_body(&body, &edits(Some("llama-3.3-70b"), false)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "llama-3.3-70b");
        assert_eq!(v["messages"][0]["content"], "the model is \"gpt-4o\"");
        assert_eq!(v["stream"], true);
    }

    #[test]
    fn rewrite_handles_unicode_and_escapes() {
        let body = Bytes::from(r#"{"model":"a\"b","note":"café ☕"}"#.as_bytes().to_vec());
        let out = prepare_body(&body, &edits(Some("new"), false)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "new");
        assert_eq!(v["note"], "café ☕");
    }

    #[test]
    fn rewrite_rejects_non_object() {
        assert!(prepare_body(&Bytes::from_static(b"[1,2]"), &edits(Some("m"), false)).is_err());
        assert!(prepare_body(&Bytes::from_static(b"not json"), &edits(Some("m"), false)).is_err());
    }

    #[test]
    fn untouched_body_is_forwarded_without_reserialization() {
        // Whitespace/key-order quirks survive because the original bytes
        // are forwarded, not re-serialized.
        let body = Bytes::from_static(b"{ \"model\" : \"m\",  \"x\": 1 }");
        let out = prepare_body(&body, &edits(None, false)).unwrap();
        assert_eq!(out, body);
    }

    #[test]
    fn injects_include_usage_when_missing() {
        let body = Bytes::from_static(br#"{"model":"m","stream":true}"#);
        let out = prepare_body(&body, &edits(None, true)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["stream_options"]["include_usage"], true);
        assert_eq!(v["stream"], true);
    }

    #[test]
    fn injection_preserves_existing_stream_options_keys() {
        let body = Bytes::from_static(br#"{"model":"m","stream_options":{"other_flag":7}}"#);
        let out = prepare_body(&body, &edits(None, true)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["stream_options"]["include_usage"], true);
        assert_eq!(v["stream_options"]["other_flag"], 7);
    }

    #[test]
    fn injection_respects_client_set_include_usage() {
        let body = Bytes::from_static(br#"{"model":"m","stream_options":{"include_usage":false}}"#);
        let out = prepare_body(&body, &edits(None, true)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["stream_options"]["include_usage"], false);
    }

    #[test]
    fn injection_composes_with_rename() {
        let body = Bytes::from_static(br#"{"model":"m","stream":true}"#);
        let out = prepare_body(&body, &edits(Some("upstream-m"), true)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "upstream-m");
        assert_eq!(v["stream_options"]["include_usage"], true);
    }

    #[test]
    fn clamp_lowers_an_over_cap_limit_per_dialect() {
        let cases = [
            (
                Dialect::OpenAiChat,
                br#"{"model":"m","max_tokens":8000}"#.as_slice(),
                "max_tokens",
            ),
            (
                Dialect::OpenAiChat,
                br#"{"model":"m","max_completion_tokens":8000}"#.as_slice(),
                "max_completion_tokens",
            ),
            (
                Dialect::OpenAiResponses,
                br#"{"model":"m","max_output_tokens":8000}"#.as_slice(),
                "max_output_tokens",
            ),
            (
                Dialect::Anthropic,
                br#"{"model":"m","max_tokens":8000}"#.as_slice(),
                "max_tokens",
            ),
        ];
        for (dialect, body, field) in cases {
            let out = prepare_body(&Bytes::from_static(body), &clamp(dialect, 1024)).unwrap();
            let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
            assert_eq!(v[field], 1024, "{dialect:?}");
        }
    }

    #[test]
    fn clamp_prefers_the_field_the_client_actually_sent() {
        // Both spellings present: the modern one wins and the legacy one is
        // left exactly as the client wrote it.
        let body =
            Bytes::from_static(br#"{"model":"m","max_tokens":50,"max_completion_tokens":9000}"#);
        let out = prepare_body(&body, &clamp(Dialect::OpenAiChat, 1024)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["max_completion_tokens"], 1024);
        assert_eq!(v["max_tokens"], 50);
    }

    #[test]
    fn clamp_leaves_a_limit_already_under_the_cap_alone() {
        let body = Bytes::from_static(br#"{"model":"m","max_tokens":100}"#);
        let out = prepare_body(&body, &clamp(Dialect::OpenAiChat, 1024)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["max_tokens"], 100);
    }

    #[test]
    fn clamp_injects_the_cap_when_the_client_set_no_limit() {
        // An omitted limit is the dangerous case: without the cap the
        // request could occupy the device indefinitely.
        let body = Bytes::from_static(br#"{"model":"m","messages":[]}"#);
        let out = prepare_body(&body, &clamp(Dialect::OpenAiChat, 512)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["max_tokens"], 512);

        // An explicit null says the same thing as omitting it.
        let body = Bytes::from_static(br#"{"model":"m","max_output_tokens":null}"#);
        let out = prepare_body(&body, &clamp(Dialect::OpenAiResponses, 512)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["max_output_tokens"], 512);
    }

    #[test]
    fn clamp_does_not_apply_to_embeddings() {
        let body = Bytes::from_static(br#"{"model":"m","input":"hi"}"#);
        let out = prepare_body(&body, &clamp(Dialect::OpenAiEmbeddings, 512)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v.as_object().unwrap().len(), 2, "nothing was added: {v}");
    }

    /// The scoring wires refuse unknown fields, so a clamped `max_tokens`
    /// injection would turn a grouped-backend request into a 400.
    #[test]
    fn clamp_does_not_apply_to_scoring() {
        let body = Bytes::from_static(br#"{"model":"m","input":"hi","questions":[]}"#);
        let out = prepare_body(&body, &clamp(Dialect::Scoring, 512)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v.as_object().unwrap().len(), 3, "nothing was added: {v}");
    }

    #[test]
    fn canonical_paths_fold_the_image_spellings() {
        for p in [
            "/images",
            "/v1/images",
            "/images/generations",
            "/v1/images/generations",
        ] {
            assert_eq!(canonical_path(p), "/v1/images/generations");
        }
        assert_eq!(Dialect::for_path("/v1/images/generations"), Dialect::Images);
        assert_eq!(required_kind("/v1/images/generations"), BackendKind::Openai);
    }

    /// The image wire has no token limit; an injected `max_tokens` would
    /// be a field the generator never asked for.
    #[test]
    fn clamp_does_not_apply_to_images() {
        let body = Bytes::from_static(br#"{"model":"m","prompt":"a fox","steps":20}"#);
        let out = prepare_body(&body, &clamp(Dialect::Images, 512)).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v.as_object().unwrap().len(), 3, "nothing was added: {v}");
    }

    #[test]
    fn canonical_paths_map_the_prefixless_spellings() {
        assert_eq!(canonical_path("/decisions"), "/v1/decisions");
        assert_eq!(canonical_path("/systemone"), "/v1/systemone");
        assert_eq!(canonical_path("/v1/decisions"), "/v1/decisions");
        // And they speak the scoring dialect once canonical.
        assert_eq!(Dialect::for_path("/v1/decisions"), Dialect::Scoring);
        assert_eq!(Dialect::for_path("/v1/systemone"), Dialect::Scoring);
    }

    const FORM: &[u8] = b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp3\"\r\n\r\n\x00\x01\r\n\
        --b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nalias\r\n--b--\r\n";

    fn form_edits(rename: Option<&str>) -> BodyEdits<'_> {
        let span = multipart::probe(FORM, "b").unwrap().model_span;
        // Everything a grouped streaming chat target would ask for, none of
        // which means anything to a form.
        BodyEdits {
            rename,
            inject_usage: true,
            clamp_tokens: Some(256),
            dialect: Dialect::OpenAiTranscriptions,
            form_model: span,
        }
    }

    #[test]
    fn form_body_is_forwarded_untouched_without_a_rename() {
        let body = Bytes::from_static(FORM);
        assert_eq!(prepare_body(&body, &form_edits(None)).unwrap(), body);
    }

    #[test]
    fn form_body_rename_splices_only_the_model_value() {
        let body = Bytes::from_static(FORM);
        let out = prepare_body(&body, &form_edits(Some("whisper-1"))).unwrap();
        let expected = String::from_utf8_lossy(FORM).replace("\r\n\r\nalias", "\r\n\r\nwhisper-1");
        assert_eq!(String::from_utf8_lossy(&out), expected);
    }

    #[test]
    fn transcription_probe_reads_model_and_stream_from_the_form() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("multipart/form-data; boundary=b"),
        );
        let (probe, span) = probe_body("/v1/audio/transcriptions", &headers, FORM).unwrap();
        assert_eq!(probe.model, "alias");
        assert_eq!(probe.stream, None);
        assert_eq!(&FORM[span.unwrap()], b"alias");

        // A JSON body on the form endpoint, and a form on a JSON endpoint,
        // are both malformed.
        let json = br#"{"model":"whisper-1"}"#;
        assert!(probe_body("/v1/audio/transcriptions", &headers, json).is_err());
        assert!(probe_body("/v1/audio/transcriptions", &HeaderMap::new(), FORM).is_err());
        assert!(probe_body("/v1/chat/completions", &headers, FORM).is_err());
    }

    #[test]
    fn clamp_composes_with_rename_and_usage_injection() {
        let body = Bytes::from_static(br#"{"model":"m","stream":true,"max_tokens":9000}"#);
        let out = prepare_body(
            &body,
            &BodyEdits {
                rename: Some("upstream-m"),
                inject_usage: true,
                clamp_tokens: Some(256),
                dialect: Dialect::OpenAiChat,
                form_model: None,
            },
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "upstream-m");
        assert_eq!(v["stream_options"]["include_usage"], true);
        assert_eq!(v["max_tokens"], 256);
    }

    #[test]
    fn probe_reads_whichever_token_limit_the_client_sent() {
        let probe = |body: &str| -> ModelProbe { serde_json::from_str(body).unwrap() };
        assert_eq!(
            probe(r#"{"model":"m","max_tokens":100}"#).requested_tokens(),
            Some(100)
        );
        assert_eq!(
            probe(r#"{"model":"m","max_completion_tokens":200}"#).requested_tokens(),
            Some(200)
        );
        assert_eq!(
            probe(r#"{"model":"m","max_output_tokens":300}"#).requested_tokens(),
            Some(300)
        );
        // The modern chat spelling wins over the deprecated one.
        assert_eq!(
            probe(r#"{"model":"m","max_tokens":1,"max_completion_tokens":2}"#).requested_tokens(),
            Some(2)
        );
        assert_eq!(probe(r#"{"model":"m"}"#).requested_tokens(), None);
        assert_eq!(
            probe(r#"{"model":"m","max_tokens":null}"#).requested_tokens(),
            None
        );
        // Unknown fields are still skipped without being materialized.
        assert_eq!(
            probe(r#"{"model":"m","messages":[{"role":"user","content":"x"}]}"#).model,
            "m"
        );
    }

    #[test]
    fn deadline_picks_the_class_the_client_asked_for() {
        let qos = RouteQos {
            max_wait_ms: Some(2_000),
            background_max_wait_ms: Some(60_000),
            on_shed_without_fallback: OnShedWithoutFallback::Wait,
        };
        let with = |pairs: &[(&str, &str)]| {
            let mut h = HeaderMap::new();
            for (k, v) in pairs {
                h.insert(
                    HeaderName::from_bytes(k.as_bytes()).unwrap(),
                    HeaderValue::from_str(v).unwrap(),
                );
            }
            h
        };

        assert_eq!(deadline_ms(&HeaderMap::new(), &qos), Some(2_000));
        assert_eq!(
            deadline_ms(&with(&[(QOS_CLASS_HEADER, "background")]), &qos),
            Some(60_000)
        );
        // Case and surrounding space are not the client's problem.
        assert_eq!(
            deadline_ms(&with(&[(QOS_CLASS_HEADER, " Background ")]), &qos),
            Some(60_000)
        );
        // An unrecognized class falls back to the interactive deadline.
        assert_eq!(
            deadline_ms(&with(&[(QOS_CLASS_HEADER, "urgent")]), &qos),
            Some(2_000)
        );

        // A client may shorten its own deadline…
        assert_eq!(
            deadline_ms(&with(&[(QOS_MAX_WAIT_HEADER, "500")]), &qos),
            Some(500)
        );
        // …but never lengthen it past what the operator sanctioned, or it
        // could pin a slot on a shared device indefinitely.
        assert_eq!(
            deadline_ms(&with(&[(QOS_MAX_WAIT_HEADER, "999999")]), &qos),
            Some(2_000)
        );
        assert_eq!(
            deadline_ms(
                &with(&[
                    (QOS_CLASS_HEADER, "background"),
                    (QOS_MAX_WAIT_HEADER, "999999")
                ]),
                &qos
            ),
            Some(60_000)
        );
        // Garbage is ignored rather than treated as zero.
        for bad in ["", "abc", "-5", "0", "1.5"] {
            assert_eq!(
                deadline_ms(&with(&[(QOS_MAX_WAIT_HEADER, bad)]), &qos),
                Some(2_000),
                "{bad:?}"
            );
        }

        // A route with no deadline waits indefinitely — the behavior every
        // route had before scheduling existed — unless a client opts in.
        let none = RouteQos::default();
        assert_eq!(deadline_ms(&HeaderMap::new(), &none), None);
        assert_eq!(
            deadline_ms(&with(&[(QOS_MAX_WAIT_HEADER, "750")]), &none),
            Some(750)
        );
    }

    #[test]
    fn qos_headers_are_not_forwarded_upstream() {
        let mut inbound = HeaderMap::new();
        inbound.insert(QOS_CLASS_HEADER, HeaderValue::from_static("background"));
        inbound.insert(QOS_MAX_WAIT_HEADER, HeaderValue::from_static("1000"));
        let out = upstream_headers(
            &inbound,
            &backend(vec![BackendKind::Openai]),
            BackendKind::Openai,
        );
        assert!(out.get(QOS_CLASS_HEADER).is_none());
        assert!(out.get(QOS_MAX_WAIT_HEADER).is_none());
    }

    #[test]
    fn join_url_preserves_base_path() {
        let base = url::Url::parse("http://host:1234/prefix").unwrap();
        assert_eq!(
            join_url(&base, "/v1/chat/completions").unwrap().as_str(),
            "http://host:1234/prefix/v1/chat/completions"
        );
        let base = url::Url::parse("https://api.openai.com").unwrap();
        assert_eq!(
            join_url(&base, "/v1/responses").unwrap().as_str(),
            "https://api.openai.com/v1/responses"
        );
    }

    #[test]
    fn join_url_does_not_double_the_v1_prefix() {
        for base in ["https://api.openai.com/v1", "https://api.openai.com/v1/"] {
            let base = url::Url::parse(base).unwrap();
            assert_eq!(
                join_url(&base, "/v1/responses").unwrap().as_str(),
                "https://api.openai.com/v1/responses"
            );
            assert_eq!(
                join_url(&base, "/v1/messages").unwrap().as_str(),
                "https://api.openai.com/v1/messages"
            );
            assert_eq!(
                join_url(&base, "/v1/chat/completions").unwrap().as_str(),
                "https://api.openai.com/v1/chat/completions"
            );
        }
        // Only an exact trailing /v1 segment counts.
        let base = url::Url::parse("http://host/v10").unwrap();
        assert_eq!(
            join_url(&base, "/v1/responses").unwrap().as_str(),
            "http://host/v10/v1/responses"
        );
    }

    #[test]
    fn canonical_path_folds_prefixless_aliases() {
        assert_eq!(canonical_path("/responses"), "/v1/responses");
        assert_eq!(canonical_path("/messages"), "/v1/messages");
        assert_eq!(canonical_path("/chat/completions"), "/v1/chat/completions");
        assert_eq!(canonical_path("/embeddings"), "/v1/embeddings");
        assert_eq!(
            canonical_path("/audio/transcriptions"),
            "/v1/audio/transcriptions"
        );
        assert_eq!(canonical_path("/v1/responses"), "/v1/responses");
        assert_eq!(canonical_path("/healthz"), "/healthz");
    }

    fn backend(kinds: Vec<BackendKind>) -> Backend {
        Backend {
            name: "b".into(),
            kinds,
            base_url: url::Url::parse("http://example.com").unwrap(),
            api_key: Some("sk-secret".into()),
            anthropic_version: "2023-06-01".into(),
            inject_usage: true,
            class: crate::catalog::SchedulingClass::Unrestricted,
            group: None,
            max_concurrency: None,
            max_tokens_cap: None,
            tok_s_estimate: None,
        }
    }

    #[test]
    fn required_kind_maps_paths() {
        assert_eq!(required_kind("/v1/messages"), BackendKind::Anthropic);
        assert_eq!(required_kind("/v1/chat/completions"), BackendKind::Openai);
        assert_eq!(required_kind("/v1/responses"), BackendKind::Openai);
        assert_eq!(required_kind("/v1/embeddings"), BackendKind::Openai);
        assert_eq!(
            required_kind("/v1/audio/transcriptions"),
            BackendKind::Openai
        );
    }

    #[test]
    fn headers_openai_injects_bearer_and_strips_client_auth() {
        let mut inbound = HeaderMap::new();
        inbound.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer sk-client"),
        );
        inbound.insert("x-api-key", HeaderValue::from_static("sk-client"));
        inbound.insert(header::ACCEPT_ENCODING, HeaderValue::from_static("gzip"));
        inbound.insert(header::HOST, HeaderValue::from_static("router.local"));

        let out = upstream_headers(
            &inbound,
            &backend(vec![BackendKind::Openai]),
            BackendKind::Openai,
        );
        assert_eq!(out.get(header::AUTHORIZATION).unwrap(), "Bearer sk-secret");
        assert!(out.get("x-api-key").is_none());
        assert!(out.get(header::ACCEPT_ENCODING).is_none());
        assert!(out.get(header::HOST).is_none());
        assert_eq!(out.get(header::CONTENT_TYPE).unwrap(), "application/json");
    }

    #[test]
    fn headers_forward_the_client_user_agent() {
        let mut inbound = HeaderMap::new();
        inbound.insert(
            header::USER_AGENT,
            HeaderValue::from_static("OpenAI/Go 3.41.0"),
        );
        inbound.insert("x-stainless-lang", HeaderValue::from_static("go"));
        inbound.insert(
            "x-stainless-package-version",
            HeaderValue::from_static("3.41.0"),
        );
        inbound.insert("x-not-stainless", HeaderValue::from_static("nope"));
        let out = upstream_headers(
            &inbound,
            &backend(vec![BackendKind::Openai]),
            BackendKind::Openai,
        );
        assert_eq!(out.get(header::USER_AGENT).unwrap(), "OpenAI/Go 3.41.0");
        assert_eq!(out.get("x-stainless-lang").unwrap(), "go");
        assert_eq!(out.get("x-stainless-package-version").unwrap(), "3.41.0");
        assert!(
            out.get("x-not-stainless").is_none(),
            "only the x-stainless- prefix passes"
        );
        // No client UA means none is invented.
        let out = upstream_headers(
            &HeaderMap::new(),
            &backend(vec![BackendKind::Openai]),
            BackendKind::Openai,
        );
        assert!(out.get(header::USER_AGENT).is_none());
    }

    #[test]
    fn service_tier_records_only_the_notable_tiers() {
        let tier = |v: &str| {
            let mut h = HeaderMap::new();
            h.insert("x-service-tier", HeaderValue::from_str(v).unwrap());
            service_tier(&h)
        };

        assert_eq!(tier("flex").as_deref(), Some("flex"));
        // Case and surrounding whitespace are the client's, not ours.
        assert_eq!(tier("  FLEX ").as_deref(), Some("flex"));
        assert_eq!(tier("priority").as_deref(), Some("priority"));
        assert_eq!(tier("scale").as_deref(), Some("scale"));

        // `auto` and `default` are what saying nothing gets you, so they
        // are recorded as nothing.
        assert_eq!(tier("auto"), None);
        assert_eq!(tier("Default"), None);
        assert_eq!(tier("   "), None);
        assert_eq!(service_tier(&HeaderMap::new()), None);

        // Client-controlled and headed for the database: a nonsense value
        // is kept (it is real, and worth seeing in the UI), an oversized
        // one is dropped rather than half-stored.
        assert_eq!(tier("made-up-tier").as_deref(), Some("made-up-tier"));
        assert_eq!(
            tier(&"x".repeat(32)).as_deref(),
            Some("x".repeat(32).as_str())
        );
        assert_eq!(tier(&"x".repeat(33)), None);
    }

    #[test]
    fn headers_forward_the_service_tier_hint() {
        let mut inbound = HeaderMap::new();
        inbound.insert("x-service-tier", HeaderValue::from_static("priority"));
        for kind in [BackendKind::Openai, BackendKind::Anthropic] {
            let out = upstream_headers(&inbound, &backend(vec![kind]), kind);
            assert_eq!(out.get("x-service-tier").unwrap(), "priority");
        }
        let out = upstream_headers(
            &HeaderMap::new(),
            &backend(vec![BackendKind::Openai]),
            BackendKind::Openai,
        );
        assert!(out.get("x-service-tier").is_none());
    }

    #[test]
    fn headers_anthropic_injects_key_and_version() {
        let inbound = HeaderMap::new();
        let out = upstream_headers(
            &inbound,
            &backend(vec![BackendKind::Anthropic]),
            BackendKind::Anthropic,
        );
        assert_eq!(out.get("x-api-key").unwrap(), "sk-secret");
        assert_eq!(out.get("anthropic-version").unwrap(), "2023-06-01");
        assert!(out.get(header::AUTHORIZATION).is_none());
    }

    #[test]
    fn headers_dual_kind_backend_follow_endpoint_kind() {
        let inbound = HeaderMap::new();
        let b = backend(vec![BackendKind::Openai, BackendKind::Anthropic]);
        let out = upstream_headers(&inbound, &b, BackendKind::Openai);
        assert_eq!(out.get(header::AUTHORIZATION).unwrap(), "Bearer sk-secret");
        assert!(out.get("x-api-key").is_none());
        let out = upstream_headers(&inbound, &b, BackendKind::Anthropic);
        assert_eq!(out.get("x-api-key").unwrap(), "sk-secret");
        assert_eq!(out.get("anthropic-version").unwrap(), "2023-06-01");
        assert!(out.get(header::AUTHORIZATION).is_none());
    }

    #[test]
    fn headers_keyless_backend_sends_no_auth() {
        let mut inbound = HeaderMap::new();
        inbound.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer sk-client"),
        );
        let mut b = backend(vec![BackendKind::Openai]);
        b.api_key = None;
        let out = upstream_headers(&inbound, &b, BackendKind::Openai);
        assert!(out.get(header::AUTHORIZATION).is_none());
        assert!(out.get("x-api-key").is_none());
    }

    #[test]
    fn headers_anthropic_client_version_wins() {
        let mut inbound = HeaderMap::new();
        inbound.insert("anthropic-version", HeaderValue::from_static("2024-01-01"));
        inbound.insert("anthropic-beta", HeaderValue::from_static("some-beta"));
        let out = upstream_headers(
            &inbound,
            &backend(vec![BackendKind::Anthropic]),
            BackendKind::Anthropic,
        );
        assert_eq!(out.get("anthropic-version").unwrap(), "2024-01-01");
        assert_eq!(out.get("anthropic-beta").unwrap(), "some-beta");
    }
}
