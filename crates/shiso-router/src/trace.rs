//! Opt-in per-request wire tracing.
//!
//! A request carrying `X-Debug: trace` gets its whole exchange printed to
//! stdout: the inbound request as the client sent it, the outbound request
//! as it leaves for the backend (after model rename / `include_usage`
//! injection), every upstream response head, the response body chunk by
//! chunk as it streams, and the head relayed back to the client. Response
//! bodies are never modified, so the body is printed once — what the tee
//! sees is byte-for-byte what the client gets.
//!
//! Each block is written under a single stdout lock, so concurrent traces
//! interleave between blocks but never inside one. Credential headers are
//! redacted; `x-debug` itself is not in the outbound allowlist, so it never
//! reaches the backend.

use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::http::HeaderMap;

/// Inbound header that turns tracing on, and the value it must carry.
pub(crate) const TRACE_HEADER: &str = "x-debug";
const TRACE_VALUE: &str = "trace";

/// Per-body cap on printed request bodies.
const MAX_BODY_BYTES: usize = 64 * 1024;
/// Cumulative cap on printed response-body bytes for one request; a long
/// stream stops printing rather than flooding stdout for minutes.
const MAX_STREAM_BYTES: usize = 1024 * 1024;

/// Header values never printed verbatim.
const REDACTED: &[&str] = &[
    "authorization",
    "x-api-key",
    "proxy-authorization",
    "cookie",
];

pub(crate) struct Tracer {
    /// Same id as the request's `request_logs` row, so a trace can be
    /// matched to the stored log entry.
    id: String,
    chunks: AtomicUsize,
    stream_bytes: AtomicUsize,
    stream_truncated: AtomicUsize,
}

impl Tracer {
    /// `Some` when the request asked for a trace (`X-Debug: trace`).
    pub(crate) fn from_headers(headers: &HeaderMap) -> Option<Arc<Tracer>> {
        let on = headers
            .get(TRACE_HEADER)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.trim().eq_ignore_ascii_case(TRACE_VALUE));
        on.then(|| {
            Arc::new(Tracer {
                id: uuid::Uuid::new_v4().to_string(),
                chunks: AtomicUsize::new(0),
                stream_bytes: AtomicUsize::new(0),
                stream_truncated: AtomicUsize::new(0),
            })
        })
    }

    pub(crate) fn id(&self) -> &str {
        &self.id
    }

    /// The request exactly as the client sent it.
    pub(crate) fn client_request(
        &self,
        method: &str,
        raw_path: &str,
        canonical_path: &str,
        headers: &HeaderMap,
        body: &[u8],
    ) {
        let mut out = self.header("client → router");
        let _ = writeln!(out, "{method} {raw_path}");
        if raw_path != canonical_path {
            let _ = writeln!(out, "(canonicalized to {canonical_path})");
        }
        push_headers(&mut out, headers);
        push_body(&mut out, "request body", body, MAX_BODY_BYTES);
        self.emit(out);
    }

    /// The request as it goes upstream: rewritten body (rename and/or
    /// `stream_options.include_usage`) and the allowlisted header set with
    /// the backend's own credentials.
    pub(crate) fn upstream_request(
        &self,
        backend: &str,
        url: &str,
        headers: &HeaderMap,
        body: &[u8],
        rewritten: bool,
    ) {
        let mut out = self.header(&format!("router → upstream ({backend})"));
        let _ = writeln!(out, "POST {url}");
        push_headers(&mut out, headers);
        let label = if rewritten {
            "request body (rewritten)"
        } else {
            "request body (forwarded unchanged)"
        };
        push_body(&mut out, label, body, MAX_BODY_BYTES);
        self.emit(out);
    }

    /// An upstream response head, before any failover decision.
    pub(crate) fn upstream_response(
        &self,
        backend: &str,
        status: u16,
        headers: &HeaderMap,
        elapsed_ms: u128,
    ) {
        let mut out = self.header(&format!(
            "upstream → router ({backend})  status: {status}  {elapsed_ms}ms"
        ));
        push_headers(&mut out, headers);
        self.emit(out);
    }

    /// The head handed back to the client (hop-by-hop headers dropped).
    pub(crate) fn client_response(&self, status: u16, headers: &HeaderMap) {
        let mut out = self.header(&format!("router → client  status: {status}"));
        push_headers(&mut out, headers);
        self.emit(out);
    }

    /// One response body chunk, as forwarded (unmodified) to the client.
    pub(crate) fn body_chunk(&self, chunk: &[u8]) {
        let n = self.chunks.fetch_add(1, Ordering::Relaxed) + 1;
        let printed = self.stream_bytes.fetch_add(chunk.len(), Ordering::Relaxed);
        if printed >= MAX_STREAM_BYTES {
            // Announce the cutoff once, then stay quiet.
            if self.stream_truncated.swap(1, Ordering::Relaxed) == 0 {
                self.event(
                    "body truncated",
                    &format!("over {MAX_STREAM_BYTES} bytes printed"),
                );
            }
            return;
        }
        let room = MAX_STREAM_BYTES - printed;
        let mut out = self.header(&format!("response chunk #{n} ({} bytes)", chunk.len()));
        push_raw(&mut out, chunk, room);
        self.emit(out);
    }

    /// A one-line note: failover, rejection, stream end.
    pub(crate) fn event(&self, title: &str, detail: &str) {
        let mut out = self.header(title);
        if !detail.is_empty() {
            let _ = writeln!(out, "{detail}");
        }
        self.emit(out);
    }

    fn header(&self, title: &str) -> String {
        format!("\n──── [trace {}] {title}  {}\n", self.id, now())
    }

    fn emit(&self, text: String) {
        let mut stdout = std::io::stdout().lock();
        let _ = stdout.write_all(text.as_bytes());
        let _ = stdout.flush();
    }
}

fn now() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_default()
}

fn push_headers(out: &mut String, headers: &HeaderMap) {
    for (name, value) in headers {
        let shown = if REDACTED.contains(&name.as_str()) {
            format!("[redacted, {} bytes]", value.len())
        } else {
            String::from_utf8_lossy(value.as_bytes()).into_owned()
        };
        let _ = writeln!(out, "{name}: {shown}");
    }
}

fn push_body(out: &mut String, label: &str, body: &[u8], cap: usize) {
    let _ = writeln!(out, "--- {label} ({} bytes) ---", body.len());
    push_raw(out, body, cap);
}

fn push_raw(out: &mut String, bytes: &[u8], cap: usize) {
    let end = bytes.len().min(cap);
    out.push_str(&String::from_utf8_lossy(&bytes[..end]));
    if end < bytes.len() {
        let _ = write!(out, "\n… [{} more bytes]", bytes.len() - end);
    }
    if !out.ends_with('\n') {
        out.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn only_the_trace_value_enables_tracing() {
        assert!(Tracer::from_headers(&HeaderMap::new()).is_none());
        assert!(Tracer::from_headers(&headers(&[("x-debug", "off")])).is_none());
        assert!(Tracer::from_headers(&headers(&[("x-debug", "trace")])).is_some());
        // Header names are case-insensitive; so is the value.
        assert!(Tracer::from_headers(&headers(&[("X-Debug", " TRACE ")])).is_some());
    }

    #[test]
    fn credentials_are_redacted_but_other_headers_are_not() {
        let mut out = String::new();
        push_headers(
            &mut out,
            &headers(&[
                ("authorization", "Bearer sk-secret"),
                ("content-type", "application/json"),
            ]),
        );
        assert!(!out.contains("sk-secret"), "{out}");
        assert!(out.contains("authorization: [redacted, 16 bytes]"), "{out}");
        assert!(out.contains("content-type: application/json"), "{out}");
    }

    #[test]
    fn body_is_capped_with_a_remainder_note() {
        let mut out = String::new();
        push_body(&mut out, "request body", &[b'x'; 100], 10);
        assert!(out.contains("--- request body (100 bytes) ---"), "{out}");
        assert!(out.contains("xxxxxxxxxx\n… [90 more bytes]"), "{out}");
    }
}
