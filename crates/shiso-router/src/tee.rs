//! Response-stream tee: relays upstream bytes to the client untouched
//! while extracting token usage, finish reason, and error details on the
//! side, then emits a request-log row when the stream ends (or the client
//! disconnects).

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;

use bytes::Bytes;
use futures_util::Stream;
use serde_json::Value;

use crate::active::ActiveGuard;
use crate::balancer::Pricing;
use crate::limits::RequestUsage;
use crate::logging::{LogRow, LogSink};
use crate::sched::SlotGuard;
use crate::tokens::KeyRecord;
use crate::trace::Tracer;

/// Which response dialect to parse, derived from the request path (each
/// path has one wire format regardless of which backend serves it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dialect {
    OpenAiChat,
    OpenAiResponses,
    OpenAiEmbeddings,
    OpenAiTranscriptions,
    /// The decision-scoring endpoints (`/v1/decisions`, `/v1/systemone`):
    /// one prefill per question, no generated text. Their usage objects
    /// spell the counts `prompt_tokens`/`completion_tokens` (decisions) and
    /// `input_tokens`/`output_tokens` (systemone), both of which
    /// `apply_openai_usage` already reads.
    Scoring,
    Anthropic,
}

impl Dialect {
    pub(crate) fn for_path(path: &str) -> Dialect {
        if path.ends_with("/v1/messages") {
            Dialect::Anthropic
        } else if path.ends_with("/v1/responses") {
            Dialect::OpenAiResponses
        } else if path.ends_with("/v1/embeddings") {
            Dialect::OpenAiEmbeddings
        } else if path.ends_with("/v1/audio/transcriptions") {
            Dialect::OpenAiTranscriptions
        } else if path.ends_with("/v1/decisions") || path.ends_with("/v1/systemone") {
            Dialect::Scoring
        } else {
            Dialect::OpenAiChat
        }
    }
}

/// Everything the tee needs to finish a log row once the response is done.
pub(crate) struct LogContext {
    pub row: LogRow,
    pub started: Instant,
    pub pricing: Option<Pricing>,
    pub dialect: Dialect,
    /// The key that made the request, for limit accrual at finalize time.
    pub key: Arc<KeyRecord>,
    /// Set when the request asked for `X-Debug: trace`; body chunks and the
    /// stream summary are printed through it.
    pub trace: Option<Arc<Tracer>>,
    /// The backend slot (and, for a grouped backend, the group's in-flight
    /// count) this request holds. It rides here purely for its lifetime:
    /// `TeeState` being dropped — on end of stream, on error, or when the
    /// client disconnects — is what releases it, so there is no explicit
    /// release call and no path that can skip one.
    pub slot: Option<SlotGuard>,
    /// This request's entry in the in-flight registry, riding here for the
    /// same reason `slot` does: dropping `TeeState` is what deregisters it,
    /// so the active panel empties on completion, error and client
    /// disconnect alike with no explicit removal call.
    pub active: Option<ActiveGuard>,
}

/// Normalized usage extracted from a response. `prompt_tokens` always
/// includes the cached portions (OpenAI reports it that way; Anthropic's
/// separate cache fields are summed in), so `cached_tokens` and
/// `cache_write_tokens` are subsets of it uniformly across providers.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Usage {
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
    pub cache_write_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub finish_reason: Option<String>,
    pub error: Option<String>,
}

const MAX_SSE_LINE: usize = 64 * 1024;
const MAX_JSON_BUFFER: usize = 4 * 1024 * 1024;
/// Size of each edge window kept by `Mode::Edges`. Only needs to cover the
/// non-payload part of the body (`usage` plus a few scalar fields), so a
/// small window is plenty — and it caps memory regardless of body size.
const MAX_EDGE_WINDOW: usize = 64 * 1024;
const ERROR_SNIPPET_BYTES: usize = 500;

/// Cheap substring gate before JSON-parsing an SSE data line; skips the
/// bulk of content chunks (Anthropic text deltas, [DONE], comments).
const PRE_FILTER: &[&str] = &[
    "usage",
    "finish_reason",
    "stop_reason",
    "message_start",
    "message_delta",
    "response.completed",
    "response.incomplete",
    "response.failed",
];

enum Mode {
    Sse {
        partial: Vec<u8>,
        skipping_oversized: bool,
    },
    Json {
        buf: Vec<u8>,
        truncated: bool,
    },
    /// Keeps only the first and last `MAX_EDGE_WINDOW` bytes. Used for
    /// embeddings, whose bodies are mostly vector data (a 2048-input batch
    /// runs to tens of MB) — buffering one whole would blow past
    /// `MAX_JSON_BUFFER` and lose usage entirely. Both edges are kept
    /// because JSON key order is not guaranteed: OpenAI and OpenRouter put
    /// `usage` after `data`, but a provider is free to emit it first.
    Edges {
        head: Vec<u8>,
        tail: Vec<u8>,
        overflowed: bool,
    },
    /// Parsing disabled (e.g. content-encoding set) — usage stays unknown.
    Skip,
}

/// Raw values as extracted per dialect; normalization happens in `finish`.
#[derive(Default)]
struct Extracted {
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    cached_tokens: Option<i64>,
    cache_write_tokens: Option<i64>,
    finish_reason: Option<String>,
    body_error: Option<String>,
}

pub(crate) struct UsageAccumulator {
    dialect: Dialect,
    mode: Mode,
    ext: Extracted,
    saw_token: bool,
}

impl UsageAccumulator {
    pub(crate) fn new(dialect: Dialect, is_sse: bool, content_encoded: bool) -> Self {
        let (mode, saw_token) = if content_encoded {
            (Mode::Skip, true)
        } else if is_sse {
            (
                Mode::Sse {
                    partial: Vec::new(),
                    skipping_oversized: false,
                },
                false,
            )
        } else if dialect == Dialect::OpenAiEmbeddings {
            (
                Mode::Edges {
                    head: Vec::new(),
                    tail: Vec::new(),
                    overflowed: false,
                },
                true,
            )
        } else {
            (
                Mode::Json {
                    buf: Vec::new(),
                    truncated: false,
                },
                true,
            )
        };
        UsageAccumulator {
            dialect,
            mode,
            ext: Extracted::default(),
            saw_token,
        }
    }

    /// Whether the stream has produced its first generated token or terminal event.
    #[must_use]
    pub(crate) fn saw_token(&self) -> bool {
        self.saw_token
    }

    pub(crate) fn push(&mut self, chunk: &[u8]) {
        match &mut self.mode {
            Mode::Skip => {}
            Mode::Json { buf, truncated } => {
                if *truncated {
                    return;
                }
                if buf.len() + chunk.len() > MAX_JSON_BUFFER {
                    *truncated = true;
                    buf.clear();
                    return;
                }
                buf.extend_from_slice(chunk);
            }
            Mode::Edges {
                head,
                tail,
                overflowed,
            } => {
                if head.len() < MAX_EDGE_WINDOW {
                    let take = (MAX_EDGE_WINDOW - head.len()).min(chunk.len());
                    head.extend_from_slice(&chunk[..take]);
                }
                tail.extend_from_slice(chunk);
                if tail.len() > MAX_EDGE_WINDOW {
                    tail.drain(..tail.len() - MAX_EDGE_WINDOW);
                    *overflowed = true;
                }
            }
            Mode::Sse {
                partial,
                skipping_oversized,
            } => {
                let mut rest = chunk;
                while let Some(pos) = rest.iter().position(|&b| b == b'\n') {
                    let (head, tail) = rest.split_at(pos);
                    if *skipping_oversized {
                        *skipping_oversized = false;
                    } else if partial.is_empty() {
                        process_data_line(self.dialect, &mut self.ext, &mut self.saw_token, head);
                    } else {
                        partial.extend_from_slice(head);
                        let line = std::mem::take(partial);
                        process_data_line(self.dialect, &mut self.ext, &mut self.saw_token, &line);
                    }
                    rest = &tail[1..];
                }
                if !rest.is_empty() && !*skipping_oversized {
                    if partial.len() + rest.len() > MAX_SSE_LINE {
                        // Give up on this line; scan for the next newline.
                        partial.clear();
                        *skipping_oversized = true;
                    } else {
                        partial.extend_from_slice(rest);
                    }
                }
            }
        }
    }

    /// Consumes the accumulator and produces the normalized usage.
    /// `status_2xx` gates error-body extraction for JSON responses.
    pub(crate) fn finish(mut self, status_2xx: bool) -> Usage {
        match &mut self.mode {
            Mode::Skip => {}
            Mode::Sse {
                partial,
                skipping_oversized,
            } => {
                if !*skipping_oversized && !partial.is_empty() {
                    let line = std::mem::take(partial);
                    process_data_line(self.dialect, &mut self.ext, &mut self.saw_token, &line);
                }
            }
            Mode::Json { buf, truncated } => {
                if !*truncated {
                    apply_whole_body(self.dialect, &mut self.ext, buf, status_2xx);
                }
            }
            Mode::Edges {
                head,
                tail,
                overflowed,
            } => {
                if !*overflowed {
                    // The window never filled, so `tail` is the whole body.
                    apply_whole_body(self.dialect, &mut self.ext, tail, status_2xx);
                } else if let Some(usage) =
                    last_usage_object(tail).or_else(|| last_usage_object(head))
                {
                    // Error bodies are small and land in the branch above,
                    // so an overflowed buffer means a successful response;
                    // only usage is worth recovering.
                    apply_openai_usage(&mut self.ext, &usage);
                }
            }
        }

        let ext = self.ext;
        // Anthropic's input_tokens excludes the cache fields; normalize to
        // "total input including cached" so all providers read the same.
        let prompt_tokens = match self.dialect {
            Dialect::Anthropic => {
                match (ext.input_tokens, ext.cached_tokens, ext.cache_write_tokens) {
                    (None, None, None) => None,
                    (i, c, w) => Some(i.unwrap_or(0) + c.unwrap_or(0) + w.unwrap_or(0)),
                }
            }
            _ => ext.input_tokens,
        };
        let total_tokens = match (prompt_tokens, ext.output_tokens) {
            (None, None) => None,
            (p, o) => Some(p.unwrap_or(0) + o.unwrap_or(0)),
        };
        Usage {
            prompt_tokens,
            completion_tokens: ext.output_tokens,
            cached_tokens: ext.cached_tokens,
            cache_write_tokens: ext.cache_write_tokens,
            total_tokens,
            finish_reason: ext.finish_reason,
            error: ext.body_error,
        }
    }
}

/// Parses a complete non-SSE response body: usage plus, for a body that
/// isn't the expected JSON shape, an error message or raw snippet.
fn apply_whole_body(dialect: Dialect, ext: &mut Extracted, buf: &[u8], status_2xx: bool) {
    if buf.is_empty() {
        return;
    }
    if let Ok(v) = serde_json::from_slice::<Value>(buf) {
        apply_event(dialect, ext, &v);
        if let Some(msg) = v["error"]["message"].as_str() {
            ext.body_error = Some(msg.to_string());
        }
    } else if !status_2xx {
        ext.body_error = Some(body_snippet(buf));
    }
}

/// Recovers the `usage` object from one edge window of a body too large to
/// buffer whole, by taking the last `"usage"` key in the window and brace
/// matching from the `{` that follows. Nesting-only matching is safe here
/// because every field of an OpenAI/OpenRouter `usage` object is a number,
/// a bool, or a nested object (`prompt_tokens_details`, `cost_details`) —
/// no string value can smuggle in an unbalanced brace.
fn last_usage_object(window: &[u8]) -> Option<Value> {
    const KEY: &[u8] = b"\"usage\"";
    let key_at = window.windows(KEY.len()).rposition(|w| w == KEY)?;
    let after = &window[key_at + KEY.len()..];

    let colon = after.iter().position(|b| !b.is_ascii_whitespace())?;
    if after[colon] != b':' {
        return None;
    }
    let value = &after[colon + 1..];
    let open = value.iter().position(|b| !b.is_ascii_whitespace())?;
    if value[open] != b'{' {
        return None;
    }

    let obj = &value[open..];
    let mut depth = 0usize;
    for (i, &b) in obj.iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_slice(&obj[..=i]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether an SSE `data:` payload is an empty prologue or heartbeat frame
/// that carries no generated tokens or completion signals.
///
/// Some streaming backends emit an initial frame (such as an empty role prologue,
/// `message_start`, or `response.created`) immediately upon admitting a request,
/// before prompt prefill starts. Treating that frame as the first token produces
/// an artificially low TTFB (recording scheduler admission rather than first
/// token generation).
fn is_prologue(dialect: Dialect, data: &str) -> bool {
    let trimmed = data.trim();
    if trimmed.is_empty() {
        return true;
    }
    if trimmed == "[DONE]" {
        return false;
    }
    let Ok(v) = serde_json::from_str::<Value>(trimmed) else {
        return false;
    };
    if v.get("error").is_some() {
        return false;
    }
    match dialect {
        Dialect::OpenAiChat => {
            if let Some(choices) = v["choices"].as_array() {
                if choices.is_empty() {
                    return v.get("usage").is_none();
                }
                for c in choices {
                    if c.get("finish_reason").is_some_and(|f| !f.is_null()) {
                        return false;
                    }
                    if c["text"].as_str().is_some_and(|text| !text.is_empty()) {
                        return false;
                    }
                    if let Some(delta) = c["delta"].as_object() {
                        if delta
                            .get("content")
                            .and_then(Value::as_str)
                            .is_some_and(|s| !s.is_empty())
                        {
                            return false;
                        }
                        if delta
                            .get("reasoning_content")
                            .and_then(Value::as_str)
                            .is_some_and(|s| !s.is_empty())
                        {
                            return false;
                        }
                        if delta
                            .get("thinking")
                            .and_then(Value::as_str)
                            .is_some_and(|s| !s.is_empty())
                        {
                            return false;
                        }
                        if delta
                            .get("tool_calls")
                            .and_then(Value::as_array)
                            .is_some_and(|a| !a.is_empty())
                        {
                            return false;
                        }
                    }
                }
                true
            } else {
                false
            }
        }
        Dialect::Anthropic => matches!(v["type"].as_str(), Some("message_start" | "ping")),
        Dialect::OpenAiResponses => {
            matches!(
                v["type"].as_str(),
                Some("response.created" | "response.in_progress")
            )
        }
        _ => false,
    }
}

fn process_data_line(dialect: Dialect, ext: &mut Extracted, saw_token: &mut bool, line: &[u8]) {
    let line = match line.last() {
        Some(b'\r') => &line[..line.len() - 1],
        _ => line,
    };
    let Some(data) = line.strip_prefix(b"data:") else {
        return;
    };
    let Ok(data) = std::str::from_utf8(data) else {
        return;
    };
    let data = data.trim_start();
    if !*saw_token && !is_prologue(dialect, data) {
        *saw_token = true;
    }
    if !PRE_FILTER.iter().any(|t| data.contains(t)) {
        return;
    }
    let Ok(v) = serde_json::from_str::<Value>(data) else {
        return;
    };
    apply_event(dialect, ext, &v);
}

fn apply_event(dialect: Dialect, ext: &mut Extracted, v: &Value) {
    match dialect {
        Dialect::OpenAiChat => {
            if let Some(choices) = v["choices"].as_array() {
                for c in choices {
                    if let Some(fr) = c["finish_reason"].as_str() {
                        ext.finish_reason = Some(fr.to_string());
                    }
                }
            }
            apply_openai_usage(ext, &v["usage"]);
        }
        Dialect::OpenAiResponses => {
            // Streaming events wrap the response object; the non-streaming
            // body IS the response object.
            let resp = if v["response"].is_object() {
                &v["response"]
            } else {
                v
            };
            if let Some(status) = resp["status"].as_str() {
                ext.finish_reason = Some(status.to_string());
            }
            apply_openai_usage(ext, &resp["usage"]);
        }
        // Embeddings report only `prompt_tokens`/`total_tokens` and have no
        // notion of a finish reason.
        Dialect::OpenAiEmbeddings => apply_openai_usage(ext, &v["usage"]),
        // Token-billed models (`gpt-4o-transcribe`, …) report
        // `input_tokens`/`output_tokens` in the JSON body or on the stream's
        // `transcript.text.done` event. `whisper-1` reports
        // `{"type":"duration","seconds":…}` instead, which has no token
        // counts to record; `text`/`srt`/`vtt` bodies carry no usage at all.
        Dialect::OpenAiTranscriptions => apply_openai_usage(ext, &v["usage"]),
        // Scoring replies have no finish reason: the answers are the reply.
        Dialect::Scoring => apply_openai_usage(ext, &v["usage"]),
        Dialect::Anthropic => match v["type"].as_str() {
            Some("message_start") => {
                apply_anthropic_usage(ext, &v["message"]["usage"]);
            }
            Some("message_delta") => {
                if let Some(sr) = v["delta"]["stop_reason"].as_str() {
                    ext.finish_reason = Some(sr.to_string());
                }
                apply_anthropic_usage(ext, &v["usage"]);
            }
            _ => {
                // Non-streaming message body.
                if let Some(sr) = v["stop_reason"].as_str() {
                    ext.finish_reason = Some(sr.to_string());
                }
                apply_anthropic_usage(ext, &v["usage"]);
            }
        },
    }
}

fn apply_openai_usage(ext: &mut Extracted, u: &Value) {
    if !u.is_object() {
        return;
    }
    // Chat uses prompt/completion_tokens; Responses uses input/output_tokens.
    // Both report totals that already include the cached subset.
    if let Some(n) = u["prompt_tokens"]
        .as_i64()
        .or_else(|| u["input_tokens"].as_i64())
    {
        ext.input_tokens = Some(n);
    }
    if let Some(n) = u["completion_tokens"]
        .as_i64()
        .or_else(|| u["output_tokens"].as_i64())
    {
        ext.output_tokens = Some(n);
    }
    if let Some(n) = u["prompt_tokens_details"]["cached_tokens"]
        .as_i64()
        .or_else(|| u["input_tokens_details"]["cached_tokens"].as_i64())
    {
        ext.cached_tokens = Some(n);
    }
}

fn apply_anthropic_usage(ext: &mut Extracted, u: &Value) {
    if !u.is_object() {
        return;
    }
    if let Some(n) = u["input_tokens"].as_i64() {
        ext.input_tokens = Some(n);
    }
    // Cumulative in message_delta — the last value wins.
    if let Some(n) = u["output_tokens"].as_i64() {
        ext.output_tokens = Some(n);
    }
    if let Some(n) = u["cache_read_input_tokens"].as_i64() {
        ext.cached_tokens = Some(n);
    }
    if let Some(n) = u["cache_creation_input_tokens"].as_i64() {
        ext.cache_write_tokens = Some(n);
    }
}

fn body_snippet(body: &[u8]) -> String {
    let end = body.len().min(ERROR_SNIPPET_BYTES);
    String::from_utf8_lossy(&body[..end]).into_owned()
}

struct TeeState {
    acc: UsageAccumulator,
    ctx: LogContext,
    sink: LogSink,
    first_byte_ms: Option<i64>,
}

/// Wraps the upstream byte stream: bytes pass through unchanged; the log
/// row is finalized and sent exactly once — on stream end, on upstream
/// error, or on drop (client disconnect).
pub(crate) struct TeeStream<S> {
    inner: S,
    state: Option<TeeState>,
    /// Upstream Content-Length, when framed. For such responses hyper can
    /// finish writing without ever polling the trailing `None`, so drop
    /// alone does not imply a client disconnect — the byte count decides.
    expected_len: Option<u64>,
    forwarded: u64,
}

impl<S> TeeStream<S> {
    pub(crate) fn new(
        inner: S,
        acc: UsageAccumulator,
        ctx: LogContext,
        sink: LogSink,
        expected_len: Option<u64>,
    ) -> Self {
        TeeStream {
            inner,
            state: Some(TeeState {
                acc,
                ctx,
                sink,
                first_byte_ms: None,
            }),
            expected_len,
            forwarded: 0,
        }
    }

    fn finalize(&mut self, stream_error: Option<String>) {
        let Some(TeeState {
            acc,
            mut ctx,
            sink,
            first_byte_ms,
        }) = self.state.take()
        else {
            return;
        };
        let forwarded = self.forwarded;
        let status_2xx = ctx.row.status.is_some_and(|s| (200..300).contains(&s));
        let usage = acc.finish(status_2xx);

        ctx.row.duration_ms = Some(ctx.started.elapsed().as_millis() as i64);
        if ctx.row.ttfb_ms.is_none() {
            ctx.row.ttfb_ms = first_byte_ms.or(ctx.row.duration_ms);
        }
        ctx.row.prompt_tokens = usage.prompt_tokens;
        ctx.row.completion_tokens = usage.completion_tokens;
        ctx.row.cached_tokens = usage.cached_tokens;
        ctx.row.cache_write_tokens = usage.cache_write_tokens;
        ctx.row.total_tokens = usage.total_tokens;
        ctx.row.finish_reason = usage.finish_reason;
        ctx.row.cost = match (
            &ctx.pricing,
            usage.prompt_tokens.or(usage.completion_tokens),
        ) {
            (Some(p), Some(_)) => Some(p.cost_usd(
                usage.prompt_tokens.unwrap_or(0),
                usage.completion_tokens.unwrap_or(0),
                usage.cached_tokens.unwrap_or(0),
                usage.cache_write_tokens.unwrap_or(0),
            )),
            _ => None,
        };

        let mut errors: Vec<String> = Vec::new();
        if let Some(e) = ctx.row.error.take() {
            errors.push(e);
        }
        if let Some(e) = usage.error {
            errors.push(e);
        }
        if let Some(e) = stream_error {
            errors.push(e);
        }
        if !errors.is_empty() {
            ctx.row.error = Some(errors.join("; "));
        }

        if let Some(t) = &ctx.trace {
            let r = &ctx.row;
            t.event(
                "response complete",
                &format!(
                    "{forwarded} body bytes  streamed={}  ttfb={:?}ms  duration={:?}ms\n\
                     prompt={:?} completion={:?} cached={:?} cache_write={:?} total={:?} cost={:?}\n\
                     finish_reason={:?} error={:?}",
                    r.streamed,
                    r.ttfb_ms,
                    r.duration_ms,
                    r.prompt_tokens,
                    r.completion_tokens,
                    r.cached_tokens,
                    r.cache_write_tokens,
                    r.total_tokens,
                    r.cost,
                    r.finish_reason,
                    r.error,
                ),
            );
        }

        // Feed the backend's throughput estimator before the slot goes: the
        // token count and the elapsed time both only exist here, and the
        // next request's wait estimate is built from them.
        if let Some(slot) = &ctx.slot {
            slot.observe_completion(ctx.row.completion_tokens);
        }

        sink.accrue(
            &ctx.key,
            RequestUsage {
                cost: ctx.row.cost.unwrap_or(0.0),
                total_tokens: ctx.row.total_tokens.unwrap_or(0),
                prompt_tokens: ctx.row.prompt_tokens.unwrap_or(0),
                completion_tokens: ctx.row.completion_tokens.unwrap_or(0),
            },
        );
        sink.send(ctx.row);
    }
}

impl<S> Stream for TeeStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                this.forwarded += chunk.len() as u64;
                if let Some(state) = &mut this.state {
                    if state.first_byte_ms.is_none() {
                        state.first_byte_ms = Some(state.ctx.started.elapsed().as_millis() as i64);
                    }
                    state.acc.push(&chunk);
                    if state.ctx.row.ttfb_ms.is_none() && state.acc.saw_token() {
                        state.ctx.row.ttfb_ms =
                            Some(state.ctx.started.elapsed().as_millis() as i64);
                    }
                    if let Some(t) = &state.ctx.trace {
                        t.body_chunk(&chunk);
                    }
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                this.finalize(Some(format!("upstream stream error: {e}")));
                Poll::Ready(Some(Err(std::io::Error::other(e))))
            }
            Poll::Ready(None) => {
                this.finalize(None);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S> Drop for TeeStream<S> {
    fn drop(&mut self) {
        // Still armed here means poll_next never saw the end of stream.
        // With a known Content-Length that's normal completion once every
        // byte has been forwarded; otherwise the client disconnected.
        let complete = self.expected_len.is_some_and(|n| self.forwarded >= n);
        self.finalize(if complete {
            None
        } else {
            Some("client disconnected".into())
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(dialect: Dialect, is_sse: bool, chunks: &[&[u8]]) -> Usage {
        let mut acc = UsageAccumulator::new(dialect, is_sse, false);
        for c in chunks {
            acc.push(c);
        }
        acc.finish(true)
    }

    #[test]
    fn openai_chat_sse_with_usage_chunk() {
        let usage = feed(
            Dialect::OpenAiChat,
            true,
            &[
                b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
                b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":20,\"prompt_tokens_details\":{\"cached_tokens\":60}}}\n\n",
                b"data: [DONE]\n\n",
            ],
        );
        assert_eq!(usage.prompt_tokens, Some(100));
        assert_eq!(usage.completion_tokens, Some(20));
        assert_eq!(usage.cached_tokens, Some(60));
        assert_eq!(usage.cache_write_tokens, None);
        assert_eq!(usage.total_tokens, Some(120));
        assert_eq!(usage.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn sse_data_line_split_across_chunks() {
        let usage = feed(
            Dialect::OpenAiChat,
            true,
            &[
                b"data: {\"choices\":[],\"usage\":{\"prompt_to",
                b"kens\":7,\"completion_tokens\":3}}\n\n",
            ],
        );
        assert_eq!(usage.prompt_tokens, Some(7));
        assert_eq!(usage.completion_tokens, Some(3));
        assert_eq!(usage.total_tokens, Some(10));
    }

    #[test]
    fn sse_final_line_without_trailing_newline_is_processed() {
        let usage = feed(
            Dialect::OpenAiChat,
            true,
            &[b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2}}"],
        );
        assert_eq!(usage.total_tokens, Some(3));
    }

    #[test]
    fn oversized_sse_line_is_skipped_without_breaking_later_lines() {
        let mut acc = UsageAccumulator::new(Dialect::OpenAiChat, true, false);
        // A giant line that mentions "usage" but never completes within cap.
        acc.push(b"data: {\"usage\":\"");
        let filler = vec![b'x'; MAX_SSE_LINE + 100];
        acc.push(&filler);
        acc.push(b"\"}\n");
        acc.push(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":5}}\n",
        );
        let usage = acc.finish(true);
        assert_eq!(usage.total_tokens, Some(10));
    }

    #[test]
    fn anthropic_sse_message_start_and_delta() {
        let usage = feed(
            Dialect::Anthropic,
            true,
            &[
                b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":40,\"cache_creation_input_tokens\":8,\"output_tokens\":1}}}\n\n",
                b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
                b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":25}}\n\n",
            ],
        );
        // input excludes cache fields upstream; normalized prompt includes them.
        assert_eq!(usage.prompt_tokens, Some(58));
        assert_eq!(
            usage.completion_tokens,
            Some(25),
            "last cumulative value wins"
        );
        assert_eq!(usage.cached_tokens, Some(40));
        assert_eq!(usage.cache_write_tokens, Some(8));
        assert_eq!(usage.total_tokens, Some(83));
        assert_eq!(usage.finish_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn openai_responses_sse_completed_event() {
        let usage = feed(
            Dialect::OpenAiResponses,
            true,
            &[
                b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
                b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":50,\"output_tokens\":9,\"input_tokens_details\":{\"cached_tokens\":30}}}}\n\n",
            ],
        );
        assert_eq!(usage.prompt_tokens, Some(50));
        assert_eq!(usage.completion_tokens, Some(9));
        assert_eq!(usage.cached_tokens, Some(30));
        assert_eq!(usage.finish_reason.as_deref(), Some("completed"));
    }

    #[test]
    fn openai_chat_json_body() {
        let usage = feed(
            Dialect::OpenAiChat,
            false,
            &[br#"{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":4}}"#],
        );
        assert_eq!(usage.prompt_tokens, Some(11));
        assert_eq!(usage.completion_tokens, Some(4));
        assert_eq!(usage.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn anthropic_json_body_with_cache_fields() {
        let usage = feed(
            Dialect::Anthropic,
            false,
            &[br#"{"type":"message","stop_reason":"end_turn","usage":{"input_tokens":5,"cache_read_input_tokens":100,"cache_creation_input_tokens":20,"output_tokens":7}}"#],
        );
        assert_eq!(usage.prompt_tokens, Some(125));
        assert_eq!(usage.cached_tokens, Some(100));
        assert_eq!(usage.cache_write_tokens, Some(20));
        assert_eq!(usage.total_tokens, Some(132));
        assert_eq!(usage.finish_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn responses_json_body_top_level() {
        let usage = feed(
            Dialect::OpenAiResponses,
            false,
            &[br#"{"id":"resp_1","status":"completed","usage":{"input_tokens":8,"output_tokens":2}}"#],
        );
        assert_eq!(usage.prompt_tokens, Some(8));
        assert_eq!(usage.completion_tokens, Some(2));
        assert_eq!(usage.finish_reason.as_deref(), Some("completed"));
    }

    #[test]
    fn embeddings_json_body() {
        let usage = feed(
            Dialect::OpenAiEmbeddings,
            false,
            &[br#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,-0.2]}],"model":"text-embedding-3-small","usage":{"prompt_tokens":8,"total_tokens":8}}"#],
        );
        assert_eq!(usage.prompt_tokens, Some(8));
        assert_eq!(usage.completion_tokens, None);
        assert_eq!(usage.total_tokens, Some(8));
        assert_eq!(usage.finish_reason, None);
    }

    #[test]
    fn transcription_json_body_with_token_usage() {
        let usage = feed(
            Dialect::OpenAiTranscriptions,
            false,
            &[br#"{"text":"hello there","usage":{"type":"tokens","input_tokens":14,"input_token_details":{"text_tokens":0,"audio_tokens":14},"output_tokens":45,"total_tokens":59}}"#],
        );
        assert_eq!(usage.prompt_tokens, Some(14));
        assert_eq!(usage.completion_tokens, Some(45));
        assert_eq!(usage.total_tokens, Some(59));
        assert_eq!(usage.finish_reason, None);
    }

    #[test]
    fn transcription_stream_takes_usage_from_the_done_event() {
        let usage = feed(
            Dialect::OpenAiTranscriptions,
            true,
            &[
                b"data: {\"type\":\"transcript.text.delta\",\"delta\":\"hel\"}\n\n",
                b"data: {\"type\":\"transcript.text.done\",\"text\":\"hello\",\"usage\":{\"type\":\"tokens\",\"input_tokens\":9,\"output_tokens\":3,\"total_tokens\":12}}\n\n",
            ],
        );
        assert_eq!(usage.prompt_tokens, Some(9));
        assert_eq!(usage.completion_tokens, Some(3));
    }

    /// `whisper-1` bills by duration, and `text`/`srt`/`vtt` bodies are not
    /// JSON at all: neither yields tokens, and neither is an error.
    #[test]
    fn transcription_without_token_usage_records_none() {
        let duration = feed(
            Dialect::OpenAiTranscriptions,
            false,
            &[br#"{"text":"hi","usage":{"type":"duration","seconds":3}}"#],
        );
        assert_eq!(duration, Usage::default());
        let srt = feed(
            Dialect::OpenAiTranscriptions,
            false,
            &[b"1\n00:00:00,000 --> 00:00:01,000\nhi\n"],
        );
        assert_eq!(srt, Usage::default());
    }

    /// Both scoring wires record usage; they just spell the counts
    /// differently (`prompt_tokens`/`completion_tokens` on decisions,
    /// `input_tokens`/`output_tokens` on systemone), and neither has a
    /// finish reason — the answers are the reply.
    #[test]
    fn scoring_json_bodies_record_usage_in_both_spellings() {
        let decisions = feed(
            Dialect::Scoring,
            false,
            &[br#"{"object":"decisions","answers":{"q1":{"type":"yes_no","answer":"no"}},"usage":{"prompt_tokens":42,"completion_tokens":0,"total_tokens":42}}"#],
        );
        assert_eq!(decisions.prompt_tokens, Some(42));
        assert_eq!(decisions.completion_tokens, Some(0));
        assert_eq!(decisions.total_tokens, Some(42));
        assert_eq!(decisions.finish_reason, None);

        let systemone = feed(
            Dialect::Scoring,
            false,
            &[br#"{"model":"served","answers":{},"usage":{"input_tokens":17,"output_tokens":0,"total_tokens":17}}"#],
        );
        assert_eq!(systemone.prompt_tokens, Some(17));
        assert_eq!(systemone.completion_tokens, Some(0));
        assert_eq!(systemone.total_tokens, Some(17));
    }

    /// OpenRouter's embeddings response adds `id` and enriches `usage` with
    /// `cost` / `cost_details` / `prompt_tokens_details` (a per-modality
    /// breakdown, *not* a cached-token subset).
    #[test]
    fn openrouter_embeddings_json_body() {
        let usage = feed(
            Dialect::OpenAiEmbeddings,
            false,
            &[br#"{"id":"embd-123","object":"list","data":[{"object":"embedding","index":0,"embedding":"eyJhIjoxfQ=="}],"model":"openai/text-embedding-3-small","usage":{"prompt_tokens":8,"total_tokens":8,"cost":0.0001,"is_byok":false,"prompt_tokens_details":{"text_tokens":8},"cost_details":{"upstream_inference_prompt_cost":0.0001,"upstream_inference_completions_cost":0.0}}}"#],
        );
        assert_eq!(usage.prompt_tokens, Some(8));
        assert_eq!(usage.total_tokens, Some(8));
        // prompt_tokens_details here has no cached_tokens — must not be
        // mistaken for one.
        assert_eq!(usage.cached_tokens, None);
        assert_eq!(usage.error, None);
    }

    /// A batch of embeddings easily runs to tens of MB; usage must still be
    /// recovered once the sliding windows have overflowed — from whichever
    /// end of the body the provider put it.
    #[test]
    fn oversized_embeddings_body_recovers_usage_from_either_edge() {
        let vector = format!(
            "{{\"object\":\"embedding\",\"index\":0,\"embedding\":[{}]}},",
            "0.0123456789,".repeat(1000)
        );
        let repeats = MAX_EDGE_WINDOW / vector.len() + 4;

        // usage last (OpenAI / OpenRouter ordering) -> found in the tail.
        let mut acc = UsageAccumulator::new(Dialect::OpenAiEmbeddings, false, false);
        acc.push(br#"{"object":"list","data":["#);
        for _ in 0..repeats {
            acc.push(vector.as_bytes());
        }
        acc.push(br#"],"model":"m","usage":{"prompt_tokens":300000,"total_tokens":300000}}"#);
        let usage = acc.finish(true);
        assert_eq!(usage.prompt_tokens, Some(300_000));
        assert_eq!(usage.total_tokens, Some(300_000));

        // usage first -> found in the head.
        let mut acc = UsageAccumulator::new(Dialect::OpenAiEmbeddings, false, false);
        acc.push(br#"{"usage":{"prompt_tokens":42,"total_tokens":42},"data":["#);
        for _ in 0..repeats {
            acc.push(vector.as_bytes());
        }
        acc.push(br#"],"model":"m"}"#);
        let usage = acc.finish(true);
        assert_eq!(usage.prompt_tokens, Some(42));
    }

    #[test]
    fn last_usage_object_ignores_earlier_and_malformed_matches() {
        // The response-level object wins over one embedded in the payload.
        let v = last_usage_object(
            br#"{"data":[{"usage":{"prompt_tokens":1}}],"usage":{"prompt_tokens":9}}"#,
        )
        .unwrap();
        assert_eq!(v["prompt_tokens"], 9);
        // A nested details object does not end the match early.
        let v = last_usage_object(
            br#"...,"usage": {"prompt_tokens":4,"cost_details":{"x":1},"total_tokens":4}}"#,
        )
        .unwrap();
        assert_eq!(v["total_tokens"], 4);
        assert!(last_usage_object(b"no usage here").is_none());
        assert!(last_usage_object(br#"{"usage":null}"#).is_none());
        assert!(last_usage_object(br#"{"usage":{"prompt_tokens":1"#).is_none());
    }

    #[test]
    fn error_body_message_is_extracted() {
        let mut acc = UsageAccumulator::new(Dialect::OpenAiChat, false, false);
        acc.push(br#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#);
        let usage = acc.finish(false);
        assert_eq!(usage.error.as_deref(), Some("rate limited"));
        assert_eq!(usage.prompt_tokens, None);
    }

    #[test]
    fn non_json_error_body_becomes_snippet() {
        let mut acc = UsageAccumulator::new(Dialect::OpenAiChat, false, false);
        acc.push(b"upstream exploded");
        let usage = acc.finish(false);
        assert_eq!(usage.error.as_deref(), Some("upstream exploded"));
    }

    #[test]
    fn truncated_json_body_yields_unknown_usage() {
        let mut acc = UsageAccumulator::new(Dialect::OpenAiChat, false, false);
        let big = vec![b'x'; MAX_JSON_BUFFER + 1];
        acc.push(&big);
        acc.push(br#"{"usage":{"prompt_tokens":1}}"#);
        let usage = acc.finish(true);
        assert_eq!(usage.prompt_tokens, None);
        assert_eq!(usage.total_tokens, None);
    }

    #[test]
    fn content_encoded_body_is_not_parsed() {
        let mut acc = UsageAccumulator::new(Dialect::OpenAiChat, false, true);
        acc.push(br#"{"usage":{"prompt_tokens":1,"completion_tokens":1}}"#);
        let usage = acc.finish(true);
        assert_eq!(usage.prompt_tokens, None);
    }

    #[test]
    fn crlf_lines_are_handled() {
        let usage = feed(
            Dialect::OpenAiChat,
            true,
            &[b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\r\n\r\n"],
        );
        assert_eq!(usage.total_tokens, Some(3));
    }

    #[test]
    fn openai_chat_prologue_skipped_until_token() {
        let mut acc = UsageAccumulator::new(Dialect::OpenAiChat, true, false);
        assert!(!acc.saw_token());

        // Prologue chunk: empty content with role assistant
        acc.push(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n");
        assert!(
            !acc.saw_token(),
            "empty role prologue must not be counted as a token"
        );

        // First token chunk
        acc.push(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n");
        assert!(
            acc.saw_token(),
            "first non-empty content chunk must be detected as a token"
        );
    }

    #[test]
    fn anthropic_prologue_skipped_until_token() {
        let mut acc = UsageAccumulator::new(Dialect::Anthropic, true, false);
        assert!(!acc.saw_token());

        // Prologue chunk: message_start
        acc.push(b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n");
        assert!(
            !acc.saw_token(),
            "message_start must not be counted as a token"
        );

        // First content delta
        acc.push(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n");
        assert!(
            acc.saw_token(),
            "content_block_delta must be detected as a token"
        );
    }

    #[test]
    fn responses_prologue_skipped_until_token() {
        let mut acc = UsageAccumulator::new(Dialect::OpenAiResponses, true, false);
        assert!(!acc.saw_token());

        // Prologue: response.created and response.in_progress
        acc.push(b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\"}}\n\n");
        assert!(!acc.saw_token());

        acc.push(b"event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"response\":{\"id\":\"r1\"}}\n\n");
        assert!(!acc.saw_token());

        // First token delta
        acc.push(b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n");
        assert!(
            acc.saw_token(),
            "response.output_text.delta must be detected as a token"
        );
    }

    #[test]
    fn non_sse_marks_token_immediately() {
        let acc = UsageAccumulator::new(Dialect::OpenAiChat, false, false);
        assert!(
            acc.saw_token(),
            "non-SSE response should immediately treat first chunk as token"
        );
    }
}
