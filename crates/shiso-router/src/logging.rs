use std::sync::Arc;

use anyhow::Result;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::limits::{RequestUsage, UsageTracker};
use crate::tokens::{Db, KeyRecord};

/// One request-log record, mirroring the `request_logs` table. Token
/// counts are normalized across providers: `prompt_tokens` is the total
/// input including cached portions, `cached_tokens` the cache-read subset,
/// `cache_write_tokens` Anthropic cache creation.
#[derive(Debug, Clone, Serialize)]
pub struct LogRow {
    /// Unique request identifier.
    pub id: String,
    /// Unix epoch milliseconds UTC, request start.
    pub created_at: i64,
    /// Client API token identifier.
    pub token_id: String,
    /// Requested model name.
    pub model: String,
    /// Dispatched backend name.
    pub backend: Option<String>,
    /// Dispatched upstream model name.
    pub upstream_model: Option<String>,
    /// Request endpoint path.
    pub path: String,
    /// Final HTTP response status code.
    pub status: Option<u16>,
    /// Whether response was streamed via SSE.
    pub streamed: bool,
    /// Time to first byte in milliseconds.
    pub ttfb_ms: Option<i64>,
    /// Total request duration in milliseconds.
    pub duration_ms: Option<i64>,
    /// Total prompt token count.
    pub prompt_tokens: Option<i64>,
    /// Completion token count.
    pub completion_tokens: Option<i64>,
    /// Cached prompt tokens read.
    pub cached_tokens: Option<i64>,
    /// Anthropic cache creation tokens written.
    pub cache_write_tokens: Option<i64>,
    /// Total token count.
    pub total_tokens: Option<i64>,
    /// Request cost in USD.
    pub cost: Option<f64>,
    /// Finish reason reported by upstream provider.
    pub finish_reason: Option<String>,
    /// Error message or failover trail.
    pub error: Option<String>,
    /// The client's `x-service-tier` hint, normalized by
    /// `proxy::service_tier` — `None` for ordinary traffic (no header, or
    /// the `auto`/`default` tiers).
    pub service_tier: Option<String>,
}

impl LogRow {
    /// A fresh row with attribution filled in and everything else pending.
    pub fn new(token_id: String, model: String, path: String) -> Self {
        LogRow {
            id: uuid::Uuid::new_v4().to_string(),
            created_at: now_ms(),
            token_id,
            model,
            backend: None,
            upstream_model: None,
            path,
            status: None,
            streamed: false,
            ttfb_ms: None,
            duration_ms: None,
            prompt_tokens: None,
            completion_tokens: None,
            cached_tokens: None,
            cache_write_tokens: None,
            total_tokens: None,
            cost: None,
            finish_reason: None,
            error: None,
            service_tier: None,
        }
    }
}

/// Returns current unix epoch in milliseconds.
pub fn now_ms() -> i64 {
    (time::OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000) as i64
}

/// Handle for emitting log rows from the request path. `send` never blocks:
/// on a full channel the row is dropped with a warning rather than stalling
/// a response stream. Also carries the usage tracker so stream finalization
/// can accrue limits synchronously (the DB write is what stays async).
#[derive(Debug, Clone)]
pub struct LogSink {
    tx: mpsc::Sender<LogRow>,
    usage: Arc<UsageTracker>,
}

impl LogSink {
    /// Sends a log row to the background persistence channel without blocking.
    pub fn send(&self, row: LogRow) {
        if let Err(e) = self.tx.try_send(row) {
            tracing::warn!(error = %e, "request log channel full or closed, dropping row");
        }
    }

    /// Accrues token usage synchronously against key limits.
    pub fn accrue(&self, record: &KeyRecord, usage: RequestUsage) {
        self.usage
            .accrue(record, usage, time::OffsetDateTime::now_utc());
    }

    /// Returns a reference to the active usage tracker.
    pub fn usage_tracker(&self) -> Arc<UsageTracker> {
        Arc::clone(&self.usage)
    }
}

const CHANNEL_CAPACITY: usize = 4096;
const MAX_BATCH: usize = 256;

/// Spawns the single writer task that drains the log channel and inserts
/// rows in batched transactions off the request hot path. The task exits
/// when every `LogSink` clone has dropped (i.e. at shutdown), flushing
/// whatever is queued; await the returned handle to be sure.
pub fn start_writer(db: Db) -> (LogSink, JoinHandle<()>) {
    let (tx, mut rx) = mpsc::channel::<LogRow>(CHANNEL_CAPACITY);
    let usage = Arc::new(UsageTracker::new());
    let handle = tokio::spawn(async move {
        while let Some(row) = rx.recv().await {
            let mut batch = vec![row];
            while batch.len() < MAX_BATCH {
                match rx.try_recv() {
                    Ok(row) => batch.push(row),
                    Err(_) => break,
                }
            }
            let db = db.clone();
            match tokio::task::spawn_blocking(move || insert_batch(&db, &batch)).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!(error = %e, "failed to write request logs"),
                Err(e) => tracing::warn!(error = %e, "request log writer task panicked"),
            }
        }
    });
    (LogSink { tx, usage }, handle)
}

fn insert_batch(db: &Db, rows: &[LogRow]) -> Result<()> {
    db.with_write(|conn| {
        let tx = conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO request_logs (
                    id, created_at, token_id, model, backend, upstream_model,
                    path, status, streamed, ttfb_ms, duration_ms,
                    prompt_tokens, completion_tokens, cached_tokens,
                    cache_write_tokens, total_tokens, cost, finish_reason, error,
                    service_tier
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                           ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)",
            )?;
            for row in rows {
                stmt.execute(rusqlite::params![
                    row.id,
                    row.created_at,
                    row.token_id,
                    row.model,
                    row.backend,
                    row.upstream_model,
                    row.path,
                    row.status,
                    row.streamed,
                    row.ttfb_ms,
                    row.duration_ms,
                    row.prompt_tokens,
                    row.completion_tokens,
                    row.cached_tokens,
                    row.cache_write_tokens,
                    row.total_tokens,
                    row.cost,
                    row.finish_reason,
                    row.error,
                    row.service_tier,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> LogRow {
        LogRow {
            id: id.into(),
            created_at: now_ms(),
            token_id: "tok".into(),
            model: "m".into(),
            backend: Some("b".into()),
            upstream_model: None,
            path: "/v1/chat/completions".into(),
            status: Some(200),
            streamed: false,
            ttfb_ms: Some(12),
            duration_ms: Some(34),
            prompt_tokens: Some(100),
            completion_tokens: Some(50),
            cached_tokens: Some(20),
            cache_write_tokens: None,
            total_tokens: Some(150),
            cost: Some(0.001),
            finish_reason: Some("stop".into()),
            error: None,
            service_tier: Some("flex".into()),
        }
    }

    #[tokio::test]
    async fn writer_persists_rows_and_flushes_on_close() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("t.db")).unwrap();
        let (sink, handle) = start_writer(db.clone());

        sink.send(row("a"));
        sink.send(row("b"));
        drop(sink);
        handle.await.unwrap();

        let count: i64 = db
            .with_read(|conn| {
                Ok(conn.query_row("SELECT COUNT(*) FROM request_logs", [], |r| r.get(0))?)
            })
            .unwrap();
        assert_eq!(count, 2);
        let (status, cached, tier): (Option<u16>, Option<i64>, Option<String>) = db
            .with_read(|conn| {
                Ok(conn.query_row(
                    "SELECT status, cached_tokens, service_tier FROM request_logs WHERE id = 'a'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )?)
            })
            .unwrap();
        assert_eq!(status, Some(200));
        assert_eq!(cached, Some(20));
        assert_eq!(tier.as_deref(), Some("flex"));
    }
}
