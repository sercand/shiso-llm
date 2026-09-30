use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::tokens::{Db, KeyRecord};

/// How often a per-key usage window resets. All windows are UTC; weekly
/// windows start Monday 00:00 UTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResetInterval {
    #[default]
    Never,
    Daily,
    Weekly,
    Monthly,
}

impl ResetInterval {
    pub fn as_str(self) -> &'static str {
        match self {
            ResetInterval::Never => "never",
            ResetInterval::Daily => "daily",
            ResetInterval::Weekly => "weekly",
            ResetInterval::Monthly => "monthly",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "never" => Some(ResetInterval::Never),
            "daily" => Some(ResetInterval::Daily),
            "weekly" => Some(ResetInterval::Weekly),
            "monthly" => Some(ResetInterval::Monthly),
            _ => None,
        }
    }
}

/// Start of the usage window containing `now`, as unix epoch milliseconds.
/// `Never` means "since forever" (epoch 0); weekly windows start Monday
/// 00:00 UTC; monthly windows on the 1st 00:00 UTC.
pub(crate) fn window_start_ms(interval: ResetInterval, now: time::OffsetDateTime) -> i64 {
    let date = match interval {
        ResetInterval::Never => return 0,
        ResetInterval::Daily => now.date(),
        ResetInterval::Weekly => {
            let days_from_monday = now.date().weekday().number_days_from_monday() as i64;
            now.date() - time::Duration::days(days_from_monday)
        }
        ResetInterval::Monthly => now.date().replace_day(1).expect("day 1 is always valid"),
    };
    date.midnight().assume_utc().unix_timestamp() * 1000
}

/// One finished request's contribution to a key's windows (or, when read
/// back from `request_logs`, a whole window's sums).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RequestUsage {
    pub cost: f64,
    /// The provider's total; limits are enforced against this, not against
    /// prompt + completion (which can disagree with it).
    pub total_tokens: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
}

impl RequestUsage {
    fn is_empty(&self) -> bool {
        self.cost == 0.0
            && self.total_tokens == 0
            && self.prompt_tokens == 0
            && self.completion_tokens == 0
    }
}

/// Current-window usage of one key, for the admin UI's limit bars.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct UsageSnapshot {
    pub cost: f64,
    pub tokens: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
}

/// One accumulation window. Only the fields matching the window's purpose
/// are used (cost for the credit window, token counts for the token
/// window). `tokens` is the provider's total and is what limits are
/// enforced against; `prompt`/`completion` are kept for display only.
#[derive(Debug, Default)]
struct Window {
    start_ms: i64,
    cost: f64,
    tokens: i64,
    prompt: i64,
    completion: i64,
}

/// If `now` has crossed into a newer window, restart it with zeroed sums —
/// a fresh window genuinely starts empty.
fn roll(w: &mut Window, interval: ResetInterval, now: time::OffsetDateTime) {
    let start = window_start_ms(interval, now);
    if start > w.start_ms {
        *w = Window {
            start_ms: start,
            ..Window::default()
        };
    }
}

#[derive(Debug, Default)]
struct KeyUsage {
    /// Separate windows: the two limits may reset on different intervals.
    credit: Mutex<Window>,
    tokens: Mutex<Window>,
}

/// Current-window usage per key, kept in memory so the request hot path
/// never touches SQLite. Warmed from `request_logs` at startup (and after
/// limit changes); accrual happens synchronously when a response stream
/// finalizes. Concurrent in-flight requests can overshoot a limit by at
/// most (concurrency x one request's usage) — accepted.
#[derive(Debug, Default)]
pub struct UsageTracker {
    map: RwLock<HashMap<String, Arc<KeyUsage>>>,
}

impl UsageTracker {
    pub fn new() -> Self {
        Self::default()
    }

    fn entry(&self, token_id: &str) -> Arc<KeyUsage> {
        if let Some(u) = self.map.read().unwrap().get(token_id) {
            return Arc::clone(u);
        }
        let mut map = self.map.write().unwrap();
        Arc::clone(map.entry(token_id.to_string()).or_default())
    }

    /// Hot-path admission check; rejects with 429 insufficient_quota when
    /// a configured limit is already exhausted in the current window.
    pub fn check(&self, record: &KeyRecord, now: time::OffsetDateTime) -> Result<(), AppError> {
        if !record.has_limits() {
            return Ok(());
        }
        let usage = self.entry(&record.id);
        if let Some(limit) = record.credit_limit {
            let mut w = usage.credit.lock().unwrap();
            roll(&mut w, record.credit_limit_reset, now);
            if w.cost >= limit {
                return Err(AppError::QuotaExceeded(format!(
                    "credit limit of ${limit} reached for this API key"
                )));
            }
        }
        if let Some(limit) = record.token_limit {
            let mut w = usage.tokens.lock().unwrap();
            roll(&mut w, record.token_limit_reset, now);
            if w.tokens >= limit {
                return Err(AppError::QuotaExceeded(format!(
                    "token limit of {limit} tokens reached for this API key"
                )));
            }
        }
        Ok(())
    }

    /// Adds one finished request's usage to the key's current windows.
    pub fn accrue(&self, record: &KeyRecord, usage: RequestUsage, now: time::OffsetDateTime) {
        if usage.is_empty() {
            return;
        }
        let entry = self.entry(&record.id);
        {
            let mut w = entry.credit.lock().unwrap();
            roll(&mut w, record.credit_limit_reset, now);
            w.cost += usage.cost;
        }
        {
            let mut w = entry.tokens.lock().unwrap();
            roll(&mut w, record.token_limit_reset, now);
            w.tokens += usage.total_tokens;
            w.prompt += usage.prompt_tokens;
            w.completion += usage.completion_tokens;
        }
    }

    /// Current-window usage for display, rolling first.
    pub fn snapshot(&self, record: &KeyRecord, now: time::OffsetDateTime) -> UsageSnapshot {
        let entry = self.entry(&record.id);
        let cost = {
            let mut w = entry.credit.lock().unwrap();
            roll(&mut w, record.credit_limit_reset, now);
            w.cost
        };
        let mut w = entry.tokens.lock().unwrap();
        roll(&mut w, record.token_limit_reset, now);
        UsageSnapshot {
            cost,
            tokens: w.tokens,
            prompt_tokens: w.prompt,
            completion_tokens: w.completion,
        }
    }

    /// Recomputes a key's windows from `request_logs`. Used at startup for
    /// every key and again when an admin changes limits or intervals.
    pub fn warm_key(
        &self,
        db: &Db,
        record: &KeyRecord,
        now: time::OffsetDateTime,
    ) -> anyhow::Result<()> {
        let credit_start = window_start_ms(record.credit_limit_reset, now);
        let token_start = window_start_ms(record.token_limit_reset, now);
        let credit_sums = sums_since(db, &record.id, credit_start)?;
        let token_sums = if token_start == credit_start {
            credit_sums
        } else {
            sums_since(db, &record.id, token_start)?
        };
        let entry = self.entry(&record.id);
        *entry.credit.lock().unwrap() = Window {
            start_ms: credit_start,
            cost: credit_sums.cost,
            ..Window::default()
        };
        *entry.tokens.lock().unwrap() = Window {
            start_ms: token_start,
            cost: 0.0,
            tokens: token_sums.total_tokens,
            prompt: token_sums.prompt_tokens,
            completion: token_sums.completion_tokens,
        };
        Ok(())
    }

    pub fn warm_all(
        &self,
        db: &Db,
        records: impl IntoIterator<Item = impl std::borrow::Borrow<KeyRecord>>,
    ) -> anyhow::Result<()> {
        let now = time::OffsetDateTime::now_utc();
        for record in records {
            self.warm_key(db, record.borrow(), now)?;
        }
        Ok(())
    }

    pub fn remove(&self, token_id: &str) {
        self.map.write().unwrap().remove(token_id);
    }
}

fn sums_since(db: &Db, token_id: &str, since_ms: i64) -> anyhow::Result<RequestUsage> {
    db.with_read(|conn| {
        let row = conn.query_row(
            "SELECT COALESCE(SUM(cost), 0.0), COALESCE(SUM(total_tokens), 0),
                    COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0)
             FROM request_logs WHERE token_id = ?1 AND created_at >= ?2",
            rusqlite::params![token_id, since_ms],
            |row| {
                Ok(RequestUsage {
                    cost: row.get(0)?,
                    total_tokens: row.get(1)?,
                    prompt_tokens: row.get(2)?,
                    completion_tokens: row.get(3)?,
                })
            },
        )?;
        Ok(row)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trips_as_str() {
        for interval in [
            ResetInterval::Never,
            ResetInterval::Daily,
            ResetInterval::Weekly,
            ResetInterval::Monthly,
        ] {
            assert_eq!(ResetInterval::parse(interval.as_str()), Some(interval));
        }
        assert_eq!(ResetInterval::parse("hourly"), None);
    }

    #[test]
    fn serde_uses_lowercase() {
        assert_eq!(
            serde_json::to_string(&ResetInterval::Monthly).unwrap(),
            "\"monthly\""
        );
        let parsed: ResetInterval = serde_json::from_str("\"weekly\"").unwrap();
        assert_eq!(parsed, ResetInterval::Weekly);
    }

    /// A request that only moves the cost and total-token counters; the
    /// prompt/completion split is display-only and irrelevant to limits.
    fn usage(cost: f64, tokens: i64) -> RequestUsage {
        RequestUsage {
            cost,
            total_tokens: tokens,
            ..RequestUsage::default()
        }
    }

    fn at(s: &str) -> time::OffsetDateTime {
        time::OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).unwrap()
    }

    fn ms(s: &str) -> i64 {
        at(s).unix_timestamp() * 1000
    }

    #[test]
    fn window_start_never_is_epoch() {
        assert_eq!(
            window_start_ms(ResetInterval::Never, at("2026-07-19T10:30:00Z")),
            0
        );
    }

    #[test]
    fn window_start_daily_is_utc_midnight() {
        assert_eq!(
            window_start_ms(ResetInterval::Daily, at("2026-07-19T23:59:59Z")),
            ms("2026-07-19T00:00:00Z")
        );
        assert_eq!(
            window_start_ms(ResetInterval::Daily, at("2026-07-19T00:00:00Z")),
            ms("2026-07-19T00:00:00Z")
        );
    }

    #[test]
    fn window_start_weekly_is_monday() {
        // 2026-07-19 is a Sunday; its week began Monday 2026-07-13.
        assert_eq!(
            window_start_ms(ResetInterval::Weekly, at("2026-07-19T10:00:00Z")),
            ms("2026-07-13T00:00:00Z")
        );
        // A Monday is its own week start.
        assert_eq!(
            window_start_ms(ResetInterval::Weekly, at("2026-07-13T00:00:01Z")),
            ms("2026-07-13T00:00:00Z")
        );
        // Week spanning a month boundary.
        assert_eq!(
            window_start_ms(ResetInterval::Weekly, at("2026-08-01T12:00:00Z")),
            ms("2026-07-27T00:00:00Z")
        );
    }

    fn limited_key(credit: Option<f64>, tokens: Option<i64>) -> KeyRecord {
        let mut rec = KeyRecord::new("k".into());
        rec.credit_limit = credit;
        rec.credit_limit_reset = ResetInterval::Daily;
        rec.token_limit = tokens;
        rec.token_limit_reset = ResetInterval::Monthly;
        rec
    }

    #[test]
    fn unlimited_key_always_passes() {
        let tracker = UsageTracker::new();
        let rec = KeyRecord::new("k".into());
        let now = at("2026-07-19T10:00:00Z");
        tracker.accrue(&rec, usage(1e9, i64::MAX / 2), now);
        assert!(tracker.check(&rec, now).is_ok());
    }

    #[test]
    fn credit_limit_rejects_once_reached_and_rolls_over() {
        let tracker = UsageTracker::new();
        let rec = limited_key(Some(0.01), None);
        let now = at("2026-07-19T10:00:00Z");
        assert!(tracker.check(&rec, now).is_ok());
        // Overshoot is allowed on accrual; the next check rejects.
        tracker.accrue(&rec, usage(0.015, 100), now);
        let err = tracker.check(&rec, now).unwrap_err();
        assert!(matches!(err, AppError::QuotaExceeded(_)));
        // Still rejected later the same day...
        assert!(tracker.check(&rec, at("2026-07-19T23:59:59Z")).is_err());
        // ...but the next UTC day starts a fresh window.
        assert!(tracker.check(&rec, at("2026-07-20T00:00:00Z")).is_ok());
    }

    #[test]
    fn token_limit_is_independent_of_credit_window() {
        let tracker = UsageTracker::new();
        let rec = limited_key(Some(100.0), Some(50));
        let now = at("2026-07-19T10:00:00Z");
        tracker.accrue(&rec, usage(0.0, 60), now);
        assert!(matches!(
            tracker.check(&rec, now),
            Err(AppError::QuotaExceeded(_))
        ));
        // Next day the daily credit window rolls but the monthly token
        // window keeps its sum — still rejected.
        assert!(tracker.check(&rec, at("2026-07-20T10:00:00Z")).is_err());
        // Next month clears the token window.
        assert!(tracker.check(&rec, at("2026-08-01T00:00:00Z")).is_ok());
    }

    #[test]
    fn zero_limit_blocks_immediately() {
        let tracker = UsageTracker::new();
        let rec = limited_key(None, Some(0));
        assert!(tracker.check(&rec, at("2026-07-19T10:00:00Z")).is_err());
    }

    #[test]
    fn accrual_in_old_window_after_rollover_starts_fresh() {
        let tracker = UsageTracker::new();
        let rec = limited_key(Some(1.0), None);
        tracker.accrue(&rec, usage(0.9, 0), at("2026-07-19T10:00:00Z"));
        // Accruing on a later day rolls the window first: the new day
        // starts from zero, not 0.9.
        tracker.accrue(&rec, usage(0.2, 0), at("2026-07-20T10:00:00Z"));
        let w = tracker.snapshot(&rec, at("2026-07-20T11:00:00Z"));
        assert!((w.cost - 0.2).abs() < 1e-12);
        assert!(tracker.check(&rec, at("2026-07-20T11:00:00Z")).is_ok());
    }

    #[test]
    fn warm_key_loads_window_sums_from_logs() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("t.db")).unwrap();
        let now = at("2026-07-19T12:00:00Z");
        let insert = |id: &str, created_ms: i64, cost: f64, tokens: i64| {
            db.with_write(|conn| {
                conn.execute(
                    "INSERT INTO request_logs (id, created_at, token_id, model, path, cost,
                                               total_tokens, prompt_tokens, completion_tokens)
                     VALUES (?1, ?2, 'key1', 'm', '/v1/chat/completions', ?3, ?4, ?5, ?6)",
                    rusqlite::params![
                        id,
                        created_ms,
                        cost,
                        tokens,
                        tokens * 8 / 10,
                        tokens * 2 / 10
                    ],
                )?;
                Ok(())
            })
            .unwrap();
        };
        // Inside today's window (credit: daily) and this month (tokens).
        insert("r1", ms("2026-07-19T01:00:00Z"), 0.5, 100);
        // Yesterday: outside daily credit window, inside monthly token window.
        insert("r2", ms("2026-07-18T01:00:00Z"), 0.25, 40);
        // Last month: outside both.
        insert("r3", ms("2026-06-01T01:00:00Z"), 9.0, 9000);

        let tracker = UsageTracker::new();
        let rec = KeyRecord {
            id: "key1".into(),
            credit_limit: Some(10.0),
            credit_limit_reset: ResetInterval::Daily,
            token_limit: Some(1000),
            token_limit_reset: ResetInterval::Monthly,
            ..KeyRecord::new("k".into())
        };
        tracker.warm_key(&db, &rec, now).unwrap();
        let w = tracker.snapshot(&rec, now);
        assert!((w.cost - 0.5).abs() < 1e-12);
        assert_eq!(w.tokens, 140);
        // The prompt/completion split comes from the same window rows.
        assert_eq!(w.prompt_tokens, 112);
        assert_eq!(w.completion_tokens, 28);
    }

    #[test]
    fn window_start_monthly_handles_boundaries() {
        assert_eq!(
            window_start_ms(ResetInterval::Monthly, at("2026-07-19T10:00:00Z")),
            ms("2026-07-01T00:00:00Z")
        );
        // December -> window starts Dec 1; January 1 begins a new window.
        assert_eq!(
            window_start_ms(ResetInterval::Monthly, at("2026-12-31T23:59:59Z")),
            ms("2026-12-01T00:00:00Z")
        );
        assert_eq!(
            window_start_ms(ResetInterval::Monthly, at("2027-01-01T00:00:00Z")),
            ms("2027-01-01T00:00:00Z")
        );
        // Leap-February.
        assert_eq!(
            window_start_ms(ResetInterval::Monthly, at("2028-02-29T12:00:00Z")),
            ms("2028-02-01T00:00:00Z")
        );
    }
}
