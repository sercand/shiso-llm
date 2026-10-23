//! The engine's periodic info line: what the last interval ran, at what rate.
//!
//! Written by the engine thread between ticks rather than by a timer thread,
//! because the cache figures come from the model and only that thread calls the
//! model. The price is that a tick longer than the interval delays the line,
//! which is also the only time a line would have nothing new to say.
//!
//! A server with nothing running prints nothing: the line after the last busy
//! interval says the rates fell to what they were, and then it goes quiet until
//! work arrives, so an idle log is not a wall of zeros.

use std::time::{Duration, Instant};

use shiso_core::page::PAGE_TOKENS;
use shiso_kvstore::CacheStats;

use super::stats::{SchedulerStats, Snapshot, Window};

/// When the next line is due and what the last one was measured from.
#[derive(Debug)]
pub(crate) struct StatsLog {
    /// `None` is the operator turning the line off.
    interval: Option<Duration>,
    last: Snapshot,
    last_at: Instant,
}

/// The gauges a line reads at the moment it is printed.
#[derive(Debug, Clone)]
pub(crate) struct Gauges {
    /// Sequences resident.
    pub(crate) running: usize,
    /// Requests waiting for a slot.
    pub(crate) waiting: usize,
    /// The cache underneath the model, when it reports one.
    pub(crate) kv: Option<CacheStats>,
}

impl StatsLog {
    /// A log whose first interval starts at `now`.
    pub(crate) fn new(interval: Option<Duration>, stats: &SchedulerStats, now: Instant) -> Self {
        Self {
            interval: interval.filter(|every| !every.is_zero()),
            last: stats.snapshot(),
            last_at: now,
        }
    }

    /// Whether a line is due at `now`: an interval has passed since the last.
    pub(crate) fn due(&self, now: Instant) -> bool {
        self.interval
            .is_some_and(|every| now.saturating_duration_since(self.last_at) >= every)
    }

    /// Prints the line for the interval ending at `now`, unless nothing ran in
    /// it, and starts the next one.
    pub(crate) fn emit(&mut self, stats: &SchedulerStats, now: Instant, gauges: &Gauges) {
        if self.interval.is_none() {
            return;
        }
        let snapshot = stats.snapshot();
        let secs = now.saturating_duration_since(self.last_at).as_secs_f64();
        let window = snapshot.since(&self.last, secs);
        self.last = snapshot;
        self.last_at = now;
        if window.is_empty() && gauges.running == 0 {
            return;
        }
        log_window(&window, gauges);
    }
}

/// The line itself. Rates to one decimal place: the digits past that are
/// scheduling noise at a ten-second window.
fn log_window(window: &Window, gauges: &Gauges) {
    let (kv_used, kv_tokens) = gauges.kv.as_ref().map_or((None, None), |kv| {
        let used = f64::from(kv.pages_used) / f64::from(kv.pool_pages.max(1));
        (
            Some(format!("{:.1}%", used * 100.0)),
            Some(u64::from(kv.pages_used) * u64::from(PAGE_TOKENS)),
        )
    });
    tracing::info!(
        prefill_tps = format_args!("{:.1}", window.prefill_tps()),
        decode_tps = format_args!("{:.1}", window.decode_tps()),
        running = gauges.running,
        waiting = gauges.waiting,
        kv_used = kv_used.map(display),
        kv_used_tokens = kv_tokens,
        prefix_hit_rate = window
            .hit_rate()
            .map(|rate| display(format!("{:.1}%", rate * 100.0))),
        spec_accept = window
            .acceptance()
            .map(|rate| display(format!("{:.1}%", rate * 100.0))),
        spec_tokens_per_step = window
            .tokens_per_step()
            .map(|mean| display(format!("{mean:.2}"))),
        score_passes = (window.score_passes > 0).then_some(window.score_passes),
        score_pass_runs = (window.score_passes > 0).then_some(window.score_pass_runs),
        "engine stats"
    );
}

#[cfg(test)]
mod tests {
    use super::StatsLog;
    use crate::scheduler::SchedulerStats;
    use std::time::{Duration, Instant};

    #[test]
    fn a_line_is_due_once_an_interval_has_passed_and_never_when_off() {
        let stats = SchedulerStats::new();
        let start = Instant::now();
        let every = StatsLog::new(Some(Duration::from_secs(10)), &stats, start);
        assert!(!every.due(start + Duration::from_secs(9)));
        assert!(every.due(start + Duration::from_secs(10)));
        let off = StatsLog::new(Some(Duration::ZERO), &stats, start);
        assert!(!off.due(start + Duration::from_secs(3600)));
        let none = StatsLog::new(None, &stats, start);
        assert!(!none.due(start + Duration::from_secs(3600)));
    }
}
