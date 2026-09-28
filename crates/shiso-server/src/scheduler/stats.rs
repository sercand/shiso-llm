//! What the coordinator has done, counted where a handler can read it.
//!
//! # Why atomics and not a message
//!
//! The counters are read by `/metrics` and `/cache`, which run on runtime
//! workers, and written by the one thread that owns the model. A request/reply
//! message for a scrape would put an HTTP request behind a decode step; a lock
//! would put the decode step behind an HTTP request. Relaxed atomics are
//! neither: the writer never waits, and a scrape that reads two counters an
//! instant apart is a scrape of a server that was running while it was scraped.
//!
//! # Why every counter is written by something
//!
//! A counter nobody increments is worse than an absent one, because a dashboard
//! built on it looks healthy for the reason that nothing is happening. Each
//! field below names the exact event that moves it, and the ones the scheduler
//! cannot observe are absent rather than present and zero.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// The relaxed ordering every counter uses.
///
/// These are counts read by a human and by a scrape, never a synchronisation
/// edge between two threads: nothing downstream branches on one and then
/// assumes something else is visible, so anything stronger would buy ordering
/// nobody depends on.
const ORDER: Ordering = Ordering::Relaxed;

/// One counter with its metric name and help text.
struct Counter<'a> {
    name: &'a str,
    help: &'a str,
    kind: &'a str,
    value: u64,
}

impl Counter<'static> {
    /// A monotonically rising count.
    const fn total(name: &'static str, help: &'static str, value: u64) -> Self {
        Self {
            name,
            help,
            kind: "counter",
            value,
        }
    }

    /// A number that goes up and down.
    const fn gauge(name: &'static str, help: &'static str, value: u64) -> Self {
        Self {
            name,
            help,
            kind: "gauge",
            value,
        }
    }
}

/// The counters one periodic log line is a difference of.
///
/// Plain numbers rather than a view of the atomics: two of these, taken an
/// interval apart, are what the line reports rates over.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// Prompt tokens run through a forward pass.
    pub prefill_tokens: u64,
    /// Tokens generated.
    pub decode_tokens: u64,
    /// Guesses a draft head made.
    pub drafted: u64,
    /// Guesses the target agreed with.
    pub accepted: u64,
    /// Speculative steps taken, one per row.
    pub spec_steps: u64,
    /// Requests given a slot and a session.
    pub admitted: u64,
    /// Requests that resumed from a stored prefix.
    pub checkpoint_hits: u64,
}

impl Snapshot {
    /// What happened between `earlier` and `self`, `secs` apart.
    ///
    /// Saturating, so a pair taken out of order reads as nothing happening
    /// rather than as a wrapped count.
    #[must_use]
    pub fn since(&self, earlier: &Self, secs: f64) -> Window {
        let d = |now: u64, then: u64| now.saturating_sub(then);
        Window {
            secs,
            prefill_tokens: d(self.prefill_tokens, earlier.prefill_tokens),
            decode_tokens: d(self.decode_tokens, earlier.decode_tokens),
            drafted: d(self.drafted, earlier.drafted),
            accepted: d(self.accepted, earlier.accepted),
            spec_steps: d(self.spec_steps, earlier.spec_steps),
            admitted: d(self.admitted, earlier.admitted),
            checkpoint_hits: d(self.checkpoint_hits, earlier.checkpoint_hits),
        }
    }
}

/// The work done over one interval, and the rates a log line reads from it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    /// How long the interval was.
    pub secs: f64,
    /// Prompt tokens prefilled in it.
    pub prefill_tokens: u64,
    /// Tokens generated in it.
    pub decode_tokens: u64,
    /// Guesses drafted in it.
    pub drafted: u64,
    /// Guesses accepted in it.
    pub accepted: u64,
    /// Speculative steps in it.
    pub spec_steps: u64,
    /// Requests admitted in it.
    pub admitted: u64,
    /// Admitted requests that resumed from a stored prefix.
    pub checkpoint_hits: u64,
}

#[expect(
    clippy::cast_precision_loss,
    reason = "rates and ratios printed for a human to read to one decimal place"
)]
impl Window {
    /// Whether anything ran in it.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.prefill_tokens == 0 && self.decode_tokens == 0 && self.admitted == 0
    }

    /// Prompt tokens a second.
    #[must_use]
    pub fn prefill_tps(&self) -> f64 {
        per_sec(self.prefill_tokens, self.secs)
    }

    /// Generated tokens a second, across every sequence.
    #[must_use]
    pub fn decode_tps(&self) -> f64 {
        per_sec(self.decode_tokens, self.secs)
    }

    /// The share of drafted guesses the target kept; `None` with no drafts.
    #[must_use]
    pub fn acceptance(&self) -> Option<f64> {
        (self.drafted > 0).then(|| self.accepted as f64 / self.drafted as f64)
    }

    /// Tokens one speculative step yielded: the accepted guesses plus the
    /// token the verify pass always produces. `None` with no speculative step.
    #[must_use]
    pub fn tokens_per_step(&self) -> Option<f64> {
        (self.spec_steps > 0)
            .then(|| (self.accepted + self.spec_steps) as f64 / self.spec_steps as f64)
    }

    /// The share of admitted requests that resumed from a stored prefix;
    /// `None` when nothing was admitted.
    #[must_use]
    pub fn hit_rate(&self) -> Option<f64> {
        (self.admitted > 0).then(|| self.checkpoint_hits as f64 / self.admitted as f64)
    }
}

/// `count` over `secs`, zero over an interval with no length.
#[expect(
    clippy::cast_precision_loss,
    reason = "a rate printed for a human to read"
)]
fn per_sec(count: u64, secs: f64) -> f64 {
    if secs > 0.0 { count as f64 / secs } else { 0.0 }
}

/// What the coordinator has done since the process started.
///
/// Shared as an `Arc` between the engine thread that writes it and the handlers
/// that read it. `#[non_exhaustive]` because this is a report a caller is handed
/// rather than one it builds: the fields are private and read through accessors,
/// and a new counter must not be a compile error at every scrape site.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct SchedulerStats {
    requests: AtomicU64,
    admitted: AtomicU64,
    rejected: AtomicU64,
    active: AtomicU64,
    pending: AtomicU64,
    prefill_tokens: AtomicU64,
    decode_tokens: AtomicU64,
    reused_tokens: AtomicU64,
    completions: AtomicU64,
    cancellations: AtomicU64,
    timeouts: AtomicU64,
    failures: AtomicU64,
    checkpoint_hits: AtomicU64,
    checkpoint_writes: AtomicU64,
    session_opens: AtomicU64,
    session_closes: AtomicU64,
    shutdown_reclaims: AtomicU64,
    ttft_ms_total: AtomicU64,
    batch_steps: AtomicU64,
    /// Guesses a draft head made.
    drafted: AtomicU64,
    /// Guesses the target agreed with.
    accepted: AtomicU64,
    /// Speculative steps taken, one per row: what the accepted guesses are
    /// averaged over to give the tokens one step yields.
    spec_steps: AtomicU64,
    batched_rows: AtomicU64,
}

impl SchedulerStats {
    /// A fresh set of counters, all zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests that reached the coordinator's queue.
    #[must_use]
    pub fn requests(&self) -> u64 {
        self.requests.load(ORDER)
    }

    /// Requests that were given a slot and a session.
    #[must_use]
    pub fn admitted(&self) -> u64 {
        self.admitted.load(ORDER)
    }

    /// Requests refused because the queue was full.
    #[must_use]
    pub fn rejected(&self) -> u64 {
        self.rejected.load(ORDER)
    }

    /// Sequences resident right now.
    #[must_use]
    pub fn active(&self) -> u64 {
        self.active.load(ORDER)
    }

    /// Requests waiting for a slot right now.
    #[must_use]
    pub fn pending(&self) -> u64 {
        self.pending.load(ORDER)
    }

    /// Prompt tokens actually run through a forward pass.
    #[must_use]
    pub fn prefill_tokens(&self) -> u64 {
        self.prefill_tokens.load(ORDER)
    }

    /// Prompt tokens a checkpoint resume made unnecessary.
    #[must_use]
    pub fn reused_tokens(&self) -> u64 {
        self.reused_tokens.load(ORDER)
    }

    /// Tokens generated.
    #[must_use]
    pub fn decode_tokens(&self) -> u64 {
        self.decode_tokens.load(ORDER)
    }

    /// Turns that ran to a terminal event.
    #[must_use]
    pub fn completions(&self) -> u64 {
        self.completions.load(ORDER)
    }

    /// Turns cut short because their client stopped reading.
    #[must_use]
    pub fn cancellations(&self) -> u64 {
        self.cancellations.load(ORDER)
    }

    /// Turns cut short because they outlived their deadline.
    #[must_use]
    pub fn timeouts(&self) -> u64 {
        self.timeouts.load(ORDER)
    }

    /// Requests that found a stored prefix.
    #[must_use]
    pub fn checkpoint_hits(&self) -> u64 {
        self.checkpoint_hits.load(ORDER)
    }

    /// Prefixes this process wrote to the checkpoint tier.
    #[must_use]
    pub fn checkpoint_writes(&self) -> u64 {
        self.checkpoint_writes.load(ORDER)
    }

    /// Sessions opened.
    #[must_use]
    pub fn session_opens(&self) -> u64 {
        self.session_opens.load(ORDER)
    }

    /// Sessions closed. Equal to [`SchedulerStats::session_opens`] whenever the
    /// coordinator is between ticks, which is the ownership invariant stated as
    /// a number an operator can watch.
    #[must_use]
    pub fn session_closes(&self) -> u64 {
        self.session_closes.load(ORDER)
    }

    /// Sessions stored on the way down by the shutdown sweep.
    #[must_use]
    pub fn shutdown_reclaims(&self) -> u64 {
        self.shutdown_reclaims.load(ORDER)
    }

    /// The share of requests that resumed from a stored prefix.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let admitted = self.admitted();
        if admitted == 0 {
            return 0.0;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "a ratio of two counts is reported to two decimal places, and a count \
                      large enough to lose precision as f64 has long since stopped mattering"
        )]
        let rate = self.checkpoint_hits() as f64 / admitted as f64;
        rate
    }

    /// The mean number of sequences one batched decode step covered.
    ///
    /// The number the whole batching machinery exists to move: one means every
    /// step ran a batch of one and the device was doing a single sequence's work
    /// for a single sequence's cost.
    #[must_use]
    pub fn mean_batch_width(&self) -> f64 {
        let steps = self.batch_steps.load(ORDER);
        if steps == 0 {
            return 0.0;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "as above: this is a mean reported for a human to read"
        )]
        let width = self.batched_rows.load(ORDER) as f64 / steps as f64;
        width
    }

    /// The mean time to first token, in milliseconds.
    #[must_use]
    pub fn mean_ttft_ms(&self) -> f64 {
        let completions = self.completions();
        if completions == 0 {
            return 0.0;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "as above: this is a mean reported for a human to read"
        )]
        let mean = self.ttft_ms_total.load(ORDER) as f64 / completions as f64;
        mean
    }

    /// One more request reached the queue.
    pub fn request_queued(&self) {
        self.requests.fetch_add(1, ORDER);
    }

    /// One more request was refused for want of room.
    pub fn request_rejected(&self) {
        self.rejected.fetch_add(1, ORDER);
    }

    /// The queue depth, as the coordinator last saw it.
    pub fn set_pending(&self, depth: usize) {
        self.pending.store(depth as u64, ORDER);
    }

    /// The resident count, as the coordinator last saw it.
    pub fn set_active(&self, count: usize) {
        self.active.store(count as u64, ORDER);
    }

    /// A request was given a slot and a session.
    pub fn admit(&self) {
        self.admitted.fetch_add(1, ORDER);
        self.session_opens.fetch_add(1, ORDER);
    }

    /// A session was released.
    pub fn session_closed(&self) {
        self.session_closes.fetch_add(1, ORDER);
    }

    /// `tokens` prompt tokens went through a forward pass.
    pub fn prefilled(&self, tokens: u64) {
        self.prefill_tokens.fetch_add(tokens, ORDER);
    }

    /// A resume covered `tokens` prompt tokens that were never prefilled.
    pub fn resumed(&self, tokens: u64) {
        self.checkpoint_hits.fetch_add(1, ORDER);
        self.reused_tokens.fetch_add(tokens, ORDER);
    }

    /// A prefix was written to the checkpoint tier.
    pub fn checkpoint_written(&self) {
        self.checkpoint_writes.fetch_add(1, ORDER);
    }

    /// The shutdown sweep stored `count` sessions.
    pub fn swept(&self, count: u64) {
        self.shutdown_reclaims.fetch_add(count, ORDER);
    }

    /// One batched decode step over `rows` sequences produced `rows` tokens.
    pub fn decoded(&self, rows: usize) {
        self.batch_steps.fetch_add(1, ORDER);
        self.batched_rows.fetch_add(rows as u64, ORDER);
        self.decode_tokens.fetch_add(rows as u64, ORDER);
    }

    /// One speculative step made `drafted` guesses and kept `accepted` of them.
    ///
    /// Counted apart from the tokens because the ratio is the number that says
    /// whether the head is earning the width the verify pass pays for it, and a
    /// token count alone cannot show it.
    pub fn speculated(&self, accepted: u32, drafted: u32) {
        self.accepted.fetch_add(u64::from(accepted), ORDER);
        self.drafted.fetch_add(u64::from(drafted), ORDER);
        self.spec_steps.fetch_add(1, ORDER);
    }

    /// The counters the periodic log line takes differences of, read at once.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            prefill_tokens: self.prefill_tokens(),
            decode_tokens: self.decode_tokens(),
            drafted: self.drafted.load(ORDER),
            accepted: self.accepted.load(ORDER),
            spec_steps: self.spec_steps.load(ORDER),
            admitted: self.admitted(),
            checkpoint_hits: self.checkpoint_hits(),
        }
    }

    /// Guesses made and guesses kept, in that order.
    #[must_use]
    pub fn speculation(&self) -> (u64, u64) {
        (self.drafted.load(ORDER), self.accepted.load(ORDER))
    }

    /// A turn ended normally, having taken `ttft` to reach its first token.
    pub fn completed(&self, ttft: std::time::Duration) {
        self.completions.fetch_add(1, ORDER);
        // Whole milliseconds, because the sum is only ever divided to produce a
        // mean for a human and sub-millisecond precision in that mean would be
        // precision the clock underneath does not have either.
        let whole = u64::try_from(ttft.as_millis()).unwrap_or(u64::MAX);
        self.ttft_ms_total.fetch_add(whole, ORDER);
    }

    /// A turn ended because its client stopped reading.
    pub fn cancelled(&self) {
        self.cancellations.fetch_add(1, ORDER);
    }

    /// A turn ended because it outlived its deadline.
    pub fn timed_out(&self) {
        self.timeouts.fetch_add(1, ORDER);
    }

    /// A turn ended in a device or vocabulary failure.
    pub fn failed(&self) {
        self.failures.fetch_add(1, ORDER);
    }

    /// Turns that ended in a failure.
    #[must_use]
    pub fn failures(&self) -> u64 {
        self.failures.load(ORDER)
    }

    /// Every counter, in the order a scrape reads them.
    fn counters(&self) -> Vec<Counter<'static>> {
        let mut all = self.work_counters();
        all.extend(self.checkpoint_counters());
        all
    }

    /// What the loop did with the requests it was given.
    fn work_counters(&self) -> Vec<Counter<'static>> {
        vec![
            Counter::total(
                "shiso_scheduler_requests_total",
                "Generations that reached the queue",
                self.requests(),
            ),
            Counter::total(
                "shiso_scheduler_admitted_total",
                "Generations given a slot and a session",
                self.admitted(),
            ),
            Counter::total(
                "shiso_scheduler_rejected_total",
                "Generations refused because the queue was full",
                self.rejected(),
            ),
            Counter::gauge(
                "shiso_scheduler_active",
                "Sequences resident now",
                self.active(),
            ),
            Counter::gauge(
                "shiso_scheduler_pending",
                "Generations waiting for a slot now",
                self.pending(),
            ),
            Counter::total(
                "shiso_scheduler_prefill_tokens_total",
                "Prompt tokens run through a forward pass",
                self.prefill_tokens(),
            ),
            Counter::total(
                "shiso_scheduler_decode_tokens_total",
                "Tokens generated",
                self.decode_tokens(),
            ),
            Counter::total(
                "shiso_scheduler_completions_total",
                "Turns that reached a terminal event",
                self.completions(),
            ),
            Counter::total(
                "shiso_scheduler_cancellations_total",
                "Turns cut short because a client stopped reading",
                self.cancellations(),
            ),
            Counter::total(
                "shiso_scheduler_timeouts_total",
                "Turns cut short by their deadline",
                self.timeouts(),
            ),
            Counter::total(
                "shiso_scheduler_failures_total",
                "Turns that ended in a failure",
                self.failures(),
            ),
            Counter::total(
                "shiso_scheduler_session_opens_total",
                "Sessions opened by the coordinator",
                self.session_opens(),
            ),
            Counter::total(
                "shiso_scheduler_session_closes_total",
                "Sessions closed by the coordinator",
                self.session_closes(),
            ),
        ]
    }

    /// What the loop made of the checkpoint tier.
    fn checkpoint_counters(&self) -> Vec<Counter<'static>> {
        vec![
            Counter::total(
                "shiso_checkpoint_hits_total",
                "Generations that resumed from a stored prefix",
                self.checkpoint_hits(),
            ),
            Counter::total(
                "shiso_checkpoint_tokens_reused_total",
                "Prompt tokens a resume made unnecessary",
                self.reused_tokens(),
            ),
            Counter::total(
                "shiso_checkpoint_writes_total",
                "Prefixes written to the checkpoint tier",
                self.checkpoint_writes(),
            ),
            Counter::total(
                "shiso_checkpoint_shutdown_reclaims_total",
                "Sessions stored by the shutdown sweep",
                self.shutdown_reclaims(),
            ),
        ]
    }

    /// Renders every counter in the text exposition format.
    ///
    /// # Errors
    ///
    /// Whatever `out` reports. Writing into a `String` cannot fail, which is
    /// what every caller in this crate does; the signature is fallible so that a
    /// caller writing into something that can is not forced to buffer first.
    pub fn render(&self, out: &mut impl fmt::Write) -> fmt::Result {
        for counter in self.counters() {
            writeln!(out, "# HELP {} {}", counter.name, counter.help)?;
            writeln!(out, "# TYPE {} {}", counter.name, counter.kind)?;
            writeln!(out, "{} {}", counter.name, counter.value)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{SchedulerStats, Snapshot};

    #[test]
    fn an_untouched_set_of_counters_renders_every_name_at_zero() {
        let stats = SchedulerStats::new();
        let mut out = String::new();
        stats.render(&mut out).expect("a String accepts writes");
        assert!(out.contains("shiso_scheduler_requests_total 0"), "{out}");
        assert!(out.contains("# TYPE shiso_scheduler_active gauge"), "{out}");
        assert!(
            out.contains("shiso_checkpoint_tokens_reused_total 0"),
            "{out}"
        );
    }

    /// The open/close pair is the ownership invariant written as a number, so a
    /// leak shows up as a gap between two counters rather than as a slow
    /// exhaustion nobody can name.
    #[test]
    fn opening_and_closing_a_session_leaves_the_two_counters_equal() {
        let stats = SchedulerStats::new();
        stats.admit();
        stats.admit();
        stats.session_closed();
        assert_eq!(stats.session_opens(), 2);
        assert_eq!(stats.session_closes(), 1);
        stats.session_closed();
        assert_eq!(stats.session_opens(), stats.session_closes());
    }

    #[test]
    fn the_hit_rate_is_zero_before_anything_is_admitted_and_a_ratio_after() {
        let stats = SchedulerStats::new();
        assert!((stats.hit_rate() - 0.0).abs() < f64::EPSILON);
        stats.admit();
        stats.admit();
        stats.resumed(40);
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
        assert_eq!(stats.reused_tokens(), 40);
    }

    /// The mean batch width is the one number that says whether batching is
    /// happening at all, so it has to count rows and steps separately rather
    /// than being derived from the token counter.
    #[test]
    fn the_mean_batch_width_reports_the_rows_each_step_actually_covered() {
        let stats = SchedulerStats::new();
        stats.decoded(3);
        stats.decoded(1);
        assert!((stats.mean_batch_width() - 2.0).abs() < 1e-9);
        assert_eq!(stats.decode_tokens(), 4);
    }

    /// The periodic line is a difference of two snapshots, so a step taken
    /// between them has to show up in the window and nothing before it may.
    #[test]
    fn a_window_counts_only_what_happened_between_its_two_snapshots() {
        let stats = SchedulerStats::new();
        stats.prefilled(100);
        let earlier = stats.snapshot();
        stats.prefilled(400);
        stats.speculated(3, 4);
        stats.decoded(4);
        stats.speculated(1, 4);
        stats.decoded(2);
        let window = stats.snapshot().since(&earlier, 2.0);
        assert_eq!(window.prefill_tokens, 400);
        assert!((window.prefill_tps() - 200.0).abs() < 1e-9);
        assert!((window.decode_tps() - 3.0).abs() < 1e-9);
        assert_eq!(window.acceptance(), Some(0.5));
        // Four accepted over two steps, plus the verify token each step makes.
        assert_eq!(window.tokens_per_step(), Some(3.0));
        assert_eq!(window.hit_rate(), None);
    }

    #[test]
    fn an_idle_window_is_empty_and_reports_no_ratios() {
        let window = Snapshot::default().since(&Snapshot::default(), 10.0);
        assert!(window.is_empty());
        assert_eq!(window.acceptance(), None);
        assert_eq!(window.tokens_per_step(), None);
        assert!((window.prefill_tps() - 0.0).abs() < f64::EPSILON);
    }
}
