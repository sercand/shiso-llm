//! What the coordinator was told, read once at boot.
//!
//! # Why a struct and not a scatter of environment reads
//!
//! Every value here is a policy the loop applies on every tick. Read from the
//! environment where it is used, the answer could change between two ticks of
//! the same request - a deadline that was five seconds when a turn started and
//! one second when it was checked is a deadline nobody configured. So the
//! environment is read exactly once, at boot, by [`SchedulerConfig::from_env`],
//! and the loop reads a struct it cannot change.
//!
//! That is also what makes the loop testable: a test builds the struct
//! directly, with a stopped clock, and never touches a process-wide variable
//! that every other concurrently running test would also see.

use std::sync::Arc;
use std::time::Duration;

use crate::scheduler::clock::{Clock, SystemClock};
use crate::tokenizer::turn_replay::TurnReplay;

/// Sequences that may be resident at once.
pub const DEFAULT_MAX_NUM_SEQS: usize = 4;
/// Requests that may wait for a slot before the queue refuses.
pub const DEFAULT_MAX_QUEUED: usize = 256;
/// Prompt tokens one prefill quantum covers while decodes are also running.
pub const DEFAULT_MIXED_QUANTUM: usize = 128;
/// Prompt tokens one prefill chunk covers when nothing else is running.
///
/// As wide as the widest pass a stack takes, because a stack splits a wider
/// chunk itself and a narrower one only costs passes: the sparse family reads
/// every routed expert's weights once per pass, and measured on a GB10 its
/// prefill ran 1580 tok/s at 8192-token chunks against 1000 at 2048.
pub const DEFAULT_SOLO_CHUNK: usize = 8192;
/// The grid a prefill chunk that does not finish its prompt ends on.
///
/// The recurrent blocks walk a prompt in runs of this many positions counted
/// from position zero, and a pass boundary inside a run cuts it into two runs
/// whose sums are not the one run's: the prompt would answer by where the
/// scheduler happened to cut it. Every stack's prompt passes are otherwise
/// computed row for row as any other cut would compute them, so a chunk end on
/// this grid is the whole of what keeps the widths, the quantum shared with
/// decoding sequences, and a resume point out of the answer.
pub const PREFILL_GRID: usize = 64;
/// How long a caller may block inside the queue before being refused.
pub const DEFAULT_QUEUE_ADMISSION_MS: u64 = 250;

/// The environment knob for the waiting bound.
const MAX_QUEUED_ENV: &str = "SHISO_MAX_QUEUED";
/// The environment knob for the per-request deadline, in milliseconds.
const REQUEST_TIMEOUT_ENV: &str = "SHISO_REQUEST_TIMEOUT_MS";
/// The environment knob for the admission wait, in milliseconds.
const QUEUE_ADMISSION_ENV: &str = "SHISO_QUEUE_ADMISSION_MS";
/// The environment knob for the checkpoint alignment.
const CKPT_ALIGN_ENV: &str = "SHISO_CKPT_BOUNDARY_ALIGN";
/// The environment knob for how far alignment may wait.
const CKPT_TRIM_ENV: &str = "SHISO_CKPT_BOUNDARY_TRIM";
/// The environment knob for the continued interval.
const CKPT_CONTINUED_ENV: &str = "SHISO_CKPT_CONTINUED_INTERVAL";
/// The environment knob for exact sampling in speculative decoding.
const MTP_EXACT_SAMPLING_ENV: &str = "SHISO_MTP_EXACT_SAMPLING";

/// Tokens a session has to add mid-answer before its frontier is offered again.
/// Zero: only the end of a turn, which every turn offers anyway.
pub const DEFAULT_CONTINUED_INTERVAL: u32 = 0;
/// Bytes of queued checkpoint pages one tick may copy off the device.
///
/// Eight pages of the sparse family's cache: a few milliseconds of copies
/// between decode steps, which is the whole of what the write-behind costs the
/// streams that are decoding.
pub const DEFAULT_PUMP_BYTES: u64 = 64 * 1024 * 1024;
/// How long a shutdown waits for queued checkpoint writes to land.
///
/// Under the 90 seconds the service unit allows a stop, with room left for the
/// teardown after it.
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(60);
/// The alignment a stored prefix is preferred to land on.
pub const DEFAULT_BOUNDARY_ALIGN: u32 = 256;
/// How far past its trigger a store may wait for that alignment.
pub const DEFAULT_BOUNDARY_TRIM: u32 = 64;

/// Reads one unsigned knob, falling back to `default`.
///
/// A value that does not parse is reported and ignored rather than failing the
/// boot: the server can serve perfectly well with the default, and refusing to
/// start over a typo in one tuning knob is a worse outcome than a log line.
fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Err(_unset) => default,
        Ok(raw) => match raw.trim().parse() {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(
                    knob = name,
                    value = raw,
                    %error,
                    "this setting is not a number; the default is being used"
                );
                default
            }
        },
    }
}

/// Reads one knob as a `usize` with a floor of one.
///
/// Zero is never a servable answer for any of the three counts that use this -
/// a scheduler with no slots serves nothing, a quantum of zero never finishes a
/// prefill - so it is clamped rather than accepted, and the clamp is said out
/// loud.
fn env_count(name: &str, default: usize) -> usize {
    let asked = usize::try_from(env_u64(name, default as u64)).unwrap_or(default);
    if asked == 0 {
        tracing::warn!(knob = name, "zero is not a servable value; using one");
        return 1;
    }
    asked
}

/// Reads one boolean knob, falling back to `default`.
fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Err(_unset) => default,
        Ok(raw) => match raw.trim() {
            "1" | "true" | "TRUE" | "yes" | "YES" => true,
            "0" | "false" | "FALSE" | "no" | "NO" => false,
            _ => {
                tracing::warn!(
                    knob = name,
                    value = raw,
                    "this setting is not a boolean; the default is being used"
                );
                default
            }
        },
    }
}

/// When a session's prefix is written to the checkpoint tier.
///
/// # Why this is a scheduler policy and not an engine one
///
/// The engine stores exactly one thing: the state a session is standing on,
/// under the key of the text that produced it. It has one intrinsic gate - a
/// prefix below a floor is not worth a file - and deliberately no opinion about
/// *which* of a turn's growing frontiers deserves one, because that answer
/// depends on how many other sequences are waiting for the device and how often
/// this deployment sees the same prompt again.
///
/// So the policy here does not choose a length to truncate to; a session cannot
/// be rewound, and offering the engine a shorter path than the one it stands on
/// is refused by design. It chooses *moments*: which of the frontiers a turn
/// passes through are offered to the store at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorePolicy {
    /// The multiple a stored prefix is preferred to end on. Zero and one both
    /// mean "any length will do".
    ///
    /// Aligning matters because the next request that resumes from an entry
    /// pays for the tokens it has to prefill on top of it, and entries that end
    /// at the same lengths are entries that can be shared by conversations that
    /// diverge later.
    pub boundary_align: u32,
    /// How many tokens past a store's trigger the policy will wait to reach
    /// [`StorePolicy::boundary_align`].
    ///
    /// Without a bound, alignment would be able to postpone a store forever on
    /// a turn that ends before the next boundary. With one, a turn that will not
    /// reach the boundary in time is stored unaligned rather than not at all.
    pub boundary_trim: u32,
    /// Tokens a session has to have added since its last stored frontier before
    /// another one is offered. Zero stores only at the end of the turn.
    pub continued_interval: u32,
    /// Bytes of queued checkpoint pages one tick copies off the device for the
    /// disk tier's writer.
    pub pump_bytes: u64,
    /// How long a shutdown waits for queued checkpoint writes.
    pub drain_timeout: Duration,
}

impl StorePolicy {
    /// The defaults: 256-token alignment, 64 tokens of patience for it, the end
    /// of each turn only, a 64 MiB pump per tick and a minute
    /// to drain on the way down.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            boundary_align: DEFAULT_BOUNDARY_ALIGN,
            boundary_trim: DEFAULT_BOUNDARY_TRIM,
            continued_interval: DEFAULT_CONTINUED_INTERVAL,
            pump_bytes: DEFAULT_PUMP_BYTES,
            drain_timeout: DEFAULT_DRAIN_TIMEOUT,
        }
    }

    /// The policy this process was configured with.
    #[must_use]
    pub fn from_env() -> Self {
        let default = Self::new();
        Self {
            boundary_align: clamp_u32(env_u64(CKPT_ALIGN_ENV, default.boundary_align.into())),
            boundary_trim: clamp_u32(env_u64(CKPT_TRIM_ENV, default.boundary_trim.into())),
            continued_interval: clamp_u32(env_u64(
                CKPT_CONTINUED_ENV,
                default.continued_interval.into(),
            )),
            // `--kvstore-pump-mb` and `--kvstore-drain-secs` are flags, set
            // over these by the binary.
            pump_bytes: default.pump_bytes,
            drain_timeout: default.drain_timeout,
        }
    }

    /// Whether a frontier of `frontier` is worth storing given that `stored` was
    /// the last length this session put on disk.
    ///
    /// `final_frontier` says the turn is over, which is what turns the
    /// alignment preference back into a bound: a turn that has ended will never
    /// reach the next boundary, so waiting for one would mean never storing.
    #[must_use]
    pub const fn stores_continued(&self, frontier: u32, stored: u32, final_frontier: bool) -> bool {
        let grown = frontier.saturating_sub(stored);
        if self.continued_interval == 0 {
            // Only the end of the turn, and only if anything was added since the
            // cold store - re-writing an identical frontier is a file the store
            // would recognise as a duplicate anyway.
            return final_frontier && grown > 0;
        }
        if grown == 0 {
            return false;
        }
        if final_frontier {
            return true;
        }
        if grown < self.continued_interval {
            return false;
        }
        // Past the trigger. Hold for an aligned frontier, but only for as long
        // as the trim allows.
        self.aligned(frontier) || grown >= self.continued_interval + self.boundary_trim
    }

    /// Whether `frontier` lands on the configured boundary.
    #[must_use]
    pub const fn aligned(&self, frontier: u32) -> bool {
        self.boundary_align <= 1 || frontier.is_multiple_of(self.boundary_align)
    }
}

impl Default for StorePolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Narrows a knob to the width the store's own counts use.
fn clamp_u32(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Everything the coordinator was told before it started.
///
/// Deliberately not `#[non_exhaustive]`: every caller that builds one is in
/// this repository and recompiles with it, so the attribute would buy nothing
/// and cost the struct expression that makes a test's configuration readable in
/// one place.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Sequences that may hold a session at once.
    ///
    /// This *is* the residency policy. A slot is held from the moment a session
    /// is opened to the one call that closes it, so a resident session is never
    /// evicted to make room for a newer one - it retires by finishing, by
    /// outliving its deadline, or by its client hanging up. Page-level pressure
    /// underneath is the cache's business and has its own eviction.
    pub max_num_seqs: usize,
    /// Requests that may wait for a slot before the queue answers "overloaded".
    pub max_queued: usize,
    /// Prompt tokens one prefill quantum covers while decodes are also running.
    ///
    /// Bounded because a prefill is a single device call that nothing can
    /// interrupt: a whole prompt run at once is a whole prompt's worth of
    /// latency added to every sequence that is already decoding.
    pub mixed_quantum: usize,
    /// Prompt tokens one prefill chunk covers when nothing is decoding.
    ///
    /// Wider than the mixed quantum, and for the opposite reason: with nobody to
    /// starve, the only thing small chunks buy is more launches.
    pub solo_chunk: usize,
    /// How long one request may run before it is cut. `None` is no deadline.
    pub request_timeout: Option<Duration>,
    /// How long a caller blocks inside the queue before being refused.
    pub queue_admission: Duration,
    /// When a session's prefix is offered to the checkpoint tier.
    pub store: StorePolicy,
    /// Where elapsed time is read from.
    pub clock: Arc<dyn Clock>,
    /// Where finished turns are kept for the renderer to reuse their ids.
    /// `None` keeps nothing.
    pub turn_replay: Option<Arc<TurnReplay>>,
    /// How often the engine prints its info line of rates and cache use.
    /// `None` prints none; the binary sets it from `--log-stats-interval`.
    pub stats_interval: Option<Duration>,
    /// Whether non-greedy sequences preserve exact sampling instead of
    /// accepting target-matching greedy drafts directly.
    pub mtp_exact_sampling: bool,
    /// Token ids no generated token may be: the control tokens a model must
    /// never write - a turn opener, a media marker - even when it is quoting
    /// one. Every row a sequence draws from is masked with it. Empty
    /// suppresses nothing; the binary fills it from the vocabulary.
    pub suppressed: Arc<[shiso_core::TokenId]>,
}

impl SchedulerConfig {
    /// The defaults, with the real clock.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            max_queued: DEFAULT_MAX_QUEUED,
            mixed_quantum: DEFAULT_MIXED_QUANTUM,
            solo_chunk: DEFAULT_SOLO_CHUNK,
            request_timeout: None,
            queue_admission: Duration::from_millis(DEFAULT_QUEUE_ADMISSION_MS),
            store: StorePolicy::new(),
            clock: Arc::new(SystemClock),
            turn_replay: None,
            stats_interval: None,
            mtp_exact_sampling: false,
            suppressed: Arc::from([]),
        }
    }

    /// The configuration this process was started with.
    ///
    /// The one place the environment is read for the scheduler, called once
    /// from the binary's entry point. Every knob is prefixed, so `env | grep`
    /// is an exhaustive list of what a running server was told.
    #[must_use]
    pub fn from_env() -> Self {
        let default = Self::new();
        let timeout_ms = env_u64(REQUEST_TIMEOUT_ENV, 0);
        Self {
            // `--max-num-seqs` (and its `SHISO_MAX_NUM_SEQS`) is a flag, set
            // over this by the binary: the KV pool is sized from the same count.
            max_num_seqs: default.max_num_seqs,
            max_queued: env_count(MAX_QUEUED_ENV, default.max_queued),
            // `--mixed-prefill-quantum` and `--solo-prefill-chunk` (or
            // `--max-num-batched-tokens`) are flags, set over these by the
            // binary, which refuses a zero rather than raising it to one.
            mixed_quantum: default.mixed_quantum,
            solo_chunk: default.solo_chunk,
            // Zero is how the knob spells "no deadline", which is a different
            // thing from a deadline of zero milliseconds and has to stay
            // distinguishable from it.
            request_timeout: (timeout_ms > 0).then(|| Duration::from_millis(timeout_ms)),
            queue_admission: Duration::from_millis(env_u64(
                QUEUE_ADMISSION_ENV,
                DEFAULT_QUEUE_ADMISSION_MS,
            )),
            store: StorePolicy::from_env(),
            clock: default.clock,
            // Shared with the renderer, so the binary builds it and sets it.
            turn_replay: None,
            // `--log-stats-interval`, set by the binary.
            stats_interval: None,
            mtp_exact_sampling: env_bool(MTP_EXACT_SAMPLING_ENV, false),
            suppressed: default.suppressed.clone(),
        }
    }

    /// The same configuration reading time from `clock`.
    ///
    /// Generic over the concrete clock rather than taking the trait object
    /// directly, so a caller hands over the `Arc` it already has instead of
    /// writing a cast at every call site - and a cast at a call site is a place
    /// the wrong type can be coerced into looking right.
    #[must_use]
    pub fn with_clock<C: Clock + 'static>(mut self, clock: Arc<C>) -> Self {
        self.clock = clock;
        self
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{SchedulerConfig, StorePolicy};

    /// The interval is what keeps a long turn from writing a file per token,
    /// and the end of a turn is what keeps a short one from writing none.
    #[test]
    fn a_growing_turn_is_stored_on_the_interval_and_again_when_it_ends() {
        let policy = StorePolicy {
            boundary_align: 0,
            boundary_trim: 0,
            continued_interval: 100,
            ..StorePolicy::new()
        };
        assert!(!policy.stores_continued(1099, 1000, false));
        assert!(policy.stores_continued(1100, 1000, false));
        assert!(
            policy.stores_continued(1001, 1000, true),
            "a turn that ends is stored whatever it added"
        );
        assert!(
            !policy.stores_continued(1000, 1000, true),
            "a frontier that has not moved is the file that is already there"
        );
    }

    /// Past the trigger the policy waits for an aligned frontier - but only as
    /// far as the trim, so a turn that never reaches one is still stored.
    #[test]
    fn alignment_delays_a_store_to_a_boundary_and_the_trim_bounds_the_delay() {
        let policy = StorePolicy {
            boundary_align: 64,
            boundary_trim: 16,
            continued_interval: 100,
            ..StorePolicy::new()
        };
        // Triggered at 1100, which is not a multiple of 64, so it waits.
        assert!(!policy.stores_continued(1100, 1000, false));
        assert!(policy.stores_continued(1152, 1000, false), "1152 = 18 * 64");

        // A run that would sail past the boundary is cut off by the trim
        // instead of waiting for one it may never see.
        let impatient = StorePolicy {
            boundary_align: 1_000_000,
            ..policy
        };
        assert!(!impatient.stores_continued(1115, 1000, false));
        assert!(impatient.stores_continued(1116, 1000, false));
    }

    /// Zero means "only when the turn is over", which is the setting a
    /// deployment that cares about disk more than about resume latency wants.
    #[test]
    fn an_interval_of_zero_stores_only_at_the_end_of_a_turn() {
        let policy = StorePolicy {
            continued_interval: 0,
            ..StorePolicy::new()
        };
        assert!(!policy.stores_continued(9999, 0, false));
        assert!(policy.stores_continued(9999, 0, true));
    }

    #[test]
    fn the_default_configuration_is_servable_without_any_environment() {
        let cfg = SchedulerConfig::new();
        assert!(cfg.max_num_seqs >= 1);
        assert!(cfg.mixed_quantum <= cfg.solo_chunk);
        assert_eq!(
            cfg.request_timeout, None,
            "a server has no deadline unless told"
        );
    }
}
