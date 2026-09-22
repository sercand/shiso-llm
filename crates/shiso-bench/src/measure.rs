//! The one place a measured number becomes a statistic.
//!
//! Every gate in this crate reports a median and a spread, and both are
//! computed here. A second implementation would be a second definition of what
//! a baseline means, and the first time the two disagreed the disagreement
//! would look like a regression.
//!
//! # The protocol, and why each part of it is there
//!
//! A single sample from a shared laptop is not a number anyone should gate on,
//! so a measured phase is wrapped in three stabilisers:
//!
//! 1. **A warmup pass that is thrown away**, and a whole warmup *repetition*
//!    besides. The first pass through a phase pays for lazily-claimed cache
//!    pages and a cold instruction path; it describes the first run rather than
//!    the steady state. The first whole repetition turned out to pay for more
//!    than that: measured here, its decode steps cost about three times what
//!    every later repetition's did, while repetitions two through seven agreed
//!    with each other to a tenth of a per cent. One cold repetition in seven is
//!    enough to move a median, so the outer loop discards one too.
//! 2. **The first decode step is dropped.** Step one is where the device
//!    schedules the work for the first time; every later step reuses that.
//! 3. **The statistic is a median, never a mean and never a best-of-N.** A mean
//!    lets one descheduled step move the record. Best-of-N is worse than either:
//!    it biases the number toward whichever run got the quietest machine, which
//!    is precisely the direction that hides a regression.
//!
//! The phase is then repeated end to end, and the *spread* across those outer
//! repetitions - the ratio of the slowest to the fastest, minus one - is
//! recorded beside the value. A noisy machine shows up as a wide spread and the
//! recorder refuses to write, rather than quietly producing a number that the
//! next run cannot reproduce.

/// The widest run-to-run spread a record may carry, as a fraction.
///
/// Two per cent is the noise policy for the pinned machines: below it the
/// repetitions agree closely enough that a five-per-cent regression threshold
/// means something, above it they do not. A machine that will not hold this is
/// quiesced and re-run. The number is never raised to make a run land - doing
/// so would widen the band the gate cannot see into, which is the one thing
/// that makes the gate useless rather than merely strict.
pub const SPREAD_MAX: f64 = 0.02;

/// The stabilisers one measured target ran with, recorded beside its numbers.
///
/// Recorded rather than assumed: a record whose protocol differs from the
/// baseline's is not comparable with it, and the only way a reader can tell is
/// if both say what they did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Protocol {
    /// Whole repetitions run and discarded before the timed ones.
    pub warmup_reps: u32,
    /// Prefill passes run and discarded before the timed ones, per repetition.
    pub warmup_prefills: u32,
    /// Prefill passes that are timed, per repetition.
    pub timed_prefills: u32,
    /// Decode steps run per repetition, the dropped ones included.
    pub decode_steps: u32,
    /// Leading decode steps discarded, per repetition.
    pub dropped_decode_steps: u32,
    /// Outer repetitions of the whole phase.
    pub reps: u32,
}

impl Protocol {
    /// The protocol the pinned machines record with.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            warmup_reps: 1,
            warmup_prefills: 1,
            // Eleven rather than a handful: a prefill of this prompt is roughly
            // a second of device work wrapped in a session open and close, and
            // the host half of that jitters by a few per cent. A median over
            // five of them inherited the jitter; a median over eleven does not,
            // which is what lets the repetitions agree inside the noise policy
            // without the policy being touched.
            timed_prefills: 11,
            decode_steps: 128,
            dropped_decode_steps: 1,
            reps: 7,
        }
    }

    /// The single-pass protocol the standalone runner uses.
    ///
    /// One repetition and no warmup repetition, so this is the cold number and
    /// says so: it is printed, never filed. The statistic within the repetition
    /// is the same one the recorder uses, which is what keeps the two runners
    /// from disagreeing about what they measured.
    #[must_use]
    pub const fn single() -> Self {
        Self {
            reps: 1,
            warmup_reps: 0,
            ..Self::standard()
        }
    }

    /// One timed repetition, with a whole repetition thrown away in front of it.
    ///
    /// The difference from [`Protocol::single`] is the discarded repetition, and
    /// it is not a refinement: for a checkpoint whose weights are a multi-gigabyte
    /// mapping, the first repetition is dominated by the first touch of every
    /// page it reads, and the figure it produces describes the operating system's
    /// paging rather than the device's arithmetic. A runner that prints a number
    /// for a family like that has to warm it first or say it is cold, and this is
    /// the first of the two.
    #[must_use]
    pub const fn warmed() -> Self {
        Self {
            reps: 1,
            warmup_reps: 1,
            ..Self::standard()
        }
    }

    /// The word the record uses for the statistic, for a reader of the JSON.
    #[must_use]
    pub const fn statistic(&self) -> &'static str {
        "median"
    }
}

/// One phase's repetitions, folded.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// The reported value: the median across repetitions.
    pub value: f64,
    /// Slowest over fastest, minus one. Zero for a single repetition.
    pub spread: f64,
    /// Each repetition's own statistic, in the order they ran.
    pub reps: Vec<f64>,
}

impl Summary {
    /// Folds one repetition per entry of `reps`.
    ///
    /// Returns `None` for an empty slice: a phase that produced no sample has
    /// no median, and inventing one - a zero, a NaN - would put a number into a
    /// record that no measurement stands behind.
    #[must_use]
    pub fn over(reps: &[f64]) -> Option<Self> {
        Some(Self {
            value: median(reps)?,
            spread: spread(reps)?,
            reps: reps.to_vec(),
        })
    }

    /// Whether the repetitions agreed closely enough to be recorded.
    #[must_use]
    pub fn steady(&self) -> bool {
        self.spread <= SPREAD_MAX
    }
}

/// The median of `samples`, or `None` when there are none.
///
/// An even count takes the mean of the two middle samples. Timings are a
/// continuous quantity, so the midpoint is the honest answer; picking one side
/// would make the statistic depend on which half of the pair ran first.
#[must_use]
pub fn median(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        f64::midpoint(sorted[mid - 1], sorted[mid])
    };
    Some(median)
}

/// Slowest over fastest, minus one, or `None` when there are no samples.
///
/// Zero when every sample is equal, and so zero for a single sample - one
/// repetition carries no evidence about reproducibility at all, which is why
/// the recorder asks for seven and the gate reads the count as well as the
/// number. `None` when a sample is not a positive duration, since the ratio
/// would then describe a clock failure rather than a machine.
#[must_use]
pub fn spread(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for sample in samples {
        low = low.min(*sample);
        high = high.max(*sample);
    }
    if low <= 0.0 || !low.is_finite() || !high.is_finite() {
        return None;
    }
    Some(high / low - 1.0)
}

/// The samples left after the leading `dropped` are discarded.
#[must_use]
pub fn after_warmup(samples: &[f64], dropped: u32) -> &[f64] {
    let dropped = usize::try_from(dropped).unwrap_or(samples.len());
    samples.get(dropped..).unwrap_or(&[])
}

/// Tokens per second from a per-token millisecond figure.
///
/// Zero for a non-positive duration, which is the only value that does not
/// claim a throughput no clock measured.
#[must_use]
pub fn tokens_per_sec(tokens: u32, total_ms: f64) -> f64 {
    if total_ms > 0.0 {
        f64::from(tokens) * 1000.0 / total_ms
    } else {
        0.0
    }
}
