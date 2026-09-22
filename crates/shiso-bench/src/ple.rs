//! Fetch-latency distribution for the streaming expert tier, and its gate.
//!
//! # Why a histogram and not a list of samples
//!
//! A prefill can wait on tens of thousands of row fetches. Keeping every one of
//! them to sort at the end would make the measurement allocate in the path it
//! is measuring, which is the classic way to measure the instrument. A bounded
//! histogram costs one increment per wait, a fixed amount of memory chosen up
//! front, and gives percentiles to within one bucket - which is far finer than
//! the five-per-cent threshold the gate applies to them.
//!
//! # Where a percentile is reported
//!
//! Bucket `i` covers `[i·w, (i+1)·w)` microseconds, and a percentile is reported
//! at its bucket's **upper** edge. So the number is never smaller than the true
//! one: rounding a latency down would let a regression of up to one bucket width
//! hide inside the rounding, and the whole point of a p99 is that it does not
//! get to hide. Samples past the last bucket land in an overflow count and are
//! reported at the largest wait actually seen, which is tracked exactly.
//!
//! # What the gate compares, and against what
//!
//! Two different questions, deliberately answered by two different references:
//!
//! - **Tail behaviour** is compared against this port's own recorded history.
//!   There is no external p99 to compare against - the implementation this port
//!   follows keeps cumulative read time per worker and prints a mean and a max,
//!   and no percentile exists anywhere in it. Gating a p99 against a number
//!   nobody measured would be inventing the reference.
//! - **Steady-state cost** is compared against the reference implementation on
//!   the metrics it can actually emit: mean wait, max wait, hit rate and mean
//!   read time. Of those, only the mean is gated. A max is one sample and moves
//!   with whatever else the machine was doing; it is recorded because it is
//!   worth reading, and not gated because it cannot be reproduced.
//!
//! # Not yet recorded anywhere
//!
//! Nothing in this tree counts waits yet: the streaming tier's statistics carry
//! lookups, hits, reads and cumulative read time, and the per-wait counter this
//! module consumes is still owed by the crate that owns the tier. So no record
//! written today carries this section, and the gate reports the section as
//! absent rather than passing - an ungated metric must never read as a green
//! one.

use serde::{Deserialize, Serialize};

use crate::baseline::{Comparison, Report};

/// A bounded histogram of fetch waits, in microseconds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaitHistogram {
    /// Bucket width in microseconds.
    width_us: u64,
    /// One counter per bucket, the first covering `[0, width_us)`.
    counts: Vec<u64>,
    /// Waits past the last bucket.
    overflow: u64,
    /// The largest wait seen, tracked exactly so the overflow tail has a value.
    max_us: u64,
    /// Sum of every wait, for the mean.
    sum_us: u128,
    /// Waits recorded.
    count: u64,
}

impl WaitHistogram {
    /// A histogram of `buckets` buckets, each `width_us` microseconds wide.
    ///
    /// Both are clamped to at least one, so a caller cannot build a histogram
    /// that divides by zero or has nowhere to put a sample.
    #[must_use]
    pub fn new(width_us: u64, buckets: usize) -> Self {
        Self {
            width_us: width_us.max(1),
            counts: vec![0; buckets.max(1)],
            overflow: 0,
            max_us: 0,
            sum_us: 0,
            count: 0,
        }
    }

    /// Records one wait.
    pub fn record(&mut self, wait_us: u64) {
        let index = usize::try_from(wait_us / self.width_us).unwrap_or(usize::MAX);
        match self.counts.get_mut(index) {
            Some(slot) => *slot += 1,
            None => self.overflow += 1,
        }
        self.max_us = self.max_us.max(wait_us);
        self.sum_us += u128::from(wait_us);
        self.count += 1;
    }

    /// Waits recorded.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Waits that landed past the last bucket.
    #[must_use]
    pub const fn overflow(&self) -> u64 {
        self.overflow
    }

    /// The `numerator/denominator` quantile, in milliseconds.
    ///
    /// Nearest-rank, in integers throughout: the answer is the smallest bucket
    /// whose cumulative count reaches `ceil(q·n)`, reported at that bucket's
    /// upper edge but never above the largest wait actually seen. Rounding up
    /// to the edge means the figure is never smaller than the truth, and
    /// clamping to the observed maximum means it is never larger than something
    /// that happened.
    ///
    /// `None` when nothing was recorded: an empty distribution has no quantile,
    /// and returning zero would put a latency nobody observed into a record.
    #[must_use]
    pub fn quantile(&self, numerator: u64, denominator: u64) -> Option<f64> {
        if self.count == 0 || denominator == 0 {
            return None;
        }
        let numerator = numerator.min(denominator);
        // ceil(numerator * count / denominator), with at least rank one so the
        // zeroth quantile asks for the first sample rather than for none.
        let rank = numerator.saturating_mul(self.count).div_ceil(denominator);
        let rank = rank.max(1);
        let mut seen = 0u64;
        for (index, count) in self.counts.iter().enumerate() {
            seen += *count;
            if seen >= rank {
                let edge = u64::try_from(index + 1).unwrap_or(u64::MAX);
                let upper = edge.saturating_mul(self.width_us).min(self.max_us);
                return Some(us_to_ms(upper));
            }
        }
        Some(us_to_ms(self.max_us))
    }

    /// The fiftieth percentile, in milliseconds.
    #[must_use]
    pub fn p50(&self) -> Option<f64> {
        self.quantile(50, 100)
    }

    /// The ninety-ninth percentile, in milliseconds. This is the gated one.
    #[must_use]
    pub fn p99(&self) -> Option<f64> {
        self.quantile(99, 100)
    }

    /// The thousandth quantile, in milliseconds.
    #[must_use]
    pub fn p999(&self) -> Option<f64> {
        self.quantile(999, 1000)
    }

    /// The mean wait in milliseconds, or `None` when nothing was recorded.
    #[must_use]
    pub fn mean_ms(&self) -> Option<f64> {
        (self.count > 0).then(|| {
            let total = u64::try_from(self.sum_us).unwrap_or(u64::MAX);
            us_to_ms(total) / widen(self.count)
        })
    }

    /// The largest wait in milliseconds, or `None` when nothing was recorded.
    #[must_use]
    pub fn max_ms(&self) -> Option<f64> {
        (self.count > 0).then(|| us_to_ms(self.max_us))
    }

    /// Folds the distribution and the tier's own counters into a record
    /// section.
    ///
    /// Returns `None` when no wait was recorded: a section of zeroes would read
    /// as a tier that never waited rather than as one that was never measured.
    #[must_use]
    pub fn section(&self, counters: TierCounters) -> Option<PleSection> {
        Some(PleSection {
            wait_mean_ms: self.mean_ms()?,
            wait_p50_ms: self.p50()?,
            wait_p99_ms: self.p99()?,
            wait_p999_ms: self.p999()?,
            wait_max_ms: self.max_ms()?,
            hit_rate: counters.hit_rate(),
            read_mean_ms: counters.read_mean_ms(),
            waits: self.count,
            reference: None,
        })
    }
}

/// The tier's own counters, as a record section needs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierCounters {
    /// Rows asked for, repeats included.
    pub lookups: u64,
    /// Rows served without a read.
    pub hits: u64,
    /// Positional reads issued.
    pub reads: u64,
    /// Cumulative time those reads took, in microseconds.
    pub read_us: u64,
}

impl TierCounters {
    /// Hits over lookups, and zero when nothing was asked for.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            widen(self.hits) / widen(self.lookups)
        }
    }

    /// Mean milliseconds per read, and zero when none were issued.
    #[must_use]
    pub fn read_mean_ms(&self) -> f64 {
        if self.reads == 0 {
            0.0
        } else {
            us_to_ms(self.read_us) / widen(self.reads)
        }
    }
}

/// The fetch-latency section of a record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PleSection {
    /// Mean wait, in milliseconds.
    pub wait_mean_ms: f64,
    /// Median wait, recorded for context.
    pub wait_p50_ms: f64,
    /// Ninety-ninth percentile wait: the gated tail figure.
    pub wait_p99_ms: f64,
    /// Thousandth-quantile wait, recorded for context.
    pub wait_p999_ms: f64,
    /// Largest wait seen. Recorded, never gated.
    pub wait_max_ms: f64,
    /// Rows served from the cache, as a fraction of rows asked for.
    pub hit_rate: f64,
    /// Mean milliseconds per positional read.
    pub read_mean_ms: f64,
    /// Waits the distribution was built from, so a reader can tell a thin
    /// sample from a thick one.
    pub waits: u64,
    /// The reference implementation's own figures, when they were taken on the
    /// same machine and checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<ReferenceStats>,
}

/// The metrics the reference implementation can emit, as it emits them.
///
/// No percentile: it does not compute one, so this block does not claim one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceStats {
    /// Mean wait, in milliseconds.
    pub mean_wait_ms: f64,
    /// Largest wait, in milliseconds. Recorded, never gated.
    pub max_wait_ms: f64,
    /// Cache hit rate, as a percentage, in the reference's own units.
    pub hit_pct: f64,
    /// Mean milliseconds per read.
    pub read_mean_ms: f64,
    /// How this block was produced, named so a reader can reproduce it.
    pub source: String,
}

/// Grades one fetch-latency section against the recorded baseline for it.
///
/// The mean is also compared against the reference block when both sides carry
/// one; the maximum is compared by neither, by design.
pub fn grade_ple(record: &PleSection, baseline: &PleSection, report: &mut Report) {
    report.comparisons.push(Comparison {
        name: "fetch wait mean ms".to_owned(),
        baseline: baseline.wait_mean_ms,
        measured: record.wait_mean_ms,
    });
    report.comparisons.push(Comparison {
        name: "fetch wait p99 ms".to_owned(),
        baseline: baseline.wait_p99_ms,
        measured: record.wait_p99_ms,
    });
    if let Some(reference) = record.reference.as_ref() {
        report.comparisons.push(Comparison {
            name: "fetch wait mean ms against the reference".to_owned(),
            baseline: reference.mean_wait_ms,
            measured: record.wait_mean_ms,
        });
    }
}

/// Microseconds as milliseconds.
fn us_to_ms(us: u64) -> f64 {
    widen(us) / 1000.0
}

/// A 64-bit count as a float, in halves.
///
/// Written out rather than cast because a cast from `u64` is the kind of
/// silent narrowing this workspace does not allow anywhere; in halves the
/// conversion is exact for every count below two to the fifty-third, which is
/// every count any of these histograms will ever hold.
fn widen(value: u64) -> f64 {
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(value & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}
