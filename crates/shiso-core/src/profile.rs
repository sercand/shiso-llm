//! GPU time per launch, for finding where a pass spends it.
//!
//! A batching backend submits hundreds of launches as one unit of work, and the
//! device reports time per submission, not per launch. So while a region is
//! open a backend submits every launch on its own and reads that submission's
//! GPU time. The serialisation costs the overlap and the submission gap a real
//! pass does not pay, which is why every report carries the region's wall time
//! beside the sum of its launches: the gap between the two is the measurement's
//! own cost, and a batched run without a region is the number a user sees.
//!
//! With no region open a backend checks one flag per launch and otherwise
//! batches exactly as it always has.

use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::time::Instant;

use serde_json::{Value, json};

use crate::backend::{Extent3, FnConst};
use crate::error::CoreResult;

/// What one timed launch is filed under; launches with equal keys are one row.
///
/// The constants are part of the key because one entry built with two
/// specialization tables is two pipelines with two costs, and the geometry is
/// because the same kernel at a decode shape and at a prefill shape is two
/// questions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaunchKey {
    /// The kernel entry point, or the backend method for a copy.
    pub entry: &'static str,
    /// Specialization constants the pipeline was built with.
    pub consts: &'static [FnConst],
    /// Threadgroups (blocks) launched; zero for a copy.
    pub grid: Extent3,
    /// Threads per threadgroup; zero for a copy.
    pub block: Extent3,
}

impl LaunchKey {
    /// The key for device work that is not a kernel, such as a blit copy.
    #[must_use]
    pub const fn copy(op: &'static str) -> Self {
        Self {
            entry: op,
            consts: &[],
            grid: Extent3::new(0, 0, 0),
            block: Extent3::new(0, 0, 0),
        }
    }
}

/// One launch's GPU time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchSample {
    /// What ran.
    pub key: LaunchKey,
    /// Nanoseconds between the device starting the launch and finishing it.
    pub gpu_ns: u64,
}

/// A backend that can time each launch on its own.
///
/// Reached through [`crate::Backend::as_profiler`]. Most callers want
/// [`ProfileRegion`], which pairs the two calls and cannot be left open.
pub trait Profiler: Send + Sync + fmt::Debug {
    /// Drains queued work, so none of it is counted, and starts timing.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when a region is already open or the
    /// drain fails.
    fn start(&self) -> CoreResult<()>;

    /// Drains, stops timing and returns every sample since
    /// [`Profiler::start`] in launch order. Batching resumes with the next
    /// launch.
    ///
    /// # Errors
    ///
    /// [`crate::CoreError::Backend`] when no region is open or the drain
    /// fails. The region is closed either way.
    fn stop(&self) -> CoreResult<Vec<LaunchSample>>;
}

/// An open profile region that closes itself.
///
/// Dropping it without [`ProfileRegion::finish`] stops the profiler and
/// discards the samples, so a caller that leaves early through `?` or a panic
/// cannot leave the backend submitting launches one at a time for the rest of
/// the process.
#[derive(Debug)]
pub struct ProfileRegion<'a> {
    profiler: &'a dyn Profiler,
    started: Instant,
    open: bool,
}

impl<'a> ProfileRegion<'a> {
    /// Starts `profiler` and opens the region.
    ///
    /// # Errors
    ///
    /// Whatever [`Profiler::start`] refuses, including a region already open.
    pub fn open(profiler: &'a dyn Profiler) -> CoreResult<Self> {
        profiler.start()?;
        Ok(Self {
            profiler,
            started: Instant::now(),
            open: true,
        })
    }

    /// Stops the profiler and aggregates what it timed.
    ///
    /// # Errors
    ///
    /// Whatever [`Profiler::stop`] refuses.
    pub fn finish(mut self) -> CoreResult<ProfileReport> {
        self.open = false;
        let samples = self.profiler.stop()?;
        Ok(ProfileReport::from_samples(
            &samples,
            elapsed_ns(self.started),
        ))
    }
}

impl Drop for ProfileRegion<'_> {
    fn drop(&mut self) {
        if self.open
            && let Err(error) = self.profiler.stop()
        {
            tracing::warn!(error = %error.chain(), "closing an abandoned profile region failed");
        }
    }
}

/// Nanoseconds since `started`, saturating at `u64::MAX`.
#[must_use]
pub fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// One row of a report: every launch filed under one [`LaunchKey`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelTiming {
    /// What ran.
    pub key: LaunchKey,
    /// How many times.
    pub launches: u32,
    /// Sum of their GPU times.
    pub total_ns: u64,
    /// The fastest launch.
    pub min_ns: u64,
    /// The lower median: for an even count, the smaller of the middle two, so
    /// the value is always one that was measured.
    pub median_ns: u64,
    /// The slowest launch.
    pub max_ns: u64,
}

/// What one region timed, aggregated per [`LaunchKey`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProfileReport {
    /// One row per key, most total time first.
    pub kernels: Vec<KernelTiming>,
    /// Launches timed in the region.
    pub launches: u32,
    /// Sum of every launch's GPU time. Exceeds a batched run of the same work
    /// by the overlap and submission gaps the serialisation added.
    pub serialized_ns: u64,
    /// Host time from opening the region to closing it.
    pub wall_ns: u64,
}

impl ProfileReport {
    /// Groups `samples` by key and orders the rows by total time, ties broken
    /// by entry name and then grid so two runs print in the same order.
    #[must_use]
    pub fn from_samples(samples: &[LaunchSample], wall_ns: u64) -> Self {
        let mut by_key: HashMap<LaunchKey, Vec<u64>> = HashMap::new();
        for sample in samples {
            by_key.entry(sample.key).or_default().push(sample.gpu_ns);
        }
        let mut kernels: Vec<KernelTiming> = by_key
            .into_iter()
            .map(|(key, mut ns)| {
                ns.sort_unstable();
                KernelTiming {
                    key,
                    launches: u32::try_from(ns.len()).unwrap_or(u32::MAX),
                    total_ns: ns.iter().sum(),
                    min_ns: ns[0],
                    median_ns: ns[(ns.len() - 1) / 2],
                    max_ns: ns[ns.len() - 1],
                }
            })
            .collect();
        kernels.sort_by(|a, b| {
            b.total_ns
                .cmp(&a.total_ns)
                .then_with(|| a.key.entry.cmp(b.key.entry))
                .then_with(|| extent(a.key.grid).cmp(&extent(b.key.grid)))
        });
        Self {
            launches: u32::try_from(samples.len()).unwrap_or(u32::MAX),
            serialized_ns: samples.iter().map(|s| s.gpu_ns).sum(),
            wall_ns,
            kernels,
        }
    }

    /// The `top` heaviest rows as a fixed-width table, with a header line
    /// giving the serialized total against the wall time.
    #[must_use]
    pub fn table(&self, top: usize) -> String {
        let mut out = String::new();
        let whole = self.serialized_ns.max(1);
        writeln!(
            out,
            "{} launches over {} kernels: serialized {:.3} ms, wall {:.3} ms",
            self.launches,
            self.kernels.len(),
            ms(self.serialized_ns),
            ms(self.wall_ns),
        )
        .expect("writing to a String cannot fail");
        writeln!(
            out,
            "{:>6} {:>10} {:>8} {:>11} {:>11}  kernel [grid / block]",
            "%", "total ms", "launches", "median us", "max us",
        )
        .expect("writing to a String cannot fail");
        for row in self.kernels.iter().take(top) {
            writeln!(
                out,
                "{:>6.2} {:>10.3} {:>8} {:>11.2} {:>11.2}  {}",
                percent(row.total_ns, whole),
                ms(row.total_ns),
                row.launches,
                us(row.median_ns),
                us(row.max_ns),
                describe(&row.key),
            )
            .expect("writing to a String cannot fail");
        }
        if self.kernels.len() > top {
            writeln!(out, "  ... {} more kernels", self.kernels.len() - top)
                .expect("writing to a String cannot fail");
        }
        out
    }

    /// The report as JSON, the one schema every writer of a profile file uses.
    ///
    /// Integers only, nanoseconds throughout, so a reader never has to guess a
    /// unit or a rounding.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let kernels: Vec<Value> = self
            .kernels
            .iter()
            .map(|row| {
                json!({
                    "entry": row.key.entry,
                    "consts": consts_text(row.key.consts),
                    "grid": extent(row.key.grid),
                    "block": extent(row.key.block),
                    "launches": row.launches,
                    "total_ns": row.total_ns,
                    "min_ns": row.min_ns,
                    "median_ns": row.median_ns,
                    "max_ns": row.max_ns,
                })
            })
            .collect();
        json!({
            "schema": 1,
            "launches": self.launches,
            "serialized_ns": self.serialized_ns,
            "wall_ns": self.wall_ns,
            "kernels": kernels,
        })
    }
}

/// An extent as an ordered triple, for sorting and for JSON.
fn extent(e: Extent3) -> [u32; 3] {
    [e.x, e.y, e.z]
}

/// `index=value` pairs, comma-separated; empty for an unspecialized entry.
fn consts_text(consts: &[FnConst]) -> String {
    consts
        .iter()
        .map(|c| format!("{}={:?}", c.index, c.value))
        .collect::<Vec<_>>()
        .join(",")
}

/// `entry<consts> [gxgygz / bxbybz]`, the label a table row carries.
fn describe(key: &LaunchKey) -> String {
    let consts = consts_text(key.consts);
    let consts = if consts.is_empty() {
        String::new()
    } else {
        format!("<{consts}>")
    };
    let [gx, gy, gz] = extent(key.grid);
    let [bx, by, bz] = extent(key.block);
    format!("{}{consts} [{gx}x{gy}x{gz} / {bx}x{by}x{bz}]", key.entry)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "nanosecond counts below 2^53 convert exactly, and a table prints three places"
)]
fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

#[expect(
    clippy::cast_precision_loss,
    reason = "nanosecond counts below 2^53 convert exactly, and a table prints two places"
)]
fn us(ns: u64) -> f64 {
    ns as f64 / 1e3
}

#[expect(clippy::cast_precision_loss, reason = "a share printed to two places")]
fn percent(part: u64, whole: u64) -> f64 {
    part as f64 * 100.0 / whole as f64
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::backend::FnConstValue;
    use crate::error::CoreError;

    const SPEC: [FnConst; 1] = [FnConst {
        index: 600,
        value: FnConstValue::I16(2),
    }];

    fn key(entry: &'static str, grid: u32) -> LaunchKey {
        LaunchKey {
            entry,
            consts: &[],
            grid: Extent3::linear(grid),
            block: Extent3::linear(32),
        }
    }

    fn sample(key: LaunchKey, gpu_ns: u64) -> LaunchSample {
        LaunchSample { key, gpu_ns }
    }

    #[test]
    fn rows_group_by_key_and_sort_by_total_time() {
        let samples = [
            sample(key("small", 1), 10),
            sample(key("big", 1), 100),
            sample(key("small", 1), 30),
            sample(key("big", 1), 300),
            sample(key("big", 1), 200),
        ];
        let report = ProfileReport::from_samples(&samples, 1_000);
        assert_eq!(report.launches, 5);
        assert_eq!(report.serialized_ns, 640);
        assert_eq!(report.wall_ns, 1_000);
        assert_eq!(report.kernels.len(), 2);
        let big = &report.kernels[0];
        assert_eq!(big.key.entry, "big");
        assert_eq!(
            (
                big.launches,
                big.total_ns,
                big.min_ns,
                big.median_ns,
                big.max_ns
            ),
            (3, 600, 100, 200, 300)
        );
        // An even count takes the lower of the middle two.
        assert_eq!(report.kernels[1].median_ns, 10);
    }

    #[test]
    fn geometry_and_constants_split_rows() {
        let mut specialized = key("k", 1);
        specialized.consts = &SPEC;
        let samples = [
            sample(key("k", 1), 5),
            sample(key("k", 2), 5),
            sample(specialized, 5),
        ];
        let report = ProfileReport::from_samples(&samples, 0);
        assert_eq!(report.kernels.len(), 3);
    }

    #[test]
    fn the_table_shows_the_top_rows_and_counts_the_rest() {
        let samples = [
            sample(key("a", 1), 3_000),
            sample(key("b", 1), 2_000),
            sample(key("c", 1), 1_000),
        ];
        let table = ProfileReport::from_samples(&samples, 10_000).table(2);
        assert!(table.contains("a [1x1x1 / 32x1x1]"), "{table}");
        assert!(table.contains("b [1x1x1 / 32x1x1]"), "{table}");
        assert!(!table.contains("c [1x1x1"), "{table}");
        assert!(table.contains("1 more kernels"), "{table}");
    }

    #[test]
    fn json_is_nanoseconds_with_a_schema_number() {
        let mut specialized = key("k", 4);
        specialized.consts = &SPEC;
        let json = ProfileReport::from_samples(&[sample(specialized, 7)], 9).to_json();
        assert_eq!(json["schema"], 1);
        assert_eq!(json["serialized_ns"], 7);
        assert_eq!(json["wall_ns"], 9);
        assert_eq!(json["kernels"][0]["entry"], "k");
        assert_eq!(json["kernels"][0]["consts"], "600=I16(2)");
        assert_eq!(json["kernels"][0]["grid"], json!([4, 1, 1]));
    }

    /// A profiler that records how often it was started and stopped.
    #[derive(Debug, Default)]
    struct Fake {
        calls: Mutex<(u32, u32)>,
    }

    impl Profiler for Fake {
        fn start(&self) -> CoreResult<()> {
            self.calls.lock().expect("not poisoned").0 += 1;
            Ok(())
        }
        fn stop(&self) -> CoreResult<Vec<LaunchSample>> {
            self.calls.lock().expect("not poisoned").1 += 1;
            Ok(vec![sample(key("k", 1), 11)])
        }
    }

    #[test]
    fn finish_stops_once_and_reports_the_samples() {
        let fake = Fake::default();
        let report = ProfileRegion::open(&fake)
            .expect("opens")
            .finish()
            .expect("finishes");
        assert_eq!(report.serialized_ns, 11);
        assert_eq!(*fake.calls.lock().expect("not poisoned"), (1, 1));
    }

    #[test]
    fn a_dropped_region_stops_the_profiler() {
        let fake = Fake::default();
        drop(ProfileRegion::open(&fake).expect("opens"));
        assert_eq!(*fake.calls.lock().expect("not poisoned"), (1, 1));
    }

    /// A profiler that refuses to start, as a backend with a region open does.
    #[derive(Debug)]
    struct Busy;

    impl Profiler for Busy {
        fn start(&self) -> CoreResult<()> {
            Err(CoreError::Unsupported {
                what: "a second region",
                backend: "fake",
            })
        }
        fn stop(&self) -> CoreResult<Vec<LaunchSample>> {
            panic!("a region that never opened must not be stopped");
        }
    }

    #[test]
    fn a_region_that_did_not_open_never_stops() {
        assert!(ProfileRegion::open(&Busy).is_err());
    }
}
