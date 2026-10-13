//! Kernel microbenchmarks: one launch at a realistic shape, timed on the
//! device many times.
//!
//! A case builds its buffers once and hands back a [`Fixture`] that launches
//! the kernel and nothing else, so a profile region around N launches is N
//! samples of one kernel. Fixtures live beside the correctness case for the
//! same kernel, because that module already owns the argument block and the
//! launch geometry, and a second copy here would drift from it.
//!
//! Throughput figures come from the bytes and flops the case declares - the
//! minimum traffic the kernel must do, not what it actually does - so a
//! number above the device's bandwidth means the declaration is wrong.

use std::fmt;

use shiso_core::{CoreError, CoreResult, ProfileRegion};

use crate::harness::{Gpu, drain};

/// Launches before the region opens: pipeline build, cache warm-up, clocks.
const WARMUP: u32 = 3;

/// One kernel, ready to launch.
pub struct Fixture {
    launch: Box<dyn Fn(&Gpu)>,
    /// Bytes the launch must move at the least: inputs read once, outputs
    /// written once.
    pub bytes: u64,
    /// Floating-point operations of the launch; `0` for one that is all
    /// traffic.
    pub flops: u64,
}

impl Fixture {
    /// A fixture whose `launch` encodes exactly one kernel.
    pub fn new(bytes: u64, flops: u64, launch: impl Fn(&Gpu) + 'static) -> Self {
        Self {
            launch: Box::new(launch),
            bytes,
            flops,
        }
    }

    /// Encodes the launch.
    ///
    /// # Panics
    ///
    /// When the device refuses it, as every launch in this crate does.
    pub fn launch(&self, gpu: &Gpu) {
        (self.launch)(gpu);
    }
}

impl fmt::Debug for Fixture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fixture")
            .field("bytes", &self.bytes)
            .field("flops", &self.flops)
            .finish_non_exhaustive()
    }
}

/// A named fixture builder.
#[derive(Debug, Clone, Copy)]
pub struct Case {
    /// What the bench prints and filters on.
    pub name: &'static str,
    /// Uploads the buffers and returns the launch.
    pub build: fn(&Gpu) -> Fixture,
}

/// Every case, decode-hot kernels first.
pub const CASES: &[Case] = &[
    Case {
        name: "q4_0_matvec_4096x4096",
        build: crate::dense::q4_0::bench_matvec,
    },
    Case {
        name: "dense_attn_decode_hd256_kv4096",
        build: crate::dense::attn::bench_decode,
    },
    Case {
        name: "swiglu_flat_14336",
        build: crate::generic::glu::bench_swiglu,
    },
    Case {
        name: "rms_norm_4096",
        build: crate::generic::norm::bench_rms_norm,
    },
];

/// What [`time`] measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchStat {
    /// The kernel that ran.
    pub entry: &'static str,
    /// Launches timed.
    pub launches: u32,
    /// The lower median launch.
    pub median_ns: u64,
    /// The fastest launch.
    pub min_ns: u64,
    /// The slowest launch.
    pub max_ns: u64,
}

impl BenchStat {
    /// `bytes` over the median launch, in GB/s.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "byte and ns counts below 2^53")]
    pub fn gb_per_s(&self, bytes: u64) -> f64 {
        bytes as f64 / self.median_ns.max(1) as f64
    }

    /// `flops` over the median launch, in TFLOP/s.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "flop and ns counts below 2^53")]
    pub fn tflops(&self, flops: u64) -> f64 {
        flops as f64 / self.median_ns.max(1) as f64 / 1e3
    }

    /// `max / min - 1`: how far apart the launches were.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "ns counts below 2^53")]
    pub fn spread(&self) -> f64 {
        self.max_ns as f64 / self.min_ns.max(1) as f64 - 1.0
    }
}

/// Launches `fixture` a few (`WARMUP`) times, then `reps` times inside a profile
/// region, and summarises the timed launches.
///
/// # Errors
///
/// - [`CoreError::InvalidValue`] for zero `reps`, or for a fixture that
///   launched other than exactly one kernel.
/// - [`CoreError::Unsupported`] for a backend that cannot time launches.
/// - Whatever the region refuses.
pub fn time(gpu: &Gpu, fixture: &Fixture, reps: u32) -> CoreResult<BenchStat> {
    if reps == 0 {
        return Err(CoreError::InvalidValue {
            key: "reps",
            value: "0".to_owned(),
        });
    }
    let profiler = gpu.device().as_profiler().ok_or(CoreError::Unsupported {
        what: "per-launch timing",
        backend: gpu.backend_name(),
    })?;
    for _ in 0..WARMUP {
        fixture.launch(gpu);
    }
    drain(gpu);
    let region = ProfileRegion::open(profiler)?;
    for _ in 0..reps {
        fixture.launch(gpu);
    }
    let report = region.finish()?;
    let [row] = report.kernels.as_slice() else {
        return Err(CoreError::InvalidValue {
            key: "bench fixture",
            value: format!(
                "{} kernels in one fixture; a case times exactly one",
                report.kernels.len()
            ),
        });
    };
    Ok(BenchStat {
        entry: row.key.entry,
        launches: row.launches,
        median_ns: row.median_ns,
        min_ns: row.min_ns,
        max_ns: row.max_ns,
    })
}
