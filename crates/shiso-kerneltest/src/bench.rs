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
    Case {
        name: "q4e_mxfp8_gemv_10240x2560",
        build: crate::qwen4exp::mxfp8::bench_gemv_10240x2560,
    },
    Case {
        name: "q4e_mxfp8_gemv_2560x6144",
        build: crate::qwen4exp::mxfp8::bench_gemv_2560x6144,
    },
    Case {
        name: "q4e_mxfp8_gemv_6144x2560",
        build: crate::qwen4exp::mxfp8::bench_gemv_6144x2560,
    },
    Case {
        name: "q4e_mxfp8_rows2_10240x2560",
        build: crate::qwen4exp::mxfp8::bench_rows2_10240x2560,
    },
    Case {
        name: "q4e_mxfp8_rows2_2560x6144",
        build: crate::qwen4exp::mxfp8::bench_rows2_2560x6144,
    },
    Case {
        name: "q4e_mxfp8_rows5_10240x2560",
        build: crate::qwen4exp::mxfp8::bench_rows5_10240x2560,
    },
    Case {
        name: "q4e_mxfp8_rows5_2560x6144",
        build: crate::qwen4exp::mxfp8::bench_rows5_2560x6144,
    },
    Case {
        name: "q4e_mxfp8_rows8_10240x2560",
        build: crate::qwen4exp::mxfp8::bench_rows8_10240x2560,
    },
    Case {
        name: "q4e_mxfp8_rows8_2560x6144",
        build: crate::qwen4exp::mxfp8::bench_rows8_2560x6144,
    },
    Case {
        name: "q4e_mxfp8_rows5_6144x2560",
        build: crate::qwen4exp::mxfp8::bench_rows5_6144x2560,
    },
    Case {
        name: "q4e_mxfp8_rows5_12288x2560",
        build: crate::qwen4exp::mxfp8::bench_rows5_12288x2560,
    },
    Case {
        name: "q4e_mxfp8_rows5_512x2560",
        build: crate::qwen4exp::mxfp8::bench_rows5_512x2560,
    },
    Case {
        name: "q4e_mxfp8_rows5_640x2560",
        build: crate::qwen4exp::mxfp8::bench_rows5_640x2560,
    },
    Case {
        name: "q4e_mxfp8_rows5_48x2560",
        build: crate::qwen4exp::mxfp8::bench_rows5_48x2560,
    },
    Case {
        name: "q4e_mxfp8_gemm2048_10240x2560",
        build: crate::qwen4exp::mxfp8::bench_gemm2048_10240x2560,
    },
    Case {
        name: "q4e_mxfp8_gemm2048_12288x2560",
        build: crate::qwen4exp::mxfp8::bench_gemm2048_12288x2560,
    },
    Case {
        name: "q4e_mxfp8_gemm2048_6144x2560",
        build: crate::qwen4exp::mxfp8::bench_gemm2048_6144x2560,
    },
    Case {
        name: "q4e_mxfp8_gemm2048_2560x6144",
        build: crate::qwen4exp::mxfp8::bench_gemm2048_2560x6144,
    },
    Case {
        name: "q4e_mxfp8_gemm512_10240x2560",
        build: crate::qwen4exp::mxfp8::bench_gemm512_10240x2560,
    },
    Case {
        name: "q4e_bf16_rows1_10240x2560",
        build: crate::qwen4exp::mxfp8::bench_bf16_10240x2560,
    },
    Case {
        name: "q4e_bf16_rows1_2560x6144",
        build: crate::qwen4exp::mxfp8::bench_bf16_2560x6144,
    },
    Case {
        name: "q4e_nvfp4_step_gate_up_t1",
        build: crate::qwen4exp::nvfp4::bench_step_gate_up_t1,
    },
    Case {
        name: "q4e_nvfp4_step_gate_up_t5",
        build: crate::qwen4exp::nvfp4::bench_step_gate_up_t5,
    },
    Case {
        name: "q4e_nvfp4_step_gate_up_t8",
        build: crate::qwen4exp::nvfp4::bench_step_gate_up_t8,
    },
    Case {
        name: "q4e_nvfp4_step_down_t1",
        build: crate::qwen4exp::nvfp4::bench_step_down_t1,
    },
    Case {
        name: "q4e_nvfp4_step_down_t5",
        build: crate::qwen4exp::nvfp4::bench_step_down_t5,
    },
    Case {
        name: "q4e_nvfp4_step_down_t8",
        build: crate::qwen4exp::nvfp4::bench_step_down_t8,
    },
    Case {
        name: "q4e_nvfp4_prompt_gate_up_2048",
        build: crate::qwen4exp::nvfp4::bench_prompt_gate_up_2048,
    },
    Case {
        name: "q4e_nvfp4_prompt_down_2048",
        build: crate::qwen4exp::nvfp4::bench_prompt_down_2048,
    },
    Case {
        name: "q4e_nvfp4_prompt_gate_up_4096",
        build: crate::qwen4exp::nvfp4::bench_prompt_gate_up_4096,
    },
    Case {
        name: "q4e_nvfp4_prompt_down_4096",
        build: crate::qwen4exp::nvfp4::bench_prompt_down_4096,
    },
    Case {
        name: "q4e_bf16_gemv_320x10240",
        build: crate::qwen4exp::bf16::bench_gemv_320x10240,
    },
    Case {
        name: "q4e_bf16_gemv_10240x320",
        build: crate::qwen4exp::bf16::bench_gemv_10240x320,
    },
    Case {
        name: "q4e_bf16_gemv_4x10240",
        build: crate::qwen4exp::bf16::bench_gemv_4x10240,
    },
    Case {
        name: "q4e_bf16_gemv_512x2560",
        build: crate::qwen4exp::bf16::bench_gemv_512x2560,
    },
    Case {
        name: "q4e_bf16_gemv_10240x2560",
        build: crate::qwen4exp::bf16::bench_gemv_10240x2560,
    },
    Case {
        name: "q4e_bf16_gemv_2560x2560",
        build: crate::qwen4exp::bf16::bench_gemv_2560x2560,
    },
    Case {
        name: "q4e_bf16_gemv_248320x2560",
        build: crate::qwen4exp::bf16::bench_gemv_248320x2560,
    },
    Case {
        name: "q4e_bf16_rows5_320x10240",
        build: crate::qwen4exp::bf16::bench_rows5_320x10240,
    },
    Case {
        name: "q4e_bf16_rows5_10240x320",
        build: crate::qwen4exp::bf16::bench_rows5_10240x320,
    },
    Case {
        name: "q4e_bf16_rows5_512x2560",
        build: crate::qwen4exp::bf16::bench_rows5_512x2560,
    },
    Case {
        name: "q4e_bf16_rows5_10240x2560",
        build: crate::qwen4exp::bf16::bench_rows5_10240x2560,
    },
    Case {
        name: "q4e_bf16_rows8_320x10240",
        build: crate::qwen4exp::bf16::bench_rows8_320x10240,
    },
    Case {
        name: "q4e_bf16_rows8_10240x2560",
        build: crate::qwen4exp::bf16::bench_rows8_10240x2560,
    },
    Case {
        name: "q4e_bf16_rows20_320x10240",
        build: crate::qwen4exp::bf16::bench_rows20_320x10240,
    },
    Case {
        name: "q4e_bf16_gemm2048_320x10240",
        build: crate::qwen4exp::bf16::bench_gemm2048_320x10240,
    },
    Case {
        name: "q4e_bf16_gemm2048_10240x320",
        build: crate::qwen4exp::bf16::bench_gemm2048_10240x320,
    },
    Case {
        name: "q4e_bf16_gemm2048_4x10240",
        build: crate::qwen4exp::bf16::bench_gemm2048_4x10240,
    },
    Case {
        name: "q4e_bf16_gemm_split2048_512x2560",
        build: crate::qwen4exp::bf16::bench_gemm_split2048_512x2560,
    },
    Case {
        name: "q4e_bf16_gemm2048_10240x2560",
        build: crate::qwen4exp::bf16::bench_gemm2048_10240x2560,
    },
    Case {
        name: "q4e_bf16_gemm2048_2560x2560",
        build: crate::qwen4exp::bf16::bench_gemm2048_2560x2560,
    },
    Case {
        name: "q4e_bf16_gemm2048_12288x2560",
        build: crate::qwen4exp::bf16::bench_gemm2048_12288x2560,
    },
    Case {
        name: "q4e_bf16_ported2048_320x10240",
        build: crate::qwen4exp::bf16::bench_ported2048_320x10240,
    },
    Case {
        name: "q4e_bf16_ported2048_512x2560",
        build: crate::qwen4exp::bf16::bench_ported2048_512x2560,
    },
    Case {
        name: "q4e_bf16_ported1_320x10240",
        build: crate::qwen4exp::bf16::bench_ported1_320x10240,
    },
    Case {
        name: "q4e_bf16_ported1_10240x2560",
        build: crate::qwen4exp::bf16::bench_ported1_10240x2560,
    },
    Case {
        name: "q4e_bf16_ported1_248320x2560",
        build: crate::qwen4exp::bf16::bench_ported1_248320x2560,
    },
    Case {
        name: "q4e_bf16_p12_1_320x10240",
        build: crate::qwen4exp::bf16::bench_p12_1_320x10240,
    },
    Case {
        name: "q4e_bf16_p12_1_10240x320",
        build: crate::qwen4exp::bf16::bench_p12_1_10240x320,
    },
    Case {
        name: "q4e_bf16_p12_1_512x2560",
        build: crate::qwen4exp::bf16::bench_p12_1_512x2560,
    },
    Case {
        name: "q4e_bf16_p12_1_10240x2560",
        build: crate::qwen4exp::bf16::bench_p12_1_10240x2560,
    },
    Case {
        name: "q4e_bf16_p12_5_320x10240",
        build: crate::qwen4exp::bf16::bench_p12_5_320x10240,
    },
    Case {
        name: "q4e_bf16_p12_5_10240x2560",
        build: crate::qwen4exp::bf16::bench_p12_5_10240x2560,
    },
    Case {
        name: "vit_q_mxfp8_qkv_4096",
        build: crate::vision::quant::bench_mxfp8_qkv_4096,
    },
    Case {
        name: "vit_q_mxfp8_proj_4096",
        build: crate::vision::quant::bench_mxfp8_proj_4096,
    },
    Case {
        name: "vit_q_nvfp4_fc1_4096",
        build: crate::vision::quant::bench_nvfp4_fc1_4096,
    },
    Case {
        name: "vit_q_nvfp4_fc2_4096",
        build: crate::vision::quant::bench_nvfp4_fc2_4096,
    },
    Case {
        name: "vit_q_nvfp4_merger_1024",
        build: crate::vision::quant::bench_nvfp4_merger_1024,
    },
    Case {
        name: "vit_q_mxfp8_qkv_16384",
        build: crate::vision::quant::bench_mxfp8_qkv_16384,
    },
    Case {
        name: "vit_q_nvfp4_fc1_16384",
        build: crate::vision::quant::bench_nvfp4_fc1_16384,
    },
    Case {
        name: "vit_q_nvfp4_fc2_16384",
        build: crate::vision::quant::bench_nvfp4_fc2_16384,
    },
    Case {
        name: "vit_q_blas_qkv_4096",
        build: crate::vision::quant::bench_blas_qkv_4096,
    },
    Case {
        name: "vit_q_blas_proj_4096",
        build: crate::vision::quant::bench_blas_proj_4096,
    },
    Case {
        name: "vit_q_blas_fc1_4096",
        build: crate::vision::quant::bench_blas_fc1_4096,
    },
    Case {
        name: "vit_q_blas_fc2_4096",
        build: crate::vision::quant::bench_blas_fc2_4096,
    },
    Case {
        name: "vit_q_blas_merger_1024",
        build: crate::vision::quant::bench_blas_merger_1024,
    },
    Case {
        name: "vit_q_blas_qkv_16384",
        build: crate::vision::quant::bench_blas_qkv_16384,
    },
    Case {
        name: "vit_q_blas_fc1_16384",
        build: crate::vision::quant::bench_blas_fc1_16384,
    },
    Case {
        name: "vit_q_blas_fc2_16384",
        build: crate::vision::quant::bench_blas_fc2_16384,
    },
    Case {
        name: "vit_q_mxfp8_proj_196_narrow",
        build: crate::vision::quant::bench_mxfp8_proj_196_narrow,
    },
    Case {
        name: "vit_q_mxfp8_proj_196",
        build: crate::vision::quant::bench_mxfp8_proj_196,
    },
    Case {
        name: "vit_q_nvfp4_fc2_196_narrow",
        build: crate::vision::quant::bench_nvfp4_fc2_196_narrow,
    },
    Case {
        name: "vit_q_blas_proj_196",
        build: crate::vision::quant::bench_blas_proj_196,
    },
    Case {
        name: "vit_q_blas_fc2_196",
        build: crate::vision::quant::bench_blas_fc2_196,
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

/// Batches [`time_wall`] splits its launches into, one sample each.
const WALL_BATCHES: u32 = 11;

/// Times `fixture` by the host clock, for a backend that cannot time a launch:
/// eleven batches of `reps` back-to-back launches, each drained, and
/// each batch's mean launch is one sample.
///
/// Back to back, the queue never runs dry, so a batch's mean is the kernel's
/// own time plus the gap the device leaves between two launches - a few
/// microseconds, which is noise on a kernel of a hundred and a bias on one of
/// ten. The entry name is not known to the host clock and is reported as
/// `"wall clock"`.
///
/// # Errors
///
/// [`CoreError::InvalidValue`] for zero `reps`, and whatever the drain reports.
pub fn time_wall(gpu: &Gpu, fixture: &Fixture, reps: u32) -> CoreResult<BenchStat> {
    if reps == 0 {
        return Err(CoreError::InvalidValue {
            key: "reps",
            value: "0".to_owned(),
        });
    }
    for _ in 0..WARMUP {
        fixture.launch(gpu);
    }
    gpu.device().sync()?;
    let mut samples = Vec::with_capacity(WALL_BATCHES as usize);
    for _ in 0..WALL_BATCHES {
        let start = std::time::Instant::now();
        for _ in 0..reps {
            fixture.launch(gpu);
        }
        gpu.device().sync()?;
        let ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        samples.push(ns / u64::from(reps));
    }
    samples.sort_unstable();
    Ok(BenchStat {
        entry: "wall clock",
        launches: reps.saturating_mul(WALL_BATCHES),
        median_ns: samples[samples.len() / 2],
        min_ns: samples[0],
        max_ns: samples[samples.len() - 1],
    })
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
