//! Single-kernel GPU time at realistic shapes, for A/B work on one kernel.
//!
//! Each case is launched a few times to warm up and then `--reps` times inside
//! a profile region, one submission per launch, so the number is the kernel's
//! own GPU time with no batching around it. Compare two variants by
//! registering both as cases and filtering on a shared prefix, or by running
//! one case before and after an MSL edit with `SHISO_METAL_SOURCE_DIR`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde_json::json;
use shiso_core::Backend;
use shiso_kerneltest::Gpu;
use shiso_kerneltest::bench::{CASES, time};
use shiso_metal::MetalBackend;

/// CLI arguments.
#[derive(Debug, Parser)]
#[command(
    name = "kernel_bench",
    about = "GPU time of single kernels at realistic shapes"
)]
struct Args {
    /// Only cases whose name contains this.
    filter: Option<String>,
    /// Timed launches per case.
    #[arg(long, default_value_t = 200)]
    reps: u32,
    /// Lists the cases and exits.
    #[arg(long)]
    list: bool,
    /// Bandwidth to report each case against, in GB/s. The default is the
    /// published figure for an M4; pass this machine's own.
    #[arg(long, default_value_t = 120.0)]
    peak_gbps: f64,
    /// Matmul throughput to report against, in TFLOP/s. The default is the
    /// fp16 ceiling measured on an M4 on 2026-10-10.
    #[arg(long, default_value_t = 3.64)]
    peak_tflops: f64,
    /// Writes every result as JSON here.
    #[arg(long)]
    json: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let selected: Vec<_> = CASES
        .iter()
        .filter(|case| args.filter.as_deref().is_none_or(|f| case.name.contains(f)))
        .collect();
    if args.list {
        for case in &selected {
            out(case.name);
        }
        return Ok(());
    }
    if selected.is_empty() {
        bail!("no case matches {:?}; --list shows them", args.filter);
    }
    if args.reps == 0 {
        bail!("--reps must be greater than zero");
    }
    let metal = MetalBackend::new().context("the Metal device")?;
    let device: Arc<dyn Backend> = metal.clone();
    let gpu = Gpu::new(device, Box::new(metal));

    out(&format!(
        "{:<34} {:>10} {:>10} {:>7} {:>8} {:>6} {:>8} {:>6}",
        "case", "median us", "min us", "spread", "GB/s", "%bw", "TFLOP/s", "%peak"
    ));
    let mut rows = Vec::with_capacity(selected.len());
    for case in selected {
        let fixture = (case.build)(&gpu);
        let stat = time(&gpu, &fixture, args.reps).with_context(|| case.name)?;
        let gbps = stat.gb_per_s(fixture.bytes);
        let tflops = stat.tflops(fixture.flops);
        let flops_cols = if fixture.flops == 0 {
            format!("{:>8} {:>6}", "-", "-")
        } else {
            format!("{tflops:>8.3} {:>5.1}%", tflops / args.peak_tflops * 100.0)
        };
        out(&format!(
            "{:<34} {:>10.2} {:>10.2} {:>6.1}% {gbps:>8.1} {:>5.1}% {flops_cols}",
            case.name,
            ns_to_us(stat.median_ns),
            ns_to_us(stat.min_ns),
            stat.spread() * 100.0,
            gbps / args.peak_gbps * 100.0,
        ));
        rows.push(json!({
            "case": case.name,
            "entry": stat.entry,
            "launches": stat.launches,
            "median_ns": stat.median_ns,
            "min_ns": stat.min_ns,
            "max_ns": stat.max_ns,
            "bytes": fixture.bytes,
            "flops": fixture.flops,
        }));
    }
    if let Some(path) = args.json.as_deref() {
        let device = shiso_metal::device_facts().map(|facts| facts.name);
        let body = json!({ "schema": 1, "device": device, "reps": args.reps, "cases": rows });
        std::fs::write(path, serde_json::to_string_pretty(&body)?)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "nanoseconds below 2^53 convert exactly"
)]
fn ns_to_us(ns: u64) -> f64 {
    ns as f64 / 1e3
}

#[expect(
    clippy::print_stdout,
    reason = "the table is this runner's output, read from stdout"
)]
fn out(line: &str) {
    println!("{line}");
}
