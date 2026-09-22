//! Aggregate decode throughput by stream count, against a resident checkpoint.
//!
//! The in-process twin of `scripts/streams.py`, and it exists because that
//! script cannot reach every family this workspace loads: the server's factory
//! has no arm for `qwen3_5_moe`, so an HTTP measurement of the `MoE` family is
//! not a number that happens to be missing - it is a number nothing can take.
//! This binary opens the checkpoint through `Rig` instead, which both
//! families reach, and reports the same four stream counts.
//!
//! What it reports is the aggregate, for the reason the script's header gives:
//! a batched decode step that is really a loop divides the per-stream rate by
//! the stream count and leaves the aggregate flat. Per-stream is printed beside
//! it because the ratio between the two is what says whether the batch is real.
//!
//!   `cargo run --release -p shiso-bench --bin streams -- <checkpoint> [config]`
//!
//! The first argument is a GGUF container or a snapshot directory; a directory
//! is the `MoE` family and a file is the dense one, which is the same rule
//! `Rig::dispatch` applies. The second is the config document, which a
//! container needs and a directory carries: this family publishes its config in
//! the MLX sidecar rather than beside the GGUF, so defaulting it to the
//! container's own parent would name a file that is not there.

use std::path::PathBuf;

use anyhow::{Result, bail};

use shiso_bench::decode::{PREFILL_TOKENS, Rig, STREAM_COUNTS, decode_batched_ms_per_step};
use shiso_bench::measure::{Protocol, tokens_per_sec};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) else {
        bail!("usage: streams <checkpoint file or snapshot directory> [config.json]");
    };
    let protocol = Protocol::single();
    let mut rig = if path.is_dir() {
        Rig::open_qwen35moe(&path)?
    } else {
        let Some(config) = std::env::args_os().nth(2).map(PathBuf::from) else {
            bail!("a container needs its config document: streams <checkpoint> <config.json>");
        };
        Rig::open_cuda(&path, &config)?
    };
    // Held until every count has run: the numbers are only comparable if they
    // came off one load of one checkpoint, and reloading between counts would
    // put a cold page cache in the middle of the series.
    let mut measured = Vec::with_capacity(STREAM_COUNTS.len());
    for rows in counts() {
        let ms = decode_batched_ms_per_step(&mut rig, protocol, rows)?;
        measured.push((rows, ms));
    }
    rig.teardown()?;
    report(&measured);
    Ok(())
}

/// The stream counts this run reports.
///
/// `STREAM_COUNTS` unless the environment names others, as a comma-separated
/// list. One count at a time is what a profiler needs: a run that decoded at
/// four widths would hand `nsys` one kernel summary covering all four, and the
/// per-width number would have to be guessed back out of it.
fn counts() -> Vec<u32> {
    let Ok(list) = std::env::var("SHISO_STREAMS") else {
        return STREAM_COUNTS.to_vec();
    };
    list.split(',')
        .filter_map(|field| field.trim().parse().ok())
        .collect()
}

/// The four lines this binary exists to print.
#[expect(
    clippy::print_stdout,
    reason = "the measured rates are this runner's output, and a caller reads \
              them from stdout"
)]
fn report(measured: &[(u32, f64)]) {
    for (rows, ms) in measured {
        let aggregate = tokens_per_sec(*rows, *ms);
        let each = tokens_per_sec(1, *ms);
        println!(
            "streams={rows}  aggregate {aggregate:6.2} tok/s  step {ms:7.2} ms  \
             per-stream {each:6.2} tok/s  prefill_tokens={PREFILL_TOKENS}"
        );
    }
}
