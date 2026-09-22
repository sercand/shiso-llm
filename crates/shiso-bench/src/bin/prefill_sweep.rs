//! Prefill throughput by prompt length, against a resident checkpoint.
//!
//! The in-process twin of `scripts/prefill_sweep.py`, and it exists for the
//! reason `crates/shiso-bench/src/bin/streams.rs` gives: the server's factory
//! has no arm for `qwen3_5_moe`, so an HTTP sweep of the sparse family is not
//! a number that happens to be missing - it is a number nothing can take. This
//! binary opens the checkpoint through `Rig`, which both families reach.
//!
//! Three lengths, and each of them is here for a reason. 64 is where every
//! committed baseline in `docs/handoff/cuda-decode-perf.md` sits, so a new
//! number can be read against the old one. 512 is what `llama-bench`'s `pp512`
//! runs and is therefore the only like-for-like comparison against llama.cpp
//! this box affords. 2048 is nearer the length a real prompt arrives at.
//!
//! Every length is reported as a **slope** and not a division. See
//! [`shiso_bench::prefill::prefill_slope`].
//!
//!   `cargo run --release -p shiso-bench --bin prefill_sweep -- <checkpoint> [config]`
//!
//! The first argument is a GGUF container or a snapshot directory; a directory
//! is the `MoE` family and a file is the dense one, which is the rule
//! `Rig::dispatch` applies. The second is the config document, which a
//! container needs and a directory carries.

use std::path::PathBuf;

use anyhow::{Result, bail};

use shiso_bench::decode::{CTX, Rig};
use shiso_bench::measure::Protocol;
use shiso_bench::prefill::{SLOPE_BASE_TOKENS, Slope, prefill_slope};

/// The prompt lengths this sweep reports.
const LENGTHS: [u32; 3] = [64, 512, 2048];

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) else {
        bail!("usage: prefill_sweep <checkpoint file or snapshot directory> [config.json]");
    };
    // Warmed rather than single: this family's weights are a multi-gigabyte
    // mapping and the first repetition times the operating system's first
    // touch of every page it reads.
    let protocol = Protocol::warmed();
    let mut rig = if path.is_dir() {
        Rig::open_qwen35moe(&path)?
    } else {
        let Some(config) = std::env::args_os().nth(2).map(PathBuf::from) else {
            bail!(
                "a container needs its config document: prefill_sweep <checkpoint> <config.json>"
            );
        };
        Rig::open_cuda(&path, &config)?
    };
    // Held until every length has run: the numbers are only comparable if they
    // came off one load of one checkpoint.
    let mut measured = Vec::with_capacity(lengths().len());
    for tokens in lengths() {
        measured.push(prefill_slope(&mut rig, protocol, tokens)?);
    }
    rig.teardown()?;
    report(&measured);
    Ok(())
}

/// The lengths this run reports, from the environment or [`LENGTHS`].
///
/// One length at a time is what a profiler needs: a run that prefilled at
/// three widths would hand `nsys` one kernel summary covering all three.
fn lengths() -> Vec<u32> {
    let Ok(list) = std::env::var("SHISO_PREFILL_LENGTHS") else {
        return LENGTHS.to_vec();
    };
    list.split(',')
        .filter_map(|field| field.trim().parse().ok())
        .collect()
}

/// The lines this binary exists to print.
#[expect(
    clippy::print_stdout,
    reason = "the measured rates are this runner's output, and a caller reads \
              them from stdout"
)]
fn report(measured: &[Slope]) {
    for slope in measured {
        println!(
            "ctx={CTX} prefill_tokens={} prefill_ms={:.3} base_tokens={SLOPE_BASE_TOKENS} \
             base_ms={:.3} prefill_tokens_per_second={:.2} naive_tokens_per_second={:.2}",
            slope.tokens,
            slope.ms,
            slope.base_ms,
            slope.tokens_per_sec(),
            slope.naive_tokens_per_sec(),
        );
    }
}
