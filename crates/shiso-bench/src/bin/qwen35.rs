//! What one dense-hybrid checkpoint costs on the local device, once.
//!
//! One pass of the same protocol the recorder uses, printed and not filed. It
//! exists for two callers: a person who wants a number now without committing
//! anything, and the resident-set test, which runs this under an external timer
//! and reads the peak the operating system reports.
//!
//! # Why this does not write a baseline
//!
//! It used to. A baseline that any run can overwrite is not a baseline - the
//! reference moves whenever somebody measures, and a regression becomes
//! invisible by being recorded as the new normal. Records are written by the
//! gate's `record` command, which measures seven times and refuses to write a
//! noisy result, and a record becomes a baseline only when a human copies it.
//!
//! # Why the measurement itself lives elsewhere
//!
//! Every line of timing here is the library's. Two implementations of "how long
//! does a decode step take" would eventually disagree, and the disagreement
//! would look exactly like the thing the gate is for.

use std::path::PathBuf;

use anyhow::Result;

use shiso_bench::decode::{CTX, PREFILL_TOKENS, Rig, decode_ms_per_step};
use shiso_bench::measure::{Protocol, tokens_per_sec};
use shiso_bench::prefill::prefill_ms;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let protocol = Protocol::single();
    let mut rig = Rig::open(&model_path())?;
    let decode_ms = decode_ms_per_step(&mut rig, protocol)?;
    let prefill = prefill_ms(&mut rig, protocol)?;
    rig.teardown()?;

    report(
        tokens_per_sec(PREFILL_TOKENS, prefill),
        tokens_per_sec(1, decode_ms),
    );
    Ok(())
}

/// The one line this binary exists to print.
#[expect(
    clippy::print_stdout,
    reason = "the measured rates are this runner's output, and a caller reads \
              them from stdout"
)]
fn report(prefill_tokens_per_sec: f64, decode_tokens_per_sec: f64) {
    let protocol = Protocol::single();
    println!(
        "ctx={CTX} prefill_tokens={PREFILL_TOKENS} prefill_tokens_per_second={prefill_tokens_per_sec:.4} \
         decode_tokens={} decode_tokens_per_second={decode_tokens_per_sec:.4}",
        protocol.decode_steps,
    );
}

/// The checkpoint the run measures.
fn model_path() -> PathBuf {
    PathBuf::from(
        std::env::var("SHISO_TEST_MODEL")
            .expect("SHISO_TEST_MODEL must name the checkpoint to measure"),
    )
}
