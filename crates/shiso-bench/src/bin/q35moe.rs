//! What the Qwen 3.5 `MoE` checkpoint costs on the local device, once.
//!
//! The sibling of `qwen35.rs` for the sparse-feed-forward family. It reads a
//! snapshot directory rather than a container file, because this family ships
//! three shards and two config documents.
//!
//! This does not write a baseline, for the reason `qwen35.rs` states: a
//! baseline any run can overwrite is not a baseline.

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
    let mut rig = Rig::open_qwen35moe(&dir())?;
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
        "ctx={CTX} prefill_tokens={PREFILL_TOKENS} \
         prefill_tokens_per_second={prefill_tokens_per_sec:.4} decode_tokens={} \
         decode_tokens_per_second={decode_tokens_per_sec:.4}",
        protocol.decode_steps,
    );
}

/// The snapshot directory the run measures.
fn dir() -> PathBuf {
    PathBuf::from(
        std::env::var("SHISO_TEST_MODEL_QWEN35MOE")
            .expect("SHISO_TEST_MODEL_QWEN35MOE must name the snapshot directory"),
    )
}
