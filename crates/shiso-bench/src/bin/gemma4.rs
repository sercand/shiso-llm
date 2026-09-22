//! What the Gemma 4 E4B checkpoint costs on the local device, once.
//!
//! The dense-hybrid runner beside this one, with two differences that are
//! forced by the family rather than chosen:
//!
//! 1. **The checkpoint's own config document is read.** That family has a shape
//!    profile recorded beside the test vectors and its baselines were taken
//!    through it; this one has no such history, so the geometry comes off the
//!    sidecar the export wrote, exactly as the server reads it.
//! 2. **A whole repetition is thrown away before the timed one.** The weights
//!    are a five-gigabyte mapping and the per-layer table is streamed off the
//!    same file. A first repetition measures the operating system faulting those
//!    pages in - on this machine it came out around three times slower than the
//!    steady state - and a runner that printed it would be publishing a number
//!    about paging under a heading about kernels.
//!
//! # The per-layer table, and why the holding is a flag
//!
//! E4B's per-layer embedding table is larger than the rest of the model put
//! together. It is streamed off the checkpoint with a row cache by default, which
//! is what the server does, and `--ple-resident` holds the whole thing in memory
//! instead. The two are different machines as far as a decode step is concerned,
//! so the holding is named on the command line and never inferred: a run that did
//! not say which one it used produced a number nobody can reproduce. That is also
//! why it is a flag and not an environment variable - `scripts/bench_gemma4.sh`
//! turns `PLE=resident` into this flag, and the flag is the only thing either the
//! bench or the server parses.
//!
//! What the reader did is reported by the model itself at teardown - rows asked
//! for, rows the cache had, and the seconds the rest cost - so `SHISO_LOG=info`
//! is enough to read it, and nothing here instruments the loop it is timing.
//!
//! # Why this writes nothing
//!
//! The same reason the dense-hybrid runner writes nothing: a baseline that any
//! run can overwrite is not a baseline. Records come from the gate's `record`
//! command, which repeats the protocol seven times and refuses a noisy result.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use shiso_bench::decode::{CTX, PREFILL_TOKENS, Rig, decode_ms_per_step};
use shiso_bench::measure::{Protocol, tokens_per_sec};
use shiso_bench::prefill::prefill_ms;
use shiso_models::gemma4::PleOptions;

/// One pass of the protocol over one checkpoint.
#[derive(Debug, Parser)]
#[command(about = "measures one Gemma 4 checkpoint on the local device", long_about = None)]
struct Cli {
    /// The checkpoint to measure. Falls back to `SHISO_TEST_MODEL`.
    #[arg(long)]
    model: Option<PathBuf>,
    /// The checkpoint's own config document. Falls back to `SHISO_TEST_CONFIG`.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Hold the whole per-layer table in memory instead of streaming it.
    ///
    /// A flag and nothing else. An environment variable saying the same thing
    /// would be a second parser for one fact, and the server already parses its
    /// own `--ple-resident` through clap: two spellings of "resident" that
    /// disagreed about what `false` means is exactly the kind of difference that
    /// files a record under the wrong holding.
    #[arg(long)]
    ple_resident: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let model = path_of(cli.model, "SHISO_TEST_MODEL");
    let config = path_of(cli.config, "SHISO_TEST_CONFIG");
    let resident = cli.ple_resident;
    let ple = PleOptions {
        resident,
        ..PleOptions::default()
    };

    let protocol = Protocol::warmed();
    let mut rig = Rig::open_with_ple(&model, &config, ple)?;
    // Discarded, not measured: this is the repetition that pays for the first
    // touch of the mapping and for a row cache nothing has filled yet.
    for _ in 0..protocol.warmup_reps {
        decode_ms_per_step(&mut rig, protocol)?;
        prefill_ms(&mut rig, protocol)?;
        tracing::info!("warmup repetition discarded");
    }
    let decode_ms = decode_ms_per_step(&mut rig, protocol)?;
    let prefill = prefill_ms(&mut rig, protocol)?;
    // The per-layer reader's counters are reported from inside the model here,
    // which is the last moment they exist.
    rig.teardown()?;

    report(
        tokens_per_sec(PREFILL_TOKENS, prefill),
        tokens_per_sec(1, decode_ms),
        resident,
    );
    Ok(())
}

/// The one line this binary exists to print.
#[expect(
    clippy::print_stdout,
    reason = "the measured rates are this runner's output, and a caller reads \
              them from stdout"
)]
fn report(prefill_tokens_per_sec: f64, decode_tokens_per_sec: f64, resident: bool) {
    let protocol = Protocol::warmed();
    let ple = if resident { "resident" } else { "streamed" };
    println!(
        "ctx={CTX} ple={ple} prefill_tokens={PREFILL_TOKENS} \
         prefill_tokens_per_second={prefill_tokens_per_sec:.4} decode_tokens={} \
         decode_tokens_per_second={decode_tokens_per_sec:.4}",
        protocol.decode_steps,
    );
}

/// The flag, then the environment, then the local default.
fn path_of(flag: Option<PathBuf>, variable: &str) -> PathBuf {
    flag.unwrap_or_else(|| {
        PathBuf::from(
            std::env::var(variable).unwrap_or_else(|_| panic!("pass the flag or set {variable}")),
        )
    })
}
