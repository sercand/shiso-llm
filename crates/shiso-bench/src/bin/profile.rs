//! Where a text model's prefill or decode step spends its GPU time.
//!
//! The phase is measured batched first, then once inside a profile region, and
//! both are printed: the region submits every launch on its own, so its total
//! is above the batched time by the overlap it gave up, and only the batched
//! number is how fast the model is.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use serde_json::json;

use shiso_bench::decode::{PREFILL_TOKENS, Rig, decode_rep};
use shiso_bench::measure::{after_warmup, median};
use shiso_bench::prefill::prefill_once;
use shiso_bench::profile;

/// Which phase is profiled.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Phase {
    /// One prefill of the prompt.
    Prefill,
    /// Steady decode steps after the prompt.
    Decode,
}

/// CLI arguments.
#[derive(Debug, Parser)]
#[command(
    name = "profile",
    about = "GPU time per kernel for one phase of a text model"
)]
struct Args {
    /// The checkpoint: a GGUF file, or a sparse checkpoint's directory.
    #[arg(long)]
    model: PathBuf,
    /// The checkpoint's own config, instead of the recorded shape profile.
    #[arg(long)]
    config: Option<PathBuf>,
    /// The phase to profile.
    #[arg(long, value_enum, default_value_t = Phase::Decode)]
    phase: Phase,
    /// Prompt length in tokens.
    #[arg(long, default_value_t = PREFILL_TOKENS)]
    prompt_tokens: u32,
    /// Decode steps inside the region.
    #[arg(long, default_value_t = 32)]
    steps: u32,
    /// Decode steps run first, outside the region.
    #[arg(long, default_value_t = 2)]
    warm_steps: u32,
    /// Rows of the printed table.
    #[arg(long, default_value_t = 25)]
    top: usize,
    /// Writes the report and its context as JSON here.
    #[arg(long)]
    json: Option<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    if args.prompt_tokens == 0 || args.steps == 0 {
        bail!("--prompt-tokens and --steps must be greater than zero");
    }

    let mut rig = if args.model.is_dir() {
        Rig::open_qwen35moe(&args.model)?
    } else if let Some(config) = args.config.as_deref() {
        Rig::open_with_config(&args.model, config)?
    } else {
        Rig::open(&args.model)?
    };
    rig.set_prompt_len(args.prompt_tokens);

    let (batched_ms, report, per) = match args.phase {
        Phase::Decode => {
            let steps = decode_rep(&mut rig, args.warm_steps + args.steps)?;
            let batched = median(after_warmup(&steps, args.warm_steps))
                .context("no decode steps were timed")?;
            let report = profile::decode_profiled(&mut rig, args.warm_steps, args.steps)?;
            (batched, report, args.steps)
        }
        Phase::Prefill => {
            prefill_once(&mut rig)?;
            prefill_once(&mut rig)?;
            let batched = prefill_once(&mut rig)?;
            (batched, profile::prefill_profiled(&mut rig)?, 1)
        }
    };
    let model = rig.model_name().to_owned();
    rig.teardown()?;

    let serialized_ms = report_ms(report.serialized_ns) / f64::from(per);
    print(&format!(
        "{model} {:?}: batched {batched_ms:.3} ms, serialized {serialized_ms:.3} ms per {}\n{}",
        args.phase,
        if per == 1 { "pass" } else { "step" },
        report.table(args.top),
    ));
    if let Some(path) = args.json.as_deref() {
        let device = shiso_metal::device_facts().map(|facts| facts.name);
        let context = json!({
            "model": model,
            "phase": format!("{:?}", args.phase),
            "prompt_tokens": args.prompt_tokens,
            "steps": per,
            "batched_ms": batched_ms,
            "device": device,
        });
        profile::write(path, &context, &report)?;
    }
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "nanoseconds below 2^53 convert exactly"
)]
fn report_ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

/// The table is this runner's output.
#[expect(
    clippy::print_stdout,
    reason = "the table is this runner's output, read from stdout"
)]
fn print(text: &str) {
    println!("{text}");
}
