//! The Qwen 3.5 `MoE` architecture from a GGUF container: what it answers, and
//! what it costs.
//!
//! ```text
//! cargo run --release -p shiso-bench --bin q35moe_gguf -- \
//!     --model <file>.gguf --tokenizer <dir> --prompt "..." --tokens 64
//! ```
//!
//! **This is the greedy-lock runbook for this container.** llama.cpp reads the
//! same file, so `llama-cli --temp 0` over the same prompt is an independent
//! implementation of the same weights, and the ids the two produce either agree
//! or name the first position where they do not. That is a stronger check than
//! any tolerance on a logits row: it grades forty slots, a routed feed-forward,
//! two expert codecs and the sampler in one number.
//!
//! It is a binary rather than a test because it needs a 22 GB file, a
//! vocabulary and a device, and because the ids it prints are read beside
//! another program's.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use shiso_core::model::{DecodeStep, Model};
use shiso_core::{Backend, TokenId};
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_engine::EngineOptions;
use shiso_models::qwen35moe::Qwen35MoeModel;

/// The runbook's command line.
#[derive(Debug, Parser)]
#[command(about = "loads a qwen35moe GGUF and decodes greedily", long_about = None)]
struct Args {
    /// The container to read.
    #[arg(long)]
    model: PathBuf,
    /// The directory holding `tokenizer.json`.
    #[arg(long)]
    tokenizer: PathBuf,
    /// What to continue.
    #[arg(
        long,
        default_value = "The capital of France is Paris, and the capital of Italy is"
    )]
    prompt: String,
    /// How many tokens to produce.
    #[arg(long, default_value_t = 32)]
    tokens: usize,
    /// Context the session is opened with.
    #[arg(long, default_value_t = 4096)]
    ctx: u32,
    /// The multi-token-prediction head's directory, when one is to be bound.
    #[arg(long)]
    draft: Option<PathBuf>,
    /// Guesses the head is asked for per step. Zero decodes plainly.
    #[arg(long, default_value_t = 0)]
    spec: u32,
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

    let tok = tokenizers::Tokenizer::from_file(args.tokenizer.join("tokenizer.json"))
        .map_err(|err| anyhow::anyhow!("the vocabulary reads: {err}"))?;
    let ids: Vec<TokenId> = tok
        .encode(args.prompt.as_str(), false)
        .map_err(|err| anyhow::anyhow!("the prompt encodes: {err}"))?
        .get_ids()
        .iter()
        .map(|id| TokenId::new(*id))
        .collect();

    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).context("a CUDA device")?;
    let mut opts = EngineOptions::default();
    opts.ctx = args.ctx;
    opts.spec = shiso_engine::mtp::SpecDepth::new(args.spec);
    let clock = Instant::now();
    let mut model =
        Qwen35MoeModel::load_gguf_with_draft(&backend, &args.model, opts, args.draft.as_deref())
            .context("the container loads")?;
    tracing::info!(seconds = clock.elapsed().as_secs_f64(), "resident");

    let vocab = usize::try_from(model.info().vocab).unwrap_or(0);
    let mut row = vec![0u8; vocab * 4];
    let session = model.open_session().context("a session opens")?;

    let clock = Instant::now();
    let last = model
        .prefill_last(session, &ids)
        .context("the prompt runs")?;
    let prefill = clock.elapsed().as_secs_f64();
    backend
        .copy_d2h(last.row(0).context("the prompt's row")?, &mut row)
        .context("the prompt's row reads back")?;
    let mut token = greedy(&row);

    let mut produced = vec![token];
    let (mut accepted, mut drafted) = (0u32, 0u32);
    let clock = Instant::now();
    while produced.len() < args.tokens {
        if args.spec > 0 {
            // The speculative step: the head drafts, the target verifies every
            // guess against its own argmax, and what comes back is the guesses
            // it kept followed by its own next pick.
            let step = model
                .spec_mut()
                .context("a build with a head answers the speculative path")?
                .speculate(session, token)
                .context("a speculative step")?;
            accepted += step.accepted;
            drafted += step.drafted;
            for id in &step.produced {
                produced.push(*id);
            }
            token = *step
                .produced
                .last()
                .context("a step produces at least one token")?;
            continue;
        }
        let out = model
            .decode_batch(&[DecodeStep { session, token }])
            .context("a decode step")?;
        backend
            .copy_d2h(out.row(0).context("the step's row")?, &mut row)
            .context("the step's row reads back")?;
        token = greedy(&row);
        produced.push(token);
    }
    let decode = clock.elapsed().as_secs_f64();

    produced.truncate(args.tokens);
    report(&tok, &ids, &produced, prefill, decode);
    if drafted > 0 {
        acceptance(accepted, drafted);
    }
    model.close_session(session).context("the session closes")?;
    model.teardown().context("the model tears down")?;
    Ok(())
}

/// What the head proposed and what the target kept.
///
/// Reported APART from the throughput, and that is deliberate: an acceptance
/// rate and a token rate answer different questions, and a head that drafts
/// badly can still raise the second on a machine where the verify pass is
/// cheap. A build that quoted one number would hide which it had.
#[expect(
    clippy::print_stdout,
    reason = "the measurement is this runner's output"
)]
fn acceptance(accepted: u32, drafted: u32) {
    println!(
        "draft_accepted={accepted} draft_proposed={drafted} acceptance={:.3}",
        f64::from(accepted) / f64::from(drafted.max(1)),
    );
}

/// A count, as a rate's numerator.
#[expect(
    clippy::cast_precision_loss,
    reason = "a prompt length and a token budget, neither of which reaches an \
              f64 mantissa"
)]
fn rate(count: usize) -> f64 {
    count as f64
}

/// The most likely id of a logits row.
fn greedy(row: &[u8]) -> TokenId {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (id, word) in row.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        if value > best.1 {
            best = (id, value);
        }
    }
    TokenId::new(u32::try_from(best.0).unwrap_or(0))
}

/// The ids and the text, which is what a reader compares with llama.cpp's.
#[expect(
    clippy::print_stdout,
    reason = "the ids and the rates are this runner's output, and a caller reads \
              them from stdout beside another program's"
)]
fn report(
    tok: &tokenizers::Tokenizer,
    prompt: &[TokenId],
    produced: &[TokenId],
    prefill: f64,
    decode: f64,
) {
    let ids: Vec<u32> = produced.iter().map(|t| t.get()).collect();
    let text = tok
        .decode(&ids, false)
        .unwrap_or_else(|_| "<undecodable>".into());
    println!("prompt_tokens={}", prompt.len());
    println!("ids={ids:?}");
    println!("text={text:?}");
    let steps = produced.len().saturating_sub(1);
    // Both counts are a prompt and a token budget, so neither reaches the
    // mantissa. Named rather than cast inline so the lint's exception sits
    // beside the reason for it.
    let (tokens, steps_f) = (rate(prompt.len()), rate(steps));
    println!(
        "prefill_tokens_per_second={:.4} decode_tokens={steps} decode_tokens_per_second={:.4}",
        tokens / prefill,
        steps_f / decode,
    );
}
