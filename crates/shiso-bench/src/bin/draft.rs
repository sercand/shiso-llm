//! The draft head's device runbook: does it propose a token, and a sensible one?
//!
//! ```text
//! cargo run --release -p shiso-bench --bin draft -- \
//!     --model <checkpoint>.gguf --draft <sidecar>.gguf \
//!     --config <sidecar>/config.json --tokenizer <sidecar> \
//!     --prompt "The capital of France is"
//! ```
//!
//! What it settles is the first question a speculative build has: the head
//! loads at the target's widths, its pass issues, and the token it proposes is
//! the one a reader would predict for the position after the target's own pick.
//! It settles nothing about acceptance - that is the verify pass's job and it
//! has its own check, which is that a speculative greedy run and a plain greedy
//! run produce the same tokens.
//!
//! It is a binary rather than a test because it needs a checkpoint, a sidecar, a
//! vocabulary and a device, and because what it prints is read by a person.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use std::fmt::Write as _;

use anyhow::{Context, Result, bail};
use clap::Parser;

use shiso_core::{Backend, HfConfig, Model, TokenId};
use shiso_engine::EngineOptions;
use shiso_engine::sparse::bind::Operand;
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::scratch::Buf;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_models::qwen4exp::{Qwen4ExpConfig, Qwen4ExpModel};

/// Context the session is opened with.
const CTX: u32 = 8192;

/// The runbook's command line.
#[derive(Debug, Parser)]
#[command(about = "runs the draft head once against a real checkpoint", long_about = None)]
struct Args {
    /// The checkpoint the target decodes from.
    #[arg(long)]
    model: PathBuf,
    /// The draft head sidecar.
    #[arg(long)]
    draft: PathBuf,
    /// The checkpoint's config document.
    #[arg(long)]
    config: PathBuf,
    /// The directory holding `tokenizer.json`.
    #[arg(long)]
    tokenizer: PathBuf,
    /// The prompt to stand at the end of.
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,
    /// How many tokens to chain, each drafting from the last one's own residual.
    #[arg(long, default_value_t = 3)]
    chain: u32,
    /// Replay a reference engine's own draft records instead of chaining.
    ///
    /// Every record carries the tokens and positions one draft pass ran, the
    /// multi-stream rows it started from and the ones it produced. Replaying
    /// them feeds this head the *reference's* input and compares its output
    /// against the reference's, which is the only check that separates a head
    /// that is right from one that is merely fluent.
    #[arg(long)]
    replay: Option<PathBuf>,
    /// Corpus token counts the draft's vocabulary head is narrowed from.
    #[arg(long)]
    draft_vocab: Option<PathBuf>,
    /// Generate this many tokens twice - once a token at a time, once
    /// speculatively - and compare.
    ///
    /// The whole correctness claim of a speculative build in one number. Greedy
    /// decoding is deterministic, so a verify that accepts the right prefix and
    /// a rollback that restores the right state produce *the same tokens* as a
    /// plain loop. Anything else is a build that answers fluently from a state
    /// the target never computed.
    #[arg(long)]
    generate: Option<u32>,
}

/// One reference draft record.
struct Record {
    tokens: Vec<TokenId>,
    positions: Vec<u32>,
    hidden: Vec<u8>,
    res: Vec<u8>,
}

/// Reads every record in `dir`, in the order they were written.
fn records(dir: &std::path::Path) -> Result<Vec<Record>> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "bin"))
        .collect();
    paths.sort();
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = std::fs::read(&path)?;
        let word = |at: usize| -> i32 {
            i32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        if bytes.len() < 20 || word(0) != 0x4d54_5031 {
            bail!("{} is not a draft record", path.display());
        }
        let n = usize::try_from(word(4)).unwrap_or(0);
        let hc = usize::try_from(word(8)).unwrap_or(0);
        let mut at = 20;
        let tokens = (0..n)
            .map(|i| TokenId::new(u32::try_from(word(at + 4 * i)).unwrap_or(0)))
            .collect();
        at += 4 * n;
        let positions = (0..n)
            .map(|i| u32::try_from(word(at + 4 * i)).unwrap_or(0))
            .collect();
        at += 4 * n;
        let span = 4 * n * hc;
        if bytes.len() < at + 2 * span {
            bail!("{} is short of its two row blocks", path.display());
        }
        out.push(Record {
            tokens,
            positions,
            hidden: bytes[at..at + span].to_vec(),
            res: bytes[at + span..at + 2 * span].to_vec(),
        });
    }
    Ok(out)
}

#[expect(
    clippy::print_stdout,
    reason = "this binary's output is read by a person"
)]
fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("SHISO_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    match run(&Args::parse()) {
        Ok(report) => {
            println!("{report}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            tracing::error!("{error:#}");
            ExitCode::FAILURE
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "the runbook is a sequence a reader follows top to bottom: open \
              the device, load, prefill, draft, report"
)]
fn run(args: &Args) -> Result<String> {
    if !shiso_cuda::cuda_available() {
        bail!("no NVIDIA device on this machine");
    }
    let backend: Arc<dyn Backend> =
        shiso_cuda::CudaBackend::new(0, shiso_cuda::CudaOptions::default())
            .context("opening the device")?;
    let document = HfConfig::from_file(&args.config).context("reading the config document")?;
    let config =
        Qwen4ExpConfig::from_hf(&document, &args.model).context("the config is not this family")?;
    let vocab = tokenizers::Tokenizer::from_file(args.tokenizer.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("loading the vocabulary")?;

    let mut engine = EngineOptions::default();
    engine.ctx = CTX;
    let opts = SparseOptions {
        engine,
        // The head is resident and drafting, so the session carries the
        // per-position rollback state a verify would need even though this
        // runbook never verifies.
        spec: SpecDepth::new(args.chain),
        ..SparseOptions::default()
    };
    let opts = SparseOptions {
        entries: shiso_engine::sparse::capture::capture(backend.as_ref()),
        co_resident_blocks: shiso_engine::sparse::capture::co_resident_blocks(
            backend.as_ref(),
            shiso_engine::sparse::capture::capture(backend.as_ref()),
        ),
        ..opts
    };
    let mut model = Qwen4ExpModel::load_with(
        &backend,
        &args.model,
        Some(&args.draft),
        args.draft_vocab.as_deref(),
        &config,
        opts,
    )
    .context("loading the checkpoint and its draft head")?;

    if let Some(dir) = args.replay.as_ref() {
        return replay(&backend, &mut model, dir);
    }
    if let Some(count) = args.generate {
        return generate(&backend, &mut model, &vocab, &args.prompt, count);
    }

    let prompt: Vec<TokenId> = vocab
        .encode(args.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .iter()
        .copied()
        .map(TokenId::new)
        .collect();
    let session = model.open_session()?;
    let logits = model.prefill_last(session, &prompt)?;
    let row = logits.row(0)?;
    let mut bytes = vec![0u8; usize::try_from(row.len()).unwrap_or(0)];
    backend.copy_d2h(row, &mut bytes)?;
    let target = argmax(&bytes);

    // The head reads the row the target just produced, together with the token
    // the target picked for the position after it, and predicts the token after
    // *that*. So the first draft's position is the prompt's length: the slot the
    // target's pick will occupy.
    let mut report = String::new();
    writeln!(
        report,
        "prompt {} tokens; target picks {target} {:?}",
        prompt.len(),
        decode(&vocab, target)
    )?;
    let stream = u64::from(config.shape().stream_dim()) * 4;
    // The row the head starts from, printed so it can be held against another
    // engine's own: a staged row that is plausible and wrong produces drafts
    // that are plausible and wrong, and nothing downstream would say so.
    {
        let last = u64::try_from(prompt.len().saturating_sub(1)).unwrap_or(0) * stream;
        let view = model
            .stack()
            .scratch_view(Buf::Residual)?
            .slice(last, stream)?;
        let mut bytes = vec![0u8; usize::try_from(stream).unwrap_or(0)];
        backend.copy_d2h(view, &mut bytes)?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        let l1: f64 = values.iter().map(|v| f64::from(v.abs())).sum();
        writeln!(
            report,
            "staged row: head {:?} l1 {l1:.3}",
            values.get(..4).unwrap_or(&[])
        )?;
    }
    let last = u64::try_from(prompt.len().saturating_sub(1)).unwrap_or(0) * stream;
    let mut token = target;
    // The head at position `p` reads the target's hidden state for `p` and the
    // token that sits at `p + 1`, and predicts the token at `p + 2`. So the
    // first draft's position is the prompt's *last*, not one past it: the token
    // it is handed is the pick that will occupy the slot after that.
    let mut at = u32::try_from(prompt.len().saturating_sub(1)).unwrap_or(0);
    let mut hidden = Operand::ScratchFrom(Buf::Residual, last);
    for step in 0..args.chain {
        let pick = model
            .stack_mut()
            .draft(session, &[TokenId::new(token)], &[at], hidden, true)
            .with_context(|| format!("draft step {step}"))?
            .unwrap_or(TokenId::new(0));
        writeln!(
            report,
            "  draft {step}: at position {at}, after {:?} -> {} {:?}",
            decode(&vocab, token),
            pick.get(),
            decode(&vocab, pick.get())
        )?;
        token = pick.get();
        at = at.saturating_add(1);
        // Every step after the first reads the head's own residual, which its
        // previous pass left in row zero.
        hidden = Operand::Scratch(Buf::Residual);
    }
    model.teardown()?;
    Ok(report)
}

/// Decodes the same continuation twice and compares the two.
#[expect(
    clippy::print_stdout,
    reason = "this binary's output is read by a person"
)]
fn generate(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    vocab: &tokenizers::Tokenizer,
    prompt: &str,
    count: u32,
) -> Result<String> {
    let prompt: Vec<TokenId> = vocab
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .get_ids()
        .iter()
        .copied()
        .map(TokenId::new)
        .collect();
    let eos = model.info().stop_tokens.primary();

    let at = std::time::Instant::now();
    let plain = decode_plain(backend, model, &prompt, count)?;
    let slow_ms = at.elapsed().as_secs_f64() * 1000.0;
    println!("plain:       {:?}", text(vocab, &plain));
    let at = std::time::Instant::now();
    let (fast, steps) = decode_speculative(backend, model, &prompt, count, eos)?;
    let fast_ms = at.elapsed().as_secs_f64() * 1000.0;
    println!("speculative: {:?}", text(vocab, &fast));
    if plain != fast {
        let at = plain
            .iter()
            .zip(&fast)
            .position(|(a, b)| a != b)
            .unwrap_or(plain.len().min(fast.len()));
        bail!(
            "the two runs diverge at token {at}: plain {:?}, speculative {:?}",
            plain.get(at).map(|t| t.get()),
            fast.get(at).map(|t| t.get())
        );
    }
    let committed = u32::try_from(fast.len()).unwrap_or(0);
    // Both walls include the prefill, which is the same work in both and so
    // shrinks the ratio rather than flattering it.
    Ok(format!(
        "identical: {committed} tokens, {:.2} per speculative step ({steps} steps)\n\
         plain {slow_ms:.0} ms ({:.2} tok/s), speculative {fast_ms:.0} ms ({:.2} tok/s), \
         {:.2}x",
        f64::from(committed) / f64::from(steps.max(1)),
        f64::from(committed) / (slow_ms / 1000.0),
        f64::from(committed) / (fast_ms / 1000.0),
        slow_ms / fast_ms.max(1e-9)
    ))
}

/// The plain loop: one token a step.
fn decode_plain(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    prompt: &[TokenId],
    count: u32,
) -> Result<Vec<TokenId>> {
    let session = model.open_session()?;
    let mut out = Vec::with_capacity(count as usize);
    let mut token = pick(backend, model.prefill_last(session, prompt)?)?;
    for _ in 0..count {
        out.push(token);
        let step = shiso_core::model::DecodeStep { session, token };
        token = pick(backend, model.decode_batch(std::slice::from_ref(&step))?)?;
    }
    model.close_session(session)?;
    Ok(out)
}

/// The speculative loop, and how many steps it took.
#[expect(
    clippy::print_stdout,
    reason = "this binary's output is read by a person"
)]
fn decode_speculative(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    prompt: &[TokenId],
    count: u32,
    eos: TokenId,
) -> Result<(Vec<TokenId>, u32)> {
    let session = model.open_session()?;
    let mut token = pick(backend, model.prefill_last(session, prompt)?)?;
    // The row the prompt's last position produced, which is the row the head's
    // first guess starts from.
    let last = u32::try_from(prompt.len().saturating_sub(1)).unwrap_or(0);
    model.stack_mut().stage_draft_row(session, last)?;

    // The caller's own first token is the one the prefill produced; every step
    // after that emits what it accepted plus the target's next pick.
    let mut out = vec![token];
    let mut steps = 0u32;
    let mut accepted = 0u32;
    let mut drafted = 0u32;
    while u32::try_from(out.len()).unwrap_or(u32::MAX) < count {
        let step = model.stack_mut().speculate(session, token, eos)?;
        steps += 1;
        accepted += step.accepted;
        drafted += step.drafted;
        out.extend_from_slice(&step.produced);
        token = *step.produced.last().unwrap_or(&token);
    }
    println!("accepted {accepted} of {drafted} drafts");
    out.truncate(count as usize);
    model.close_session(session)?;
    Ok((out, steps))
}

/// The greedy pick of a pass's first logits row.
fn pick(backend: &Arc<dyn Backend>, logits: shiso_core::Logits<'_>) -> Result<TokenId> {
    let row = logits.row(0)?;
    let mut bytes = vec![0u8; usize::try_from(row.len()).unwrap_or(0)];
    backend.copy_d2h(row, &mut bytes)?;
    Ok(TokenId::new(argmax(&bytes)))
}

/// A token run as text.
fn text(vocab: &tokenizers::Tokenizer, tokens: &[TokenId]) -> String {
    let ids: Vec<u32> = tokens.iter().map(|t| t.get()).collect();
    vocab.decode(&ids, false).unwrap_or_default()
}

/// Replays a reference engine's draft records through this head.
///
/// Each record's own `hidden` is written into the residual before the pass
/// runs, so the two heads are given the same input rather than the same
/// history: a divergence then belongs to this pass and not to everything that
/// ran before it.
#[expect(
    clippy::print_stdout,
    reason = "this binary's output is read by a person"
)]
fn replay(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    dir: &std::path::Path,
) -> Result<String> {
    let records = records(dir)?;
    if records.is_empty() {
        bail!("no draft records under {}", dir.display());
    }
    let session = model.open_session()?;
    let mut worst = 0.0f64;
    for (index, record) in records.iter().enumerate() {
        let rows = u32::try_from(record.tokens.len()).unwrap_or(0);
        model.stack_mut().reserve(session, rows)?;
        let view = model.stack().scratch_view(Buf::Residual)?;
        let span = u64::try_from(record.hidden.len()).unwrap_or(0);
        backend.copy_h2d(view.slice(0, span)?, &record.hidden)?;
        let pick = model
            .stack_mut()
            .draft(
                session,
                &record.tokens,
                &record.positions,
                Operand::Scratch(Buf::Residual),
                true,
            )?
            .unwrap_or(TokenId::new(0));
        let mut got = vec![0u8; record.res.len()];
        let view = model.stack().scratch_view(Buf::Residual)?;
        backend.copy_d2h(view.slice(0, span)?, &mut got)?;
        let (rel, l1_ref, l1_got) = compare(&record.res, &got);
        worst = worst.max(rel);
        println!(
            "record {index}: {rows} rows at {:?} -> pick {}; residual |l1| ref {l1_ref:.3} got              {l1_got:.3}, relative {rel:.3e}",
            record.positions,
            pick.get()
        );
    }
    Ok(format!("worst relative residual difference {worst:.3e}"))
}

/// Two row blocks as one number: how far apart their absolute sums are.
fn compare(want: &[u8], got: &[u8]) -> (f64, f64, f64) {
    let sum = |bytes: &[u8]| -> f64 {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f64::from(f32::from_le_bytes(*w).abs()))
            .sum()
    };
    let (a, b) = (sum(want), sum(got));
    ((a - b).abs() / a.max(1e-9), a, b)
}

/// The greedy pick of one logits row.
fn argmax(bytes: &[u8]) -> u32 {
    let mut best = 0u32;
    let mut high = f32::NEG_INFINITY;
    for (index, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let value = f32::from_le_bytes(*word);
        if value > high {
            high = value;
            best = u32::try_from(index).unwrap_or(0);
        }
    }
    best
}

/// One token's text, for the human half of the report.
fn decode(vocab: &tokenizers::Tokenizer, token: u32) -> String {
    vocab
        .decode(&[token], false)
        .unwrap_or_else(|_| format!("<{token}>"))
}
