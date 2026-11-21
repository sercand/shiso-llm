//! The Inkling-2 device runbook: do the multi-row paths answer what the
//! one-row path answers, and what does a pass cost?
//!
//! ```text
//! cargo run --release -p shiso-bench --bin inkling2 -- \
//!     --model ~/.shiso/models/inkling-2-mini parity --tokens 1500
//! cargo run --release -p shiso-bench --bin inkling2 -- --model <dir> spec --steps 200
//! cargo run --release -p shiso-bench --bin inkling2 -- --model <dir> perf --tokens 512
//! ```
//!
//! The family has no independent reference on this machine, so the one-row
//! decode path is the reference for everything wider than it: a chunked
//! prefill has to leave the logits a token-at-a-time walk leaves, and a
//! speculative greedy run has to emit the tokens a plain greedy run emits.
//! `parity` and `spec` settle those; `dump` writes a row of logits so two builds
//! can be compared; `perf` times. The parity paths print no duration and the
//! timing path prints no logit.
//!
//! It is a binary rather than a test because it needs the 20 GB checkpoint and
//! a device, and because what it prints is read by a person.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use shiso_bench::replay::{Device, open_backend};
use shiso_core::{Backend, DecodeStep, Logits, Model, SessionId, TokenId};
use shiso_models::inkling2::Inkling2Model;

/// Context every session is opened with.
const CTX: u32 = 16384;

/// The runbook's command line.
#[derive(Debug, Parser)]
#[command(about = "Inkling-2 parity, speculation and timing on the local device", long_about = None)]
struct Args {
    /// The checkpoint directory.
    #[arg(long)]
    model: PathBuf,
    /// A text file the prompt tokens are cut from; the repository's README
    /// when absent. It is repeated until it covers the length asked for.
    #[arg(long)]
    text: Option<PathBuf>,
    /// What to run.
    #[command(subcommand)]
    command: Command,
}

/// One runbook.
#[derive(Debug, Subcommand)]
enum Command {
    /// A chunked prefill against a token-at-a-time walk of the same tokens.
    Parity {
        /// Prompt length.
        #[arg(long, default_value_t = 1500)]
        tokens: usize,
        /// Rows a prefill pass carries; zero leaves the model its own long
        /// passes. A pass shorter than 32 rows runs the kernels a decode row
        /// runs, so its logits must be the walk's bit for bit: that is the
        /// check of the key ring, the window and the carried states. A long
        /// pass multiplies on tiles in another order, and is held to the same
        /// picks and a cosine only.
        #[arg(long, default_value_t = 0)]
        pass_rows: usize,
    },
    /// Writes the prefill's last row and the row after `steps` greedy steps.
    Dump {
        /// Prompt length.
        #[arg(long, default_value_t = 512)]
        tokens: usize,
        /// Greedy steps after the prefill.
        #[arg(long, default_value_t = 16)]
        steps: usize,
        /// Where the rows go, as little-endian f32.
        #[arg(long)]
        out: PathBuf,
    },
    /// Compares two files `dump` wrote.
    Diff {
        /// The earlier build's rows.
        a: PathBuf,
        /// The later build's rows.
        b: PathBuf,
    },
    /// A plain greedy run against a speculative one from the same prompt.
    Spec {
        /// Prompt length (of the text) ahead of the generation.
        #[arg(long, default_value_t = 64)]
        tokens: usize,
        /// Tokens to generate.
        #[arg(long, default_value_t = 200)]
        steps: usize,
        /// A chat question to ask instead of continuing the text.
        #[arg(long)]
        ask: Option<String>,
        /// Tokens a speculative step drafts.
        #[arg(long, default_value_t = 7)]
        depth: u32,
        /// Speculative rounds to run again inside a profile region afterwards.
        #[arg(long, default_value_t = 0)]
        profile_rounds: u32,
    },
    /// Prefill and decode rates.
    Perf {
        /// Prompt length.
        #[arg(long, default_value_t = 512)]
        tokens: usize,
        /// Decode steps timed after the prefill.
        #[arg(long, default_value_t = 48)]
        steps: usize,
        /// Prefill repetitions after one warm-up.
        #[arg(long, default_value_t = 3)]
        reps: usize,
    },
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
    if let Command::Diff { a, b } = &args.command {
        return diff(a, b);
    }
    let vocab = tokenizers::Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
    let backend = open_backend(platform()).context("opening the device")?;
    let spec = matches!(args.command, Command::Spec { .. });
    let model = Inkling2Model::load_with_options(Arc::clone(&backend), &args.model, CTX, spec)
        .context("loading the checkpoint")?;
    let model = match args.command {
        Command::Spec { depth, .. } => model.with_draft_depth(depth)?,
        _ => model,
    };
    let mut rig = Rig {
        backend,
        model,
        vocab,
        row: Vec::new(),
    };
    let text = match &args.text {
        Some(path) => {
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
        }
        None => include_str!("../../../../README.md").to_owned(),
    };
    match args.command {
        Command::Parity { tokens, pass_rows } => {
            let prompt = rig_tokens(&rig, &text, tokens)?;
            parity(&mut rig, &prompt, pass_rows)
        }
        Command::Dump { tokens, steps, out } => {
            let prompt = rig_tokens(&rig, &text, tokens)?;
            dump(&mut rig, &prompt, steps, &out)
        }
        Command::Spec {
            tokens,
            steps,
            ask,
            profile_rounds,
            ..
        } => {
            let prompt = match ask {
                Some(question) => rig.encode(&format!(
                    "<|message_user|><|content_text|>{question}<|end_message|>\
                     <|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
                     <|message_model|>"
                ))?,
                None => rig_tokens(&rig, &text, tokens)?,
            };
            speculation(&mut rig, &prompt, steps, profile_rounds)
        }
        Command::Perf {
            tokens,
            steps,
            reps,
        } => {
            let prompt = rig_tokens(&rig, &text, tokens)?;
            perf(&mut rig, &prompt, steps, reps)
        }
        Command::Diff { .. } => Ok(()),
    }
}

/// The device this platform has.
const fn platform() -> Device {
    if cfg!(target_os = "macos") {
        Device::Metal
    } else {
        Device::Cuda
    }
}

/// What every runbook holds.
struct Rig {
    backend: Arc<dyn Backend>,
    model: Inkling2Model,
    vocab: tokenizers::Tokenizer,
    row: Vec<u8>,
}

/// `count` tokens of `text`, the text repeated as often as that takes.
fn rig_tokens(rig: &Rig, text: &str, count: usize) -> Result<Vec<TokenId>> {
    let once = rig.encode(text)?;
    if once.is_empty() {
        bail!("the text encodes to no tokens");
    }
    Ok(once.iter().copied().cycle().take(count).collect())
}

impl Rig {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        let encoding = self
            .vocab
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encoding: {e}"))?;
        Ok(encoding
            .get_ids()
            .iter()
            .map(|&id| TokenId::new(id))
            .collect())
    }

    fn text(&self, tokens: &[TokenId]) -> String {
        let ids: Vec<u32> = tokens.iter().map(|t| t.get()).collect();
        self.vocab.decode(&ids, false).unwrap_or_default()
    }

    /// Reads row 0 of `logits` to the host.
    fn read(
        backend: &Arc<dyn Backend>,
        row: &mut Vec<u8>,
        logits: &Logits<'_>,
    ) -> Result<Vec<f32>> {
        let view = logits.row(0)?;
        row.resize(usize::try_from(view.len())?, 0);
        backend.sync()?;
        backend.copy_d2h(view, row)?;
        Ok(row
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    fn prefill(&mut self, session: SessionId, tokens: &[TokenId]) -> Result<Vec<f32>> {
        let logits = self.model.prefill_last(session, tokens)?;
        Self::read(&self.backend, &mut self.row, &logits)
    }

    fn step(&mut self, session: SessionId, token: TokenId) -> Result<Vec<f32>> {
        let logits = self.model.decode_batch(&[DecodeStep { session, token }])?;
        Self::read(&self.backend, &mut self.row, &logits)
    }
}

/// Why a profiled speculative round stopped.
#[derive(Debug, thiserror::Error)]
enum ProfileFailure {
    /// The profiler refused.
    #[error(transparent)]
    Profile(#[from] shiso_bench::profile::ProfileError),
    /// The device refused.
    #[error(transparent)]
    Core(#[from] shiso_core::CoreError),
    /// The checkpoint was loaded without its drafter, or a round made nothing.
    #[error("no drafter, or a round that produced nothing")]
    NoDrafter,
}

/// The lowest id attaining the row's maximum.
fn argmax(row: &[f32]) -> TokenId {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    TokenId::new(u32::try_from(best).unwrap_or(0))
}

/// How far two rows are apart: the largest difference, the cosine, and whether
/// they pick the same token.
fn compare(a: &[f32], b: &[f32]) -> (f32, f64, bool) {
    let mut worst = 0f32;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        worst = worst.max((x - y).abs());
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    (worst, dot / (na.sqrt() * nb.sqrt()), argmax(a) == argmax(b))
}

fn parity(rig: &mut Rig, prompt: &[TokenId], pass_rows: usize) -> Result<()> {
    let session = rig.model.open_session()?;
    let chunked = if pass_rows == 0 {
        rig.prefill(session, prompt)?
    } else {
        // Every pass but the last commits without logits.
        let last = (prompt.len() - 1) / pass_rows * pass_rows;
        for pass in prompt[..last].chunks(pass_rows) {
            rig.model.prefill_chunk(session, pass)?;
        }
        rig.prefill(session, &prompt[last..])?
    };
    let chunked_next = rig.step(session, argmax(&chunked))?;
    rig.model.close_session(session)?;

    let session = rig.model.open_session()?;
    let mut walked = rig.prefill(session, &prompt[..1])?;
    for &token in &prompt[1..] {
        walked = rig.step(session, token)?;
    }
    let walked_next = rig.step(session, argmax(&chunked))?;
    rig.model.close_session(session)?;

    let (worst, cosine, same) = compare(&chunked, &walked);
    let (worst_next, cosine_next, same_next) = compare(&chunked_next, &walked_next);
    tracing::info!(
        tokens = prompt.len(),
        worst,
        cosine,
        same_pick = same,
        worst_next,
        cosine_next,
        same_pick_next = same_next,
        "chunked prefill against the one-row walk"
    );
    let identical = chunked
        .iter()
        .zip(&walked)
        .chain(chunked_next.iter().zip(&walked_next))
        .all(|(a, b)| a.to_bits() == b.to_bits());
    tracing::info!(pass_rows, identical, "bit for bit");
    if pass_rows > 0 && pass_rows < 32 {
        if !identical {
            bail!("a short pass does not leave the bits the one-row walk leaves");
        }
        return Ok(());
    }
    // A long pass sums on tiles in another order, and a router near-tie that
    // flips on the difference moves a row by more than rounding does - as far
    // as a wrong window moved it (0.997), which is why the short-pass run
    // above, not this bound, is the check of the logic.
    if !(same && same_next && cosine > 0.99 && cosine_next > 0.99) {
        bail!("the chunked prefill does not answer what the one-row walk answers");
    }
    Ok(())
}

fn dump(rig: &mut Rig, prompt: &[TokenId], steps: usize, out: &Path) -> Result<()> {
    let session = rig.model.open_session()?;
    let first = rig.prefill(session, prompt)?;
    let mut last = first.clone();
    let mut picked = Vec::with_capacity(steps);
    for _ in 0..steps {
        let token = argmax(&last);
        picked.push(token);
        last = rig.step(session, token)?;
    }
    rig.model.close_session(session)?;
    let mut bytes = Vec::with_capacity((first.len() + last.len()) * 4);
    for v in first.iter().chain(&last) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(out, bytes).with_context(|| format!("writing {}", out.display()))?;
    tracing::info!(
        tokens = prompt.len(),
        steps,
        text = %rig.text(&picked),
        "rows written"
    );
    Ok(())
}

fn diff(a: &Path, b: &Path) -> Result<()> {
    let read = |path: &Path| -> Result<Vec<f32>> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    };
    let (a, b) = (read(a)?, read(b)?);
    if a.len() != b.len() || a.len() % 2 != 0 {
        bail!("the two files do not hold the same rows");
    }
    let half = a.len() / 2;
    let (worst, cosine, same) = compare(&a[..half], &b[..half]);
    let (worst_last, cosine_last, same_last) = compare(&a[half..], &b[half..]);
    tracing::info!(
        worst,
        cosine,
        same_pick = same,
        worst_last,
        cosine_last,
        same_pick_last = same_last,
        identical = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
        "prefill row, then the row after the greedy steps"
    );
    if !(same && same_last) {
        bail!("the two builds pick different tokens");
    }
    Ok(())
}

fn speculation(rig: &mut Rig, prompt: &[TokenId], steps: usize, profile_rounds: u32) -> Result<()> {
    // Plain greedy.
    let session = rig.model.open_session()?;
    let mut row = rig.prefill(session, prompt)?;
    let mut plain = Vec::with_capacity(steps);
    let started = Instant::now();
    while plain.len() < steps {
        let token = argmax(&row);
        plain.push(token);
        row = rig.step(session, token)?;
    }
    let plain_s = started.elapsed().as_secs_f64();
    rig.model.close_session(session)?;

    // Speculative greedy from the same prompt.
    let session = rig.model.open_session()?;
    let row = rig.prefill(session, prompt)?;
    let mut spec = vec![argmax(&row)];
    let (mut rounds, mut drafted, mut accepted) = (0u32, 0u32, 0u32);
    let started = Instant::now();
    while spec.len() < steps {
        let first = *spec.last().context("a first token")?;
        let step = rig
            .model
            .spec_mut()
            .context("the checkpoint was loaded without its drafter")?
            .speculate(session, first)?;
        rounds += 1;
        drafted += step.drafted;
        accepted += step.accepted;
        spec.extend_from_slice(&step.produced);
    }
    let spec_s = started.elapsed().as_secs_f64();
    let made = spec.len() - 1;
    if profile_rounds > 0 {
        let backend = Arc::clone(&rig.backend);
        let mut first = *spec.last().context("a first token")?;
        let ((), report) =
            shiso_bench::profile::scoped(backend.as_ref(), || -> Result<(), ProfileFailure> {
                for _ in 0..profile_rounds {
                    let step = rig
                        .model
                        .spec_mut()
                        .ok_or(ProfileFailure::NoDrafter)?
                        .speculate(session, first)?;
                    first = *step.produced.last().ok_or(ProfileFailure::NoDrafter)?;
                }
                Ok(())
            })?;
        tracing::info!("{profile_rounds} profiled rounds\n{}", report.table(40));
    }
    rig.model.close_session(session)?;
    spec.truncate(steps);

    let same = plain.iter().zip(&spec).take_while(|(a, b)| a == b).count();
    #[expect(
        clippy::cast_precision_loss,
        reason = "token counts are far below 2^52"
    )]
    {
        tracing::info!(
            prompt_tokens = prompt.len(),
            plain_tokens_per_second = steps as f64 / plain_s,
            spec_tokens_per_second = made as f64 / spec_s,
            rounds,
            tokens_per_round = made as f64 / f64::from(rounds.max(1)),
            acceptance = f64::from(accepted) / f64::from(drafted.max(1)),
            ms_per_round = spec_s * 1000.0 / f64::from(rounds.max(1)),
            agree_until = same,
            of = steps,
            "plain greedy against speculative greedy"
        );
    }
    tracing::info!(text = %rig.text(&plain), "plain");
    if same < steps.min(spec.len()) {
        tracing::info!(text = %rig.text(&spec), "speculative");
        bail!("the speculative run leaves the plain run at token {same}");
    }
    Ok(())
}

fn perf(rig: &mut Rig, prompt: &[TokenId], steps: usize, reps: usize) -> Result<()> {
    let mut prefill_s = Vec::with_capacity(reps);
    let mut decode_ms = Vec::new();
    for rep in 0..=reps {
        let session = rig.model.open_session()?;
        let started = Instant::now();
        let mut row = rig.prefill(session, prompt)?;
        let took = started.elapsed().as_secs_f64();
        if rep > 0 {
            prefill_s.push(took);
        }
        if rep == reps {
            for _ in 0..steps {
                let token = argmax(&row);
                let started = Instant::now();
                row = rig.step(session, token)?;
                decode_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            }
        }
        rig.model.close_session(session)?;
    }
    let prefill = median(&mut prefill_s).context("no prefill was timed")?;
    // The first step lays the work out for the first time.
    let decode = median(decode_ms.get_mut(1..).unwrap_or(&mut [])).unwrap_or(f64::NAN);
    #[expect(
        clippy::cast_precision_loss,
        reason = "token counts are far below 2^52"
    )]
    {
        tracing::info!(
            prompt_tokens = prompt.len(),
            prefill_seconds = prefill,
            prefill_tokens_per_second = prompt.len() as f64 / prefill,
            prefill_spread = spread(&prefill_s),
            decode_ms_per_step = decode,
            decode_tokens_per_second = 1000.0 / decode,
            "median rates"
        );
    }
    Ok(())
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

/// The widest relative gap between the repetitions.
fn spread(sorted: &[f64]) -> f64 {
    match (sorted.first(), sorted.last()) {
        (Some(lo), Some(hi)) if *lo > 0.0 => (hi - lo) / lo,
        _ => 0.0,
    }
}
