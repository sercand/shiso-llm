//! What a resume saves, measured against the cold prefill it replaces.
//!
//! One prompt, twice. The first run prefills from nothing and stores its prefix;
//! the second finds that prefix on disk and stands on it. The number worth having
//! is the ratio, because that is what the disk tier is for - and the number worth
//! *checking* is that both runs continue identically, since a resume that is fast
//! and wrong is worse than no resume at all.
//!
//! # Why this exercises more than it measures
//!
//! The checkpoint layer was built and unit-tested against a fake device, where a
//! byte image is a byte image. This is the first thing that asks a real device to
//! read its recurrent stack back, write it to a file, and reinstall it into a
//! second session well enough that the model carries on mid-sentence. The timing
//! is the point of the file; the continuation check is the point of the run.
//!
//! Records; never gates. The regression gate over this file belongs to the task
//! that owns gates.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::{DecodeStep, Model};
use shiso_engine::{Checkpointed, CkptConfig, EngineOptions, Reason, argmax};
use shiso_metal::{MetalBackend, metal_available};
use shiso_models::qwen35::{Qwen35Config, Qwen35Model};

/// Prompt length, in tokens. Long enough that a cold prefill is worth avoiding
/// and that the stored prefix clears the tier's own minimum.
const PROMPT_TOKENS: u32 = 1024;

/// Greedy steps each run takes past the prompt. The two runs' outputs are
/// compared over exactly these.
const CONTINUATION_TOKENS: u32 = 16;

/// Context the sessions are opened with.
const CTX: u32 = 2048;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let path = model_path();
    if !path.is_file() {
        bail!("no checkpoint at {}", path.display());
    }
    if !metal_available() {
        bail!("no Metal device on this machine");
    }
    let config = profile()?;
    let backend: Arc<dyn Backend> = MetalBackend::new().context("opening the device")?;

    // A directory of its own, removed on the way out: a measurement that reused a
    // warm tier would report a hit it did not earn.
    let dir = std::env::temp_dir().join(format!("shiso-kv-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let outcome = run(&backend, &path, &config, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    let record = outcome?;

    tracing::info!(
        cold_prefill_ms = record.cold_prefill_ms,
        resume_ms = record.resume_ms,
        tokens_reused = record.tokens_reused,
        continuation_tokens_equal = record.continuation_equal,
        "measured"
    );
    if !record.continuation_equal {
        bail!(
            "the resumed run continued differently from the cold one; \
             the timing below describes a resume that is not a resume"
        );
    }
    write_record(&record, &path)?;
    Ok(())
}

/// One run's numbers.
struct Record {
    prompt_tokens: u32,
    cold_prefill_ms: f64,
    resume_ms: f64,
    tokens_reused: u32,
    continuation_equal: bool,
}

/// Prefills cold and stores, then resumes and compares.
fn run(
    backend: &Arc<dyn Backend>,
    path: &Path,
    config: &Qwen35Config,
    dir: &Path,
) -> Result<Record> {
    let mut opts = EngineOptions::default();
    opts.ctx = CTX;
    let mut model =
        Qwen35Model::load(backend, path, config, opts).context("loading the checkpoint")?;
    model
        .kv_enable(&CkptConfig::new(dir))
        .context("opening the checkpoint directory")?;

    let prompt = prompt(config);

    let cold = Instant::now();
    let (session, first) = stand_on(&mut model, backend, &prompt, false)?;
    let cold_prefill_ms = cold.elapsed().as_secs_f64() * 1000.0;

    // Stored here rather than after the continuation: a save names the position
    // the session is standing on, and a session that has decoded past its prompt
    // is somewhere else. The prompt's own commit already queued it; the save
    // finds it queued, and the drain lands it.
    let outcome = model
        .kv_save(session, Reason::Cold)
        .context("storing the prefix")?;
    let Some(id) = outcome.stored() else {
        bail!("the prefix was not stored: {outcome:?}");
    };
    tracing::info!(?id, "prefix stored");
    if !model.kv_drain(std::time::Duration::from_secs(60)) {
        bail!("the stored prefix did not reach disk in a minute");
    }
    let cold_tail = continue_from(&mut model, backend, session, first)?;
    model
        .close_session(session)
        .context("closing the session")?;

    let warm = Instant::now();
    let (session, first) = stand_on(&mut model, backend, &prompt, true)?;
    let resume_ms = warm.elapsed().as_secs_f64() * 1000.0;
    let warm_tail = continue_from(&mut model, backend, session, first)?;
    let reused = u32::try_from(prompt.len()).unwrap_or(u32::MAX);
    model
        .close_session(session)
        .context("closing the session")?;
    model.teardown().context("releasing the model")?;

    Ok(Record {
        prompt_tokens: reused,
        cold_prefill_ms,
        resume_ms,
        tokens_reused: reused,
        continuation_equal: cold_tail == warm_tail,
    })
}

/// Opens a session and stands it at the end of the prompt, returning the first
/// token the model would produce next.
///
/// With `resume`, the prompt is looked up in the cache - the tree, or the disk
/// behind it - and the session is installed on what was found; without it, the
/// whole prompt is prefilled. The session is left exactly at its frontier.
fn stand_on(
    model: &mut Qwen35Model,
    backend: &Arc<dyn Backend>,
    prompt: &[TokenId],
    resume: bool,
) -> Result<(shiso_core::ids::SessionId, TokenId)> {
    let vocab = usize::try_from(model.info().vocab).unwrap_or(0);
    let mut row = vec![0u8; vocab * 4];
    let mut scratch = vec![0f32; vocab];

    let session = model.open_session().context("opening a session")?;
    let token = if resume {
        let resumed = model
            .kv_resume(session, prompt)
            .context("looking for a stored prefix")?;
        let Some(info) = resumed else {
            bail!("the stored prefix did not match on the second run");
        };
        let Some(logits) = info.frontier_logits else {
            bail!("a resume of the whole prompt came back without its row");
        };
        // The entry carries the row the next token is drawn from, so a resume
        // re-runs no decode step. Reading it here is what proves that.
        backend
            .copy_d2h(logits.row(0)?, &mut row)
            .context("reading the restored frontier row")?;
        greedy(&row, &mut scratch)
    } else {
        let last = model
            .prefill_last(session, prompt)
            .context("the prompt through prefill")?;
        backend
            .copy_d2h(last.row(0)?, &mut row)
            .context("reading the prompt's logits row")?;
        greedy(&row, &mut scratch)
    };

    Ok((session, token))
}

/// Continues a session that is standing at its frontier, greedily.
fn continue_from(
    model: &mut Qwen35Model,
    backend: &Arc<dyn Backend>,
    session: shiso_core::ids::SessionId,
    first: TokenId,
) -> Result<Vec<TokenId>> {
    let vocab = usize::try_from(model.info().vocab).unwrap_or(0);
    let mut row = vec![0u8; vocab * 4];
    let mut scratch = vec![0f32; vocab];
    let mut token = first;
    let mut tail = Vec::with_capacity(CONTINUATION_TOKENS as usize);
    tail.push(token);
    for _ in 1..CONTINUATION_TOKENS {
        let out = model
            .decode_batch(&[DecodeStep { session, token }])
            .context("one decode step")?;
        backend
            .copy_d2h(out.row(0)?, &mut row)
            .context("reading a step's logits row")?;
        token = greedy(&row, &mut scratch);
        tail.push(token);
    }
    Ok(tail)
}

/// The highest-scoring id of one logits row.
fn greedy(row: &[u8], scratch: &mut [f32]) -> TokenId {
    for (slot, bytes) in scratch.iter_mut().zip(row.as_chunks::<4>().0) {
        *slot = f32::from_le_bytes(*bytes);
    }
    argmax(scratch)
}

/// A fixed span of ids, every one inside the vocabulary.
fn prompt(config: &Qwen35Config) -> Vec<TokenId> {
    let vocab = config.shape().vocab.max(1);
    (0..PROMPT_TOKENS)
        .map(|i| TokenId::new(i.saturating_mul(2_654_435_761) % vocab))
        .collect()
}

/// The checkpoint the run measures.
fn model_path() -> PathBuf {
    PathBuf::from(
        std::env::var("SHISO_TEST_MODEL")
            .expect("SHISO_TEST_MODEL must name the checkpoint to measure"),
    )
}

/// The geometry, read from the recorded config.
fn profile() -> Result<Qwen35Config> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/qwen35-config.json");
    let config = shiso_core::HfConfig::from_file(&path)
        .with_context(|| format!("reading the config at {}", path.display()))?;
    Ok(Qwen35Config::from_hf(&config)?)
}

/// Merges this run's numbers into the recorded baseline.
fn write_record(record: &Record, model: &Path) -> Result<()> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("baseline");
    std::fs::create_dir_all(&dir).context("the baseline directory")?;
    let path = dir.join("mac16-10-qwen35-resume.json");
    let mut value = match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|_| serde_json::json!({})),
        Err(_) => serde_json::json!({}),
    };
    let machine = read_command("sysctl", &["-n", "hw.model"]).unwrap_or_else(|| "unknown".into());
    let entry = serde_json::json!({
        "machine": machine,
        "model": model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        "ctx": CTX,
        "prompt_tokens": record.prompt_tokens,
        "cold_prefill_ms": record.cold_prefill_ms,
        "resume_ms": record.resume_ms,
        "tokens_reused": record.tokens_reused,
        "continuation_tokens_equal": record.continuation_equal,
    });
    if let Some(object) = value.as_object_mut() {
        object.insert(machine.clone(), entry);
    }
    std::fs::write(&path, serde_json::to_string_pretty(&value)?)
        .with_context(|| format!("writing {}", path.display()))?;
    tracing::info!(path = %path.display(), "baseline recorded");
    Ok(())
}

/// One line of a command's output, or `None` when it could not be run.
fn read_command(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_owned())
}
