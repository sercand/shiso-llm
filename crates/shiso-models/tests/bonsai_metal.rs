//! A rotated ternary checkpoint of this family, end to end on Metal, from each
//! of its three published encodings.
//!
//! Two claims:
//!
//! - **It answers.** A greedy continuation of a short chat turn names the
//!   right city. The encodings are read by different kernels and different
//!   loaders, so each is checked on its own.
//! - **The encodings are one model.** All three store the same trits: the
//!   2-bit GGUF in two-bit slots, the trit GGUF packed five to a byte, the MLX
//!   build as a lossless re-encoding of the 2-bit GGUF. Teacher-forced over the
//!   same tokens they agree on every top-1. Not to the bit: the MLX build
//!   stores the recurrent decay as `log(-a)`, which this family's loader turns
//!   back into `-exp(.)`, one rounding away from the GGUF's stored `a`
//!   (measured: max |dlogit| 2.5e-4 for the 2-bit GGUF, 5.7e-4 for the trit
//!   one, both against the MLX build).
//!
//! ```text
//! SHISO_TEST_MODEL_BONSAI_MLX=<mlx dir> \
//! SHISO_TEST_MODEL_BONSAI_PQ2=<PQ2_0 gguf> SHISO_TEST_MODEL_BONSAI_PTQ1=<PTQ1_0 gguf> \
//!   cargo test -p shiso-models --release --features shiso-metal/metal \
//!   --test bonsai_metal -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The MLX directory is required by every case: its `config.json` and
//! `tokenizer.json` describe the GGUF files too.
#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_core::HfConfig;
use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::{DecodeStep, Model};
use shiso_engine::{EngineOptions, argmax};
use shiso_models::qwen35::{Qwen35Config, Qwen35Model};

/// The chat turn every case runs, thinking disabled the way the template does.
const PROMPT: &str = "<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Steps a greedy continuation runs to.
const STEPS: usize = 16;

#[expect(
    clippy::print_stdout,
    reason = "the skip report and the measurements are the test's output"
)]
fn say(what: &str) {
    println!("{what}");
}

/// The MLX directory, the other two paths, or `None` with the reason said.
fn paths() -> Option<(PathBuf, Option<PathBuf>, Option<PathBuf>)> {
    let var = |key: &str| std::env::var(key).ok().map(PathBuf::from);
    let Some(mlx) = var("SHISO_TEST_MODEL_BONSAI_MLX").filter(|p| p.is_dir()) else {
        say("SKIP: SHISO_TEST_MODEL_BONSAI_MLX names no directory");
        return None;
    };
    let pq2 = var("SHISO_TEST_MODEL_BONSAI_PQ2").filter(|p| p.is_file());
    let ptq1 = var("SHISO_TEST_MODEL_BONSAI_PTQ1").filter(|p| p.is_file());
    Some((mlx, pq2, ptq1))
}

fn device() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        say("SKIP: no Metal device");
        return None;
    }
    let backend: Arc<dyn Backend> = shiso_metal::MetalBackend::new().expect("a Metal device");
    Some(backend)
}

fn load(backend: &Arc<dyn Backend>, config_dir: &Path, model: &Path) -> (Qwen35Model, u32) {
    let config = Qwen35Config::from_hf(
        &HfConfig::from_file(&config_dir.join("config.json")).expect("config"),
    )
    .expect("the rotated export is this family");
    let mut opts = EngineOptions::default();
    opts.ctx = 1024;
    opts.kv_pool_pages = 32;
    let started = Instant::now();
    let model = Qwen35Model::load(backend, model, &config, opts).expect("the checkpoint loads");
    say(&format!(
        "loaded in {:.1}s",
        started.elapsed().as_secs_f32()
    ));
    (model, config.shape().vocab)
}

fn tokenizer(mlx: &Path) -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(mlx.join("tokenizer.json")).expect("the tokenizer loads")
}

fn encode(tok: &tokenizers::Tokenizer, text: &str) -> Vec<TokenId> {
    tok.encode(text, false)
        .expect("the prompt encodes")
        .get_ids()
        .iter()
        .map(|&id| TokenId::new(id))
        .collect()
}

fn row(
    backend: &Arc<dyn Backend>,
    view: shiso_core::buffer::DeviceView<'_>,
    vocab: u32,
) -> Vec<f32> {
    let mut bytes = vec![0u8; vocab as usize * 4];
    backend
        .copy_d2h(view, &mut bytes)
        .expect("the row reads back");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The greedy continuation of `prompt`, and the logits rows it was picked from.
fn greedy(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen35Model,
    vocab: u32,
    prompt: &[TokenId],
    forced: Option<&[TokenId]>,
) -> (Vec<TokenId>, Vec<Vec<f32>>) {
    let session = model.open_session().expect("a session");
    let started = Instant::now();
    let last = model
        .prefill_last(session, prompt)
        .expect("the prompt runs");
    let mut logits = vec![row(backend, last.row(0).expect("row"), vocab)];
    let prefill = started.elapsed();
    let mut produced = vec![argmax(&logits[0])];
    let started = Instant::now();
    for step in 1..STEPS {
        let token = forced.map_or(produced[step - 1], |f| f[step - 1]);
        let out = model
            .decode_batch(&[DecodeStep { session, token }])
            .expect("a step runs");
        logits.push(row(backend, out.row(0).expect("row"), vocab));
        produced.push(argmax(logits.last().expect("a row")));
    }
    let decode = started.elapsed();
    say(&format!(
        "prefill {} tokens {:.0} ms; decode {:.1} tok/s",
        prompt.len(),
        prefill.as_secs_f64() * 1e3,
        f64::from(u32::try_from(STEPS - 1).unwrap_or(0)) / decode.as_secs_f64()
    ));
    model.close_session(session).expect("closes");
    (produced, logits)
}

/// One encoding's label, greedy ids and the logits rows they came from.
type Run = (&'static str, Vec<TokenId>, Vec<Vec<f32>>);

/// Every encoding that is present: label and path.
fn variants(
    mlx: &Path,
    pq2: Option<&PathBuf>,
    ptq1: Option<&PathBuf>,
) -> Vec<(&'static str, PathBuf)> {
    let mut out = vec![("mlx-2bit", mlx.to_path_buf())];
    if let Some(p) = pq2 {
        out.push(("pq2_0", p.clone()));
    }
    if let Some(p) = ptq1 {
        out.push(("ptq1_0", p.clone()));
    }
    out
}

#[test]
#[ignore = "needs the local rotated checkpoints and a Metal device"]
fn every_encoding_answers_and_they_agree() {
    let Some((mlx, pq2, ptq1)) = paths() else {
        return;
    };
    let Some(backend) = device() else { return };
    let tok = tokenizer(&mlx);
    let prompt = encode(&tok, PROMPT);
    let mut runs: Vec<Run> = Vec::new();
    let mut reference: Option<Vec<TokenId>> = None;
    for (label, path) in variants(&mlx, pq2.as_ref(), ptq1.as_ref()) {
        say(&format!("--- {label}: {}", path.display()));
        let (mut model, vocab) = load(&backend, &mlx, &path);
        let (ids, logits) = greedy(&backend, &mut model, vocab, &prompt, reference.as_deref());
        let raw: Vec<u32> = ids.iter().map(|t| t.get()).collect();
        let text = tok.decode(&raw, false).expect("decodes");
        say(&format!("{label}: {text:?}"));
        assert!(
            text.contains("Paris"),
            "{label} did not answer Paris: {text:?}"
        );
        reference.get_or_insert(ids.clone());
        model.teardown().expect("tears down");
        drop(model);
        runs.push((label, ids, logits));
    }
    let (_, first_ids, first_logits) = &runs[0];
    for (label, ids, logits) in &runs[1..] {
        let mut max_delta = 0.0_f32;
        for (a, b) in first_logits.iter().zip(logits) {
            for (x, y) in a.iter().zip(b) {
                max_delta = max_delta.max((x - y).abs());
            }
        }
        say(&format!("{label} vs mlx-2bit: max |dlogit| {max_delta}"));
        assert_eq!(first_ids, ids, "{label} disagrees with mlx-2bit on a top-1");
    }
}
