//! What an LFM2 container answers, held to an independent reference.
//!
//! `test-vectors/references/d1/lfm2-logits.json` was captured from the
//! checkpoint's own forward in float32 by
//! `scripts/reference/lfm2_logits_reference.py`: per prompt, the top-64
//! log-probabilities at the last position and the log-probabilities of every
//! token a decision readout scores. The prompts are a chat turn, the three
//! decision prompts of the model card's example, and two long corpus prompts
//! that cross several prefill chunks.
//!
//! Graded per container: the top-1 id must match exactly, and the largest
//! difference over the reference's top-64 and over the readout tokens is held
//! to a budget measured on this port (printed, so a drift shows before it
//! fails). The `q4_K_M` export is graded twice: loosely against the float32
//! reference, whose distance is mostly the export's own rounding, and tightly
//! against `lfm2-q4_k_m-llama.json` - the same container run by llama.cpp
//! (`scripts/reference/lfm2_llama_reference.py`), which shares no kernel with
//! this port and so isolates this port's error from the quantization's. A second case reads one prompt in two chunks split at an odd point
//! and holds the row to the one-shot read: the convolution window and the
//! cache have to carry the first chunk into the second exactly.
//!
//! Needs the container (`SHISO_TEST_MODEL_D1`, and `SHISO_TEST_MODEL_D1_Q8`
//! for the higher-precision export), the fixture and a Metal device, so every
//! case is `#[ignore]`d and says on stdout why it skipped.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::Model;
use shiso_engine::EngineOptions;
use shiso_metal::{MetalBackend, metal_available};
use shiso_modelfile::GgufFile;
use shiso_models::lfm2::{Lfm2Config, Lfm2Model};

/// The largest log-probability difference allowed over a reference's top-64
/// and readout tokens, per container.
///
/// Measured on an M1 Max against the float32 reference: the budgets sit above
/// the worst case seen so that a quantization's own error passes and a
/// structural error (a wrong rotation, a misread gate, a lost window) - which
/// moves log-probabilities by whole nats - does not.
const BUDGET_Q4_K_M: f64 = 1.0;
/// The `q8_0` export's budget against the float32 reference.
const BUDGET_Q8_0: f64 = 0.15;
/// The `q4_K_M` export's budget against the same container run by llama.cpp.
const BUDGET_Q4_K_M_LLAMA: f64 = 0.15;

/// The float32 reference.
const F32_REFERENCE: &str = "lfm2-logits.json";
/// The `q4_K_M` container's llama.cpp reference.
const LLAMA_Q4_K_M: &str = "lfm2-q4_k_m-llama.json";

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Reports what a run measured.
#[expect(
    clippy::print_stdout,
    reason = "the measured distance is a test's output"
)]
fn report(what: &str) {
    println!("{what}");
}

fn fixture(name: &str) -> Option<serde_json::Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/d1")
        .join(name);
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// A token id or a width as a host index.
fn index(value: u64) -> usize {
    usize::try_from(value).expect("an index fits the host")
}

fn container(var: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(var).ok()?);
    path.is_file().then_some(path)
}

fn options() -> EngineOptions {
    let mut opts = EngineOptions::default();
    opts.ctx = 8192;
    opts
}

/// A loaded container and the device it is on.
struct Loaded {
    backend: Arc<dyn Backend>,
    model: Lfm2Model,
    vocab: usize,
}

fn load(path: &Path) -> Loaded {
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("a Metal device");
    let file = GgufFile::open_shared(path).expect("the container opens");
    let config = Lfm2Config::from_gguf(&file).expect("the container's geometry");
    drop(file);
    let model = Lfm2Model::load(&backend, path, &config, options()).expect("the container loads");
    let vocab = index(u64::from(config.shape().vocab));
    Loaded {
        backend,
        model,
        vocab,
    }
}

/// The log-softmax of the last row after reading `chunks` in order.
fn logz(loaded: &mut Loaded, chunks: &[&[TokenId]]) -> Vec<f64> {
    let session = loaded.model.open_session().expect("a session opens");
    let (last, head) = chunks.split_last().expect("at least one chunk");
    for chunk in head {
        loaded
            .model
            .prefill_chunk(session, chunk)
            .expect("a chunk runs");
    }
    let mut bytes = vec![0u8; loaded.vocab * 4];
    {
        let logits = loaded
            .model
            .prefill_last(session, last)
            .expect("the prompt runs");
        loaded
            .backend
            .copy_d2h(logits.row(0).expect("the row"), &mut bytes)
            .expect("the row reads back");
    }
    loaded
        .model
        .close_session(session)
        .expect("the session closes");
    let row: Vec<f64> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f64::from(f32::from_le_bytes(*b)))
        .collect();
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
    let lse = max + sum.ln();
    row.iter().map(|v| v - lse).collect()
}

fn ids(case: &serde_json::Value) -> Vec<TokenId> {
    case["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| TokenId::new(u32::try_from(v.as_u64().expect("an id")).expect("a u32 id")))
        .collect()
}

/// Grades every reference case against the container at `var`.
fn grade(var: &str, reference_name: &str, budget: f64) {
    if !metal_available() {
        skip("no Metal device");
        return;
    }
    let Some(path) = container(var) else {
        skip(&format!("{var} does not name a local container"));
        return;
    };
    let Some(reference) = fixture(reference_name) else {
        skip(&format!("no test-vectors/references/d1/{reference_name}"));
        return;
    };
    let mut loaded = load(&path);
    let mut worst_overall = 0.0f64;
    for case in reference["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("a name");
        let have = logz(&mut loaded, &[&ids(case)]);
        let top = case["top"].as_array().expect("top");
        let want_top = top[0][0].as_u64().map(index).expect("an id");
        let have_top = have
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |best, (i, v)| {
                if *v > best.1 { (i, *v) } else { best }
            })
            .0;
        let mut worst = 0.0f64;
        let pairs = top
            .iter()
            .map(|e| (e[0].as_u64().expect("id"), e[1].as_f64().expect("lp")));
        let readout = case["readout"]
            .as_object()
            .expect("readout")
            .values()
            .map(|e| (e[0].as_u64().expect("id"), e[1].as_f64().expect("lp")));
        for (id, want) in pairs.chain(readout) {
            // Only the plausible part of the distribution is graded: below
            // e^-12 a log-probability is noise in both implementations.
            if want < -12.0 {
                continue;
            }
            worst = worst.max((have[index(id)] - want).abs());
        }
        report(&format!(
            "{var} vs {reference_name} {name}: top-1 {have_top} (reference {want_top}), worst |dlogp| {worst:.4}"
        ));
        assert_eq!(
            have_top, want_top,
            "{name}: top-1 differs from the reference"
        );
        worst_overall = worst_overall.max(worst);
    }
    report(&format!(
        "{var}: worst over all cases {worst_overall:.4}, budget {budget}"
    ));
    assert!(
        worst_overall <= budget,
        "worst |dlogp| {worst_overall:.4} exceeds the budget {budget}"
    );
    loaded.model.teardown().expect("the model tears down");
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_D1, the reference fixture and a Metal device"]
fn the_q4_k_m_container_agrees_with_the_reference() {
    grade("SHISO_TEST_MODEL_D1", F32_REFERENCE, BUDGET_Q4_K_M);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_D1, the llama.cpp fixture and a Metal device"]
fn the_q4_k_m_container_agrees_with_llama_cpp_on_the_same_container() {
    grade("SHISO_TEST_MODEL_D1", LLAMA_Q4_K_M, BUDGET_Q4_K_M_LLAMA);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_D1_Q8, the reference fixture and a Metal device"]
fn the_q8_0_container_agrees_with_the_reference() {
    grade("SHISO_TEST_MODEL_D1_Q8", F32_REFERENCE, BUDGET_Q8_0);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_D1, the reference fixture and a Metal device"]
fn a_prompt_read_in_two_chunks_answers_as_one_read() {
    if !metal_available() {
        skip("no Metal device");
        return;
    }
    let Some(path) = container("SHISO_TEST_MODEL_D1") else {
        skip("SHISO_TEST_MODEL_D1 does not name a local container");
        return;
    };
    let Some(reference) = fixture(F32_REFERENCE) else {
        skip("no test-vectors/references/d1/lfm2-logits.json");
        return;
    };
    let mut loaded = load(&path);
    let case = &reference["cases"][2];
    let prompt = ids(case);
    let whole = logz(&mut loaded, &[&prompt]);
    // An odd split, so neither half is a multiple of anything a kernel tiles
    // by, and the second half starts mid-window.
    let (a, b) = prompt.split_at(prompt.len() / 2 + 1);
    let split = logz(&mut loaded, &[a, b]);
    let worst = whole
        .iter()
        .zip(&split)
        .filter(|(w, _)| **w > -12.0)
        .map(|(w, s)| (w - s).abs())
        .fold(0.0f64, f64::max);
    report(&format!("two chunks against one: worst |dlogp| {worst:.6}"));
    assert!(worst < 1e-3, "a split read moved the row by {worst}");
    loaded.model.teardown().expect("the model tears down");
}
