//! What a Clef container over a dense llama backbone answers, held to an
//! independent reference.
//!
//! Two fixtures under `test-vectors/references/clef-llama/`, both captured by
//! `scripts/reference/clef_llama_reference.py` from the checkpoint's own
//! backbone and the release's own joint schema head, in float32 on the CPU:
//! `reference.json` over the checkpoint's bf16 weights, and
//! `reference-gguf.json` over this very container's `q8_0` weights,
//! dequantized. Per prompt, the ids and the top-64 next-token
//! log-probabilities; per decision record, the spans the release lays out, the
//! backbone's final normalized rows and the head's per-option probabilities.
//!
//! The second fixture is what separates this port's error from the export's.
//! The `q8_0` rounding alone moves the 135M backbone's log-probabilities by up
//! to 0.42 nats from the bf16 checkpoint (the two fixtures against each other),
//! so the bf16 grade is loose and the same-weights grade is the tight one.
//!
//! Three grades per fixture, so a disagreement says where it is:
//!
//! - the backbone's next-token row against the reference's, top-1 exactly and
//!   the plausible log-probabilities within a budget;
//! - the head alone, fed the reference's own rows, against the reference's
//!   probabilities - the head's arithmetic with no backbone drift in it;
//! - the whole decision path, this port's rows into this port's head, against
//!   the same probabilities.
//!
//! The budgets sit above what an M1 Max measured, and every run prints its
//! distances so a drift shows before it fails.
//!
//! Needs the container (`SHISO_TEST_MODEL_CLEF_LLAMA`), the fixture and a Metal
//! device, so every case is `#[ignore]`d and says on stdout why it skipped.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::{DecisionQuestion, Model};
use shiso_engine::EngineOptions;
use shiso_metal::{MetalBackend, metal_available};
use shiso_modelfile::GgufFile;
use shiso_models::clef::ClefHead;
use shiso_models::llama::{LlamaConfig, LlamaModel};

/// One fixture and what it is graded at.
struct Grade {
    /// The fixture's file name.
    file: &'static str,
    /// The largest log-probability difference allowed over the reference's
    /// plausible top-64.
    logp: f64,
    /// The largest per-option probability difference allowed when the head
    /// reads the reference's own rows: the head's `q8_0` linears against its
    /// float32 ones, and nothing else.
    head: f64,
    /// The largest per-option probability difference allowed over the whole
    /// decision path.
    decision: f64,
}

/// The bf16 checkpoint: the export's rounding plus this port's. Measured
/// 0.418 nats, 1.8e-4 and 0.0070.
const BF16: Grade = Grade {
    file: "reference.json",
    logp: 0.5,
    head: 0.005,
    decision: 0.02,
};

/// This container's own weights: this port's error alone.
const SAME_WEIGHTS: Grade = Grade {
    file: "reference-gguf.json",
    logp: 0.05,
    head: 0.005,
    decision: 0.005,
};

/// The container variable.
const VAR: &str = "SHISO_TEST_MODEL_CLEF_LLAMA";

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

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/clef-llama")
}

fn fixture(file: &str) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(fixture_dir().join(file)).ok()?;
    serde_json::from_str(&text).ok()
}

/// The container, the fixture and the device, or the reason one is missing.
fn inputs(file: &str) -> Option<(PathBuf, serde_json::Value)> {
    if !metal_available() {
        skip("no Metal device");
        return None;
    }
    let Some(path) = std::env::var(VAR)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_file())
    else {
        skip(&format!("{VAR} does not name a local container"));
        return None;
    };
    let Some(reference) = fixture(file) else {
        skip(&format!("no test-vectors/references/clef-llama/{file}"));
        return None;
    };
    Some((path, reference))
}

fn load(path: &Path) -> (Arc<dyn Backend>, LlamaModel) {
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("a Metal device");
    let file = GgufFile::open_shared(path).expect("the container opens");
    let config = LlamaConfig::from_gguf(&file).expect("the container's geometry");
    drop(file);
    let mut opts = EngineOptions::default();
    opts.ctx = 8192;
    let model = LlamaModel::load(&backend, path, &config, opts).expect("the container loads");
    (backend, model)
}

/// A host index out of a fixture number.
fn index(value: &serde_json::Value) -> usize {
    usize::try_from(value.as_u64().expect("a non-negative integer")).expect("fits the host")
}

fn ids(case: &serde_json::Value) -> Vec<TokenId> {
    case["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| TokenId::new(u32::try_from(v.as_u64().expect("an id")).expect("a u32 id")))
        .collect()
}

/// The questions a decision case's spans describe, in the head's order.
fn questions(case: &serde_json::Value) -> Vec<DecisionQuestion> {
    let span = |v: &serde_json::Value| index(&v[0])..index(&v[1]);
    case["questions"]
        .as_array()
        .expect("questions")
        .iter()
        .map(|q| DecisionQuestion {
            kind: u32::try_from(q["kind"].as_u64().expect("a kind")).expect("a small kind"),
            span: span(&q["span"]),
            options: q["options"]
                .as_array()
                .expect("options")
                .iter()
                .map(span)
                .collect(),
        })
        .collect()
}

fn decision_cases(reference: &serde_json::Value) -> Vec<&serde_json::Value> {
    reference["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .filter(|c| c.get("questions").is_some())
        .collect()
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f64> = logits.iter().map(|v| f64::from(v - top).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

/// The largest per-option probability difference between `logits` and the
/// reference's probabilities for `case`.
fn worst_probability(case: &serde_json::Value, logits: &[Vec<f32>]) -> f64 {
    let mut worst = 0.0f64;
    for (q, have) in case["questions"]
        .as_array()
        .expect("questions")
        .iter()
        .zip(logits)
    {
        let want: Vec<f64> = q["probabilities"]
            .as_array()
            .expect("probabilities")
            .iter()
            .map(|p| p.as_f64().expect("a probability"))
            .collect();
        let have = softmax(have);
        assert_eq!(have.len(), want.len(), "{}: option count", q["id"]);
        for (h, w) in have.iter().zip(&want) {
            worst = worst.max((h - w).abs());
        }
    }
    worst
}

/// The log-softmax of the last row after reading `prompt`.
fn logz(backend: &Arc<dyn Backend>, model: &mut LlamaModel, prompt: &[TokenId]) -> Vec<f64> {
    let session = model.open_session().expect("a session opens");
    let mut bytes = vec![0u8; model.info().vocab as usize * 4];
    {
        let logits = model
            .prefill_last(session, prompt)
            .expect("the prompt runs");
        backend
            .copy_d2h(logits.row(0).expect("the row"), &mut bytes)
            .expect("the row reads back");
    }
    model.close_session(session).expect("the session closes");
    let row: Vec<f64> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f64::from(f32::from_le_bytes(*b)))
        .collect();
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
    row.iter().map(|v| v - lse).collect()
}

/// The backbone's next-token rows against `grade`'s fixture.
fn backbone(grade: &Grade) {
    let Some((path, reference)) = inputs(grade.file) else {
        return;
    };
    let (backend, mut model) = load(&path);
    let mut worst_overall = 0.0f64;
    for case in reference["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("a name");
        let have = logz(&backend, &mut model, &ids(case));
        let want_ids = case["top"]["ids"].as_array().expect("top ids");
        let want_lp = case["top"]["logprobs"].as_array().expect("top logprobs");
        let have_top = have
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |best, (i, v)| {
                if *v > best.1 { (i, *v) } else { best }
            })
            .0;
        let mut worst = 0.0f64;
        for (id, lp) in want_ids.iter().zip(want_lp) {
            let want = lp.as_f64().expect("a log-probability");
            // Below e^-12 a log-probability is noise in both implementations.
            if want < -12.0 {
                continue;
            }
            worst = worst.max((have[index(id)] - want).abs());
        }
        let want_top = index(&want_ids[0]);
        report(&format!(
            "{}: {name}: {} tokens, top-1 {have_top} (reference {want_top}), worst |dlogp| {worst:.4}",
            grade.file,
            case["ids"].as_array().map_or(0, Vec::len)
        ));
        assert_eq!(
            have_top, want_top,
            "{name}: top-1 differs from the reference"
        );
        worst_overall = worst_overall.max(worst);
    }
    report(&format!(
        "{}: worst over all cases {worst_overall:.4}, budget {}",
        grade.file, grade.logp
    ));
    assert!(
        worst_overall <= grade.logp,
        "worst |dlogp| {worst_overall:.4}"
    );
    model.teardown().expect("the model tears down");
}

/// The head alone, on `grade`'s fixture's own rows.
fn head_on_reference_rows(grade: &Grade) {
    let Some((path, reference)) = inputs(grade.file) else {
        return;
    };
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("a Metal device");
    let head = ClefHead::load(&backend, &path).expect("the head loads");
    let mut worst_overall = 0.0f64;
    for case in decision_cases(&reference) {
        let name = case["name"].as_str().expect("a name");
        let bytes = std::fs::read(fixture_dir().join(case["hidden_file"].as_str().expect("file")))
            .expect("the rows read");
        let rows: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let logits = head
            .score(&rows, &ids(case), &questions(case))
            .expect("the head scores");
        let worst = worst_probability(case, &logits);
        report(&format!(
            "{}: {name}: head on reference rows, worst |dp| {worst:.5}",
            grade.file
        ));
        worst_overall = worst_overall.max(worst);
    }
    report(&format!("worst {worst_overall:.5}, budget {}", grade.head));
    assert!(worst_overall <= grade.head, "worst |dp| {worst_overall:.5}");
}

/// The whole decision path against `grade`'s fixture's probabilities.
fn decision_path(grade: &Grade) {
    let Some((path, reference)) = inputs(grade.file) else {
        return;
    };
    let (_backend, mut model) = load(&path);
    let mut worst_overall = 0.0f64;
    for case in decision_cases(&reference) {
        let name = case["name"].as_str().expect("a name");
        let prompt = ids(case);
        let session = model.open_session().expect("a session opens");
        let decision = model.decision_mut().expect("the head is resident");
        decision.capture_hidden(session).expect("capturing");
        // The next-token row is not read: the rows the capture kept are.
        let _ = model
            .prefill_last(session, &prompt)
            .expect("the prompt runs");
        let logits = model
            .decision_mut()
            .expect("the head is resident")
            .decide(session, &prompt, &questions(case))
            .expect("the head scores");
        model.close_session(session).expect("the session closes");
        let worst = worst_probability(case, &logits);
        report(&format!(
            "{}: {name}: decision path, worst |dp| {worst:.5}",
            grade.file
        ));
        worst_overall = worst_overall.max(worst);
    }
    report(&format!(
        "worst {worst_overall:.5}, budget {}",
        grade.decision
    ));
    assert!(
        worst_overall <= grade.decision,
        "worst |dp| {worst_overall:.5}"
    );
    model.teardown().expect("the model tears down");
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA, the reference fixtures and a Metal device"]
fn the_backbone_agrees_with_the_same_weights() {
    backbone(&SAME_WEIGHTS);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA, the reference fixtures and a Metal device"]
fn the_backbone_agrees_with_the_bf16_checkpoint() {
    backbone(&BF16);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA, the reference fixtures and a Metal device"]
fn the_head_on_the_reference_rows_agrees_with_the_reference_head() {
    head_on_reference_rows(&SAME_WEIGHTS);
    head_on_reference_rows(&BF16);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA, the reference fixtures and a Metal device"]
fn the_decision_path_agrees_with_the_same_weights() {
    decision_path(&SAME_WEIGHTS);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA, the reference fixtures and a Metal device"]
fn the_decision_path_agrees_with_the_bf16_checkpoint() {
    decision_path(&BF16);
}
