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
//! Every grade runs twice: on Metal, and on the CPU device (`cpu_` twins, same
//! fixtures and the same budgets). `cpu_and_metal_agree_on_the_same_prompts`
//! prints how far the two devices sit from each other, which is where a
//! disagreement against the reference is bisected.
//!
//! Needs the container (`SHISO_TEST_MODEL_CLEF_LLAMA`) and the fixture, plus a
//! Metal device for the Metal cases, so every case is `#[ignore]`d and says on
//! stdout why it skipped.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::{DecisionQuestion, DecodeStep, Logits, Model};
use shiso_cpu::{CpuBackend, CpuOptions};
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

/// The device a case runs on.
#[derive(Clone, Copy)]
enum Device {
    /// The Metal device.
    Metal,
    /// The CPU device with this many worker threads (`None`: the host's).
    Cpu(Option<usize>),
}

impl Device {
    /// Whether this device can be opened on this machine.
    fn available(self) -> bool {
        match self {
            Self::Metal => metal_available(),
            Self::Cpu(_) => true,
        }
    }

    /// Opens the device.
    fn open(self) -> Arc<dyn Backend> {
        match self {
            Self::Metal => MetalBackend::new().expect("a Metal device"),
            Self::Cpu(threads) => CpuBackend::new(CpuOptions {
                threads: threads.map(|n| NonZeroUsize::new(n).expect("a thread count")),
            })
            .expect("the CPU device"),
        }
    }
}

/// The container, the fixture and the device, or the reason one is missing.
fn inputs(file: &str, device: Device) -> Option<(PathBuf, serde_json::Value)> {
    if !device.available() {
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

/// The container loaded on `backend`.
fn load_on(backend: Arc<dyn Backend>, path: &Path) -> (Arc<dyn Backend>, LlamaModel) {
    let file = GgufFile::open_shared(path).expect("the container opens");
    let config = LlamaConfig::from_gguf(&file).expect("the container's geometry");
    drop(file);
    let mut opts = EngineOptions::default();
    opts.ctx = 8192;
    let model = LlamaModel::load(&backend, path, &config, opts).expect("the container loads");
    (backend, model)
}

/// The Clef head loaded on `backend`.
fn head_on(backend: &Arc<dyn Backend>, path: &Path) -> ClefHead {
    ClefHead::load(backend, path).expect("the head loads")
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

/// The log-softmax of row `i` of `logits`.
fn row_logp(backend: &Arc<dyn Backend>, logits: &Logits<'_>, i: u32, vocab: usize) -> Vec<f64> {
    let mut bytes = vec![0u8; vocab * 4];
    backend
        .copy_d2h(logits.row(i).expect("the row"), &mut bytes)
        .expect("the row reads back");
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

/// The log-softmax of the last row after reading `prompt`.
fn logz(backend: &Arc<dyn Backend>, model: &mut LlamaModel, prompt: &[TokenId]) -> Vec<f64> {
    let session = model.open_session().expect("a session opens");
    let vocab = model.info().vocab as usize;
    let row = {
        let logits = model
            .prefill_last(session, prompt)
            .expect("the prompt runs");
        row_logp(backend, &logits, 0, vocab)
    };
    model.close_session(session).expect("the session closes");
    row
}

/// The index of the largest value.
fn argmax(row: &[f64]) -> usize {
    row.iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |best, (i, v)| {
            if *v > best.1 { (i, *v) } else { best }
        })
        .0
}

/// The head's per-option logits for a decision case: capture the backbone's
/// rows while reading the prompt, then decide.
fn decide_logits(model: &mut LlamaModel, case: &serde_json::Value) -> Vec<Vec<f32>> {
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
    logits
}

/// The backbone's next-token rows against `grade`'s fixture.
fn backbone(grade: &Grade, device: Device) {
    let Some((path, reference)) = inputs(grade.file, device) else {
        return;
    };
    let (backend, mut model) = load_on(device.open(), &path);
    let mut worst_overall = 0.0f64;
    for case in reference["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("a name");
        let have = logz(&backend, &mut model, &ids(case));
        let want_ids = case["top"]["ids"].as_array().expect("top ids");
        let want_lp = case["top"]["logprobs"].as_array().expect("top logprobs");
        let have_top = argmax(&have);
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
fn head_on_reference_rows(grade: &Grade, device: Device) {
    let Some((path, reference)) = inputs(grade.file, device) else {
        return;
    };
    let backend = device.open();
    let head = head_on(&backend, &path);
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
fn decision_path(grade: &Grade, device: Device) {
    let Some((path, reference)) = inputs(grade.file, device) else {
        return;
    };
    let (_backend, mut model) = load_on(device.open(), &path);
    let mut worst_overall = 0.0f64;
    for case in decision_cases(&reference) {
        let name = case["name"].as_str().expect("a name");
        let logits = decide_logits(&mut model, case);
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
    backbone(&SAME_WEIGHTS, Device::Metal);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA and the reference fixtures; run with --release"]
fn cpu_the_backbone_agrees_with_the_same_weights() {
    backbone(&SAME_WEIGHTS, Device::Cpu(None));
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA, the reference fixtures and a Metal device"]
fn the_backbone_agrees_with_the_bf16_checkpoint() {
    backbone(&BF16, Device::Metal);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA and the reference fixtures; run with --release"]
fn cpu_the_backbone_agrees_with_the_bf16_checkpoint() {
    backbone(&BF16, Device::Cpu(None));
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA, the reference fixtures and a Metal device"]
fn the_head_on_the_reference_rows_agrees_with_the_reference_head() {
    head_on_reference_rows(&SAME_WEIGHTS, Device::Metal);
    head_on_reference_rows(&BF16, Device::Metal);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA and the reference fixtures; run with --release"]
fn cpu_the_head_on_the_reference_rows_agrees_with_the_reference_head() {
    head_on_reference_rows(&SAME_WEIGHTS, Device::Cpu(None));
    head_on_reference_rows(&BF16, Device::Cpu(None));
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA, the reference fixtures and a Metal device"]
fn the_decision_path_agrees_with_the_same_weights() {
    decision_path(&SAME_WEIGHTS, Device::Metal);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA and the reference fixtures; run with --release"]
fn cpu_the_decision_path_agrees_with_the_same_weights() {
    decision_path(&SAME_WEIGHTS, Device::Cpu(None));
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA, the reference fixtures and a Metal device"]
fn the_decision_path_agrees_with_the_bf16_checkpoint() {
    decision_path(&BF16, Device::Metal);
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA and the reference fixtures; run with --release"]
fn cpu_the_decision_path_agrees_with_the_bf16_checkpoint() {
    decision_path(&BF16, Device::Cpu(None));
}

/// The CPU device against Metal on every prompt of the same-weights fixture:
/// the largest log-probability distance over the plausible tokens and the
/// largest per-option probability distance over the decision cases. Asserts
/// only that both are finite; the numbers are for bisecting a grade a device
/// misses.
#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA and the reference fixtures; run with --release"]
fn cpu_and_metal_agree_on_the_same_prompts() {
    if !metal_available() {
        skip("no Metal device");
        return;
    }
    let Some((path, reference)) = inputs(SAME_WEIGHTS.file, Device::Cpu(None)) else {
        return;
    };
    let (metal_backend, mut metal) = load_on(Device::Metal.open(), &path);
    let (cpu_backend, mut cpu) = load_on(Device::Cpu(None).open(), &path);
    let mut worst_logp = 0.0f64;
    for case in reference["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("a name");
        let prompt = ids(case);
        let m = logz(&metal_backend, &mut metal, &prompt);
        let c = logz(&cpu_backend, &mut cpu, &prompt);
        let worst = m
            .iter()
            .zip(&c)
            .filter(|(m, c)| **m > -12.0 || **c > -12.0)
            .map(|(m, c)| (m - c).abs())
            .fold(0.0f64, f64::max);
        report(&format!(
            "cpu vs metal: {name}: top-1 {} / {}, worst |dlogp| {worst:.5}",
            argmax(&c),
            argmax(&m)
        ));
        assert!(worst.is_finite(), "{name}: a non-finite distance");
        worst_logp = worst_logp.max(worst);
    }
    let mut worst_dp = 0.0f64;
    for case in decision_cases(&reference) {
        let name = case["name"].as_str().expect("a name");
        let m = decide_logits(&mut metal, case);
        let c = decide_logits(&mut cpu, case);
        let mut worst = 0.0f64;
        for (m, c) in m.iter().zip(&c) {
            for (a, b) in softmax(m).iter().zip(softmax(c)) {
                worst = worst.max((a - b).abs());
            }
        }
        report(&format!(
            "cpu vs metal: {name}: decision worst |dp| {worst:.6}"
        ));
        assert!(worst.is_finite(), "{name}: a non-finite distance");
        worst_dp = worst_dp.max(worst);
    }
    report(&format!(
        "cpu vs metal: worst |dlogp| {worst_logp:.5}, worst decision |dp| {worst_dp:.6}"
    ));
    metal.teardown().expect("the model tears down");
    cpu.teardown().expect("the model tears down");
}

/// The decision logits and next-token rows of the same-weights fixture, bit
/// for bit, at one worker thread and at four: the launch partitions rows
/// across threads and never splits a reduction, so the thread count may not
/// change a single bit.
#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA and the reference fixtures; run with --release"]
fn cpu_decisions_are_bit_identical_at_one_and_many_threads() {
    let Some((path, reference)) = inputs(SAME_WEIGHTS.file, Device::Cpu(None)) else {
        return;
    };
    let run = |threads: usize| {
        let (backend, mut model) = load_on(Device::Cpu(Some(threads)).open(), &path);
        let mut decisions = Vec::new();
        for case in decision_cases(&reference) {
            decisions.push(decide_logits(&mut model, case));
        }
        let mut rows = Vec::new();
        for case in reference["cases"].as_array().expect("cases") {
            rows.push(logz(&backend, &mut model, &ids(case)));
        }
        model.teardown().expect("the model tears down");
        (decisions, rows)
    };
    let (d1, r1) = run(1);
    let (d4, r4) = run(4);
    assert!(!d1.is_empty(), "the fixture has decision cases");
    let same_decisions = d1.iter().flatten().flatten().count()
        == d4.iter().flatten().flatten().count()
        && d1
            .iter()
            .flatten()
            .flatten()
            .zip(d4.iter().flatten().flatten())
            .all(|(a, b)| a.to_bits() == b.to_bits());
    let same_rows = r1
        .iter()
        .flatten()
        .zip(r4.iter().flatten())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    report(&format!(
        "1 vs 4 threads: {} decision cases and {} rows, decisions identical {same_decisions}, rows identical {same_rows}",
        d1.len(),
        r1.len()
    ));
    assert!(
        same_decisions,
        "decision logits differ across thread counts"
    );
    assert!(same_rows, "next-token rows differ across thread counts");
}

/// Two sessions with different prompts decoding one step together must read
/// what each reads decoding alone: the only model-level coverage of the
/// two-row matvec and of per-row page tables.
#[test]
#[ignore = "needs SHISO_TEST_MODEL_CLEF_LLAMA and the reference fixtures; run with --release"]
fn cpu_batched_decode_matches_single_decodes() {
    let Some((path, reference)) = inputs(SAME_WEIGHTS.file, Device::Cpu(None)) else {
        return;
    };
    let (backend, mut model) = load_on(Device::Cpu(None).open(), &path);
    let vocab = model.info().vocab as usize;
    let mut cases: Vec<Vec<TokenId>> = reference["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(ids)
        .collect();
    cases.sort_by_key(Vec::len);
    cases.dedup_by_key(|c| c.len());
    assert!(cases.len() >= 2, "two prompts of different lengths");
    let prompts = [cases[0].clone(), cases[cases.len() - 1].clone()];
    assert_ne!(prompts[0].len(), prompts[1].len());

    // Reads both prompts, picks each one's own top-1 as the token to feed.
    let open = |model: &mut LlamaModel| {
        let mut out = Vec::new();
        for prompt in &prompts {
            let session = model.open_session().expect("a session opens");
            let row = {
                let logits = model
                    .prefill_last(session, prompt)
                    .expect("the prompt runs");
                row_logp(&backend, &logits, 0, vocab)
            };
            let token = TokenId::new(u32::try_from(argmax(&row)).expect("a u32 id"));
            out.push(DecodeStep { session, token });
        }
        out
    };
    let solo_steps = open(&mut model);
    let mut solo = Vec::new();
    for step in &solo_steps {
        let row = {
            let logits = model.decode_batch(&[*step]).expect("a solo step runs");
            row_logp(&backend, &logits, 0, vocab)
        };
        solo.push(row);
    }
    let batch_steps = open(&mut model);
    let together = {
        let logits = model.decode_batch(&batch_steps).expect("a batch runs");
        [
            row_logp(&backend, &logits, 0, vocab),
            row_logp(&backend, &logits, 1, vocab),
        ]
    };
    for (i, (alone, both)) in solo.iter().zip(&together).enumerate() {
        let worst = alone
            .iter()
            .zip(both)
            .filter(|(a, b)| **a > -12.0 || **b > -12.0)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        report(&format!(
            "batched decode: prompt {i} ({} tokens): top-1 {} alone / {} batched, worst |dlogp| {worst:.6}",
            prompts[i].len(),
            argmax(alone),
            argmax(both)
        ));
        assert_eq!(argmax(alone), argmax(both), "prompt {i}: top-1 differs");
        assert!(
            worst <= SAME_WEIGHTS.logp,
            "prompt {i}: worst |dlogp| {worst:.6}"
        );
    }
    model.teardown().expect("the model tears down");
}
