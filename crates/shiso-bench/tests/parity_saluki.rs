//! The mixed i-quant Saluki 27B against the reference runtime.
//!
//! The recording was made by a different implementation of the same file -
//! the reference runtime named in `PROVENANCE.md`, served on this machine's
//! Metal device, by `scripts/reference/llama_server_vec.py`, which sends the
//! ids this vocabulary produces - so a disagreement is a defect in one of the two stacks rather
//! than a changed answer. The reference shares no kernel with this port: not
//! the nine i-quant decoders, not the GDN recurrence, not the reduction order.
//!
//! The file is the GSQ-RCO "IQ2-mix" export: 851 tensors spread over nine
//! block formats, with `token_embd.weight` in `IQ1_M` (row gather) and
//! `output.weight` in `IQ3_S` (matvec), so every frontier exercises the mixed
//! dispatch end to end.
//!
//! ```text
//! SHISO_TEST_MODEL_SALUKI=<the GGUF> \
//! SHISO_TEST_MODEL_SALUKI_SIDECAR=<the Qwen3.8-27B directory> \
//!   cargo test -p shiso-bench --release --test parity_saluki \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The sidecar directory is required: its `config.json` describes the GGUF's
//! geometry and its `tokenizer.json` is the vocabulary the capture encoded with
//! (the same file the Bonsai captures used, digest `ce3f4653fe1911fb`).
//!
//! Measured on an M1 Max: top-1 and all 64 of the top-64 agree at every
//! frontier, the worst top-20 log-probability gap is 0.003-0.019 nats on the
//! three short frontiers and 0.006-0.012 on the two long ones - inside
//! [`REFERENCE`] everywhere, though only the short ones are held to it.

use std::path::{Path, PathBuf};

use shiso_bench::compare::{FLOOR, REFERENCE, compare};
use shiso_bench::replay::{Device, Harness, Vocabulary};
use shiso_bench::run::{REPLAY_CHUNK, case_tokens, logprob_top_k, prompts_digest, tokens_digest};
use shiso_bench::vecfile::{self, Body, LocalCase, VecFile};
use shiso_core::ids::TokenId;

/// The IQ2-mix GGUF's capture.
const GOLDEN: &str = "references/saluki27b-iq2_mix-llama-golden.vec";

/// The compute device this platform's text lane runs on.
#[cfg(target_os = "macos")]
const DEVICE: Device = Device::Metal;
/// The compute device this platform's text lane runs on.
#[cfg(not(target_os = "macos"))]
const DEVICE: Device = Device::Cuda;

#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

#[expect(clippy::print_stdout, reason = "the measurement is a test's output")]
fn report(what: &str) {
    println!("{what}");
}

fn vectors_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors")
}

/// A named path, when this box has it.
fn checkpoint(var: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(var).ok()?);
    path.exists().then_some(path)
}

fn device_available() -> bool {
    match DEVICE {
        Device::Metal => shiso_metal::metal_available(),
        Device::Cuda => shiso_cuda::cuda_available(),
    }
}

fn open(ctx: u32) -> Option<(Harness, Vocabulary)> {
    let Some(sidecar) = checkpoint("SHISO_TEST_MODEL_SALUKI_SIDECAR") else {
        skip("SHISO_TEST_MODEL_SALUKI_SIDECAR names no Qwen3.8-27B directory");
        return None;
    };
    let Some(model) = checkpoint("SHISO_TEST_MODEL_SALUKI") else {
        skip("SHISO_TEST_MODEL_SALUKI names no checkpoint");
        return None;
    };
    if !device_available() {
        skip("no device on this machine");
        return None;
    }
    let vocabulary =
        Vocabulary::load(&sidecar.join("tokenizer.json")).expect("the sidecar's vocabulary loads");
    let harness = Harness::open(DEVICE, &model, &sidecar.join("config.json"), ctx)
        .expect("the checkpoint loads onto the device");
    Some((harness, vocabulary))
}

fn cases(file: &VecFile) -> Vec<LocalCase> {
    match &file.body {
        Body::Local(cases) => cases.clone(),
        Body::Official(_) | Body::Tower(_) => panic!("the reference fixture is a token-id fixture"),
    }
}

/// The case's tokens, checked against the ids the capture recorded.
fn agreed_tokens(file: &VecFile, vocabulary: &Vocabulary, case: &LocalCase) -> Vec<TokenId> {
    let tokens = case_tokens(vocabulary, &vectors_dir().join("references"), case)
        .expect("the prompt encodes");
    assert_eq!(
        u32::try_from(tokens.len()).unwrap_or(u32::MAX),
        case.frontier,
        "case {}: the prompt is {} tokens here and the reference evaluated {}",
        case.id,
        tokens.len(),
        case.frontier
    );
    let key = format!("tokens_digest_{}", case.id);
    assert_eq!(
        Some(&tokens_digest(&tokens)),
        file.headers.get(&key),
        "case {}: the two sides tokenized this prompt into different ids",
        case.id
    );
    tokens
}

/// Frontiers at or below this many tokens are held to the budgets a caller
/// grades; longer ones to [`FLOOR`] with the rest printed, for the reason the
/// dense 27B's own references give (`parity_qwen38.rs`): 3.5k-4k steps of the
/// recurrence summed in a different order by two implementations.
const SHORT_FRONTIER: u32 = 1024;

/// Replays the fixture, asserting `graded` and printing `reported`.
fn replay(graded: &[&shiso_bench::compare::Budget], reported: &[&shiso_bench::compare::Budget]) {
    let file = vecfile::read(&vectors_dir().join(GOLDEN)).expect("the fixture parses");
    assert_eq!(
        file.headers.get("values").map(String::as_str),
        Some("logprob"),
        "this fixture does not declare log-probabilities"
    );
    let cases = cases(&file);
    let refs: Vec<&LocalCase> = cases.iter().collect();
    assert_eq!(
        file.headers.get("prompts_digest"),
        Some(
            &prompts_digest(&vectors_dir().join("references"), &refs)
                .expect("the prompt files read")
        ),
        "a prompt file changed since the capture"
    );
    let ctx = cases.iter().map(|c| c.ctx).max().unwrap_or(4096);
    let Some((mut harness, vocabulary)) = open(ctx) else {
        return;
    };
    assert_eq!(
        Some(&vocabulary.digest().expect("the vocabulary reads")),
        file.headers.get("vocabulary_digest"),
        "this is not the tokenizer.json the capture encoded with"
    );
    let mut failures = Vec::new();
    for case in &cases {
        let tokens = agreed_tokens(&file, &vocabulary, case);
        let row = harness
            .frontier_row(&tokens, REPLAY_CHUNK)
            .expect("the case runs");
        let produced = logprob_top_k(&row, usize::try_from(case.top_count).unwrap_or(0));
        let short = case.frontier <= SHORT_FRONTIER;
        for budget in graded.iter().filter(|b| short || b.name == FLOOR.name) {
            let verdict = compare(&case.id, &case.tops, &produced, budget);
            report(&verdict.report_line());
            if !verdict.passed() {
                failures.push(format!("{}: {:?}", verdict.report_line(), verdict.failures));
            }
        }
        for budget in reported
            .iter()
            .chain(graded.iter().filter(|b| !short && b.name != FLOOR.name))
        {
            report(&compare(&case.id, &case.tops, &produced, budget).report_line());
        }
    }
    harness.close().expect("the device releases");
    assert!(
        failures.is_empty(),
        "this port disagrees with the reference runtime about the same file \
         ({GOLDEN}): a defect to locate, never a budget to widen.\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the local Saluki checkpoint and a device"]
fn the_mixed_iquant_gguf_replays_the_reference() {
    replay(&[&FLOOR, &REFERENCE], &[]);
}
