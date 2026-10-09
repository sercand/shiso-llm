//! The rotated ternary Bonsai 27B against the checkpoint vendor's own runtime
//! (named in `PROVENANCE.md`; "the fork" below).
//!
//! The recordings were made by a different implementation of the same files -
//! the fork's server on this machine's Metal device, by
//! `scripts/reference/llama_server_vec.py`, which sends the ids this
//! vocabulary produces - so a disagreement is a defect in one of the two
//! stacks rather than a changed answer. The fork shares no kernel with this
//! port: not the ternary decode, not the rotation, not the reduction order.
//!
//! Two captures, one per GGUF encoding; they agree with each other to about
//! 5e-6 nats at every frontier, which is the fork saying the two files are one
//! model. The MLX build is a lossless re-encoding of the 2-bit file and is
//! graded against that capture.
//!
//! ```text
//! SHISO_TEST_MODEL_BONSAI_MLX=<mlx dir> SHISO_TEST_MODEL_BONSAI_PQ2=<gguf> \
//! SHISO_TEST_MODEL_BONSAI_PTQ1=<gguf> cargo test -p shiso-bench --release \
//!   --test parity_bonsai -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The MLX directory is required by every case: its `config.json` describes the
//! GGUF files too, and its `tokenizer.json` is the vocabulary the capture used.
//!
//! Measured on an M1 Max, all three encodings: top-1 and all 64 of the top-64
//! agree at every frontier, the worst top-20 log-probability gap is 0.002-0.003
//! nats on the three short frontiers and 0.006-0.008 on the two long ones -
//! inside [`REFERENCE`] everywhere, though only the short ones are held to it.

use std::path::{Path, PathBuf};

use shiso_bench::compare::{FLOOR, REFERENCE, compare};
use shiso_bench::replay::{Device, Harness, Vocabulary};
use shiso_bench::run::{REPLAY_CHUNK, case_tokens, logprob_top_k, prompts_digest, tokens_digest};
use shiso_bench::vecfile::{self, Body, LocalCase, VecFile};
use shiso_core::ids::TokenId;

/// The 2-bit GGUF's capture.
const GOLDEN_PQ2: &str = "references/bonsai27b-pq2_0-llama-golden.vec";

/// The trit GGUF's capture.
const GOLDEN_PTQ1: &str = "references/bonsai27b-ptq1_0-llama-golden.vec";

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

/// One encoding: the variable naming it, and whether it is a file or a
/// directory.
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

fn open(var: &str, ctx: u32) -> Option<(Harness, Vocabulary)> {
    let Some(mlx) = checkpoint("SHISO_TEST_MODEL_BONSAI_MLX") else {
        skip("SHISO_TEST_MODEL_BONSAI_MLX names no export directory");
        return None;
    };
    let Some(model) = checkpoint(var) else {
        skip(&format!("{var} names no checkpoint"));
        return None;
    };
    if !device_available() {
        skip("no device on this machine");
        return None;
    }
    let vocabulary =
        Vocabulary::load(&mlx.join("tokenizer.json")).expect("the export's vocabulary loads");
    let harness = Harness::open(DEVICE, &model, &mlx.join("config.json"), ctx)
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

/// Replays one fixture, asserting `graded` and printing `reported`.
fn replay(
    var: &str,
    golden: &str,
    graded: &[&shiso_bench::compare::Budget],
    reported: &[&shiso_bench::compare::Budget],
) {
    let file = vecfile::read(&vectors_dir().join(golden)).expect("the fixture parses");
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
    let Some((mut harness, vocabulary)) = open(var, ctx) else {
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
        "this port disagrees with the fork about the same file ({golden}): a defect to locate, \n\
         never a budget to widen.\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the local Bonsai checkpoints and a device"]
fn the_2bit_gguf_replays_the_fork() {
    replay(
        "SHISO_TEST_MODEL_BONSAI_PQ2",
        GOLDEN_PQ2,
        &[&FLOOR, &REFERENCE],
        &[],
    );
}

#[test]
#[ignore = "needs the local Bonsai checkpoints and a device"]
fn the_trit_gguf_replays_the_fork() {
    replay(
        "SHISO_TEST_MODEL_BONSAI_PTQ1",
        GOLDEN_PTQ1,
        &[&FLOOR, &REFERENCE],
        &[],
    );
}

#[test]
#[ignore = "needs the local Bonsai checkpoints and a device"]
fn the_mlx_build_replays_the_fork_on_its_source_file() {
    replay(
        "SHISO_TEST_MODEL_BONSAI_MLX",
        GOLDEN_PQ2,
        &[&FLOOR, &REFERENCE],
        &[],
    );
}
