//! The dense 27B checkpoint of the Qwen 3.5 family against `mlx-lm`'s own rows.
//!
//! The recording was made by a different implementation of the same MLX 4-bit
//! export - `mlx-lm`, by `scripts/reference/mlx_logit_vectors.py` - so a
//! disagreement is a defect in one of the two stacks rather than a changed
//! answer. It grades the same five frontiers the 4B checkpoint's references
//! use, over a 64-layer stack that shares every kernel with that one and
//! differs only in its widths and depth.
//!
//! ```text
//! SHISO_TEST_MODEL_QWEN38_27B=<the MLX export directory> cargo test -p shiso-bench \
//!   --release --test parity_qwen38 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Two captures of the same five frontiers. The one held to [`REFERENCE`] ran
//! `mlx-lm` with its activations cast to `float32`, the width this port
//! computes at. The default `bfloat16` capture is held only to [`FLOOR`] - the
//! answer and the top five - and its [`MLX_REFERENCE`] verdict is printed: over
//! 64 layers and 4k tokens `mlx-lm` in `bfloat16` sits 0.59 nats from `mlx-lm`
//! in `float32` on `long_code_audit` (measured on an M1 Max), which is the whole
//! of this port's distance from it there, so a budget on that fixture would be
//! a budget on the reference's number format.

use std::path::{Path, PathBuf};

use shiso_bench::compare::{FLOOR, MLX_REFERENCE, REFERENCE, compare};
use shiso_bench::replay::{Device, Harness, Vocabulary};
use shiso_bench::run::{REPLAY_CHUNK, case_tokens, logprob_top_k, prompts_digest, tokens_digest};
use shiso_bench::vecfile::{self, Body, LocalCase, VecFile};
use shiso_core::ids::TokenId;

/// The reference captured with `mlx-lm`'s activations in `float32`: the width
/// this port computes at, so it is held to [`REFERENCE`].
const GOLDEN_F32: &str = "references/qwen38-27b-mlx-f32-golden.vec";

/// The reference captured the way `mlx-lm` runs by default, in `bfloat16`.
const GOLDEN_BF16: &str = "references/qwen38-27b-mlx-golden.vec";

/// The greedy trajectory of the `float32` reference.
const GREEDY_LOCK: &str = "references/qwen38-27b-mlx-f32-greedy-64.json";

/// Leading greedy ids that have to agree; see `parity_gemma4.rs` for why not 64.
const GREEDY_MIN_AGREEMENT: usize = 8;

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

/// The export directory, or `None` when this box has none.
fn model_dir() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_QWEN38_27B").ok()?);
    path.is_dir().then_some(path)
}

fn device_available() -> bool {
    match DEVICE {
        Device::Metal => shiso_metal::metal_available(),
        Device::Cuda => shiso_cuda::cuda_available(),
    }
}

fn open(ctx: u32) -> Option<(Harness, Vocabulary)> {
    let Some(dir) = model_dir() else {
        skip("SHISO_TEST_MODEL_QWEN38_27B names no export directory");
        return None;
    };
    if !device_available() {
        skip("no device on this machine");
        return None;
    }
    let vocabulary =
        Vocabulary::load(&dir.join("tokenizer.json")).expect("the export's vocabulary loads");
    let harness = Harness::open(DEVICE, &dir, &dir.join("config.json"), ctx)
        .expect("the export loads onto the device");
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
/// grades; longer ones to [`FLOOR`] with the rest printed.
///
/// Measured on an M1 Max against the `float32` capture: the three short
/// frontiers (15 to 23 tokens) sit 0.013 to 0.037 nats from it, and the two long
/// ones (3493 and 3957 tokens) 0.092 and 0.158, with the answer and the top five
/// still agreeing and a 64-token greedy continuation agreeing id for id. The
/// long gap is 3.5k-4k steps of the recurrence summed in a different order by
/// two implementations - this port's chunked prefill against the reference's
/// per-token loop - and the 4B checkpoint's own reference is graded the same
/// way for the same reason (see `test-vectors/README.md`).
const SHORT_FRONTIER: u32 = 1024;

/// Replays one fixture, asserting `graded` and printing `reported`.
fn replay(
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
        "this port disagrees with mlx-lm about the same export ({golden}): a defect to locate, \n\
         never a budget to widen.\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the local Qwen3.8-27B MLX export and a device"]
fn the_float32_reference_row_replays_at_the_reference_budget() {
    replay(GOLDEN_F32, &[&FLOOR, &REFERENCE], &[]);
}

#[test]
#[ignore = "needs the local Qwen3.8-27B MLX export and a device"]
fn the_bfloat16_reference_row_agrees_on_the_answer() {
    replay(GOLDEN_BF16, &[&FLOOR], &[&MLX_REFERENCE]);
}

#[test]
#[ignore = "needs the local Qwen3.8-27B MLX export and a device"]
fn the_greedy_trajectory_follows_the_reference_one() {
    let Some((mut harness, vocabulary)) = open(1024) else {
        return;
    };
    let lock: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(vectors_dir().join(GREEDY_LOCK)).expect("the lock is present"),
    )
    .expect("the lock parses");
    let prompt: Vec<TokenId> = lock["prompt_tokens"]
        .as_array()
        .expect("the lock records its prompt ids")
        .iter()
        .map(|v| TokenId::new(u32::try_from(v.as_u64().expect("an id")).expect("a u32 id")))
        .collect();
    let want: Vec<u32> = lock["tokens"]
        .as_array()
        .expect("the lock carries a token array")
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("an id")).expect("a u32 id"))
        .collect();
    let produced = harness
        .greedy_after(&prompt, REPLAY_CHUNK, want.len())
        .expect("the continuation runs");
    harness.close().expect("the device releases");
    let ids: Vec<u32> = produced.iter().map(|t| t.get()).collect();
    let agreed = ids.iter().zip(&want).take_while(|(a, b)| a == b).count();
    report(&format!(
        "qwen38-27b greedy lock: {agreed}/{} ids agree with mlx-lm",
        want.len()
    ));
    report(&format!(
        "qwen38-27b greedy text: {:?}",
        vocabulary.decode(&produced).unwrap_or_default()
    ));
    assert!(
        agreed >= GREEDY_MIN_AGREEMENT,
        "the two stacks part after {agreed} greedy ids: shiso {ids:?}; mlx-lm {want:?}"
    );
}
