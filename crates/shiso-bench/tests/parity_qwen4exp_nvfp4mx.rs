//! NVIDIA's mixed-precision Qwen3.8-Flash-Next export against the reference
//! server's own rows over the same checkpoint.
//!
//! The recording was made by a different implementation of the same weights
//! (`scripts/reference/vllm_server_vec.py`), with its own NVFP4 and MXFP8
//! kernels, so a disagreement is a defect in one of the two stacks. It grades
//! the five frontiers the 27B references use.
//!
//! The reference is graded at [`FLOOR`] only, with the tighter budgets
//! printed. Its numerics are not a fixed point to approach: it serves the
//! MXFP8 projections with activations it quantizes to MXFP8 or keeps in brain
//! floats depending on batch size, rounds intermediates to brain floats, and
//! quantizes the routed experts' activations to FP4 on prompt rows. This port
//! keeps activations in single precision against the same weights. The answer
//! and the greedy trajectory are what both must agree on.
//!
//! ```text
//! SHISO_TEST_MODEL_QWEN4EXP_NVFP4MX=<the export directory> cargo test -p shiso-bench \
//!   --release --test parity_qwen4exp_nvfp4mx -- --ignored \
//!   --nocapture --test-threads=1
//! ```

use std::path::{Path, PathBuf};

use shiso_bench::compare::{FLOOR, MLX_REFERENCE, REFERENCE, compare};
use shiso_bench::replay::{Device, Harness, HarnessOptions, Vocabulary};
use shiso_bench::run::{REPLAY_CHUNK, case_tokens, logprob_top_k, prompts_digest, tokens_digest};
use shiso_bench::vecfile::{self, Body, LocalCase, VecFile};
use shiso_core::ids::TokenId;
use shiso_engine::sparse::matmul::PrefillActivations;

/// The reference rows, captured from the reference server over the same
/// checkpoint as that checkpoint's card serves it: FP4 activations on prompt
/// rows, MXFP8 or brain-float activations by batch size.
const GOLDEN: &str = "references/qwen38-flash-fp4local-vllm-golden.vec";

/// The same capture with every activation kept in brain floats, which is the
/// nearer arithmetic to this port's. The two references themselves sit 0.5 to
/// 1.25 nats apart on their top 20 and part after five greedy ids, which is the
/// measure of how much of a gap here is the reference's own quantization.
const GOLDEN_A16: &str = "references/qwen38-flash-fp4local-vllm-a16-golden.vec";

/// The library's own modules over the same weights, dequantized to brain
/// floats, activations in brain floats, logits in single precision
/// (`scripts/reference/qwen4exp_modelopt_mixed_ref.py`): the reference that
/// shares no kernel and no quantized arithmetic with either server. It agrees
/// with the served reference's greedy trajectory on all 64 ids and still sits
/// 0.5 to 1.5 nats from it on the top 20.
const GOLDEN_HF: &str = "references/qwen38-flash-fp4local-hf-a16-golden.vec";

/// The library reference's greedy trajectory after the first prompt.
const GREEDY_LOCK_HF: &str = "references/qwen38-flash-fp4local-hf-a16-greedy-64.json";

/// The reference server's greedy trajectory after the first prompt.
const GREEDY_LOCK: &str = "references/qwen38-flash-fp4local-vllm-greedy-64.json";

/// Leading greedy ids that have to agree.
///
/// Five, measured rather than chosen: the sixth id after the first prompt is a
/// near-tie between two phrasings ("asks" and "is asking"), and the
/// independent references split on it themselves - the served reference in
/// brain floats picks "is asking" as this port does, the library reference and
/// the served reference as published pick "asks". Every case still has to agree
/// up to it.
const GREEDY_MIN_AGREEMENT: usize = 5;

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
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_QWEN4EXP_NVFP4MX").ok()?);
    path.is_dir().then_some(path)
}

fn device_available() -> bool {
    match DEVICE {
        Device::Metal => shiso_metal::metal_available(),
        Device::Cuda => shiso_cuda::cuda_available(),
    }
}

fn open(ctx: u32) -> Option<(Harness, Vocabulary)> {
    open_as(ctx, PrefillActivations::A16)
}

fn open_as(ctx: u32, prefill: PrefillActivations) -> Option<(Harness, Vocabulary)> {
    let Some(dir) = model_dir() else {
        skip("SHISO_TEST_MODEL_QWEN4EXP_NVFP4MX names no export directory");
        return None;
    };
    if !device_available() {
        skip("no device on this machine");
        return None;
    }
    let vocabulary =
        Vocabulary::load(&dir.join("tokenizer.json")).expect("the export's vocabulary loads");
    let harness = Harness::open_with_options(
        DEVICE,
        &dir,
        &dir.join("config.json"),
        ctx,
        HarnessOptions {
            prefill_activations: prefill,
            ..HarnessOptions::default()
        },
    )
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
    replay_as(golden, graded, reported, PrefillActivations::A16);
}

/// [`replay`] at a chosen prompt-pass activation precision.
fn replay_as(
    golden: &str,
    graded: &[&shiso_bench::compare::Budget],
    reported: &[&shiso_bench::compare::Budget],
    prefill: PrefillActivations,
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
    let Some((mut harness, vocabulary)) = open_as(ctx, prefill) else {
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
        "this port disagrees with the reference server about the same checkpoint ({golden}): a \n\
         defect to locate, never a budget to widen.\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the local Qwen3.8-Flash-Next FP4-Local export and a device"]
fn the_reference_rows_agree_on_the_answer() {
    replay(GOLDEN, &[&FLOOR], &[&MLX_REFERENCE, &REFERENCE]);
}

#[test]
#[ignore = "needs the local Qwen3.8-Flash-Next FP4-Local export and a device"]
fn the_brain_float_reference_rows_agree_on_the_answer() {
    replay(GOLDEN_A16, &[&FLOOR], &[&MLX_REFERENCE, &REFERENCE]);
}

/// The prompt passes with low-precision activations (NVFP4 into the routed
/// experts, MXFP8 into the dense projections), graded against the same library
/// reference: what the activation quantization costs on real prompts.
#[test]
#[ignore = "needs the local Qwen3.8-Flash-Next FP4-Local export and a device"]
fn low_precision_prompt_rows_agree_with_the_library_reference() {
    replay_as(
        GOLDEN_HF,
        &[&FLOOR],
        &[&MLX_REFERENCE, &REFERENCE],
        PrefillActivations::Lowp,
    );
}

#[test]
#[ignore = "needs the local Qwen3.8-Flash-Next FP4-Local export and a device"]
fn the_library_reference_rows_agree_on_the_answer() {
    replay(GOLDEN_HF, &[&FLOOR], &[&MLX_REFERENCE, &REFERENCE]);
}

#[test]
#[ignore = "needs the local Qwen3.8-Flash-Next FP4-Local export and a device"]
fn the_greedy_trajectory_follows_the_reference_one() {
    greedy(GREEDY_LOCK);
}

#[test]
#[ignore = "needs the local Qwen3.8-Flash-Next FP4-Local export and a device"]
fn the_greedy_trajectory_follows_the_library_reference() {
    greedy(GREEDY_LOCK_HF);
}

/// Replays one greedy lock and checks the leading ids agree.
fn greedy(lock_path: &str) {
    let Some((mut harness, vocabulary)) = open(1024) else {
        return;
    };
    let lock: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(vectors_dir().join(lock_path)).expect("the lock is present"),
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
        "fp4-local greedy lock ({lock_path}): {agreed}/{} ids agree",
        want.len()
    ));
    report(&format!(
        "fp4-local greedy text: {:?}",
        vocabulary.decode(&produced).unwrap_or_default()
    ));
    assert!(
        agreed >= GREEDY_MIN_AGREEMENT,
        "the two stacks part after {agreed} greedy ids: shiso {ids:?}; reference {want:?}"
    );
}
