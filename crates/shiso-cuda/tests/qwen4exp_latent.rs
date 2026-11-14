//! The latent pack, served: it answers a chat prompt, its prompt pass and its
//! decode step agree, and a pool that holds a twelfth of its experts says the
//! same thing as one that holds them all.
//!
//! The pack is a directory (see `qwen4exp::latent`): the skeleton container,
//! the expert records, the dense latent projections and the n-gram table,
//! beside the config, tokenizer and chat template of the base model. Three
//! claims, in one case because each load is tens of gigabytes and the machine
//! has room for one model at a time:
//!
//! 1. greedy decoding of a chat-formatted question with thinking off names
//!    Paris - the first thing a reader can read and disbelieve;
//! 2. a 600-token prompt prefilled in one pass and fed a token at a time ends
//!    on the same top-1 token, its logits within bounds taken from the same
//!    comparison on the GGUF arm. They are not bit-equal by construction (see
//!    the constants below);
//! 3. the same 48 greedy tokens from a pool of 2048 streamed slots.
//!
//! A second case, `latent_pack_speculates`, loads the pack's own draft head
//! from its `mtp/` folder and holds speculative greedy decoding (depth 4) to
//! the plain run's tokens, for two prompts, token for token.
//!
//! ```text
//! SHISO_CUDA_COMPILE=1 \
//!   SHISO_TEST_MODEL_QWEN4EXP_LATENT=/path/to/servable/directory \
//!   cargo test --release --locked -p shiso-cuda --test qwen4exp_latent -- --ignored --nocapture
//! ```
#![cfg(target_os = "linux")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use shiso_core::HfConfig;
use shiso_core::backend::Backend;
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::{DecodeStep, Model};
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_engine::sparse::capture;
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_engine::{EngineOptions, argmax};
use shiso_models::qwen4exp::{EXPERTS, N_LAYERS, Qwen4ExpModel, StreamedLoad};

/// Context the model is loaded with.
const CTX: u32 = 4_096;

/// The widest prompt pass; the 600-token prompt fits in one.
const PREFILL: usize = 1_024;

/// Greedy tokens each decode run produces.
const STEPS: usize = 48;

/// Tokens of the one-pass versus stepped prompt.
const LONG: usize = 600;

/// Slots of the streamed run: a twelfth of the pack's experts.
const STREAMED_SLOTS: u32 = 2_048;

/// Bounds on one 600-token pass against the same prompt fed a token at a
/// time, from two measurements of that very comparison (same prompt, same
/// method, `the_gguf_arm_pass_vs_steps_reference` for the first):
///
/// - the family's GGUF arm (`UD-IQ3_XXS`, abliterated Qwen3.8-Flash-Next):
///   max 0.724, mean 0.0783, top-5 overlap 4, same top-1;
/// - this pack: max 0.324, mean 0.0492, top-5 overlap 5, same top-1.
///
/// The bounds hold the pack to its own measurement with a margin: 0.5 is
/// about 1.5 times its max (0.324) and 0.08 about 1.6 times its mean (0.0492),
/// both under the GGUF arm's own (0.724, 0.0783), from which this test's
/// earlier bounds of 1.1 and 0.12 were derived. Top-5 overlap is the smaller
/// of the two arms' (4). The difference is expected where a prompt pass and a step differ in
/// the kernels they run: the pass's dense projections and routed stages take
/// activations rounded to half precision into tensor-core tiles
/// (`kernels/cuda/q4e/ggml.cu`, "Prompt passes on tensor cores"; the routed
/// intermediate is converted by `q4e_g_swiglu_half_kernel`), where the step's
/// matrix-vector kernels read single precision. That is read off the kernels'
/// documentation and source; no per-stage attribution was measured, since the
/// pack's difference is below the GGUF arm's.
const MIN_TOP5: usize = 4;
/// Mean absolute logit difference bound.
const MEAN_BOUND: f64 = 0.08;
/// Widest absolute logit difference bound.
const MAX_BOUND: f64 = 0.5;

/// `count` events in `seconds`, as a rate.
fn per_second(count: usize, seconds: f64) -> f64 {
    f64::from(u32::try_from(count).expect("a small count")) / seconds
}

/// Held for the whole of every device test in this file.
///
/// The machine's memory is one pool shared by the CPU and the GPU, and an
/// allocation is only backed when first written, so two models loaded at once
/// do not fail at allocation: they take the machine down when the second one
/// fills. The harness runs tests of one binary on parallel threads, so the
/// exclusion is the tests' own.
static ONE_MODEL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Takes [`ONE_MODEL`], whether or not a failed test poisoned it: the lock
/// guards a resource, not data.
fn one_model() -> std::sync::MutexGuard<'static, ()> {
    ONE_MODEL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stderr, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    eprintln!("SKIP: {why}");
}

/// Prints what a run measured, which is what a reader grades it by.
#[expect(clippy::print_stdout, reason = "the measurement is a test's output")]
fn report(what: &str) {
    println!("{what}");
}

/// The servable directory named by the variable, when it is one.
fn directory() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_QWEN4EXP_LATENT").ok()?);
    dir.join("config.json").is_file().then_some(dir)
}

/// Loads the pack with `slots` expert slots and no draft head.
fn load(backend: &Arc<dyn Backend>, dir: &std::path::Path, slots: u32) -> Qwen4ExpModel {
    load_speculating(backend, dir, slots, 0)
}

/// Loads the pack with `slots` expert slots and, for a `depth` above zero,
/// its draft head at that speculation depth.
fn load_speculating(
    backend: &Arc<dyn Backend>,
    dir: &std::path::Path,
    slots: u32,
    depth: u32,
) -> Qwen4ExpModel {
    let document = HfConfig::from_file(&dir.join("config.json")).expect("the config document");
    let mut engine = EngineOptions::default();
    engine.ctx = CTX;
    let entries = capture::capture(backend.as_ref());
    let opts = SparseOptions {
        engine,
        co_resident_blocks: capture::co_resident_blocks(backend.as_ref(), entries),
        entries,
        spec: SpecDepth::new(depth),
        prefill_rows: SparseOptions::prefill_rows_for(PREFILL),
        ..SparseOptions::default()
    };
    let clock = Instant::now();
    let model = Qwen4ExpModel::load_latent(
        backend,
        dir,
        &document,
        opts,
        &StreamedLoad {
            slots: Some(slots),
            reserve: 6 << 30,
            draft: depth > 0,
            counts: None,
        },
    )
    .expect("the pack loads");
    report(&format!(
        "loaded with {slots} slots in {:.1} s",
        clock.elapsed().as_secs_f64()
    ));
    model
}

/// One logits row, read back.
fn row_of(backend: &Arc<dyn Backend>, view: shiso_core::DeviceView<'_>) -> Vec<f32> {
    let mut bytes = vec![0_u8; usize::try_from(view.len()).expect("a host index")];
    backend
        .copy_d2h(view, &mut bytes)
        .expect("the row reads back");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// The tokens that end a turn: `<|im_end|>` and `<|endoftext|>`.
const END_OF_TURN: [u32; 2] = [248_046, 248_044];

/// Greedy-decodes [`STEPS`] tokens after `prompt`, or up to and including the
/// first end-of-turn token when `stop` is set.
fn generate(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    prompt: &[TokenId],
    stop: bool,
) -> (Vec<u32>, f64) {
    let session = model.open_session().expect("a session opens");
    let first = {
        let rows = model
            .prefill_last(session, prompt)
            .expect("the prompt runs");
        argmax(&row_of(backend, rows.row(0).expect("one row")))
    };
    let mut ids = vec![first.get()];
    let mut token = first;
    let clock = Instant::now();
    while ids.len() < STEPS && !(stop && END_OF_TURN.contains(&token.get())) {
        let rows = model
            .decode_batch(&[DecodeStep { session, token }])
            .expect("a step runs");
        token = argmax(&row_of(backend, rows.row(0).expect("one row")));
        ids.push(token.get());
    }
    let seconds = clock.elapsed().as_secs_f64();
    model.close_session(session).expect("the session closes");
    (ids, seconds)
}

/// The chat prompt, thinking off: the template's `enable_thinking=false` path.
fn chat(tokenizer: &tokenizers::Tokenizer) -> Vec<TokenId> {
    let prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one \
                  word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let encoded = tokenizer.encode(prompt, false).expect("the prompt encodes");
    encoded
        .get_ids()
        .iter()
        .map(|id| TokenId::new(*id))
        .collect()
}

/// [`LONG`] tokens of ordinary text.
fn long_prompt(tokenizer: &tokenizers::Tokenizer) -> Vec<TokenId> {
    let text = "The river wound through the valley past orchards and old stone mills, \
                and the farmers who lived beside it kept careful records of every flood. "
        .repeat(40);
    let encoded = tokenizer.encode(text, false).expect("the text encodes");
    let ids: Vec<TokenId> = encoded
        .get_ids()
        .iter()
        .take(LONG)
        .map(|id| TokenId::new(*id))
        .collect();
    assert_eq!(ids.len(), LONG, "the text is long enough");
    ids
}

/// How far two logits rows over one vocabulary are apart.
#[derive(Debug)]
struct Diff {
    /// The largest absolute difference.
    max: f64,
    /// The mean absolute difference.
    mean: f64,
    /// How many of the first row's top five are among the second's.
    top5: usize,
    /// Whether the two rows pick the same token.
    top1: bool,
}

/// The ids of the five largest entries.
fn top5(row: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
    order.truncate(5);
    order
}

/// Compares two rows.
fn diff(whole: &[f32], steps: &[f32]) -> Diff {
    assert_eq!(whole.len(), steps.len());
    let gaps: Vec<f64> = whole
        .iter()
        .zip(steps)
        .map(|(a, b)| f64::from((a - b).abs()))
        .collect();
    let (a, b) = (top5(whole), top5(steps));
    Diff {
        max: gaps.iter().copied().fold(0.0, f64::max),
        mean: gaps.iter().sum::<f64>()
            / f64::from(u32::try_from(gaps.len()).expect("a vocabulary")),
        top5: a.iter().filter(|id| b.contains(id)).count(),
        top1: a[0] == b[0],
    }
}

/// The 600-token prompt in one pass against the same prompt a token at a
/// time, on a fresh session each.
fn pass_vs_steps(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    long: &[TokenId],
    label: &str,
) -> Diff {
    let session = model.open_session().expect("a session opens");
    let clock = Instant::now();
    let whole = {
        let rows = model.prefill_last(session, long).expect("the pass runs");
        row_of(backend, rows.row(0).expect("one row"))
    };
    let pass = clock.elapsed().as_secs_f64();
    report(&format!(
        "{label} prefill: {LONG} tokens in one pass, {pass:.2} s, {:.1} tok/s",
        per_second(LONG, pass)
    ));
    model.close_session(session).expect("the session closes");
    let session = model.open_session().expect("a session opens");
    let steps = stepped(backend, model, session, long);
    model.close_session(session).expect("the session closes");
    let found = diff(&whole, &steps);
    report(&format!("{label} one pass vs steps: {found:?}"));
    found
}

/// Feeds `tokens` a token at a time and returns the row after the last.
fn stepped(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    session: SessionId,
    tokens: &[TokenId],
) -> Vec<f32> {
    let mut row = Vec::new();
    for token in tokens {
        let rows = model
            .decode_batch(&[DecodeStep {
                session,
                token: *token,
            }])
            .expect("a step runs");
        row = row_of(backend, rows.row(0).expect("one row"));
    }
    row
}

#[test]
#[ignore = "needs a CUDA device and the latent pack's servable directory"]
fn latent_pack_generates() {
    let _alone = one_model();
    let Some(dir) = directory() else {
        skip("no servable directory named by SHISO_TEST_MODEL_QWEN4EXP_LATENT");
        return;
    };
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("the vocabulary");
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");

    // (a) and (b), every expert resident.
    let mut model = load(&backend, &dir, EXPERTS * N_LAYERS);
    let prompt = chat(&tokenizer);
    let (turn, _) = generate(&backend, &mut model, &prompt, true);
    let answer = tokenizer.decode(&turn, false).expect("the answer decodes");
    report(&format!("greedy to end of turn: {answer:?} {turn:?}"));
    assert!(
        turn.last().is_some_and(|id| END_OF_TURN.contains(id)),
        "the answer ends its turn within {STEPS} tokens: {turn:?}"
    );
    let first_end = turn
        .iter()
        .position(|id| END_OF_TURN.contains(id))
        .expect("checked above");
    let before = tokenizer
        .decode(&turn[..first_end], false)
        .expect("the answer decodes");
    assert!(
        before.contains("Paris"),
        "Paris before the turn ends: {before:?}"
    );
    let (ids, seconds) = generate(&backend, &mut model, &prompt, false);
    report(&format!(
        "greedy {STEPS} past the turn: {:?}",
        tokenizer.decode(&ids, false).expect("decodes")
    ));
    report(&format!(
        "decode: {:.2} tok/s over {} steps",
        per_second(STEPS - 1, seconds),
        STEPS - 1
    ));

    let long = long_prompt(&tokenizer);
    let found = pass_vs_steps(&backend, &mut model, &long, "latent pack");
    assert!(found.top1, "the same top-1 token: {found:?}");
    assert!(found.top5 >= MIN_TOP5, "top-5 overlap: {found:?}");
    assert!(found.mean <= MEAN_BOUND, "mean difference: {found:?}");
    assert!(found.max <= MAX_BOUND, "widest difference: {found:?}");
    model.teardown().expect("the model tears down");
    drop(model);

    // (c) the same answer from a streamed pool.
    let mut model = load(&backend, &dir, STREAMED_SLOTS);
    let (again, seconds) = generate(&backend, &mut model, &prompt, false);
    report(&format!(
        "streamed {STREAMED_SLOTS} slots: {:.2} tok/s",
        per_second(STEPS - 1, seconds)
    ));
    assert_eq!(
        again, ids,
        "the streamed pool says what the resident one did"
    );
    model.teardown().expect("the model tears down");
}

/// The reference measurement the bounds above come from: the same comparison
/// on the family's GGUF checkpoint, whose dense projections run the same
/// `Q8_0` and prompt kernels. `SHISO_TEST_MODEL_QWEN4EXP` names its first
/// shard; the tokenizer is the one beside it or in its parent.
#[test]
#[ignore = "needs a CUDA device and a GGUF checkpoint of this family"]
fn the_gguf_arm_pass_vs_steps_reference() {
    let _alone = one_model();
    let Ok(path) = std::env::var("SHISO_TEST_MODEL_QWEN4EXP") else {
        skip("no checkpoint named by SHISO_TEST_MODEL_QWEN4EXP");
        return;
    };
    let path = PathBuf::from(path);
    let Some(tokenizer) = path
        .ancestors()
        .skip(1)
        .take(2)
        .map(|d| d.join("tokenizer.json"))
        .find(|t| t.is_file())
    else {
        skip("no tokenizer.json beside the checkpoint");
        return;
    };
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer).expect("the vocabulary");
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let file = shiso_modelfile::GgufFile::open(&path).expect("the checkpoint opens");
    let config = shiso_models::qwen4exp::Qwen4ExpConfig::from_metadata(file.metadata())
        .expect("the metadata agrees");
    drop(file);
    let mut engine = EngineOptions::default();
    engine.ctx = CTX;
    let entries = capture::capture(backend.as_ref());
    let opts = SparseOptions {
        engine,
        co_resident_blocks: capture::co_resident_blocks(backend.as_ref(), entries),
        entries,
        prefill_rows: SparseOptions::prefill_rows_for(PREFILL),
        ..SparseOptions::default()
    };
    let mut model = Qwen4ExpModel::load_with(&backend, &path, None, None, &config, opts)
        .expect("the checkpoint loads");
    let long = long_prompt(&tokenizer);
    let found = pass_vs_steps(&backend, &mut model, &long, "gguf arm");
    assert!(found.top1, "the same top-1 token: {found:?}");
    model.teardown().expect("the model tears down");
}

/// Tokens each speculation prompt may produce.
const SPEC_TOKENS: usize = 256;

/// The speculation depth the server's flag `--spec-k 4` asks for.
const SPEC_DEPTH: u32 = 4;

/// A chat prompt with thinking off, rendered as the template renders it.
fn user_turn(tokenizer: &tokenizers::Tokenizer, question: &str) -> Vec<TokenId> {
    let prompt = format!(
        "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let encoded = tokenizer.encode(prompt, false).expect("the prompt encodes");
    encoded
        .get_ids()
        .iter()
        .map(|id| TokenId::new(*id))
        .collect()
}

/// The ids up to and including the first end-of-turn token, and no more than
/// [`SPEC_TOKENS`] of them.
fn turn_of(ids: &[u32]) -> Vec<u32> {
    let end = ids
        .iter()
        .position(|id| END_OF_TURN.contains(id))
        .map_or(ids.len(), |at| at + 1);
    ids[..end.min(SPEC_TOKENS)].to_vec()
}

/// Greedy plain decode of up to [`SPEC_TOKENS`] tokens or the end of the turn,
/// and its decode rate.
fn plain_turn(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    prompt: &[TokenId],
) -> (Vec<u32>, f64) {
    let session = model.open_session().expect("a session opens");
    let first = {
        let rows = model
            .prefill_last(session, prompt)
            .expect("the prompt runs");
        argmax(&row_of(backend, rows.row(0).expect("one row")))
    };
    let mut ids = vec![first.get()];
    let mut token = first;
    let clock = Instant::now();
    while ids.len() < SPEC_TOKENS && !END_OF_TURN.contains(&token.get()) {
        let rows = model
            .decode_batch(&[DecodeStep { session, token }])
            .expect("a step runs");
        token = argmax(&row_of(backend, rows.row(0).expect("one row")));
        ids.push(token.get());
    }
    let seconds = clock.elapsed().as_secs_f64();
    model.close_session(session).expect("the session closes");
    let steps = ids.len() - 1;
    (ids, per_second(steps, seconds))
}

/// What a speculative run did: its tokens, its decode rate, and how many of
/// the head's guesses the target kept.
struct Speculative {
    /// The run's tokens, the first from the prompt pass.
    ids: Vec<u32>,
    /// Tokens a second after the prompt pass.
    rate: f64,
    /// Guesses the target agreed with.
    accepted: u32,
    /// Guesses the head made.
    drafted: u32,
}

/// Greedy speculative decode of the same turn.
fn speculative_turn(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    prompt: &[TokenId],
) -> Speculative {
    let eos = model.info().stop_tokens.primary();
    let session = model.open_session().expect("a session opens");
    let first = {
        let rows = model
            .prefill_last(session, prompt)
            .expect("the prompt runs");
        argmax(&row_of(backend, rows.row(0).expect("one row")))
    };
    let last = u32::try_from(prompt.len() - 1).expect("a position fits");
    model
        .stack_mut()
        .stage_draft_row(session, last)
        .expect("the prompt's last row stages");
    let mut ids = vec![first.get()];
    let (mut accepted, mut drafted) = (0, 0);
    let clock = Instant::now();
    while ids.len() < SPEC_TOKENS && !ids.iter().any(|id| END_OF_TURN.contains(id)) {
        let token = TokenId::new(*ids.last().expect("one token at least"));
        let step = model
            .stack_mut()
            .speculate(session, token, eos)
            .expect("a speculative step runs");
        accepted += step.accepted;
        drafted += step.drafted;
        ids.extend(step.produced.iter().map(|t| t.get()));
    }
    let seconds = clock.elapsed().as_secs_f64();
    model.close_session(session).expect("the session closes");
    let produced = ids.len() - 1;
    Speculative {
        ids,
        rate: per_second(produced, seconds),
        accepted,
        drafted,
    }
}

/// The pack's own draft head, loaded from `mtp/`: greedy speculation at depth
/// [`SPEC_DEPTH`] emits exactly the plain greedy tokens, for an English
/// question and for a request for code, thinking off, and the head's
/// acceptance rate is reported. Below about 30 % the head is mis-bound.
#[test]
#[ignore = "needs a CUDA device and the latent pack's servable directory"]
fn latent_pack_speculates() {
    let _alone = one_model();
    let Some(dir) = directory() else {
        skip("no servable directory named by SHISO_TEST_MODEL_QWEN4EXP_LATENT");
        return;
    };
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("the vocabulary");
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let mut model = load_speculating(&backend, &dir, EXPERTS * N_LAYERS, SPEC_DEPTH);
    let questions = [
        (
            "prose",
            "Why is the sky blue during the day but red at sunset? Explain it to a curious child.",
        ),
        (
            "code",
            "Write a Python function that returns the n-th Fibonacci number iteratively.",
        ),
    ];
    for (label, question) in questions {
        let prompt = user_turn(&tokenizer, question);
        let (plain, plain_rate) = plain_turn(&backend, &mut model, &prompt);
        let spec = speculative_turn(&backend, &mut model, &prompt);
        let (want, got) = (turn_of(&plain), turn_of(&spec.ids));
        let rate = f64::from(spec.accepted) / f64::from(spec.drafted.max(1));
        report(&format!(
            "{label}: {} tokens; plain {plain_rate:.2} tok/s, speculative depth {SPEC_DEPTH} {:.2} tok/s; \
             drafts accepted {}/{} = {:.1} %; equal: {}",
            want.len(),
            spec.rate,
            spec.accepted,
            spec.drafted,
            rate * 100.0,
            want == got
        ));
        report(&format!(
            "{label}: {:?}",
            tokenizer.decode(&want, false).expect("decodes")
        ));
        assert_eq!(got, want, "{label}: speculation changed the greedy text");
        assert!(spec.drafted > 0, "{label}: the head drafted nothing");
        assert!(
            rate >= 0.30,
            "{label}: acceptance {:.1} % is below 30 %, the head is mis-bound",
            rate * 100.0
        );
    }
    model.teardown().expect("the model tears down");
}
