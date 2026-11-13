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
//!    on the same top-1 token, its logits within a tolerance. They are not
//!    bit-equal by construction: the pass runs the weight-stationary matrix
//!    kernels and the step the matrix-vector ones, which sum in different
//!    orders;
//! 3. the same 48 greedy tokens from a pool of 2048 streamed slots.
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

/// The widest a logit may differ between the prompt pass and the steps.
///
/// The two sum the same products in different orders over eight latent groups
/// and 48 layers, and the routers can flip an expert at a near tie, so the
/// bound is on the logit scale (these rows span tens of units) rather than
/// the last bit.
const LOGIT_TOLERANCE: f32 = 0.5;

/// `count` events in `seconds`, as a rate.
fn per_second(count: usize, seconds: f64) -> f64 {
    f64::from(u32::try_from(count).expect("a small count")) / seconds
}

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
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
    let document = HfConfig::from_file(&dir.join("config.json")).expect("the config document");
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
    let clock = Instant::now();
    let model = Qwen4ExpModel::load_latent(
        backend,
        dir,
        &document,
        opts,
        &StreamedLoad {
            slots: Some(slots),
            reserve: 6 << 30,
            draft: false,
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

/// Greedy-decodes [`STEPS`] tokens after `prompt`.
fn generate(
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
    while ids.len() < STEPS {
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
    let (ids, seconds) = generate(&backend, &mut model, &prompt);
    let text = tokenizer.decode(&ids, false).expect("the answer decodes");
    report(&format!("greedy {STEPS}: {text:?}"));
    report(&format!("ids: {ids:?}"));
    report(&format!(
        "decode: {:.2} tok/s over {} steps",
        per_second(STEPS - 1, seconds),
        STEPS - 1
    ));
    assert!(text.contains("Paris"), "the answer names Paris: {text:?}");

    let long = long_prompt(&tokenizer);
    let session = model.open_session().expect("a session opens");
    let clock = Instant::now();
    let whole = {
        let rows = model.prefill_last(session, &long).expect("the pass runs");
        row_of(&backend, rows.row(0).expect("one row"))
    };
    let pass = clock.elapsed().as_secs_f64();
    report(&format!(
        "prefill: {LONG} tokens in one pass, {pass:.2} s, {:.1} tok/s",
        per_second(LONG, pass)
    ));
    model.close_session(session).expect("the session closes");
    let session = model.open_session().expect("a session opens");
    let steps = stepped(&backend, &mut model, session, &long);
    model.close_session(session).expect("the session closes");
    let (top_whole, top_steps) = (argmax(&whole), argmax(&steps));
    let widest = whole
        .iter()
        .zip(&steps)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    report(&format!(
        "one pass vs steps: top-1 {} vs {}, logit {:.4} vs {:.4}, widest difference {widest:.4}",
        top_whole.get(),
        top_steps.get(),
        whole[top_whole.get() as usize],
        steps[top_steps.get() as usize],
    ));
    assert_eq!(top_whole, top_steps, "the same top-1 token");
    assert!(
        widest <= LOGIT_TOLERANCE,
        "logits differ by {widest}, tolerance {LOGIT_TOLERANCE}"
    );
    model.teardown().expect("the model tears down");
    drop(model);

    // (c) the same answer from a streamed pool.
    let mut model = load(&backend, &dir, STREAMED_SLOTS);
    let (again, seconds) = generate(&backend, &mut model, &prompt);
    report(&format!(
        "streamed {STREAMED_SLOTS} slots: {:.2} tok/s, {:?}",
        per_second(STEPS - 1, seconds),
        tokenizer.decode(&again, false).expect("the answer decodes")
    ));
    assert_eq!(
        again, ids,
        "the streamed pool says what the resident one did"
    );
    model.teardown().expect("the model tears down");
}
