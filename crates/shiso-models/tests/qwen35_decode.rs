//! The model tests: a greedy decode that has to keep answering the same way, and
//! the resident-set ceiling the run has to stay under.
//!
//! Every case here needs a checkpoint and a device, so every case is `#[ignore]`d
//! and skips with a printed note when either is missing. Nothing downloads
//! anything: the paths are local, overridable by `SHISO_TEST_MODEL` and
//! `SHISO_TEST_MODEL_OPTIQ`.
//!
//! The resident-set ceiling is measured where the binary it measures lives, in
//! `shiso-bench/tests/rss.rs`: `CARGO_BIN_EXE_*` names only the binaries of the
//! crate the test belongs to, and a test that guessed at a path under `target/`
//! would break on the first profile change.
//!
//! # Why a recorded sequence is a real assertion
//!
//! Greedy sampling over a bit-exact sampler is deterministic, so the same build
//! against the same checkpoint produces the same 64 ids every time. The sequence
//! was recorded once, with its decoded text read by a human and found to be
//! coherent English; from then on it proves that nothing in the kernel stack has
//! moved. It is not a claim about what the model *should* say - it is a claim that
//! today's answer is the answer that was checked.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::HfConfig;
use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::{DecodeStep, Model};
use shiso_engine::{EngineOptions, argmax};
use shiso_metal::{MetalBackend, metal_available};
use shiso_models::qwen35::{Qwen35Config, Qwen35Model};

/// The prompt the golden sequence continues.
const PROMPT: &str = "The capital of France is";

/// Ids the golden sequence records.
const GOLDEN_STEPS: usize = 64;

/// The id that ends a turn, which must not appear inside the recorded span.
const EOS: u32 = 248_044;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Reports what a run produced, for the eye that records a golden file.
#[expect(
    clippy::print_stdout,
    reason = "the recorded sequence is a test's output"
)]
fn report(what: &str) {
    println!("{what}");
}

/// The checkpoint the model tests run against.
fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL").ok()?);
    path.is_file().then_some(path)
}

/// The vocabulary the prompt is encoded with.
///
/// Read from the sibling model directory rather than committed beside the test:
/// the file is twenty megabytes, which is not a thing to carry in a source tree
/// for one prompt. A machine without it skips, which is the same outcome as a
/// machine without the checkpoint.
fn tokenizer_path() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_OPTIQ").ok()?);
    let path = dir.join("tokenizer.json");
    path.is_file().then_some(path)
}

/// Where the recorded sequence lives.
fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/goldens/qwen35-greedy-64.json")
}

/// The shape profile the tests load against.
fn profile() -> Qwen35Config {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/qwen35-config.json");
    Qwen35Config::from_hf(&HfConfig::from_file(&path).expect("the recorded config parses"))
        .expect("the recorded config is this family's")
}

/// The highest-scoring id of one logits row.
fn greedy(row: &[u8], scratch: &mut [f32]) -> TokenId {
    for (slot, bytes) in scratch.iter_mut().zip(row.as_chunks::<4>().0) {
        *slot = f32::from_le_bytes(*bytes);
    }
    argmax(scratch)
}

/// Runs the prompt and `GOLDEN_STEPS` greedy steps, returning the ids produced.
fn decode_greedy(prompt: &[TokenId]) -> Vec<TokenId> {
    let path = model_path().expect("caller checked the checkpoint is present");
    let profile = profile();
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("a Metal device");
    let mut opts = EngineOptions::default();
    opts.ctx = 1024;
    opts.kv_pool_pages = 32;
    let mut model =
        Qwen35Model::load(&backend, &path, &profile, opts).expect("the checkpoint loads");

    let vocab = usize::try_from(profile.shape().vocab).expect("a vocabulary fits a host index");
    let mut row = vec![0u8; vocab * 4];
    let mut scratch = vec![0f32; vocab];

    let session = model.open_session().expect("a session opens");
    let last = model
        .prefill_last(session, prompt)
        .expect("the prompt runs");
    backend
        .copy_d2h(last.row(0).expect("the prompt's row"), &mut row)
        .expect("the prompt's row reads back");

    let mut produced = Vec::with_capacity(GOLDEN_STEPS);
    let mut token = greedy(&row, &mut scratch);
    produced.push(token);
    for _ in 1..GOLDEN_STEPS {
        let out = model
            .decode_batch(&[DecodeStep { session, token }])
            .expect("a decode step runs");
        backend
            .copy_d2h(out.row(0).expect("the step's row"), &mut row)
            .expect("the step's row reads back");
        token = greedy(&row, &mut scratch);
        produced.push(token);
    }
    model.close_session(session).expect("the session closes");
    model.teardown().expect("the model tears down");
    produced
}

#[test]
#[ignore = "needs the local checkpoint and a Metal device"]
fn a_greedy_continuation_is_the_one_that_was_recorded() {
    let Some(_model) = model_path() else {
        skip("no local checkpoint");
        return;
    };
    let Some(vocabulary) = tokenizer_path() else {
        skip("no local tokenizer beside the checkpoint");
        return;
    };
    if !metal_available() {
        skip("no Metal device");
        return;
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&vocabulary).expect("the vocabulary loads");
    let encoded = tokenizer.encode(PROMPT, false).expect("the prompt encodes");
    let prompt: Vec<TokenId> = encoded
        .get_ids()
        .iter()
        .copied()
        .map(TokenId::new)
        .collect();

    let produced = decode_greedy(&prompt);
    let ids: Vec<u32> = produced.iter().map(|t| t.get()).collect();
    let text = tokenizer.decode(&ids, true).expect("the answer decodes");
    report(&format!("greedy continuation: {text:?}"));

    assert!(
        !ids.contains(&EOS),
        "the recorded span stops before the turn ends, so an end-of-turn id inside it \
         means the run diverged: {ids:?}"
    );

    let path = golden_path();
    let Ok(recorded) = std::fs::read_to_string(&path) else {
        // Recording, not asserting: the first run on a machine with the checkpoint
        // writes the file, and its decoded text is what a human checks before it is
        // committed. Every run after this one asserts against it. A recorded file
        // is therefore a claim that somebody read the text and found it coherent -
        // never commit one that was written by a run nobody looked at.
        let record = serde_json::json!({
            "prompt": PROMPT,
            "text": text,
            "tokens": ids,
        });
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&record).expect("the record serializes"),
        )
        .expect("the record writes");
        report(&format!("RECORDED {}", path.display()));
        return;
    };
    let golden: serde_json::Value =
        serde_json::from_str(&recorded).expect("the recorded sequence parses");
    let want: Vec<u32> = golden["tokens"]
        .as_array()
        .expect("the record carries a token array")
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("a token is a number")).expect("a token is u32"))
        .collect();
    assert_eq!(
        ids, want,
        "the greedy continuation moved; either a kernel changed or the checkpoint did"
    );
}
