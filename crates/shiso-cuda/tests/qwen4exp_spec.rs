//! A verify pass must compute each of its rows exactly as a decode step would.
//!
//! Speculative greedy decoding promises the plain run's text byte for byte,
//! and it can only keep that promise if the target's multi-row pass and its
//! single-row step agree to the bit on every row: a verify that is merely
//! close answers the same until some later pick is a near tie, and then
//! leaves the plain run's text for good. Nothing fails when that happens, so
//! the comparison here is of logits, bitwise, rather than of tokens.
//!
//! The case sits where the two passes used to part: the indexer budget. A pass
//! decided whether its attention reads a selection once, from its last row, so
//! a verify that started a row or two short of the budget attended those rows
//! with the selected-block kernel while the decode steps it stands in for
//! attended them densely.
//!
//! It runs here rather than in `shiso-models` for the reason `qwen35_spec.rs`
//! gives: that crate names no device, so a case written there skips on the one
//! machine that could run it.
//!
//! ```text
//! SHISO_CUDA_COMPILE=1 \
//!   SHISO_TEST_MODEL_QWEN4EXP=/path/to/checkpoint.gguf \
//!   SHISO_TEST_MODEL_QWEN4EXP_DRAFT=/path/to/draft-head.gguf \
//!   cargo test --release -p shiso-cuda --test qwen4exp_spec -- --ignored --nocapture
//! ```
//!
//! The draft head is optional: without it the logits comparison still runs,
//! and only the end-to-end speculative run is skipped.
#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::{DecodeStep, Model};
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_engine::EngineOptions;
use shiso_engine::sparse::capture;
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_modelfile::GgufFile;
use shiso_models::qwen4exp::{Qwen4ExpConfig, Qwen4ExpModel};

/// Context the model is loaded with: room for the budget and a little past it.
const CTX: u32 = 4096;

/// Rows of the verify-shaped chunk: the committed token and three guesses,
/// which is the depth the server speculates at.
const WIDTH: u32 = 4;

/// Greedy tokens each run of the end-to-end comparison produces past the
/// prompt, which straddles the budget.
const STEPS: usize = 48;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// A path named by `var`, when it names a file.
fn file_from(var: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(var).ok()?);
    path.is_file().then_some(path)
}

/// `count` token ids from a fixed pseudo-random walk over ordinary text ids.
///
/// Synthetic rather than a rendered prompt so the case needs no vocabulary; the
/// ids stay clear of the special tokens at the top of the table.
fn prompt(count: u32) -> Vec<TokenId> {
    let mut state = 0x2545_f491_u32;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            TokenId::new(1_000 + (state >> 8) % 60_000)
        })
        .collect()
}

/// One logits row, as floats.
fn row_of(backend: &Arc<dyn Backend>, logits: &shiso_core::Logits<'_>, row: u32) -> Vec<f32> {
    let view = logits.row(row).expect("the row exists");
    let mut bytes = vec![0u8; usize::try_from(view.len()).expect("a row fits a host index")];
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

/// The highest-scoring id of one row.
fn greedy(row: &[f32]) -> TokenId {
    shiso_engine::argmax(row)
}

/// Loads the checkpoint, with the draft head when one is named.
fn load(backend: &Arc<dyn Backend>, path: &Path, draft: Option<&Path>) -> Qwen4ExpModel {
    let file = GgufFile::open(path).expect("the checkpoint opens");
    let config = Qwen4ExpConfig::from_metadata(file.metadata()).expect("the metadata agrees");
    drop(file);
    let mut engine = EngineOptions::default();
    engine.ctx = CTX;
    let entries = capture::capture(backend.as_ref());
    let opts = SparseOptions {
        engine,
        spec: SpecDepth::new(if draft.is_some() { WIDTH - 1 } else { 0 }),
        co_resident_blocks: capture::co_resident_blocks(backend.as_ref(), entries),
        entries,
        ..SparseOptions::default()
    };
    Qwen4ExpModel::load_with(backend, path, draft, None, &config, opts)
        .expect("the checkpoint loads")
}

/// A prompt length that puts a verify-shaped chunk two rows short of the
/// budget: its first two rows' own steps attend densely, its last two over a
/// selection.
fn crossing_prompt_len(model: &Qwen4ExpModel) -> u32 {
    let indexer = model.stack().shape().indexer;
    // The first frontier whose single-row step selects.
    let first = (0..CTX)
        .find(|&frontier| indexer.sparse_at(frontier + 1))
        .expect("the budget sits inside the context");
    first - 2
}

/// Plain greedy decode of `steps` tokens after `prompt`, keeping every row.
fn decode_rows(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    prompt: &[TokenId],
    steps: usize,
) -> (Vec<TokenId>, Vec<Vec<f32>>) {
    let session = model.open_session().expect("a session opens");
    let first = row_of(
        backend,
        &model.prefill_last(session, prompt).expect("prefill"),
        0,
    );
    let mut ids = vec![greedy(&first)];
    let mut rows = Vec::with_capacity(steps);
    for at in 0..steps {
        let step = DecodeStep {
            session,
            token: ids[at],
        };
        let row = row_of(
            backend,
            &model.decode_batch(&[step]).expect("a decode step"),
            0,
        );
        ids.push(greedy(&row));
        rows.push(row);
    }
    model.close_session(session).expect("the session closes");
    (ids, rows)
}

/// A chunk of verify width that crosses the indexer budget produces, at its
/// last row, the very logits the decode steps over the same tokens produce.
///
/// Its last row reads the key and value rows the chunk's earlier rows wrote in
/// every attention layer, so it is exact only if those rows were each computed
/// as their own decode step would compute them - which is the property a
/// verify pass needs for every row.
#[test]
#[ignore = "needs a CUDA device and a checkpoint of this family"]
fn a_verify_width_chunk_across_the_indexer_budget_matches_decode_to_the_bit() {
    let Some(path) = file_from("SHISO_TEST_MODEL_QWEN4EXP") else {
        skip("no checkpoint named by SHISO_TEST_MODEL_QWEN4EXP");
        return;
    };
    let draft = file_from("SHISO_TEST_MODEL_QWEN4EXP_DRAFT");
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let mut model = load(&backend, &path, draft.as_deref());
    let prompt = prompt(crossing_prompt_len(&model));

    let (ids, rows) = decode_rows(&backend, &mut model, &prompt, WIDTH as usize);

    let session = model.open_session().expect("a session opens");
    let first = row_of(
        &backend,
        &model.prefill_last(session, &prompt).expect("prefill"),
        0,
    );
    assert_eq!(greedy(&first), ids[0], "the two prompts agree");
    let chunk = row_of(
        &backend,
        &model
            .prefill_last(session, &ids[..WIDTH as usize])
            .expect("the crossing chunk runs"),
        0,
    );
    model.close_session(session).expect("the session closes");
    let want = &rows[WIDTH as usize - 1];
    let worst = want
        .iter()
        .zip(&chunk)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        want.iter()
            .zip(&chunk)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the chunk's last row differs from the decode step's by up to {worst:e}"
    );

    if let Some(_head) = draft {
        speculative_matches_plain(&backend, &mut model, &prompt);
    } else {
        skip("no draft head named by SHISO_TEST_MODEL_QWEN4EXP_DRAFT; the speculative run");
    }
    model.teardown().expect("the model tears down");
}

/// The end-to-end form of the same claim: a greedy speculative run across the
/// budget emits exactly the plain run's tokens.
fn speculative_matches_plain(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    prompt: &[TokenId],
) {
    let (plain, _) = decode_rows(backend, model, prompt, STEPS);
    let eos = model.info().stop_tokens.primary();
    let session = model.open_session().expect("a session opens");
    let first = row_of(
        backend,
        &model.prefill_last(session, prompt).expect("prefill"),
        0,
    );
    let last = u32::try_from(prompt.len() - 1).expect("a position fits");
    model
        .stack_mut()
        .stage_draft_row(session, last)
        .expect("the prompt's row stages");
    let mut ids = vec![greedy(&first)];
    while ids.len() <= STEPS {
        let token = *ids.last().expect("one token at least");
        let step = model
            .stack_mut()
            .speculate(session, token, eos)
            .expect("a speculative step runs");
        ids.extend_from_slice(&step.produced);
    }
    model.close_session(session).expect("the session closes");
    ids.truncate(plain.len());
    assert_eq!(ids, plain, "speculation changed the greedy text");
}
