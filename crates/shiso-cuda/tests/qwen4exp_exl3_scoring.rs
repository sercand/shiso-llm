//! The served EXL3 export under a scoring session.
//!
//! A scoring session (`kv_set_scoring`, `kv_set_hints`) must read a prompt
//! exactly as a plain one does: it changes where the stack may stop and
//! resume, not what a pass computes. This file is also the harness the
//! scoring-prefill plan times and profiles, because it loads the export the
//! server serves rather than a GGUF.
//!
//! ```text
//! SHISO_CUDA_COMPILE=1 \
//!   SHISO_TEST_MODEL_QWEN4EXP_EXL3=$HOME/.shiso/models/qwen3.8-flash/exl3-405/base \
//!   cargo test --release -p shiso-cuda --test qwen4exp_exl3_scoring -- --ignored --nocapture
//! ```
#![cfg(target_os = "linux")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use shiso_core::backend::Backend;
use shiso_core::config::HfConfig;
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::Model;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_engine::EngineOptions;
use shiso_engine::ckpt::Checkpointed as _;
use shiso_engine::sparse::capture;
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_models::qwen4exp::Qwen4ExpModel;

/// Context the model is loaded with.
const CTX: u32 = 4_096;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// The served trellis export, with its own draft head when `draft` is set.
///
/// The path is a directory, so it is read from the variable directly rather
/// than through `file_from`, which accepts only files.
fn load_served(backend: &Arc<dyn Backend>, draft: bool) -> Option<Qwen4ExpModel> {
    let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_QWEN4EXP_EXL3").ok()?);
    if !dir.is_dir() {
        return None;
    }
    let hf = HfConfig::from_file(&dir.join("config.json")).expect("the config reads");
    let mut engine = EngineOptions::default();
    engine.ctx = CTX;
    let entries = capture::capture(backend.as_ref());
    let opts = SparseOptions {
        engine,
        spec: SpecDepth::new(if draft { 3 } else { 0 }),
        co_resident_blocks: capture::co_resident_blocks(backend.as_ref(), entries),
        entries,
        prefill_rows: SparseOptions::prefill_rows_for(4_096),
        ..SparseOptions::default()
    };
    Some(
        Qwen4ExpModel::load_exl3(
            backend,
            &dir,
            &dir.join("ngram_embedding.safetensors"),
            &hf,
            draft,
            None,
            opts,
        )
        .expect("the export loads"),
    )
}

/// `count` token ids from a fixed pseudo-random walk over ordinary text ids.
fn prompt(count: u32) -> Vec<TokenId> {
    let mut state = 0x2545_f491_u32;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            TokenId::new(1_000 + (state >> 8) % 60_000)
        })
        .collect()
}

/// The last row a session leaves after reading `tokens` in one prefill.
fn last_row(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    session: SessionId,
    tokens: &[TokenId],
) -> Vec<u8> {
    let rows = model
        .prefill_last(session, tokens)
        .expect("the prompt runs");
    let view = rows.row(0).expect("one row");
    let mut bytes = vec![0_u8; usize::try_from(view.len()).expect("a host index")];
    backend
        .copy_d2h(view, &mut bytes)
        .expect("the row reads back");
    bytes
}

/// The last row a session leaves after reading `tokens[..at]` as a chunk and
/// the rest as the final piece: the shape a sibling question's read has after
/// resuming at the shared boundary, without a checkpoint tier to resume from.
fn split_row(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    session: SessionId,
    tokens: &[TokenId],
    at: usize,
) -> Vec<u8> {
    model
        .prefill_chunk(session, &tokens[..at])
        .expect("the shared part runs");
    assert_eq!(
        model.stack().session(session).expect("open").written(),
        u32::try_from(at).expect("a position"),
        "the tail must start where the chunk stopped"
    );
    last_row(backend, model, session, &tokens[at..])
}

#[test]
#[ignore = "needs a CUDA device and the served qwen4exp EXL3 export"]
fn a_scoring_session_reads_the_row_a_plain_one_reads() {
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let Some(mut model) = load_served(&backend, true) else {
        skip("no export named by SHISO_TEST_MODEL_QWEN4EXP_EXL3");
        return;
    };
    let tokens = prompt(1_200);
    let plain = model.open_session().expect("a session");
    let want = last_row(&backend, &mut model, plain, &tokens);
    model.close_session(plain).expect("closes");
    let scoring = model.open_session().expect("a session");
    model.kv_set_scoring(scoring).expect("marked");
    model.kv_set_hints(scoring, &[1_024]).expect("hinted");
    let got = last_row(&backend, &mut model, scoring, &tokens);
    model.close_session(scoring).expect("closes");
    assert!(want == got, "marking a session as scoring moved its row");
    model.teardown().expect("the model tears down");
}

#[test]
#[ignore = "needs a CUDA device and the served qwen4exp EXL3 export"]
fn a_scoring_tail_answers_the_same_row_with_or_without_the_draft_warm_up() {
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let Some(mut model) = load_served(&backend, true) else {
        skip("no export named by SHISO_TEST_MODEL_QWEN4EXP_EXL3");
        return;
    };
    let tokens = prompt(1_200);
    // Both sessions read the prompt as a chunk to the boundary and a 176-row
    // tail (still a prompt-role pass), so the scoring tail is a separate piece
    // that starts at the boundary, where its draft warm-up is skipped. A plain
    // session warms the draft head over every row.
    let plain = model.open_session().expect("a session");
    let want = split_row(&backend, &mut model, plain, &tokens, 1_024);
    model.close_session(plain).expect("closes");
    // A scoring session with its shared boundary at 1024 warms only below it.
    let scoring = model.open_session().expect("a session");
    model.kv_set_scoring(scoring).expect("marked");
    model.kv_set_hints(scoring, &[1_024]).expect("hinted");
    assert_eq!(
        model.stack().session(scoring).expect("open").warm_until(),
        1_024
    );
    let got = split_row(&backend, &mut model, scoring, &tokens, 1_024);
    model.close_session(scoring).expect("closes");
    assert!(want == got, "the draft head never writes the target's rows");
    model.teardown().expect("the model tears down");
}

/// Times plain prefills of a short prompt, for the profile of the pass a
/// scoring request's questions run.
///
/// `SCORING_PROFILE_TOKENS` sets the prompt width (34 by default) and
/// `SCORING_PROFILE_PASSES` the timed passes (5); `SCORING_PROFILE_DRAFT=0`
/// loads the export without its draft head; one untimed pass warms the
/// kernels first.
#[test]
#[ignore = "needs a CUDA device and the served qwen4exp EXL3 export"]
#[expect(clippy::print_stdout, reason = "the timings are the test's output")]
fn time_a_short_prefill() {
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let draft = std::env::var("SCORING_PROFILE_DRAFT").map_or(true, |v| v != "0");
    let Some(mut model) = load_served(&backend, draft) else {
        skip("no export named by SHISO_TEST_MODEL_QWEN4EXP_EXL3");
        return;
    };
    let width = std::env::var("SCORING_PROFILE_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(34);
    let passes: u32 = std::env::var("SCORING_PROFILE_PASSES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let tokens = prompt(width);
    for pass in 0..=passes {
        let session = model.open_session().expect("a session");
        let started = Instant::now();
        let _ = last_row(&backend, &mut model, session, &tokens);
        let ms = started.elapsed().as_secs_f64() * 1e3;
        model.close_session(session).expect("closes");
        let label = if pass == 0 { "warm-up" } else { "timed" };
        println!("PREFILL draft={draft} tokens={width} pass={pass} {label} ms={ms:.2}");
    }
    model.teardown().expect("the model tears down");
}
