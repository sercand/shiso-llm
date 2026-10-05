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

use shiso_core::CoreError;
use shiso_core::backend::Backend;
use shiso_core::config::HfConfig;
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::Logits;
use shiso_core::model::Model;
use shiso_core::model::PromptRun;
use shiso_core::model::Speculating as _;
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

/// The id of a logits row's largest value.
fn argmax(row: &[u8]) -> u32 {
    let best = row
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        })
        .0;
    u32::try_from(best).expect("a vocabulary id")
}

/// Copies logits row `row` of `rows` back.
fn read_row(backend: &Arc<dyn Backend>, rows: &Logits<'_>, row: u32) -> Vec<u8> {
    let view = rows.row(row).expect("a row");
    let mut bytes = vec![0_u8; usize::try_from(view.len()).expect("a host index")];
    backend
        .copy_d2h(view, &mut bytes)
        .expect("the row reads back");
    bytes
}

/// A scoring session whose shared boundary is the end of `prefix`, standing on
/// `prefix` read as one chunk: where a sibling question stands once it has
/// resumed at its request's shared prefix.
fn scoring_on(model: &mut Qwen4ExpModel, prefix: &[TokenId]) -> SessionId {
    let session = model.open_session().expect("a session");
    model.kv_set_scoring(session).expect("marked");
    model
        .kv_set_hints(session, &[u32::try_from(prefix.len()).expect("a position")])
        .expect("hinted");
    model
        .prefill_chunk(session, prefix)
        .expect("the prefix runs");
    session
}

/// The row a solo prompt pass leaves at the end of `tail`, read after `prefix`.
fn solo_tail(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    prefix: &[TokenId],
    tail: &[TokenId],
) -> Vec<u8> {
    let session = scoring_on(model, prefix);
    let row = last_row(backend, model, session, tail);
    model.close_session(session).expect("closes");
    row
}

/// Each run's row from one batched prompt pass, every run on a scoring session
/// of its own standing on its own prefix.
fn batched_tails(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    runs: &[(&[TokenId], &[TokenId])],
) -> Vec<Vec<u8>> {
    let sessions = runs
        .iter()
        .map(|(prefix, _)| scoring_on(model, prefix))
        .collect::<Vec<_>>();
    let pairs = sessions
        .iter()
        .copied()
        .zip(runs.iter().map(|(_, tail)| *tail))
        .collect::<Vec<_>>();
    let rows = {
        let logits = model
            .stack_mut()
            .prefill_segments(&pairs)
            .expect("the batched pass runs");
        (0..logits.rows())
            .map(|r| read_row(backend, &logits, r))
            .collect::<Vec<_>>()
    };
    for session in sessions {
        model.close_session(session).expect("closes");
    }
    rows
}

/// Max |a-b| over two f32 rows, and how many elements differ.
fn diff(a: &[u8], b: &[u8]) -> (f32, usize) {
    let fa = a
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    let fb = b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    fa.zip(fb).fold((0.0_f32, 0), |(m, n), (x, y)| {
        if x.to_bits() == y.to_bits() {
            (m, n)
        } else {
            (m.max((x - y).abs()), n + 1)
        }
    })
}

/// Every row byte-equal to its solo row, or a failure naming the first that
/// is not, how far it moved and whether its top token did.
fn assert_rows_match(got: &[Vec<u8>], want: &[Vec<u8>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: one row a run");
    for (run, (got, want)) in got.iter().zip(want).enumerate() {
        if got != want {
            let (max, differing) = diff(got, want);
            panic!(
                "{what}: run {run} differs from its solo row in {differing} logits, max |d| {max:e}, argmax {} vs {}",
                argmax(got),
                argmax(want)
            );
        }
    }
}

/// The draft row a session has staged, read back.
fn staged_row(backend: &Arc<dyn Backend>, model: &Qwen4ExpModel, session: SessionId) -> Vec<u8> {
    let live = model.stack().session(session).expect("open");
    let view = live.pending_draft().expect("a staged row");
    let mut bytes = vec![0_u8; usize::try_from(view.len()).expect("a host index")];
    backend
        .copy_d2h(view, &mut bytes)
        .expect("the row reads back");
    bytes
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

/// What four greedy sessions produce over eight batched speculative steps, one
/// line per session, against a file the first run records.
///
/// A segmented verify's rows must sum exactly as they always have, whatever
/// the segmented pass learns to carry besides: a change that moved one sum
/// moves a pick, and a batched speculative step then keeps different tokens.
/// The GGUF verify suite cannot run on this machine, so this is the lock. A
/// missing golden fails the test; `SHISO_RECORD_GOLDEN=1` records one instead,
/// on code that is known to be right.
#[test]
#[ignore = "needs a CUDA device and the served qwen4exp EXL3 export"]
#[expect(
    clippy::print_stdout,
    reason = "the recording notice is the test's output"
)]
fn a_batched_speculative_step_keeps_its_tokens() {
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let Some(mut model) = load_served(&backend, true) else {
        skip("no export named by SHISO_TEST_MODEL_QWEN4EXP_EXL3");
        return;
    };
    let tokens = prompt(1_400);
    let sessions = (0..4)
        .map(|_| model.open_session().expect("a session"))
        .collect::<Vec<_>>();
    let mut next = Vec::new();
    for (at, session) in sessions.iter().enumerate() {
        let row = last_row(
            &backend,
            &mut model,
            *session,
            &tokens[at * 100..at * 100 + 600],
        );
        next.push(TokenId::new(argmax(&row)));
    }
    let mut streams = vec![Vec::new(); sessions.len()];
    for _ in 0..8 {
        let steps = sessions
            .iter()
            .copied()
            .zip(next.iter().copied())
            .collect::<Vec<_>>();
        let produced = model
            .speculate_batch(&steps)
            .expect("the batched step runs");
        for ((stream, out), first) in streams.iter_mut().zip(&produced).zip(next.iter_mut()) {
            stream.extend(out.produced.iter().map(|token| token.get()));
            if let Some(last) = out.produced.last() {
                *first = *last;
            }
        }
    }
    let text = streams
        .iter()
        .map(|stream| format!("{stream:?}"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/scoring-golden/speculate-batch.txt");
    if std::env::var_os("SHISO_RECORD_GOLDEN").is_some_and(|v| v == "1") {
        std::fs::create_dir_all(golden.parent().expect("a parent")).expect("the directory");
        std::fs::write(&golden, &text).expect("the golden writes");
        println!("RECORDED {} (SHISO_RECORD_GOLDEN=1)", golden.display());
    } else {
        let want = std::fs::read_to_string(&golden).unwrap_or_else(|error| {
            panic!(
                "no golden at {} ({error}): record one with SHISO_RECORD_GOLDEN=1 on code known \
                 to be right",
                golden.display()
            )
        });
        assert_eq!(text, want, "a batched speculative step changed its tokens");
    }
    for session in sessions {
        model.close_session(session).expect("closes");
    }
    model.teardown().expect("the model tears down");
}

#[test]
#[ignore = "needs a CUDA device and the served qwen4exp EXL3 export"]
fn batched_scoring_tails_read_the_rows_their_solo_prefills_read() {
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let Some(mut model) = load_served(&backend, true) else {
        skip("no export named by SHISO_TEST_MODEL_QWEN4EXP_EXL3");
        return;
    };
    let tokens = prompt(1_600);
    let prefix = &tokens[..1_024];
    // 40 rows (past the expert sort's and the decode tier's widths), 7 rows
    // (under both), 256 rows (wider than the old one-byte run table).
    let tails: [&[TokenId]; 3] = [
        &tokens[1_024..1_064],
        &tokens[1_100..1_107],
        &tokens[1_200..1_456],
    ];
    let want = tails.map(|tail| solo_tail(&backend, &mut model, prefix, tail));
    let got = batched_tails(
        &backend,
        &mut model,
        &[(prefix, tails[0]), (prefix, tails[1]), (prefix, tails[2])],
    );
    assert_rows_match(&got, &want, "runs of 40, 7 and 256 rows");
    let got = batched_tails(
        &backend,
        &mut model,
        &[(prefix, tails[2]), (prefix, tails[1]), (prefix, tails[0])],
    );
    assert_rows_match(
        &got,
        &[want[2].clone(), want[1].clone(), want[0].clone()],
        "the same runs reversed",
    );
    let got = batched_tails(&backend, &mut model, &[(prefix, tails[1])]);
    assert_rows_match(&got, &want[1..2], "one run alone");
    // A pass narrower than every width threshold: 12 rows in all, while
    // its 5- and 4-row solo tails sit on the other side of the grouping one.
    let small: [&[TokenId]; 3] = [
        &tokens[1_500..1_505],
        &tokens[1_510..1_513],
        &tokens[1_520..1_524],
    ];
    let want = small.map(|tail| solo_tail(&backend, &mut model, prefix, tail));
    let got = batched_tails(
        &backend,
        &mut model,
        &[(prefix, small[0]), (prefix, small[1]), (prefix, small[2])],
    );
    assert_rows_match(&got, &want, "runs of 5, 3 and 4 rows");
    model.teardown().expect("the model tears down");
}

/// Past the indexer budget (2048 positions) each run attends over its own
/// selection, and a run whose own rows cross the budget is split where its
/// solo pass splits it. The spike's contexts were 1054 positions.
#[test]
#[ignore = "needs a CUDA device and the served qwen4exp EXL3 export"]
fn batched_scoring_tails_past_the_indexer_budget_read_their_solo_rows() {
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let Some(mut model) = load_served(&backend, true) else {
        skip("no export named by SHISO_TEST_MODEL_QWEN4EXP_EXL3");
        return;
    };
    let tokens = prompt(3_000);
    let long = &tokens[..2_560];
    let crossing = &tokens[..2_000];
    let prefixes: [&[TokenId]; 3] = [long, crossing, long];
    let tails: [&[TokenId]; 3] = [
        &tokens[2_560..2_593],
        &tokens[2_000..2_090],
        &tokens[2_600..2_800],
    ];
    let want = [0, 1, 2].map(|i| solo_tail(&backend, &mut model, prefixes[i], tails[i]));
    let got = batched_tails(
        &backend,
        &mut model,
        &[(long, tails[0]), (crossing, tails[1]), (long, tails[2])],
    );
    assert_rows_match(&got, &want, "contexts past the indexer budget");
    model.teardown().expect("the model tears down");
}

/// After a batched tail a session stands exactly where a solo read leaves it:
/// the same staged draft row (the run's own last pass row, not row `r`), and
/// the same row for whatever it reads next - which needs its cache, recurrent
/// state and n-gram window to be the solo read's.
#[test]
#[ignore = "needs a CUDA device and the served qwen4exp EXL3 export"]
fn a_batched_tail_leaves_its_session_where_a_solo_read_leaves_it() {
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let Some(mut model) = load_served(&backend, true) else {
        skip("no export named by SHISO_TEST_MODEL_QWEN4EXP_EXL3");
        return;
    };
    let tokens = prompt(1_400);
    let prefix = &tokens[..1_024];
    let (tail, other, next) = (
        &tokens[1_024..1_064],
        &tokens[1_100..1_130],
        &tokens[1_300..1_309],
    );
    let solo = scoring_on(&mut model, prefix);
    let _ = last_row(&backend, &mut model, solo, tail);
    let solo_staged = staged_row(&backend, &model, solo);
    let solo_next = last_row(&backend, &mut model, solo, next);
    model.close_session(solo).expect("closes");

    let a = scoring_on(&mut model, prefix);
    let b = scoring_on(&mut model, prefix);
    // `a` is the second run, so its last pass row is 69 and not 1.
    let _ = model
        .stack_mut()
        .prefill_segments(&[(b, other), (a, tail)])
        .expect("the pass runs");
    let live = model.stack().session(a).expect("open");
    assert_eq!(live.written(), 1_064);
    assert!(
        live.staged_row_current(),
        "the staged row belongs to the new frontier"
    );
    assert!(
        staged_row(&backend, &model, a) == solo_staged,
        "the staged draft row is the run's own last row"
    );
    let got_next = last_row(&backend, &mut model, a, next);
    assert!(
        got_next == solo_next,
        "the session continues as one that read its tail alone"
    );
    for session in [a, b] {
        model.close_session(session).expect("closes");
    }
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

#[test]
#[ignore = "needs a CUDA device and the served qwen4exp EXL3 export"]
fn the_model_reads_batched_tails_through_prefill_last_many() {
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let Some(mut model) = load_served(&backend, true) else {
        skip("no export named by SHISO_TEST_MODEL_QWEN4EXP_EXL3");
        return;
    };
    let tokens = prompt(1_200);
    let prefix = &tokens[..1_024];
    let (a, b) = (&tokens[1_024..1_064], &tokens[1_100..1_107]);
    let want = [
        solo_tail(&backend, &mut model, prefix, a),
        solo_tail(&backend, &mut model, prefix, b),
    ];
    let (sa, sb) = (
        scoring_on(&mut model, prefix),
        scoring_on(&mut model, prefix),
    );
    let got = {
        let logits = model
            .prefill_last_many(&[
                PromptRun {
                    session: sa,
                    tokens: a,
                },
                PromptRun {
                    session: sb,
                    tokens: b,
                },
            ])
            .expect("the batched pass runs");
        (0..2)
            .map(|r| read_row(&backend, &logits, r))
            .collect::<Vec<_>>()
    };
    assert_rows_match(&got, &want, "through the model");
    let plain = model.open_session().expect("a session");
    model.prefill_chunk(plain, prefix).expect("the prefix runs");
    let error = model
        .prefill_last_many(&[PromptRun {
            session: plain,
            tokens: a,
        }])
        .expect_err("a plain session is read alone");
    assert!(matches!(error, CoreError::Unsupported { .. }), "{error:?}");
    assert_eq!(model.stack().session(plain).expect("open").written(), 1_024);
    for session in [sa, sb, plain] {
        model.close_session(session).expect("closes");
    }
    model.teardown().expect("the model tears down");
}

/// Prints the last steps of a batched plan over eight-token tails on
/// `sessions`, where the per-run steps before the suppression end, so a
/// profile can name the vocabulary projection's kernel.
#[expect(clippy::print_stdout, reason = "the plan's tail is the test's output")]
fn print_plan_tail(model: &Qwen4ExpModel, sessions: &[SessionId]) {
    let runs = sessions.len();
    let plan = model
        .stack()
        .plan_prompt_segments(sessions, &vec![8; runs])
        .expect("plans");
    let steps = plan.plan.steps();
    for step in &steps[steps.len().saturating_sub(3 * runs + 1)..] {
        let line = format!("{step:?}");
        println!(
            "PLAN-TAIL runs={runs} {}",
            line.chars().take(200).collect::<String>()
        );
    }
}

/// Milliseconds the solo reads of `tails` take, summed, each on a scoring
/// session of its own standing on `prefix`, clocked once the prefix drains.
fn time_solo_tails(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    prefix: &[TokenId],
    tails: &[&[TokenId]],
) -> f64 {
    let mut ms = 0.0;
    for tail in tails {
        let session = scoring_on(model, prefix);
        backend.sync().expect("the queue drains");
        let started = Instant::now();
        let _ = last_row(backend, model, session, tail);
        ms += started.elapsed().as_secs_f64() * 1e3;
        model.close_session(session).expect("closes");
    }
    ms
}

/// Times one batched prompt pass over `runs` eight-token tails against the
/// same tails read one at a time, and prints the plan's tail so a profile can
/// name the vocabulary projection's kernel.
///
/// `SCORING_RUNS` (default `1,4,8,16,32,64`) lists the run counts. Solo sums
/// are timed up to eight runs. The export loads without its draft head, so a
/// session is its recurrent state alone and sixty-four fit beside the scratch.
/// If a count cannot open its sessions, the test prints how many did and stops.
#[test]
#[ignore = "needs a CUDA device and the served qwen4exp EXL3 export"]
#[expect(clippy::print_stdout, reason = "the timings are the test's output")]
fn time_batched_scoring_tails() {
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let Some(mut model) = load_served(&backend, false) else {
        skip("no export named by SHISO_TEST_MODEL_QWEN4EXP_EXL3");
        return;
    };
    let counts = std::env::var("SCORING_RUNS").map_or_else(
        |_| vec![1, 4, 8, 16, 32, 64],
        |v| {
            v.split(',')
                .filter_map(|n| n.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        },
    );
    let tokens = prompt(1_024 + 8 * 64);
    let prefix = &tokens[..1_024];
    'counts: for runs in counts {
        let tails = (0..runs)
            .map(|r| &tokens[1_024 + 8 * r..1_032 + 8 * r])
            .collect::<Vec<_>>();
        let (mut batched, mut solo) = (Vec::new(), Vec::new());
        for pass in 0..=5 {
            let mut sessions = Vec::new();
            for _ in 0..runs {
                match model.open_session() {
                    Ok(session) => sessions.push(session),
                    Err(error) => {
                        println!("BATCH runs={runs} opened={} error={error}", sessions.len());
                        break 'counts;
                    }
                }
            }
            for session in &sessions {
                model.kv_set_scoring(*session).expect("marked");
                model.kv_set_hints(*session, &[1_024]).expect("hinted");
                model
                    .prefill_chunk(*session, prefix)
                    .expect("the prefix runs");
            }
            if pass == 0 {
                print_plan_tail(&model, &sessions);
            }
            let pieces = sessions
                .iter()
                .zip(&tails)
                .map(|(session, tail)| PromptRun {
                    session: *session,
                    tokens: tail,
                })
                .collect::<Vec<_>>();
            // The prefixes' passes are still queued; the clock starts once they drain.
            backend.sync().expect("the queue drains");
            let started = Instant::now();
            {
                let logits = model.prefill_last_many(&pieces).expect("the pass runs");
                for r in 0..u32::try_from(runs).expect("runs") {
                    let _ = read_row(&backend, &logits, r);
                }
            }
            let ms = started.elapsed().as_secs_f64() * 1e3;
            for session in sessions {
                model.close_session(session).expect("closes");
            }
            let solo_ms = if runs <= 8 {
                time_solo_tails(&backend, &mut model, prefix, &tails)
            } else {
                0.0
            };
            if pass > 0 {
                batched.push(ms);
                solo.push(solo_ms);
            }
            println!("BATCH runs={runs} pass={pass} batched_ms={ms:.2} solo_sum_ms={solo_ms:.2}");
        }
        let mean = |v: &[f64]| {
            v.iter().sum::<f64>() / f64::from(u32::try_from(v.len().max(1)).unwrap_or(1))
        };
        println!(
            "BATCH runs={runs} MEAN batched_ms={:.2} solo_sum_ms={:.2}",
            mean(&batched),
            mean(&solo)
        );
    }
    model.teardown().expect("the model tears down");
}

#[test]
#[ignore = "needs a CUDA device and the served qwen4exp EXL3 export"]
fn batched_rows_match_their_solo_rows_at_every_run_count() {
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let Some(mut model) = load_served(&backend, false) else {
        skip("no export named by SHISO_TEST_MODEL_QWEN4EXP_EXL3");
        return;
    };
    let tokens = prompt(1_024 + 8 * 64);
    let prefix = &tokens[..1_024];
    let tails = (0..64)
        .map(|r| &tokens[1_024 + 8 * r..1_032 + 8 * r])
        .collect::<Vec<_>>();
    let want = tails
        .iter()
        .map(|tail| solo_tail(&backend, &mut model, prefix, tail))
        .collect::<Vec<_>>();
    for runs in [2_usize, 7, 16, 32, 64] {
        let pairs = tails[..runs]
            .iter()
            .map(|tail| (prefix, *tail))
            .collect::<Vec<_>>();
        let got = batched_tails(&backend, &mut model, &pairs);
        assert_rows_match(&got, &want[..runs], &format!("{runs} runs"));
    }
    model.teardown().expect("the model tears down");
}
