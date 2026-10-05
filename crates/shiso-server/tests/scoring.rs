//! Scoring through the engine: one prefill, an answer of label log-probabilities,
//! and no generation.
//!
//! These drive [`EngineHandle::score`] against the host-memory fake the other
//! scheduler tests share. The fake's scripted rows are the point: the answer a
//! score gets must be the log-probability of each label under the *exact* row
//! the model produced for the prompt's last position, so a test scripts that row
//! as numbers and asserts on the arithmetic, not on a spike the sampler turns
//! into a token.

mod common;

use futures::StreamExt as _;
use shiso_core::{Backend, SamplingParams, Seed, TokenId};
use shiso_ir::StreamEvent;
use shiso_server::error::AppError;
use shiso_server::scheduler::{
    EngineFailure, EngineHandle, GenerateRequest, SchedulerConfig, ScoreRequest,
    spawn_engine_configured,
};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// The fake's vocabulary width, so a scripted row covers every logit the
/// scheduler reads. A row narrower than this leaves the tail of the previous
/// row in the logits buffer, and a log-sum-exp would silently include it.
const VOCAB: usize = 64;

/// A full logits row: `[1.0, 2.0, 3.0, 0, 0, ...]` over the fake vocabulary.
fn label_row() -> Vec<f32> {
    let mut row = vec![0f32; VOCAB];
    row[0] = 1.0;
    row[1] = 2.0;
    row[2] = 3.0;
    row
}

/// A scoring request over `prompt` for `labels`, in the standard class.
fn score_request(prompt: &[u32], labels: &[u32]) -> ScoreRequest {
    ScoreRequest::new(
        Arc::from(prompt.iter().copied().map(TokenId::new).collect::<Vec<_>>()),
        Arc::from(labels.iter().copied().map(TokenId::new).collect::<Vec<_>>()),
    )
}

/// Argmax sampling, so a scripted spike in the logits row is what comes out.
fn argmax() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::neutral()
    }
}

/// An engine over scripted prefill rows that holds exactly one sequence at a
/// time: admission of a second request is then proof that the first released
/// its slot, rather than proof that the table had room anyway.
fn single_slot_rows(
    rows: Vec<Vec<f32>>,
) -> (
    EngineHandle,
    std::thread::JoinHandle<Result<(), EngineFailure>>,
    Arc<common::Audit>,
) {
    let backend: Arc<dyn Backend> = common::FakeBackend::new();
    let (model, audit) = common::FakeModel::new(&backend, Vec::new());
    let model = model.with_prefill_rows(rows);
    let cfg = SchedulerConfig {
        max_num_seqs: 1,
        ..SchedulerConfig::new()
    };
    let (engine, join) = spawn_engine_configured(
        backend,
        Box::new(model),
        None,
        "fake-model",
        common::CAPACITY,
        cfg,
    );
    (engine, join, audit)
}

/// The log-sum-exp of `row`, computed here in `f64` by the same formula, so the
/// assertion is independent of the terminus's own arithmetic.
fn logsumexp(row: &[f32]) -> f64 {
    let max = row.iter().fold(f32::NEG_INFINITY, |best, v| best.max(*v));
    f64::from(max)
        + row
            .iter()
            .map(|v| (f64::from(*v) - f64::from(max)).exp())
            .sum::<f64>()
            .ln()
}

#[tokio::test]
async fn a_score_replies_label_logprobs_from_the_prompt_last_row() {
    let row = label_row();
    let (engine, join, _audit) = common::fake_engine_rows(vec![row.clone()], None);

    let outcome = engine
        .score(score_request(&[1, 2, 3], &[0, 1, 2]))
        .await
        .expect("the score is answered");

    // Each label's log-probability is its row value less the row's
    // log-sum-exp. The expected vector is computed here, in `f64`, by the
    // same formula, so the assertion is arithmetic a reader can check rather
    // than a constant copied from the terminus.
    let norm = logsumexp(&row);
    let expected: Vec<f64> = [0usize, 1, 2]
        .iter()
        .map(|at| f64::from(row[*at]) - norm)
        .collect();
    assert_eq!(outcome.label_logprobs.len(), 3);
    for (got, want) in outcome.label_logprobs.iter().zip(&expected) {
        assert!(
            (got - want).abs() < 1e-5,
            "label logprob {got} is not {want} for row {row:?}"
        );
    }

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

#[tokio::test]
async fn a_score_closes_its_session_and_frees_its_slot() {
    let (engine, join, audit) = single_slot_rows(vec![label_row()]);

    engine
        .score(score_request(&[1, 2], &[0, 1]))
        .await
        .expect("the score is answered");

    // The reply is sent at the prefill terminus; the session closes later in
    // the same tick, on the retirement path. Wait for the close rather than
    // sleeping: the assertions below are about what the coordinator does, not
    // about how fast this machine runs.
    for _ in 0..2_000 {
        if audit.closes.load(Ordering::Relaxed) >= 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    assert_eq!(audit.opens.load(Ordering::Relaxed), 1);
    assert_eq!(audit.closes.load(Ordering::Relaxed), 1);
    assert!(audit.live.lock().is_empty(), "nothing stays open");

    // The only slot is free again, so a generation is admitted. If the score
    // had kept its slot, this would wait out the admission window and come
    // back Overloaded.
    let mut stream = engine
        .generate(GenerateRequest::new(
            Arc::from(vec![TokenId::new(1)]),
            argmax(),
            Seed::new(7),
            4,
        ))
        .await
        .expect("a generation is admitted after the score released its slot");
    while let Some(_event) = stream.next().await {}
    assert_eq!(audit.opens.load(Ordering::Relaxed), 2);
    assert_eq!(audit.closes.load(Ordering::Relaxed), 2);

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

#[tokio::test]
async fn a_score_records_its_prompt_and_labels_at_the_model() {
    // One quantum covers the whole prompt: the model saw exactly these ids.
    let (engine, join, audit) = common::fake_engine_rows(vec![label_row()], None);
    let outcome = engine
        .score(score_request(&[7, 8, 9], &[0, 1, 2]))
        .await
        .expect("the score is answered");
    assert_eq!(
        outcome.label_logprobs.len(),
        3,
        "one logprob per label, in the request's order"
    );
    assert_eq!(*audit.prefill_prompts.lock(), vec![vec![7u32, 8, 9]]);
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");

    // A prompt wider than one quantum is read in chunks, and still replies
    // exactly once: the answer comes from the chunk that finishes the prompt,
    // and the chunks concatenated are the prompt the caller sent.
    let prompt: Vec<u32> = (0..200u32).map(|t| t % 60).collect();
    let backend: Arc<dyn Backend> = common::FakeBackend::new();
    let (model, audit) = common::FakeModel::new(&backend, Vec::new());
    let model = model.with_prefill_rows(vec![label_row()]);
    let cfg = SchedulerConfig {
        solo_chunk: 64,
        ..SchedulerConfig::new()
    };
    let (engine, join) = spawn_engine_configured(
        backend,
        Box::new(model),
        None,
        "fake-model",
        common::CAPACITY,
        cfg,
    );
    let outcome = engine
        .score(score_request(&prompt, &[5, 9]))
        .await
        .expect("a prompt read in pieces still answers once");
    assert_eq!(outcome.label_logprobs.len(), 2);
    let chunks = audit.prefill_prompts.lock();
    assert!(
        chunks.len() > 1,
        "the prompt was actually split: {chunks:?}"
    );
    let read: Vec<u32> = chunks.iter().flatten().copied().collect();
    assert_eq!(read, prompt, "the chunks are the prompt, in order");
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

#[tokio::test]
async fn a_score_replies_err_when_the_sequence_fails() {
    let (engine, join, audit) = common::fake_engine_rows(vec![label_row()], None);

    for (prompt, labels) in [(&[][..], &[0u32][..]), (&[1u32][..], &[][..])] {
        let error = engine
            .score(score_request(prompt, labels))
            .await
            .expect_err("a prompt or a label set is empty: the request is refused");
        match error {
            AppError::BadRequest { message, code, .. } => {
                assert_eq!(
                    message,
                    "a scoring request needs a prompt and at least one label"
                );
                assert_eq!(code, "invalid_request");
            }
            other => panic!("an empty scoring request is a bad request, not {other:?}"),
        }
    }
    assert_eq!(
        audit.opens.load(Ordering::Relaxed),
        0,
        "a refused request opens no session"
    );

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

#[tokio::test]
async fn scores_and_generate_share_the_engine() {
    let (engine, join, audit) = common::fake_engine_rows(vec![label_row()], None);

    let score_prompt = vec![1u32, 2, 3];
    let generate_prompt = vec![4u32, 5];
    let (scored, streamed) = tokio::join!(
        engine.score(score_request(&score_prompt, &[0, 1, 2])),
        engine.generate(GenerateRequest::new(
            Arc::from(
                generate_prompt
                    .iter()
                    .copied()
                    .map(TokenId::new)
                    .collect::<Vec<_>>()
            ),
            argmax(),
            Seed::new(3),
            4,
        )),
    );
    let outcome = scored.expect("the score is answered beside the generation");
    assert_eq!(outcome.label_logprobs.len(), 3);

    let mut stream = streamed.expect("the generation is admitted beside the score");
    let mut stopped = false;
    while let Some(event) = stream.next().await {
        stopped |= matches!(event, StreamEvent::Stop { .. });
    }
    assert!(stopped, "the generation ran to its end");

    // Both prompts reached the model. Which one popped the scripted row is
    // not fixed - the two share the one prefill slot per tick - so this
    // asserts both reads, not their order.
    let prompts = audit.prefill_prompts.lock();
    assert!(
        prompts.iter().any(|read| read == &score_prompt),
        "the score's prompt was not read: {prompts:?}"
    );
    assert!(
        prompts.iter().any(|read| read == &generate_prompt),
        "the generate's prompt was not read: {prompts:?}"
    );

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// One question of a scoring request: its prompt and how much of it the
/// request's questions share.
struct Question {
    prompt: Vec<u32>,
    shared: u32,
}

/// A question over `shared` tokens of a common prefix and a 4-token tail
/// ending in `last`, which keys the fake's answer.
fn question(shared: u32, last: u32) -> Question {
    let mut prompt = (0..shared).map(|i| 13 + i % 40).collect::<Vec<_>>();
    prompt.extend([3, 4, 5, last]);
    Question { prompt, shared }
}

/// Asks `questions`, in order, while a generation holds the engine in a
/// decode step, so all of them are resident - admitted in this order - before
/// the first is planned. Returns each answer in question order, and the
/// audit.
async fn ask_while_decoding(
    cfg: SchedulerConfig,
    batched: bool,
    hit: Option<u32>,
    questions: &[Question],
) -> (Vec<Vec<f64>>, Arc<common::Audit>) {
    let gate = common::Gate::closed();
    let (engine, join, audit) =
        common::batched_scoring_engine(cfg, Arc::clone(&gate), batched, hit);
    let chat = engine
        .generate(GenerateRequest::new(
            Arc::from(vec![TokenId::new(1)]),
            argmax(),
            Seed::new(7),
            50,
        ))
        .await
        .expect("the generation is admitted");
    common::wait_for_calls(&audit, 1).await;
    let mut asks = Vec::new();
    for (at, ask) in (2_u64..).zip(questions) {
        let request = score_request(&ask.prompt, &[0, 1, 2]).with_shared_prefix(ask.shared);
        let engine_for_ask = engine.clone();
        asks.push(tokio::spawn(
            async move { engine_for_ask.score(request).await },
        ));
        common::wait_for_queued(&engine, at).await;
    }
    gate.open();
    let mut answers = Vec::new();
    for ask in asks {
        answers.push(
            ask.await
                .expect("not cancelled")
                .expect("answered")
                .label_logprobs,
        );
    }
    let _ = chat.collect::<Vec<_>>().await;
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    (answers, audit)
}

/// A scheduler for eight sequences whose checkpoint floor is `floor`.
fn sibling_cfg(floor: u32) -> SchedulerConfig {
    let mut cfg = SchedulerConfig {
        max_num_seqs: 8,
        ..SchedulerConfig::new()
    };
    cfg.store.min_tokens = floor;
    cfg
}

/// Six sibling questions: a 64-token shared prefix the checkpoint tier can
/// stand them on, and a 4-token tail each, whose last tokens differ.
async fn six_siblings(batched: bool, hit: Option<u32>) -> (Vec<Vec<f64>>, Arc<common::Audit>) {
    let questions = (0..6_u32).map(|q| question(64, 6 + q)).collect::<Vec<_>>();
    ask_while_decoding(sibling_cfg(64), batched, hit, &questions).await
}

/// Each answer is the keyed row's log-probabilities for its own last token.
fn assert_keyed_answers(answers: &[Vec<f64>]) {
    for (q, got) in answers.iter().enumerate() {
        let row = common::keyed_row(6 + u32::try_from(q).expect("a handful"));
        let norm = logsumexp(&row);
        for (label, value) in got.iter().enumerate() {
            let want = f64::from(row[label]) - norm;
            assert!(
                (value - want).abs() < 1e-5,
                "question {q} label {label}: {value} vs {want}"
            );
        }
    }
}

/// The resumes asked for scoring prompts: the generation's one-token prompt
/// left out.
fn score_resumes(audit: &common::Audit) -> usize {
    audit.resumes.lock().iter().filter(|len| **len > 1).count()
}

/// The prompt-reading calls, in order: decode steps left out.
fn prompt_calls(audit: &common::Audit) -> Vec<common::Call> {
    audit
        .calls
        .lock()
        .iter()
        .copied()
        .filter(|call| !matches!(call, common::Call::Decode(_)))
        .collect()
}

#[tokio::test]
async fn sibling_scores_share_one_prompt_pass_and_answer_as_they_do_one_at_a_time() {
    let (together, audit) = six_siblings(true, Some(64)).await;
    let (alone, alone_audit) = six_siblings(false, Some(64)).await;
    assert_keyed_answers(&together);
    assert_keyed_answers(&alone);
    // All six are resident before the first is planned, so the expected log is
    // one pass of six; what must hold whatever the admission timing is that
    // some pass carried several tails and every tail was read exactly once.
    let batched = audit.batched_runs();
    assert!(
        batched.iter().any(|runs| *runs > 1),
        "no batched pass: {:?}",
        audit.calls.lock()
    );
    assert_eq!(
        batched.iter().sum::<usize>() + audit.prefill_chunks().len() - 1,
        6,
        "each tail read once, together or alone: {:?}",
        audit.calls.lock()
    );
    assert!(
        alone_audit.batched_runs().is_empty(),
        "a model without the pass reads none"
    );
    assert_eq!(
        alone_audit.prefill_chunks().len(),
        7,
        "the generation's prompt and six tails, one call each"
    );
    assert_eq!(
        alone_audit.batch_refusals.load(Ordering::Relaxed),
        1,
        "one refusal is enough: the planner stops offering the pass"
    );
}

#[tokio::test]
async fn siblings_whose_shared_prefix_nobody_has_read_wait_for_the_first_to_read_it() {
    let (answers, audit) = six_siblings(true, None).await;
    assert_keyed_answers(&answers);
    assert!(
        audit.batched_runs().is_empty(),
        "no checkpoint to stand on, so nothing was batched"
    );
    assert_eq!(
        audit.prefill_chunks().len(),
        7,
        "each question read once, alone"
    );
}

/// A family whose shared prefix ends below the checkpoint tier's floor has
/// no checkpoint to wait for: its questions are read whole, in one pass, and
/// answer as they do one at a time.
#[tokio::test]
async fn a_family_sharing_less_than_the_floor_is_read_whole_in_one_pass() {
    let cfg = || SchedulerConfig {
        mixed_quantum: 512,
        ..sibling_cfg(512)
    };
    let questions = (0..6_u32).map(|q| question(20, 6 + q)).collect::<Vec<_>>();
    let (together, audit) = ask_while_decoding(cfg(), true, None, &questions).await;
    let (alone, alone_audit) = ask_while_decoding(cfg(), false, None, &questions).await;
    assert_keyed_answers(&together);
    assert_keyed_answers(&alone);
    assert_eq!(audit.batched_runs(), vec![6], "{:?}", audit.calls.lock());
    assert_eq!(
        audit.prefill_chunks().len(),
        1,
        "only the generation's prompt read alone"
    );
    assert_eq!(
        score_resumes(&audit),
        6,
        "each question resumed once, never re-armed"
    );
    assert_eq!(alone_audit.prefill_chunks().len(), 7);
}

/// A lead whose resume did not stand it on its shared prefix reads first,
/// alone, while the siblings that do stand wait a tick: passing it over
/// would hold up everything admitted behind it.
#[tokio::test]
async fn a_lead_that_missed_its_shared_prefix_reads_before_the_batch() {
    // The lead shares 128 tokens with nobody resident and resumes only 64;
    // the two after it share 64 and stand where they resume.
    let questions = [question(128, 6), question(64, 7), question(64, 8)];
    let (answers, audit) = ask_while_decoding(sibling_cfg(64), true, Some(64), &questions).await;
    assert_keyed_answers(&answers);
    assert_eq!(
        prompt_calls(&audit),
        vec![
            common::Call::Prefill {
                tokens: 1,
                last: true
            },
            common::Call::Prefill {
                tokens: 68,
                last: true
            },
            common::Call::PrefillMany { runs: 2, tokens: 8 },
        ],
        "the lead's 68 unresumed tokens first, then its siblings' tails together"
    );
}

/// Siblings waiting on a first read that takes several quanta ask for their
/// checkpoint again only once that read has crossed the shared prefix, not
/// once a tick.
#[tokio::test]
async fn waiting_siblings_do_not_resume_again_until_a_read_crosses_the_shared_prefix() {
    let questions = (0..3_u32).map(|q| question(256, 6 + q)).collect::<Vec<_>>();
    let (answers, audit) = ask_while_decoding(sibling_cfg(64), true, None, &questions).await;
    assert_keyed_answers(&answers);
    // First tick: all three ask and miss, the lead reads 128. Its second
    // quantum crosses 256, so the two waiting ask once more and miss; the
    // second reads alone without asking again, its second quantum crosses,
    // and the third asks one last time before reading alone too.
    assert_eq!(
        score_resumes(&audit),
        6,
        "{:?} {:?}",
        audit.resumes.lock(),
        prompt_calls(&audit)
    );
}
