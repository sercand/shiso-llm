//! When the coordinator asks the checkpoint tier something, and what it offers.
//!
//! The model decides whether anything is stored - a prefix below its floor, one
//! it already holds, a session that read media - and it files a prompt's end on
//! its own, from the commit that ends the prompt. What the scheduler decides is
//! *which other moments* are offered: the finished turn, a turn cut short, and a
//! sweep on the way down; and what it hands the model beforehand: the prompt's
//! tokens to resume from and its message boundaries.

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use futures::StreamExt as _;
use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_engine::Reason;
use shiso_server::scheduler::{GenerateRequest, SchedulerConfig, ScoreRequest, StorePolicy};

fn argmax() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::neutral()
    }
}

/// A request whose rendered text is `text`.
fn request(text: &str, prompt: usize, max_tokens: u32) -> GenerateRequest {
    GenerateRequest::new(
        Arc::from(vec![TokenId::new(1); prompt]),
        argmax(),
        Seed::new(7),
        max_tokens,
    )
    .with_text(Arc::from(text))
}

/// A policy that offers a frontier at the end of a turn and never in between,
/// so a test can count offers rather than race them.
fn ends_only(max_num_seqs: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs,
        store: StorePolicy {
            boundary_align: 0,
            boundary_trim: 0,
            continued_interval: 0,
            ..StorePolicy::new()
        },
        ..SchedulerConfig::new()
    }
}

/// Reads a stream to its end.
async fn drain(mut stream: shiso_ir::EventStream) {
    while stream.next().await.is_some() {}
}

/// A finished turn is offered once, at the last position its session ran.
///
/// The prompt's end is not offered here: the model files it itself, from the
/// commit that ends the prompt. The turn's end is the prompt plus every
/// generated token a decode step ran; the end-of-turn id that stopped it was
/// sampled and never run, and the next turn's prompt supplies it again.
#[tokio::test]
async fn a_finished_turn_offers_the_last_position_its_session_ran() {
    let (engine, join, audit) = common::checkpointed_engine(vec![10, 11, 12], ends_only(2), None);
    drain(
        engine
            .generate(request("a rendered prompt", 6, 16))
            .await
            .expect("the engine accepts the request"),
    )
    .await;

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");

    let saves = audit.saves.lock().clone();
    assert_eq!(saves.len(), 1, "one offer, not one per token: {saves:?}");
    assert_eq!(saves[0].reason, Reason::Continued);
    assert_eq!(
        saves[0].tokens, 9,
        "the prompt plus the three generated ids, each run by the step after it"
    );
}

/// A prompt the tier covers whole starts decoding from the row the resume
/// hands back, and no prompt token goes through a forward pass.
#[tokio::test]
async fn a_prompt_the_tier_covers_whole_is_not_prefilled() {
    let (engine, join, audit) =
        common::checkpointed_engine_hitting(vec![10, 11], ends_only(2), None, Some(6));
    drain(
        engine
            .generate(request("a covered prompt", 6, 4))
            .await
            .expect("the engine accepts the request"),
    )
    .await;
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert_eq!(audit.resumes.lock().clone(), vec![6]);
    assert!(
        audit.prefill_chunks().is_empty(),
        "a covered prompt read nothing: {:?}",
        audit.prefill_chunks()
    );
}

/// A prompt the tier covers in part prefills exactly the rest.
#[tokio::test]
async fn a_prompt_the_tier_covers_in_part_prefills_only_the_rest() {
    let (engine, join, audit) =
        common::checkpointed_engine_hitting(vec![10, 11], ends_only(2), None, Some(4));
    drain(
        engine
            .generate(request("a longer prompt", 10, 4))
            .await
            .expect("the engine accepts the request"),
    )
    .await;
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert_eq!(
        audit.prefill_chunks(),
        vec![6],
        "the six tokens past the resume"
    );
}

/// The model is told where the prompt's message boundaries are before it reads
/// the prompt: just past the first end-of-turn token, and the start of the final
/// message. The boundary just before the generation prompt is not handed over:
/// the prompt's own end, a few tokens later, is checkpointed anyway.
#[tokio::test]
async fn the_first_and_last_message_boundaries_are_handed_over() {
    let (engine, join, audit) = common::checkpointed_engine(vec![10, 11], ends_only(2), None);
    let eos = TokenId::new(common::EOS);
    let one = TokenId::new(1);
    let prompt = vec![one, one, eos, one, eos, one, eos, one, one, eos, one];
    let req = GenerateRequest::new(Arc::from(prompt), argmax(), Seed::new(7), 4);
    drain(engine.generate(req).await.expect("accepted")).await;
    // One message and a generation prompt: nothing worth a stop.
    let single = vec![one, one, one, eos, one];
    let req = GenerateRequest::new(Arc::from(single), argmax(), Seed::new(7), 4);
    drain(engine.generate(req).await.expect("accepted")).await;
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert_eq!(audit.hints.lock().clone(), vec![vec![3, 7], vec![]]);
}

/// A turn whose client hung up is state being discarded rather than a
/// conversation somebody may continue, and the reason on the file says so.
#[tokio::test]
async fn an_abandoned_turn_is_stored_as_state_leaving_memory() {
    // Held still at the first decode step, so the turn has read its prompt and
    // is genuinely mid-flight when its client goes away. Dropping a stream
    // before that is a different case - nothing has been computed, so there is
    // nothing to file - and it is the one the next test covers.
    let gate = common::Gate::closed();
    let (engine, join, audit) =
        common::checkpointed_engine(vec![10; 400], ends_only(1), Arc::clone(&gate).into());

    let stream = engine
        .generate(request("abandoned", 4, 300))
        .await
        .expect("the engine accepts the request");
    common::wait_for_live(&audit, &gate, 1).await;
    // And until the prompt has actually been read: a session exists before its
    // turn has started, and a stream dropped in that window is the other case -
    // nothing has been computed, so there is nothing to file.
    common::wait_for_calls(&audit, 1).await;
    drop(stream);
    gate.open();

    for _ in 0..2_000 {
        if audit.closes.load(Ordering::Relaxed) >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");

    let saves = audit.saves.lock().clone();
    assert!(
        saves.iter().any(|note| note.reason == Reason::Evict),
        "an abandoned turn's state is stored as leaving memory: {saves:?}"
    );
}

/// A prompt with no rendered text is still looked up and still filed: the tier
/// keys by tokens, and a legacy completion's tokens are as good a key as a chat
/// prompt's.
#[tokio::test]
async fn a_prompt_with_no_rendered_text_is_still_resumed_and_stored() {
    let (engine, join, audit) = common::checkpointed_engine(vec![10, 11], ends_only(2), None);
    let textless = GenerateRequest::new(
        Arc::from(vec![TokenId::new(1); 4]),
        argmax(),
        Seed::new(7),
        8,
    );
    drain(
        engine
            .generate(textless)
            .await
            .expect("the engine accepts the request"),
    )
    .await;

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");

    assert_eq!(audit.resumes.lock().clone(), vec![4], "it was looked up");
    assert_eq!(audit.saves.lock().len(), 1, "its turn's end was offered");
}

/// A score sequence never files a continued store entry. Its prompt is read
/// once and kept by nobody, and with the interval path on every tick, a
/// mid-prefill score frontier would otherwise reach the disk tier under a key
/// no follow-up will ever ask for - the ruling that keeps scores off the tier
/// is enforced at the retirement site *and* in the `store_continued` filter.
#[tokio::test]
async fn a_score_sequence_never_files_a_continued_store_entry() {
    let cfg = SchedulerConfig {
        // Two solo chunks of a prompt wider than the prefill grid, so the
        // score's prefill spans two ticks: the slot must be alive and
        // unfinished when the interval path looks at it, which a one-tick
        // prefill never is.
        solo_chunk: 128,
        mixed_quantum: 128,
        store: StorePolicy {
            boundary_align: 0,
            boundary_trim: 0,
            continued_interval: 1,
            ..StorePolicy::new()
        },
        ..SchedulerConfig::new()
    };
    let (engine, join, audit) = common::checkpointed_engine(Vec::new(), cfg, None);
    engine
        .score(ScoreRequest::new(
            Arc::from(vec![TokenId::new(1); 192]),
            Arc::from(vec![TokenId::new(0), TokenId::new(1)]),
        ))
        .await
        .expect("the score is answered");
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");

    let saves = audit.saves.lock().clone();
    assert!(
        saves.iter().all(|note| note.reason != Reason::Continued),
        "a score sequence offered a continued frontier: {saves:?}"
    );
}

/// The sweep is the safety net for sessions the retirement path never reached.
/// It runs on the way down whether or not it finds anything, because a shutdown
/// that skipped it when it *did* have something to store would turn work
/// already paid for into a prefill somebody else has to pay.
#[tokio::test]
async fn the_shutdown_sweep_runs_exactly_once_on_the_way_down() {
    let (engine, join, audit) = common::checkpointed_engine(vec![10, 11], ends_only(2), None);
    let stream = engine
        .generate(request("closing", 2, 8))
        .await
        .expect("the engine accepts the request");
    drop(engine);
    drain(stream).await;
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");

    assert_eq!(
        audit.sweeps.load(Ordering::Relaxed),
        1,
        "the sweep runs once, whether or not it finds anything"
    );
    let saves = audit.saves.lock().clone();
    assert!(
        saves
            .iter()
            .any(|note| matches!(note.reason, Reason::Shutdown | Reason::Continued)),
        "the turn's end was offered too: {saves:?}"
    );
}
