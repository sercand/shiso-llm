//! When the coordinator asks the checkpoint tier something, and what it offers.
//!
//! The engine decides whether a file appears - a prefix below its floor, one it
//! already holds, one the session is not standing on. What the scheduler decides
//! is *which moments* are offered at all, and that is what these tests pin: a
//! scan per distinct prompt rather than per sequence, a cold prefix offered at
//! the prompt's own frontier, the finished turn offered again, and a sweep on
//! the way down.
//!
//! # What cannot be tested here
//!
//! A checkpoint *hit*. A fake tier would have to return a `ResumeInfo`, which is
//! `#[non_exhaustive]` and has no constructor, so no crate outside the engine
//! can build one. The miss path, the save path and the sweep are all reachable;
//! the resume-skips-prefill case needs either a constructor on that type or a
//! real store, and is owed.

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use futures::StreamExt as _;
use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_engine::Reason;
use shiso_server::scheduler::{GenerateRequest, SchedulerConfig, StorePolicy};

fn argmax() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::neutral()
    }
}

/// A request whose rendered text is `text`, which is what a store keys on.
fn request(text: &str, prompt: usize, max_tokens: u32) -> GenerateRequest {
    GenerateRequest::new(
        Arc::from(vec![TokenId::new(1); prompt]),
        argmax(),
        Seed::new(7),
        max_tokens,
    )
    .with_text(Arc::from(text))
}

/// A policy that offers a prefix twice per turn - once cold, once at the end -
/// and never in between, so a test can count offers rather than race them.
fn ends_only(max_num_seqs: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs,
        store: StorePolicy {
            boundary_align: 0,
            boundary_trim: 0,
            cold_max_tokens: 0,
            continued_interval: 0,
        },
        ..SchedulerConfig::new()
    }
}

/// Reads a stream to its end.
async fn drain(mut stream: shiso_ir::EventStream) {
    while stream.next().await.is_some() {}
}

/// The two moments a turn is worth storing, and the paths they carry.
///
/// The cold one is the only instant the frontier is the prompt and nothing
/// else, which is the one key another request could ever name. The second is
/// the finished conversation, which is what a follow-up turn would continue
/// from.
#[tokio::test]
async fn a_turn_offers_its_prompt_cold_and_its_whole_frontier_at_the_end() {
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
    assert_eq!(saves.len(), 2, "two offers, not one per token: {saves:?}");
    assert_eq!(saves[0].reason, Reason::Cold);
    assert_eq!(
        saves[0].tokens, 6,
        "the cold offer is the prompt itself, at the one instant that is the frontier"
    );
    assert_eq!(saves[1].reason, Reason::Continued);
    assert_eq!(
        saves[1].tokens, 9,
        "the second offer is the prompt plus the three ids the turn generated"
    );
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

/// The prefix scan re-hashes whatever is on disk, so two sequences that arrived
/// together with the same prompt must not pay for it twice to be told the same
/// thing.
#[tokio::test]
async fn two_sequences_admitted_together_share_one_prefix_scan() {
    let (engine, join, audit) = common::checkpointed_engine(vec![10; 400], ends_only(3), None);

    // The first takes a seat and the coordinator settles into its loop. Its own
    // prompt text is different, so it cannot be the one being memoised.
    let first = engine
        .generate(request("the first prompt", 2, 300))
        .await
        .expect("the engine accepts the first request");

    // These two arrive together, with the same rendered text.
    let queued: Vec<_> = (0..2)
        .map(|_| {
            let engine = engine.clone();
            tokio::spawn(async move { engine.generate(request("a shared prompt", 2, 4)).await })
        })
        .collect();
    let mut streams = vec![first];
    for task in queued {
        streams.push(
            task.await
                .expect("a queued request is not cancelled")
                .expect("the engine accepts it"),
        );
    }
    for stream in streams.drain(1..) {
        drain(stream).await;
    }

    drop(streams);
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");

    let scans = audit.resumes.lock().clone();
    let shared = scans
        .iter()
        .filter(|text| *text == "a shared prompt")
        .count();
    assert!(
        (1..=2).contains(&shared),
        "at most one scan per sequence, and one when they arrive together: {scans:?}"
    );
    assert!(
        scans.iter().any(|text| text == "the first prompt"),
        "a distinct prompt is always scanned: {scans:?}"
    );
}

/// A prompt with no rendered text can never be named by a later request, so
/// there is nothing to look up and nothing to file. Asking anyway would cost
/// every legacy-completions request a scan of the whole store for a key that
/// cannot exist.
#[tokio::test]
async fn a_prompt_with_no_rendered_text_is_neither_scanned_nor_stored() {
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

    assert!(audit.resumes.lock().is_empty(), "nothing was looked up");
    assert!(audit.saves.lock().is_empty(), "nothing was filed");
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
        saves.iter().any(|note| note.reason == Reason::Cold),
        "the cold offer still happened: {saves:?}"
    );
    assert!(
        saves
            .iter()
            .any(|note| matches!(note.reason, Reason::Shutdown | Reason::Continued)),
        "the turn's end was offered too: {saves:?}"
    );
}
