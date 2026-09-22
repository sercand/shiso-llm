//! Two turns at once, on one thread, sharing one forward pass.
//!
//! This is the claim the whole scheduler exists to make, and it is the one that
//! cannot be checked by looking at a single request: a loop that ran two
//! requests one after the other would produce exactly the same two streams. So
//! what is asserted here is the *shape of the calls* the model saw - one decode
//! step covering both sequences - and the ownership that has to survive it: a
//! session opened for each, closed once each, and the model torn down once.

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use futures::StreamExt as _;
use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_ir::{StopReason, StreamEvent};
use shiso_server::scheduler::{GenerateRequest, SchedulerConfig};

/// Argmax sampling, so a scripted spike in the logits row is what comes out.
fn argmax() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::neutral()
    }
}

/// A request for `max_tokens` tokens from a one-token prompt.
fn request(max_tokens: u32) -> GenerateRequest {
    GenerateRequest::new(
        Arc::from(vec![TokenId::new(1)]),
        argmax(),
        Seed::new(7),
        max_tokens,
    )
}

/// A configuration that holds two sequences and reads prompt in small quanta.
fn two_seats() -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs: 2,
        mixed_quantum: 8,
        solo_chunk: 64,
        ..SchedulerConfig::new()
    }
}

/// Reads a stream to its end.
async fn drain(mut stream: shiso_ir::EventStream) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

#[tokio::test]
async fn two_concurrent_turns_share_one_decode_step() {
    let gate = common::Gate::closed();
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], two_seats(), Some(Arc::clone(&gate)));

    let first = engine
        .generate(request(300))
        .await
        .expect("the engine accepts the first request");
    let second = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(request(300)).await }
    });
    common::wait_for_live(&audit, &gate, 2).await;
    let second = second
        .await
        .expect("the second request is not cancelled")
        .expect("the engine accepts the second request");

    gate.open();
    let (first, second) = tokio::join!(drain(first), drain(second));

    assert!(
        audit.widest_batch() >= 2,
        "two resident sequences share a forward pass: widths were {:?}",
        audit.decode_widths()
    );
    for events in [&first, &second] {
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::Stop { .. }))
                .count(),
            1,
            "a turn ends exactly once: {events:?}"
        );
    }

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert_eq!(audit.opens.load(Ordering::Relaxed), 2);
    assert_eq!(
        audit.closes.load(Ordering::Relaxed),
        2,
        "every session the coordinator opened it also closed"
    );
    assert!(
        audit.live.lock().is_empty(),
        "nothing is still holding a session"
    );
    assert_eq!(audit.teardowns.load(Ordering::Relaxed), 1);
}

/// Both sequences have to move every tick. A scheduler that admitted two and
/// then ran them to completion one at a time would pass the batch-width
/// assertion above the moment the two happened to overlap for one step, and
/// would still be serving the second request's client nothing for the whole of
/// the first request's turn.
#[tokio::test]
async fn neither_turn_waits_for_the_other_to_finish() {
    let gate = common::Gate::closed();
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], two_seats(), Some(Arc::clone(&gate)));

    let first = engine
        .generate(request(300))
        .await
        .expect("the engine accepts the first request");
    let second = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(request(300)).await }
    });
    common::wait_for_live(&audit, &gate, 2).await;
    let second = second
        .await
        .expect("the second request is not cancelled")
        .expect("the engine accepts the second request");

    // One step at a time from here, so the assertion is about the scheduler's
    // interleaving rather than about which thread the operating system
    // happened to run.
    for _ in 0..8 {
        gate.allow(1);
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let widths = audit.decode_widths();
    assert!(
        widths.iter().filter(|width| **width == 2).count() >= 4,
        "both sequences advance together, step after step: {widths:?}"
    );

    gate.open();
    drop(first);
    drop(second);
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert_eq!(
        audit.opens.load(Ordering::Relaxed),
        audit.closes.load(Ordering::Relaxed),
        "abandoning both streams still releases both sessions"
    );
}

/// One request is still one request: the batching machinery must not change
/// what a single turn produces, or every existing wire test is asserting on
/// something the server no longer does.
#[tokio::test]
async fn a_single_turn_is_unchanged_by_the_machinery_that_batches() {
    let (engine, join, audit) = common::scheduled_engine(vec![10, 11, 12], two_seats(), None);
    let events = drain(
        engine
            .generate(request(16))
            .await
            .expect("the engine accepts the request"),
    )
    .await;

    match events.last() {
        Some(StreamEvent::Stop { stop, usage, .. }) => {
            assert_eq!(*stop, StopReason::EndTurn);
            assert_eq!(usage.prompt_tokens, 1);
            assert_eq!(usage.completion_tokens, 4, "three ids and the marker");
        }
        other => panic!("a turn ends with its accounting, not {other:?}"),
    }
    assert_eq!(
        audit.decode_widths(),
        vec![1, 1, 1],
        "one resident sequence is a batch of one, every step"
    );

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}
