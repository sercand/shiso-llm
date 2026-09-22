//! A client that gives up, and what the server does with what it was holding.
//!
//! The property being pinned is an ownership one. A client cannot free a
//! session - it has no way to name one, and a request that could would be a
//! request that frees device state in the middle of somebody else's forward
//! pass. What a client can do is stop reading, and the coordinator discovers
//! that on its next send and retires the sequence itself, at a step boundary,
//! on the one thread that owns the model.

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use futures::StreamExt as _;
use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_ir::StreamEvent;
use shiso_server::scheduler::{GenerateRequest, SchedulerConfig};

/// Argmax sampling, so a scripted spike in the logits row is what comes out.
fn argmax() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::neutral()
    }
}

fn request(max_tokens: u32) -> GenerateRequest {
    GenerateRequest::new(
        Arc::from(vec![TokenId::new(1)]),
        argmax(),
        Seed::new(7),
        max_tokens,
    )
}

fn two_seats() -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs: 2,
        ..SchedulerConfig::new()
    }
}

/// A client that hangs up mid-turn loses its session - and only its own.
#[tokio::test]
async fn a_client_that_hangs_up_has_its_session_released_by_the_worker() {
    let gate = common::Gate::closed();
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], two_seats(), Arc::clone(&gate).into());

    let abandoned = engine
        .generate(request(300))
        .await
        .expect("the engine accepts the first request");
    let kept = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(request(6)).await }
    });
    common::wait_for_live(&audit, &gate, 2).await;
    let kept = kept
        .await
        .expect("the second request is not cancelled")
        .expect("the engine accepts the second request");
    // Both prompts have been read, so both turns have genuinely started; a
    // stream dropped before that is a different case, and the test would be
    // testing it on whichever runs it lost the race.
    common::wait_for_calls(&audit, 2).await;

    // The hang-up. Nothing tells the engine; the next send is what discovers it.
    drop(abandoned);
    gate.open();

    let mut events = Vec::new();
    let mut kept = kept;
    while let Some(event) = kept.next().await {
        events.push(event);
    }
    assert!(
        matches!(events.last(), Some(StreamEvent::Stop { .. })),
        "the sequence that kept reading finished its turn: {events:?}"
    );

    // Both sessions are released by the coordinator. The abandoned one is
    // released because the coordinator noticed, not because the client asked.
    for _ in 0..2_000 {
        if audit.closes.load(Ordering::Relaxed) >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(audit.opens.load(Ordering::Relaxed), 2);
    assert_eq!(
        audit.closes.load(Ordering::Relaxed),
        2,
        "the abandoned sequence's session was released too"
    );
    assert_eq!(
        engine.stats().cancellations(),
        1,
        "exactly one turn was counted as abandoned, not one per dropped event"
    );

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert!(audit.live.lock().is_empty());
}

/// A turn nobody ever read is still a turn the server has to unwind: the
/// session is opened before the first token exists, so abandoning the stream
/// immediately is the case where a leak would be invisible.
#[tokio::test]
async fn a_stream_dropped_before_it_is_read_still_frees_its_session() {
    let (engine, join, audit) = common::scheduled_engine(vec![10; 200], two_seats(), None);
    let stream = engine
        .generate(request(200))
        .await
        .expect("the engine accepts the request");
    drop(stream);

    // The proof that the loop is not wedged: a second request is served at all.
    let mut second = engine
        .generate(request(4))
        .await
        .expect("the engine accepts a second request");
    let mut seen = 0;
    while second.next().await.is_some() {
        seen += 1;
    }
    assert!(seen > 0, "the second request produced a turn");

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert_eq!(
        audit.opens.load(Ordering::Relaxed),
        audit.closes.load(Ordering::Relaxed),
        "every session opened was closed"
    );
    assert_eq!(audit.teardowns.load(Ordering::Relaxed), 1);
}

/// Shutdown with a sequence still running ends it rather than leaving its
/// client waiting on a stream that will never say anything again.
#[tokio::test]
async fn a_shutdown_ends_every_live_turn_and_releases_its_session() {
    let gate = common::Gate::closed();
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], two_seats(), Arc::clone(&gate).into());

    let mut live = engine
        .generate(request(300))
        .await
        .expect("the engine accepts the request");
    common::wait_for_live(&audit, &gate, 1).await;

    // Dropping the last handle closes the queue. The coordinator keeps going
    // until the sequences it already admitted have ended, which is why the gate
    // has to open for the loop to finish at all.
    drop(engine);
    gate.open();

    let mut events = Vec::new();
    while let Some(event) = live.next().await {
        events.push(event);
    }
    assert!(
        matches!(events.last(), Some(StreamEvent::Stop { .. })),
        "a turn that was running when the server stopped still ends: {events:?}"
    );

    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert_eq!(audit.opens.load(Ordering::Relaxed), 1);
    assert_eq!(audit.closes.load(Ordering::Relaxed), 1);
    assert_eq!(audit.teardowns.load(Ordering::Relaxed), 1);
}
