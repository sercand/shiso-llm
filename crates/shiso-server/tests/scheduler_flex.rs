//! The flex class: served only while no standard request has work.
//!
//! Every assertion is read off what the model was asked to do - how wide a
//! decode step was, how many sessions were opened - rather than off timing,
//! because the claim is about who gets the device, and the device is the fake
//! model's call log. A flex row that ran beside a standard one shows up as a
//! decode step two rows wide; a flex request admitted too early shows up as a
//! session opened too early.

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::http::StatusCode;
use futures::StreamExt as _;
use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_ir::{ServiceClass, StopReason, StreamEvent};
use shiso_server::scheduler::{GenerateRequest, SchedulerConfig};

use common::Gate;

fn argmax() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::neutral()
    }
}

/// A standard request over a one-token prompt.
fn standard(max_tokens: u32) -> GenerateRequest {
    GenerateRequest::new(
        Arc::from(vec![TokenId::new(1)]),
        argmax(),
        Seed::new(7),
        max_tokens,
    )
}

/// The same request in the flex class.
fn flex(max_tokens: u32) -> GenerateRequest {
    standard(max_tokens).with_service_class(ServiceClass::Flex)
}

fn cfg(max_num_seqs: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs,
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

/// The tokens a finished turn was charged for, and why it stopped.
fn ending(events: &[StreamEvent]) -> (StopReason, u64) {
    match events.last() {
        Some(StreamEvent::Stop { stop, usage, .. }) => (stop.clone(), usage.completion_tokens),
        other => panic!("a turn ends with its accounting, not {other:?}"),
    }
}

/// Lets the coordinator take one decode step at a time until `done` holds.
///
/// The coordinator only pulls the queue between steps, so a request queued
/// while it is held in the gate is answered only once a step is allowed.
async fn drive_until(gate: &Gate, mut done: impl FnMut() -> bool) {
    for _ in 0..2_000 {
        if done() {
            return;
        }
        gate.allow(1);
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("the coordinator never reached the state the test waited for");
}

/// Lets `steps` more decode steps through, one at a time.
async fn step(gate: &Gate, steps: usize) {
    for _ in 0..steps {
        gate.allow(1);
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// The pause: a flex answer that is already streaming gets no decode row while
/// a standard one is resident, and picks up where it stood once it has gone.
#[tokio::test]
async fn a_streaming_flex_answer_pauses_for_a_standard_one_and_then_finishes() {
    let gate = Gate::closed();
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], cfg(2), Some(Arc::clone(&gate)));

    let flex_stream = engine
        .generate(flex(40))
        .await
        .expect("the engine accepts the flex request");
    common::wait_for_live(&audit, &gate, 1).await;
    step(&gate, 3).await;

    let arriving = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(standard(8)).await }
    });
    common::wait_for_live(&audit, &gate, 2).await;
    let standard_stream = arriving
        .await
        .expect("the standard request is not cancelled")
        .expect("the engine accepts the standard request");

    gate.open();
    let standard_events = drain(standard_stream).await;
    assert_eq!(ending(&standard_events), (StopReason::MaxTokens, 8));
    let flex_events = drain(flex_stream).await;
    assert_eq!(
        ending(&flex_events),
        (StopReason::MaxTokens, 40),
        "the paused flex answer resumed and ran to its budget"
    );
    assert_eq!(
        audit.widest_batch(),
        1,
        "both were resident at once and no step ever carried both: {:?}",
        audit.decode_widths()
    );

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// A flex request that arrives while standard work is resident gets its
/// stream at once and no session until the standard work has gone.
#[tokio::test]
async fn a_flex_request_is_held_while_standard_work_is_resident() {
    let gate = Gate::closed();
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], cfg(4), Some(Arc::clone(&gate)));

    let standard_stream = engine
        .generate(standard(200))
        .await
        .expect("the engine accepts the standard request");
    common::wait_for_live(&audit, &gate, 1).await;

    let queued = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(flex(4)).await }
    });
    drive_until(&gate, || queued.is_finished()).await;
    let flex_stream = queued
        .await
        .expect("the flex request is not cancelled")
        .expect("a held flex request still gets its stream");
    step(&gate, 5).await;
    assert_eq!(
        audit.opens.load(Ordering::Relaxed),
        1,
        "a free slot is not enough: the flex request waits for the standard one"
    );

    drop(standard_stream);
    gate.open();
    let flex_events = drain(flex_stream).await;
    assert_eq!(ending(&flex_events), (StopReason::MaxTokens, 4));
    assert_eq!(audit.opens.load(Ordering::Relaxed), 2);

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// A held flex request never stands in front of a standard one: a standard
/// request that came off the queue behind it is admitted past it.
#[tokio::test]
async fn a_standard_request_behind_a_held_flex_one_is_admitted_first() {
    let gate = Gate::closed();
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], cfg(4), Some(Arc::clone(&gate)));

    let first = engine
        .generate(standard(200))
        .await
        .expect("the engine accepts the first standard request");
    common::wait_for_live(&audit, &gate, 1).await;

    let held = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(flex(4)).await }
    });
    drive_until(&gate, || held.is_finished()).await;
    let flex_stream = held
        .await
        .expect("the flex request is not cancelled")
        .expect("a held flex request still gets its stream");

    let behind = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(standard(200)).await }
    });
    common::wait_for_live(&audit, &gate, 2).await;
    let second = behind
        .await
        .expect("the second standard request is not cancelled")
        .expect("the engine accepts the second standard request");
    assert_eq!(
        audit.opens.load(Ordering::Relaxed),
        2,
        "the two resident sessions are the two standard requests"
    );

    drop(first);
    drop(second);
    gate.open();
    let flex_events = drain(flex_stream).await;
    assert_eq!(ending(&flex_events), (StopReason::MaxTokens, 4));
    assert_eq!(audit.opens.load(Ordering::Relaxed), 3);

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// Flex answers never take the last slot, so a standard request arriving
/// while flex work fills everything else is admitted at once rather than
/// waiting for a flex answer to finish.
#[tokio::test]
async fn flex_work_leaves_a_slot_for_a_standard_request() {
    let gate = Gate::closed();
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], cfg(2), Some(Arc::clone(&gate)));

    let first = engine
        .generate(flex(200))
        .await
        .expect("the engine accepts the first flex request");
    common::wait_for_live(&audit, &gate, 1).await;

    let queued = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(flex(200)).await }
    });
    drive_until(&gate, || queued.is_finished()).await;
    let second = queued
        .await
        .expect("the second flex request is not cancelled")
        .expect("a held flex request still gets its stream");
    step(&gate, 5).await;
    assert_eq!(
        audit.live.lock().len(),
        1,
        "with two slots, one flex answer is the most that may be resident"
    );

    let arriving = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(standard(4)).await }
    });
    common::wait_for_live(&audit, &gate, 2).await;
    let standard_stream = arriving
        .await
        .expect("the standard request is not cancelled")
        .expect("the engine accepts the standard request");

    gate.open();
    let standard_events = drain(standard_stream).await;
    assert_eq!(ending(&standard_events), (StopReason::MaxTokens, 4));

    drop(first);
    drop(second);
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// A held flex request whose caller went away is let go without ever
/// opening a session.
#[tokio::test]
async fn an_abandoned_held_flex_request_never_opens_a_session() {
    let gate = Gate::closed();
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], cfg(4), Some(Arc::clone(&gate)));

    let standard_stream = engine
        .generate(standard(200))
        .await
        .expect("the engine accepts the standard request");
    common::wait_for_live(&audit, &gate, 1).await;

    let held = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(flex(4)).await }
    });
    drive_until(&gate, || held.is_finished()).await;
    drop(
        held.await
            .expect("the flex request is not cancelled")
            .expect("a held flex request still gets its stream"),
    );

    drop(standard_stream);
    gate.open();
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert_eq!(
        audit.opens.load(Ordering::Relaxed),
        1,
        "only the standard request ever had a session"
    );
}

/// A full queue refuses a standard caller once its admission window passes,
/// and makes a flex caller wait for room instead.
#[tokio::test]
async fn a_full_queue_makes_a_flex_caller_wait_rather_than_refusing_it() {
    let gate = Gate::closed();
    let cfg = SchedulerConfig {
        max_num_seqs: 1,
        max_queued: 1,
        queue_admission: Duration::from_millis(50),
        ..SchedulerConfig::new()
    };
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], cfg, Some(Arc::clone(&gate)));

    let running = engine
        .generate(standard(300))
        .await
        .expect("the engine accepts the first request");
    common::wait_for_live(&audit, &gate, 1).await;
    let queued = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(standard(4)).await }
    });
    common::wait_for_queued(&engine, 2).await;

    let waiting = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(flex(4)).await }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        !waiting.is_finished(),
        "six admission windows later the flex caller is still waiting, not refused"
    );
    let Err(refused) = engine.generate(standard(4)).await else {
        panic!("a standard caller with nowhere to wait was admitted anyway");
    };
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);

    drop(running);
    gate.open();
    let queued = queued
        .await
        .expect("the queued request is not cancelled")
        .expect("the engine accepts the queued request");
    drain(queued).await;
    let flex_stream = waiting
        .await
        .expect("the flex request is not cancelled")
        .expect("the flex request is admitted once there is room");
    let flex_events = drain(flex_stream).await;
    assert_eq!(ending(&flex_events), (StopReason::MaxTokens, 4));
    assert_eq!(engine.stats().rejected(), 1, "only the standard caller");

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}
