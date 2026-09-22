//! A request that outlives its deadline, and the word the client is told.
//!
//! Time here comes from a clock the test moves, not from the clock on the wall.
//! A deadline test that slept would be either slow or flaky depending on how
//! long it slept, and the thing under test is the policy - which reason a cut
//! turn reports - rather than how fast the machine ran the loop.
//!
//! The reason matters more than it looks. A deadline reported as "length" is
//! indistinguishable to a client from a truncated answer, and the two call for
//! opposite responses: a truncated answer can be continued, a timed-out one
//! should be retried or asked for more cheaply.

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use futures::StreamExt as _;
use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_ir::{StopReason, StreamEvent};
use shiso_server::scheduler::{FakeClock, GenerateRequest, SchedulerConfig};

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

#[tokio::test]
async fn a_request_past_its_deadline_is_cut_and_says_so() {
    let clock = FakeClock::new();
    let gate = common::Gate::closed();
    let cfg = SchedulerConfig {
        max_num_seqs: 2,
        request_timeout: Some(Duration::from_secs(30)),
        ..SchedulerConfig::new()
    }
    .with_clock(Arc::clone(&clock));
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], cfg, Arc::clone(&gate).into());

    let mut stream = engine
        .generate(request(300))
        .await
        .expect("the engine accepts the request");
    common::wait_for_live(&audit, &gate, 1).await;

    // The turn is mid-flight and the deadline has not passed. Moving the clock
    // is what makes it pass; nothing sleeps.
    clock.advance(Duration::from_secs(31));
    gate.open();

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    match events.last() {
        Some(StreamEvent::Stop { stop, .. }) => assert_eq!(
            *stop,
            StopReason::Timeout,
            "a deadline is reported as a deadline, never as an exhausted budget"
        ),
        other => panic!("a cut turn still ends with its accounting, not {other:?}"),
    }
    assert_eq!(engine.stats().timeouts(), 1);

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert_eq!(
        audit.closes.load(Ordering::Relaxed),
        1,
        "a cut turn releases its session like any other"
    );
}

/// A server with no deadline never cuts anything, however long a turn runs.
/// That is what makes the deadline opt-in rather than a default nobody chose.
#[tokio::test]
async fn a_server_with_no_deadline_lets_a_turn_run_to_its_own_end() {
    let clock = FakeClock::new();
    let cfg = SchedulerConfig {
        request_timeout: None,
        ..SchedulerConfig::new()
    }
    .with_clock(Arc::clone(&clock));
    let (engine, join, _audit) = common::scheduled_engine(vec![10, 11, 12], cfg, None);

    // Years pass. Nothing is cut, because nothing has a deadline to miss.
    // `Duration::from_days` is unstable on the pinned 1.98.1 toolchain.
    #[expect(
        clippy::duration_suboptimal_units,
        reason = "Duration::from_days is unstable on the pinned toolchain"
    )]
    let a_year = Duration::from_secs(60 * 60 * 24 * 365);
    clock.advance(a_year);
    let mut stream = engine
        .generate(request(16))
        .await
        .expect("the engine accepts the request");
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    match events.last() {
        Some(StreamEvent::Stop { stop, .. }) => assert_eq!(*stop, StopReason::EndTurn),
        other => panic!("a turn ends with its accounting, not {other:?}"),
    }
    assert_eq!(engine.stats().timeouts(), 0);

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}
