//! What a caller is told when the server has more work than it can hold.
//!
//! The alternative to refusing is waiting, and waiting without a bound turns
//! overload into latency a client cannot tell from a hung server: it sees no
//! bytes either way, and its own timeout fires with nothing to report but
//! "nothing happened". A bounded wait followed by a refusal is a different
//! answer - it names the condition, it arrives quickly, and it is a status
//! every client library already knows how to back off from.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_server::scheduler::{GenerateRequest, SchedulerConfig};

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
async fn a_caller_that_cannot_be_queued_is_refused_rather_than_left_waiting() {
    let gate = common::Gate::closed();
    let cfg = SchedulerConfig {
        // One seat and one place in the queue, so the third caller has nowhere
        // to go. The same thing happens at four and two hundred and fifty six;
        // small numbers are what make it happen in a test.
        max_num_seqs: 1,
        max_queued: 1,
        queue_admission: Duration::from_millis(120),
        ..SchedulerConfig::new()
    };
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], cfg, Arc::clone(&gate).into());

    // The first takes the seat and the coordinator then blocks inside the
    // model, which is what a real server under load looks like from here.
    let running = engine
        .generate(request(300))
        .await
        .expect("the engine accepts the first request");
    common::wait_for_live(&audit, &gate, 1).await;

    // The second takes the one place in the queue. Its own admission never
    // answers, because the seat is not free.
    let queued = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(request(4)).await }
    });
    common::wait_for_queued(&engine, 2).await;

    let started = Instant::now();
    // `expect_err` is not available here: the success arm is a boxed stream
    // with no `Debug`, and a test that unwrapped it would be asserting on a
    // type rather than on the refusal.
    let Err(refused) = engine.generate(request(4)).await else {
        panic!("the third caller had nowhere to wait and was admitted anyway");
    };
    let waited = started.elapsed();

    assert_eq!(
        refused.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "an overloaded server says so with a status a client can act on: {refused}"
    );
    assert!(
        refused.to_string().contains("overloaded"),
        "the message names the condition: {refused}"
    );
    assert!(
        waited < Duration::from_secs(5),
        "the refusal arrives on the admission window, not on some other timeout: {waited:?}"
    );
    assert_eq!(engine.stats().rejected(), 1);

    gate.open();
    drop(running);
    drop(engine);
    let _ = queued.await.expect("the queued request is not cancelled");
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}
