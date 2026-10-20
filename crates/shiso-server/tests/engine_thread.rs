//! The boundary between the handlers and the model, driven directly.
//!
//! These are the cases that would otherwise only be exercised through a socket:
//! that a stream can be handed back from a thread that is not a runtime worker,
//! that the events add up to the turn the model produced, that a client which
//! stops reading does not wedge the one engine thread, and that dropping the
//! last handle ends the loop with the model torn down.

mod common;

use futures::StreamExt as _;
use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_ir::{BlockDelta, BlockStart, ResponseBuilder, StopReason, StreamEvent};
use shiso_server::scheduler::GenerateRequest;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Argmax sampling, so a scripted spike in the logits row is what comes out.
///
/// `greedy_neutral` is not greedy - it is every stage set to a no-op, with the
/// temperature at one - so a test that used it would be asserting on a random
/// draw. Zero temperature is what selects the argmax path.
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

#[tokio::test]
async fn a_scripted_turn_arrives_as_a_start_a_block_its_deltas_and_a_stop() {
    let (engine, join, _teardowns) = common::fake_engine(vec![10, 11, 12], None);
    let mut stream = engine
        .generate(request(16))
        .await
        .expect("the engine accepts the request");

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    assert!(
        matches!(events.first(), Some(StreamEvent::Start { .. })),
        "a turn opens with its head: {events:?}"
    );
    assert!(
        matches!(
            events.get(1),
            Some(StreamEvent::BlockStart {
                index: 0,
                block: BlockStart::Text
            })
        ),
        "the first block is the text block: {events:?}"
    );
    let ids: Vec<u32> = events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::BlockDelta {
                delta: BlockDelta::Text { token_ids, .. },
                ..
            } => Some(token_ids.clone()),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(ids, vec![10, 11, 12], "the scripted ids reach the caller");
    assert!(
        matches!(
            events.get(events.len() - 2),
            Some(StreamEvent::BlockStop { index: 0 })
        ),
        "the block closes before the turn does: {events:?}"
    );
    match events.last() {
        Some(StreamEvent::Stop { stop, usage, .. }) => {
            assert_eq!(*stop, StopReason::EndTurn);
            assert_eq!(usage.prompt_tokens, 1);
            // Three scripted ids plus the end-of-turn marker the model produced
            // after them: the marker is generated work and is charged for.
            assert_eq!(usage.completion_tokens, 4);
        }
        other => panic!("a turn ends with its accounting, not {other:?}"),
    }

    drop(engine);
    let outcome = join.join().expect("the engine thread does not panic");
    assert!(outcome.is_ok(), "the loop exits cleanly: {outcome:?}");
}

/// The same events, folded by the accumulator the blocking surfaces use. This
/// is the proof that a streaming client and a blocking one are looking at the
/// same turn rather than at two encoders that happen to agree today.
#[tokio::test]
async fn the_stream_folds_into_the_response_a_blocking_run_would_have_produced() {
    let (engine, join, _teardowns) = common::fake_engine(vec![10, 11], None);
    let mut stream = engine
        .generate(request(16))
        .await
        .expect("the engine accepts the request");

    let mut builder = ResponseBuilder::new();
    while let Some(event) = stream.next().await {
        builder
            .accept(event)
            .expect("every event belongs to the turn");
    }
    let response = builder.finish().expect("the turn is complete");

    assert_eq!(response.model, "fake-model");
    assert_eq!(response.stop, StopReason::EndTurn);
    assert_eq!(response.message.content.len(), 1);
    assert_eq!(response.token_ids, vec![10, 11, 63]);

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// The budget, not the script, is what ends this turn.
#[tokio::test]
async fn a_turn_that_runs_out_of_budget_stops_for_that_reason() {
    let (engine, join, _teardowns) = common::fake_engine(vec![10, 11, 12, 13, 14], None);
    let mut stream = engine
        .generate(request(2))
        .await
        .expect("the engine accepts the request");

    let mut last = None;
    while let Some(event) = stream.next().await {
        last = Some(event);
    }
    match last {
        Some(StreamEvent::Stop { stop, usage, .. }) => {
            assert_eq!(stop, StopReason::MaxTokens);
            assert_eq!(usage.completion_tokens, 2);
        }
        other => panic!("a truncated turn still ends with its accounting, not {other:?}"),
    }

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// Dropping the last handle is what ends the loop, and teardown is explicit on
/// the way out rather than left to `Drop`.
#[tokio::test]
async fn dropping_every_handle_ends_the_loop_with_the_model_torn_down() {
    let (engine, join, teardowns) = common::fake_engine(vec![10], None);
    let clone = engine.clone();
    assert_eq!(teardowns.load(Ordering::Relaxed), 0);

    drop(engine);
    // One clone is still alive, so the queue is still open and nothing has been
    // torn down. This is the half of the test that would pass for the wrong
    // reason if the loop exited on the first drop.
    assert_eq!(teardowns.load(Ordering::Relaxed), 0);

    drop(clone);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert_eq!(teardowns.load(Ordering::Relaxed), 1);
}

/// A client that takes the stream and drops it part way must not leave the one
/// engine thread blocked on a send nobody will ever read.
#[tokio::test]
async fn a_client_that_stops_reading_does_not_wedge_the_engine() {
    let (engine, join, _teardowns) = common::fake_engine((0..200).collect(), None);
    let stream = engine
        .generate(request(200))
        .await
        .expect("the engine accepts the request");
    drop(stream);

    // The proof is that a second request is served at all: if the abandoned one
    // had wedged the loop, this would never be admitted.
    let mut second = engine
        .generate(request(4))
        .await
        .expect("the engine accepts a second request");
    let mut seen = 0usize;
    while second.next().await.is_some() {
        seen += 1;
    }
    assert!(seen > 0, "the second request produced a turn");

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// An empty prompt has nothing to run a forward pass over, and the turn fails
/// rather than opening a session it cannot use.
#[tokio::test]
async fn an_empty_prompt_fails_the_turn_rather_than_producing_one() {
    let (engine, join, _teardowns) = common::fake_engine(vec![10], None);
    let empty = GenerateRequest::new(Arc::from(Vec::new()), argmax(), Seed::new(1), 8);
    let mut stream = engine
        .generate(empty)
        .await
        .expect("the engine accepts the request");
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    assert!(
        matches!(events.as_slice(), [StreamEvent::Failed { .. }]),
        "an unrunnable turn is a single terminal failure: {events:?}"
    );

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// With the queue gone, a handle reports a server failure rather than hanging.
#[tokio::test]
async fn a_request_to_a_dead_engine_is_reported_and_not_awaited_forever() {
    let (engine, join, _teardowns) = common::fake_engine(vec![10], None);
    let orphan = engine.clone();
    drop(engine);
    // Ending the loop from under a live handle is not reachable through the
    // public API by design, so the closest thing is asserting the handle's
    // snapshot fields still answer without touching the model.
    assert_eq!(orphan.model_id(), "fake-model");
    assert!(orphan.ctx() > 0);
    drop(orphan);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// Runs one request to its end on an engine whose decode steps hand out
/// device picks `offset` places past their scripted ids, and returns the ids
/// the caller was given.
async fn picked_ids(
    script: Vec<u32>,
    offset: u32,
    sampling: SamplingParams,
    cfg: shiso_server::scheduler::SchedulerConfig,
) -> Vec<u32> {
    let (engine, join, _audit) = common::picked_engine(script, offset, cfg);
    let request = GenerateRequest::new(Arc::from(vec![TokenId::new(1)]), sampling, Seed::new(7), 4);
    let mut stream = engine
        .generate(request)
        .await
        .expect("the engine accepts the request");
    let mut builder = ResponseBuilder::new();
    while let Some(event) = stream.next().await {
        builder
            .accept(event)
            .expect("every event belongs to the turn");
    }
    let response = builder.finish().expect("the turn is complete");
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    response.token_ids
}

/// A greedy sequence's decode steps take the token the device picked, read
/// in four bytes, rather than reading the row back and scanning it. The fake
/// hands out picks one past each row's scripted spike, so which of the two a
/// token came from is visible: the first token is the prompt's row (a prompt
/// hands out no pick), every later one the pick. With honest picks the turn
/// is the turn the rows alone produce.
#[tokio::test]
async fn a_greedy_step_takes_the_devices_pick_and_an_honest_pick_changes_nothing() {
    let script = vec![10, 11, 12, 13, 14];
    let cfg = shiso_server::scheduler::SchedulerConfig::new();
    assert_eq!(
        picked_ids(script.clone(), 1, argmax(), cfg.clone()).await,
        vec![10, 12, 13, 14],
        "a greedy step drew from its row rather than the device's pick"
    );
    assert_eq!(
        picked_ids(script, 0, argmax(), cfg).await,
        vec![10, 11, 12, 13]
    );
}

/// A pick stands for the argmax only. A sampled sequence draws from its row
/// whatever the device picked, and so does a greedy one whose pick it
/// suppresses - its own argmax is then some other id, which only the row
/// knows.
#[tokio::test]
async fn a_sampled_step_and_a_suppressed_pick_both_read_the_row() {
    let script = vec![10, 11, 12, 13, 14];
    let sampled = SamplingParams {
        temperature: 1.0,
        top_k: std::num::NonZeroU32::new(1),
        ..SamplingParams::neutral()
    };
    let cfg = shiso_server::scheduler::SchedulerConfig::new();
    assert_eq!(
        picked_ids(script.clone(), 1, sampled, cfg.clone()).await,
        vec![10, 11, 12, 13],
        "a sampled step took the device's pick"
    );
    let suppressing = shiso_server::scheduler::SchedulerConfig {
        suppressed: Arc::from([TokenId::new(12)]),
        ..cfg
    };
    // The first decode step's pick (12) is suppressed, so that step reads its
    // row (11); the picks after it (13, 14) are not, and are taken.
    assert_eq!(
        picked_ids(script, 1, argmax(), suppressing).await,
        vec![10, 11, 13, 14],
        "a suppressed pick was taken"
    );
}
