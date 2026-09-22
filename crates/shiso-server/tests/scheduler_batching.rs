//! How wide a batch gets, how much prompt one call reads, and in what order.
//!
//! Every assertion here is about the calls the model saw rather than about the
//! answers it gave, because the answers are identical under a scheduler that
//! batches and one that does not. What separates them is the width of a decode
//! step, the size of a prompt quantum, and whether a prompt is allowed to run
//! between two of somebody else's tokens.

mod common;

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt as _;
use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_server::scheduler::{GenerateRequest, SchedulerConfig};

use common::Call;

/// Argmax sampling, so a scripted spike in the logits row is what comes out.
fn argmax() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..SamplingParams::neutral()
    }
}

/// A request over a prompt of `prompt` tokens.
fn request(prompt: usize, max_tokens: u32) -> GenerateRequest {
    GenerateRequest::new(
        Arc::from(vec![TokenId::new(1); prompt]),
        argmax(),
        Seed::new(7),
        max_tokens,
    )
}

/// Quanta small enough that a hundred-token prompt takes several of them.
fn cfg(max_num_seqs: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs,
        mixed_quantum: 8,
        solo_chunk: 64,
        ..SchedulerConfig::new()
    }
}

/// Reads a stream to its end and counts the events.
async fn drain(mut stream: shiso_ir::EventStream) -> usize {
    let mut seen = 0;
    while stream.next().await.is_some() {
        seen += 1;
    }
    seen
}

/// With nobody to starve there is no reason to read the prompt in small
/// pieces, and the widths are the two configured numbers rather than a third
/// one derived by accident.
#[tokio::test]
async fn a_lone_prompt_is_read_at_the_solo_width() {
    let (engine, join, audit) = common::scheduled_engine(vec![10, 11], cfg(2), None);
    drain(
        engine
            .generate(request(100, 4))
            .await
            .expect("the engine accepts the request"),
    )
    .await;

    assert_eq!(
        audit.prefill_chunks(),
        vec![64, 36],
        "a hundred-token prompt is one solo chunk and its remainder"
    );

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// A prompt arriving beside a turn that is already streaming is read in pieces
/// small enough that the streaming turn never waits a whole prompt for its next
/// token.
#[tokio::test]
async fn a_prompt_read_beside_a_decoding_turn_is_read_in_quanta() {
    let gate = common::Gate::closed();
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], cfg(2), Some(Arc::clone(&gate)));

    let streaming = engine
        .generate(request(1, 300))
        .await
        .expect("the engine accepts the streaming request");
    let arriving = tokio::spawn({
        let engine = engine.clone();
        async move { engine.generate(request(100, 4)).await }
    });
    common::wait_for_live(&audit, &gate, 2).await;
    let arriving = arriving
        .await
        .expect("the second request is not cancelled")
        .expect("the engine accepts the second request");

    // Let the prompt be read one quantum per step, so the assertion is about
    // the widths the scheduler chose rather than about how far it got.
    for _ in 0..20 {
        gate.allow(1);
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    let quanta: Vec<usize> = audit
        .calls
        .lock()
        .iter()
        .filter_map(|call| match call {
            Call::Prefill { tokens, .. } => Some(*tokens),
            Call::Decode(_) => None,
        })
        .collect();
    assert!(
        quanta.len() > 2,
        "a hundred-token prompt beside a decoder takes several quanta: {quanta:?}"
    );
    assert!(
        quanta.iter().all(|tokens| *tokens <= 8),
        "no quantum is wider than the shared cap: {quanta:?}"
    );

    gate.open();
    drop(streaming);
    drop(arriving);
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// The fairness rule, read off the call log: once anything is decoding, no two
/// prompt quanta run back to back. A scheduler that read a whole prompt while a
/// turn was streaming would show two adjacent quanta here, and its client would
/// see a gap in the stream.
#[tokio::test]
async fn no_two_prompt_quanta_run_between_a_sequences_decode_steps() {
    let gate = common::Gate::closed();
    let (engine, join, audit) =
        common::scheduled_engine(vec![10; 400], cfg(3), Some(Arc::clone(&gate)));

    let streaming = engine
        .generate(request(1, 300))
        .await
        .expect("the engine accepts the streaming request");
    let queued: Vec<_> = (0..2)
        .map(|_| {
            let engine = engine.clone();
            tokio::spawn(async move { engine.generate(request(100, 4)).await })
        })
        .collect();
    common::wait_for_live(&audit, &gate, 3).await;
    let mut streams = vec![streaming];
    for task in queued {
        streams.push(
            task.await
                .expect("a queued request is not cancelled")
                .expect("the engine accepts it"),
        );
    }
    for _ in 0..40 {
        gate.allow(1);
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    let calls = audit.calls.lock().clone();
    let first_decode = calls
        .iter()
        .position(|call| matches!(call, Call::Decode(_)))
        .expect("something decoded");
    for pair in calls[first_decode..].windows(2) {
        assert!(
            !matches!(
                (pair[0], pair[1]),
                (Call::Prefill { .. }, Call::Prefill { .. })
            ),
            "a decoding sequence gets a step between two prompt quanta: {calls:?}"
        );
    }

    gate.open();
    drop(streams);
    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// The resident bound is what makes a batch's width predictable, and it is
/// enforced by the slot table rather than by a counter beside it - so a third
/// request waits for a seat instead of widening the batch.
#[tokio::test]
async fn no_decode_step_is_wider_than_the_resident_bound() {
    let (engine, join, audit) = common::scheduled_engine(vec![10; 8], cfg(2), None);
    let mut tasks = Vec::new();
    for _ in 0..3 {
        let engine = engine.clone();
        tasks.push(tokio::spawn(async move {
            let stream = engine
                .generate(request(4, 6))
                .await
                .expect("the engine accepts the request");
            drain(stream).await
        }));
    }
    for task in tasks {
        assert!(
            task.await.expect("a turn is not cancelled") > 0,
            "every request produced a turn"
        );
    }

    let widths = audit.decode_widths();
    assert!(
        widths.iter().all(|width| *width <= 2),
        "the resident bound is two, so no step covers three: {widths:?}"
    );

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
    assert_eq!(
        audit.opens.load(std::sync::atomic::Ordering::Relaxed),
        3,
        "all three were served, one after another where they did not fit"
    );
    assert_eq!(audit.closes.load(std::sync::atomic::Ordering::Relaxed), 3);
}
