//! The embedding thread: take the waiting requests, embed their inputs in
//! one pass, report to each.
//!
//! The only thread that calls the model. The loop ends when every service
//! handle is dropped, which is when the process is done serving.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use shiso_models::embedder::{EmbedPart, Embedder, Embedding};
use tokio::sync::mpsc;

use super::{Queued, refuse};
use crate::error::AppError;

/// The most inputs one combined pass gathers from the queue. The model
/// itself closes a pack at its token limit; this only bounds how long the
/// first caller in a burst waits behind the others.
const MAX_GATHERED_INPUTS: usize = 64;

/// The thread body: runs until every service handle is gone.
///
/// Requests waiting together run as one pass: four callers with a short
/// query each cost one pass's launches, not four. The model gives each input
/// the vector it would get alone, so gathering is invisible in the answers.
pub(super) fn run(mut embedder: Box<dyn Embedder>, mut rx: mpsc::Receiver<Queued>) {
    while let Some(first) = rx.blocking_recv() {
        let mut batch = vec![first];
        let mut inputs = batch[0].inputs.len();
        while inputs < MAX_GATHERED_INPUTS {
            match rx.try_recv() {
                Ok(next) => {
                    inputs += next.inputs.len();
                    batch.push(next);
                }
                Err(_) => break,
            }
        }
        // A caller that hung up while its request waited is not worth the
        // passes: nothing is listening for the vectors any more.
        batch.retain(|q| !q.reply.is_closed());
        match batch.len() {
            0 => {}
            1 => {
                let Queued { inputs, reply } = batch.remove(0);
                let _gone = reply.send(embed(embedder.as_mut(), &inputs));
            }
            _ => run_gathered(embedder.as_mut(), batch),
        }
    }
}

/// Several requests as one pass, each answered with its own vectors. A pass
/// that fails is retried request by request, so one caller's bad input is
/// refused alone rather than failing everyone it was gathered with.
fn run_gathered(embedder: &mut dyn Embedder, batch: Vec<Queued>) {
    let mut all = Vec::new();
    let mut callers = Vec::with_capacity(batch.len());
    for Queued { inputs, reply } in batch {
        callers.push((inputs.len(), reply));
        all.extend(inputs);
    }
    if let Ok(vectors) = embed(embedder, &all)
        && vectors.len() == all.len()
    {
        let mut vectors = vectors.into_iter();
        for (count, reply) in callers {
            let _gone = reply.send(Ok(vectors.by_ref().take(count).collect()));
        }
        return;
    }
    let mut all = all.into_iter();
    for (count, reply) in callers {
        let inputs: Vec<_> = all.by_ref().take(count).collect();
        let _gone = reply.send(embed(embedder, &inputs));
    }
}

/// One `embed_batch`, a panic inside the model reported as a 500.
fn embed(
    embedder: &mut dyn Embedder,
    inputs: &[Vec<EmbedPart>],
) -> Result<Vec<Embedding>, AppError> {
    let began = Instant::now();
    // A panic inside the model ends that pass, not the thread: the queue
    // holds other callers who did nothing wrong.
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        embedder.embed_batch(inputs).map_err(refuse)
    }))
    .unwrap_or_else(|payload| {
        let detail = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panicked with non-string payload".to_owned());
        tracing::error!(
            detail = detail.as_str(),
            "an embedding pass panicked; the worker goes on"
        );
        Err(AppError::internal("embedding panicked", EmbedderPanicked))
    });
    tracing::debug!(
        inputs = inputs.len(),
        seconds = began.elapsed().as_secs_f64(),
        ok = outcome.is_ok(),
        "embedding pass finished"
    );
    outcome
}

/// A pass panicked inside the model; what the caller's 500 keeps as its
/// cause.
#[derive(Debug, thiserror::Error)]
#[error("the embedding model panicked")]
struct EmbedderPanicked;
