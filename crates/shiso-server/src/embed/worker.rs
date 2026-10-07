//! The embedding thread: take a request, embed each of its inputs, report.
//!
//! The only thread that calls the model. The loop ends when every service
//! handle is dropped, which is when the process is done serving.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use shiso_models::embedder::Embedder;
use tokio::sync::mpsc;

use super::{Queued, refuse};
use crate::error::AppError;

/// The thread body: runs until every service handle is gone.
pub(super) fn run(mut embedder: Box<dyn Embedder>, mut rx: mpsc::Receiver<Queued>) {
    while let Some(Queued { inputs, reply }) = rx.blocking_recv() {
        // A caller that hung up while its request waited is not worth the
        // passes: nothing is listening for the vectors any more.
        if reply.is_closed() {
            continue;
        }
        let began = Instant::now();
        let count = inputs.len();
        // A panic inside the model ends that request, not the thread: the
        // queue holds other callers who did nothing wrong.
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            embedder.embed_batch(&inputs).map_err(refuse)
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
            inputs = count,
            seconds = began.elapsed().as_secs_f64(),
            ok = outcome.is_ok(),
            "embedding request finished"
        );
        // `Err` is a caller that hung up mid-request; there is nowhere left
        // to report to.
        let _gone = reply.send(outcome);
    }
}

/// A pass panicked inside the model; what the caller's 500 keeps as its
/// cause.
#[derive(Debug, thiserror::Error)]
#[error("the embedding model panicked")]
struct EmbedderPanicked;
