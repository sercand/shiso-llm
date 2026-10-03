//! The image thread: take a job, run it to its end, report.
//!
//! One generation at a time, in arrival order: a denoising step saturates the
//! device, so interleaving two would make both slower and double the peak.
//! The loop ends when every service handle is dropped, which is when the
//! process is done serving.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use shiso_vision::generator::ImageStepEvent;
use tokio::sync::mpsc;

use super::{GeneratorPanicked, ImageServiceDesc, Queued, refuse};
use crate::error::AppError;

/// The thread body: runs until every service handle is gone.
pub(super) fn run_worker(desc: &ImageServiceDesc, mut rx: mpsc::Receiver<Queued>) {
    while let Some(request) = rx.blocking_recv() {
        let Queued {
            job,
            progress,
            reply,
        } = request;
        // A caller that hung up while its job waited is not worth a full
        // generation: nothing is listening for the answer any more.
        if reply.is_closed() {
            continue;
        }
        let began = Instant::now();
        let emit = progress.as_ref().map(|sink| {
            move |event: ImageStepEvent| {
                // A streaming caller that stopped reading does not stop the
                // generation; its final answer still goes to its reply.
                let _gone = sink.send(event);
            }
        });
        // Coerced at a binding rather than with `as`: the cast form is what
        // the compiler calls a trivial cast and warns about.
        let progress_cb: Option<&dyn Fn(ImageStepEvent)> = match &emit {
            Some(emit) => Some(emit),
            None => None,
        };
        // A panic inside the generator ends that job, not the thread: the
        // queue holds other callers who did nothing wrong.
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            desc.generator.generate(&job, progress_cb)
        }));
        let outcome = match outcome {
            Ok(Ok(image)) => Ok(image),
            Ok(Err(error)) => Err(refuse(error)),
            Err(payload) => {
                // The payload is whatever `panic!` was handed; the two
                // string forms are almost all of it, and the named fallback
                // keeps a non-string payload (a `panic_any` of some struct,
                // say) from erasing the fact of the panic from the log.
                let detail = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panicked with non-string payload".to_owned());
                tracing::error!(
                    detail = detail.as_str(),
                    "an image generation panicked; the worker goes on"
                );
                Err(AppError::internal(
                    "image generation panicked",
                    GeneratorPanicked,
                ))
            }
        };
        tracing::info!(
            seconds = began.elapsed().as_secs_f64(),
            ok = outcome.is_ok(),
            "image generation finished"
        );
        // `Err` here is a caller that hung up mid-generation; there is
        // nowhere left to report to.
        let _gone = reply.send(outcome);
    }
}
