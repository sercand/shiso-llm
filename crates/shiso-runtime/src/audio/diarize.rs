//! The diarizer's thread: whole-clip passes, advanced round robin, beside the
//! recognizer's thread.
//!
//! A diarized job needs its speakers only once its words are known, so its
//! pass runs here, over a backend instance of its own, while the job is
//! recognised. The recognizer's decode is bound by round trips to the device,
//! and the diarizer's encoder fills the device between them; on one thread the
//! two would take turns and the clip would pay for both.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc as std_mpsc;

use shiso_audio::transcript::SpeakerActivity;
use shiso_audio::{AudioError, DiarizationPass, Diarizer, MelFrames};
use tokio::sync::mpsc;

use super::TranscriptEvent;

/// What a finished pass answers: the activity, or why there is none. A reply
/// channel closed without an answer means the pass panicked.
pub(super) type Reply = Result<SpeakerActivity, AudioError>;

/// One clip to diarize.
pub(super) struct Request {
    /// The clip's mel rows, at the diarizer's own spec.
    pub(super) mel: MelFrames,
    /// The job's events, watched only for closing: a caller that has gone
    /// ends the pass at its next step.
    pub(super) events: mpsc::UnboundedSender<TranscriptEvent>,
    /// Where the answer goes.
    pub(super) reply: std_mpsc::Sender<Reply>,
}

struct Running {
    pass: Box<dyn DiarizationPass>,
    events: mpsc::UnboundedSender<TranscriptEvent>,
    reply: std_mpsc::Sender<Reply>,
    /// When the pass began, for the debug line its answer logs.
    began: std::time::Instant,
}

/// The thread body: runs until the recognizer's thread drops its sender.
pub(super) fn run(diarizer: &dyn Diarizer, rx: &std_mpsc::Receiver<Request>) {
    let mut running: Vec<Running> = Vec::new();
    loop {
        if running.is_empty() {
            let Ok(request) = rx.recv() else {
                return;
            };
            begin(diarizer, request, &mut running);
        }
        while let Ok(request) = rx.try_recv() {
            begin(diarizer, request, &mut running);
        }
        // One step of each pass per round, so a short clip queued behind a
        // long one is not held for the whole of it.
        let mut i = 0;
        while i < running.len() {
            if running[i].events.is_closed() {
                running.swap_remove(i);
                continue;
            }
            let stepped = catch_unwind(AssertUnwindSafe(|| running[i].pass.step()));
            match stepped {
                Ok(Ok(None)) => i += 1,
                Ok(Ok(Some(activity))) => {
                    let done = running.swap_remove(i);
                    tracing::debug!(
                        pass_ms = done.began.elapsed().as_secs_f64() * 1e3,
                        "a diarization pass answered"
                    );
                    let _gone = done.reply.send(Ok(activity));
                }
                Ok(Err(error)) => {
                    let _gone = running.swap_remove(i).reply.send(Err(error));
                }
                Err(_panic) => {
                    // Dropping the reply unanswered is how the job learns.
                    tracing::error!("a diarization pass panicked; its job fails alone");
                    running.swap_remove(i);
                }
            }
        }
    }
}

fn begin(diarizer: &dyn Diarizer, request: Request, running: &mut Vec<Running>) {
    let Request { mel, events, reply } = request;
    if events.is_closed() {
        return;
    }
    match catch_unwind(AssertUnwindSafe(|| diarizer.begin(mel))) {
        Ok(Ok(pass)) => running.push(Running {
            pass,
            events,
            reply,
            began: std::time::Instant::now(),
        }),
        Ok(Err(error)) => {
            let _gone = reply.send(Err(error));
        }
        Err(_panic) => {
            tracing::error!("starting a diarization pass panicked; its job fails alone");
        }
    }
}
