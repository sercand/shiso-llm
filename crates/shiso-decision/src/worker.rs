//! The worker thread that owns a model, and the [`Engine`] it runs requests on.

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;

use futures::channel::oneshot;
use shiso_core::{
    Backend, CoreResult, DecisionLayout, DecisionPrompt, DecisionReadout, Model, SessionId, TokenId,
};

use crate::error::DecideError;
use crate::load::PromptTokens;
use crate::math::label_logprobs;
use crate::schema::{Answer, DecisionRequest};
use crate::score::{RunScore, ScoreRun, Scorer, decide_with_head, decide_with_readout};

/// What the worker runs a request on. The seam a test puts a fake behind.
pub(crate) trait Engine: Send {
    /// Answers one request.
    fn decide(&mut self, req: &DecisionRequest) -> Result<Vec<Answer>, DecideError>;

    /// Releases the model; the worker calls it once, as it stops.
    fn close(&mut self);
}

/// A message to the worker.
#[derive(Debug)]
pub(crate) enum Msg {
    /// A request to answer.
    Job(Job),
    /// Leave the loop: queued behind everything already asked.
    Stop,
}

/// A request and where its answer goes.
#[derive(Debug)]
pub(crate) struct Job {
    pub(crate) req: DecisionRequest,
    pub(crate) reply: oneshot::Sender<Result<Vec<Answer>, DecideError>>,
}

/// The error of a request the `Decider` was dropped under.
fn closed() -> DecideError {
    DecideError::Engine("the decider was dropped before it answered".to_owned())
}

/// The worker's loop: one request at a time until told to stop.
///
/// Once `closing` is set no request is run, and one that was running when it
/// was set fails with the rest: whoever dropped the `Decider` has no use for
/// the answer, and the caller must not take a result for one the decider
/// stood behind.
pub(crate) fn run(mut engine: Box<dyn Engine>, rx: &Receiver<Msg>, closing: &AtomicBool) {
    while let Ok(Msg::Job(job)) = rx.recv() {
        if closing.load(Ordering::SeqCst) {
            let _ = job.reply.send(Err(closed()));
            continue;
        }
        let Ok(result) = catch_unwind(AssertUnwindSafe(|| engine.decide(&job.req))) else {
            // The model may be half-way through a pass; nothing it says after
            // this is worth trusting, so the worker stops and every later
            // request fails at the queue.
            tracing::error!("the model panicked while deciding; the worker stops");
            let _ = job
                .reply
                .send(Err(DecideError::Engine("the model panicked".to_owned())));
            break;
        };
        let result = if closing.load(Ordering::SeqCst) {
            Err(closed())
        } else {
            result
        };
        let _ = job.reply.send(result);
    }
    engine.close();
}

/// How a model answers: from the next-token rows at each answer slot, or from
/// a decision head.
pub(crate) enum Kind {
    /// A checkpoint trained to answer in one token.
    Readout(Arc<dyn DecisionReadout>),
    /// A checkpoint with a schema decision head.
    Head(Arc<dyn DecisionLayout>),
}

/// A loaded model, answering through [`Scorer`].
pub(crate) struct ModelEngine {
    model: RefCell<Box<dyn Model>>,
    backend: Arc<dyn Backend>,
    tokens: PromptTokens,
    kind: Kind,
    /// The most tokens a prompt may hold: one row short of the context, so the
    /// answer row has room after it, as the server's routes leave it.
    budget: usize,
}

impl ModelEngine {
    /// An engine over a model already loaded on `backend`.
    pub(crate) fn new(
        model: Box<dyn Model>,
        backend: Arc<dyn Backend>,
        tokens: PromptTokens,
        kind: Kind,
        ctx: usize,
    ) -> Self {
        Self {
            model: RefCell::new(model),
            backend,
            tokens,
            kind,
            budget: ctx.saturating_sub(1),
        }
    }

    /// Runs `body` in a session of its own and closes it however `body` ends.
    fn in_session<T>(
        &self,
        body: impl FnOnce(SessionId) -> CoreResult<T>,
    ) -> Result<T, DecideError> {
        let session = self
            .model
            .borrow_mut()
            .open_session()
            .map_err(|error| engine_error(&error))?;
        let outcome = body(session);
        if let Err(error) = self.model.borrow_mut().close_session(session) {
            tracing::warn!(error = %error.chain(), "a decision session did not close");
        }
        outcome.map_err(|error| engine_error(&error))
    }

    /// The logits row after reading `tokens` in `session`, on the host.
    fn last_row(&self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Vec<f32>> {
        let mut model = self.model.borrow_mut();
        let logits = model.prefill_last(session, tokens)?;
        let view = logits.row(0)?;
        let mut bytes = vec![0u8; usize::try_from(view.len()).unwrap_or(0)];
        self.backend.copy_d2h(view, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect())
    }
}

fn engine_error(error: &shiso_core::CoreError) -> DecideError {
    DecideError::Engine(error.chain().to_string())
}

impl Scorer for ModelEngine {
    /// Reads each run whole, one session each. A shared prefix is not reused:
    /// a decision prompt is short, and a request holds a handful of questions.
    fn score(&self, runs: &[ScoreRun]) -> Result<Vec<RunScore>, DecideError> {
        runs.iter()
            .map(|run| {
                let row = self.in_session(|session| self.last_row(session, &run.tokens))?;
                Ok(RunScore {
                    label_logprobs: label_logprobs(&row, &run.labels),
                    cached_tokens: 0,
                })
            })
            .collect()
    }

    fn score_decision(&self, prompt: &DecisionPrompt) -> Result<Vec<Vec<f32>>, DecideError> {
        self.in_session(|session| {
            {
                let mut model = self.model.borrow_mut();
                let head = model
                    .decision_mut()
                    .ok_or(shiso_core::CoreError::Unsupported {
                        what: "a decision pass on a model without a decision head",
                        backend: "decision",
                    })?;
                // Before the prompt is read, so the stack keeps every row the
                // head will pool.
                head.capture_hidden(session)?;
            }
            self.last_row(session, &prompt.tokens)?;
            let mut model = self.model.borrow_mut();
            let head = model
                .decision_mut()
                .ok_or(shiso_core::CoreError::Unsupported {
                    what: "a decision pass on a model without a decision head",
                    backend: "decision",
                })?;
            head.decide(session, &prompt.tokens, &prompt.questions)
        })
    }
}

impl Engine for ModelEngine {
    fn decide(&mut self, req: &DecisionRequest) -> Result<Vec<Answer>, DecideError> {
        match &self.kind {
            Kind::Readout(readout) => {
                decide_with_readout(readout.as_ref(), &self.tokens, &*self, req, self.budget)
            }
            Kind::Head(layout) => {
                decide_with_head(layout.as_ref(), &self.tokens, &*self, req, self.budget)
            }
        }
    }

    fn close(&mut self) {
        if let Err(error) = self.model.get_mut().teardown() {
            tracing::warn!(error = %error.chain(), "the decision model did not tear down cleanly");
        }
    }
}
