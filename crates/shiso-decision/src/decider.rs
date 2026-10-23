//! [`Decider`]: a decision checkpoint opened in this process.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

use futures::channel::oneshot;

use crate::error::DecideError;
use crate::load;
use crate::schema::{Answer, Choice, DecisionRequest, Question};
use crate::worker::{self, Engine, Job, Msg, Shared};

/// The context length [`OpenDesc::new`] asks for: a decision prompt is a state
/// and a few short questions, so the engine does not reserve more.
pub const DEFAULT_CTX: usize = 2048;

/// What [`Decider::open`] needs to find and load a model.
#[derive(Debug, Clone, Copy)]
pub struct OpenDesc<'a> {
    /// The model directory: `config.json`, `tokenizer.json` and one `.gguf`
    /// checkpoint, either directly in it or under its `config/` and `base/`
    /// subdirectories, which is how an installed model is laid out. A
    /// `mmproj*` file is a vision sidecar, not the checkpoint.
    pub model_dir: &'a Path,
    /// The context length: the longest prompt, answer row included, a request
    /// may lay out.
    pub ctx: usize,
}

impl<'a> OpenDesc<'a> {
    /// A description of `model_dir` at the [`DEFAULT_CTX`].
    #[must_use]
    pub const fn new(model_dir: &'a Path) -> Self {
        Self {
            model_dir,
            ctx: DEFAULT_CTX,
        }
    }
}

/// The throwaway request [`Decider::open`] warms the model with: the smallest
/// valid question, so the pass costs the device's set-up and little else.
fn warm_up_request() -> DecisionRequest {
    let choice = |value: &str| Choice {
        value: value.to_owned(),
        description: None,
    };
    DecisionRequest {
        input: "warm-up".to_owned(),
        questions: vec![Question::Choice {
            name: "warm_up".to_owned(),
            instructions: "pick one".to_owned(),
            choices: vec![choice("a"), choice("b")],
        }],
        temperature: None,
    }
}

/// A decision model owned by a worker thread of its own.
///
/// Dropping a `Decider` joins its worker, so the drop can block for as long as
/// one pass the worker is in the middle of. Do not drop it on an async
/// executor's thread while a call is in flight; drop it from a thread that may
/// block (`spawn_blocking`, or the owner's shutdown path).
///
/// The model lives on that thread for good; requests reach it over a channel
/// and are answered one at a time, in arrival order, so any number of threads
/// may share one `Decider`. Dropping it stops the worker and joins it: a
/// request that was queued, or that the worker was running, then fails with
/// [`DecideError::Engine`] rather than returning an answer nobody can still
/// be waiting for.
#[derive(Debug)]
pub struct Decider {
    queue: mpsc::Sender<Msg>,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl Decider {
    /// Opens the model in `desc` on the device this build was compiled for.
    ///
    /// Blocks while the weights load, then runs one throwaway pass of a tiny
    /// fixed prompt through the worker, so the device's first-use cost (kernel
    /// and buffer set-up, measured at about 1.5 s on Metal) is paid here and
    /// the first real call is as fast as any other. `open` takes that much
    /// longer.
    ///
    /// # Errors
    ///
    /// [`DecideError::Request`] for a context under 2 (a prompt and its answer
    /// row need two positions), [`DecideError::Load`] for a directory that is not a loadable decision
    /// model (including a checkpoint with neither a decision readout nor a
    /// head), [`DecideError::NoDevice`] when this build has no backend feature
    /// or the device would not open, and [`DecideError::Io`] for a file that
    /// cannot be read; the warm-up pass can fail with [`DecideError::Layout`]
    /// or [`DecideError::Engine`], which then fails the open.
    pub fn open(desc: &OpenDesc<'_>) -> Result<Self, DecideError> {
        if desc.ctx < 2 {
            return Err(DecideError::Request(format!(
                "ctx must be at least 2 (a prompt and its answer row), got {}",
                desc.ctx
            )));
        }
        let engine = load::open(desc)?;
        Self::start(Box::new(engine))
    }

    /// Spawns the worker over `engine` and warms it. A failed warm-up drops
    /// the decider, which stops the worker and closes the engine.
    pub(crate) fn start(engine: Box<dyn Engine>) -> Result<Self, DecideError> {
        let decider = Self::spawn(engine)?;
        decider.decide_blocking(&warm_up_request())?;
        Ok(decider)
    }

    /// Starts the worker thread over `engine`.
    pub(crate) fn spawn(engine: Box<dyn Engine>) -> Result<Self, DecideError> {
        Self::spawn_with(engine, |body| {
            std::thread::Builder::new()
                .name("shiso-decision".to_owned())
                .spawn(body)
        })
    }

    /// Starts the worker with `start`, which a test makes fail. When it does,
    /// the engine is closed before the error is returned: the model holds
    /// device memory, and nothing else will ever release it.
    pub(crate) fn spawn_with(
        engine: Box<dyn Engine>,
        start: impl FnOnce(Box<dyn FnOnce() + Send>) -> std::io::Result<JoinHandle<()>>,
    ) -> Result<Self, DecideError> {
        let (queue, rx) = mpsc::channel();
        let shared = Arc::new(Shared::default());
        // The thread takes the engine out of the slot; a thread that never
        // started leaves it there to be closed.
        let slot = Arc::new(Mutex::new(Some(engine)));
        let body: Box<dyn FnOnce() + Send> = {
            let (slot, shared) = (Arc::clone(&slot), Arc::clone(&shared));
            Box::new(move || {
                let taken = slot.lock().ok().and_then(|mut slot| slot.take());
                if let Some(engine) = taken {
                    worker::run(engine, &rx, &shared);
                }
            })
        };
        match start(body) {
            Ok(worker) => Ok(Self {
                queue,
                shared,
                worker: Some(worker),
            }),
            Err(error) => {
                if let Some(mut engine) = slot.lock().ok().and_then(|mut slot| slot.take()) {
                    engine.close();
                }
                Err(DecideError::Io(error))
            }
        }
    }

    /// Answers `req`, blocking the calling thread until the worker has.
    ///
    /// # Errors
    ///
    /// [`DecideError::Request`], [`DecideError::Layout`] or
    /// [`DecideError::Engine`] as [`crate::score::decide_with_readout`]
    /// reports them, and [`DecideError::Engine`] when the decider was dropped
    /// or its worker stopped before answering.
    pub fn decide_blocking(&self, req: &DecisionRequest) -> Result<Vec<Answer>, DecideError> {
        futures::executor::block_on(self.decide(req))
    }

    /// Answers `req`. The future holds no thread: the worker wakes it.
    ///
    /// # Errors
    ///
    /// As [`Decider::decide_blocking`].
    pub async fn decide(&self, req: &DecisionRequest) -> Result<Vec<Answer>, DecideError> {
        self.handle().decide(req.clone()).await
    }

    /// What the worker sees, so a test can tell when a drop has begun.
    #[cfg(test)]
    pub(crate) fn shared(&self) -> Arc<Shared> {
        Arc::clone(&self.shared)
    }

    /// A handle that can ask for answers without borrowing the `Decider`, so
    /// a test can have a call in flight while the `Decider` is dropped.
    pub(crate) fn handle(&self) -> Handle {
        Handle {
            queue: self.queue.clone(),
            shared: Arc::clone(&self.shared),
        }
    }
}

impl Drop for Decider {
    fn drop(&mut self) {
        self.shared.closing.store(true, Ordering::SeqCst);
        // Behind whatever is already queued: the worker fails those and then
        // leaves its loop.
        let _ = self.queue.send(Msg::Stop);
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            tracing::warn!("the decision worker panicked");
        }
    }
}

/// A way to queue requests that outlives a borrow of the [`Decider`].
#[derive(Debug, Clone)]
pub(crate) struct Handle {
    queue: mpsc::Sender<Msg>,
    shared: Arc<Shared>,
}

impl Handle {
    /// Queues `req` and waits for its answer.
    pub(crate) async fn decide(&self, req: DecisionRequest) -> Result<Vec<Answer>, DecideError> {
        let (reply, answer) = oneshot::channel();
        self.queue
            .send(Msg::Job(Job { req, reply }))
            .map_err(|_| self.shared.stopped())?;
        // A dropped reply means the worker went away with the job unanswered.
        answer.await.map_err(|_| self.shared.stopped())?
    }

    /// As [`Handle::decide`], blocking the calling thread.
    #[cfg(test)]
    pub(crate) fn decide_blocking(&self, req: DecisionRequest) -> Result<Vec<Answer>, DecideError> {
        futures::executor::block_on(self.decide(req))
    }
}
