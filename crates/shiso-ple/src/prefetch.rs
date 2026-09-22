//! Warming the cache for the chunk after this one.
//!
//! During a prefill the rows a later chunk will ask for are already known, and
//! the current chunk is about to spend tens of milliseconds on the device. So
//! one thread reads them into the cache while that happens, and the fetch that
//! eventually asks for them hits. The bytes that thread reads are thrown away -
//! the point was the residency, not the copy.
//!
//! # Why one thread and not a pool
//!
//! A second prefetch arriving while the first is still running waits for it.
//! That is deliberate: two overlapping prefetches would contend for the same
//! serialized fetch lock and for the same cache slots, and the second one is
//! for a chunk that is further away in any case. Waiting also bounds the memory
//! this holds to one batch's worth.
//!
//! # Why no async runtime
//!
//! The model runs on a plain scheduler thread and the model trait is
//! synchronous, so introducing a runtime here would mean introducing one there.
//! A thread and a condition variable are the whole mechanism.

use std::sync::Arc;
use std::thread::JoinHandle;

use parking_lot::{Condvar, Mutex};

use crate::handle::Shared;
use crate::{PleError, PleResult};

/// The prefetch thread and the queue it reads from.
#[derive(Debug)]
pub(crate) struct Prefetch {
    state: Arc<State>,
    join: JoinHandle<()>,
}

/// What the owner and the thread share.
#[derive(Debug)]
struct State {
    stream: Arc<Shared>,
    queue: Mutex<Queue>,
    signal: Condvar,
}

/// The one pending batch, and the two flags that drive the thread.
#[derive(Debug, Default)]
struct Queue {
    rows: Vec<u64>,
    busy: bool,
    shutdown: bool,
}

impl Prefetch {
    /// Starts the thread.
    ///
    /// # Errors
    ///
    /// [`PleError::Io`] when the thread cannot be spawned, which is the one
    /// failure this has: everything else it does is waiting.
    pub(crate) fn start(stream: Arc<Shared>) -> PleResult<Self> {
        let state = Arc::new(State {
            stream,
            queue: Mutex::new(Queue::default()),
            signal: Condvar::new(),
        });
        let thread_state = Arc::clone(&state);
        let join = std::thread::Builder::new()
            .name("shiso-ple-prefetch".to_owned())
            .spawn(move || run(&thread_state))
            .map_err(PleError::Io)?;
        Ok(Self { state, join })
    }

    /// Hands the thread a batch, waiting for the previous one to drain.
    pub(crate) fn submit(&self, rows: &[u64]) {
        let mut queue = self.state.queue.lock();
        while queue.busy && !queue.shutdown {
            self.state.signal.wait(&mut queue);
        }
        if queue.shutdown {
            return;
        }
        queue.rows.clear();
        queue.rows.extend_from_slice(rows);
        queue.busy = true;
        drop(queue);
        self.state.signal.notify_all();
    }

    /// Stops the thread and waits for it.
    ///
    /// # Errors
    ///
    /// [`PleError::Io`] when the thread ended by panicking. The stream is still
    /// safe to drop after that - the thread is gone either way - but a panic in
    /// a background reader is worth surfacing rather than swallowing.
    pub(crate) fn stop(self) -> PleResult<()> {
        {
            let mut queue = self.state.queue.lock();
            queue.shutdown = true;
        }
        self.state.signal.notify_all();
        self.join
            .join()
            .map_err(|_| PleError::Io(std::io::Error::other("the n-gram prefetch thread panicked")))
    }
}

/// Reads whatever is queued, for as long as the stream is open.
fn run(state: &Arc<State>) {
    let mut scratch: Vec<u8> = Vec::new();
    let mut rows: Vec<u64> = Vec::new();
    loop {
        {
            let mut queue = state.queue.lock();
            while !queue.shutdown && !queue.busy {
                state.signal.wait(&mut queue);
            }
            if queue.shutdown {
                return;
            }
            rows.clear();
            rows.extend_from_slice(&queue.rows);
        }

        scratch.clear();
        scratch.resize(rows.len() * state.stream.row_bytes(), 0);
        // The bytes are discarded: what the fetch leaves behind is the
        // residency, and a prefetch that fails costs a later fetch its hit and
        // nothing else.
        if let Err(err) = state.stream.fetch(&rows, &mut scratch) {
            tracing::debug!(%err, rows = rows.len(), "an n-gram prefetch did not complete");
        }

        let mut queue = state.queue.lock();
        queue.busy = false;
        drop(queue);
        state.signal.notify_all();
    }
}
