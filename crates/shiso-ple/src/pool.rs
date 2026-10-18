//! The fallback reader: several threads, each blocked on one positional read.
//!
//! One miss is a hundred-odd byte random read, so a batch is latency bound
//! rather than bandwidth bound. A serial loop over a prefill chunk's twenty
//! thousand misses runs the device at a queue depth of one and pays its full
//! round trip for every row. The threads exist purely to keep several of those
//! in flight; nothing here is trying to use more processor, and the depth is
//! what it buys.
//!
//! # Why the threads stand rather than being spawned per batch
//!
//! A decode step of the sharded NVFP4 table asks for sixteen rows, nearly all
//! of them misses, and the stack waits for them between its first layer and
//! the rest. Spawning and joining fifteen scoped helpers for that batch cost
//! 290 us at the median and 490 us at p90 on a GB10 before a single read was
//! issued, and the whole gather measured 614 us median (sixteen cold rows,
//! `tests/ple_decode_latency.rs`) - time the device spent idle, because the
//! layers after the n-gram block are launched only once the rows land. So the
//! helpers are started once, on the first batch that wants them, and park on a
//! condition variable between batches; a batch wakes as many as it has rows
//! for.
//!
//! The helpers never borrow from the batch. Each batch is an `Arc` holding the
//! reader, its own copy of the row identifiers and a staging buffer behind a
//! lock, so a helper that wakes after the batch has already been drained by
//! the others finds nothing left to claim and goes back to sleep, and the
//! submitting thread waits for rows, not for helpers. That keeps the whole
//! thing free of `unsafe` and of any lifetime the borrow checker cannot see.
//!
//! # Why the work is taken rather than dealt
//!
//! Every reader takes the next unclaimed read until the list is drained, so a
//! slow read holds up one row rather than a fixed stride of them. Dealing the
//! list out in equal stripes would make the batch as slow as its unluckiest
//! stripe. A row of a sharded table is several reads, one per array, and each
//! is its own unit of work: one reader walking a row's arrays in turn would
//! pay a cold row's round trip once per array.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;

use parking_lot::{Condvar, Mutex};

use crate::cache::Tally;
use crate::handle::{Counters, RowReader};
use crate::{PleError, PleResult};

/// Below this many misses a wakeup costs more than the read it would overlap,
/// so the submitting thread does them alone.
const MIN_MISSES: usize = 4;

/// Stack each helper runs on. A helper's frames are a read and a copy into a
/// row-sized buffer, so the default eight mebibytes would only reserve address
/// space nobody touches.
const HELPER_STACK: usize = 256 << 10;

/// A batch's worth of reads, spread over standing threads.
#[derive(Debug)]
pub(crate) struct Pool {
    workers: usize,
    /// The helpers, started on the first batch that wants them.
    crew: Mutex<Option<Crew>>,
}

/// The standing helpers and the queue they take batches from.
#[derive(Debug)]
struct Crew {
    shared: Arc<CrewShared>,
    joins: Vec<JoinHandle<()>>,
}

/// What the helpers and the submitting thread share.
#[derive(Debug, Default)]
struct CrewShared {
    queue: Mutex<CrewQueue>,
    wake: Condvar,
}

/// One entry per helper a batch asked for, and the flag that ends them.
#[derive(Debug, Default)]
struct CrewQueue {
    tickets: VecDeque<Arc<Batch>>,
    shutdown: bool,
}

/// One batch, owned jointly by its submitter and the helpers working on it.
#[derive(Debug)]
struct Batch {
    reader: Arc<RowReader>,
    rows: Vec<u64>,
    stride: usize,
    /// Reads one row takes: its arrays, for a sharded table.
    parts: usize,
    /// The next unclaimed read, row-major over the parts.
    next: AtomicUsize,
    landed: Mutex<Landed>,
    done: Condvar,
}

/// The rows read so far, in order, and what reading them cost.
#[derive(Debug)]
struct Landed {
    staging: Vec<u8>,
    /// Reads finished, landed or failed.
    finished: usize,
    tally: Tally,
    failure: Option<PleError>,
}

impl Pool {
    /// A pool that will use at most `workers` threads, the submitting one
    /// included.
    pub(crate) const fn new(workers: u32) -> Self {
        Self {
            workers: workers as usize,
            crew: Mutex::new(None),
        }
    }

    /// Fills `staging` with one row per identifier, in order.
    ///
    /// # Errors
    ///
    /// The first failure any reader hit. The other rows are still read - their
    /// readers are already blocked in the kernel - and their time is still
    /// counted, so the statistics do not quietly lose a batch that failed
    /// halfway; [`PleError::Io`] when the helpers cannot be started.
    pub(crate) fn read_rows(
        &self,
        reader: &Arc<RowReader>,
        rows: &[u64],
        staging: &mut [u8],
        stats: &Counters,
    ) -> PleResult<()> {
        let stride = reader.row_bytes();
        let parts = reader.parts();
        let units = rows.len() * parts;
        let helpers = if self.workers > 1 && rows.len() >= MIN_MISSES {
            units.min(self.workers) - 1
        } else {
            0
        };
        let batch = Arc::new(Batch {
            reader: Arc::clone(reader),
            rows: rows.to_vec(),
            stride,
            parts,
            next: AtomicUsize::new(0),
            landed: Mutex::new(Landed {
                staging: vec![0u8; rows.len() * stride],
                finished: 0,
                tally: Tally::default(),
                failure: None,
            }),
            done: Condvar::new(),
        });
        if helpers > 0 {
            self.dispatch(&batch, helpers)?;
        }
        drain(&batch);
        let mut landed = batch.landed.lock();
        while landed.finished < units {
            batch.done.wait(&mut landed);
        }
        stats.merge(landed.tally);
        if let Some(err) = landed.failure.take() {
            return Err(err);
        }
        staging[..landed.staging.len()].copy_from_slice(&landed.staging);
        Ok(())
    }

    /// Hands `helpers` tickets for `batch` to the crew, starting it first if
    /// this is the first batch that wanted one.
    fn dispatch(&self, batch: &Arc<Batch>, helpers: usize) -> PleResult<()> {
        let mut crew = self.crew.lock();
        if crew.is_none() {
            *crew = Some(Crew::start(self.workers - 1)?);
        }
        let Some(crew) = crew.as_ref() else {
            return Ok(());
        };
        {
            let mut queue = crew.shared.queue.lock();
            for _ in 0..helpers {
                queue.tickets.push_back(Arc::clone(batch));
            }
        }
        for _ in 0..helpers {
            crew.shared.wake.notify_one();
        }
        Ok(())
    }
}

impl Crew {
    /// Starts `n` parked helpers.
    fn start(n: usize) -> PleResult<Self> {
        let shared = Arc::new(CrewShared::default());
        let mut joins = Vec::with_capacity(n);
        for _ in 0..n {
            let shared = Arc::clone(&shared);
            let join = std::thread::Builder::new()
                .name("shiso-ple-read".to_owned())
                .stack_size(HELPER_STACK)
                .spawn(move || helper(&shared))
                .map_err(PleError::Io)?;
            joins.push(join);
        }
        Ok(Self { shared, joins })
    }
}

impl Drop for Crew {
    /// Ends the helpers and waits for them. A helper in the middle of a batch
    /// finishes its read first; the queue is empty by then, because every
    /// batch is waited for by its submitter before the pool can be dropped.
    fn drop(&mut self) {
        self.shared.queue.lock().shutdown = true;
        self.shared.wake.notify_all();
        for join in self.joins.drain(..) {
            if join.join().is_err() {
                tracing::error!("an n-gram reader thread panicked");
            }
        }
    }
}

/// A helper's life: take a ticket, drain its batch, park.
fn helper(shared: &CrewShared) {
    loop {
        let batch = {
            let mut queue = shared.queue.lock();
            loop {
                if queue.shutdown {
                    return;
                }
                if let Some(batch) = queue.tickets.pop_front() {
                    break batch;
                }
                shared.wake.wait(&mut queue);
            }
        };
        drain(&batch);
    }
}

/// Takes reads until the batch's list is drained, each into this thread's
/// own buffer and then copied into its place in the batch.
fn drain(batch: &Batch) {
    let units = batch.rows.len() * batch.parts;
    let mut buf = vec![0u8; batch.stride];
    let mut bounce = Vec::new();
    loop {
        let u = batch.next.fetch_add(1, Ordering::Relaxed);
        if u >= units {
            return;
        }
        let (i, part) = (u / batch.parts, u % batch.parts);
        let (at, len) = if batch.parts == 1 {
            (0, batch.stride)
        } else {
            batch.reader.part_span(part)
        };
        let outcome = if batch.parts == 1 {
            batch.reader.read_row(batch.rows[i], &mut buf, &mut bounce)
        } else {
            batch.reader.read_part(batch.rows[i], part, &mut buf[..len])
        };
        let mut landed = batch.landed.lock();
        match outcome {
            Ok(tally) => {
                landed.tally.merge(tally);
                let to = i * batch.stride + at;
                landed.staging[to..to + len].copy_from_slice(&buf[..len]);
            }
            Err(err) => {
                if landed.failure.is_none() {
                    landed.failure = Some(err);
                }
            }
        }
        landed.finished += 1;
        if landed.finished == units {
            batch.done.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_under_the_threshold_never_leaves_the_submitting_thread() {
        // Nothing here observes the thread count directly; what it pins is that
        // the boundary is where it is meant to be, so a batch of a handful of
        // rows does not wake a helper.
        assert_eq!(MIN_MISSES, 4);
        let pool = Pool::new(64);
        assert_eq!(pool.workers, 64);
        assert!(pool.crew.lock().is_none());
    }
}
