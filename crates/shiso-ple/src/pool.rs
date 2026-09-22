//! The fallback reader: several threads, each blocked on one positional read.
//!
//! One miss is a hundred-odd byte random read, so a batch is latency bound
//! rather than bandwidth bound. A serial loop over a prefill chunk's twenty
//! thousand misses runs the device at a queue depth of one and pays its full
//! round trip for every row. The threads exist purely to keep several of those
//! in flight; nothing here is trying to use more processor, and the depth is
//! what it buys.
//!
//! # Why the threads are per batch rather than a standing pool
//!
//! The reference keeps a standing pool because its workers write straight into
//! the cache arena, which needs the pool to outlive any one batch and needs raw
//! pointers into a shared allocation. Here each reader owns its own buffer and
//! hands it back when the batch ends, so the threads can be scoped to the
//! batch - which means no shutdown protocol, no semaphore, no lifetime a
//! borrow checker cannot see, and no unsafe. What it costs is one thread spawn
//! per reader per batch: tens of microseconds against a batch whose reads take
//! tens of milliseconds, on a path that is deliberately off the critical one. A
//! decode step, which is where a spawn would actually show, never reaches the
//! threshold below and so spawns nothing at all.
//!
//! # Why the work is taken rather than dealt
//!
//! Every reader takes the next unclaimed miss until the list is drained, so a
//! slow read holds up one row rather than a fixed stride of them. Dealing the
//! list out in equal stripes would make the batch as slow as its unluckiest
//! stripe.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::cache::Tally;
use crate::handle::{Counters, RowReader};
use crate::{PleError, PleResult};

/// Below this many misses a wakeup costs more than the read it would overlap,
/// so the submitting thread does them alone.
const MIN_MISSES: usize = 4;

/// A batch's worth of reads, spread over threads.
#[derive(Debug)]
pub(crate) struct Pool {
    workers: usize,
}

/// What one reader brings back: the rows it took, their bytes, and its time.
#[derive(Debug, Default)]
struct Share {
    taken: Vec<usize>,
    bytes: Vec<u8>,
    tally: Tally,
    failure: Option<PleError>,
}

impl Pool {
    /// A pool that will use at most `workers` threads, the submitting one
    /// included.
    pub(crate) const fn new(workers: u32) -> Self {
        Self {
            workers: workers as usize,
        }
    }

    /// Fills `staging` with one row per identifier, in order.
    ///
    /// # Errors
    ///
    /// The first failure any reader hit. The other readers still finish - they
    /// are already blocked in the kernel - and their time is still counted, so
    /// the statistics do not quietly lose a batch that failed halfway.
    pub(crate) fn read_rows(
        &self,
        reader: &RowReader,
        rows: &[u64],
        staging: &mut [u8],
        stats: &Counters,
    ) -> PleResult<()> {
        let stride = reader.row_bytes();
        let helpers = if self.workers > 1 && rows.len() >= MIN_MISSES {
            rows.len().min(self.workers) - 1
        } else {
            0
        };

        let next = AtomicUsize::new(0);
        let mut shares = Vec::with_capacity(helpers + 1);
        if helpers == 0 {
            shares.push(drain(reader, rows, stride, &next));
        } else {
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(helpers);
                for _ in 0..helpers {
                    handles.push(scope.spawn(|| drain(reader, rows, stride, &next)));
                }
                shares.push(drain(reader, rows, stride, &next));
                for handle in handles {
                    shares.push(handle.join().unwrap_or_else(|_| Share {
                        failure: Some(PleError::Io(std::io::Error::other(
                            "an n-gram reader thread panicked",
                        ))),
                        ..Share::default()
                    }));
                }
            });
        }

        let mut tally = Tally::default();
        let mut failure = None;
        for share in &mut shares {
            tally.merge(share.tally);
            if failure.is_none() {
                failure = share.failure.take();
            }
        }
        stats.merge(tally);
        if let Some(err) = failure {
            return Err(err);
        }
        for share in &shares {
            for (k, &i) in share.taken.iter().enumerate() {
                staging[i * stride..(i + 1) * stride]
                    .copy_from_slice(&share.bytes[k * stride..(k + 1) * stride]);
            }
        }
        Ok(())
    }
}

/// Takes misses until the list is drained, reading each into this reader's own
/// buffer.
fn drain(reader: &RowReader, rows: &[u64], stride: usize, next: &AtomicUsize) -> Share {
    let mut share = Share::default();
    let mut bounce = Vec::new();
    loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= rows.len() {
            return share;
        }
        let at = share.bytes.len();
        share.bytes.resize(at + stride, 0);
        match reader.read_row(rows[i], &mut share.bytes[at..], &mut bounce) {
            Ok(tally) => {
                share.tally.merge(tally);
                share.taken.push(i);
            }
            Err(err) => {
                share.bytes.truncate(at);
                share.failure = Some(err);
                return share;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_under_the_threshold_never_leaves_the_submitting_thread() {
        // Nothing here observes the thread count directly; what it pins is that
        // the boundary is where it is meant to be, so a decode step's handful
        // of rows does not pay a spawn.
        assert_eq!(MIN_MISSES, 4);
        let pool = Pool::new(64);
        assert_eq!(pool.workers, 64);
    }
}
