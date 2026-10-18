//! Rows read off a streamed table while the host is busy enqueuing something
//! else.
//!
//! Two tiers of this crate read a table too large to hold in memory, one row per
//! (position, head) and one row per token respectively, and both want the same
//! thing from the reader: start the blocking reads now, do a chunk's worth of
//! host-side work, and pick the bytes up at the point they are actually needed.
//! That shape is here rather than in either tier, because it is about the
//! reader and not about what the rows mean.
//!
//! [`issue_rows`] is the start and [`PleRows::await_rows`] is the pick-up.
//! Everything enqueued between the two has overlapped the reads. What the row
//! identifiers ARE is the caller's business: the n-gram tier hashes them out of
//! the session's trailing grams, and the per-layer tier is handed them by the
//! token ids themselves.

use std::sync::mpsc::{Receiver, Sender, SyncSender, channel, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};

use shiso_core::error::{CoreError, CoreResult};
use shiso_ple::{PleResult, PleStream};

/// Where a gather's bytes come from.
#[derive(Debug)]
enum Landing {
    /// A reader thread is fetching them.
    Pending(Receiver<PleResult<Vec<u8>>>),
    /// They were fetched on the caller's own thread.
    Ready(PleResult<Vec<u8>>),
}

/// A gather whose reads are in flight.
///
/// Separate from whatever issued it so that awaiting rows nobody asked for is
/// not a state this crate can be in. Dropping one without awaiting is allowed
/// and costs only the reads already issued: the thread finishes into a buffer
/// that is then thrown away, which is exactly what a cancelled chunk wants.
#[derive(Debug)]
pub struct PleRows {
    done: Landing,
    rows: u64,
    row_bytes: u32,
}

impl PleRows {
    /// Bytes the device buffer these rows are uploaded into has to hold.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.rows * self.row_bytes as u64
    }

    /// Rows in flight.
    #[must_use]
    pub const fn rows(&self) -> u64 {
        self.rows
    }

    /// Waits for the reads and hands back the raw quantized rows.
    ///
    /// This is the await site the whole handle exists for: everything the
    /// caller enqueued between [`issue_rows`] and here has overlapped the
    /// reads.
    ///
    /// # Errors
    ///
    /// Whatever the reader reports, as [`CoreError::Backend`], and
    /// [`CoreError::Io`] if the fetch thread ended by panicking.
    pub fn await_rows(self) -> CoreResult<Vec<u8>> {
        match self.done {
            Landing::Ready(result) => Ok(result?),
            Landing::Pending(done) => match done.recv() {
                Ok(result) => Ok(result?),
                Err(_) => Err(CoreError::Io(std::io::Error::other(
                    "the per-row fetch thread panicked",
                ))),
            },
        }
    }
}

/// One gather's work, as the reader threads receive it.
type Job = Box<dyn FnOnce() + Send>;

/// Reader threads the gathers are handed to.
///
/// Four, because a decode batch issues one gather per row and each gather's
/// misses already go out as one batch of reads; more threads would only wait
/// on the same device queue.
const READERS: usize = 4;

/// The queue the reader threads take gathers from, started on first use.
///
/// Long-lived rather than a thread per gather: a gather on the decode path
/// runs once a token, and spawning its thread and joining it cost 200-600 us
/// of a token's host time on a GB10 even when every row was a cache hit and
/// the reads themselves took 6 us - time the device spent idle, because the
/// step is launched after the rows land.
fn readers() -> CoreResult<&'static Mutex<Sender<Job>>> {
    static QUEUE: OnceLock<Mutex<Sender<Job>>> = OnceLock::new();
    if let Some(queue) = QUEUE.get() {
        return Ok(queue);
    }
    let (send, recv) = channel::<Job>();
    let recv = Arc::new(Mutex::new(recv));
    for _ in 0..READERS {
        let recv = Arc::clone(&recv);
        std::thread::Builder::new()
            .name("shiso-ple-gather".to_owned())
            .spawn(move || {
                loop {
                    // The lock is held only for the take, so the readers run
                    // their jobs side by side.
                    let job = match recv.lock() {
                        Ok(queue) => queue.recv(),
                        Err(_) => return,
                    };
                    match job {
                        Ok(job) => job(),
                        Err(_) => return,
                    }
                }
            })
            .map_err(CoreError::Io)?;
    }
    Ok(QUEUE.get_or_init(|| Mutex::new(send)))
}

/// Starts the reads for `ids` on one of the reader threads.
///
/// The identifiers are computed by the caller, on the caller's thread, because
/// that is arithmetic over things it already holds; only the blocking reads
/// move. The row width is the table's own, so a caller cannot get the stride
/// wrong and read every row after the first from the middle of its predecessor.
///
/// # Errors
///
/// [`CoreError::Io`] when the reader threads cannot be started or have gone.
pub fn issue_rows(table: &Arc<PleStream>, ids: Vec<u64>) -> CoreResult<PleRows> {
    let row_bytes = table.row_bytes();
    let rows = u64::try_from(ids.len()).unwrap_or(u64::MAX);
    let table = Arc::clone(table);
    let stride = row_bytes as usize;
    let (tx, done): (SyncSender<PleResult<Vec<u8>>>, _) = sync_channel(1);
    let job: Job = Box::new(move || {
        let mut out = vec![0u8; ids.len() * stride];
        let result = table.fetch(&ids, &mut out).map(|()| out);
        // A gather dropped without being awaited has hung up; its rows are
        // thrown away, which is what a cancelled chunk wants.
        let _ = tx.send(result);
    });
    let gone = || {
        CoreError::Io(std::io::Error::other(
            "the n-gram reader threads have stopped",
        ))
    };
    readers()?
        .lock()
        .map_err(|_| gone())?
        .send(job)
        .map_err(|_| gone())?;
    Ok(PleRows {
        done: Landing::Pending(done),
        rows,
        row_bytes,
    })
}

/// Reads the rows for `ids` on the caller's thread, for a caller with nothing
/// to overlap them with.
///
/// The sparse stack reads its rows here, on its own thread, after issuing the
/// launches in front of the island that reads them: the device has those to
/// run meanwhile, so the overlap needs no second thread, and a hand-off to a
/// reader thread would cost two thread wake-ups - 200-600 us a token on a GB10
/// whose host cores idle while the device runs, against 6 us for sixteen
/// cached rows read in place.
#[must_use]
pub fn fetch_rows(table: &PleStream, ids: &[u64]) -> PleRows {
    let row_bytes = table.row_bytes();
    let mut out = vec![0u8; ids.len() * row_bytes as usize];
    let result = table.fetch(ids, &mut out).map(|()| out);
    PleRows {
        done: Landing::Ready(result),
        rows: u64::try_from(ids.len()).unwrap_or(u64::MAX),
        row_bytes,
    }
}
