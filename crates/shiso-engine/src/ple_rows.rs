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

use std::sync::Arc;
use std::thread::JoinHandle;

use shiso_core::error::{CoreError, CoreResult};
use shiso_ple::{PleResult, PleStream};

/// A gather whose reads are in flight.
///
/// Separate from whatever issued it so that awaiting rows nobody asked for is
/// not a state this crate can be in. Dropping one without awaiting is allowed
/// and costs only the reads already issued: the thread finishes into a buffer
/// that is then thrown away, which is exactly what a cancelled chunk wants.
#[derive(Debug)]
pub struct PleRows {
    join: JoinHandle<PleResult<Vec<u8>>>,
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
        match self.join.join() {
            Ok(result) => Ok(result?),
            Err(_) => Err(CoreError::Io(std::io::Error::other(
                "the per-row fetch thread panicked",
            ))),
        }
    }
}

/// Starts the reads for `ids` on a thread of its own.
///
/// The identifiers are computed by the caller, on the caller's thread, because
/// that is arithmetic over things it already holds; only the blocking reads
/// move. The row width is the table's own, so a caller cannot get the stride
/// wrong and read every row after the first from the middle of its predecessor.
///
/// # Errors
///
/// [`CoreError::Io`] when the fetch thread cannot be spawned.
pub fn issue_rows(table: &Arc<PleStream>, ids: Vec<u64>) -> CoreResult<PleRows> {
    let row_bytes = table.row_bytes();
    let rows = u64::try_from(ids.len()).unwrap_or(u64::MAX);
    let table = Arc::clone(table);
    let stride = row_bytes as usize;
    let join = std::thread::Builder::new()
        .name("shiso-ple-gather".to_owned())
        .spawn(move || -> PleResult<Vec<u8>> {
            let mut out = vec![0u8; ids.len() * stride];
            table.fetch(&ids, &mut out)?;
            Ok(out)
        })
        .map_err(CoreError::Io)?;
    Ok(PleRows {
        join,
        rows,
        row_bytes,
    })
}
