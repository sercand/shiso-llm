//! Reading a stored path back: pages streamed off disk while the caller
//! uploads them.
//!
//! A restore is a few hundred page files and one entry file. Read one at a time
//! on the scheduler thread, the drive would idle between requests and the
//! device between uploads. So a reader thread keeps several reads in flight
//! through [`shiso_ple::bulk`], checks each page against the name the entry
//! gave it, and hands it over a bounded channel; the caller uploads page `k`
//! while page `k + depth` is still being read. Buffers travel back the other
//! way, so a restore of any length touches `depth + 2` page buffers.
//!
//! # A bad page ends the restore
//!
//! A page that is missing, foreign or corrupt makes the entry unusable - a path
//! with a hole in it cannot be resumed from - so the stream stops at the first
//! one and says which, and the caller discards the entry and prefills.

use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

use shiso_core::TokenId;
use shiso_ple::bulk::{BulkReader, FileRead};

use crate::store::format::{self, PageExpect, Unusable};
use crate::store::key::{Identity, PageKey};

/// One page the reader has checked.
#[derive(Debug)]
pub(crate) struct PageBytes {
    /// Which page of the path this is.
    pub(crate) index: usize,
    /// The whole file.
    pub(crate) buf: Vec<u8>,
    /// Where the payload starts inside `buf`.
    pub(crate) payload_at: usize,
}

impl PageBytes {
    /// The page's rows.
    #[must_use]
    pub(crate) fn payload(&self) -> &[u8] {
        &self.buf[self.payload_at..]
    }
}

/// Why a page could not be delivered.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PageFailure {
    pub(crate) index: usize,
    pub(crate) why: Unusable,
}

/// One page to read: where it is and what it must turn out to be.
#[derive(Debug)]
pub(crate) struct PageJob {
    pub(crate) index: usize,
    pub(crate) path: std::path::PathBuf,
    pub(crate) key: PageKey,
    pub(crate) parent: Option<PageKey>,
    pub(crate) tokens: Vec<TokenId>,
}

/// A running page stream.
#[derive(Debug)]
pub(crate) struct PageStream {
    rx: Receiver<Result<PageBytes, PageFailure>>,
    recycle: SyncSender<Vec<u8>>,
    thread: Option<JoinHandle<()>>,
}

impl PageStream {
    /// Starts reading `jobs` in order, `depth` at a time.
    pub(crate) fn start(
        jobs: Vec<PageJob>,
        identity: Identity,
        payload_bytes: u64,
        depth: u32,
    ) -> std::io::Result<Self> {
        let depth = depth.max(1);
        let lanes = usize::try_from(depth).unwrap_or(1);
        let (tx, rx) = sync_channel(lanes);
        let (recycle, spare) = sync_channel::<Vec<u8>>(lanes * 2 + 2);
        let thread = std::thread::Builder::new()
            .name("shiso-kv-read".to_owned())
            .spawn(move || read_all(&jobs, identity, payload_bytes, depth, &tx, &spare))?;
        Ok(Self {
            rx,
            recycle,
            thread: Some(thread),
        })
    }

    /// The next page, in path order; `None` once every page was delivered.
    pub(crate) fn next(&self) -> Option<Result<PageBytes, PageFailure>> {
        self.rx.recv().ok()
    }

    /// Hands a buffer back for the next read.
    pub(crate) fn give_back(&self, buf: Vec<u8>) {
        let _ = self.recycle.try_send(buf);
    }
}

impl Drop for PageStream {
    fn drop(&mut self) {
        // Draining the channel is what unblocks a reader parked on a full one;
        // it then finds the receiver gone at its next send and stops.
        while self.rx.try_recv().is_ok() {}
        if let Some(thread) = self.thread.take() {
            let (_, rx) = sync_channel::<Result<PageBytes, PageFailure>>(0);
            drop(std::mem::replace(&mut self.rx, rx));
            let _ = thread.join();
        }
    }
}

/// The reader thread's body.
fn read_all(
    jobs: &[PageJob],
    identity: Identity,
    payload_bytes: u64,
    depth: u32,
    tx: &SyncSender<Result<PageBytes, PageFailure>>,
    spare: &Receiver<Vec<u8>>,
) {
    let mut reader = BulkReader::new(depth);
    let lanes = usize::try_from(depth).unwrap_or(1);
    for batch in jobs.chunks(lanes) {
        let mut reads: Vec<FileRead> = batch
            .iter()
            .map(|job| FileRead::new(job.path.clone(), spare.try_recv().unwrap_or_default()))
            .collect();
        reader.read(&mut reads);
        for (job, read) in batch.iter().zip(reads) {
            let delivered = if read.result.is_err() {
                Err(PageFailure {
                    index: job.index,
                    why: Unusable::Missing,
                })
            } else {
                let expect = PageExpect {
                    key: job.key,
                    identity,
                    parent: job.parent,
                    tokens: &job.tokens,
                    payload_bytes,
                };
                format::verify_page(&read.buf, &expect)
                    .map(|payload_at| PageBytes {
                        index: job.index,
                        buf: read.buf,
                        payload_at,
                    })
                    .map_err(|why| PageFailure {
                        index: job.index,
                        why,
                    })
            };
            let failed = delivered.is_err();
            if tx.send(delivered).is_err() || failed {
                return;
            }
        }
    }
}
