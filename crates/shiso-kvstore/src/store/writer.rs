//! The writer thread: every file this store writes, renames or unlinks.
//!
//! The scheduler thread copies bytes off the device and hands them over; this
//! thread does everything that waits on the disk. One thread is enough, because
//! the drive is the bottleneck and one writer keeps the operations in the order
//! they were queued - which is the whole of the consistency argument:
//!
//! - a page's write is queued before the write of any entry that lists it, so an
//!   entry is only written once every page it names is on disk, and an entry
//!   whose page failed is abandoned rather than written;
//! - an unlink queued by a budget pass runs before a later write of the same
//!   name, so a page released and then wanted again ends up on disk.
//!
//! The thread takes the index lock only to record what it did, never across a
//! write.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex as StdMutex};

use shiso_core::TokenId;

use crate::store::format::{self, EntryWrite, PageWrite};
use crate::store::key::{EntryKey, Identity, PageKey};
use crate::store::reason::Reason;
use crate::store::{Counters, Shared};

/// One piece of work for the writer.
#[derive(Debug)]
pub(crate) enum Op {
    /// Write a page file from bytes already copied off the device.
    Page {
        key: PageKey,
        parent: Option<PageKey>,
        tokens: Vec<TokenId>,
        buf: Vec<u8>,
    },
    /// Write an entry file, if every page it lists landed.
    Entry(Box<EntryOp>),
    /// Unlink files a budget pass released.
    Delete(Vec<PathBuf>),
    /// Rewrite an entry's read count and last-read stamp.
    Touch {
        key: EntryKey,
        last_used: u64,
        hits: u32,
    },
    /// Answer once everything queued before this has been done.
    Flush(Sender<()>),
}

/// An entry's write: its header fields, its page list and its two blobs.
///
/// The blobs are shared with the checkpoint slot they came from, so queueing
/// one costs a reference count rather than a copy of the recurrent state.
#[derive(Debug)]
pub(crate) struct EntryOp {
    pub(crate) key: EntryKey,
    pub(crate) pos: u32,
    pub(crate) reason: Reason,
    pub(crate) pages: Vec<PageKey>,
    pub(crate) state: Arc<[u8]>,
    pub(crate) logits: Option<Arc<[u8]>>,
    pub(crate) created_at: u64,
}

/// Page-sized buffers the writer hands back for the next copy, so the pump does
/// not allocate and fault in a fresh page buffer per page.
#[derive(Debug, Default)]
pub(crate) struct Recycle {
    buffers: StdMutex<Vec<Vec<u8>>>,
}

/// Buffers kept for reuse; more than this are freed.
const RECYCLE_KEEP: usize = 16;

impl Recycle {
    /// A buffer of exactly `len` bytes, reused when one is spare.
    pub(crate) fn take(&self, len: usize) -> Vec<u8> {
        let mut buf = self
            .buffers
            .lock()
            .ok()
            .and_then(|mut buffers| buffers.pop())
            .unwrap_or_default();
        buf.resize(len, 0);
        buf
    }

    /// Returns a buffer for reuse.
    pub(crate) fn give(&self, buf: Vec<u8>) {
        if let Ok(mut buffers) = self.buffers.lock()
            && buffers.len() < RECYCLE_KEEP
        {
            buffers.push(buf);
        }
    }
}

/// What the writer needs to know that never changes.
#[derive(Debug)]
pub(crate) struct WriterSetup {
    pub(crate) identity: Identity,
    pub(crate) shared: Arc<Shared>,
    /// Set when the store is being dropped without a drain: every queued write
    /// is skipped so the thread can be joined promptly.
    pub(crate) abandon: Arc<AtomicBool>,
}

/// The thread body. Runs until every sender is gone.
pub(crate) fn run(setup: &WriterSetup, rx: &Receiver<Op>) {
    while let Ok(op) = rx.recv() {
        let abandoning = setup.abandon.load(Ordering::Relaxed);
        match op {
            Op::Page {
                key,
                parent,
                tokens,
                buf,
            } => {
                let len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
                if abandoning {
                    setup.shared.index.lock().page_dropped(&key);
                } else {
                    write_page(setup, key, parent, &tokens, &buf);
                }
                setup
                    .shared
                    .counters
                    .staged_bytes
                    .fetch_sub(len, Ordering::Relaxed);
                setup.shared.recycle.give(buf);
            }
            Op::Entry(entry) => {
                if abandoning {
                    let _ = setup.shared.index.lock().release(&entry.key);
                } else {
                    write_entry(setup, &entry);
                }
            }
            Op::Delete(paths) => unlink_all(&paths),
            Op::Touch {
                key,
                last_used,
                hits,
            } => {
                if !abandoning {
                    let path = setup.shared.index.lock().entry_path(&key);
                    if let Err(error) = format::touch_entry(&path, last_used, hits) {
                        tracing::debug!(entry = %key, %error, "an entry's read stamp was not updated");
                    }
                }
            }
            Op::Flush(reply) => {
                let _ = reply.send(());
            }
        }
    }
}

/// Writes one page, unless every entry that wanted it has gone.
fn write_page(
    setup: &WriterSetup,
    key: PageKey,
    parent: Option<PageKey>,
    tokens: &[TokenId],
    payload: &[u8],
) {
    let path = {
        let index = setup.shared.index.lock();
        if !index.page_wanted(&key) {
            drop(index);
            setup.shared.index.lock().page_dropped(&key);
            return;
        }
        index.page_path(&key)
    };
    let page = PageWrite {
        identity: setup.identity,
        parent,
        tokens,
        payload,
    };
    match format::write_page(&path, &page) {
        Ok(bytes) => {
            let counters: &Counters = &setup.shared.counters;
            counters.bytes_written.fetch_add(bytes, Ordering::Relaxed);
            counters.pages_written.fetch_add(1, Ordering::Relaxed);
            let orphan = setup.shared.index.lock().page_written(&key, bytes);
            if let Some(orphan) = orphan {
                unlink_all(&[orphan]);
            }
        }
        Err(error) => {
            tracing::warn!(page = %key, path = %path.display(), %error, "a page could not be written");
            setup
                .shared
                .counters
                .write_failures
                .fetch_add(1, Ordering::Relaxed);
            setup.shared.index.lock().page_dropped(&key);
        }
    }
}

/// Writes one entry if every page it lists is on disk, and abandons it if not.
fn write_entry(setup: &WriterSetup, entry: &EntryOp) {
    let path = {
        let index = setup.shared.index.lock();
        let ready =
            index.knows(&entry.key) && entry.pages.iter().all(|page| index.page_durable(page));
        if !ready {
            drop(index);
            let released = setup.shared.index.lock().release(&entry.key);
            unlink_all(&released);
            setup
                .shared
                .counters
                .entries_abandoned
                .fetch_add(1, Ordering::Relaxed);
            tracing::debug!(entry = %entry.key, "an entry lost a page on the way to disk; abandoned");
            return;
        }
        index.entry_path(&entry.key)
    };
    let write = EntryWrite {
        identity: setup.identity,
        pos: entry.pos,
        reason: entry.reason,
        pages: &entry.pages,
        logits: entry.logits.as_deref(),
        state: &entry.state,
        created_at: entry.created_at,
    };
    match format::write_entry(&path, &write) {
        Ok(bytes) => {
            let counters = &setup.shared.counters;
            counters.bytes_written.fetch_add(bytes, Ordering::Relaxed);
            counters.entries_written.fetch_add(1, Ordering::Relaxed);
            counters.by_reason[usize::from(entry.reason.as_u8())].fetch_add(1, Ordering::Relaxed);
            setup.shared.index.lock().entry_written(&entry.key, bytes);
            tracing::debug!(
                entry = %entry.key,
                tokens = entry.pos,
                pages = entry.pages.len(),
                bytes,
                reason = ?entry.reason,
                "checkpoint entry written"
            );
        }
        Err(error) => {
            tracing::warn!(entry = %entry.key, path = %path.display(), %error, "an entry could not be written");
            setup
                .shared
                .counters
                .write_failures
                .fetch_add(1, Ordering::Relaxed);
            let released = setup.shared.index.lock().release(&entry.key);
            unlink_all(&released);
        }
    }
}

/// Unlinks every path, treating one already gone as done.
pub(crate) fn unlink_all(paths: &[PathBuf]) {
    for path in paths {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "a released file could not be removed");
            }
        }
    }
}
