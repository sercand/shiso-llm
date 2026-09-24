//! The disk tier: content-addressed pages, separate checkpoints, written behind.
//!
//! The tree keeps a path alive as long as memory allows. This is what is left
//! when memory does not, and when the process does not: every page a stored
//! path covers, as its own file named by the prefix it holds, and every stored
//! checkpoint as a file listing those pages beside the recurrent state at one
//! exact position.
//!
//! # Why pages and checkpoints are separate files
//!
//! A conversation that grows by a turn shares every page but the last few with
//! the turn before. Stored as one file per checkpoint, each turn would rewrite
//! the whole conversation, and a fifty-turn session would write its first turn
//! fifty times. Stored as pages named by content, a turn writes its new pages
//! and one checkpoint, and the pages it shares are listed rather than copied.
//! Traffic and space grow with new tokens.
//!
//! # A hybrid model resumes only at a checkpoint
//!
//! Key/value pages alone cannot resume a model with recurrent layers: those
//! need their state at the exact resume position. So a lookup finds the deepest
//! *checkpoint* whose path is a prefix of the prompt, and pages past it are
//! worth nothing on their own. Nothing here stores a page without an entry that
//! lists it.
//!
//! # Nothing waits on the disk
//!
//! Accepting an entry only reserves its bytes and pins the pool pages it still
//! needs copied. The scheduler thread copies those pages off the device a few at
//! a time between steps ([`crate::KvCache::pump`]) - bounded by a staging
//! budget - and one writer thread does every write, sync and rename. A queue
//! that is full drops its least valuable job instead of blocking anything.
//!
//! # Identity
//!
//! Every key folds in the build's identity: the layout words and the kernel set.
//! A file written by a build whose kernels compute different bits is never
//! looked up at all, which is the only safe answer - it would resume into a
//! state this build does not produce and answer differently from a cold run.

mod format;
mod index;
pub(crate) mod key;
mod read;
mod reason;
mod writer;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex as StdMutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use shiso_core::TokenId;

use crate::backing::Backing;
use crate::error::KvError;
use crate::geom::{CacheGeometry, PathLayout};
use crate::ids::PageId;
use crate::pool::PagePool;

pub use key::EntryKey;
pub use reason::Reason;

use format::{ENTRIES_DIR, ENTRY_HEADER_BYTES, PAGE_HEADER_BYTES, PAGES_DIR};
pub(crate) use format::{EntryBody, EntryHead, Unusable};
pub(crate) use index::Hit;
use index::{Index, NotAccepted, Offer, Pricing};
use key::{Identity, PageKey, PathKeys};
use read::PageJob;
pub(crate) use read::{PageFailure, PageStream};
use writer::{EntryOp, Op, Recycle, WriterSetup};

/// Bytes in a mebibyte.
const MIB: u64 = 1024 * 1024;
/// Checkpoints below this many tokens are not stored by default: the file
/// costs more to write and read than the prefill it saves.
pub const DEFAULT_MIN_TOKENS: u32 = 512;
/// Host bytes the pump may have copied and not yet seen written, by default.
pub const DEFAULT_STAGING_MB: u64 = 1024;
/// Jobs that may wait for the pump at once, by default.
pub const DEFAULT_MAX_JOBS: u32 = 32;
/// Page reads a restore keeps in flight: sixteen whole-page reads keep a
/// solid-state drive's queue deep enough to stream at its sequential rate.
const READ_DEPTH: u32 = 16;
/// Read throughput assumed before any restore has been measured: a
/// conservative figure for a local solid-state drive.
const DEFAULT_READ_BPS: f64 = 1.5e9;
/// Idle seconds that halve a stored entry's standing: six hours, long enough
/// that a session's own prefix outlives a working day.
const DISK_DECAY_S: f64 = 6.0 * 3600.0;
/// How much of a new throughput sample the running estimate takes.
const EWMA_WEIGHT: f64 = 0.3;

/// How the disk tier is opened.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DiskConfig {
    /// The directory; created at owner-only mode if absent.
    pub dir: PathBuf,
    /// Bytes the directory may hold. Zero is refused at open rather than read
    /// as a default.
    pub budget_bytes: u64,
    /// Host bytes copied off the device and not yet written, at most.
    pub staging_bytes: u64,
    /// Checkpoints below this many tokens are neither stored nor looked up.
    pub min_tokens: u32,
    /// Jobs that may wait for the pump before the least valuable is dropped.
    pub max_jobs: u32,
    /// What names the arithmetic the device runs: two builds whose kernels
    /// differ must not read each other's rows.
    pub numerics: String,
}

impl DiskConfig {
    /// A tier at `dir` holding at most `budget_bytes`, with the default
    /// staging, floor and queue.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>, budget_bytes: u64) -> Self {
        Self {
            dir: dir.into(),
            budget_bytes,
            staging_bytes: DEFAULT_STAGING_MB * MIB,
            min_tokens: DEFAULT_MIN_TOKENS,
            max_jobs: DEFAULT_MAX_JOBS,
            numerics: String::new(),
        }
    }
}

/// What the disk tier holds and has done, as an operator sees it.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct DiskStats {
    /// The directory.
    pub dir: PathBuf,
    /// Bytes it may hold.
    pub budget_bytes: u64,
    /// Bytes its files occupy.
    pub durable_bytes: u64,
    /// Bytes promised to writes still queued.
    pub reserved_bytes: u64,
    /// Entries of this build a lookup can find.
    pub entries: u32,
    /// Entries accepted and not yet on disk.
    pub pending_entries: u32,
    /// Page files.
    pub pages: u32,
    /// Jobs waiting for the pump.
    pub queued_jobs: u32,
    /// Host bytes copied and not yet written.
    pub staged_bytes: u64,
    /// Bytes written since open: the drive-wear figure.
    pub bytes_written: u64,
    /// Bytes read back by restores since open.
    pub bytes_read: u64,
    /// Page files written since open.
    pub pages_written: u64,
    /// Entry files written since open.
    pub entries_written: u64,
    /// Entries written since open, per [`Reason`], in [`Reason::ALL`] order.
    pub entries_by_reason: [u64; 7],
    /// Writes that failed.
    pub write_failures: u64,
    /// Entries that lost a page on the way to disk.
    pub entries_abandoned: u64,
    /// Jobs dropped because the queue was full.
    pub jobs_dropped: u64,
    /// Entries the budget released.
    pub entries_evicted: u64,
    /// Restores that stood a session up from disk.
    pub restores: u64,
    /// Restores that found a file unusable partway and fell back to prefill.
    pub restore_failures: u64,
    /// Seconds spent restoring.
    pub restore_seconds: f64,
    /// Seconds of prefill the restores replaced, by the prefill estimate.
    pub restore_prefill_seconds: f64,
    /// Lookups that found an entry but judged a prefill cheaper.
    pub restores_declined: u64,
    /// Read throughput the cost model currently assumes, bytes per second.
    pub read_bps: f64,
    /// Prefetches started ahead of a request.
    pub prefetches: u64,
}

/// Counters the writer and the scheduler thread both bump.
#[derive(Debug, Default)]
pub(crate) struct Counters {
    pub(crate) bytes_written: AtomicU64,
    pub(crate) bytes_read: AtomicU64,
    pub(crate) pages_written: AtomicU64,
    pub(crate) entries_written: AtomicU64,
    pub(crate) by_reason: [AtomicU64; 7],
    pub(crate) write_failures: AtomicU64,
    pub(crate) entries_abandoned: AtomicU64,
    pub(crate) jobs_dropped: AtomicU64,
    pub(crate) entries_evicted: AtomicU64,
    pub(crate) staged_bytes: AtomicU64,
    pub(crate) restores: AtomicU64,
    pub(crate) restore_failures: AtomicU64,
    pub(crate) restore_micros: AtomicU64,
    pub(crate) restore_prefill_micros: AtomicU64,
    pub(crate) restores_declined: AtomicU64,
    pub(crate) prefetches: AtomicU64,
}

/// What the writer thread, the scheduler thread and a prefetching caller share.
#[derive(Debug)]
pub(crate) struct Shared {
    pub(crate) index: parking_lot::Mutex<Index>,
    pub(crate) counters: Counters,
    pub(crate) recycle: Recycle,
    identity: Identity,
    page_tokens: u32,
    min_tokens: u32,
    /// The geometry the cache was opened with, which says whether an entry a
    /// prompt ran past may be continued from.
    geometry: CacheGeometry,
    /// Set while a prefetch thread is running; one at a time is plenty.
    prefetching: AtomicBool,
}

/// Why an entry was not queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PersistRefusal {
    /// The cache has no disk tier.
    NotConfigured,
    /// The node has no live checkpoint, so nothing could resume from it.
    NoCheckpoint,
    /// The node's span is not on the tree any more.
    NodeGone,
    /// The path is shorter than the tier's floor.
    BelowMinTokens {
        /// Tokens the path covers.
        tokens: u32,
        /// The floor.
        min: u32,
    },
    /// The entry would not fit the budget even after releasing everything
    /// releasable.
    OverBudget {
        /// Bytes the entry needs.
        required: u64,
        /// Bytes the tier may hold.
        budget: u64,
    },
}

/// What a persist did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PersistOutcome {
    /// Accepted; the pump copies its new pages and the writer lands it.
    Queued {
        /// The entry's name.
        key: EntryKey,
        /// Pages it still needs copied off the device.
        pages: u32,
    },
    /// The same entry is already queued or on disk.
    AlreadyStored {
        /// The entry's name.
        key: EntryKey,
    },
    /// Not queued, and why.
    Refused(PersistRefusal),
}

impl PersistOutcome {
    /// The entry's name when it is queued or stored.
    #[must_use]
    pub const fn key(&self) -> Option<EntryKey> {
        match self {
            Self::Queued { key, .. } | Self::AlreadyStored { key } => Some(*key),
            Self::Refused(_) => None,
        }
    }
}

/// What one pump call did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PumpReport {
    /// Pages copied off the device and handed to the writer.
    pub pages: u32,
    /// Entries handed to the writer.
    pub entries: u32,
    /// Jobs still waiting.
    pub queued: u32,
}

/// One page a job still has to copy: its place in the path, its name, and the
/// pool page holding its rows, pinned until the copy is made.
#[derive(Debug)]
struct Copy {
    index: usize,
    key: PageKey,
    page: PageId,
}

/// An accepted entry waiting for the pump.
///
/// Holds a pool reference on every page it still needs copied, which is what
/// keeps those rows from being handed to another sequence before they reach
/// disk. `Drop` returns them, so a dropped job can never strand a page.
#[derive(Debug)]
pub(crate) struct Job {
    key: EntryKey,
    pos: u32,
    reason: Reason,
    pages: Vec<PageKey>,
    tokens: Vec<TokenId>,
    copies: VecDeque<Copy>,
    state: Arc<[u8]>,
    logits: Option<Arc<[u8]>>,
    created_at: u64,
    /// Tokens of prefill per byte this job will write; the drop policy's price.
    value: f64,
    pool: Arc<PagePool>,
}

impl Job {
    /// Whether this job pins `page`.
    pub(crate) fn pins(&self, page: PageId) -> bool {
        self.copies.iter().any(|copy| copy.page == page)
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        for copy in self.copies.drain(..) {
            self.pool.unref(copy.page);
        }
    }
}

/// What a persist hands the store, read off the tree and the slot.
#[derive(Debug)]
pub(crate) struct PersistSpec<'a> {
    pub(crate) tokens: &'a [TokenId],
    pub(crate) page_ids: &'a [PageId],
    pub(crate) state: Arc<[u8]>,
    pub(crate) logits: Option<Arc<[u8]>>,
    pub(crate) reason: Reason,
}

/// The open tier.
#[derive(Debug)]
pub(crate) struct DiskStore {
    root: PathBuf,
    identity: Identity,
    pub(crate) shared: Arc<Shared>,
    tx: Option<Sender<Op>>,
    writer: Option<JoinHandle<()>>,
    abandon: Arc<AtomicBool>,
    staging_cap: u64,
    max_jobs: usize,
    budget: u64,
    pub(crate) jobs: VecDeque<Job>,
    read_bps: f64,
    page_tokens: u32,
    page_bytes: u64,
}

impl DiskStore {
    /// Opens the tier: creates the directory, rebuilds the index from what is
    /// there, brings it inside the budget and starts the writer.
    ///
    /// # Errors
    ///
    /// [`KvError::BadGeometry`] for a zero budget or staging size, and
    /// [`KvError::Io`] when the directory cannot be created or the writer
    /// thread cannot start.
    pub(crate) fn open(
        config: &DiskConfig,
        geometry: &CacheGeometry,
        layout: &PathLayout,
    ) -> Result<Self, KvError> {
        if config.budget_bytes == 0 {
            return Err(KvError::bad_geometry("disk budget", "is zero"));
        }
        if config.staging_bytes < geometry.page_bytes {
            return Err(KvError::bad_geometry(
                "disk staging",
                "is smaller than one page, so no page could ever be copied",
            ));
        }
        for sub in [PAGES_DIR, ENTRIES_DIR] {
            let dir = config.dir.join(sub);
            format::ensure_dir(&dir).map_err(|source| KvError::io(&dir, source))?;
        }
        let identity = Identity::of(layout, geometry, &config.numerics);
        let pricing = Pricing {
            page_tokens: geometry.page_tokens,
            prefill_s_per_token: geometry.prefill_s_per_token,
            decay_s: DISK_DECAY_S,
        };
        let mut index = Index::empty(&config.dir, config.budget_bytes, pricing);
        let started = Instant::now();
        let scanned = index.scan(identity);
        let released = index.fit_budget(format::now_secs());
        writer::unlink_all(&released);
        log_legacy(&config.dir);
        tracing::info!(
            dir = %config.dir.display(),
            entries = scanned.entries,
            foreign = scanned.foreign,
            removed = scanned.removed,
            released = released.len(),
            bytes = index.durable_bytes,
            budget = config.budget_bytes,
            scan_ms = started.elapsed().as_millis(),
            numerics = %config.numerics,
            "disk tier open"
        );
        let shared = Arc::new(Shared {
            index: parking_lot::Mutex::new(index),
            counters: Counters::default(),
            recycle: Recycle::default(),
            identity,
            page_tokens: geometry.page_tokens,
            min_tokens: config.min_tokens,
            geometry: *geometry,
            prefetching: AtomicBool::new(false),
        });
        let (tx, rx) = channel();
        let abandon = Arc::new(AtomicBool::new(false));
        let setup = WriterSetup {
            identity,
            shared: Arc::clone(&shared),
            abandon: Arc::clone(&abandon),
        };
        let writer = std::thread::Builder::new()
            .name("shiso-kv-write".to_owned())
            .spawn(move || writer::run(&setup, &rx))
            .map_err(|source| KvError::io(&config.dir, source))?;
        Ok(Self {
            root: config.dir.clone(),
            identity,
            shared,
            tx: Some(tx),
            writer: Some(writer),
            abandon,
            staging_cap: config.staging_bytes,
            max_jobs: usize::try_from(config.max_jobs.max(1)).unwrap_or(usize::MAX),
            budget: config.budget_bytes,
            jobs: VecDeque::new(),
            read_bps: DEFAULT_READ_BPS,
            page_tokens: geometry.page_tokens,
            page_bytes: geometry.page_bytes,
        })
    }

    /// The floor below which nothing is stored or looked up.
    #[must_use]
    pub(crate) fn min_tokens(&self) -> u32 {
        self.shared.min_tokens
    }

    /// The read throughput the cost model assumes, bytes per second.
    #[must_use]
    pub(crate) const fn read_bps(&self) -> f64 {
        self.read_bps
    }

    /// Queues one operation for the writer.
    fn send(&self, op: Op) {
        if let Some(tx) = self.tx.as_ref()
            && tx.send(op).is_err()
        {
            tracing::error!("the disk writer has stopped; a write was lost");
        }
    }

    /// The key a checkpoint at the end of `tokens` would be stored under.
    #[must_use]
    pub(crate) fn entry_key_of(&self, tokens: &[TokenId]) -> Option<EntryKey> {
        PathKeys::new(self.identity, tokens, self.page_tokens)
            .entry_at(u32::try_from(tokens.len()).unwrap_or(u32::MAX))
    }

    /// Whether an entry is queued or on disk.
    #[must_use]
    pub(crate) fn knows(&self, key: &EntryKey) -> bool {
        self.shared.index.lock().knows(key)
    }

    /// The deepest stored entry on `tokens`' path at or above `floor`.
    #[must_use]
    pub(crate) fn lookup(&self, tokens: &[TokenId], floor: u32) -> Option<Hit> {
        let len = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        if len < self.shared.min_tokens {
            return None;
        }
        let keys = PathKeys::new(self.identity, tokens, self.page_tokens);
        self.shared.index.lock().lookup(
            &keys,
            len,
            floor.max(self.shared.min_tokens),
            |pos, logits| self.shared.geometry.continuable(pos, logits),
        )
    }

    /// Bytes an entry file for this spec will occupy.
    fn entry_bytes(pages: usize, spec: &PersistSpec<'_>) -> u64 {
        let fixed = ENTRY_HEADER_BYTES + pages * key::KEY_BYTES;
        u64::try_from(fixed + spec.state.len() + spec.logits.as_ref().map_or(0, |row| row.len()))
            .unwrap_or(u64::MAX)
    }

    /// The reservation for a page file: a full page's header, ids and rows.
    fn page_file_bytes(&self) -> u64 {
        u64::try_from(PAGE_HEADER_BYTES)
            .unwrap_or(u64::MAX)
            .saturating_add(u64::from(self.page_tokens) * 4)
            .saturating_add(self.page_bytes)
    }

    /// Accepts one checkpoint for writing and queues its job.
    ///
    /// `pool` is the pool `spec.page_ids` belong to: the job takes one
    /// reference on every page it still has to copy.
    pub(crate) fn persist(
        &mut self,
        spec: &PersistSpec<'_>,
        pool: &Arc<PagePool>,
    ) -> PersistOutcome {
        let pos = u32::try_from(spec.tokens.len()).unwrap_or(u32::MAX);
        if pos < self.shared.min_tokens || pos == 0 {
            return PersistOutcome::Refused(PersistRefusal::BelowMinTokens {
                tokens: pos,
                min: self.shared.min_tokens,
            });
        }
        let keys = PathKeys::new(self.identity, spec.tokens, self.page_tokens);
        let pages = keys.pages_to(pos);
        let Some(key) = keys.entry_at(pos) else {
            return PersistOutcome::Refused(PersistRefusal::NodeGone);
        };
        if pages.len() != spec.page_ids.len() {
            return PersistOutcome::Refused(PersistRefusal::NodeGone);
        }
        let now = format::now_secs();
        let offer = Offer {
            key,
            pos,
            pages: &pages,
            has_logits: spec.logits.is_some(),
            entry_bytes: Self::entry_bytes(pages.len(), spec),
            page_file_bytes: self.page_file_bytes(),
            now,
        };
        let (accepted, copies) = {
            let mut index = self.shared.index.lock();
            let accepted = match index.accept(&offer) {
                Ok(accepted) => accepted,
                Err(NotAccepted::AlreadyStored) => return PersistOutcome::AlreadyStored { key },
                Err(NotAccepted::OverBudget { required, budget }) => {
                    let orphaned = index.take_orphaned();
                    drop(index);
                    if !orphaned.is_empty() {
                        self.send(Op::Delete(orphaned));
                    }
                    return PersistOutcome::Refused(PersistRefusal::OverBudget {
                        required,
                        budget,
                    });
                }
            };
            let copies: VecDeque<Copy> = pages
                .iter()
                .enumerate()
                .filter(|(_, page)| !index.page_ready(page))
                .map(|(at, page)| Copy {
                    index: at,
                    key: *page,
                    page: spec.page_ids[at],
                })
                .collect();
            (accepted, copies)
        };
        if accepted.evicted > 0 {
            self.shared
                .counters
                .entries_evicted
                .fetch_add(u64::from(accepted.evicted), Ordering::Relaxed);
        }
        if !accepted.deletions.is_empty() {
            self.send(Op::Delete(accepted.deletions));
        }
        for copy in &copies {
            pool.add_holder(copy.page);
        }
        let new_bytes = u64::try_from(copies.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(self.page_file_bytes())
            .saturating_add(offer.entry_bytes)
            .max(1);
        let n_copies = u32::try_from(copies.len()).unwrap_or(u32::MAX);
        self.jobs.push_back(Job {
            key,
            pos,
            reason: spec.reason,
            pages,
            tokens: spec.tokens.to_vec(),
            copies,
            state: Arc::clone(&spec.state),
            logits: spec.logits.clone(),
            created_at: now,
            value: f64::from(pos) / as_f64(new_bytes),
            pool: Arc::clone(pool),
        });
        self.enforce_queue();
        PersistOutcome::Queued {
            key,
            pages: n_copies,
        }
    }

    /// Drops the least valuable jobs while the queue is over its bound.
    fn enforce_queue(&mut self) {
        while self.jobs.len() > self.max_jobs {
            let Some(victim) = self
                .jobs
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.value.total_cmp(&b.value))
                .map(|(at, _)| at)
            else {
                return;
            };
            self.drop_job(victim);
        }
    }

    /// Drops one queued job: its entry is released and its pins returned.
    pub(crate) fn drop_job(&mut self, at: usize) {
        let Some(job) = self.jobs.remove(at) else {
            return;
        };
        let released = self.shared.index.lock().release(&job.key);
        if !released.is_empty() {
            self.send(Op::Delete(released));
        }
        self.shared
            .counters
            .jobs_dropped
            .fetch_add(1, Ordering::Relaxed);
        tracing::debug!(entry = %job.key, tokens = job.pos, "a queued checkpoint write was dropped");
    }

    /// Copies queued pages off the device and hands them, and every entry whose
    /// pages are all handed over, to the writer.
    ///
    /// Stops at `max_bytes` of copies, and - unless `force` - wherever the
    /// staging budget is full: the writer frees staging as pages land, and the
    /// next call carries on.
    ///
    /// # Errors
    ///
    /// Whatever the byte mover reports for a copy. The job whose copy failed is
    /// dropped; the others stay queued.
    pub(crate) fn pump(
        &mut self,
        backing: &dyn Backing,
        max_bytes: u64,
        only: Option<usize>,
        force: bool,
    ) -> Result<PumpReport, KvError> {
        let mut report = PumpReport::default();
        let mut spent = 0u64;
        let page_len = usize::try_from(self.page_bytes).unwrap_or(usize::MAX);
        let at = only.unwrap_or(0);
        while at < self.jobs.len() {
            let mut stalled = false;
            while let Some(copy) = self.jobs[at].copies.front() {
                let (index, key, page) = (copy.index, copy.key, copy.page);
                if !self.shared.index.lock().claim_page(&key) {
                    // Copied by another job, already on disk, or released.
                    if let Some(done) = self.jobs[at].copies.pop_front() {
                        self.jobs[at].pool.unref(done.page);
                    }
                    continue;
                }
                let staged = self.shared.counters.staged_bytes.load(Ordering::Relaxed);
                if !force && (spent >= max_bytes || staged + self.page_bytes > self.staging_cap) {
                    // Put the claim back: nothing was copied.
                    self.shared.index.lock().unclaim_page(&key);
                    stalled = true;
                    break;
                }
                let mut buf = self.shared.recycle.take(page_len);
                if let Err(error) = backing.page_to_host(page, &mut buf) {
                    self.shared.index.lock().unclaim_page(&key);
                    self.shared.recycle.give(buf);
                    self.drop_job(at);
                    return Err(error);
                }
                let job = &self.jobs[at];
                let parent = index.checked_sub(1).map(|up| job.pages[up]);
                let start = index * usize::try_from(self.page_tokens).unwrap_or(usize::MAX);
                let end = (start + usize::try_from(self.page_tokens).unwrap_or(usize::MAX))
                    .min(job.tokens.len());
                let tokens = job.tokens[start..end].to_vec();
                self.shared
                    .counters
                    .staged_bytes
                    .fetch_add(self.page_bytes, Ordering::Relaxed);
                self.send(Op::Page {
                    key,
                    parent,
                    tokens,
                    buf,
                });
                spent += self.page_bytes;
                report.pages += 1;
                if let Some(done) = self.jobs[at].copies.pop_front() {
                    self.jobs[at].pool.unref(done.page);
                }
            }
            if stalled {
                break;
            }
            // Every page is handed over or durable: the entry can follow.
            let ready = {
                let index = self.shared.index.lock();
                self.jobs[at]
                    .pages
                    .iter()
                    .all(|page| index.page_ready(page))
            };
            let Some(job) = self.jobs.remove(at) else {
                break;
            };
            if ready {
                self.send(Op::Entry(Box::new(EntryOp {
                    key: job.key,
                    pos: job.pos,
                    reason: job.reason,
                    pages: job.pages.clone(),
                    state: Arc::clone(&job.state),
                    logits: job.logits.clone(),
                    created_at: job.created_at,
                })));
                report.entries += 1;
            } else {
                // A page this job counted on was released under it; the entry
                // cannot be whole.
                let released = self.shared.index.lock().release(&job.key);
                if !released.is_empty() {
                    self.send(Op::Delete(released));
                }
                self.shared
                    .counters
                    .entries_abandoned
                    .fetch_add(1, Ordering::Relaxed);
            }
            if only.is_some() {
                break;
            }
        }
        report.queued = u32::try_from(self.jobs.len()).unwrap_or(u32::MAX);
        Ok(report)
    }

    /// Reads an entry's head for a restore. A file that turns out unusable is
    /// released from the index and unlinked.
    pub(crate) fn read_head(&self, key: EntryKey) -> Result<EntryHead, Unusable> {
        let path = self.shared.index.lock().entry_path(&key);
        format::read_entry_head(&path, key, self.identity).inspect_err(|why| self.forget(key, *why))
    }

    /// Starts reading an entry's body on a thread of its own, so the state
    /// arrives while the pages stream in.
    pub(crate) fn read_body(
        &self,
        head: EntryHead,
    ) -> std::io::Result<JoinHandle<Result<EntryBody, Unusable>>> {
        let shared = Arc::clone(&self.shared);
        std::thread::Builder::new()
            .name("shiso-kv-body".to_owned())
            .spawn(move || {
                let body = head.read_body()?;
                shared
                    .counters
                    .bytes_read
                    .fetch_add(head.header.file_bytes(), Ordering::Relaxed);
                Ok(body)
            })
    }

    /// Starts every read a restore of `hit` needs: the page stream from page
    /// `from` on, and the entry's body on a thread of its own.
    ///
    /// The page reads start first, from the index's copy of the page list, so
    /// the drive is already busy while the entry's head is read and compared
    /// with it; the body - the recurrent state, a hundred mebibytes - then
    /// arrives while the pages upload. `None` when the entry proved unusable
    /// (it is discarded) or a reader thread could not start.
    pub(crate) fn start_restore(
        &self,
        hit: &Hit,
        tokens: &[TokenId],
        from: usize,
    ) -> Option<(PageStream, JoinHandle<Result<EntryBody, Unusable>>)> {
        let stream = match self.stream_pages(&hit.pages, tokens, from) {
            Ok(stream) => stream,
            Err(error) => {
                tracing::warn!(%error, "the restore reader could not start");
                self.restore_failed();
                return None;
            }
        };
        let Ok(head) = self.read_head(hit.key) else {
            self.restore_failed();
            return None;
        };
        if head.pages != hit.pages {
            self.forget(hit.key, Unusable::Malformed);
            self.restore_failed();
            return None;
        }
        match self.read_body(head) {
            Ok(body) => Some((stream, body)),
            Err(error) => {
                tracing::warn!(%error, "the restore's state reader could not start");
                self.restore_failed();
                None
            }
        }
    }

    /// Waits for a restore's body and checks its state is `want_state` bytes.
    /// `None` when it is not usable; the entry is discarded.
    pub(crate) fn finish_body(
        &self,
        hit: &Hit,
        body: JoinHandle<Result<EntryBody, Unusable>>,
        want_state: usize,
    ) -> Option<EntryBody> {
        let why = match body.join() {
            Ok(Ok(entry)) if entry.state.len() == want_state => return Some(entry),
            Ok(Ok(_)) => Some(Unusable::Malformed),
            Ok(Err(why)) => Some(why),
            Err(_) => None,
        };
        if let Some(why) = why {
            self.forget(hit.key, why);
        }
        self.restore_failed();
        None
    }

    /// Releases an entry that proved unusable, and unlinks its files.
    pub(crate) fn forget(&self, key: EntryKey, why: Unusable) {
        tracing::warn!(entry = %key, ?why, "a stored checkpoint is unusable; discarding it");
        let released = self.shared.index.lock().release(&key);
        if !released.is_empty() {
            self.send(Op::Delete(released));
        }
    }

    /// Starts streaming pages `from..` of a restored path, checked against
    /// `tokens`.
    pub(crate) fn stream_pages(
        &self,
        pages: &[PageKey],
        tokens: &[TokenId],
        from: usize,
    ) -> std::io::Result<PageStream> {
        let pt = usize::try_from(self.page_tokens).unwrap_or(usize::MAX);
        let jobs = {
            let index = self.shared.index.lock();
            pages
                .iter()
                .enumerate()
                .skip(from)
                .map(|(at, key)| {
                    let start = (at * pt).min(tokens.len());
                    let end = ((at + 1) * pt).min(tokens.len());
                    PageJob {
                        index: at,
                        path: index.page_path(key),
                        key: *key,
                        parent: at.checked_sub(1).map(|up| pages[up]),
                        tokens: tokens[start..end].to_vec(),
                    }
                })
                .collect()
        };
        PageStream::start(jobs, self.identity, self.page_bytes, READ_DEPTH)
    }

    /// Records a finished restore: its bytes and time feed the throughput
    /// estimate, and the entry's read stamp is updated.
    pub(crate) fn restored(&mut self, key: EntryKey, bytes: u64, took: Duration, prefill_s: f64) {
        let counters = &self.shared.counters;
        counters.restores.fetch_add(1, Ordering::Relaxed);
        counters
            .restore_micros
            .fetch_add(duration_micros(took), Ordering::Relaxed);
        counters.restore_prefill_micros.fetch_add(
            duration_micros(Duration::from_secs_f64(prefill_s.max(0.0))),
            Ordering::Relaxed,
        );
        let seconds = took.as_secs_f64();
        if seconds > 0.0 && bytes > 0 {
            let sample = as_f64(bytes) / seconds;
            self.read_bps = self.read_bps * (1.0 - EWMA_WEIGHT) + sample * EWMA_WEIGHT;
        }
        let touched = self.shared.index.lock().touch(&key, format::now_secs());
        if let Some((last_used, hits)) = touched {
            self.send(Op::Touch {
                key,
                last_used,
                hits,
            });
        }
    }

    /// Counts a restore that failed partway.
    pub(crate) fn restore_failed(&self) {
        self.shared
            .counters
            .restore_failures
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Counts a lookup whose restore the cost model declined.
    pub(crate) fn restore_declined(&self) {
        self.shared
            .counters
            .restores_declined
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Adds page bytes a restore read to the counters.
    pub(crate) fn count_read(&self, bytes: u64) {
        self.shared
            .counters
            .bytes_read
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Waits until every write queued so far has landed, or `timeout` passes.
    /// Returns whether the writer caught up.
    pub(crate) fn flush(&self, timeout: Duration) -> bool {
        let (tx, rx) = channel();
        self.send(Op::Flush(tx));
        rx.recv_timeout(timeout).is_ok()
    }

    /// The tier's numbers.
    #[must_use]
    pub(crate) fn stats(&self) -> DiskStats {
        let index = self.shared.index.lock();
        let c = &self.shared.counters;
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        let mut by_reason = [0u64; 7];
        for (slot, counter) in by_reason.iter_mut().zip(c.by_reason.iter()) {
            *slot = load(counter);
        }
        DiskStats {
            dir: self.root.clone(),
            budget_bytes: self.budget,
            durable_bytes: index.durable_bytes,
            reserved_bytes: index.reserved_bytes,
            entries: index.usable_entries(),
            pending_entries: index.pending_entries(),
            pages: u32::try_from(index.pages.len()).unwrap_or(u32::MAX),
            queued_jobs: u32::try_from(self.jobs.len()).unwrap_or(u32::MAX),
            staged_bytes: load(&c.staged_bytes),
            bytes_written: load(&c.bytes_written),
            bytes_read: load(&c.bytes_read),
            pages_written: load(&c.pages_written),
            entries_written: load(&c.entries_written),
            entries_by_reason: by_reason,
            write_failures: load(&c.write_failures),
            entries_abandoned: load(&c.entries_abandoned),
            jobs_dropped: load(&c.jobs_dropped),
            entries_evicted: load(&c.entries_evicted),
            restores: load(&c.restores),
            restore_failures: load(&c.restore_failures),
            restore_seconds: micros_secs(load(&c.restore_micros)),
            restore_prefill_seconds: micros_secs(load(&c.restore_prefill_micros)),
            restores_declined: load(&c.restores_declined),
            read_bps: self.read_bps,
            prefetches: load(&c.prefetches),
        }
    }

    /// A handle a request thread can warm the page cache through.
    #[must_use]
    pub(crate) fn prefetcher(&self) -> Prefetcher {
        Prefetcher {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl Drop for DiskStore {
    fn drop(&mut self) {
        // Jobs that never reached the writer go with their pins; what the
        // writer already has is written unless the caller gave up on a drain.
        self.jobs.clear();
        drop(self.tx.take());
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

impl DiskStore {
    /// Makes every write still queued a no-op, so dropping the store returns
    /// promptly. For a shutdown whose drain ran out of time.
    pub(crate) fn abandon_queued(&self) {
        self.abandon.store(true, Ordering::Relaxed);
    }
}

/// Warms the page cache for a request that has not been scheduled yet.
///
/// A request that waits in the queue behind others would otherwise read its
/// stored path cold the moment it is admitted. This reads the files it will
/// need on a background thread while it waits, so the restore reads from
/// memory. It touches only the index, never the cache lock, so a request thread
/// may call it.
#[derive(Debug, Clone)]
pub struct Prefetcher {
    shared: Arc<Shared>,
}

impl Prefetcher {
    /// Starts reading the files the deepest stored entry on `tokens`' path
    /// would restore from. Returns whether a prefetch started; one already
    /// running, or nothing stored, is `false`.
    #[must_use = "whether a prefetch started is the caller's to log or ignore"]
    pub fn prefetch(&self, tokens: &[TokenId]) -> bool {
        let len = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        if len < self.shared.min_tokens {
            return false;
        }
        let keys = PathKeys::new(self.shared.identity, tokens, self.shared.page_tokens);
        let paths: Vec<PathBuf> = {
            let index = self.shared.index.lock();
            let Some(hit) = index.lookup(&keys, len, self.shared.min_tokens, |pos, logits| {
                self.shared.geometry.continuable(pos, logits)
            }) else {
                return false;
            };
            std::iter::once(index.entry_path(&hit.key))
                .chain(hit.pages.iter().map(|page| index.page_path(page)))
                .collect()
        };
        if self.shared.prefetching.swap(true, Ordering::AcqRel) {
            return false;
        }
        self.shared
            .counters
            .prefetches
            .fetch_add(1, Ordering::Relaxed);
        let shared = Arc::clone(&self.shared);
        let spawned = std::thread::Builder::new()
            .name("shiso-kv-prefetch".to_owned())
            .spawn(move || {
                let mut reader = shiso_ple::bulk::BulkReader::new(READ_DEPTH);
                let buffers = StdMutex::new(Vec::<Vec<u8>>::new());
                for batch in paths.chunks(usize::try_from(READ_DEPTH).unwrap_or(1)) {
                    let mut reads: Vec<shiso_ple::bulk::FileRead> = batch
                        .iter()
                        .map(|path| {
                            let buf = buffers
                                .lock()
                                .ok()
                                .and_then(|mut spare| spare.pop())
                                .unwrap_or_default();
                            shiso_ple::bulk::FileRead::new(path.clone(), buf)
                        })
                        .collect();
                    reader.read(&mut reads);
                    if let Ok(mut spare) = buffers.lock() {
                        spare.extend(reads.into_iter().map(|read| read.buf));
                    }
                }
                shared.prefetching.store(false, Ordering::Release);
            });
        if spawned.is_err() {
            self.shared.prefetching.store(false, Ordering::Release);
            return false;
        }
        true
    }
}

/// Mentions a directory still holding files of the previous format, once.
fn log_legacy(dir: &Path) {
    let Ok(listing) = std::fs::read_dir(dir) else {
        return;
    };
    let (count, bytes) = listing
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".kv"))
        .filter_map(|entry| entry.metadata().ok())
        .fold((0u64, 0u64), |(n, b), meta| (n + 1, b + meta.len()));
    if count > 0 {
        tracing::info!(
            dir = %dir.display(),
            files = count,
            bytes,
            "files of the previous disk format are ignored; delete them to reclaim the space"
        );
    }
}

/// A duration in whole microseconds, saturating.
fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

/// Microseconds as seconds.
fn micros_secs(micros: u64) -> f64 {
    as_f64(micros) / 1e6
}

/// A byte count as a float; exact below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "exact below 2^53, which no byte count here reaches"
)]
const fn as_f64(value: u64) -> f64 {
    value as f64
}
