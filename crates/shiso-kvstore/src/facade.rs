//! The cache itself: one tree, one pool, one slot table, five verbs.
//!
//! Everything else in this crate is a mechanism. This module is the only place that
//! decides *when* a mechanism runs, and the five verbs are the whole interface to it:
//!
//! | verb | question it answers | mutates the tree? |
//! |---|---|---|
//! | [`KvCache::plan`] | what would a prompt reuse, and from where? | no |
//! | [`KvCache::lookup`] | give me a frontier I can generate from | no, it only claims pages |
//! | [`KvCache::commit`] | the frontier I just prefilled is worth keeping | yes, by adding |
//! | [`KvCache::evict_to`] | make room for `n` more pages | yes, by dropping leaves |
//! | [`KvCache::persist`] / [`KvCache::resume`] | put a checkpoint on disk, stand on the deepest one there or in the tree | yes, on a disk resume |
//!
//! # Two locks, and where this module takes them
//!
//! `inner` guards the tree, the slot table and the byte mover; the pool has its own
//! mutex. Every method here takes `inner` exactly once, at the top, and holds it for
//! the whole operation, which is what makes the tree consistent across a
//! walk-then-split-then-add sequence. The pool lock is taken only from inside that
//! critical section, never the other way round, and that direction is what lets a
//! [`PageTable`] or [`LiveGuard`] be dropped mid-method without deadlocking -- see
//! [`crate::pool`].
//!
//! # Time is an argument
//!
//! Every verb that touches a `hit` stamp takes `now: f64`. That is not fussiness: the
//! utility formulas decay against it, so a test that wants to prove "a crowded
//! checkpoint is evicted before a lone one" has to be able to move the clock. A cache
//! that read a clock internally could only prove that after ten minutes of sleeping.
//! The engine passes its own monotonic seconds; the disk tier's timestamps are wall
//! clock, because they outlive the process.

use std::sync::Arc;
use std::time::{Duration, Instant};

use shiso_core::TokenId;

use crate::backing::Backing;
use crate::ckpt::{Admission, CkptRef, CkptStore};
use crate::error::KvError;
use crate::evict::ckpt_utility;
use crate::geom::CacheGeometry;
use crate::ids::{NodeId, SlotId, to_u32};
use crate::page_table::PageTable;
use crate::pool::{InnerGuard, PagePool};
use crate::store::{
    DiskConfig, DiskStats, DiskStore, EntryKey, Hit, PersistOutcome, PersistRefusal, PersistSpec,
    Prefetcher, PumpReport, Reason,
};
use crate::tree::SpanTree;

/// Where a resume would come from, which changes what the engine has to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReuseSource {
    /// Nothing usable: prefill from token zero.
    Cold,
    /// The caller's own frontier already stands where the prompt ends, so no
    /// recomputation at all is needed -- only the logits row it is standing on.
    Live,
    /// A checkpoint on the path the caller is already standing on: the same
    /// conversation rewinding to an earlier message boundary.
    Checkpoint,
    /// A checkpoint the tree remembered from some other conversation.
    Tree,
    /// A checkpoint read back from the disk tier.
    Disk,
}

/// What a prompt would reuse, without taking anything.
///
/// A pure answer about positions, so it is safe to ask speculatively on every
/// request: it claims no page, drops nothing, and leaves the span a match stopped
/// inside unsplit. The boundary a match stops inside is created by the `commit` that
/// follows, which is the only verb with the mutable borrow splitting needs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Lookup {
    /// The deepest span the tokens matched.
    pub node: NodeId,
    /// Positions matched, after any clamp the caller asked for.
    pub matched: u32,
    /// The checkpoint a resume would load, if there is one worth loading.
    pub ckpt: Option<CkptRef>,
    /// The node that checkpoint belongs to.
    pub ckpt_node: Option<NodeId>,
    /// The position the checkpoint's state is valid at.
    pub ckpt_pos: u32,
    /// Which of the four cases this is.
    pub source: ReuseSource,
}

impl Lookup {
    /// Positions a caller may actually reuse: a checkpoint is only usable down to
    /// where its state is valid, and never past what matched.
    #[must_use]
    pub fn reusable(&self) -> u32 {
        match self.source {
            ReuseSource::Live => self.matched,
            ReuseSource::Cold => 0,
            ReuseSource::Checkpoint | ReuseSource::Tree | ReuseSource::Disk => {
                self.ckpt_pos.min(self.matched)
            }
        }
    }
}

/// A frontier handed to the caller: the pages to index and the span they end at.
#[derive(Debug)]
#[non_exhaustive]
pub struct Resume {
    /// What the caller now owns. Dropping it returns everything.
    pub frontier: Frontier,
    /// The checkpoint whose state was loaded, if one was.
    pub ckpt: Option<CkptRef>,
    /// The position the loaded state is valid at -- the caller's new position.
    pub ckpt_pos: u32,
    /// Positions the tree had already, which is what the caller does not recompute.
    pub matched: u32,
    /// Where the state came from, for the caller's own accounting.
    pub source: ReuseSource,
}

/// What a live sequence holds: the node it reached and the pages its path covers.
///
/// Dropping it *is* the release. The page refs return through [`PageTable`]'s `Drop`
/// and the node's live count returns through the guard it holds, so an abandoned
/// request cannot strand a page or leave a node permanently unevictable -- the two
/// leaks a hand-unbalanced mark/unmark pair produced whenever an early return skipped
/// one of the unmark calls.
#[derive(Debug)]
pub struct Frontier {
    live: crate::tree::LiveGuard,
    table: PageTable,
}

impl Frontier {
    /// The deepest span the sequence's tokens matched.
    #[must_use]
    pub fn node(&self) -> NodeId {
        self.live.node()
    }

    /// The page index array the engine uploads for this sequence.
    #[must_use]
    pub fn table(&self) -> &PageTable {
        &self.table
    }

    /// The pages the path covers, which is what page pressure is measured against.
    #[must_use]
    pub fn pages(&self) -> usize {
        self.table.len()
    }

    /// Claims pages until the table covers positions `[0, n_pos)`.
    ///
    /// A sequence that has generated past the end of its last span needs somewhere
    /// for those rows to go before it can commit them, and this is the only way to
    /// get them. Returns `false` without claiming anything when the pool is short, so
    /// the caller can [`KvCache::evict_to`] and retry rather than ending the request.
    ///
    /// # Errors
    ///
    /// [`KvError::PoolEmpty`] if another holder's frontier claimed a page between the
    /// count and the claim -- the pool is shared and this method does not hold `inner`,
    /// so the shortage check is an estimate. Pages already claimed here are released by
    /// `Drop`, so a failed grow leaves the table exactly as long as it was.
    pub fn grow_to(&mut self, n_pos: u32) -> Result<bool, KvError> {
        self.table.grow_to(n_pos)
    }

    /// Shortens the path to at most `n` pages, releasing the rest.
    ///
    /// Used when a conversation is truncated back to a branch point: the pages above
    /// it hold rows for tokens that are no longer in the conversation, and leaving
    /// them held would be a slow leak with a page count.
    pub fn truncate(&mut self, n: usize) {
        self.table.truncate(n);
    }
}

/// What one `commit` should write.
///
/// `state` and `logits` are borrowed rather than owned: both are large (the state
/// blob is tens of mebibytes for a recurrent model) and the engine owns a reusable
/// staging buffer it reuses per request.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct CommitOpts<'a> {
    /// Commit the vocab logits row with the state, so a resume from here samples
    /// identically to a run that never stopped.
    pub with_logits: bool,
    /// A chunk boundary rather than a rule-earned boundary: the checkpoint taken for
    /// it is speculative and is the first thing evicted.
    pub chunk_only: bool,
    /// Message boundaries: positions the engine would rather stop a span at, because
    /// a later request is likely to end there.
    pub hints: &'a [u32],
    /// The opaque recurrent state at the end of the span. Without it there is no
    /// checkpoint to admit, and the span is still committed.
    pub state: Option<&'a [u8]>,
    /// The vocab-sized logits row, committed only at a refresh boundary.
    pub logits: Option<&'a [u8]>,
    /// Queue the checkpoint for the disk tier once it is admitted, filed under
    /// this reason. Ignored for a speculative (chunk-end) checkpoint and on a
    /// cache with no disk tier.
    pub persist: Option<Reason>,
}

impl<'a> CommitOpts<'a> {
    /// A commit that registers the span and admits no checkpoint.
    ///
    /// A constructor rather than a struct literal because the type is
    /// `#[non_exhaustive]`: a crate outside this one cannot write the literal at
    /// all, and every caller of `commit` is outside this one.
    #[must_use]
    pub const fn span(hints: &'a [u32]) -> Self {
        Self {
            with_logits: false,
            chunk_only: false,
            hints,
            state: None,
            logits: None,
            persist: None,
        }
    }

    /// The same commit, carrying a checkpoint blob.
    ///
    /// `earned` says the boundary was one of the rules' own - a message hint or the
    /// branch point - rather than a chunk end the engine happened to stop at. A
    /// chunk end is speculative and is the first thing eviction takes back, which
    /// is the whole difference between the two.
    #[must_use]
    pub const fn checkpoint(mut self, state: &'a [u8], earned: bool) -> Self {
        self.state = Some(state);
        self.chunk_only = !earned;
        self
    }

    /// The same commit, carrying the logits row beside the checkpoint.
    ///
    /// Only meaningful with a checkpoint: the row is what a resume at this position
    /// samples its next token from, and without the state beside it there is
    /// nothing to resume.
    #[must_use]
    pub const fn logits_row(mut self, logits: &'a [u8]) -> Self {
        self.logits = Some(logits);
        self.with_logits = true;
        self
    }

    /// The same commit, asking for the admitted checkpoint to be written to
    /// the disk tier under `reason`.
    ///
    /// The write is behind: this queues it, and the pump and the writer thread
    /// do the copying and the file work later. A commit is never slower for it
    /// than by a reference count and a page pin per page not yet on disk.
    #[must_use]
    pub const fn persist(mut self, reason: Reason) -> Self {
        self.persist = Some(reason);
        self
    }
}

/// What one eviction pass gave back.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct EvictReport {
    /// Pages returned to the pool.
    pub pages_freed: u32,
    /// Spans dropped from the tree.
    pub nodes_dropped: u32,
    /// Checkpoints released, including the ones dropped to make room for one.
    pub ckpts_dropped: u32,
}

/// Everything an operator can ask about the cache without draining it.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CacheStats {
    /// Spans in the tree.
    pub nodes: u32,
    /// Pages in the pool.
    pub pool_pages: u32,
    /// Pages currently held.
    pub pages_used: u32,
    /// Checkpoint slots in use.
    pub ckpt_used: u32,
    /// Slots the store will hold.
    pub ckpt_capacity: u32,
    /// Spans dropped by page pressure.
    pub page_evictions: u64,
    /// Checkpoints evicted, whether for room or for pages.
    pub ckpt_evictions: u64,
    /// Resumes served from a checkpoint the tree held.
    pub resumes_tree: u64,
    /// Resumes served from the disk tier.
    pub resumes_disk: u64,
    /// The prefill rate the resume cost model assumes, seconds per token.
    pub prefill_s_per_token: f64,
    /// The disk tier's own numbers, when one was opened.
    pub disk: Option<DiskStats>,
}

/// The KV cache.
///
/// One tree of token spans, one pool of pages, one table of checkpoint slots, and a
/// byte mover the engine owns. The three are always consulted together, which is why
/// they sit behind one mutex rather than three.
#[derive(Debug)]
pub struct KvCache {
    geometry: CacheGeometry,
    /// Shared with every page table this cache hands out. Held here too so a stats
    /// read does not need the cache lock to ask how full the pool is.
    pool: Arc<PagePool>,
    /// The disk tier's request-thread door, kept outside the cache lock so a
    /// prefetch never waits on a step.
    prefetcher: Option<Prefetcher>,
    inner: parking_lot::Mutex<Inner>,
}

/// Everything the cache lock covers.
#[derive(Debug)]
struct Inner {
    geometry: CacheGeometry,
    /// The same `Arc` the tree and every page table handed out holds.
    pool: Arc<PagePool>,
    tree: SpanTree,
    ckpt: CkptStore,
    /// Held exclusively, and behind the same lock as the tree, because a
    /// copy-on-write must not interleave with the span changes that decided which page
    /// to copy.
    backing: Box<dyn Backing>,
    disk: Option<DiskStore>,
    /// Prefill seconds per token, as measured by the engine when it says so and
    /// as the geometry estimated before that: what a resume from disk is priced
    /// against.
    prefill_s_per_token: f64,
    resumes_tree: u64,
    resumes_disk: u64,
}

/// How much of a new prefill-rate sample the running estimate takes.
const PREFILL_EWMA_WEIGHT: f64 = 0.3;

/// How long a drain waits for the writer to free staging before copying more.
const DRAIN_WAIT: Duration = Duration::from_millis(5);

impl KvCache {
    /// Builds a cache with `pool_pages` pages of key/value storage and room for
    /// `ckpt_slots` checkpoints.
    ///
    /// Both sizes are the engine's decision, made from how much memory it can spare:
    /// the pool is `pool_pages * page_bytes` and the slot table is at worst
    /// `ckpt_slots * ckpt_bytes`. Neither is allocated here -- pages are ids until a
    /// sequence claims one, and a slot's blob appears when it is first filled -- so an
    /// over-generous pool costs a `Vec` of counters, not the bytes.
    ///
    /// `disk` opens the disk tier, which needs the geometry's layout: every key
    /// it writes names the build that computed the bytes. Without it the cache
    /// is purely in-memory and [`Self::persist`] answers
    /// [`PersistRefusal::NotConfigured`].
    ///
    /// # Errors
    ///
    /// [`KvError::BadGeometry`] if the geometry cannot be built from, or a disk
    /// tier is asked for without a layout, and [`KvError::Io`] if the disk
    /// directory cannot be created.
    pub fn open(
        geometry: &CacheGeometry,
        backing: Box<dyn Backing>,
        pool_pages: u32,
        ckpt_slots: u32,
        disk: Option<DiskConfig>,
    ) -> Result<Self, KvError> {
        geometry.validate()?;
        let pool = Arc::new(PagePool::new(pool_pages));
        let tree = SpanTree::new(*geometry, Arc::clone(&pool));
        let ckpt = CkptStore::new(*geometry, ckpt_slots);
        let disk = match disk {
            Some(config) => {
                let layout = geometry.layout.ok_or(KvError::bad_geometry(
                    "layout",
                    "is required by a disk tier, whose keys name the build",
                ))?;
                Some(DiskStore::open(&config, geometry, &layout)?)
            }
            None => None,
        };
        let prefetcher = disk.as_ref().map(DiskStore::prefetcher);
        Ok(Self {
            inner: parking_lot::Mutex::new(Inner {
                geometry: *geometry,
                pool: Arc::clone(&pool),
                tree,
                ckpt,
                backing,
                disk,
                prefill_s_per_token: geometry.prefill_s_per_token,
                resumes_tree: 0,
                resumes_disk: 0,
            }),
            geometry: *geometry,
            pool,
            prefetcher,
        })
    }

    /// The sizes the cache was built from.
    #[must_use]
    pub const fn geometry(&self) -> &CacheGeometry {
        &self.geometry
    }

    /// What a prompt would reuse, without taking a page reference or a slot.
    ///
    /// Cheap enough to call on every request and safe to call speculatively, because it
    /// changes nothing at all: no page claim, no slot, and no span boundary either. The
    /// boundary a match stops inside is created by the [`Self::commit`] that follows, so
    /// a caller that only wanted to know pays nothing for asking.
    #[must_use]
    pub fn plan(&self, tokens: &[TokenId], now: f64) -> Lookup {
        self.plan_with(None, tokens, None, now)
    }

    /// As [`Self::plan`], for a caller that is already standing somewhere.
    ///
    /// The standing frontier is what makes [`ReuseSource::Live`] and
    /// [`ReuseSource::Checkpoint`] reachable at all: both mean "this is the same
    /// conversation", which is a fact about a sequence rather than about a prompt.
    #[must_use]
    pub fn plan_from(&self, standing: &Frontier, tokens: &[TokenId], now: f64) -> Lookup {
        self.plan_with(Some(standing), tokens, None, now)
    }

    /// The full query, with the two things the two-argument forms assume.
    ///
    /// `clamp` is the first position the match must not go past, which the vision path
    /// uses: a run of image-pad tokens is the same token id repeated, so two *different*
    /// pictures of the same length produce identical key runs and would match each
    /// other -- reusing key/value rows computed from the wrong picture, with no error
    /// and a plausible-looking answer. Clamping to the first media item's start keeps
    /// the text before it (system prompt, prior turns), which is where most of the win
    /// is, and re-prefills only the media span. A text-only prompt cannot over-match
    /// past it, because reaching the clamp requires carrying the same pad ids.
    ///
    /// `None` means no clamp.
    #[must_use]
    pub fn plan_with(
        &self,
        standing: Option<&Frontier>,
        tokens: &[TokenId],
        clamp: Option<u32>,
        now: f64,
    ) -> Lookup {
        let _held = InnerGuard::new();
        let inner = self.inner.lock();
        inner.plan_with(standing, tokens, clamp, now)
    }

    /// Move a sequence onto the deepest checkpoint its prompt can use.
    ///
    /// Returns `Ok(None)` when there is nothing to resume from, which is the normal
    /// answer for a first request and not a failure: the caller prefills cold.
    ///
    /// # Errors
    ///
    /// [`KvError::PoolEmpty`] if the one page a resume lands inside cannot be copied
    /// because the pool is full and nothing may be evicted, [`KvError::PathGap`] if the
    /// matched path has a page nobody owns, and whatever the byte mover reports. On
    /// every error path the pages and the pin taken here are released before returning,
    /// so a failed resume leaves the cache exactly as full as it was.
    pub fn lookup(&self, tokens: &[TokenId], now: f64) -> Result<Option<Resume>, KvError> {
        self.lookup_with(None, tokens, None, now)
    }

    /// As [`Self::lookup`], for a caller already standing somewhere and/or holding a
    /// media clamp. See [`Self::plan_with`].
    ///
    /// `Ok(None)` carries [`Self::lookup`]'s meaning unchanged: there is nothing to
    /// resume from, which is the normal answer and not a failure.
    ///
    /// # Errors
    ///
    /// As [`Self::lookup`]: [`KvError::PoolEmpty`] when the one page a resume lands
    /// inside cannot be copied because the pool is full and nothing may be evicted,
    /// [`KvError::PathGap`] when the matched path has a page nobody owns, and whatever
    /// the byte mover reports.
    pub fn lookup_with(
        &self,
        standing: Option<&Frontier>,
        tokens: &[TokenId],
        clamp: Option<u32>,
        now: f64,
    ) -> Result<Option<Resume>, KvError> {
        let _held = InnerGuard::new();
        let mut inner = self.inner.lock();
        inner.lookup_with(standing, tokens, clamp, now)
    }

    /// Hand the tree the span the caller has just produced, and the checkpoint at its
    /// end if the caller brought one.
    ///
    /// Up to the branch point the tokens are already in the tree, so the shared path is
    /// split rather than copied; only above it does this sequence contribute spans and
    /// pages of its own. The walk down is repeated from where the frontier stands
    /// rather than trusting anything from an earlier `plan`, because the tree may have
    /// changed in between -- a split, a prune, another sequence's commit -- and a turn
    /// that re-renders its history can share a longer prefix than the prompt-time match
    /// found. Stopping short there leaves a sibling carrying the new span's first token,
    /// which `add` refuses, and then every later turn of that conversation re-prefills
    /// the whole tail.
    ///
    /// A span that cannot join the tree is logged and dropped, not an error: the caller
    /// keeps its pages and its frontier and the request is fine. What it costs is the
    /// next turn's prefill, which is why it is a `warn` and not a `debug`.
    ///
    /// # Errors
    ///
    /// [`KvError::BufferLen`] if `opts.state` is not a whole checkpoint blob, and
    /// whatever the slot table reports. A refused checkpoint is never an error: the
    /// span is still committed, and the next boundary will come around.
    pub fn commit(
        &self,
        fr: &mut Frontier,
        tokens: &[TokenId],
        q: u32,
        opts: CommitOpts<'_>,
        now: f64,
    ) -> Result<(), KvError> {
        let _held = InnerGuard::new();
        let mut inner = self.inner.lock();
        inner.commit(fr, tokens, q, &opts, now)
    }

    /// Drop whole spans until the pool holds `want_free_pages` free pages.
    ///
    /// Returns as soon as the target is met, and returns what it managed if it ran out
    /// of victims first -- every leaf is either pinned by a live sequence or the caller
    /// is asking for more pages than the cache has. A caller that still cannot get a
    /// page after this is out of memory, not out of patience.
    #[must_use]
    pub fn evict_to(&self, want_free_pages: u32, now: f64) -> EvictReport {
        let _held = InnerGuard::new();
        let mut inner = self.inner.lock();
        inner.free_pages(want_free_pages, now)
    }

    /// The next position worth stopping a chunk at, or `None` when the request ends
    /// before any of them.
    ///
    /// The host asks *before* choosing a chunk end, because a checkpoint snapshot is a
    /// large device-to-host copy that must never be paid speculatively: the engine
    /// snapshots only when this answers `Some(new_end)`. `commit` re-derives the same
    /// answer internally, so a host that never calls this stays correct -- it just pays
    /// that copy on every commit.
    ///
    /// The stop is the earliest of a message-boundary hint, the branch point where this
    /// path stops matching the tree, and -- only while a checkpoint slot is going spare
    /// -- the next multiple of `stride`. See [`crate::next_admission_pos`] for the
    /// ordering rules, including why stride positions are absolute multiples.
    #[must_use]
    pub fn next_admission(
        &self,
        fr: &Frontier,
        tokens: &[TokenId],
        hints: &[u32],
        stride: u32,
    ) -> Option<u32> {
        let _held = InnerGuard::new();
        let inner = self.inner.lock();
        let len = to_u32(tokens.len());
        let pos = inner.tree.node(fr.node()).end;
        if pos >= len {
            return None;
        }
        // The branch point is where this path stops matching the tree, which is only
        // known by walking it from the frontier -- the frontier's own match is behind
        // it by construction.
        let ahead = inner
            .tree
            .walk_down(&tokens[usize::try_from(pos).unwrap_or(tokens.len())..]);
        let branch = pos + ahead.matched;
        let stop = crate::evict::next_admission_pos(
            len,
            pos,
            hints,
            branch,
            stride,
            inner.ckpt.any_free(),
        );
        (stop > pos).then_some(stop)
    }

    /// Copies a checkpoint's opaque state blob into a caller-owned buffer.
    ///
    /// Not `-> Vec<u8>`: the blob is tens of mebibytes and the engine already owns a
    /// reusable staging buffer. `buf.len()` must equal `geometry().ckpt_bytes`, and an
    /// evicted slot is an error rather than an empty buffer -- resuming from a zeroed
    /// recurrent state produces confident nonsense.
    ///
    /// # Errors
    ///
    /// [`KvError::BufferLen`] if `buf` is not exactly a checkpoint blob, and
    /// [`KvError::StaleCheckpoint`] if the slot has since been evicted.
    pub fn ckpt_state_into(&self, ckpt: CkptRef, buf: &mut [u8]) -> Result<(), KvError> {
        let _held = InnerGuard::new();
        let inner = self.inner.lock();
        inner.ckpt.state_into(ckpt, buf)
    }

    /// Copies a checkpoint's logits row into a caller-owned buffer.
    ///
    /// # Errors
    ///
    /// [`KvError::BufferLen`] if `buf` is not the row's length, and
    /// [`KvError::StaleCheckpoint`] if the slot has since been evicted or never had a
    /// row.
    pub fn ckpt_logits_into(&self, ckpt: CkptRef, buf: &mut [u8]) -> Result<(), KvError> {
        let _held = InnerGuard::new();
        let inner = self.inner.lock();
        inner.ckpt.logits_into(ckpt, buf)
    }

    /// What the cache and the disk tier hold, for an operator or a `/cache` endpoint.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let _held = InnerGuard::new();
        let inner = self.inner.lock();
        CacheStats {
            nodes: inner.tree.node_count(),
            pool_pages: self.pool.capacity(),
            pages_used: self.pool.n_used(),
            ckpt_used: inner.ckpt.filled(),
            ckpt_capacity: inner.ckpt.capacity(),
            page_evictions: inner.tree.page_evictions(),
            ckpt_evictions: inner.ckpt.evictions(),
            resumes_tree: inner.resumes_tree,
            resumes_disk: inner.resumes_disk,
            prefill_s_per_token: inner.prefill_s_per_token,
            disk: inner.disk.as_ref().map(DiskStore::stats),
        }
    }

    /// The disk tier's numbers alone, when one is open.
    #[must_use]
    pub fn disk_stats(&self) -> Option<DiskStats> {
        let _held = InnerGuard::new();
        let inner = self.inner.lock();
        inner.disk.as_ref().map(DiskStore::stats)
    }

    /// The request-thread door to the disk tier: warms the page cache for a
    /// request that is still queued. `None` without a disk tier.
    #[must_use]
    pub fn prefetcher(&self) -> Option<Prefetcher> {
        self.prefetcher.clone()
    }
}

impl Inner {
    /// The resume half of `lookup_with`, with the lock held. `Ok(None)` is
    /// the public method's: nothing in the tree to resume from.
    fn lookup_with(
        &mut self,
        standing: Option<&Frontier>,
        tokens: &[TokenId],
        clamp: Option<u32>,
        now: f64,
    ) -> Result<Option<Resume>, KvError> {
        let plan = self.plan_with(standing, tokens, clamp, now);
        let (Some(ckpt), Some(target)) = (plan.ckpt, plan.ckpt_node) else {
            return Ok(None);
        };
        let cpos = ckpt.pos;

        // Pin the path being moved onto *before* anything that can evict. The copy
        // below may need a page, and a page shortage runs the victim scan; if the
        // target were still an unpinned childless leaf at that moment, the scan could
        // drop the very node and checkpoint being restored. Pinning afterwards is the
        // bug, not the ordering.
        let live = self.tree.live_guard(target);
        let pages = match self.tree.page_array_from_path(target, cpos) {
            Ok(pages) => pages,
            Err(error) => {
                drop(live);
                return Err(error);
            }
        };
        let geometry = self.geometry;
        let mut table = PageTable::adopt(Arc::clone(&self.pool), geometry, pages);

        // The page holding `cpos - 1` is shared and immutable, and the next prefill is
        // about to rewrite its tail, so this sequence takes a private copy of that one
        // page. A checkpoint on a page boundary -- every chunk boundary is one -- needs
        // no copy at all, which is why resuming at a chunk boundary is free.
        if !geometry.is_page_start(cpos) && !table.is_empty() {
            let last = table.len() - 1;
            let src = table.as_slice()[last];
            if self.pool.n_free() == 0 {
                self.free_pages(1, now);
            }
            let Some(dst) = self.pool.take() else {
                drop(live);
                return Err(KvError::PoolEmpty);
            };
            if let Err(error) = self.backing.copy_page(dst, src) {
                self.pool.unref(dst);
                drop(live);
                return Err(error);
            }
            table.replace(last, dst);
        }

        self.tree.touch(target, now);
        self.ckpt.touch(ckpt.slot, now);
        Ok(Some(Resume {
            frontier: Frontier { live, table },
            ckpt: Some(ckpt),
            ckpt_pos: cpos,
            matched: plan.matched,
            source: plan.source,
        }))
    }
}

impl Inner {
    /// The query half, with the lock held.
    fn plan_with(
        &self,
        standing: Option<&Frontier>,
        tokens: &[TokenId],
        clamp: Option<u32>,
        now: f64,
    ) -> Lookup {
        let _ = now;
        let walk = self.tree.walk_down(tokens);
        let end = to_u32(tokens.len());
        let mut node = walk.node;
        let mut matched = clamp.map_or(walk.matched, |limit| walk.matched.min(limit));
        // A clamp can land before the span the walk reached, which means the match
        // really ends above it: the frontier is the node whose span contains the clamped
        // position, not the node the walk stopped in.
        while matched < self.tree.node(node).start {
            let Some(parent) = self.tree.node(node).parent else {
                break;
            };
            node = parent;
        }
        matched = matched.min(self.tree.node(node).end);

        // The deepest usable checkpoint on the matched path. A node above the match
        // cannot be resumed from -- its state describes positions the caller is not
        // standing on -- and a checkpoint at the very end of the prompt is only usable
        // with its logits row, because resuming there means answering the next token
        // from that row.
        let mut found: Option<(NodeId, CkptRef)> = None;
        for cand in self.tree.path_to(node).into_iter().rev() {
            let cand_end = self.tree.node(cand).end;
            if cand_end > matched {
                continue;
            }
            let Some(slot) = self.tree.ckpt_at(cand) else {
                continue;
            };
            if !self.ckpt.is_live(slot) {
                continue;
            }
            if cand_end == end && !self.ckpt.has_logits(slot) {
                continue;
            }
            if cand_end < end
                && !self
                    .geometry
                    .continuable(cand_end, self.ckpt.has_logits(slot))
            {
                continue;
            }
            if let Some(ckpt) = self.ckpt.reference(slot, cand_end) {
                found = Some((cand, ckpt));
            }
            break;
        }

        // Whether the caller is already standing where the prompt ends. That is a fact
        // about a sequence, not about a prompt, which is why `plan` can never report it.
        let live_pos = standing.map_or(0, |fr| self.tree.node(fr.node()).end);
        let standing_on_path = standing.is_some_and(|fr| walk.path.contains(&fr.node()));
        let live_logits = standing.is_some_and(|fr| {
            self.tree
                .ckpt_at(fr.node())
                .is_some_and(|slot| self.ckpt.has_logits(slot))
        });
        let live = standing_on_path
            && live_pos > 0
            && matched >= live_pos
            && (live_pos < end || live_logits);

        let ckpt_node = found.map(|(node, _)| node);
        let ckpt_pos = found.map_or(0, |(_, ckpt)| ckpt.pos);
        let source = if found.is_some() && ckpt_pos > live_pos {
            match ckpt_node {
                Some(node) if standing.is_some_and(|fr| self.tree.is_ancestor(node, fr.node())) => {
                    ReuseSource::Checkpoint
                }
                _ => ReuseSource::Tree,
            }
        } else if live {
            ReuseSource::Live
        } else {
            ReuseSource::Cold
        };
        // Standing on the frontier wins outright: the checkpoint it is standing *on* is
        // the one already loaded, so reporting a deeper one would make the caller rewind
        // work it has already done.
        let (ckpt, ckpt_node, ckpt_pos, matched) =
            if source == ReuseSource::Live || source == ReuseSource::Cold {
                (
                    None,
                    None,
                    0,
                    if source == ReuseSource::Live {
                        live_pos
                    } else {
                        matched
                    },
                )
            } else {
                (found.map(|(_, ckpt)| ckpt), ckpt_node, ckpt_pos, matched)
            };
        Lookup {
            node,
            matched,
            ckpt,
            ckpt_node,
            ckpt_pos,
            source,
        }
    }

    /// What every checkpoint currently held is worth, on the same scale as a span.
    ///
    /// Computed before the slot table is mutated, because pricing reads the tree and the
    /// table cannot be read and evicted from at the same time.
    fn ckpt_prices(&self, now: f64) -> Vec<(SlotId, f64)> {
        self.ckpt
            .priceable()
            .into_iter()
            .map(|slot| {
                let utility = self.ckpt.node_of(slot).map_or(0.0, |owner| {
                    let distance = self
                        .tree
                        .node(owner)
                        .end
                        .saturating_sub(self.tree.lower_ckpt_pos(owner));
                    ckpt_utility(distance, self.tree.geometry(), self.ckpt.hit_of(slot), now)
                });
                (slot, utility)
            })
            .collect()
    }

    /// The tokens from the root to `node`'s end, which is what the disk tier keys a path by.
    fn path_tokens(&self, node: NodeId) -> Vec<TokenId> {
        let mut out = Vec::new();
        for step in self.tree.path_to(node) {
            out.extend_from_slice(&self.tree.node(step).tokens);
        }
        out
    }
}

impl Frontier {
    /// Moves this frontier's pin to `node`.
    ///
    /// The new holder is taken before the old one is dropped -- assignment evaluates the
    /// right-hand side first -- so a frontier walking down its own path is never
    /// momentarily unpinned, which is the window an eviction scan would need to drop the
    /// span it is standing on.
    fn set_node(&mut self, tree: &SpanTree, node: NodeId) {
        self.live = tree.live_guard(node);
    }
}

impl Inner {
    /// Walk down from where the frontier stands for as long as the tree already holds
    /// this sequence's tokens, splitting the node the path leaves partway.
    ///
    /// Only ever moves down, and re-reads the tree at every step rather than trusting
    /// anything cached on the way in: an earlier commit may have split the span the
    /// frontier is standing on, and the id it held would then name the later half of a
    /// span it has already left.
    fn descend(&mut self, fr: &mut Frontier, tokens: &[TokenId], q: u32, now: f64) -> NodeId {
        let mut cur = fr.node();
        while self.tree.node(cur).end < q {
            let pos = self.tree.node(cur).end;
            let Some(at) = usize::try_from(pos).ok() else {
                break;
            };
            let Some(&first) = tokens.get(at) else { break };
            let Some(child) = self
                .tree
                .children(cur)
                .into_iter()
                .find(|child| self.tree.node(*child).tokens.first() == Some(&first))
            else {
                break;
            };
            // How far the two sequences agree inside that child. The child was found by
            // its first token, so the answer is never zero, which matters: a split at the
            // child's own start would ask the tree for an empty span.
            let span = self.tree.node(child).tokens.len();
            let mut common = 0usize;
            while common < span {
                let Some(&want) = tokens.get(at + common) else {
                    break;
                };
                if self.tree.node(child).tokens[common] != want {
                    break;
                }
                common += 1;
            }
            let next = if common < span {
                match self.tree.split_at(child, pos + to_u32(common)) {
                    Ok(head) => head,
                    Err(error) => {
                        // The frontier stops at the parent. The span above it is still
                        // reachable, so the walk simply has nowhere further to go.
                        tracing::debug!(%error, "the shared span could not be split");
                        break;
                    }
                }
            } else {
                child
            };
            cur = next;
            self.tree.touch(cur, now);
            fr.set_node(&self.tree, cur);
        }
        cur
    }

    /// Hang the span above `parent`'s end, and the pages the caller wrote for it, under
    /// `parent`.
    ///
    /// `None` when the span is lost: a page table that stops short of the frontier, or a
    /// tree that refuses the span. Both are logged and neither is an error, because the
    /// caller still holds its own pages and its next plan will read the sequence as cold
    /// -- the same outcome as a cache that never held it, at the cost of one recomputation.
    ///
    /// `entry` labels the log lines with the stored entry the span came from, when it
    /// came from one; a commit's span has no entry to name.
    fn adopt_span(
        &mut self,
        parent: NodeId,
        table: &PageTable,
        tokens: &[TokenId],
        q: u32,
        now: f64,
        entry: Option<EntryKey>,
    ) -> Option<NodeId> {
        // The span starts where `parent` ends, which is where a caller's frontier
        // stands: taking the position from the tree rather than from the caller is what
        // makes it impossible to hand the tree a span beginning somewhere its parent
        // does not end.
        let start = self.tree.node(parent).end;
        let base = usize::try_from(self.geometry.page_of(start)).unwrap_or(usize::MAX);
        let want = usize::try_from(self.geometry.pages_for(q)).unwrap_or(usize::MAX);
        let at = usize::try_from(start).unwrap_or(usize::MAX);
        let end = usize::try_from(q).unwrap_or(usize::MAX);
        let Some(span) = tokens.get(at..end) else {
            tracing::warn!(entry = ?entry, start, q, "lost span: the tokens stop short");
            return None;
        };
        let owned = table
            .as_slice()
            .get(base..want)
            .filter(|pages| pages.len() == want.saturating_sub(base));
        let Some(pages) = owned else {
            tracing::warn!(entry = ?entry, start, q, table = table.len(), "lost span: the page table stops short");
            return None;
        };
        match self.tree.add(parent, span, pages, now) {
            Ok(node) => Some(node),
            Err(error) => {
                tracing::warn!(entry = ?entry, %error, start, q, "lost span: the tree refused it");
                None
            }
        }
    }

    /// Clear a node's checkpoint pointer when its slot was just taken from it.
    ///
    /// Left pointing at the slot, the node would advertise a checkpoint that is not
    /// there, and the eviction scan would go on picking it because the drop it provokes
    /// refuses. The pointer is only cleared when it really names the slot that moved: a
    /// node handed a different slot in the meantime is telling the truth about itself.
    fn clear_displaced(&mut self, owner: Option<NodeId>, slot: SlotId) {
        if let Some(owner) = owner
            && self.tree.ckpt_at(owner) == Some(slot)
        {
            self.tree.detach_ckpt(owner);
        }
    }

    /// The admission half, with the lock held.
    fn commit(
        &mut self,
        fr: &mut Frontier,
        tokens: &[TokenId],
        q: u32,
        opts: &CommitOpts<'_>,
        now: f64,
    ) -> Result<(), KvError> {
        let q = q.min(to_u32(tokens.len()));
        // A position the frontier already stands on is still worth a call when
        // it brings a checkpoint: that is how a finished turn is checkpointed at
        // the last position its steps committed one token at a time.
        let end = self.tree.node(fr.node()).end;
        if q == 0 || q < end || (q == end && opts.state.is_none()) {
            return Ok(());
        }
        let want_blob = usize::try_from(self.geometry.ckpt_bytes).unwrap_or(usize::MAX);
        // Checked before anything is claimed, so a caller's sizing bug cannot cost the
        // cache a good checkpoint.
        if let Some(state) = opts.state
            && state.len() != want_blob
        {
            return Err(KvError::BufferLen {
                want: want_blob,
                got: state.len(),
            });
        }

        let cur = self.descend(fr, tokens, q, now);

        // Above the branch point this sequence contributes a span of its own, and the
        // pages it wrote for it.
        let start = self.tree.node(cur).end;
        if start < q
            && let Some(node) = self.adopt_span(cur, fr.table(), tokens, q, now, None)
        {
            fr.set_node(&self.tree, node);
        }
        let cur = fr.node();
        self.tree.touch(cur, now);

        let Some(state) = opts.state else {
            return Ok(());
        };
        let logits = opts.logits.filter(|_| opts.with_logits);
        // A span that already has a live checkpoint is *refreshed* in place rather than
        // given a second slot: the state at its end has moved on, and a second slot would
        // price the same positions twice.
        let refresh = self
            .tree
            .ckpt_at(cur)
            .filter(|slot| self.ckpt.is_live(*slot));
        let admission = if let Some(slot) = refresh {
            Some(Admission {
                slot,
                displaced: None,
            })
        } else {
            let prices = self.ckpt_prices(now);
            self.ckpt
                .alloc(now, !opts.chunk_only, opts.chunk_only, &prices)
        };

        let Some(Admission { slot, displaced }) = admission else {
            // A chunk boundary never evicts to get a slot, so a full store is a no-op
            // here rather than a failure: the span is committed and the next boundary
            // will come around.
            tracing::debug!(q, "no checkpoint slot for this position");
            return Ok(());
        };
        self.clear_displaced(displaced, slot);
        match self.ckpt.fill(slot, cur, q, state, logits, now) {
            Ok(_) => {
                self.tree.attach_ckpt(cur, slot);
                if let Some(reason) = opts.persist
                    && !opts.chunk_only
                {
                    match self.persist_node(cur, reason) {
                        Ok(outcome) => tracing::debug!(
                            ?outcome,
                            q,
                            ?reason,
                            "checkpoint offered to the disk tier"
                        ),
                        Err(error) => {
                            tracing::warn!(%error, q, "a checkpoint could not be offered to the disk tier");
                        }
                    }
                }
                Ok(())
            }
            Err(error) => {
                self.ckpt.evict(slot);
                Err(error)
            }
        }
    }

    /// Drop leaves until the pool holds `want` free pages.
    ///
    /// The loop insists on progress. The victim scan is deterministic, so a node the drop
    /// refuses would be picked again on every pass and this would never end. Clearing a
    /// dropped node's checkpoint pointer before the drop is what removes the usual cause;
    /// breaking on any other refusal is what stops the loop from ever spinning, and a
    /// successful drop is progress even when the free count does not move, because the
    /// only page it can fail to release is a boundary page its parent also lists.
    fn free_pages(&mut self, want: u32, now: f64) -> EvictReport {
        let mut report = EvictReport::default();
        let geometry = self.geometry;
        while self.pool.n_free() < want {
            let victim = {
                let tree = &self.tree;
                let ckpt = &self.ckpt;
                tree.pick_victim(now, &|id: NodeId, node: &crate::tree::TreeNode| {
                    // A checkpoint at the span's end is priced on the same scale as the
                    // span's pages, so page pressure cannot silently spend a valuable one.
                    let Some(slot) = node.ckpt else {
                        return 0.0;
                    };
                    if !ckpt.is_live(slot) {
                        return 0.0;
                    }
                    let distance = node.end.saturating_sub(tree.lower_ckpt_pos(id));
                    ckpt_utility(distance, &geometry, ckpt.hit_of(slot), now)
                })
            };
            let Some(victim) = victim else {
                break;
            };
            self.before_drop(victim);
            if let Some(slot) = self.tree.ckpt_at(victim) {
                self.ckpt.evict(slot);
                self.tree.detach_ckpt(victim);
                report.ckpts_dropped += 1;
            }
            match self.tree.detach(victim) {
                Ok(freed) => {
                    report.pages_freed += to_u32(freed.len());
                    report.nodes_dropped += 1;
                }
                Err(error) => {
                    tracing::debug!(node = ?victim, %error, "victim refused the drop");
                    break;
                }
            }
        }
        // Pages a queued write still pins are held by nothing else once their
        // spans are gone, and no victim scan can reach them: copy them now,
        // whatever the staging budget says, rather than fail the claim.
        if self.pool.n_free() < want {
            let Inner { disk, backing, .. } = self;
            if let Some(disk) = disk.as_mut()
                && !disk.jobs.is_empty()
                && let Err(error) = disk.pump(backing.as_ref(), u64::MAX, None, true)
            {
                tracing::warn!(%error, "queued checkpoints could not be copied to free their pages");
            }
        }
        if report.nodes_dropped > 0 {
            tracing::debug!(?report, "evicted");
        }
        report
    }
}

impl KvCache {
    /// A frontier for a sequence that has matched nothing: an empty page table pinned
    /// at the root.
    ///
    /// A cold request still needs something to hand to [`Self::commit`], and this is the
    /// only way to get a frontier without a checkpoint to resume from. The pin is not
    /// decoration: it is what makes a frontier uniform -- every one holds a node and a
    /// table, so there is no "empty frontier" state for a caller to special-case and no
    /// branch where a release gets skipped.
    ///
    /// The caller then [`Frontier::grow_to`] as many pages as its chunk needs and
    /// commits the span; the tree hangs it under the root.
    #[must_use]
    pub fn cold_frontier(&self) -> Frontier {
        let _held = InnerGuard::new();
        let inner = self.inner.lock();
        Frontier {
            live: inner.tree.live_guard(inner.tree.root()),
            table: PageTable::new(Arc::clone(&self.pool), self.geometry),
        }
    }

    /// Queues the checkpoint the frontier stands on for the disk tier.
    ///
    /// The frontier's own span has to carry a live checkpoint at its end - the
    /// one a prompt's last chunk or a finished turn admitted - and the path from
    /// the root to it is what the entry is keyed by. Nothing is written here:
    /// the pages still to copy are pinned, and [`Self::pump`] and the writer
    /// thread do the rest.
    ///
    /// # Errors
    ///
    /// [`KvError::PathGap`] when the tree's half of the path has a hole in it.
    /// Every refusal is an `Ok` outcome.
    pub fn persist(&self, fr: &Frontier, reason: Reason) -> Result<PersistOutcome, KvError> {
        let _held = InnerGuard::new();
        let mut inner = self.inner.lock();
        inner.persist_node(fr.node(), reason)
    }

    /// The live checkpoint at the end of the frontier's span, when there is one.
    #[must_use]
    pub fn frontier_ckpt(&self, fr: &Frontier) -> Option<CkptRef> {
        let _held = InnerGuard::new();
        let inner = self.inner.lock();
        let node = fr.node();
        let slot = inner.tree.ckpt_at(node)?;
        inner.ckpt.reference(slot, inner.tree.node(node).end)
    }

    /// Copies up to `max_bytes` of queued pages off the device for the writer,
    /// and hands over every entry whose pages are all copied.
    ///
    /// The host calls this between steps. It stops early wherever the staging
    /// budget is full, so one call never holds more than that budget of copies,
    /// and a cache with nothing queued answers at once.
    ///
    /// # Errors
    ///
    /// Whatever the byte mover reports for a copy; the job it was for is
    /// dropped and the rest stay queued.
    pub fn pump(&self, max_bytes: u64) -> Result<PumpReport, KvError> {
        let _held = InnerGuard::new();
        let mut inner = self.inner.lock();
        let Inner { disk, backing, .. } = &mut *inner;
        match disk.as_mut() {
            Some(disk) if !disk.jobs.is_empty() => {
                let started = Instant::now();
                let report = disk.pump(backing.as_ref(), max_bytes, None, false)?;
                if report.pages > 0 || report.entries > 0 {
                    tracing::debug!(
                        pages = report.pages,
                        entries = report.entries,
                        queued = report.queued,
                        pump_us = started.elapsed().as_micros(),
                        "copied queued checkpoint pages for the writer"
                    );
                }
                Ok(report)
            }
            _ => Ok(PumpReport::default()),
        }
    }

    /// Copies every queued page and waits up to `timeout` for the writer to land
    /// everything. Returns whether it caught up; when it did not, the writes
    /// still queued are abandoned so that dropping the cache returns promptly.
    ///
    /// For a shutdown. The copies still respect the staging budget - the writer
    /// frees it as pages land, and a short wait lets it - so draining a full
    /// queue does not hold every queued page in host memory at once.
    pub fn drain(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let _held = InnerGuard::new();
        let mut inner = self.inner.lock();
        let Inner { disk, backing, .. } = &mut *inner;
        let Some(disk) = disk.as_mut() else {
            return true;
        };
        while !disk.jobs.is_empty() && Instant::now() < deadline {
            match disk.pump(backing.as_ref(), u64::MAX, None, false) {
                Ok(report) if report.pages == 0 && report.entries == 0 => {
                    std::thread::sleep(DRAIN_WAIT);
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(%error, "a queued checkpoint could not be copied on the way down");
                }
            }
        }
        let dropped = disk.jobs.len();
        while !disk.jobs.is_empty() {
            disk.drop_job(0);
        }
        let caught_up =
            dropped == 0 && disk.flush(deadline.saturating_duration_since(Instant::now()));
        if !caught_up {
            tracing::warn!(
                timeout_s = timeout.as_secs_f64(),
                dropped,
                "the disk writer did not finish in time; abandoning the writes still queued"
            );
            disk.abandon_queued();
        }
        caught_up
    }

    /// Records a measured prefill: `tokens` computed in `seconds`. The resume
    /// cost model prices a disk read against this rate.
    pub fn observe_prefill(&self, tokens: u32, seconds: f64) {
        if tokens == 0 || seconds.is_nan() || seconds <= 0.0 {
            return;
        }
        let sample = seconds / f64::from(tokens);
        let _held = InnerGuard::new();
        let mut inner = self.inner.lock();
        inner.prefill_s_per_token =
            inner.prefill_s_per_token * (1.0 - PREFILL_EWMA_WEIGHT) + sample * PREFILL_EWMA_WEIGHT;
    }

    /// Stands a fresh sequence on the deepest checkpoint its prompt can use,
    /// from the tree or from the disk tier.
    ///
    /// The tree is asked first and is free: its pages are resident. The disk is
    /// asked for a checkpoint *deeper* than the tree's, and read only when the
    /// cost model says reading it beats prefilling the difference: bytes at the
    /// measured read rate against tokens at the measured prefill rate. A disk
    /// restore that fails partway - a missing or corrupt file - discards the
    /// entry and falls back to the tree's answer.
    ///
    /// A checkpoint at the very end of `tokens` is only used when it carries its
    /// logits row, because resuming there means answering the next token from
    /// that row. `Ok(None)` means nothing usable was found: the caller prefills
    /// from zero, exactly as with an empty cache.
    ///
    /// # Errors
    ///
    /// [`KvError::PoolEmpty`] when the pages a resume needs cannot be claimed
    /// even after eviction, [`KvError::PathGap`] for a hole in the tree's half of
    /// a path, and whatever the byte mover reports.
    pub fn resume(&self, tokens: &[TokenId], now: f64) -> Result<Option<Resume>, KvError> {
        let _held = InnerGuard::new();
        let mut inner = self.inner.lock();
        inner.resume(tokens, now)
    }

    /// Where `tokens` stops matching the tree: the branch point a prompt leaves
    /// the paths the cache already holds at.
    ///
    /// A checkpoint there is what a third prompt sharing the same opening
    /// resumes from, which is why a host stops a chunk at it. Walked from the
    /// root, so it is right whatever the caller's own frontier holds.
    #[must_use]
    pub fn branch_point(&self, tokens: &[TokenId]) -> u32 {
        let _held = InnerGuard::new();
        let inner = self.inner.lock();
        inner.tree.walk_down(tokens).matched
    }
}

impl Inner {
    /// The persist half, with the lock held.
    fn persist_node(&mut self, node: NodeId, reason: Reason) -> Result<PersistOutcome, KvError> {
        let refused = |why| Ok(PersistOutcome::Refused(why));
        if self.disk.is_none() {
            return refused(PersistRefusal::NotConfigured);
        }
        if node == self.tree.root() || self.tree.node(node).dropped {
            return refused(PersistRefusal::NodeGone);
        }
        let Some(slot) = self
            .tree
            .ckpt_at(node)
            .filter(|slot| self.ckpt.is_live(*slot))
        else {
            return refused(PersistRefusal::NoCheckpoint);
        };
        let pos = self.tree.node(node).end;
        let min = self.disk.as_ref().map_or(0, DiskStore::min_tokens);
        if pos < min {
            return refused(PersistRefusal::BelowMinTokens { tokens: pos, min });
        }
        let Some(state) = self.ckpt.state_arc(slot) else {
            return refused(PersistRefusal::NoCheckpoint);
        };
        let tokens = self.path_tokens(node);
        let page_ids = self.tree.page_array_from_path(node, pos)?;
        let spec = PersistSpec {
            tokens: &tokens,
            page_ids: &page_ids,
            state,
            logits: self.ckpt.logits_arc(slot),
            reason,
        };
        let pool = Arc::clone(&self.pool);
        let Some(disk) = self.disk.as_mut() else {
            return refused(PersistRefusal::NotConfigured);
        };
        Ok(disk.persist(&spec, &pool))
    }

    /// Makes sure nothing is lost when `victim`'s pages leave the pool.
    ///
    /// An earned checkpoint at the victim's end that is not on disk yet is
    /// queued first - spill on evict, the safety net under the writes a turn
    /// already queued - and every queued job pinning one of the victim's pages
    /// is copied now, whatever the staging budget says, because a pin left in
    /// place would keep the page the eviction is trying to free.
    fn before_drop(&mut self, victim: NodeId) {
        let Some(disk) = self.disk.as_ref() else {
            return;
        };
        if let Some(slot) = self.tree.ckpt_at(victim)
            && self.ckpt.is_live(slot)
            && !self.ckpt.is_speculative(slot)
            && self.tree.node(victim).end >= disk.min_tokens()
        {
            let tokens = self.path_tokens(victim);
            let stored = disk
                .entry_key_of(&tokens)
                .is_none_or(|key| disk.knows(&key));
            if !stored {
                match self.persist_node(victim, Reason::Evict) {
                    Ok(outcome) => {
                        tracing::debug!(?outcome, "an evicted checkpoint was queued for disk");
                    }
                    Err(error) => {
                        tracing::warn!(%error, "an evicted checkpoint could not be queued");
                    }
                }
            }
        }
        let pages = self.tree.node(victim).pages.clone();
        let Inner { disk, backing, .. } = self;
        let Some(disk) = disk.as_mut() else {
            return;
        };
        while let Some(at) = disk
            .jobs
            .iter()
            .position(|job| pages.iter().any(|page| job.pins(*page)))
        {
            if let Err(error) = disk.pump(backing.as_ref(), u64::MAX, Some(at), true) {
                tracing::warn!(%error, "a queued checkpoint could not be copied before its pages went");
            }
        }
    }

    /// The resume half, with the lock held. `Ok(None)` is the public method's:
    /// nothing usable in the tree or on disk, a cold start.
    fn resume(&mut self, tokens: &[TokenId], now: f64) -> Result<Option<Resume>, KvError> {
        let plan = self.plan_with(None, tokens, None, now);
        let tree_pos = if plan.ckpt.is_some() {
            plan.reusable()
        } else {
            0
        };
        let hit = self
            .disk
            .as_ref()
            .and_then(|disk| disk.lookup(tokens, tree_pos.saturating_add(1)));
        if let Some(hit) = hit {
            let prefill_s = f64::from(hit.pos - tree_pos) * self.prefill_s_per_token;
            let restore_s = self.restore_estimate(&hit, plan.matched);
            if restore_s < prefill_s {
                match self.restore_disk(tokens, &hit, prefill_s, now) {
                    Ok(Some(resume)) => {
                        self.resumes_disk += 1;
                        return Ok(Some(resume));
                    }
                    Ok(None) => {}
                    Err(error) => {
                        tracing::warn!(%error, entry = %hit.key, "a disk resume failed; trying the tree");
                        if let Some(disk) = self.disk.as_ref() {
                            disk.restore_failed();
                        }
                    }
                }
            } else {
                tracing::debug!(
                    entry = %hit.key,
                    tokens = hit.pos,
                    restore_s,
                    prefill_s,
                    "a stored checkpoint costs more to read than to prefill"
                );
                if let Some(disk) = self.disk.as_ref() {
                    disk.restore_declined();
                }
            }
        }
        let resumed = self.lookup_with(None, tokens, None, now)?;
        if resumed.is_some() {
            self.resumes_tree += 1;
        }
        Ok(resumed)
    }

    /// Seconds a disk restore of `hit` is expected to take: the bytes not
    /// already resident, at the measured read rate.
    fn restore_estimate(&self, hit: &Hit, matched: u32) -> f64 {
        let Some(disk) = self.disk.as_ref() else {
            return f64::INFINITY;
        };
        let pages = hit.pages.len().max(1);
        let resident = usize::try_from(self.geometry.page_of(matched.min(hit.pos)))
            .unwrap_or(0)
            .min(pages);
        let fraction = f64::from(to_u32(pages - resident)) / f64::from(to_u32(pages));
        let bytes = crate::ids::to_u32_wide(hit.bytes);
        f64::from(bytes) * fraction / disk.read_bps().max(1.0)
    }

    /// Reads a stored path back and stands a frontier on it: the tree's own
    /// pages below the position the tree still agrees with, pages read off disk
    /// above it, and the stored state in a fresh checkpoint slot.
    ///
    /// `Ok(None)` when the entry turned out unusable partway - it is discarded -
    /// or no slot could hold its state; the caller falls back to the tree.
    fn restore_disk(
        &mut self,
        tokens: &[TokenId],
        hit: &Hit,
        prefill_s: f64,
        now: f64,
    ) -> Result<Option<Resume>, KvError> {
        let started = Instant::now();
        let geometry = self.geometry;
        let want_state = usize::try_from(geometry.ckpt_bytes).unwrap_or(usize::MAX);
        let Some(disk) = self.disk.as_ref() else {
            return Ok(None);
        };
        let pos = hit.pos;
        let path = &tokens[..usize::try_from(pos)
            .unwrap_or(tokens.len())
            .min(tokens.len())];
        let walk = self.tree.walk_down(path);
        let matched = walk.matched.min(pos);
        let own_from = usize::try_from(geometry.page_of(matched)).unwrap_or(usize::MAX);
        let Some((stream, body)) = disk.start_restore(hit, path, own_from) else {
            return Ok(None);
        };
        let mut target = self.tree.frontier_at(&walk)?;
        // Pinned before anything is claimed: the claims below can evict, and an
        // unpinned target is a leaf the victim scan would drop from under the
        // restore about to stand on it.
        let mut live = self.tree.live_guard(target);
        let table = self.restore_table(target, own_from, pos, now)?;
        let Some(disk) = self.disk.as_ref() else {
            return Ok(None);
        };
        let read = match upload_pages(
            &stream,
            self.backing.as_mut(),
            &table,
            own_from,
            hit.pages.len(),
        )? {
            Ok(read) => read,
            Err(failure) => {
                tracing::warn!(entry = %hit.key, page = failure.index, why = ?failure.why, "a stored page is unusable");
                disk.forget(hit.key, failure.why);
                disk.restore_failed();
                return Ok(None);
            }
        };
        drop(stream);
        disk.count_read(read);
        let pages_done = started.elapsed();
        let Some(entry) = disk.finish_body(hit, body, want_state) else {
            return Ok(None);
        };

        // The span above the match joins the tree, so the next request walking
        // this prefix matches it instead of reading the files again.
        if matched < pos
            && let Some(node) = self.adopt_span(target, &table, path, pos, now, Some(hit.key))
        {
            target = node;
            live = self.tree.live_guard(target);
        }
        self.tree.touch(target, now);
        let Some(ckpt) = self.checkpoint_restored(
            hit.key,
            target,
            pos,
            &entry.state,
            entry.logits.as_deref(),
            now,
        ) else {
            return Ok(None);
        };
        let took = started.elapsed();
        if let Some(disk) = self.disk.as_mut() {
            disk.restored(hit.key, hit.bytes, took, prefill_s);
        }
        tracing::info!(
            entry = %hit.key,
            tokens = pos,
            tree_tokens = matched,
            read_bytes = read,
            restore_ms = took.as_millis(),
            pages_ms = pages_done.as_millis(),
            prefill_estimate_s = prefill_s,
            "resumed a checkpoint from disk"
        );
        Ok(Some(Resume {
            frontier: Frontier { live, table },
            ckpt: Some(ckpt),
            ckpt_pos: pos,
            matched: pos,
            source: ReuseSource::Disk,
        }))
    }

    /// The page table a disk restore fills: the tree's own pages below page
    /// `own_from`, and freshly claimed ones up to `pos` for the reads to land in.
    ///
    /// # Errors
    ///
    /// [`KvError::PathGap`] for a hole in the tree's half, and
    /// [`KvError::PoolEmpty`] when the pages cannot be claimed even after an
    /// eviction pass.
    fn restore_table(
        &mut self,
        target: NodeId,
        own_from: usize,
        pos: u32,
        now: f64,
    ) -> Result<PageTable, KvError> {
        let geometry = self.geometry;
        let prefix = self
            .tree
            .page_array_from_path(target, geometry.first_position(to_u32(own_from)))?;
        let mut table = PageTable::adopt(Arc::clone(&self.pool), geometry, prefix);
        if !table.grow_to(pos)? {
            self.free_pages(geometry.pages_for(pos), now);
            if !table.grow_to(pos)? {
                return Err(KvError::PoolEmpty);
            }
        }
        Ok(table)
    }

    /// Gives restored state a checkpoint on the node the path ends at.
    ///
    /// That is what makes a second restore of the same prefix unnecessary and
    /// what lets the eviction scan price the prefix against the rest of the
    /// cache. A slot that cannot be claimed, or a blob whose length disagrees
    /// with this cache, costs the restore its checkpoint - and a checkpoint is
    /// the whole of a hybrid resume, so the caller falls back.
    fn checkpoint_restored(
        &mut self,
        key: EntryKey,
        target: NodeId,
        pos: u32,
        state: &[u8],
        logits: Option<&[u8]>,
        now: f64,
    ) -> Option<CkptRef> {
        if self.geometry.ckpt_bytes > 0 && state.is_empty() {
            return None;
        }
        let prices = self.ckpt_prices(now);
        let Admission { slot, displaced } = self.ckpt.alloc(now, true, false, &prices)?;
        self.clear_displaced(displaced, slot);
        match self.ckpt.fill(slot, target, pos, state, logits, now) {
            Ok(reference) => {
                self.tree.attach_ckpt(target, slot);
                Some(reference)
            }
            Err(error) => {
                tracing::warn!(entry = %key, %error, "restored state is not the size this cache uses");
                self.ckpt.evict(slot);
                None
            }
        }
    }
}

/// Uploads pages `from..to` of a restored path as the reader delivers them.
///
/// The outer `Result` is the byte mover's; the inner one is the entry's: a
/// page that did not arrive, arrived out of order, or failed its checks makes
/// the whole entry unusable, and the caller discards it.
fn upload_pages(
    stream: &crate::store::PageStream,
    backing: &mut dyn Backing,
    table: &PageTable,
    from: usize,
    to: usize,
) -> Result<Result<u64, crate::store::PageFailure>, KvError> {
    let mut read = 0u64;
    for expected in from..to {
        match stream.next() {
            Some(Ok(page)) if page.index == expected => {
                backing.page_from_host(table.as_slice()[expected], page.payload())?;
                read += u64::try_from(page.buf.len()).unwrap_or(u64::MAX);
                stream.give_back(page.buf);
            }
            Some(Err(failure)) => return Ok(Err(failure)),
            Some(Ok(_)) | None => {
                return Ok(Err(crate::store::PageFailure {
                    index: expected,
                    why: crate::store::Unusable::Missing,
                }));
            }
        }
    }
    Ok(Ok(read))
}
