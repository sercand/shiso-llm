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
//! | [`KvCache::spill`] / [`KvCache::restore`] | put a prefix on disk, take it back | yes, on restore |
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

use shiso_core::TokenId;

use crate::backing::Backing;
use crate::ckpt::{Admission, CkptRef, CkptStore};
use crate::error::KvError;
use crate::evict::ckpt_utility;
use crate::geom::CacheGeometry;
use crate::ids::{NodeId, SlotId, SpillId, to_u32};
use crate::page_table::PageTable;
use crate::path::{DecodedPath, PathHeader, PathMismatch, PathSections, decode_path, encode_path};
use crate::pool::{InnerGuard, PagePool};
use crate::spill::{Incoming, Reason, SpillConfig, SpillStore, SpillWrite};
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
    /// The node that checkpoint belongs to, which is what [`KvCache::spill`] takes.
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
            ReuseSource::Checkpoint | ReuseSource::Tree => self.ckpt_pos.min(self.matched),
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
    /// The disk tier's own numbers, when a spill directory was opened.
    pub spill: Option<crate::spill::SpillStats>,
}

/// Why a spill did not happen, or what it produced.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SpillOutcome {
    /// The prefix was written; the id is the sha1 of the bytes it caches.
    Written {
        /// The key the payload is stored under.
        id: SpillId,
        /// Bytes written, header included.
        bytes: u64,
    },
    /// The same prefix was already on disk, so nothing was written.
    AlreadyStored {
        /// The existing entry.
        id: SpillId,
    },
    /// Nothing was written, and why.
    Refused {
        /// The reason, in the engine's words.
        reason: SpillRefusal,
    },
}

/// Why `spill` declined. Split out because the engine's answer differs by case: a
/// cold prefix is a decision to revisit later, while a missing logits row is a caller
/// bug it should not retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpillRefusal {
    /// The node has no checkpoint, so a resume from it would have to prefill the whole
    /// prefix anyway.
    NoCheckpoint,
    /// The checkpoint's logits row is missing, so a resume at the end of the prefix
    /// would not be sample-identical.
    StaleLogits,
    /// The node's span is not on the tree any more.
    NodeGone,
    /// The cache was opened without a spill directory, or without the byte
    /// description a payload needs.
    NotConfigured,
    /// The payload would not fit the directory budget even after evicting.
    OverBudget {
        /// Bytes the write needs.
        required: u64,
        /// Bytes the directory may hold.
        budget: u64,
    },
    /// The checkpoint blob and the byte description disagree about how large a state
    /// section is, so the payload would not read back. A configuration error, and the
    /// one refusal the engine cannot fix at runtime.
    LayoutMismatch {
        /// Bytes the description says the state section is.
        want: u64,
        /// Bytes the checkpoint blob is.
        got: u64,
    },
}

/// A prefix read back from the disk tier, ready to generate from.
///
/// The state and the page bytes are already where a frontier keeps them -- in a
/// checkpoint slot and in pool pages -- so the caller generates from
/// [`Restored::frontier`] and never handles the bytes. That is the whole point of
/// returning a frontier rather than a bag of buffers: a caller that had to adopt the
/// pages itself could forget to, and the payload's pages would never come back.
#[derive(Debug)]
#[non_exhaustive]
pub struct Restored {
    /// The prefix's pages and the span it ends at.
    pub frontier: Frontier,
    /// The token prefix the payload covers.
    pub tokens: Vec<TokenId>,
    /// The logits row that came with the state, so a re-send of the same prompt can
    /// answer without recomputing one.
    pub logits: Option<Vec<u8>>,
    /// The checkpoint the state was installed in, so the caller can read the blob back
    /// with [`KvCache::ckpt_state_into`] if it wants it in its own buffer.
    pub ckpt: Option<CkptRef>,
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
    /// Crate-visible so the modules that are the cache's other halves -- the disk
    /// index in [`crate::entries`] -- can take the same lock in the same order
    /// rather than reaching the tree through a second one.
    pub(crate) inner: parking_lot::Mutex<Inner>,
}

/// Everything the cache lock covers.
#[derive(Debug)]
pub(crate) struct Inner {
    geometry: CacheGeometry,
    /// The same `Arc` the tree and every page table handed out holds.
    pool: Arc<PagePool>,
    tree: SpanTree,
    ckpt: CkptStore,
    /// Held exclusively, and behind the same lock as the tree, because a
    /// copy-on-write must not interleave with the span changes that decided which page
    /// to copy.
    backing: Box<dyn Backing>,
    pub(crate) spill: Option<SpillStore>,
}

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
    /// `spill` opens the disk tier. Without it the cache is purely in-memory and
    /// [`Self::spill`] answers [`SpillRefusal::NotConfigured`].
    ///
    /// # Errors
    ///
    /// [`KvError::BadGeometry`] if the geometry cannot be built from, and
    /// [`KvError::Io`] if the spill directory cannot be created. A directory full of
    /// files this build cannot read is not an error: those are simply not entries.
    pub fn open(
        geometry: &CacheGeometry,
        backing: Box<dyn Backing>,
        pool_pages: u32,
        ckpt_slots: u32,
        spill: Option<SpillConfig>,
    ) -> Result<Self, KvError> {
        geometry.validate()?;
        let pool = Arc::new(PagePool::new(pool_pages));
        let tree = SpanTree::new(*geometry, Arc::clone(&pool));
        let ckpt = CkptStore::new(*geometry, ckpt_slots);
        let spill = match spill {
            Some(config) => Some(SpillStore::open(&config)?),
            None => None,
        };
        Ok(Self {
            inner: parking_lot::Mutex::new(Inner {
                geometry: *geometry,
                pool: Arc::clone(&pool),
                tree,
                ckpt,
                backing,
                spill,
            }),
            geometry: *geometry,
            pool,
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
        let plan = inner.plan_with(standing, tokens, clamp, now);
        let (Some(ckpt), Some(target)) = (plan.ckpt, plan.ckpt_node) else {
            return Ok(None);
        };
        let cpos = ckpt.pos;

        // Pin the path being moved onto *before* anything that can evict. The copy
        // below may need a page, and a page shortage runs the victim scan; if the
        // target were still an unpinned childless leaf at that moment, the scan could
        // drop the very node and checkpoint being restored. Pinning afterwards is the
        // bug, not the ordering.
        let live = inner.tree.live_guard(target);
        let pages = match inner.tree.page_array_from_path(target, cpos) {
            Ok(pages) => pages,
            Err(error) => {
                drop(live);
                return Err(error);
            }
        };
        let mut table = PageTable::adopt(Arc::clone(&self.pool), self.geometry, pages);

        // The page holding `cpos - 1` is shared and immutable, and the next prefill is
        // about to rewrite its tail, so this sequence takes a private copy of that one
        // page. A checkpoint on a page boundary -- every chunk boundary is one -- needs
        // no copy at all, which is why resuming at a chunk boundary is free.
        if !self.geometry.is_page_start(cpos) && !table.is_empty() {
            let last = table.len() - 1;
            let src = table.as_slice()[last];
            if self.pool.n_free() == 0 {
                inner.free_pages(1, now);
            }
            let Some(dst) = self.pool.take() else {
                drop(live);
                return Err(KvError::PoolEmpty);
            };
            if let Err(error) = inner.backing.copy_page(dst, src) {
                self.pool.unref(dst);
                drop(live);
                return Err(error);
            }
            table.replace(last, dst);
        }

        inner.tree.touch(target, now);
        inner.ckpt.touch(ckpt.slot, now);
        Ok(Some(Resume {
            frontier: Frontier { live, table },
            ckpt: Some(ckpt),
            ckpt_pos: cpos,
            matched: plan.matched,
            source: plan.source,
        }))
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
            spill: inner.spill.as_ref().map(SpillStore::stats),
        }
    }

    /// Every entry the disk tier holds, newest first by the store's own order.
    #[must_use]
    pub fn spill_stats(&self) -> Option<crate::spill::SpillStats> {
        let _held = InnerGuard::new();
        let inner = self.inner.lock();
        inner.spill.as_ref().map(SpillStore::stats)
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

    /// The tokens from the root to `node`'s end, which is what a spill keys a path by.
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
    /// `entry` labels the log lines with the spilled entry the span came from, when it
    /// came from one; a commit's span has no entry to name.
    fn adopt_span(
        &mut self,
        parent: NodeId,
        table: &PageTable,
        tokens: &[TokenId],
        q: u32,
        now: f64,
        entry: Option<SpillId>,
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
        if q == 0 || q <= self.tree.node(fr.node()).end {
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

    /// Write the path below `node` to the disk tier, keyed by `text`.
    ///
    /// `text` is the rendered prefix the entry is *named* by -- the bytes a later
    /// request will have in hand when it looks for this one -- and it is not the
    /// token list, because the token run a media span produces is the same id repeated
    /// and cannot distinguish two different pictures. The caller is the only party that
    /// knows what a request rendered to, so the key is passed in rather than derived.
    ///
    /// A node without a checkpoint is refused rather than written: a prefix whose
    /// recurrent state is missing has to be re-prefilled anyway, so its pages are worth
    /// less than the disk space they would take. A checkpoint whose logits row is gone is
    /// refused for a different reason -- resuming at the very end of a prefix needs that
    /// row to sample identically -- and the engine treats that one as a bug in itself
    /// rather than as a cold prefix.
    ///
    /// `reason` is what the budget pass prices the entry at: an entry written because the
    /// process is shutting down, or because the engine expects to come back to it,
    /// outlives one written because a leaf was merely cold.
    ///
    /// # Errors
    ///
    /// [`KvError::Io`] if the directory rejects the write, and whatever the byte mover
    /// reports while reading the path's pages. A refusal is `Ok`, not an error.
    pub fn spill(
        &self,
        node: NodeId,
        text: &[u8],
        reason: Reason,
    ) -> Result<SpillOutcome, KvError> {
        let _held = InnerGuard::new();
        let mut inner = self.inner.lock();
        inner.spill(node, text, reason)
    }

    /// Read a spilled path back and hand over a frontier that stands at its end.
    ///
    /// `Ok(None)` means the entry is gone, was never there, or describes a build this
    /// one can never be. A file that this build can never read is deleted on the way
    /// past -- it is worthless wherever it sits -- while a file that is only wrong
    /// *as configured* is kept, because the same binary started with the other toggle
    /// value can read it and the key does not mention the toggles.
    ///
    /// # Errors
    ///
    /// [`KvError::ShaMismatch`] if the file's contents do not hash to its name,
    /// [`KvError::PoolEmpty`] if the pages the payload needs cannot be claimed even after
    /// eviction, [`KvError::PathGap`] if the tree's half of the path has a hole in it, and
    /// whatever the byte mover reports.
    pub fn restore(&self, id: SpillId, now: f64) -> Result<Option<Restored>, KvError> {
        let _held = InnerGuard::new();
        let mut inner = self.inner.lock();
        inner.restore(id, now)
    }
}

impl Inner {
    /// The spill half, with the lock held.
    fn spill(
        &mut self,
        node: NodeId,
        text: &[u8],
        reason: Reason,
    ) -> Result<SpillOutcome, KvError> {
        let refused = |reason: SpillRefusal| Ok(SpillOutcome::Refused { reason });
        if self.tree.node(node).dropped || node == self.tree.root() {
            return refused(SpillRefusal::NodeGone);
        }
        let geometry = self.geometry;
        let Some(layout) = geometry.layout else {
            return refused(SpillRefusal::NotConfigured);
        };
        if self.spill.is_none() {
            return refused(SpillRefusal::NotConfigured);
        }
        let Some(slot) = self.tree.ckpt_at(node) else {
            return refused(SpillRefusal::NoCheckpoint);
        };
        if !self.ckpt.is_live(slot) {
            return refused(SpillRefusal::NoCheckpoint);
        }
        if !self.ckpt.has_logits(slot) {
            return refused(SpillRefusal::StaleLogits);
        }

        let pos = self.tree.node(node).end;
        let tokens = self.path_tokens(node);
        let pages = self.tree.page_array_from_path(node, pos)?;
        let header = PathHeader::from_layout(
            &layout,
            &geometry,
            to_u32(tokens.len()),
            to_u32(pages.len()),
        );
        let reference = self
            .ckpt
            .reference(slot, pos)
            .ok_or(KvError::StaleCheckpoint { slot })?;

        let mut state = vec![0u8; usize::try_from(geometry.ckpt_bytes).unwrap_or(usize::MAX)];
        self.ckpt.state_into(reference, &mut state)?;
        let described = header.state_bytes();
        let carried = u64::try_from(state.len()).unwrap_or(u64::MAX);
        if described != carried {
            // The payload's sections are sized from the layout's element counts and the
            // blob is sized from the geometry. If those disagree the file would read back
            // with its sections shifted, which is worse than not writing it.
            return refused(SpillRefusal::LayoutMismatch {
                want: described,
                got: carried,
            });
        }
        let logits_len = self.ckpt.logits_len(slot).unwrap_or(0);
        let mut logits = vec![0u8; logits_len];
        self.ckpt.logits_into(reference, &mut logits)?;

        let page_bytes = usize::try_from(geometry.page_bytes).unwrap_or(usize::MAX);
        let mut page_section = Vec::with_capacity(pages.len().saturating_mul(page_bytes));
        let mut buffer = vec![0u8; page_bytes];
        for page in &pages {
            self.backing.page_to_host(*page, &mut buffer)?;
            page_section.extend_from_slice(&buffer);
        }

        let payload = encode_path(
            &header,
            &PathSections {
                tokens: &tokens,
                logits: &logits,
                state: &state,
                pages: &page_section,
            },
        );
        let incoming = Incoming {
            text,
            tokens: to_u32(tokens.len()),
            quant_bits: layout.quant_bits,
            model_id: layout.model_id,
            ctx_size: layout.ctx_size,
            reason,
            ext_flags: 0,
            payload: &payload,
        };
        let store = self.spill.as_mut().ok_or(KvError::BadHeader {
            reason: "the spill directory vanished mid-write",
        })?;
        match store.write(&incoming) {
            Ok(SpillWrite::Written { id, bytes }) => Ok(SpillOutcome::Written { id, bytes }),
            Ok(SpillWrite::AlreadyStored { id }) => Ok(SpillOutcome::AlreadyStored { id }),
            Err(KvError::OverBudget { required, budget }) => {
                refused(SpillRefusal::OverBudget { required, budget })
            }
            Err(error) => Err(error),
        }
    }

    /// The restore half, with the lock held.
    ///
    /// A path that cannot be installed still costs nothing but the recomputation: the
    /// caller gets `None` and prefills, exactly as it would have with an empty cache.
    fn restore(&mut self, id: SpillId, now: f64) -> Result<Option<Restored>, KvError> {
        let geometry = self.geometry;
        let Some(layout) = geometry.layout else {
            return Ok(None);
        };
        let Some(store) = self.spill.as_mut() else {
            return Ok(None);
        };
        let Some(entry) = store.read(id)? else {
            return Ok(None);
        };
        let expect = PathHeader::from_layout(&layout, &geometry, 0, 0);
        let Some(decoded) = Self::read_spilled(store, id, &entry.payload, &expect) else {
            return Ok(None);
        };
        let _ = store.touch(id);

        let logits = (!decoded.logits.is_empty()).then_some(decoded.logits);
        let (frontier, target) = self.install_restored(id, &decoded.tokens, &decoded.pages, now)?;
        let ckpt = self.checkpoint_restored(
            id,
            target,
            to_u32(decoded.tokens.len()),
            &decoded.state,
            logits.as_deref(),
            now,
        );
        Ok(Some(Restored {
            frontier,
            tokens: decoded.tokens,
            logits,
            ckpt,
        }))
    }

    /// Decode a payload and decide what happens to the file it came from.
    ///
    /// `None` means the caller gets a cold answer and nothing is restored. Which of those
    /// refusals also *deletes* is decided by the mismatch kind rather than by this
    /// function's convenience -- see [`PathMismatch`].
    fn read_spilled(
        store: &SpillStore,
        id: SpillId,
        payload: &[u8],
        expect: &PathHeader,
    ) -> Option<DecodedPath> {
        match decode_path(payload, expect) {
            Ok(decoded) => Some(decoded),
            Err(PathMismatch::Foreign { field, found, want }) => {
                tracing::warn!(
                    entry = %id, field, found, want,
                    "spilled path describes a build this one can never be; discarding",
                );
                discard(store, id);
                None
            }
            Err(PathMismatch::Incompatible { field, found, want }) => {
                // Kept: the same build started with the other toggle value can read this,
                // and the key is the rendered text rather than the toggles, so deleting it
                // would wipe a shared directory every time an operator flips a flag.
                tracing::info!(
                    entry = %id, field, found, want,
                    "spilled path is not usable as configured; keeping it",
                );
                None
            }
            Err(PathMismatch::Truncated { needed, have }) => {
                tracing::warn!(entry = %id, needed, have, "spilled path is short; discarding");
                discard(store, id);
                None
            }
        }
    }

    /// Rebuild a frontier for a spilled path: the tree's own pages below the position the
    /// tree still agrees with, and fresh pages read from the payload above it.
    ///
    /// The node comes back too, because that is where the restored state is a checkpoint
    /// *at*, and rediscovering it would mean walking the tree a second time.
    ///
    /// # Errors
    ///
    /// [`KvError::PathGap`] if the matched path has a page nobody owns, [`KvError::PoolEmpty`]
    /// if the pages above the match cannot be claimed even after an eviction pass, and
    /// whatever the byte mover reports while reading the payload in.
    fn install_restored(
        &mut self,
        id: SpillId,
        tokens: &[TokenId],
        pages: &[Vec<u8>],
        now: f64,
    ) -> Result<(Frontier, NodeId), KvError> {
        let geometry = self.geometry;
        let pos = to_u32(tokens.len());
        let walk = self.tree.walk_down(tokens);
        let matched = walk.matched.min(pos);
        let mut target = self.tree.frontier_at(&walk)?;
        // Pin before claiming anything: the page claims below can evict, and an
        // unpinned target is a childless leaf the victim scan would happily drop out from
        // under the restore that is about to stand on it.
        let mut live = self.tree.live_guard(target);

        // Everything below the page the match lands inside is already owned by the tree
        // and adopted rather than re-read. That one page is the exception: the tree's copy
        // is only valid up to the match, so the payload's copy -- which has the whole
        // page -- is what gets used, and it is read in rather than copied from a page the
        // tree is still holding.
        let own_from = usize::try_from(geometry.page_of(matched)).unwrap_or(usize::MAX);
        let prefix_len = geometry.first_position(to_u32(own_from));
        let prefix = self.tree.page_array_from_path(target, prefix_len)?;
        let mut table = PageTable::adopt(Arc::clone(&self.pool), geometry, prefix);
        if !table.grow_to(pos)? {
            self.free_pages(geometry.pages_for(pos), now);
            if !table.grow_to(pos)? {
                return Err(KvError::PoolEmpty);
            }
        }
        let fresh = table.as_slice()[own_from..].to_vec();
        for (page, bytes) in fresh.iter().zip(pages.iter().skip(own_from)) {
            self.backing.page_from_host(*page, bytes)?;
        }

        // The span above the match joins the tree, so the next request that walks this
        // prefix matches it instead of re-reading the file.
        if matched < pos
            && let Some(node) = self.adopt_span(target, &table, tokens, pos, now, Some(id))
        {
            target = node;
            live = self.tree.live_guard(target);
        }
        self.tree.touch(target, now);
        Ok((Frontier { live, table }, target))
    }

    /// Give the restored state a checkpoint on the node the path ends at.
    ///
    /// That is what makes a second restore of the same prefix unnecessary and what lets
    /// the eviction scan price the prefix against the rest of the cache. A slot that
    /// cannot be claimed, or a state blob whose length disagrees with this cache, costs
    /// the restored path its checkpoint and nothing else: the caller still gets the pages
    /// and the frontier, and can decode from position zero.
    fn checkpoint_restored(
        &mut self,
        id: SpillId,
        target: NodeId,
        pos: u32,
        state: &[u8],
        logits: Option<&[u8]>,
        now: f64,
    ) -> Option<CkptRef> {
        // An empty blob is a missing one only where a blob was expected. A cache whose
        // geometry says the model has no recurrent state is carrying a checkpoint that
        // is the frontier logits row alone, and dropping it would cost the restored
        // path the prefill it was written to save.
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
                tracing::warn!(entry = %id, %error, "restored state is not the size this cache uses");
                self.ckpt.evict(slot);
                None
            }
        }
    }
}

/// Delete an entry that this build can never read. that this build can never read.
///
/// Failure is ignored on purpose: the entry will be refused again on the next read, and
/// a directory the process cannot write to is not a reason to fail the request that
/// merely asked what was in it.
fn discard(store: &SpillStore, id: SpillId) {
    let path = store.path_of(id);
    if let Err(error) = std::fs::remove_file(&path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(path = %path.display(), %error, "could not discard an unreadable path");
    }
}
