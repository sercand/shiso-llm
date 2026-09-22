//! The span tree: which token prefixes already have KV, and who owns the pages.
//!
//! # Why a tree
//!
//! Sequences share prefixes — a system prompt, a prior turn, a re-sent history —
//! and once the frontier has moved past a page nobody is writing it any more. That
//! is what makes two sequences safe to index the same page without coordinating.
//! Nodes are token spans, children branch by first token, and a node owns the pool
//! pages covering `[start, end)`.
//!
//! # Splits
//!
//! A match usually ends in the middle of a node's span, and the boundary between a
//! shared prefix and a diverging continuation rarely lands on a page edge. Splitting
//! there gives the earlier half to a new node and leaves the original holding the
//! later half, its children and its checkpoint, because all of those hang off the
//! *end* of the span. The page the split lands inside is held by both halves: each
//! owns different rows of it, and neither can release it while the other reads its
//! own rows.
//!
//! # Indices, not pointers
//!
//! Nodes live in an arena and link by index. The topology is a hand-linked tree —
//! parent, first child, next sibling — and every pointer operation has an index
//! twin, but nothing can dangle and the whole tree sits behind one mutex.
//!
//! # Live counts
//!
//! A node's live count keeps its pages out of eviction while a sequence is writing
//! them. The counts sit behind their own mutex rather than in the arena because a
//! pin is released by `Drop`, and drops happen inside the cache's own critical
//! sections. Counting in the arena would make that drop retake a lock the same
//! thread holds. Same rule as the pool: the cache lock may be held while taking
//! this one, never the reverse.

use std::sync::Arc;

use parking_lot::Mutex;
use shiso_core::TokenId;

use crate::error::KvError;
use crate::geom::CacheGeometry;
use crate::ids::{NodeId, PageId, SlotId, to_u32};
use crate::pool::PagePool;

/// One span of tokens, the pages holding their rows, and what hangs off it.
#[derive(Debug)]
pub(crate) struct TreeNode {
    /// The span owned, as positions `[start, end)`.
    pub(crate) start: u32,
    pub(crate) end: u32,
    /// The tokens of the span, which is what a walk compares.
    pub(crate) tokens: Vec<TokenId>,
    /// The pages of logical pages `[page_of(start), pages_for(end))`.
    pub(crate) pages: Vec<PageId>,
    /// The checkpoint whose state is valid at `end`, if one was admitted here.
    pub(crate) ckpt: Option<SlotId>,
    /// Monotonic time of the last touch, which the decay is measured against.
    pub(crate) hit: f64,
    pub(crate) parent: Option<NodeId>,
    /// First child; further siblings chain through [`TreeNode::next`].
    pub(crate) child: Option<NodeId>,
    pub(crate) next: Option<NodeId>,
    /// Set when the node was dropped, so an arena slot is never mistaken for a
    /// leaf that eviction could pick again.
    pub(crate) dropped: bool,
}

impl TreeNode {
    /// The root: an empty span at position zero that owns no tokens and no pages.
    fn root_node() -> Self {
        Self {
            start: 0,
            end: 0,
            tokens: Vec::new(),
            pages: Vec::new(),
            ckpt: None,
            hit: 0.0,
            parent: None,
            child: None,
            next: None,
            dropped: false,
        }
    }
}

/// The deepest node a walk reached, the path to it, and how much matched.
///
/// `matched` can land anywhere inside `node`'s span, including at its start when
/// the first token of the child's span was the one that diverged. That is why a
/// match is followed by a split rather than by a search for a node ending there.
#[derive(Debug)]
pub(crate) struct Walk {
    /// The deepest node reached, possibly only partway into its span.
    pub(crate) node: NodeId,
    /// Every node from the root to [`Walk::node`], root first.
    pub(crate) path: Vec<NodeId>,
    /// Tokens matched, satisfying `node.start <= matched <= node.end`.
    pub(crate) matched: u32,
}

/// Holder counts per arena slot, shared with every [`LiveGuard`].
///
/// A guard outlives the borrow that minted it and still has to decrement when it
/// drops, so the counts sit behind their own lock rather than in the arena. That
/// is also what lets a frontier be released from inside a cache-lock critical
/// section: releasing a holder takes this lock and nothing else, so it cannot
/// invert the inner-then-pool-then-live order.
#[derive(Debug, Clone)]
pub(crate) struct LiveTable(Arc<Mutex<Vec<u32>>>);

impl LiveTable {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }

    /// Takes one holder, growing the table alongside the arena.
    fn pin(&self, id: NodeId) {
        let mut counts = self.0.lock();
        let index = id.index();
        if counts.len() <= index {
            counts.resize(index + 1, 0);
        }
        counts[index] += 1;
    }

    fn release(&self, id: NodeId) {
        let mut counts = self.0.lock();
        if let Some(count) = counts.get_mut(id.index()) {
            *count = count.saturating_sub(1);
        }
    }

    fn holders(&self, id: NodeId) -> u32 {
        self.0.lock().get(id.index()).copied().unwrap_or(0)
    }
}

/// Keeps a node's holder count above zero for as long as a caller borrows it.
///
/// The increment happens on creation and the decrement in [`Drop`], so there is no
/// signed-delta call for a caller to forget on an early return, and no path by
/// which a cancelled request pins a node forever.
#[derive(Debug)]
pub(crate) struct LiveGuard {
    node: NodeId,
    live: LiveTable,
}

impl LiveGuard {
    /// The node this guard keeps pinned.
    #[must_use]
    pub(crate) fn node(&self) -> NodeId {
        self.node
    }
}

impl Drop for LiveGuard {
    fn drop(&mut self) {
        self.live.release(self.node);
    }
}

/// The span tree: an arena of spans, the pool their pages came from, and the live
/// counts that keep a span out of eviction.
///
/// # Slot reclamation
///
/// A dropped span returns its arena slot to a free list, so a process that keeps
/// evicting and re-adding spans does not grow the arena forever. A slot is reused
/// only when its holder count is back to zero, which is checked rather than
/// assumed: a slot with holders left is left alone instead of handed out with a
/// count that belongs to a span nobody can name.
#[derive(Debug)]
pub(crate) struct SpanTree {
    /// Nodes, indexed by [`NodeId`].
    arena: Vec<TreeNode>,
    /// Recycled arena slots, lowest index first.
    free: Vec<NodeId>,
    root: NodeId,
    geometry: CacheGeometry,
    pool: Arc<PagePool>,
    live: LiveTable,
    /// Pages released because their span was dropped.
    page_evictions: u64,
}

impl SpanTree {
    /// An empty tree: the root alone, which is the empty prefix.
    pub(crate) fn new(geometry: CacheGeometry, pool: Arc<PagePool>) -> Self {
        Self {
            arena: vec![TreeNode::root_node()],
            free: Vec::new(),
            root: NodeId::new(0),
            geometry,
            pool,
            live: LiveTable::new(),
            page_evictions: 0,
        }
    }

    /// The root, which is the empty prefix every walk starts from.
    pub(crate) const fn root(&self) -> NodeId {
        self.root
    }

    pub(crate) const fn geometry(&self) -> &CacheGeometry {
        &self.geometry
    }

    /// Live spans, which is what `CacheStats::tree_nodes` reports.
    pub(crate) fn node_count(&self) -> u32 {
        to_u32(self.arena.len() - self.free.len())
    }

    /// Pages released because their span was dropped.
    pub(crate) const fn page_evictions(&self) -> u64 {
        self.page_evictions
    }

    /// The span behind an id, which is also what a page-table build walks up.
    pub(crate) fn node(&self, id: NodeId) -> &TreeNode {
        &self.arena[id.index()]
    }

    /// Holders of one node, which is zero for a slot that never had one.
    pub(crate) fn holders(&self, id: NodeId) -> u32 {
        self.live.holders(id)
    }

    /// Pins a node for as long as the returned guard lives.
    ///
    /// Taking the holder before the caller adopts any page is the point: a resume
    /// that adopted pages first could have them returned to the free stack by a
    /// concurrent eviction and handed to a new span mid-read.
    pub(crate) fn live_guard(&self, id: NodeId) -> LiveGuard {
        self.live.pin(id);
        LiveGuard {
            node: id,
            live: self.live.clone(),
        }
    }

    /// Records that a span was just used, which is what the decay measures.
    pub(crate) fn touch(&mut self, id: NodeId, now: f64) {
        self.arena[id.index()].hit = now;
    }

    /// The checkpoint attached to a span's end, if one is still there.
    pub(crate) fn ckpt_at(&self, id: NodeId) -> Option<SlotId> {
        self.arena[id.index()].ckpt
    }

    /// Points a span at the checkpoint admitted at its end.
    pub(crate) fn attach_ckpt(&mut self, id: NodeId, slot: SlotId) {
        self.arena[id.index()].ckpt = Some(slot);
    }

    /// Forgets a span's checkpoint without touching the slot.
    ///
    /// The release path clears the slot first, and a cleared slot has no owner
    /// left to clear, so the pointer is removed here. A span is never refused for
    /// pointing at a slot that is already gone.
    pub(crate) fn detach_ckpt(&mut self, id: NodeId) {
        self.arena[id.index()].ckpt = None;
    }

    /// The end of the nearest checkpoint strictly below this node, or zero.
    ///
    /// This is what a checkpoint's value is measured against: dropping it costs
    /// the prefill between it and the next checkpoint below, not the whole
    /// sequence, which is what makes a crowded checkpoint cheap and a lone one
    /// expensive.
    pub(crate) fn lower_ckpt_pos(&self, id: NodeId) -> u32 {
        let mut cur = self.arena[id.index()].parent;
        while let Some(parent) = cur {
            let node = &self.arena[parent.index()];
            if node.ckpt.is_some() {
                return node.end;
            }
            cur = node.parent;
        }
        0
    }

    /// Whether `ancestor` is `id` or one of its ancestors.
    pub(crate) fn is_ancestor(&self, ancestor: NodeId, id: NodeId) -> bool {
        let mut cur = Some(id);
        while let Some(node) = cur {
            if node == ancestor {
                return true;
            }
            cur = self.arena[node.index()].parent;
        }
        false
    }

    /// The children of a node, newest first, which is the order they were added.
    pub(crate) fn children(&self, id: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut cur = self.arena[id.index()].child;
        while let Some(child) = cur {
            out.push(child);
            cur = self.arena[child.index()].next;
        }
        out
    }

    /// Every live leaf in arena order, root excluded.
    ///
    /// The root is excluded because it owns no pages: dropping it would empty the
    /// cache while freeing nothing, and an eviction loop would make no progress.
    pub(crate) fn leaves(&self) -> Vec<NodeId> {
        (0..self.arena.len())
            .filter(|index| {
                let node = &self.arena[*index];
                !node.dropped && node.child.is_none() && *index != self.root.index()
            })
            .map(|index| NodeId::new(to_u32(index)))
            .collect()
    }

    /// The path from the root to `id`, root first.
    pub(crate) fn path_to(&self, id: NodeId) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut cur = Some(id);
        while let Some(node) = cur {
            path.push(node);
            cur = self.arena[node.index()].parent;
        }
        path.reverse();
        path
    }
}

impl SpanTree {
    // --------------------------------------------------------------
    // The write path
    // --------------------------------------------------------------

    /// The deepest node whose span matches `tokens`, and how much matched.
    ///
    /// The walk descends by first token, which is enough because two children of
    /// one node can never start with the same token: a commit that would create
    /// such a sibling is refused rather than allowed to make the walk ambiguous.
    ///
    /// Nothing is carried between steps but the node just reached. An eviction can
    /// drop the node a walk stopped at, and a stale id would then name whatever
    /// recycled its arena slot, so every step re-reads the arena through the link it
    /// just followed.
    #[must_use]
    pub(crate) fn walk_down(&self, tokens: &[TokenId]) -> Walk {
        let mut path = vec![self.root];
        let mut node = self.root;
        let mut matched = 0u32;
        while let Ok(at) = usize::try_from(matched) {
            let Some(first) = tokens.get(at).copied() else {
                break;
            };
            let Some(child) = self.child_for(node, first) else {
                break;
            };
            let common = self.arena[child.index()]
                .tokens
                .iter()
                .zip(tokens[at..].iter())
                .take_while(|(cached, want)| cached == want)
                .count();
            matched += to_u32(common);
            node = child;
            path.push(child);
            if common < self.arena[child.index()].tokens.len() {
                // The spans diverged partway in, so `matched` sits inside this node.
                break;
            }
        }
        Walk {
            node,
            path,
            matched,
        }
    }

    /// The child of `parent` whose span starts with `first`.
    fn child_for(&self, parent: NodeId, first: TokenId) -> Option<NodeId> {
        let mut cur = self.arena[parent.index()].child;
        while let Some(id) = cur {
            let child = &self.arena[id.index()];
            if child.tokens.first() == Some(&first) {
                return Some(id);
            }
            cur = child.next;
        }
        None
    }

    /// Make `walk.matched` a node boundary and return the node whose span ends there.
    ///
    /// A match usually ends inside a span, and the pages a commit writes next have
    /// to line up with a node end, so the boundary is created here rather than
    /// searched for. Splitting is the only way a read changes the tree's shape:
    /// nothing is added or dropped, and the two halves keep the same tokens and
    /// pages between them.
    ///
    /// When the match stops at the very start of the reached node's span, the first
    /// token of that span was the one that diverged, so the frontier is the node
    /// above it instead.
    pub(crate) fn frontier_at(&mut self, walk: &Walk) -> Result<NodeId, KvError> {
        let node = walk.node;
        if self.arena[node.index()].end == walk.matched {
            return Ok(node);
        }
        if walk.matched == self.arena[node.index()].start {
            return self.arena[node.index()].parent.ok_or(KvError::NodeRefused {
                reason: "the root owns no span to split",
            });
        }
        self.split_at(node, walk.matched)
    }
    /// Cut `tail`'s span at `at`, returning the new node that ends there.
    ///
    /// The earlier half becomes a new arena slot and the original keeps the later
    /// half, its children and its checkpoint, because all of those hang off the *end*
    /// of a span. The page `at` sits inside is held by both halves: each owns
    /// different rows of it, and neither may release it while the other still reads
    /// its own rows, so it gains a holder instead of moving.
    ///
    /// A live count follows the rows rather than the id: a guard was minted for the
    /// node as it was before the split, and the sequence holding it reads the earlier
    /// half, which is the half the new node owns.
    pub(crate) fn split_at(&mut self, tail: NodeId, at: u32) -> Result<NodeId, KvError> {
        let (start, end, parent, sibling, hit, tokens, pages) = {
            let node = &self.arena[tail.index()];
            (
                node.start,
                node.end,
                node.parent,
                node.next,
                node.hit,
                node.tokens.clone(),
                node.pages.clone(),
            )
        };
        if at <= start || at >= end {
            return Err(KvError::NodeRefused {
                reason: "the split falls outside the span",
            });
        }
        let parent = parent.ok_or(KvError::NodeRefused {
            reason: "the root owns no span to split",
        })?;
        // Both page counts come from the geometry before any array is touched. The
        // head owns every page from the one `start` sits in up to and including the one
        // `at` sits in; the tail keeps from the page `at` starts in. Those are the same
        // page when the split is interior and neighbours when it is not, which is the
        // whole difference between an interior split and a page-aligned one.
        //
        // The lower bound is the page `start` sits in, not the number of pages below
        // it: a node whose span starts mid-page holds that page's earlier rows too, and
        // counting from the page boundary above would hand the head one page short and
        // leave the path with a gap no node owns.
        let head_n = usize::try_from(self.geometry.pages_for(at) - self.geometry.page_of(start))
            .unwrap_or(usize::MAX);
        let keep = usize::try_from(self.geometry.page_of(at) - self.geometry.page_of(start))
            .unwrap_or(usize::MAX);
        let offset = usize::try_from(at - start).unwrap_or(usize::MAX);
        let head = self.alloc_slot();
        let split = offset.min(tokens.len());
        let head_tokens = tokens[..split].to_vec();
        let tail_tokens = tokens[split..].to_vec();
        let head_pages = pages[..head_n.min(pages.len())].to_vec();
        let tail_pages = pages[keep.min(pages.len())..].to_vec();
        // The head co-holds its own pages. The tail already holds all of them, so
        // only a shared boundary page's count changes here, which is what keeps that
        // page alive while either half can still read a row of it.
        for page in &head_pages {
            self.pool.add_holder(*page);
        }
        {
            let node = &mut self.arena[head.index()];
            node.start = start;
            node.end = at;
            node.tokens = head_tokens;
            node.pages = head_pages;
            node.hit = hit;
            node.parent = Some(parent);
            node.child = Some(tail);
            node.next = sibling;
        }
        {
            let node = &mut self.arena[tail.index()];
            node.start = at;
            node.tokens = tail_tokens;
            node.pages = tail_pages;
            node.parent = Some(head);
            node.next = None;
        }
        // The head takes the tail's place in the parent's child list, so a walk
        // reaches the shared prefix without knowing a split happened.
        let mut cur = self.arena[parent.index()].child;
        let mut prev: Option<NodeId> = None;
        while let Some(child) = cur {
            if child == tail {
                match prev {
                    Some(sib) => self.arena[sib.index()].next = Some(head),
                    None => self.arena[parent.index()].child = Some(head),
                }
                break;
            }
            prev = Some(child);
            cur = self.arena[child.index()].next;
        }
        // The holder count stays on `tail`. A guard was minted for that slot and is
        // the only thing that can give the count back, so a count carried to the new
        // head would be decremented nowhere and that head would stay pinned for the
        // rest of the process. The head is unreachable to eviction anyway: it has a
        // child now, and only a leaf is ever picked. The pin is on the path.
        Ok(head)
    }
    /// A zeroed arena slot for a new span, reusing a dropped one when possible.
    ///
    /// Reuse is last-in-first-out. A dropped slot is unreachable from the root the
    /// moment it is dropped, so handing it back later cannot hand a walk a node it
    /// can still reach: the only way back into a slot is from the root.
    fn alloc_slot(&mut self) -> NodeId {
        if let Some(id) = self.free.pop() {
            let node = &mut self.arena[id.index()];
            node.start = 0;
            node.end = 0;
            node.tokens = Vec::new();
            node.pages = Vec::new();
            node.ckpt = None;
            node.hit = 0.0;
            node.parent = None;
            node.child = None;
            node.next = None;
            node.dropped = false;
            return id;
        }
        self.arena.push(TreeNode {
            start: 0,
            end: 0,
            tokens: Vec::new(),
            pages: Vec::new(),
            ckpt: None,
            hit: 0.0,
            parent: None,
            child: None,
            next: None,
            dropped: false,
        });
        NodeId::new(to_u32(self.arena.len() - 1))
    }

    /// Hang `tokens`, and the pages holding their rows, under `parent`.
    ///
    /// The span starts at `parent.end` by construction: a node owns the positions
    /// directly after its parent's, and anything else would leave a gap the tree
    /// claims to cover but no node owns -- a page a later `PageTable::from_path`
    /// could not account for.
    ///
    /// The tree takes its own holder on each page rather than assuming the caller's
    /// page table will hand its own over, so a caller that keeps writing into those
    /// pages and one that releases them both leave the refcount right.
    ///
    /// A refused add costs a cached span and nothing else: the caller keeps its pages
    /// and the frontier it came in with, and the tree is exactly as it was.
    pub(crate) fn add(
        &mut self,
        parent: NodeId,
        tokens: &[TokenId],
        pages: &[PageId],
        now: f64,
    ) -> Result<NodeId, KvError> {
        if tokens.is_empty() {
            return Err(KvError::NodeRefused {
                reason: "an empty span would be reachable by every path",
            });
        }
        let node = &self.arena[parent.index()];
        if node.dropped {
            return Err(KvError::NodeRefused {
                reason: "the parent span was dropped",
            });
        }
        let start = node.end;
        let end = start + to_u32(tokens.len());
        // The span's first page is the one `start` sits in, which the parent also holds
        // whenever `start` is mid-page -- the two own different rows of it, and the
        // deeper node's copy is the one a path read keeps. Counting from the page
        // boundary above `start` instead would ask for one page fewer than the span
        // covers and leave the page array with a hole in it.
        let want = usize::try_from(self.geometry.pages_for(end) - self.geometry.page_of(start))
            .unwrap_or(usize::MAX);
        if pages.len() != want {
            return Err(KvError::NodeRefused {
                reason: "the page array does not cover the span",
            });
        }
        // The tree takes its own holder on every page it lists. `detach` releases one
        // holder per page it lists, so a span that never took one would hand back a page
        // its caller is still writing into, and the pool would hand that page to a second
        // sequence while the first was still writing rows into it.
        for page in pages {
            self.pool.add_holder(*page);
        }
        let id = self.alloc_slot();
        {
            let node = &mut self.arena[id.index()];
            node.start = start;
            node.end = end;
            node.tokens = tokens.to_vec();
            node.pages = pages.to_vec();
            node.hit = now;
            node.parent = Some(parent);
        }
        // is walked here both to reject a duplicate first token and to find the tail
        // to append to.
        let mut cur = self.arena[parent.index()].child;
        let mut prev: Option<NodeId> = None;
        while let Some(child) = cur {
            if self.arena[child.index()].tokens.first() == tokens.first() {
                return Err(KvError::NodeRefused {
                    reason: "two children would start with the same token",
                });
            }
            prev = Some(child);
            cur = self.arena[child.index()].next;
        }
        match prev {
            Some(sib) => self.arena[sib.index()].next = Some(id),
            None => self.arena[parent.index()].child = Some(id),
        }
        Ok(id)
    }

    /// Drop a leaf span and report what it gave up.
    ///
    /// Only a leaf can be dropped: an interior drop would either orphan its subtree
    /// or leave a child pointing at positions the tree no longer covers, and either
    /// way a later walk would read a page nobody owns.
    ///
    /// The pages are unreffed here, so a page shared with a sibling at a split
    /// survives: its count does not reach zero until the last holder goes, and only
    /// the drop that takes the count to zero is reported as freed.
    ///
    /// The slot is recycled only after it is unlinked, so no live path can walk into
    /// a recycled slot: a dropped slot is unreachable from the root either way, and
    /// a caller that pinned it could not have reached this call at all.
    pub(crate) fn detach(&mut self, id: NodeId) -> Result<Vec<PageId>, KvError> {
        let (pages, ckpt, parent, next, child) = {
            let node = &self.arena[id.index()];
            if node.dropped {
                return Err(KvError::NodeRefused {
                    reason: "the span was already dropped",
                });
            }
            if node.child.is_some() {
                return Err(KvError::NodeRefused {
                    reason: "an interior span would orphan its continuation",
                });
            }
            (
                node.pages.clone(),
                node.ckpt,
                node.parent,
                node.next,
                node.child,
            )
        };
        let _ = child;
        // Counted before the drop: a page shared with a sibling at a split has more
        // than one holder, and only the last drop actually frees it.
        let freed = pages
            .iter()
            .filter(|page| self.pool.holders(**page) == Some(1))
            .copied()
            .collect::<Vec<_>>();
        for page in &pages {
            self.pool.unref(*page);
        }
        self.page_evictions += u64::try_from(freed.len()).unwrap_or(u64::MAX);
        if let Some(parent) = parent {
            let mut cur = self.arena[parent.index()].child;
            let mut prev: Option<NodeId> = None;
            while let Some(sib) = cur {
                if sib == id {
                    match prev {
                        Some(sib) => self.arena[sib.index()].next = next,
                        None => self.arena[parent.index()].child = next,
                    }
                    break;
                }
                prev = Some(sib);
                cur = self.arena[sib.index()].next;
            }
        }
        let node = &mut self.arena[id.index()];
        node.dropped = true;
        node.tokens = Vec::new();
        node.pages = Vec::new();
        node.parent = None;
        node.next = None;
        self.free.push(id);
        // The checkpoint pointer dies with the slot it was recycled into: the caller
        // released the slot before asking for the drop, which is the only order that
        // cannot leave a node advertising state that has been handed to someone else.
        let _ = ckpt;
        Ok(freed)
    }

    /// The page array for positions `[0, n_pos)`, gathered from the root to `node`.
    ///
    /// A page a node owns is placed at its logical page index, so a page two nodes
    /// share -- the one a split landed inside -- is written by both, and the deepest
    /// node wins. That is the rule a resumed sequence needs: the one page a shared
    /// prefix and its continuation share is the one the deeper node wrote.
    ///
    /// A position no node on the path owns is a gap rather than a zero entry, because
    /// a page table with a hole in it would hand a kernel a page index that means
    /// nothing.
    pub(crate) fn page_array_from_path(
        &self,
        node: NodeId,
        n_pos: u32,
    ) -> Result<Vec<PageId>, KvError> {
        let pages = usize::try_from(self.geometry.pages_for(n_pos)).unwrap_or(usize::MAX);
        let mut out: Vec<Option<PageId>> = vec![None; pages];
        for node in self.path_to(node) {
            let node = &self.arena[node.index()];
            let base = self.geometry.page_of(node.start);
            for (i, page) in node.pages.iter().enumerate() {
                let index = usize::try_from(base + to_u32(i)).unwrap_or(usize::MAX);
                if index < out.len() {
                    out[index] = Some(*page);
                }
            }
        }
        out.into_iter()
            .enumerate()
            .map(|(index, page)| {
                page.ok_or(KvError::PathGap {
                    position: to_u32(index),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shiso_core::TokenId;

    const PAGE_TOKENS: u32 = 256;

    fn geometry() -> CacheGeometry {
        CacheGeometry::new(PAGE_TOKENS, 8_388_608, 1_048_576, 0.001, 600.0)
    }

    fn tokens(first: u32, len: u32) -> Vec<TokenId> {
        (0..len).map(|i| TokenId::new(first + i)).collect()
    }

    /// Claims `n` pages the way a caller's page table would: one holder each.
    fn claim(pool: &PagePool, n: usize) -> Vec<PageId> {
        (0..n).map(|_| pool.take().expect("a free page")).collect()
    }
    #[test]
    fn the_tree_holds_its_own_reference_on_the_pages_it_lists() {
        let pool = Arc::new(PagePool::new(4));
        let mut tree = SpanTree::new(geometry(), Arc::clone(&pool));
        let pages = claim(&pool, 2);
        let node = tree
            .add(tree.root, &tokens(0, 300), &pages, 1.0)
            .expect("a span the pool has pages for");
        for page in &pages {
            assert_eq!(pool.holders(*page), Some(2), "the tree must hold its own");
        }
        let freed = tree.detach(node).expect("the leaf drops");
        assert!(freed.is_empty(), "a page its caller lists cannot be freed");
        for page in &pages {
            assert_eq!(pool.holders(*page), Some(1), "the caller's holder survives");
        }
    }
    #[test]
    fn a_span_starting_mid_page_owns_the_page_it_starts_in() {
        let pool = Arc::new(PagePool::new(8));
        let mut tree = SpanTree::new(geometry(), Arc::clone(&pool));
        let pages = claim(&pool, 2);
        let tail = tree
            .add(tree.root, &tokens(0, 300), &pages, 1.0)
            .expect("a span");
        let head = tree.split_at(tail, 100).expect("a split mid page");
        assert_eq!(tree.node(head).pages.len(), 1);
        assert_eq!(tree.node(tail).pages.len(), 2);
        assert_eq!(pool.holders(pages[0]), Some(3));
        let array = tree
            .page_array_from_path(tail, 300)
            .expect("the deeper node owns the shared page's later rows");
        assert_eq!(array, vec![pages[0], pages[1]]);
    }

    #[test]
    fn a_live_count_stays_with_the_guard_that_minted_it() {
        let pool = Arc::new(PagePool::new(8));
        let mut tree = SpanTree::new(geometry(), Arc::clone(&pool));
        let pages = claim(&pool, 2);
        let tail = tree
            .add(tree.root, &tokens(0, 300), &pages, 1.0)
            .expect("a span");
        let guard = tree.live_guard(tail);
        let head = tree.split_at(tail, 100).expect("a split mid page");

        // The count is still on the slot the guard will decrement, and the head is
        // protected by having a child rather than by a count: eviction only ever
        // picks a leaf, so the earlier half cannot go while the later half is alive.
        assert_eq!(tree.holders(tail), 1);
        assert_eq!(tree.holders(head), 0);
        assert_eq!(tree.pick_victim(2.0, &|_id, _node| 0.0), None);

        drop(guard);
        assert_eq!(tree.holders(tail), 0);
        assert_eq!(tree.pick_victim(2.0, &|_id, _node| 0.0), Some(tail));
    }

    #[test]
    fn two_children_cannot_start_with_the_same_token() {
        let pool = Arc::new(PagePool::new(8));
        let mut tree = SpanTree::new(geometry(), Arc::clone(&pool));
        tree.add(tree.root, &tokens(0, 10), &claim(&pool, 1), 1.0)
            .expect("the first child");
        let error = tree
            .add(tree.root, &tokens(0, 12), &claim(&pool, 1), 1.0)
            .expect_err("two children cannot start with the same token");
        assert!(matches!(error, KvError::NodeRefused { .. }));
    }

    #[test]
    fn a_span_without_a_page_for_every_it_owns_is_refused() {
        let pool = Arc::new(PagePool::new(8));
        let mut tree = SpanTree::new(geometry(), Arc::clone(&pool));
        let error = tree
            .add(tree.root, &tokens(0, 300), &claim(&pool, 1), 1.0)
            .expect_err("one page cannot cover 300 positions");
        assert!(matches!(error, KvError::NodeRefused { .. }));
    }

    #[test]
    fn a_path_past_what_its_spans_own_is_a_gap_not_a_zero() {
        let pool = Arc::new(PagePool::new(8));
        let mut tree = SpanTree::new(geometry(), Arc::clone(&pool));
        let node = tree
            .add(tree.root, &tokens(0, 300), &claim(&pool, 2), 1.0)
            .expect("a span");
        assert!(tree.page_array_from_path(node, 600).is_err());
        assert_eq!(
            tree.page_array_from_path(node, 300).expect("covered").len(),
            2
        );
    }

    #[test]
    fn a_dropped_slot_is_reused_rather_than_appended_forever() {
        let pool = Arc::new(PagePool::new(8));
        let mut tree = SpanTree::new(geometry(), Arc::clone(&pool));
        let first = tree
            .add(tree.root, &tokens(0, 10), &claim(&pool, 1), 1.0)
            .expect("a span");
        let after = tree.node_count();
        tree.detach(first).expect("the leaf drops");
        let again = tree
            .add(tree.root, &tokens(9, 10), &claim(&pool, 1), 2.0)
            .expect("the next span");
        assert_eq!(again, first);
        assert_eq!(tree.node_count(), after);
    }
}
