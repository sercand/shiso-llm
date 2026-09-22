//! The page pool: which page ids exist and who is holding each one.
//!
//! # Lock order
//!
//! This crate has three locks and one legal order: **cache inner, then pool**.
//! `KvCache` keeps its tree, checkpoint slots and counters behind one mutex; the
//! pool keeps its free stack and refcounts behind another, held by `Arc` and
//! cloned into every `PageTable`.
//!
//! The split exists because of `PageTable::Drop`. A page table returns its pages
//! when it goes out of scope, and page tables go out of scope *inside* the
//! cache's own critical sections: `commit` replaces one, `evict_to` orphans a
//! node's table, `restore` discards one. With one mutex for the whole cache, that drop
//! would try to retake a lock the same thread already holds. With the pool behind
//! its own mutex the drop is legal, and the cost of the split is one rule a
//! reviewer never has to remember by hand: never take the inner lock while holding
//! the pool lock.
//!
//! The rule is machine-checked rather than left to review. A thread-local flag
//! records which lock the current thread holds, taking the inner lock while the
//! pool lock is held trips a `debug_assert`, and the pool's own methods assert
//! that they take nothing else.
//!
//! # Why refcounts rather than an owner
//!
//! A page is routinely held by more than one party: the tree node whose span
//! covers it, the page table of every sequence indexing it, and — for the one page
//! a split lands inside — both halves of that span. The last holder returns it to
//! the free stack, so no single party has to know about the others.

use std::cell::Cell;
use std::collections::VecDeque;
use std::fmt;

use parking_lot::Mutex;

use crate::ids::{PageId, to_u32};

thread_local! {
    /// Set while the current thread holds the cache's inner mutex.
    static HOLDS_INNER: Cell<bool> = const { Cell::new(false) };
    /// Set while the current thread holds the pool lock.
    static HOLDS_POOL: Cell<bool> = const { Cell::new(false) };
}

/// Marks the current thread as holding the cache's inner lock.
///
/// Taking the inner lock while already holding the pool lock is the forbidden
/// direction, so it is asserted against rather than left to review.
#[derive(Debug)]
pub(crate) struct InnerGuard;

impl InnerGuard {
    #[must_use]
    pub(crate) fn new() -> Self {
        debug_assert!(
            !HOLDS_POOL.with(Cell::get),
            "lock order violated: taking the cache lock while holding the pool lock"
        );
        HOLDS_INNER.with(|held| held.set(true));
        Self
    }
}

impl Drop for InnerGuard {
    fn drop(&mut self) {
        HOLDS_INNER.with(|held| held.set(false));
    }
}

/// Marks the current thread as holding the pool lock.
///
/// The pool lock may be taken while holding the inner lock, which is the normal
/// direction, and may be taken alone, which is what `PageTable::Drop` does.
#[derive(Debug)]
struct PoolGuard;

impl PoolGuard {
    fn new() -> Self {
        HOLDS_POOL.with(|held| held.set(true));
        Self
    }
}

impl Drop for PoolGuard {
    fn drop(&mut self) {
        HOLDS_POOL.with(|held| held.set(false));
    }
}

/// Every page id the cache can hand out, with a holder count per page.
#[derive(Debug)]
pub(crate) struct PagePool {
    inner: Mutex<PoolInner>,
}

#[derive(Debug)]
struct PoolInner {
    /// Free page ids. Popping the tail hands out highest-id-first; draining the
    /// front is lowest-first.
    free_ids: VecDeque<PageId>,
    /// Holders per page id. A zero entry means the page is free.
    refs: Vec<u32>,
    /// Hand pages out from the tail instead of the front. Test-only: it makes the
    /// scrambled allocation order observable without changing any rule.
    reverse: bool,
}

impl PagePool {
    /// Builds a pool of `n_pages` pages, all free.
    #[must_use]
    pub(crate) fn new(n_pages: u32) -> Self {
        Self::with_order(n_pages, false)
    }

    /// As [`Self::new`], with the allocation order chosen.
    ///
    /// `reverse` hands pages out highest-id-first. Nothing in the cache depends on
    /// which id a page has, so a test can run the same scenario under both orders
    /// and prove the invariants are about counts rather than identities.
    #[must_use]
    pub(crate) fn with_order(n_pages: u32, reverse: bool) -> Self {
        let free_ids = (0..n_pages).map(PageId::new).collect();
        Self {
            inner: Mutex::new(PoolInner {
                free_ids,
                refs: vec![0; n_pages as usize],
                reverse,
            }),
        }
    }

    /// Claims a free page, or `None` when the pool is exhausted.
    ///
    /// A fresh page arrives with one holder, the caller that took it, so a caller
    /// never has to remember to take a reference as well.
    #[must_use]
    pub(crate) fn take(&self) -> Option<PageId> {
        let _held = PoolGuard::new();
        let mut inner = self.inner.lock();
        let id = if inner.reverse {
            inner.free_ids.pop_back()?
        } else {
            inner.free_ids.pop_front()?
        };
        let count = inner.refs.get_mut(id.index())?;
        debug_assert_eq!(*count, 0, "a page on the free stack cannot have holders");
        *count = 1;
        Some(id)
    }

    /// Adds a holder to a page the caller already knows exists: the tree taking a
    /// node's share of a page a sequence wrote, or a page table adopting a path.
    pub(crate) fn add_holder(&self, id: PageId) {
        let _held = PoolGuard::new();
        let mut inner = self.inner.lock();
        if let Some(count) = inner.refs.get_mut(id.index()) {
            *count += 1;
        }
    }

    /// Releases one holder; the page returns to the free stack with the last.
    pub(crate) fn unref(&self, id: PageId) {
        let _held = PoolGuard::new();
        let mut inner = self.inner.lock();
        let Some(count) = inner.refs.get_mut(id.index()) else {
            return;
        };
        if *count == 0 {
            // Already free, so this release has no holder behind it and changes
            // nothing. A page joins the free stack exactly once, when its last
            // holder goes; letting an unbalanced release through would put one page
            // on the free stack twice and hand the same page out to two callers.
            return;
        }
        *count -= 1;
        if *count == 0 {
            debug_assert!(
                !inner.free_ids.contains(&id),
                "page {id} returned to the free stack twice"
            );
            inner.free_ids.push_back(id);
        }
    }

    /// Pages currently unclaimed.
    #[must_use]
    pub(crate) fn n_free(&self) -> u32 {
        let _held = PoolGuard::new();
        to_u32(self.inner.lock().free_ids.len())
    }

    /// Pages in the pool, which is also the ceiling on [`PageId`].
    #[must_use]
    pub(crate) fn capacity(&self) -> u32 {
        let _held = PoolGuard::new();
        to_u32(self.inner.lock().refs.len())
    }

    /// Pages currently claimed, which is what `CacheStats::pages_used` reports.
    #[must_use]
    pub(crate) fn n_used(&self) -> u32 {
        let _held = PoolGuard::new();
        let inner = self.inner.lock();
        to_u32(inner.refs.len() - inner.free_ids.len())
    }

    /// Holders of one page, or `None` if the id is outside the pool. A test uses
    /// this to assert a page came back completely.
    #[must_use]
    pub(crate) fn holders(&self, id: PageId) -> Option<u32> {
        let _held = PoolGuard::new();
        self.inner.lock().refs.get(id.index()).copied()
    }

    /// Every page still held, with its holder count: the leak check this crate's own
    /// tests assert on after dropping a page table.
    ///
    /// Test-only because the cache itself has no use for a census of holders -- every
    /// release is a `Drop` -- while "the count came back to zero" is the one assertion
    /// that proves a `Drop` actually ran.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn leak_report(&self) -> Vec<(PageId, u32)> {
        let _held = PoolGuard::new();
        let inner = self.inner.lock();
        inner
            .refs
            .iter()
            .enumerate()
            .filter(|(_, count)| **count != 0)
            .map(|(index, count)| (PageId::new(to_u32(index)), *count))
            .collect()
    }
}

impl fmt::Display for PagePool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.lock();
        write!(
            f,
            "PagePool({} of {} held)",
            inner.refs.len() - inner.free_ids.len(),
            inner.refs.len()
        )
    }
}

/// The pool's allocation and release rules, checked one scenario at a time.
/// The page-table half of the same behaviour lives with `PageTable`.
#[cfg(test)]
mod tests {
    use super::{PageId, PagePool};

    /// Pages come out lowest-id-first, each with exactly one holder: the taker.
    #[test]
    fn hands_out_pages_lowest_first() {
        let pool = PagePool::new(4);
        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.n_free(), 4);
        assert_eq!(pool.n_used(), 0);
        for expected in 0..4u32 {
            let page = pool.take().expect("the pool still has pages");
            assert_eq!(page, PageId::new(expected));
            assert_eq!(pool.n_free(), 3 - expected);
            // Step 3 keeps a refcount per page, so leak_report lists every held
            // page: each take adds one rather than replacing the list.
            let held: Vec<(PageId, u32)> = (0..=expected).map(|id| (PageId::new(id), 1)).collect();
            assert_eq!(pool.leak_report(), held);
        }
        assert_eq!(pool.n_free(), 0);
        assert_eq!(pool.n_used(), 4);
    }

    /// A second holder keeps a page out of the free stack, and the page returns
    /// only when the last holder has gone.
    #[test]
    fn a_second_holder_keeps_a_page_out_of_the_free_stack() {
        let pool = PagePool::new(4);
        let page = pool.take().expect("the pool has pages");
        assert_eq!(pool.n_free(), 3, "the other three pages are still free");
        pool.add_holder(page);
        pool.unref(page);
        // Step 3: n_free counts the whole pool, so the page's own holder count is
        // what shows the last holder had not gone yet.
        assert_eq!(pool.holders(page), Some(1), "the tree still holds the page");
        assert_eq!(
            pool.n_free(),
            3,
            "the held page never reached the free stack"
        );
        pool.unref(page);
        assert_eq!(pool.n_free(), 4, "the last holder returned the page");
        assert_eq!(pool.n_used(), 0);
    }

    /// Step 3: the last holder returns the page. A release with no holder left is
    /// ignored, which is what stops an unbalanced release from putting one page on
    /// the free stack twice.
    #[test]
    fn an_extra_release_does_not_duplicate_a_page() {
        let pool = PagePool::new(2);
        let page = pool.take().expect("a page");
        pool.unref(page);
        assert_eq!(pool.n_free(), 2, "the last holder returned the page");
        pool.unref(page);
        assert_eq!(
            pool.n_free(),
            2,
            "a release with no holder left changes nothing"
        );
        assert_eq!(pool.n_used(), 0);
        assert_eq!(pool.take(), Some(PageId::new(1)));
        assert_eq!(pool.take(), Some(PageId::new(0)));
        assert_eq!(pool.take(), None, "no duplicate page was left to hand out");
    }

    /// An id outside the pool is refused rather than counted.
    #[test]
    fn an_id_outside_the_pool_holds_nothing() {
        let pool = PagePool::new(2);
        let outside = PageId::new(2);
        pool.add_holder(outside); // ignored, and it must not create a count
        pool.unref(outside);
        assert_eq!(pool.n_free(), 2);
        assert!(pool.leak_report().is_empty());
    }

    /// The allocation order is a test knob, not a rule: reversing it must not
    /// change any count.
    #[test]
    fn reverse_order_hands_out_the_highest_id_first() {
        let pool = PagePool::with_order(4, true);
        for expected in (0..4u32).rev() {
            let page = pool.take().expect("a page");
            assert_eq!(page, PageId::new(expected));
        }
    }

    /// Every page returns when its last holder does, which is the property
    /// `PageTable::Drop` relies on.
    #[test]
    fn every_page_comes_back_when_the_last_holder_drops() {
        let pool = PagePool::new(8);
        let pages: Vec<PageId> = (0..8).map(|_| pool.take().expect("a page")).collect();
        assert_eq!(pool.n_free(), 0);
        assert_eq!(pool.leak_report().len(), 8);

        for page in &pages {
            pool.add_holder(*page);
        }
        for page in &pages {
            pool.unref(*page);
        }
        assert_eq!(pool.n_free(), 0, "one holder per page remains");
        for page in pages {
            pool.unref(page);
        }
        assert_eq!(pool.n_free(), 8);
        assert_eq!(pool.n_used(), 0);
        assert!(pool.leak_report().is_empty());
    }
}
