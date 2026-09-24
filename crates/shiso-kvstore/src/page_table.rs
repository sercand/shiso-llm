//! A live sequence's page table: one page id per logical page position.
//!
//! A page table is the session's view of the cache — the array the KV kernels
//! translate a position through, plus the holders that keep those pages allocated.
//! It is the only type in the crate whose whole purpose is to give something back
//! on `Drop`, and the reason it can: `Drop` takes the pool lock and nothing else,
//! so it is legal inside a cache-lock critical section (see the module doc in
//! [`crate::pool`]).
//!
//! # The boundary-page rule
//!
//! A page can be owned by a tree node *and* still be written by a live sequence:
//! the page a shared prefix ends in is half shared rows, half the new tokens. When
//! the sequence commits that span, the node takes a holder and the sequence keeps
//! its own, so the refcount goes to two and the page outlives either side alone.
//!
//! When a child node's first page is its parent's last page, the **child's copy is
//! the one the tree keeps**: the child wrote the later rows, and the parent's rows
//! inside that page are already covered by the child's span. Getting this backwards
//! leaves a node advertising rows nobody wrote.
//!
//! # Short pages
//!
//! The last page of a path is usually only partly filled. It is still a whole
//! page's worth of allocation and a whole page's worth of eviction credit, and the
//! cache does not pretend otherwise: a path of 257 tokens costs two pages.

use std::sync::Arc;

use crate::error::KvError;
use crate::geom::CacheGeometry;
use crate::ids::PageId;
use crate::pool::PagePool;

/// One page id per logical page position of a live sequence.
#[derive(Debug)]
pub struct PageTable {
    pool: Arc<PagePool>,
    geometry: CacheGeometry,
    /// The array the kernels index through.
    pages: Vec<PageId>,
}

impl PageTable {
    /// An empty table for a sequence that has written nothing yet.
    pub(crate) fn new(pool: Arc<PagePool>, geometry: CacheGeometry) -> Self {
        Self {
            pool,
            geometry,
            pages: Vec::new(),
        }
    }

    /// A table covering `pages`, taking one holder on each.
    ///
    /// Used by `lookup` for a path the tree already owns and by `restore` for a
    /// path rebuilt from stored bytes: in both cases the pages exist and somebody
    /// else holds them too, so this adopts rather than claims.
    pub(crate) fn adopt(pool: Arc<PagePool>, geometry: CacheGeometry, pages: Vec<PageId>) -> Self {
        for page in &pages {
            pool.add_holder(*page);
        }
        Self {
            pool,
            geometry,
            pages,
        }
    }

    /// Logical pages covered so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Whether the table covers no positions at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// The page holding `position`, or `None` past the end of the path.
    #[must_use]
    pub fn page_of(&self, position: u32) -> Option<PageId> {
        self.pages
            .get(usize::try_from(self.geometry.page_of(position)).unwrap_or(usize::MAX))
            .copied()
    }

    /// The page index array the KV kernels translate through.
    ///
    /// Borrowed, because the engine re-uploads it only when it changes.
    #[must_use]
    pub fn as_slice(&self) -> &[PageId] {
        &self.pages
    }

    /// Claims pages until the table covers positions `[0, n_pos)`.
    ///
    /// Returns `false` without claiming anything when the pool is short: a partial
    /// table would let a sequence write into positions the cache cannot account
    /// for, and the caller's alternative — evict and retry — is always available.
    pub(crate) fn grow_to(&mut self, n_pos: u32) -> Result<bool, KvError> {
        let want = usize::try_from(self.geometry.pages_for(n_pos)).unwrap_or(usize::MAX);
        let needed = want.saturating_sub(self.pages.len());
        if needed == 0 {
            return Ok(true);
        }
        if needed > usize::try_from(self.pool.n_free()).unwrap_or(usize::MAX) {
            return Ok(false);
        }
        for _ in 0..needed {
            let page = self.pool.take().ok_or(KvError::PoolEmpty)?;
            self.pages.push(page);
        }
        Ok(true)
    }

    /// Shortens the table to at most `n` pages, releasing the ones above that.
    ///
    /// The same shape as `Vec::truncate`: a sequence truncated back to a branch
    /// point keeps `pages_for(matched_end)` pages, which is a count to keep rather
    /// than a count to drop.
    pub(crate) fn truncate(&mut self, n: usize) {
        while self.pages.len() > n {
            if let Some(page) = self.pages.pop() {
                self.pool.unref(page);
            }
        }
    }

    /// Swaps the page at `index` for `page`, releasing this table's holder on the
    /// page that was there.
    ///
    /// This is the commit half of copy-on-write: a resume takes a private copy of the
    /// one page its checkpoint lands inside and installs it here, so the shared page
    /// keeps only the holders that were reading it before. The new page arrives with
    /// its one holder already counted, which is why it is *installed* rather than
    /// referenced.
    ///
    /// Panics when `index` is past the end, which no call site can reach: the only
    /// caller indexes the last page of a table it has just built.
    pub(crate) fn replace(&mut self, index: usize, page: PageId) {
        let old = std::mem::replace(&mut self.pages[index], page);
        self.pool.unref(old);
    }

    /// The geometry this table's pages were sized against.
    #[must_use]
    pub fn geometry(&self) -> &CacheGeometry {
        &self.geometry
    }
}

impl Drop for PageTable {
    /// Releases every holder this table took.
    ///
    /// Only the pool lock is taken, which is the whole reason the pool has its own
    /// mutex: a table is routinely dropped while the caller holds the cache lock,
    /// and a single-mutex design would deadlock right here.
    fn drop(&mut self) {
        for page in &self.pages {
            self.pool.unref(*page);
        }
    }
}

/// The page-table half of the upstream page-table scenario: grow, adopt, read
/// back, truncate. The path-walking half needs a span tree and lives with the
/// tree tests instead.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::PageTable;
    use crate::geom::CacheGeometry;
    use crate::ids::PageId;
    use crate::pool::PagePool;

    fn geometry() -> CacheGeometry {
        CacheGeometry::new(256, 8_388_608, 53_684_224, 0.0014, 600.0)
    }

    /// A page covers the whole first chunk of positions, and one more page arrives
    /// only when a position lands past it.
    #[test]
    fn growing_claims_one_page_per_page_of_positions() {
        let pool = Arc::new(PagePool::new(4));
        let mut table = PageTable::new(Arc::clone(&pool), geometry());
        assert!(table.is_empty());
        assert!(table.grow_to(700).expect("the pool has pages"));
        assert_eq!(table.len(), 3);
        assert_eq!(
            table.as_slice(),
            &[PageId::new(0), PageId::new(1), PageId::new(2)]
        );
        // Step 4 sizes a table in pages, so its extent is whole pages rather than
        // the position it was grown to: three pages of 256 cover 0..=767, which
        // holds 700, and only 768 is past the end.
        assert_eq!(table.page_of(700), Some(PageId::new(2)));
        assert_eq!(table.page_of(767), Some(PageId::new(2)));
        assert_eq!(table.page_of(768), None);
        assert_eq!(pool.n_free(), 1);
        assert!(table.grow_to(700).expect("idempotent"));
        assert_eq!(table.len(), 3);
    }

    /// A full pool refuses rather than handing a page out twice, and claims nothing
    /// on the way to refusing.
    #[test]
    fn a_short_pool_refuses_without_claiming_anything() {
        let pool = Arc::new(PagePool::new(4));
        let mut table = PageTable::new(Arc::clone(&pool), geometry());
        assert!(table.grow_to(4 * 256).expect("the pool has pages"));
        assert_eq!(table.len(), 4);
        assert_eq!(pool.n_free(), 0);
        assert!(
            !table
                .grow_to(4 * 256 + 1)
                .expect("a refusal, not an io error")
        );
        assert_eq!(table.len(), 4, "a refusal claims nothing");
        assert_eq!(pool.n_free(), 0);
    }

    /// A second holder keeps the page out of the free stack after the table lets
    /// go of it: the tree's share of a page a sequence wrote.
    #[test]
    fn another_holder_keeps_a_released_page_alive() {
        let pool = Arc::new(PagePool::new(4));
        let mut table = PageTable::new(Arc::clone(&pool), geometry());
        assert!(table.grow_to(4 * 256).expect("the pool has pages"));
        let last = PageId::new(3);
        pool.add_holder(last);
        table.truncate(3);
        assert_eq!(table.len(), 3);
        assert_eq!(pool.n_free(), 0, "the tree still holds the fourth page");
        pool.unref(last);
        assert_eq!(pool.n_free(), 1);
    }

    /// Dropping the table is the release; nothing else has to run.
    #[test]
    fn dropping_returns_every_page_it_held() {
        let pool = Arc::new(PagePool::new(6));
        {
            let mut table = PageTable::new(Arc::clone(&pool), geometry());
            assert!(table.grow_to(700).expect("the pool has pages"));
            assert_eq!(pool.n_free(), 3);
            assert_eq!(pool.leak_report().len(), 3);
        }
        assert_eq!(pool.n_free(), 6, "dropping is the release");
        assert!(pool.leak_report().is_empty());
    }
}
