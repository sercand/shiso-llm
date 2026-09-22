//! Fixtures shared by the cache's integration tests.
//!
//! The [`FakeBacking`] here is the whole reason these tests can say anything about
//! copying: it is the only place a scenario can see that a resume moved exactly one
//! page, rather than trusting the cache's bookkeeping to agree with itself.

#![allow(
    dead_code,
    unreachable_pub,
    reason = "every test binary compiles its own copy of these fixtures and uses a subset \
              of them"
)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use shiso_core::TokenId;
use shiso_kvstore::{
    Backing, CacheGeometry, CommitOpts, Frontier, KvCache, KvError, PageId, Resume,
};

/// Positions in one page, for every scenario in this directory.
pub const PAGE_TOKENS: u32 = 256;
/// Bytes in one page. Small on purpose: a scenario about page counts should not need a
/// mebibyte of host memory to make its point.
pub const PAGE_BYTES: u64 = 4_096;
/// Bytes in one checkpoint's opaque state blob.
pub const CKPT_BYTES: u64 = 1_024;

/// The geometry every scenario builds its cache from.
#[must_use]
pub fn geometry() -> CacheGeometry {
    CacheGeometry::new(PAGE_TOKENS, PAGE_BYTES, CKPT_BYTES, 0.001, 600.0)
}

/// A prompt of `n` tokens anchored at `start`, so two scenarios' prompts differ
/// wherever they are meant to and match exactly where they are meant to.
#[must_use]
pub fn seq(start: u32, n: u32) -> Vec<TokenId> {
    (0..n).map(|i| TokenId::from(start + i)).collect()
}

/// A state blob exactly as large as the geometry says a checkpoint's is.
#[must_use]
pub fn blob() -> Vec<u8> {
    (0..u32::try_from(CKPT_BYTES).expect("fits"))
        .map(|i| u8::try_from(i % 251).expect("remainder fits"))
        .collect()
}

/// A logits row. The store keeps the bytes it is handed and nothing else.
#[must_use]
pub fn logits_row() -> Vec<u8> {
    vec![0xAB; 64]
}

/// A cache over [`FakeBacking`], plus the handle a scenario reads while the cache owns
/// the mover.
#[must_use]
pub fn open(pool_pages: u32, ckpt_slots: u32) -> (KvCache, Pages) {
    let (backing, pages) = FakeBacking::new();
    let cache = KvCache::open(&geometry(), Box::new(backing), pool_pages, ckpt_slots, None)
        .expect("a geometry this file builds is a valid one");
    (cache, pages)
}

/// Hands the tree the span a frontier has just produced, with a checkpoint at `q`.
///
/// The blob and the logits row are built here so a scenario cannot accidentally commit
/// a state of the wrong size, which the cache rejects as a caller bug.
pub fn commit_span(cache: &KvCache, frontier: &mut Frontier, tokens: &[TokenId], q: u32, now: f64) {
    let state = blob();
    let row = logits_row();
    let mut opts = CommitOpts::default();
    opts.state = Some(&state);
    opts.logits = Some(&row);
    opts.with_logits = true;
    cache
        .commit(frontier, tokens, q, opts, now)
        .expect("a commit of a well-sized blob is not an error");
}

/// Moves a conversation onto the deepest checkpoint its prompt can reuse, claims pages
/// up to `q`, and commits the span above the branch point.
///
/// This is the order a real turn happens in, and the only order in which two
/// conversations end up sharing the pages below their branch: the sharing comes from the
/// resume, not from the commit.
#[must_use]
pub fn continue_prompt(cache: &KvCache, prompt: &[TokenId], q: u32, now: f64) -> Frontier {
    let mut frontier = resume(cache, prompt, now)
        .expect("the prompt was built to reuse something")
        .frontier;
    let grew = frontier
        .grow_to(q)
        .expect("a pool sized for the prompt answers, not errors");
    assert!(grew, "the pool covered every position of the prompt");
    commit_span(cache, &mut frontier, prompt, q, now);
    frontier
}

/// Spans in the tree, not counting the root the tree is built with.
///
/// [`CacheStats::nodes`](shiso_kvstore::CacheStats::nodes) counts the root, and a
/// scenario that says "one turn wrote one span" means one span *above* it.
#[must_use]
pub fn spans(cache: &KvCache) -> u32 {
    cache.stats().nodes - 1
}

/// `lookup`, with the two layers of `Option` and `Result` peeled off by a scenario that
/// knows which one it expects.
#[must_use]
pub fn resume(cache: &KvCache, prompt: &[TokenId], now: f64) -> Option<Resume> {
    cache
        .lookup(prompt, now)
        .expect("a resume over a pool sized for the prompt does not fail")
}

/// Prefills `tokens` as one span with a checkpoint at `q` and hands back the frontier
/// that wrote it.
///
/// A scenario that wants the tree to stand on its own drops the frontier; one that is
/// playing the part of a live conversation keeps it.
#[must_use]
pub fn prefill(cache: &KvCache, tokens: &[TokenId], q: u32, now: f64) -> Frontier {
    let mut frontier = cache.cold_frontier();
    let grew = frontier
        .grow_to(q)
        .expect("a pool sized for the prompt answers, not errors");
    assert!(grew, "the pool covered every position of the prompt");
    commit_span(cache, &mut frontier, tokens, q, now);
    frontier
}

/// The read side of [`FakeBacking`], cheap enough for a scenario to keep one.
#[derive(Debug, Clone, Default)]
pub struct Pages(Arc<Mutex<Store>>);

impl Pages {
    /// Whole-page copies the mover has been asked for, which is what the
    /// copy-on-write scenarios count.
    #[must_use]
    pub fn copies(&self) -> usize {
        self.lock().copies
    }

    /// The bytes the backing holds for one page, or `None` if it holds nothing.
    #[must_use]
    pub fn bytes_of(&self, page: PageId) -> Option<Vec<u8>> {
        self.lock().pages.get(&u32::from(page)).cloned()
    }

    /// Puts a page's bytes in the backing the way the engine would have after writing
    /// the rows that produced them.
    pub fn seed(&self, page: PageId, bytes: &[u8]) {
        self.lock().pages.insert(u32::from(page), bytes.to_vec());
    }

    /// Pages the backing holds anything for.
    #[must_use]
    pub fn stored(&self) -> usize {
        self.lock().pages.len()
    }

    fn lock(&self) -> MutexGuard<'_, Store> {
        self.0.lock().expect("the fixture store is never poisoned")
    }
}

#[derive(Debug, Default)]
struct Store {
    pages: HashMap<u32, Vec<u8>>,
    copies: usize,
}

/// A `Backing` over host bytes that counts the copies it is asked for.
#[derive(Debug)]
pub struct FakeBacking {
    store: Arc<Mutex<Store>>,
}

impl FakeBacking {
    /// The mover, and the handle that reads what it did.
    #[must_use]
    pub fn new() -> (Self, Pages) {
        let store = Arc::new(Mutex::new(Store::default()));
        (
            Self {
                store: Arc::clone(&store),
            },
            Pages(store),
        )
    }

    fn lock(&self) -> MutexGuard<'_, Store> {
        self.store
            .lock()
            .expect("the fixture store is never poisoned")
    }
}

impl Backing for FakeBacking {
    fn copy_page(&mut self, dst: PageId, src: PageId) -> Result<(), KvError> {
        let mut store = self.lock();
        let bytes = store
            .pages
            .get(&u32::from(src))
            .cloned()
            .unwrap_or_else(|| vec![0; usize::try_from(PAGE_BYTES).expect("fits")]);
        store.copies += 1;
        store.pages.insert(u32::from(dst), bytes);
        Ok(())
    }

    fn page_to_host(&self, page: PageId, buf: &mut [u8]) -> Result<(), KvError> {
        let store = self.lock();
        let held = store.pages.get(&u32::from(page));
        let want = usize::try_from(PAGE_BYTES).expect("fits");
        let Some(bytes) = held else {
            return Err(KvError::BufferLen { want, got: 0 });
        };
        if bytes.len() != buf.len() {
            return Err(KvError::BufferLen {
                want: buf.len(),
                got: bytes.len(),
            });
        }
        buf.copy_from_slice(bytes);
        Ok(())
    }

    fn page_from_host(&mut self, page: PageId, buf: &[u8]) -> Result<(), KvError> {
        self.lock().pages.insert(u32::from(page), buf.to_vec());
        Ok(())
    }
}
