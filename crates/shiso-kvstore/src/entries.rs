//! The disk tier as an index: what it holds, and how an entry is taken out of it.
//!
//! [`crate::facade`] has the five verbs a request goes through. These two are not
//! verbs of a request at all - they are what a caller uses to *decide* which entry to
//! ask for, and to say that a stored entry must never be offered again.
//!
//! # Why listing is separate from restoring
//!
//! [`KvCache::restore`] reads a whole payload: the tokens, the recurrent state and
//! every page. At full context that is on the order of a hundred mebibytes, and a
//! caller looking for "the longest stored prefix of this prompt" would have to pay it
//! once per candidate. [`KvCache::spill_entries`] reads only the headers, so the
//! search costs a directory walk and the restore happens exactly once, for the entry
//! that won.
//!
//! # Why forgetting lives here and not in the engine
//!
//! The store owns the files. What it does not own is the *policy* that decides a file
//! is worthless - which model, which quantization, which context size a build can act
//! on are all facts about a model, and this crate holds none of them. So the policy
//! stays above and reaches down through [`KvCache::forget_spill`], which is the only
//! way anything outside this crate deletes an entry.

use crate::error::KvError;
use crate::facade::KvCache;
use crate::ids::SpillId;
use crate::pool::InnerGuard;
use crate::spill::{SpillEntry, SpillStore};

impl KvCache {
    /// Every entry the disk tier holds, header only.
    ///
    /// Empty when the cache was opened without a spill directory, which is the same
    /// answer an empty directory gives: a caller searching for a prefix finds none
    /// either way, and neither is a failure.
    #[must_use]
    pub fn spill_entries(&self) -> Vec<SpillEntry> {
        let _held = InnerGuard::new();
        let inner = self.inner.lock();
        inner
            .spill
            .as_ref()
            .map_or_else(Vec::new, SpillStore::entries)
    }

    /// Unlinks one stored entry, reporting whether there was one to unlink.
    ///
    /// `Ok(false)` when nothing was stored under `id` - including when there is no
    /// spill directory at all - because the caller's aim was that the entry not be
    /// there, and it is not.
    ///
    /// This deletes a file. It is for an entry the caller has determined this build
    /// can never act on, not for one it merely cannot use right now: an entry that a
    /// differently configured run of the same build could still read is worth more
    /// where it is than the bytes it costs.
    ///
    /// # Errors
    ///
    /// [`KvError::Io`] when the entry exists and the directory refuses to remove it.
    pub fn forget_spill(&self, id: SpillId) -> Result<bool, KvError> {
        let _held = InnerGuard::new();
        let mut inner = self.inner.lock();
        match inner.spill.as_mut() {
            Some(store) => store.forget(id),
            None => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::backing::Backing;
    use crate::error::KvError;
    use crate::facade::KvCache;
    use crate::geom::{CacheGeometry, PathLayout};
    use crate::ids::{PageId, SpillId};
    use crate::spill::{Reason, SpillConfig};
    use shiso_core::TokenId;

    /// Positions per page, small enough that a short path still spans two of them.
    const PAGE_TOKENS: u32 = 4;
    /// Bytes of one page, which is what the fake mover moves.
    const PAGE_BYTES: usize = 64;
    /// Bytes of one checkpoint blob, which the layout below has to imply exactly.
    const CKPT_BYTES: usize = 4 * 4;

    /// A byte mover over a flat `Vec`, so a page copy is a `memcpy` and a restore can
    /// be compared byte for byte.
    #[derive(Debug)]
    struct VecBacking {
        pages: Vec<Vec<u8>>,
    }

    impl Backing for VecBacking {
        fn copy_page(&mut self, dst: PageId, src: PageId) -> Result<(), KvError> {
            let from = self.pages[src.get() as usize].clone();
            self.pages[dst.get() as usize] = from;
            Ok(())
        }

        fn page_to_host(&self, page: PageId, buf: &mut [u8]) -> Result<(), KvError> {
            buf.copy_from_slice(&self.pages[page.get() as usize]);
            Ok(())
        }

        fn page_from_host(&mut self, page: PageId, buf: &[u8]) -> Result<(), KvError> {
            self.pages[page.get() as usize].copy_from_slice(buf);
            Ok(())
        }
    }

    /// The geometry every test here opens with: one recurrent layer whose conv window
    /// and state matrix add up to [`CKPT_BYTES`].
    fn geometry() -> CacheGeometry {
        let page_bytes = u64::try_from(PAGE_BYTES).expect("the fixture page fits a u64");
        let ckpt_bytes = u64::try_from(CKPT_BYTES).expect("the fixture blob fits a u64");
        CacheGeometry::new(PAGE_TOKENS, page_bytes, ckpt_bytes, 0.004, 600.0).with_layout(
            PathLayout {
                model_id: 5,
                quant_bits: 4,
                ctx_size: 64,
                kv_width: 8,
                n_layers: 2,
                n_attn_layers: 1,
                n_gdn_layers: 1,
                vocab: 4,
                gdn_conv_elements: 2,
                gdn_state_elements: 2,
                ple_window_elements: 0,
                hc_width: 0,
                indexer: false,
                draft_head: false,
            },
        )
    }

    /// A cache with a spill directory, plus the directory so it outlives the test.
    fn cache(dir: &tempfile::TempDir) -> KvCache {
        let backing = VecBacking {
            pages: vec![vec![0u8; PAGE_BYTES]; 16],
        };
        KvCache::open(
            &geometry(),
            Box::new(backing),
            16,
            4,
            Some(SpillConfig::new(dir.path(), 64)),
        )
        .expect("the cache opens over a temporary spill directory")
    }

    /// Prefills one span, commits it with a checkpoint and a logits row, and spills
    /// it under `text`.
    fn store(cache: &KvCache, text: &[u8], tokens: &[TokenId]) -> SpillId {
        let mut frontier = cache.cold_frontier();
        let n = u32::try_from(tokens.len()).expect("a fixture path is short");
        assert!(frontier.grow_to(n).expect("the pool has pages"));
        let state = vec![7u8; CKPT_BYTES];
        let logits = vec![9u8; 4 * 4];
        cache
            .commit(
                &mut frontier,
                tokens,
                n,
                crate::facade::CommitOpts::span(&[])
                    .checkpoint(&state, true)
                    .logits_row(&logits),
                1.0,
            )
            .expect("the span commits with a checkpoint");
        match cache.spill(frontier.node(), text, Reason::Cold) {
            Ok(crate::facade::SpillOutcome::Written { id, .. }) => id,
            other => panic!("the path did not spill: {other:?}"),
        }
    }

    fn path(n: u32) -> Vec<TokenId> {
        (0..n).map(TokenId::new).collect()
    }

    #[test]
    fn a_forgotten_entry_is_gone_and_forgetting_it_again_says_there_was_nothing_there() {
        let dir = tempfile::TempDir::new().expect("a temporary directory");
        let cache = cache(&dir);
        let id = store(&cache, b"the capital of France is", &path(6));

        assert_eq!(
            cache.spill_entries().len(),
            1,
            "the entry is in the directory index before it is forgotten"
        );
        assert!(
            cache
                .forget_spill(id)
                .expect("the directory allows the removal"),
            "forgetting an entry that is there reports that it was"
        );
        assert!(
            cache
                .restore(id, 2.0)
                .expect("a restore of a missing entry is not an error")
                .is_none(),
            "a forgotten entry cannot be restored"
        );
        assert!(
            cache.spill_entries().is_empty(),
            "a forgotten entry leaves the directory index"
        );
        assert!(
            !cache
                .forget_spill(id)
                .expect("a second removal is not an error"),
            "forgetting an entry that was already gone reports that there was nothing"
        );
    }

    #[test]
    fn the_directory_index_reports_each_entrys_own_header_without_reading_its_pages() {
        let dir = tempfile::TempDir::new().expect("a temporary directory");
        let cache = cache(&dir);
        let short = b"the capital".as_slice();
        let long = b"the capital of France is".as_slice();
        store(&cache, short, &path(4));
        store(&cache, long, &path(6));

        let mut entries = cache.spill_entries();
        entries.sort_by_key(|entry| entry.text_bytes);
        assert_eq!(entries.len(), 2, "both prefixes are indexed");
        assert_eq!(entries[0].text_bytes, short.len() as u64);
        assert_eq!(entries[0].tokens, 4);
        assert_eq!(entries[1].text_bytes, long.len() as u64);
        assert_eq!(entries[1].tokens, 6);
        for entry in &entries {
            assert_eq!(
                entry.model_id, 5,
                "the layout's model tag reaches the index"
            );
            assert_eq!(entry.quant_bits, 4);
            assert_eq!(entry.ctx_size, 64);
            assert_eq!(entry.reason, Reason::Cold);
        }
    }

    #[test]
    fn a_cache_without_a_spill_directory_has_nothing_to_list_and_nothing_to_forget() {
        let backing = VecBacking {
            pages: vec![vec![0u8; PAGE_BYTES]; 4],
        };
        let cache = KvCache::open(&geometry(), Box::new(backing), 4, 2, None)
            .expect("an in-memory cache opens");
        assert!(cache.spill_entries().is_empty());
        assert!(
            !cache
                .forget_spill(SpillId::of(b"never stored"))
                .expect("forgetting without a directory is not an error"),
            "with no directory there is nothing to forget, which is not a failure"
        );
        drop(Arc::new(()));
    }
}
