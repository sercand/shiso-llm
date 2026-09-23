//! What a sparse stack tells the memory ledger beyond its own allocations.
//!
//! Every device buffer the stack makes is tagged where it is allocated; see
//! [`shiso_core::ledger`]. What is left are the consumers whose size moves
//! under the cache's own lock - pages sessions hold, checkpoint blobs in host
//! memory, the disk tier - and the n-gram reader's host row cache. Each gets a
//! probe the ledger calls when it is read, so none of them costs a step
//! anything.
//!
//! The probes hold the cache weakly. A checkpoint tier rebuilds the cache, and
//! a probe that kept the old one alive would report a cache nothing uses while
//! holding its bookkeeping in memory.

use std::sync::{Arc, Weak};

use shiso_core::backend::Backend;
use shiso_core::ledger::{MemTag, Usage};
use shiso_kvstore::KvCache;

use crate::sparse::ple::NgramTable;

/// Registers the cache's probes: pages in use, checkpoint slots, and the disk
/// tier when the cache has one. Replaces whatever an earlier cache registered.
///
/// `page_bytes` is what one pool page costs on the device, every plane
/// included, and `ckpt_bytes` one checkpoint blob.
pub(crate) fn watch_cache(
    backend: &Arc<dyn Backend>,
    cache: &Arc<KvCache>,
    page_bytes: u64,
    ckpt_bytes: u64,
) {
    let Some(ledger) = backend.ledger() else {
        return;
    };
    let weak = Arc::downgrade(cache);
    ledger.probe(
        MemTag::KvInUse,
        stats_probe(weak.clone(), move |stats| Usage {
            bytes: u64::from(stats.pages_used).saturating_mul(page_bytes),
            limit: Some(u64::from(stats.pool_pages).saturating_mul(page_bytes)),
        }),
    );
    ledger.probe(
        MemTag::CheckpointSlots,
        stats_probe(weak.clone(), move |stats| Usage {
            bytes: u64::from(stats.ckpt_used).saturating_mul(ckpt_bytes),
            limit: Some(u64::from(stats.ckpt_capacity).saturating_mul(ckpt_bytes)),
        }),
    );
    if cache.stats().spill.is_some() {
        ledger.probe(
            MemTag::DiskTier,
            stats_probe(weak, |stats| {
                stats.spill.map_or_else(Usage::default, |spill| Usage {
                    bytes: spill.total_bytes,
                    limit: Some(spill.budget_bytes),
                })
            }),
        );
    }
}

/// Registers the n-gram reader's host row cache, whose size is fixed once
/// the reader is open. `limit` is the budget the reader was given.
pub(crate) fn watch_ngram(backend: &Arc<dyn Backend>, table: &NgramTable, limit: u64) {
    let Some(ledger) = backend.ledger() else {
        return;
    };
    let bytes = table.cache_bytes();
    ledger.probe(MemTag::NgramRowCache, move || Usage {
        bytes,
        limit: Some(limit),
    });
}

/// A probe that reads one figure out of the cache's statistics, and reports
/// nothing once the cache is gone.
fn stats_probe(
    cache: Weak<KvCache>,
    read: impl Fn(shiso_kvstore::CacheStats) -> Usage + Send + Sync,
) -> impl Fn() -> Usage + Send + Sync {
    move || {
        cache
            .upgrade()
            .map_or_else(Usage::default, |cache| read(cache.stats()))
    }
}
