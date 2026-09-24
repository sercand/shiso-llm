//! The KV cache: a radix tree of token spans over refcounted paged KV, plus the
//! recurrent-state checkpoint store, utility eviction and the disk tier.
//!
//! Three things are true at once here, and most of the design rules fall out of
//! keeping them separate.
//!
//! - **Pages are immutable past the frontier.** A page is written once, by the
//!   sequence that filled it, and past the frontier it is never rewritten. That is what
//!   lets one page be indexed by every path that shares the span, and it is why
//!   resuming *inside* a page is a copy rather than a write.
//! - Because pages are immutable, a page's holders are tree nodes and live page tables,
//!   and nothing has to invalidate a page it thought it owned.
//! - A checkpoint is worth the prefill between it and the nearest lower checkpoint on its
//!   path, which is why one 300 tokens above its neighbour is nearly worthless and a lone
//!   one 20k deep is worth tens of seconds. Ranking either entry by recency instead --
//!   plain LRU -- throws away exactly the large recurrent-state entries that are worth
//!   the most, precisely because they are large.
//!
//! # Start at [`KvCache`]
//!
//! The verbs are `plan`, `lookup`, `commit`, `evict_to` and `persist`/`resume`;
//! [`pool`] explains the two-lock design they all depend on and [`facade`] the order in
//! which they take those locks.
//!
//! # Two locks, one order
//!
//! [`KvCache`]'s inner mutex guards the span tree, the checkpoint slots and the byte
//! mover. [`pool::PagePool`] keeps its free stack and refcounts behind a mutex of its own. The
//! order is always inner then pool, never the reverse, and it exists so that
//! [`PageTable`]'s and [`Frontier`]'s `Drop` can run inside a cache critical section:
//! see the module docs of [`pool`].
//!
//! # Nothing about a model is in here
//!
//! Sizes arrive in [`CacheGeometry`] and bytes move through [`Backing`], so the same tree
//! serves a recurrent 24-layer model and a dense one, and the crate compiles without a
//! backend dependency. The store never touches device memory: a page is an index the
//! engine translates, and a checkpoint is an opaque blob the engine snapshots.
#![forbid(unsafe_code)]

pub mod backing;

mod ckpt;
mod error;
mod evict;
mod facade;
mod geom;
mod ids;
mod page_table;
mod pool;
mod store;
mod tree;

pub use backing::Backing;
pub use ckpt::CkptRef;
pub use error::KvError;
pub use evict::next_admission_pos;
pub use facade::{
    CacheStats, CommitOpts, EvictReport, Frontier, KvCache, Lookup, Resume, ReuseSource,
};
pub use geom::{CacheGeometry, PathLayout};
pub use ids::{NodeId, PageId, SlotId};
pub use page_table::PageTable;
pub use store::{
    DEFAULT_MAX_JOBS, DEFAULT_MIN_TOKENS, DEFAULT_STAGING_MB, DiskConfig, DiskStats, EntryKey,
    PersistOutcome, PersistRefusal, Prefetcher, PumpReport, Reason,
};
