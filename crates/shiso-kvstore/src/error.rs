//! Every way an operation on the cache can be refused.
//!
//! Each refusal names itself rather than arriving as a `String` the caller has to
//! match on, and the ones that gate a retry carry the numbers the caller needs:
//! `OverBudget` carries both sides of the comparison so the disk tier can decide
//! whether to evict and retry or give up, and `StaleCheckpoint` carries the slot so
//! the engine can drop the reference it was holding instead of guessing which one
//! went away.
//!
//! Refusals are split by who could have prevented them. `NodeRefused`, `PathGap` and
//! `BufferLen` describe a caller bug — a span the page table does not cover, a
//! staging buffer of the wrong length — and are never a runtime condition to retry.
//! `PoolEmpty` and `OverBudget` are conditions the caller can act on by evicting and
//! trying again.

/// Anything that can go wrong in the cache.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KvError {
    /// A disk-tier file could not be read, written or renamed.
    #[error("io error for {path}: {source}")]
    Io {
        /// The file the operation was on.
        path: std::path::PathBuf,
        /// The OS error behind it.
        source: std::io::Error,
    },

    /// The page pool has no free page and the caller cannot evict.
    #[error("page pool is empty")]
    PoolEmpty,

    /// A token path walked off the end of a span before the span ended.
    #[error("span tree has a gap at position {position}")]
    PathGap {
        /// The first position no node on the path owns a page for.
        position: u32,
    },

    /// A node could not be added to the tree.
    #[error("node refused: {reason}")]
    NodeRefused {
        /// Which rule the node broke.
        reason: &'static str,
    },

    /// A checkpoint could not be admitted.
    #[error("checkpoint refused: {reason}")]
    SlotRefused {
        /// Which rule refused it.
        reason: &'static str,
    },

    /// A disk-tier file's header is not one this build understands.
    #[error("bad disk-tier header: {reason}")]
    BadHeader {
        /// Which field was wrong.
        reason: &'static str,
    },

    /// A write would put the disk tier over its byte budget.
    #[error("a disk-tier write needs {required} bytes, budget is {budget}")]
    OverBudget {
        /// Bytes the write would need.
        required: u64,
        /// Bytes the directory may hold.
        budget: u64,
    },

    /// A checkpoint the caller still holds a reference to has been evicted.
    #[error("checkpoint in slot {slot} is gone")]
    StaleCheckpoint {
        /// The slot the reference pointed at.
        slot: crate::ids::SlotId,
    },

    /// A caller-owned staging buffer is not the length the geometry requires.
    #[error("buffer length {got}, geometry wants {want}")]
    BufferLen {
        /// Bytes the geometry requires.
        want: usize,
        /// Bytes the caller passed.
        got: usize,
    },

    /// A `CacheGeometry` the cache cannot be built from.
    #[error("invalid cache geometry: {field} {reason}")]
    BadGeometry {
        /// Which field failed.
        field: &'static str,
        /// What was wrong with it.
        reason: &'static str,
    },
}

impl KvError {
    pub(crate) const fn bad_geometry(field: &'static str, reason: &'static str) -> Self {
        Self::BadGeometry { field, reason }
    }

    pub(crate) fn io(path: impl Into<std::path::PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
