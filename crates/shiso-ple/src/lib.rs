//! Streaming reader for an n-gram embedding table that cannot be resident.
//!
//! The table is hundreds of millions of rows of a hundred and sixty quantized
//! weights - tens of gibibytes, against a model that already wants most of the
//! machine. It is also the one tensor that is never streamed in bulk: a token
//! needs one row per hash head, chosen by a hash of its trailing n-gram, so the
//! access pattern is a scattered gather of ninety- or hundred-and-twenty-two
//! byte rows over the whole file and a decode step touches barely a kibibyte of
//! it.
//!
//! That is why this is its own reader rather than a second use of the routed
//! expert cache: that cache is keyed by layer and expert over multi-megabyte
//! slices, which is the wrong index and the wrong granularity here.
//!
//! # The shape of the thing
//!
//! [`codec`] is pure: row identifiers out of a token history, and one row out
//! of its quantized bytes. It allocates nothing, touches no file, and is the
//! part that has to be bit-exact, so it is separated from everything that could
//! make it hard to test.
//!
//! [`PleStream`] is the I/O tier. It owns the descriptor, a CLOCK row cache,
//! and one of two readers: a ring where the platform has `io_uring`, otherwise a
//! pool of threads doing blocking positional reads. Both reach a fast device's
//! random-read ceiling; the ring reaches it from a single thread at roughly a
//! quarter of the processor, which is what matters on a machine that also runs
//! work beside the model.
//!
//! # Why the gather is started early
//!
//! Every row a step needs is known before the step's first layer runs, and the
//! reads are cold. So a caller computes the identifiers up front, starts the
//! fetch, lets the layers ahead of the n-gram block enqueue, and waits only at
//! the block itself. [`PleStream::prefetch`] is the same idea one chunk further
//! out: it warms the cache for the chunk after this one, on a thread of its
//! own, and the fetch that follows hits.
//!
//! # The one environment knob
//!
//! `SHISO_PLE_IO` takes `uring` or `pool`. `pool` forces the thread pool;
//! `uring` insists on the ring and makes its absence an error rather than a
//! silent downgrade, which is the only way a test can tell "the ring ran" from
//! "something ran". Queue depth, worker count and cache size are constructor
//! parameters, because they are per-stream facts rather than per-process ones.

#![allow(
    unsafe_code,
    reason = "the ring arm registers bounce buffers with the kernel and pushes submission entries, which is FFI; every block carries its own SAFETY note"
)]

pub mod bulk;
pub mod codec;

mod cache;
mod handle;
mod pool;
mod prefetch;
mod ring;

use thiserror::Error;

pub use crate::handle::{
    PleBackend, PleConfig, PleIo, PleParams, PleParamsDesc, PleRowType, PleSegment, PleShards,
    PleStats, PleStream,
};

/// Everything this crate can refuse to do.
///
/// Structured rather than formatted: a caller that wants to know which row ran
/// past the end of the table reads a field instead of parsing a sentence. The
/// one `String` is the description of a metadata value that was wrong, which
/// has no fixed shape because the metadata does not.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PleError {
    /// The table metadata cannot describe a table.
    #[error("{what}: {value}")]
    InvalidParams {
        /// The parameter that could not be accepted.
        what: &'static str,
        /// What it said, as it was supplied.
        value: String,
    },
    /// A requested row identifier is past the end of the table.
    ///
    /// Always a bug in the hash parameters or in the row count the caller
    /// opened with, never a transient condition, so it is refused rather than
    /// clamped: a clamped row decodes to finite numbers and is silently wrong.
    #[error("row {row} is outside a table of {n_rows} rows")]
    RowOutOfRange {
        /// The offending identifier.
        row: u64,
        /// Rows the table holds.
        n_rows: u64,
    },
    /// A caller-supplied buffer is too small for what was asked of it.
    #[error("{what}: {want} needed, {got} supplied")]
    BufferTooSmall {
        /// The buffer that was short.
        what: &'static str,
        /// Elements the call needed.
        want: u64,
        /// Elements it was given.
        got: u64,
    },
    /// A read ended before the row it was asked for was covered.
    #[error("row {row} needed {want} bytes and the file gave {got}")]
    ShortRead {
        /// The row that could not be completed.
        row: u64,
        /// Bytes that landed.
        got: u64,
        /// Bytes the row needed.
        want: u64,
    },
    /// The ring exists on this target but could not be brought up.
    ///
    /// Distinct from [`PleError::BackendRefused`]: there is a kernel to ask and
    /// it answered, so the answer is worth keeping as a real `source` rather
    /// than collapsing into a sentence.
    #[error("io_uring: {op} failed")]
    RingUnavailable {
        /// The step of ring setup that failed.
        op: &'static str,
        /// What the kernel said.
        #[source]
        source: std::io::Error,
    },
    /// One fetch asked for more distinct rows than the cache has slots.
    ///
    /// Every miss in a batch claims a slot and holds it until the reads land,
    /// so a batch with more distinct rows than the cache holds cannot be
    /// served: there is no slot left to claim and no slot that may be evicted,
    /// because every one of them has a read in flight into it. This used to be
    /// an unbounded spin in the eviction hand - the one refusal in this crate
    /// that was written as an assumption ("a batch is a few thousand rows
    /// against millions of slots") rather than as a check.
    ///
    /// It is a sizing error at the call site, not a transient condition: open
    /// the stream with a cache of at least as many rows as one fetch will ask
    /// for, or fetch in smaller batches.
    #[error(
        "a fetch of {distinct} distinct rows needs at least that many cache slots, and the cache \
         has {slots}"
    )]
    CacheTooSmall {
        /// Distinct rows this batch had reached when it ran out of slots.
        distinct: u64,
        /// Slots the cache holds.
        slots: u64,
    },
    /// The I/O backend the caller insisted on is not available here.
    ///
    /// Loud on purpose. A silent downgrade to the other backend would make a
    /// test that means to exercise the ring pass without ever touching one.
    #[error("{0}")]
    BackendRefused(&'static str),
    /// A file operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl PleError {
    /// The operation name this failure travels under when it crosses into the
    /// shared error vocabulary.
    #[must_use]
    pub const fn op(&self) -> &'static str {
        match self {
            Self::InvalidParams { .. } => "ple_params",
            Self::RowOutOfRange { .. }
            | Self::BufferTooSmall { .. }
            | Self::CacheTooSmall { .. } => "ple_fetch",
            Self::ShortRead { .. } | Self::Io(_) => "ple_read",
            Self::BackendRefused(_) | Self::RingUnavailable { .. } => "ple_open",
        }
    }
}

impl From<PleError> for shiso_core::CoreError {
    /// Carries a streaming failure into the vocabulary every other crate speaks.
    ///
    /// The original stays reachable as `source()`, so `{:#}` chains it and a
    /// caller that cares which row failed can still downcast to it.
    fn from(err: PleError) -> Self {
        Self::Backend {
            backend: "ple",
            op: err.op(),
            source: Box::new(err),
        }
    }
}

/// Result of a streaming operation.
///
/// Named rather than shadowing the prelude's `Result`: a crate-root alias
/// called `Result` makes every signature in the crate ambiguous to read.
pub type PleResult<T> = Result<T, PleError>;
