//! Typed error vocabulary shared by every crate in the workspace.
//!
//! Failures are structured, never formatted strings: the backend name, the
//! failing operation and the driver error stay separate fields so a caller can
//! branch on the cause. `Backend` wraps the driver error as a real `source`, so
//! `{:#}` chains it.

use std::error::Error as StdError;
use std::fmt;
use std::io;

use thiserror::Error;

use crate::ids::SessionId;

/// Every failure surfaced by this crate and by the crates layered on it.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CoreError {
    /// The config names an architecture this build cannot run.
    #[error("unknown model architecture {model_type:?}")]
    UnknownArch {
        /// The `model_type` string as it appeared in the config.
        model_type: String,
    },
    /// A config field disagrees with the fixed shape table for its
    /// architecture. Field-level, so a wrong constant names itself.
    #[error("{field}: expected {expected}, got {got}")]
    ShapeMismatch {
        /// Dotted field path naming the value that disagreed.
        field: &'static str,
        /// Value the port requires.
        expected: u64,
        /// Value found in the input.
        got: u64,
    },
    /// A required config key is absent.
    #[error("config is missing required key {key:?}")]
    ConfigMissing {
        /// Dotted path of the missing key.
        key: String,
    },
    /// A code path the caller asked for that this build or device cannot serve.
    #[error("{what} is not supported by the {backend} backend")]
    Unsupported {
        /// The capability that is missing.
        what: &'static str,
        /// Name of the backend that refused, or `"core"` when no device is
        /// involved.
        backend: &'static str,
    },
    /// A value whose spelling this build does not implement.
    ///
    /// Distinct from [`CoreError::Unsupported`]: the device is not the problem,
    /// the input is, and the offending spelling is the only thing a caller can
    /// act on. It stays a field rather than being interpolated into a `String`
    /// message, so the variant remains structured.
    #[error("{key} has unsupported value {value:?}")]
    InvalidValue {
        /// Config key or value vocabulary that was being parsed.
        key: &'static str,
        /// The value as it appeared in the input.
        value: String,
    },
    /// A sub-range ran past the end of an allocation. Never silently clamped.
    #[error("range {offset}..{} exceeds buffer of {buffer_bytes} bytes", offset + len)]
    OutOfBounds {
        /// Requested start offset in bytes.
        offset: u64,
        /// Requested length in bytes.
        len: u64,
        /// Size of the buffer the range was taken from.
        buffer_bytes: u64,
    },
    /// The session handle is not one the model handed out.
    #[error("no such session: {0:?}")]
    UnknownSession(SessionId),
    /// A session asked for one token more than its configured context.
    #[error("session {session:?} reached its context capacity of {cap} tokens")]
    ContextOverflow {
        /// The overflowing session.
        session: SessionId,
        /// The capacity that was hit.
        cap: u32,
    },
    /// A device-level failure. The driver or runtime error is a real `source`
    /// so `{:#}` and error-chain walkers reach it.
    #[error("{backend} backend failed during {op}")]
    Backend {
        /// Name of the backend that reported the failure.
        backend: &'static str,
        /// The [`crate::Backend`] method or kernel that failed.
        op: &'static str,
        /// The underlying driver error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Host I/O failure while reading a config, weights or cache file.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// JSON failure while parsing a config.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl CoreError {
    /// This error and every cause behind it, as one line.
    ///
    /// `Display` on a variant prints that variant and stops, which is right for
    /// a message a user reads and wrong for a log line: the field a `Backend`
    /// variant names is the kernel, and the sentence that says *why* the kernel
    /// refused is in the driver error underneath. Formatting through this
    /// walks [`std::error::Error::source`] to the bottom, so a log site that
    /// writes `%error.chain()` records the whole story rather than its title.
    #[must_use]
    pub fn chain(&self) -> Chain<'_> {
        Chain { head: self }
    }
}

/// An error and its causes, rendered `outer: inner: innermost`.
///
/// Built by [`CoreError::chain`] and by [`chain_of`] for the foreign error types
/// this workspace also logs. Borrowed rather than owned so that a log site that
/// never formats - a filtered-out level - allocates nothing.
#[derive(Clone, Copy)]
pub struct Chain<'e> {
    /// The error the walk starts at.
    head: &'e (dyn StdError + 'static),
}

impl fmt::Display for Chain<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.head)?;
        let mut cause = self.head.source();
        while let Some(err) = cause {
            write!(f, ": {err}")?;
            cause = err.source();
        }
        Ok(())
    }
}

impl fmt::Debug for Chain<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

/// The same walk for an error that is not a [`CoreError`].
///
/// Every crate that logs a failure it did not define - a loader error, a driver
/// error - reaches for this, so that "the cause is in the source" is one rule
/// with one spelling rather than a habit each crate keeps or forgets.
#[must_use]
pub const fn chain_of<'e>(error: &'e (dyn StdError + 'static)) -> Chain<'e> {
    Chain { head: error }
}

/// Result alias for every fallible operation in the workspace.
///
/// Crates define their own aliases when they need one, but they always resolve
/// to [`CoreError`]; there is no crate-wide `Result<T>` at the root.
pub type CoreResult<T> = Result<T, CoreError>;
