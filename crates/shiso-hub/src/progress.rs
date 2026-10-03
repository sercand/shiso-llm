//! Progress reporting events for downloads and installations (§5.2, §6.1).

use serde::{Deserialize, Serialize};

/// Progress event emitted during recipe resolution, file download and model installation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum ProgressEvent {
    /// Model recipe and variant resolved; download plan formed.
    Resolved {
        /// Model name.
        model: String,
        /// Selected variant key.
        variant: String,
        /// Total missing bytes to download across all files.
        total: u64,
    },

    /// Download began for a specific component file.
    FileStarted {
        /// Model name.
        model: String,
        /// Variant key.
        variant: String,
        /// Role name (e.g. "base", "vision").
        role: String,
        /// File name.
        file: String,
        /// Total file size in bytes.
        total: u64,
    },

    /// Chunk progress update for an in-flight file download.
    FileProgress {
        /// Model name.
        model: String,
        /// Variant key.
        variant: String,
        /// Role name.
        role: String,
        /// File name.
        file: String,
        /// Total bytes transferred so far for this file.
        bytes: u64,
        /// Total expected size in bytes.
        total: u64,
    },

    /// File download finished and SHA-256 digest verified.
    FileVerified {
        /// Model name.
        model: String,
        /// Variant key.
        variant: String,
        /// Role name.
        role: String,
        /// File name.
        file: String,
        /// Verified size in bytes.
        total: u64,
    },

    /// All files present, staged, and atomically committed into place.
    Installed {
        /// Model name.
        model: String,
        /// Variant key.
        variant: String,
    },

    /// An error occurred during download or staging.
    Error {
        /// Model name.
        model: String,
        /// Variant key, if known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        variant: Option<String>,
        /// Human-readable error description.
        message: String,
    },
}
