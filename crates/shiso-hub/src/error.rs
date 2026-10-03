//! Typed errors for `shiso-hub`.

use std::path::PathBuf;
use thiserror::Error;

/// Everything that can fail during recipe resolution, variant selection,
/// download, verification, locking or blob storage.
#[derive(Debug, Error)]
pub enum HubError {
    /// An I/O error occurred on a specific filesystem path.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// The path where the I/O error occurred.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A network request failed.
    #[error("network error for {url}: {source}")]
    Network {
        /// The URL of the failed network request.
        url: String,
        /// The underlying HTTP/network error.
        #[source]
        source: reqwest::Error,
    },

    /// An HTTP response had an unsuccessful status code.
    #[error("HTTP error {status} for {url}")]
    HttpStatus {
        /// The URL that returned the status.
        url: String,
        /// The HTTP status code.
        status: reqwest::StatusCode,
    },

    /// File contents did not match the expected SHA-256 digest.
    #[error("digest mismatch for {file}: expected {expected}, got {actual}")]
    DigestMismatch {
        /// The file or URL that had a digest mismatch.
        file: String,
        /// The expected hexadecimal SHA-256 digest.
        expected: String,
        /// The computed actual hexadecimal SHA-256 digest.
        actual: String,
    },

    /// File size did not match the manifest.
    #[error("size mismatch for {file}: expected {expected} bytes, got {actual} bytes")]
    SizeMismatch {
        /// The file that had a size mismatch.
        file: String,
        /// The expected size in bytes.
        expected: u64,
        /// The actual size in bytes.
        actual: u64,
    },

    /// Free disk space on the home filesystem is insufficient for the download.
    #[error("insufficient disk space: need {needed_bytes} bytes, have {free_bytes} bytes")]
    InsufficientDiskSpace {
        /// The bytes needed for missing blobs.
        needed_bytes: u64,
        /// The free bytes available on the filesystem.
        free_bytes: u64,
    },

    /// A recipe or manifest failed structural validation.
    #[error("validation error: {reason}")]
    Validation {
        /// Detailed description of the validation failure.
        reason: String,
    },

    /// The requested model name or alias was not found in the index.
    #[error("model '{name}' not found in registry")]
    ModelNotFound {
        /// The model name or alias searched for.
        name: String,
    },

    /// The specified variant key was not found in the recipe.
    #[error("variant '{variant}' not found for model '{model}'")]
    VariantNotFound {
        /// The model name.
        model: String,
        /// The missing variant key.
        variant: String,
    },

    /// No variant matches the host's profile.
    #[error("no compatible variant found for model '{model}':\n{reasons}")]
    NoCompatibleVariant {
        /// The model name.
        model: String,
        /// Rejection reason for each variant in the recipe.
        reasons: String,
    },

    /// The recipe requires a newer schema version than this client supports.
    #[error("unsupported schema version {found}, maximum supported is {supported}")]
    UnsupportedSchemaVersion {
        /// Schema version found in manifest.
        found: u32,
        /// Maximum schema version known to this build.
        supported: u32,
    },

    /// The recipe requires a newer shiso version than this client.
    #[error(
        "this recipe needs a newer shiso: requires shiso >= {min_version}, running {current_version}"
    )]
    VersionMismatch {
        /// Minimum version specified in recipe.
        min_version: String,
        /// Currently running shiso version.
        current_version: String,
    },

    /// The model is currently locked by a running server.
    #[error("model '{name}:{variant}' is in use by a running server")]
    ModelInUse {
        /// The model name.
        name: String,
        /// The variant key.
        variant: String,
    },

    /// Serialization or deserialization failed.
    #[error("JSON error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Hugging Face authentication failed (401/403).
    #[error("authentication failed for repository '{repo}' (token was sent: {token_sent})")]
    Auth {
        /// The repository that rejected the request.
        repo: String,
        /// Whether an Authorization header was included.
        token_sent: bool,
    },

    /// The auth.json file has invalid permissions or format.
    #[error("auth token file error at {path}: {reason}")]
    AuthFile {
        /// Path to the auth file.
        path: PathBuf,
        /// The reason the file was rejected.
        reason: String,
    },

    /// A generic or unspecified error.
    #[error("{0}")]
    Other(String),
}
