//! Hub caller configuration and authentication types (§5.1).

use std::fmt;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use base64::Engine;
use ed25519_dalek::VerifyingKey;
use url::Url;

use crate::error::HubError;

/// Default static registry URL.
pub const DEFAULT_REGISTRY_URL: &str = "https://models.shiso.run/v1/";

/// Default Hugging Face endpoint URL.
pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";

/// Default maximum parallel file downloads.
pub const DEFAULT_MAX_PARALLEL_FILES: usize = 4;

/// Compiled-in ed25519 public key bytes for `models.shiso.run`.
pub const COMPILED_REGISTRY_PUBKEY: &[u8; 32] = &[
    0x3f, 0xe9, 0x64, 0x12, 0xe1, 0xe7, 0xf2, 0x33, 0xf8, 0xc9, 0x24, 0x42, 0xb6, 0x18, 0x1d, 0x04,
    0x3c, 0x99, 0x27, 0x8c, 0xc3, 0x4a, 0x26, 0x18, 0x1d, 0xad, 0x40, 0x28, 0xfb, 0xa2, 0x47, 0xd8,
];

/// Secret Hugging Face bearer token.
///
/// Implements `Debug` by redacting the secret: `HfToken(***)`.
#[derive(Clone, PartialEq, Eq)]
pub struct HfToken(String);

impl HfToken {
    /// Wraps a token string.
    #[must_use]
    pub fn new(token: impl Into<String>) -> Self {
        Self(token.into())
    }

    /// Returns the raw token string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for HfToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HfToken(***)")
    }
}

impl fmt::Display for HfToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HfToken(***)")
    }
}

/// Registry static endpoint URL and its Ed25519 verification key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryEndpoint {
    /// Base URL of the registry (e.g. `https://models.shiso.run/v1`).
    pub url: Url,
    /// Ed25519 verifying key used to verify `index.json.sig`.
    pub verifying_key: VerifyingKey,
}

impl RegistryEndpoint {
    /// Constructs the default production registry endpoint.
    ///
    /// # Panics
    /// Panics only if the compiled-in constants are invalid (checked by unit tests).
    #[must_use]
    pub fn default_production() -> Self {
        let url = Url::parse(DEFAULT_REGISTRY_URL).expect("valid default registry URL");
        let verifying_key =
            VerifyingKey::from_bytes(COMPILED_REGISTRY_PUBKEY).expect("valid compiled-in key");
        Self { url, verifying_key }
    }

    /// Constructs an endpoint from a URL and a key string (hex, base64, or path to key file).
    ///
    /// # Errors
    /// Returns [`HubError::Validation`] if the URL or key cannot be parsed.
    pub fn parse(url_str: &str, key_str: &str) -> Result<Self, HubError> {
        let mut url = Url::parse(url_str).map_err(|e| HubError::Validation {
            reason: format!("invalid registry URL '{url_str}': {e}"),
        })?;
        if !url.path().ends_with('/') {
            let mut path = url.path().to_string();
            path.push('/');
            url.set_path(&path);
        }
        let verifying_key = parse_verifying_key(key_str)?;
        Ok(Self { url, verifying_key })
    }
}

/// Parses an ed25519 verifying key from hex, base64, or a file path.
///
/// # Errors
/// Returns [`HubError::Validation`] or [`HubError::Io`] if parsing fails.
pub fn parse_verifying_key(key_input: &str) -> Result<VerifyingKey, HubError> {
    let trimmed = key_input.trim();

    // Check if input is an existing file path
    let path = Path::new(trimmed);
    if path.is_file() {
        let content = std::fs::read(path).map_err(|e| HubError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        if content.len() == 32 {
            let bytes: [u8; 32] = content.try_into().map_err(|_| HubError::Validation {
                reason: "key must be 32 bytes".to_string(),
            })?;
            return VerifyingKey::from_bytes(&bytes).map_err(|e| HubError::Validation {
                reason: format!("invalid ed25519 key bytes in {}: {e}", path.display()),
            });
        }
        let text = String::from_utf8_lossy(&content);
        return parse_verifying_key_text(text.trim());
    }

    parse_verifying_key_text(trimmed)
}

fn parse_verifying_key_text(text: &str) -> Result<VerifyingKey, HubError> {
    // 64-hex
    if text.len() == 64 && text.bytes().all(|b| b.is_ascii_hexdigit()) {
        let bytes = hex::decode(text).map_err(|e| HubError::Validation {
            reason: format!("invalid hex in ed25519 key: {e}"),
        })?;
        let array: [u8; 32] = bytes.try_into().map_err(|_| HubError::Validation {
            reason: "key must be 32 bytes".to_string(),
        })?;
        return VerifyingKey::from_bytes(&array).map_err(|e| HubError::Validation {
            reason: format!("invalid ed25519 key: {e}"),
        });
    }

    // Base64
    let clean: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if let Ok(bytes) = base64::prelude::BASE64_STANDARD.decode(&clean)
        && bytes.len() == 32
    {
        let array: [u8; 32] = bytes.try_into().map_err(|_| HubError::Validation {
            reason: "key must be 32 bytes".to_string(),
        })?;
        return VerifyingKey::from_bytes(&array).map_err(|e| HubError::Validation {
            reason: format!("invalid ed25519 key: {e}"),
        });
    }

    Err(HubError::Validation {
        reason: "registry key must be 32 bytes encoded as 64-character hex or base64, or a path to a key file".to_string(),
    })
}

/// Everything the hub needs from its caller; nothing is read from the environment (§5.1).
#[derive(Debug, Clone)]
pub struct HubConfig {
    /// Local storage home directory (default `~/.shiso`).
    pub home: PathBuf,
    /// Registry endpoint and verification key.
    pub registry: RegistryEndpoint,
    /// Hugging Face endpoint URL (default `https://huggingface.co`).
    pub hf_endpoint: Url,
    /// Optional Hugging Face authentication token.
    pub hf_token: Option<HfToken>,
    /// Maximum concurrent file downloads.
    pub max_parallel_files: NonZeroUsize,
}

impl HubConfig {
    /// Resolves default home directory (`~/.shiso` or override).
    #[must_use]
    pub fn default_home() -> PathBuf {
        if let Some(user_dirs) = std::env::var_os("HOME") {
            PathBuf::from(user_dirs).join(".shiso")
        } else {
            PathBuf::from(".shiso")
        }
    }

    /// Constructs a configuration with standard production defaults.
    ///
    /// # Panics
    /// Panics only if default URLs or key constants are invalid.
    #[must_use]
    pub fn with_defaults(home: PathBuf) -> Self {
        let registry = RegistryEndpoint::default_production();
        let hf_endpoint = Url::parse(DEFAULT_HF_ENDPOINT).expect("valid default HF endpoint");
        let max_parallel_files =
            NonZeroUsize::new(DEFAULT_MAX_PARALLEL_FILES).expect("nonzero default jobs");

        Self {
            home,
            registry,
            hf_endpoint,
            hf_token: None,
            max_parallel_files,
        }
    }
}
