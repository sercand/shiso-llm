//! Hub caller configuration and authentication types (§5.1).

use std::fmt;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use url::Url;

use crate::error::HubError;

/// Default static registry URL. The registry repository is served statically,
/// so the index lives at `<host>/registry/v1/index.json` under its root.
pub const DEFAULT_REGISTRY_URL: &str = "https://hub.shiso.run/registry/v1/";

/// Default Hugging Face endpoint URL.
pub const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";

/// Default maximum parallel file downloads.
pub const DEFAULT_MAX_PARALLEL_FILES: usize = 4;

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

/// Registry static endpoint URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryEndpoint {
    /// Base URL of the registry (e.g. `https://hub.shiso.run/registry/v1`).
    pub url: Url,
}

impl RegistryEndpoint {
    /// Constructs the default production registry endpoint.
    ///
    /// # Panics
    /// Panics only if the compiled-in default URL is invalid (checked by unit tests).
    #[must_use]
    pub fn default_production() -> Self {
        let url = Url::parse(DEFAULT_REGISTRY_URL).expect("valid default registry URL");
        Self { url }
    }

    /// Constructs an endpoint from a URL string, normalizing a trailing `/`.
    ///
    /// # Errors
    /// Returns [`HubError::Validation`] if the URL cannot be parsed.
    pub fn parse(url_str: &str) -> Result<Self, HubError> {
        let mut url = Url::parse(url_str).map_err(|e| HubError::Validation {
            reason: format!("invalid registry URL '{url_str}': {e}"),
        })?;
        if !url.path().ends_with('/') {
            let mut path = url.path().to_string();
            path.push('/');
            url.set_path(&path);
        }
        Ok(Self { url })
    }
}

/// Everything the hub needs from its caller; nothing is read from the environment (§5.1).
#[derive(Debug, Clone)]
pub struct HubConfig {
    /// Local storage home directory (default `~/.shiso`).
    pub home: PathBuf,
    /// Registry endpoint.
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
    /// Panics only if default URLs are invalid.
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
