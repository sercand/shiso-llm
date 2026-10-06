//! Static HTTPS registry client with `ETag` caching (§4.2).

use std::fs;
use std::path::{Path, PathBuf};

use reqwest::header::{ETAG, IF_NONE_MATCH};
use sha2::{Digest, Sha256 as Sha256Hasher};

use crate::config::HubConfig;
use crate::error::HubError;
use crate::recipe::{Index, IndexModelEntry, ModelName, Recipe};
use crate::validate::{ValidationMode, validate_recipe};

/// Client for interacting with a static HTTPS registry.
#[derive(Debug, Clone)]
pub struct RegistryClient {
    client: reqwest::Client,
    config: HubConfig,
}

impl RegistryClient {
    /// Constructs a new `RegistryClient` with the given configuration.
    #[must_use]
    pub fn new(config: HubConfig) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client, config }
    }
    fn base_url(&self) -> url::Url {
        let mut u = self.config.registry.url.clone();
        if !u.path().ends_with('/') {
            let mut p = u.path().to_string();
            p.push('/');
            u.set_path(&p);
        }
        u
    }

    /// Returns the registry cache directory (`{home}/registry`).
    #[must_use]
    pub fn cache_dir(&self) -> PathBuf {
        self.config.home.join("registry")
    }

    /// Fetches the registry index, using `ETag` caching if available.
    ///
    /// # Errors
    /// Returns [`HubError`] if downloading or caching fails.
    pub async fn fetch_index(&self) -> Result<Index, HubError> {
        let cache_dir = self.cache_dir();
        fs::create_dir_all(&cache_dir).map_err(|e| HubError::Io {
            path: cache_dir.clone(),
            source: e,
        })?;

        let index_path = cache_dir.join("index.json");
        let etag_path = cache_dir.join("index.etag");

        let cached_etag = fs::read_to_string(&etag_path).ok();

        let base = self.base_url();
        let index_url = base.join("index.json").map_err(|e| HubError::Validation {
            reason: format!("failed to form index URL: {e}"),
        })?;

        // Try conditional GET for index.json
        let mut req = self.client.get(index_url.clone());
        if let Some(etag) = &cached_etag {
            req = req.header(IF_NONE_MATCH, etag.trim());
        }

        match req.send().await {
            Ok(resp) if resp.status() == reqwest::StatusCode::NOT_MODIFIED => {
                let index = Self::read_cached_index(&index_path)?;
                Ok(index)
            }
            Ok(resp) if resp.status().is_success() => {
                self.download_fresh_index(resp, &index_path, &etag_path)
                    .await
            }
            Ok(resp) => {
                if index_path.is_file()
                    && let Ok(index) = Self::read_cached_index(&index_path)
                {
                    return Ok(index);
                }
                Err(HubError::HttpStatus {
                    url: index_url.to_string(),
                    status: resp.status(),
                })
            }
            Err(e) => {
                if index_path.is_file()
                    && let Ok(index) = Self::read_cached_index(&index_path)
                {
                    return Ok(index);
                }
                Err(HubError::Network {
                    url: index_url.to_string(),
                    source: e,
                })
            }
        }
    }

    async fn download_fresh_index(
        &self,
        resp: reqwest::Response,
        index_path: &Path,
        etag_path: &Path,
    ) -> Result<Index, HubError> {
        let new_etag = resp
            .headers()
            .get(ETAG)
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);
        let index_bytes = resp.bytes().await.map_err(|e| HubError::Network {
            url: index_path.display().to_string(),
            source: e,
        })?;

        atomic_write(index_path, &index_bytes)?;
        if let Some(etag) = new_etag {
            atomic_write(etag_path, etag.as_bytes())?;
        }

        let index: Index = serde_json::from_slice(&index_bytes)?;
        Ok(index)
    }

    /// Reads a locally cached index.
    fn read_cached_index(index_path: &Path) -> Result<Index, HubError> {
        let index_bytes = fs::read(index_path).map_err(|e| HubError::Io {
            path: index_path.to_path_buf(),
            source: e,
        })?;
        let index: Index = serde_json::from_slice(&index_bytes)?;
        Ok(index)
    }

    /// Resolves a model name or alias against the index.
    #[must_use]
    pub fn resolve_model<'a>(
        &self,
        index: &'a Index,
        name_or_alias: &str,
    ) -> Option<&'a IndexModelEntry> {
        let query = name_or_alias.to_lowercase();
        index
            .models
            .iter()
            .find(|m| m.name.as_str() == query || m.aliases.iter().any(|a| a.as_str() == query))
    }

    /// Fetches, verifies, and parses a recipe manifest for `model_entry`.
    ///
    /// # Errors
    /// Returns [`HubError`] if fetching, digest verification, or recipe validation fails.
    pub async fn fetch_manifest(
        &self,
        model_entry: &IndexModelEntry,
        current_shiso_version: &semver::Version,
    ) -> Result<Recipe, HubError> {
        let models_dir = self.cache_dir().join("models");
        fs::create_dir_all(&models_dir).map_err(|e| HubError::Io {
            path: models_dir.clone(),
            source: e,
        })?;

        let canonical_name = model_entry.name.as_str();
        let manifest_path = models_dir.join(format!("{canonical_name}.json"));
        let etag_path = models_dir.join(format!("{canonical_name}.etag"));

        let cached_etag = fs::read_to_string(&etag_path).ok();
        let manifest_rel = manifest_path_of(model_entry)?;
        let manifest_url =
            self.base_url()
                .join(manifest_rel)
                .map_err(|e| HubError::Validation {
                    reason: format!("failed to form manifest URL: {e}"),
                })?;

        let mut req = self.client.get(manifest_url.clone());
        if let Some(etag) = &cached_etag {
            req = req.header(IF_NONE_MATCH, etag.trim());
        }

        let manifest_bytes = match req.send().await {
            Ok(resp) if resp.status() == reqwest::StatusCode::NOT_MODIFIED => {
                fs::read(&manifest_path).map_err(|e| HubError::Io {
                    path: manifest_path.clone(),
                    source: e,
                })?
            }
            Ok(resp) if resp.status().is_success() => {
                let new_etag = resp
                    .headers()
                    .get(ETAG)
                    .and_then(|v| v.to_str().ok())
                    .map(ToString::to_string);
                let bytes = resp.bytes().await.map_err(|e| HubError::Network {
                    url: manifest_url.to_string(),
                    source: e,
                })?;

                atomic_write(&manifest_path, &bytes)?;
                if let Some(etag) = new_etag {
                    atomic_write(&etag_path, etag.as_bytes())?;
                }
                bytes.to_vec()
            }
            Ok(resp) => {
                if manifest_path.is_file() {
                    fs::read(&manifest_path).map_err(|e| HubError::Io {
                        path: manifest_path.clone(),
                        source: e,
                    })?
                } else {
                    return Err(HubError::HttpStatus {
                        url: manifest_url.to_string(),
                        status: resp.status(),
                    });
                }
            }
            Err(e) => {
                if manifest_path.is_file() {
                    fs::read(&manifest_path).map_err(|e| HubError::Io {
                        path: manifest_path.clone(),
                        source: e,
                    })?
                } else {
                    return Err(HubError::Network {
                        url: manifest_url.to_string(),
                        source: e,
                    });
                }
            }
        };

        // Verify SHA-256 against the index entry the manifest was resolved from
        let actual_digest = hex::encode(Sha256Hasher::digest(&manifest_bytes));
        if actual_digest != model_entry.manifest_sha256.as_str() {
            return Err(HubError::DigestMismatch {
                file: manifest_rel.to_string(),
                expected: model_entry.manifest_sha256.to_string(),
                actual: actual_digest,
            });
        }

        // Deserialize in client mode (ignoring unknown fields for additive changes)
        let recipe: Recipe = serde_json::from_slice(&manifest_bytes)?;

        // The version gate first: a recipe written for a newer shiso may use
        // rules this build does not know, and "upgrade shiso" is the answer
        // that helps, where a validation message about a rule would not.
        if recipe.min_shiso_version > *current_shiso_version {
            return Err(HubError::VersionMismatch {
                min_version: recipe.min_shiso_version.to_string(),
                current_version: current_shiso_version.to_string(),
            });
        }

        // Validate basic rules
        validate_recipe(&recipe, ValidationMode::Client)?;

        Ok(recipe)
    }

    /// Reads a cached recipe manifest from local storage if present.
    #[must_use]
    pub fn read_cached_recipe(&self, model_name: &ModelName) -> Option<Recipe> {
        let manifest_path = self
            .cache_dir()
            .join("models")
            .join(format!("{}.json", model_name.as_str()));
        if let Ok(bytes) = fs::read(manifest_path) {
            serde_json::from_slice(&bytes).ok()
        } else {
            None
        }
    }
}

/// The entry's manifest path, refused unless it is a plain relative path.
///
/// The path is joined onto the registry base URL, so an absolute path, a
/// scheme or a `..` segment would let one index entry point the client at a
/// manifest outside the registry it trusts.
fn manifest_path_of(entry: &IndexModelEntry) -> Result<&str, HubError> {
    let rel = entry.manifest.as_str();
    let refuse = |why: &str| HubError::Validation {
        reason: format!(
            "index entry '{}' has manifest path '{rel}' that {why}",
            entry.name
        ),
    };
    if rel.is_empty() {
        return Err(refuse("is empty"));
    }
    if rel.starts_with('/') || rel.contains(['\\', ':', '?', '#', '%']) {
        return Err(refuse("is not a plain relative path"));
    }
    if rel
        .split('/')
        .any(|seg| seg.is_empty() || seg == "." || seg == "..")
    {
        return Err(refuse("has an empty, '.' or '..' segment"));
    }
    Ok(rel)
}

/// Atomically writes `bytes` to `dest` using a temporary file.
pub(crate) fn atomic_write(dest: &Path, bytes: &[u8]) -> Result<(), HubError> {
    let parent = dest.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|e| HubError::Io {
        path: parent.to_path_buf(),
        source: e,
    })?;

    let tmp = parent.join(format!(
        ".tmp_{}_{}",
        std::process::id(),
        dest.file_name().and_then(|f| f.to_str()).unwrap_or("file")
    ));

    fs::write(&tmp, bytes).map_err(|e| HubError::Io {
        path: tmp.clone(),
        source: e,
    })?;

    fs::rename(&tmp, dest).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        HubError::Io {
            path: dest.to_path_buf(),
            source: e,
        }
    })?;

    Ok(())
}
