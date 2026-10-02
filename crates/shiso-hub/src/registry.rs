//! Static HTTPS registry client, `ETag` caching and signature verification (§4.2, §4.3).

use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use ed25519_dalek::{Signature, Verifier};
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

    /// Fetches and verifies the registry index, using `ETag` caching if available.
    ///
    /// # Errors
    /// Returns [`HubError`] if downloading, caching, or signature verification fails.
    pub async fn fetch_index(&self) -> Result<Index, HubError> {
        let cache_dir = self.cache_dir();
        fs::create_dir_all(&cache_dir).map_err(|e| HubError::Io {
            path: cache_dir.clone(),
            source: e,
        })?;

        let index_path = cache_dir.join("index.json");
        let sig_path = cache_dir.join("index.json.sig");
        let etag_path = cache_dir.join("index.etag");

        let cached_etag = fs::read_to_string(&etag_path).ok();

        let base = self.base_url();
        let index_url = base.join("index.json").map_err(|e| HubError::Validation {
            reason: format!("failed to form index URL: {e}"),
        })?;
        let sig_url = base
            .join("index.json.sig")
            .map_err(|e| HubError::Validation {
                reason: format!("failed to form sig URL: {e}"),
            })?;

        // Try conditional GET for index.json
        let mut req = self.client.get(index_url.clone());
        if let Some(etag) = &cached_etag {
            req = req.header(IF_NONE_MATCH, etag.trim());
        }

        match req.send().await {
            Ok(resp) if resp.status() == reqwest::StatusCode::NOT_MODIFIED => {
                let index = self.read_cached_index(&index_path, &sig_path)?;
                Ok(index)
            }
            Ok(resp) if resp.status().is_success() => {
                self.download_fresh_index(resp, &index_path, &sig_path, &etag_path, &sig_url)
                    .await
            }
            Ok(resp) => {
                if index_path.is_file()
                    && sig_path.is_file()
                    && let Ok(index) = self.read_cached_index(&index_path, &sig_path)
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
                    && sig_path.is_file()
                    && let Ok(index) = self.read_cached_index(&index_path, &sig_path)
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
        sig_path: &Path,
        etag_path: &Path,
        sig_url: &url::Url,
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

        let sig_resp =
            self.client
                .get(sig_url.clone())
                .send()
                .await
                .map_err(|e| HubError::Network {
                    url: sig_url.to_string(),
                    source: e,
                })?;
        if !sig_resp.status().is_success() {
            return Err(HubError::HttpStatus {
                url: sig_url.to_string(),
                status: sig_resp.status(),
            });
        }
        let sig_bytes = sig_resp.bytes().await.map_err(|e| HubError::Network {
            url: sig_url.to_string(),
            source: e,
        })?;

        self.verify_signature(&index_bytes, &sig_bytes)?;

        atomic_write(index_path, &index_bytes)?;
        atomic_write(sig_path, &sig_bytes)?;
        if let Some(etag) = new_etag {
            atomic_write(etag_path, etag.as_bytes())?;
        }

        let index: Index = serde_json::from_slice(&index_bytes)?;
        Ok(index)
    }

    /// Reads and verifies a locally cached index.
    fn read_cached_index(&self, index_path: &Path, sig_path: &Path) -> Result<Index, HubError> {
        let index_bytes = fs::read(index_path).map_err(|e| HubError::Io {
            path: index_path.to_path_buf(),
            source: e,
        })?;
        let sig_bytes = fs::read(sig_path).map_err(|e| HubError::Io {
            path: sig_path.to_path_buf(),
            source: e,
        })?;
        self.verify_signature(&index_bytes, &sig_bytes)?;
        let index: Index = serde_json::from_slice(&index_bytes)?;
        Ok(index)
    }

    /// Verifies the Ed25519 signature over index bytes.
    ///
    /// # Errors
    /// Returns [`HubError::SignatureVerification`] if the signature does not match.
    pub fn verify_signature(&self, index_bytes: &[u8], sig_bytes: &[u8]) -> Result<(), HubError> {
        let signature = parse_signature(sig_bytes)?;
        self.config
            .registry
            .verifying_key
            .verify(index_bytes, &signature)
            .map_err(|e| HubError::SignatureVerification {
                reason: format!("Ed25519 verification failed: {e}"),
            })
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
        let manifest_rel = format!("models/{canonical_name}.json");
        let manifest_url =
            self.base_url()
                .join(&manifest_rel)
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

        // Verify SHA-256 against signed index
        let actual_digest = hex::encode(Sha256Hasher::digest(&manifest_bytes));
        if actual_digest != model_entry.manifest_sha256.as_str() {
            return Err(HubError::DigestMismatch {
                file: format!("models/{canonical_name}.json"),
                expected: model_entry.manifest_sha256.to_string(),
                actual: actual_digest,
            });
        }

        // Deserialize in client mode (ignoring unknown fields for additive changes)
        let recipe: Recipe = serde_json::from_slice(&manifest_bytes)?;

        // Validate basic rules
        validate_recipe(&recipe, ValidationMode::Client, None)?;

        // Check min_shiso_version
        if recipe.min_shiso_version > *current_shiso_version {
            return Err(HubError::VersionMismatch {
                min_version: recipe.min_shiso_version.to_string(),
                current_version: current_shiso_version.to_string(),
            });
        }

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

/// Parses an ed25519 signature from 64 raw bytes, 128 hex characters, or base64.
fn parse_signature(bytes: &[u8]) -> Result<Signature, HubError> {
    if bytes.len() == 64 {
        let arr: [u8; 64] = bytes
            .try_into()
            .map_err(|_| HubError::SignatureVerification {
                reason: "signature must be 64 bytes".to_string(),
            })?;
        return Ok(Signature::from_bytes(&arr));
    }

    let text = String::from_utf8_lossy(bytes).trim().to_string();
    if text.len() == 128 && text.bytes().all(|b| b.is_ascii_hexdigit()) {
        let raw = hex::decode(&text).map_err(|e| HubError::SignatureVerification {
            reason: format!("invalid signature hex: {e}"),
        })?;
        let arr: [u8; 64] = raw
            .try_into()
            .map_err(|_| HubError::SignatureVerification {
                reason: "signature must be 64 bytes".to_string(),
            })?;
        return Ok(Signature::from_bytes(&arr));
    }

    let clean: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if let Ok(raw) = base64::prelude::BASE64_STANDARD.decode(&clean)
        && raw.len() == 64
    {
        let arr: [u8; 64] = raw
            .try_into()
            .map_err(|_| HubError::SignatureVerification {
                reason: "signature must be 64 bytes".to_string(),
            })?;
        return Ok(Signature::from_bytes(&arr));
    }

    Err(HubError::SignatureVerification {
        reason: "signature must be 64 bytes (raw, 128-hex, or base64)".to_string(),
    })
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
