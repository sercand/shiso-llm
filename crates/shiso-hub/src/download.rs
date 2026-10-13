//! Resumable, digest-verified HTTP downloader and parallel transfer manager (§5.5).

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures_util::StreamExt;
use reqwest::header::{AUTHORIZATION, HeaderValue, RANGE};
use reqwest::redirect::Policy;
use sha2::{Digest, Sha256 as Sha256Hasher};
use tokio::sync::{Semaphore, mpsc};
use url::Url;

use crate::config::HubConfig;
use crate::error::HubError;
use crate::host::probe_free_disk;
use crate::progress::ProgressEvent;
use crate::recipe::{FileEntry, Role, Sha256, Variant};

/// Where one file is downloaded from.
#[derive(Debug, Clone, Copy)]
pub enum FileOrigin<'a> {
    /// `<hf endpoint>/<repo>/resolve/<revision>/<name>`.
    HuggingFace {
        /// Repository (`owner/name`).
        repo: &'a str,
        /// Commit SHA.
        revision: &'a str,
    },
    /// `<registry url>/models/<model>/<name>`: a file the hub hosts beside the
    /// model's manifest. No Hugging Face token is ever sent there.
    Hub,
}

/// Description of a single file download operation (§5.5).
#[derive(Debug, Clone)]
pub struct DownloadFileDesc<'a> {
    /// Canonical model name.
    pub model_name: &'a str,
    /// Variant key string.
    pub variant_key: &'a str,
    /// Component role.
    pub role: Role,
    /// Where the file comes from.
    pub origin: FileOrigin<'a>,
    /// File entry metadata.
    pub file: &'a FileEntry,
    /// Optional progress event channel.
    pub progress_tx: Option<&'a mpsc::Sender<ProgressEvent>>,
}

#[derive(Debug)]
struct DownloadItem {
    role: Role,
    /// The Hugging Face repository and commit; `None` is a hub-hosted file
    /// (validation guarantees a Hugging Face component has both).
    hf: Option<(String, String)>,
    file: FileEntry,
}

/// Downloader engine managing HTTP transfers and resuming partial files.
#[derive(Debug, Clone)]
pub struct Downloader {
    client: reqwest::Client,
    config: HubConfig,
}

impl Downloader {
    /// Constructs a new `Downloader` with custom redirect policy.
    #[must_use]
    pub fn new(config: HubConfig) -> Self {
        let redirect_policy = Policy::custom(|attempt| {
            if attempt.previous().len() >= 10 {
                attempt.error("too many redirects")
            } else {
                attempt.follow()
            }
        });

        // Building a client loads the platform trust store, which on macOS is a
        // keychain round trip that took 35-60s on a developer machine. An
        // endpoint pair that is plain http (a local mock or registry) never
        // does TLS, so it skips that load.
        let plain_http =
            config.registry.url.scheme() == "http" && config.hf_endpoint.scheme() == "http";
        let client = reqwest::Client::builder()
            .tls_built_in_root_certs(!plain_http)
            .redirect(redirect_policy)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self { client, config }
    }

    /// The hub URL of `name` beside model `model`'s manifest.
    ///
    /// # Errors
    /// Returns [`HubError::Validation`] if the joined URL does not parse.
    pub fn hub_file_url(&self, model: &str, name: &str) -> Result<Url, HubError> {
        let mut base = self.config.registry.url.clone();
        // `join` replaces the last segment of a base without a trailing
        // slash, which would drop the registry's `v1`.
        if !base.path().ends_with('/') {
            let path = format!("{}/", base.path());
            base.set_path(&path);
        }
        base.join(&format!("models/{model}/{name}"))
            .map_err(|e| HubError::Validation {
                reason: format!("invalid hub file URL for '{model}/{name}': {e}"),
            })
    }

    /// Returns the blobs directory (`{home}/blobs/sha256`).
    #[must_use]
    pub fn blobs_dir(&self) -> PathBuf {
        self.config.home.join("blobs").join("sha256")
    }

    /// Returns the temporary directory (`{home}/tmp`).
    #[must_use]
    pub fn tmp_dir(&self) -> PathBuf {
        self.config.home.join("tmp")
    }

    /// Path to a committed blob.
    #[must_use]
    pub fn blob_path(&self, sha256: &Sha256) -> PathBuf {
        self.blobs_dir().join(sha256.as_str())
    }

    /// Checks if a blob already exists and has the expected size.
    #[must_use]
    pub fn is_blob_present(&self, sha256: &Sha256, expected_size: u64) -> bool {
        let path = self.blob_path(sha256);
        if let Ok(meta) = fs::metadata(&path) {
            meta.len() == expected_size
        } else {
            false
        }
    }

    /// Downloads all missing blobs for `variant` in parallel, emitting progress events.
    ///
    /// # Errors
    /// Returns [`HubError`] if disk space is insufficient, transfer fails, or digest mismatches.
    pub async fn download_variant(
        &self,
        model_name: &str,
        variant: &Variant,
        progress_tx: Option<&mpsc::Sender<ProgressEvent>>,
    ) -> Result<(), HubError> {
        fs::create_dir_all(self.blobs_dir()).map_err(|e| HubError::Io {
            path: self.blobs_dir(),
            source: e,
        })?;
        fs::create_dir_all(self.tmp_dir()).map_err(|e| HubError::Io {
            path: self.tmp_dir(),
            source: e,
        })?;

        let mut items = Vec::new();
        let mut missing_bytes: u64 = 0;

        for (role, comp) in &variant.components {
            for file in &comp.files {
                if !self.is_blob_present(&file.sha256, file.size) {
                    missing_bytes = missing_bytes.saturating_add(file.size);
                }
                items.push(DownloadItem {
                    role: *role,
                    hf: comp
                        .hf_source()
                        .map(|(repo, rev)| (repo.to_string(), rev.to_string())),
                    file: file.clone(),
                });
            }
        }

        if missing_bytes > 0 {
            let free = probe_free_disk(&self.config.home)?;
            if free < missing_bytes {
                return Err(HubError::InsufficientDiskSpace {
                    needed_bytes: missing_bytes,
                    free_bytes: free,
                });
            }
        }

        if let Some(tx) = progress_tx {
            let _ = tx
                .send(ProgressEvent::Resolved {
                    model: model_name.to_string(),
                    variant: variant.key.to_string(),
                    total: missing_bytes,
                })
                .await;
        }

        if missing_bytes == 0 {
            return Ok(());
        }

        let max_jobs = self.config.max_parallel_files.get();
        let semaphore = Arc::new(Semaphore::new(max_jobs));
        let mut tasks = Vec::new();

        for item in items {
            if self.is_blob_present(&item.file.sha256, item.file.size) {
                continue;
            }

            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| HubError::Other("download semaphore closed".to_string()))?;

            let downloader = self.clone();
            let model = model_name.to_string();
            let variant_key = variant.key.to_string();
            let tx = progress_tx.cloned();

            tasks.push(tokio::spawn(async move {
                let _permit = permit;
                let desc = DownloadFileDesc {
                    model_name: &model,
                    variant_key: &variant_key,
                    role: item.role,
                    origin: match &item.hf {
                        Some((repo, revision)) => FileOrigin::HuggingFace { repo, revision },
                        None => FileOrigin::Hub,
                    },
                    file: &item.file,
                    progress_tx: tx.as_ref(),
                };
                downloader.download_file(desc).await
            }));
        }

        for task in tasks {
            let res = task
                .await
                .map_err(|e| HubError::Other(format!("download task panicked: {e}")))?;
            res?;
        }

        Ok(())
    }

    /// Downloads and verifies a single file with Range resume.
    ///
    /// # Errors
    /// Returns [`HubError`] on network, authentication, I/O, or verification errors.
    pub async fn download_file(&self, desc: DownloadFileDesc<'_>) -> Result<(), HubError> {
        let blob_path = self.blob_path(&desc.file.sha256);
        if self.is_blob_present(&desc.file.sha256, desc.file.size) {
            return Ok(());
        }

        fs::create_dir_all(self.tmp_dir()).map_err(|e| HubError::Io {
            path: self.tmp_dir(),
            source: e,
        })?;
        fs::create_dir_all(self.blobs_dir()).map_err(|e| HubError::Io {
            path: self.blobs_dir(),
            source: e,
        })?;

        let lock_path = self
            .tmp_dir()
            .join(format!("{}.lock", desc.file.sha256.as_str()));
        let part_path = self
            .tmp_dir()
            .join(format!("{}.part", desc.file.sha256.as_str()));

        let lock_file = self.acquire_lock(&lock_path).await?;

        if self.is_blob_present(&desc.file.sha256, desc.file.size) {
            let _ = lock_file.unlock();
            return Ok(());
        }

        if let Some((existing_bytes, initial_hasher)) =
            Self::check_existing_part(&part_path, &blob_path, desc.file)?
        {
            if existing_bytes == desc.file.size {
                let _ = lock_file.unlock();
                return Ok(());
            }

            self.perform_download_and_verify(
                &desc,
                &part_path,
                &blob_path,
                existing_bytes,
                initial_hasher,
            )
            .await?;
        } else {
            self.perform_download_and_verify(&desc, &part_path, &blob_path, 0, Sha256Hasher::new())
                .await?;
        }

        let _ = lock_file.unlock();
        Ok(())
    }

    async fn acquire_lock(&self, lock_path: &Path) -> Result<File, HubError> {
        let lp = lock_path.to_path_buf();
        tokio::task::spawn_blocking(move || -> Result<File, HubError> {
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&lp)
                .map_err(|e| HubError::Io {
                    path: lp.clone(),
                    source: e,
                })?;
            f.lock().map_err(|e| HubError::Io {
                path: lp,
                source: e,
            })?;
            Ok(f)
        })
        .await
        .map_err(|e| HubError::Other(format!("lock task failed: {e}")))?
    }

    // None is not a refusal: no valid part file exists on disk to resume from.
    fn check_existing_part(
        part_path: &Path,
        blob_path: &Path,
        file: &FileEntry,
    ) -> Result<Option<(u64, Sha256Hasher)>, HubError> {
        if !part_path.is_file() {
            return Ok(None);
        }

        let meta = fs::metadata(part_path).map_err(|e| HubError::Io {
            path: part_path.to_path_buf(),
            source: e,
        })?;

        if meta.len() > file.size {
            let _ = fs::remove_file(part_path);
            return Ok(None);
        }

        if meta.len() == file.size {
            if let Ok(mut f) = File::open(part_path) {
                let mut buf = vec![0u8; 64 * 1024];
                let mut h = Sha256Hasher::new();
                while let Ok(n) = f.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    h.update(&buf[..n]);
                }
                if hex::encode(h.finalize()) == file.sha256.as_str() {
                    commit_blob(part_path, blob_path)?;
                    return Ok(Some((file.size, Sha256Hasher::new())));
                }
            }
            let _ = fs::remove_file(part_path);
            return Ok(None);
        }

        if meta.len() > 0
            && let Ok(mut f) = File::open(part_path)
        {
            let mut buf = vec![0u8; 64 * 1024];
            let mut h = Sha256Hasher::new();
            while let Ok(n) = f.read(&mut buf) {
                if n == 0 {
                    break;
                }
                h.update(&buf[..n]);
            }
            return Ok(Some((meta.len(), h)));
        }

        Ok(None)
    }

    async fn perform_download_and_verify(
        &self,
        desc: &DownloadFileDesc<'_>,
        part_path: &Path,
        blob_path: &Path,
        existing_bytes: u64,
        initial_hasher: Sha256Hasher,
    ) -> Result<(), HubError> {
        let url = match desc.origin {
            FileOrigin::HuggingFace { repo, revision } => {
                let url_str = format!(
                    "{}/{}/resolve/{}/{}",
                    self.config.hf_endpoint.as_str().trim_end_matches('/'),
                    repo,
                    revision,
                    desc.file.name
                );
                Url::parse(&url_str).map_err(|e| HubError::Validation {
                    reason: format!("invalid download URL '{url_str}': {e}"),
                })?
            }
            FileOrigin::Hub => self.hub_file_url(desc.model_name, &desc.file.name)?,
        };
        // The token belongs to Hugging Face: a hub file never carries it, even
        // when the registry and the HF endpoint share a host.
        let send_token = matches!(desc.origin, FileOrigin::HuggingFace { .. })
            && url.host_str() == self.config.hf_endpoint.host_str();

        if let Some(tx) = desc.progress_tx {
            let _ = tx
                .send(ProgressEvent::FileStarted {
                    model: desc.model_name.to_string(),
                    variant: desc.variant_key.to_string(),
                    role: desc.role.as_str().to_string(),
                    file: desc.file.name.clone(),
                    total: desc.file.size,
                })
                .await;
        }

        let mut req = self.client.get(url.clone());
        if let Some(token) = &self.config.hf_token
            && send_token
        {
            let auth_val = format!("Bearer {}", token.as_str());
            if let Ok(hv) = HeaderValue::from_str(&auth_val) {
                req = req.header(AUTHORIZATION, hv);
            }
        }

        if existing_bytes > 0 {
            req = req.header(RANGE, format!("bytes={existing_bytes}-"));
        }

        let resp = req.send().await.map_err(|e| HubError::Network {
            url: url.to_string(),
            source: e,
        })?;

        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(HubError::Auth {
                repo: match desc.origin {
                    FileOrigin::HuggingFace { repo, .. } => repo.to_string(),
                    FileOrigin::Hub => format!("hub:{}/{}", desc.model_name, desc.file.name),
                },
                token_sent: send_token && self.config.hf_token.is_some(),
            });
        }

        let (part_file, hasher, downloaded_so_far) =
            if status == reqwest::StatusCode::PARTIAL_CONTENT && existing_bytes > 0 {
                let f = OpenOptions::new()
                    .append(true)
                    .open(part_path)
                    .map_err(|e| HubError::Io {
                        path: part_path.to_path_buf(),
                        source: e,
                    })?;
                (f, initial_hasher, existing_bytes)
            } else if status.is_success() {
                let f = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(part_path)
                    .map_err(|e| HubError::Io {
                        path: part_path.to_path_buf(),
                        source: e,
                    })?;
                (f, Sha256Hasher::new(), 0)
            } else {
                return Err(HubError::HttpStatus {
                    url: url.to_string(),
                    status,
                });
            };

        stream_response_chunks(resp, part_file, hasher, downloaded_so_far, desc, part_path).await?;
        commit_blob(part_path, blob_path)?;

        if let Some(tx) = desc.progress_tx {
            let _ = tx
                .send(ProgressEvent::FileVerified {
                    model: desc.model_name.to_string(),
                    variant: desc.variant_key.to_string(),
                    role: desc.role.as_str().to_string(),
                    file: desc.file.name.clone(),
                    total: desc.file.size,
                })
                .await;
        }

        Ok(())
    }
}

async fn stream_response_chunks(
    resp: reqwest::Response,
    mut part_file: File,
    mut hasher: Sha256Hasher,
    mut downloaded_so_far: u64,
    desc: &DownloadFileDesc<'_>,
    part_path: &Path,
) -> Result<(), HubError> {
    let mut stream = resp.bytes_stream();
    let mut last_progress_bytes = downloaded_so_far;
    let progress_interval = (desc.file.size / 20).max(512 * 1024);

    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.map_err(|e| HubError::Network {
            url: desc.file.name.clone(),
            source: e,
        })?;

        part_file.write_all(&chunk).map_err(|e| HubError::Io {
            path: part_path.to_path_buf(),
            source: e,
        })?;
        hasher.update(&chunk);
        downloaded_so_far += chunk.len() as u64;

        if downloaded_so_far - last_progress_bytes >= progress_interval
            || downloaded_so_far == desc.file.size
        {
            last_progress_bytes = downloaded_so_far;
            if let Some(tx) = desc.progress_tx {
                let _ = tx
                    .send(ProgressEvent::FileProgress {
                        model: desc.model_name.to_string(),
                        variant: desc.variant_key.to_string(),
                        role: desc.role.as_str().to_string(),
                        file: desc.file.name.clone(),
                        bytes: downloaded_so_far,
                        total: desc.file.size,
                    })
                    .await;
            }
        }
    }

    part_file.flush().map_err(|e| HubError::Io {
        path: part_path.to_path_buf(),
        source: e,
    })?;
    part_file.sync_all().map_err(|e| HubError::Io {
        path: part_path.to_path_buf(),
        source: e,
    })?;
    drop(part_file);

    if downloaded_so_far != desc.file.size {
        let _ = fs::remove_file(part_path);
        return Err(HubError::SizeMismatch {
            file: desc.file.name.clone(),
            expected: desc.file.size,
            actual: downloaded_so_far,
        });
    }

    let computed_digest = hex::encode(hasher.finalize());
    if computed_digest != desc.file.sha256.as_str() {
        let _ = fs::remove_file(part_path);
        return Err(HubError::DigestMismatch {
            file: desc.file.name.clone(),
            expected: desc.file.sha256.to_string(),
            actual: computed_digest,
        });
    }

    Ok(())
}

fn commit_blob(part_path: &Path, blob_path: &Path) -> Result<(), HubError> {
    if let Some(parent) = blob_path.parent() {
        fs::create_dir_all(parent).map_err(|e| HubError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }

    fs::rename(part_path, blob_path).map_err(|e| HubError::Io {
        path: blob_path.to_path_buf(),
        source: e,
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(blob_path, fs::Permissions::from_mode(0o444));
    }

    Ok(())
}
