//! Content-addressed blob store, staging transactions, file locks, and garbage collection (§4.4, §5.6, §5.7).

use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256 as Sha256Hasher};

use crate::config::{HfToken, HubConfig};
use crate::error::HubError;
use crate::recipe::{
    InstalledFileEntry, InstalledMetadata, ModelName, Recipe, Role, Sha256, Variant, VariantKey,
};
#[derive(serde::Deserialize)]
struct AuthDoc {
    token: Option<String>,
    hf_token: Option<String>,
}

/// Summary of an installed model variant found on disk.
#[derive(Debug, Clone)]
pub struct InstalledVariantSummary {
    /// Canonical model name.
    pub model: ModelName,
    /// Variant key.
    pub variant: VariantKey,
    /// Target backend.
    pub backend: String,
    /// Quantization label.
    pub quant: String,
    /// Roles present in installation.
    pub roles: Vec<Role>,
    /// Total disk size in bytes.
    pub disk_bytes: u64,
    /// Installed manifest SHA-256.
    pub manifest_sha256: Sha256,
    /// Cached recipe if readable.
    pub recipe: Option<Recipe>,
}

/// Statistics reported by garbage collection (§5.7).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GcStats {
    /// Number of unreferenced blobs deleted.
    pub deleted_blobs: usize,
    /// Bytes of disk space reclaimed from unreferenced blobs.
    pub reclaimed_bytes: u64,
    /// Number of stale partial downloads deleted.
    pub deleted_parts: usize,
}

/// Storage manager handling blob hardlinks, locking, staging, and GC.
#[derive(Debug, Clone)]
pub struct BlobStore {
    config: HubConfig,
}

impl BlobStore {
    /// Constructs a `BlobStore` for `config`.
    #[must_use]
    pub fn new(config: HubConfig) -> Self {
        Self { config }
    }

    /// Root directory for installed models (`{home}/models`).
    #[must_use]
    pub fn models_dir(&self) -> PathBuf {
        self.config.home.join("models")
    }

    /// Directory for one model family (`{home}/models/<name>`).
    #[must_use]
    pub fn model_dir(&self, name: &ModelName) -> PathBuf {
        self.models_dir().join(name.as_str())
    }

    /// Directory for one installed variant (`{home}/models/<name>/<key>`).
    #[must_use]
    pub fn variant_dir(&self, name: &ModelName, key: &VariantKey) -> PathBuf {
        self.model_dir(name).join(key.as_str())
    }

    /// Path to lock file for variant (`{home}/models/<name>/<key>/.lock`).
    #[must_use]
    pub fn variant_lock_path(&self, name: &ModelName, key: &VariantKey) -> PathBuf {
        self.variant_dir(name, key).join(".lock")
    }

    /// Checks if `variant` is installed and up to date with `manifest_sha256`.
    #[must_use]
    pub fn is_up_to_date(
        &self,
        name: &ModelName,
        key: &VariantKey,
        manifest_sha256: &Sha256,
        variant: &Variant,
    ) -> bool {
        let vdir = self.variant_dir(name, key);
        let installed_json_path = vdir.join("installed.json");
        let recipe_json_path = vdir.join("recipe.json");

        if !installed_json_path.is_file() || !recipe_json_path.is_file() {
            return false;
        }

        let Ok(installed_bytes) = fs::read(&installed_json_path) else {
            return false;
        };

        let Ok(metadata) = serde_json::from_slice::<InstalledMetadata>(&installed_bytes) else {
            return false;
        };

        if &metadata.manifest_sha256 != manifest_sha256 {
            return false;
        }

        // Verify that every component file exists and matches size
        for (role, comp) in &variant.components {
            for file in &comp.files {
                let file_path = vdir.join(role.as_str()).join(&file.name);
                match fs::metadata(&file_path) {
                    Ok(meta) if meta.len() == file.size => {}
                    _ => return false,
                }
            }
        }

        true
    }

    /// Takes the variant lock shared (for `serve`) or exclusive (for `pull` / `rm`).
    ///
    /// # Errors
    /// Returns [`HubError::Io`] if creating or opening the lock file fails.
    pub fn open_variant_lock(&self, name: &ModelName, key: &VariantKey) -> Result<File, HubError> {
        let vdir = self.variant_dir(name, key);
        fs::create_dir_all(&vdir).map_err(|e| HubError::Io {
            path: vdir.clone(),
            source: e,
        })?;
        let lock_path = self.variant_lock_path(name, key);
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| HubError::Io {
                path: lock_path,
                source: e,
            })
    }

    /// Atomically installs a downloaded variant into `models/<name>/<key>`.
    ///
    /// # Errors
    /// Returns [`HubError`] if staging, hardlinking, or swapping fails.
    pub fn install_variant(
        &self,
        recipe: &Recipe,
        variant: &Variant,
        manifest_sha256: &Sha256,
    ) -> Result<(), HubError> {
        let model_name = &recipe.name;
        let variant_key = &variant.key;
        let mdir = self.model_dir(model_name);
        fs::create_dir_all(&mdir).map_err(|e| HubError::Io {
            path: mdir.clone(),
            source: e,
        })?;

        // 1. Take exclusive lock on the target variant directory
        let lock_file = self.open_variant_lock(model_name, variant_key)?;
        lock_file.lock().map_err(|e| HubError::Io {
            path: self.variant_lock_path(model_name, variant_key),
            source: e,
        })?;

        let staging_dir = mdir.join(format!(".{}.staging", variant_key.as_str()));
        let old_dir = mdir.join(format!(".{}.old", variant_key.as_str()));
        let final_dir = self.variant_dir(model_name, variant_key);

        // Clean up previous abandoned staging or old directories
        let _ = fs::remove_dir_all(&staging_dir);
        let _ = fs::remove_dir_all(&old_dir);

        fs::create_dir_all(&staging_dir).map_err(|e| HubError::Io {
            path: staging_dir.clone(),
            source: e,
        })?;

        // 2. Populate hardlinks into staging
        let blobs_dir = self.config.home.join("blobs").join("sha256");
        let mut installed_files = Vec::new();

        for (role, comp) in &variant.components {
            let role_dir = staging_dir.join(role.as_str());
            fs::create_dir_all(&role_dir).map_err(|e| HubError::Io {
                path: role_dir.clone(),
                source: e,
            })?;

            for file in &comp.files {
                let blob_file = blobs_dir.join(file.sha256.as_str());
                if !blob_file.is_file() {
                    return Err(HubError::Validation {
                        reason: format!(
                            "blob for file '{}' ({}) is missing",
                            file.name, file.sha256
                        ),
                    });
                }

                let target_file = role_dir.join(&file.name);
                if let Some(parent) = target_file.parent() {
                    fs::create_dir_all(parent).map_err(|e| HubError::Io {
                        path: parent.to_path_buf(),
                        source: e,
                    })?;
                }

                // Create hardlink
                fs::hard_link(&blob_file, &target_file).map_err(|e| HubError::Io {
                    path: target_file.clone(),
                    source: e,
                })?;

                installed_files.push(InstalledFileEntry {
                    role: *role,
                    name: file.name.clone(),
                    size: file.size,
                    sha256: file.sha256.clone(),
                });
            }
        }

        // 3. Write recipe.json and installed.json in staging
        let recipe_json = serde_json::to_vec_pretty(recipe)?;
        fs::write(staging_dir.join("recipe.json"), recipe_json).map_err(|e| HubError::Io {
            path: staging_dir.join("recipe.json"),
            source: e,
        })?;

        let metadata = InstalledMetadata {
            manifest_sha256: manifest_sha256.clone(),
            installed_at: current_iso8601(),
            files: installed_files,
        };
        let installed_json = serde_json::to_vec_pretty(&metadata)?;
        fs::write(staging_dir.join("installed.json"), installed_json).map_err(|e| {
            HubError::Io {
                path: staging_dir.join("installed.json"),
                source: e,
            }
        })?;

        // Ensure lock file exists in staging before swapping
        let staging_lock = staging_dir.join(".lock");
        let _ = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&staging_lock);

        // 4. Atomic swap
        if final_dir.is_dir() {
            fs::rename(&final_dir, &old_dir).map_err(|e| HubError::Io {
                path: old_dir.clone(),
                source: e,
            })?;
        }

        fs::rename(&staging_dir, &final_dir).map_err(|e| HubError::Io {
            path: final_dir.clone(),
            source: e,
        })?;

        let _ = fs::remove_dir_all(&old_dir);

        // 5. Unlock and run blob GC
        let _ = lock_file.unlock();
        let _ = self.run_gc();

        Ok(())
    }

    /// Removes an installed variant, with lock protection or `--force`.
    ///
    /// # Errors
    /// Returns [`HubError::ModelInUse`] if locked by a running server and `force` is false.
    pub fn remove_variant(
        &self,
        name: &ModelName,
        key: &VariantKey,
        force: bool,
    ) -> Result<(), HubError> {
        let vdir = self.variant_dir(name, key);
        if !vdir.is_dir() {
            return Err(HubError::VariantNotFound {
                model: name.to_string(),
                variant: key.to_string(),
            });
        }

        if !force {
            let lock_path = self.variant_lock_path(name, key);
            if lock_path.is_file() {
                let lock_file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&lock_path)
                    .map_err(|e| HubError::Io {
                        path: lock_path.clone(),
                        source: e,
                    })?;
                if lock_file.try_lock().is_err() {
                    return Err(HubError::ModelInUse {
                        name: name.to_string(),
                        variant: key.to_string(),
                    });
                }
                let _ = lock_file.unlock();
            }
        }

        fs::remove_dir_all(&vdir).map_err(|e| HubError::Io {
            path: vdir,
            source: e,
        })?;

        // If model directory is empty, remove it too
        let mdir = self.model_dir(name);
        if let Ok(mut entries) = fs::read_dir(&mdir)
            && entries.next().is_none()
        {
            let _ = fs::remove_dir(&mdir);
        }

        let _ = self.run_gc();
        Ok(())
    }

    /// Removes all installed variants of a model.
    ///
    /// # Errors
    /// Returns [`HubError`] if any variant fails to remove.
    pub fn remove_model(&self, name: &ModelName, force: bool) -> Result<usize, HubError> {
        let mdir = self.model_dir(name);
        if !mdir.is_dir() {
            return Err(HubError::ModelNotFound {
                name: name.to_string(),
            });
        }

        let mut count = 0;
        let entries = fs::read_dir(&mdir).map_err(|e| HubError::Io {
            path: mdir.clone(),
            source: e,
        })?;

        for entry_res in entries {
            let entry = entry_res.map_err(|e| HubError::Io {
                path: mdir.clone(),
                source: e,
            })?;
            let path = entry.path();
            if path.is_dir() {
                let file_name = entry.file_name().to_string_lossy().to_string();
                if !file_name.starts_with('.')
                    && let Ok(key) = VariantKey::parse(&file_name)
                {
                    self.remove_variant(name, &key, force)?;
                    count += 1;
                }
            }
        }

        let _ = fs::remove_dir_all(&mdir);
        let _ = self.run_gc();
        Ok(count)
    }

    /// Verifies all files of an installed variant against `installed.json`.
    ///
    /// Returns a list of `(file_name, ok_bool)`.
    ///
    /// # Errors
    /// Returns [`HubError`] if metadata cannot be read.
    pub fn verify_variant(
        &self,
        name: &ModelName,
        key: &VariantKey,
    ) -> Result<Vec<(String, bool)>, HubError> {
        let vdir = self.variant_dir(name, key);
        let installed_json = vdir.join("installed.json");
        let bytes = fs::read(&installed_json).map_err(|e| HubError::Io {
            path: installed_json,
            source: e,
        })?;

        let metadata: InstalledMetadata = serde_json::from_slice(&bytes)?;
        let mut results = Vec::new();

        for file in metadata.files {
            let file_path = vdir.join(file.role.as_str()).join(&file.name);
            let ok = if let Ok(mut f) = File::open(&file_path) {
                let mut hasher = Sha256Hasher::new();
                let mut buf = vec![0u8; 64 * 1024];
                let mut total = 0;
                let read_ok = true;
                while let Ok(n) = f.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    hasher.update(&buf[..n]);
                    total += n as u64;
                }
                read_ok
                    && total == file.size
                    && hex::encode(hasher.finalize()) == file.sha256.as_str()
            } else {
                false
            };
            results.push((format!("{}/{}", file.role.as_str(), file.name), ok));
        }

        Ok(results)
    }

    /// Lists all installed models and variants across the store.
    ///
    /// # Errors
    /// Returns [`HubError`] on filesystem read errors.
    pub fn list_installed(&self) -> Result<Vec<InstalledVariantSummary>, HubError> {
        let models_dir = self.models_dir();
        if !models_dir.is_dir() {
            return Ok(Vec::new());
        }

        let mut list = Vec::new();
        let model_entries = fs::read_dir(&models_dir).map_err(|e| HubError::Io {
            path: models_dir.clone(),
            source: e,
        })?;

        for model_entry_res in model_entries {
            let Ok(model_entry) = model_entry_res else {
                continue;
            };
            let mpath = model_entry.path();
            if !mpath.is_dir() {
                continue;
            }
            let model_name_str = model_entry.file_name().to_string_lossy().to_string();
            let Ok(model_name) = ModelName::parse(&model_name_str) else {
                continue;
            };

            let Ok(variant_entries) = fs::read_dir(&mpath) else {
                continue;
            };

            for v_entry_res in variant_entries {
                let Ok(v_entry) = v_entry_res else {
                    continue;
                };
                let vpath = v_entry.path();
                if !vpath.is_dir() {
                    continue;
                }
                let vname = v_entry.file_name().to_string_lossy().to_string();
                if vname.starts_with('.') {
                    continue;
                }
                let Ok(variant_key) = VariantKey::parse(&vname) else {
                    continue;
                };
                let installed_json = vpath.join("installed.json");
                let metadata: Option<InstalledMetadata> = fs::read(&installed_json)
                    .ok()
                    .and_then(|b| serde_json::from_slice(&b).ok());

                let recipe_json = vpath.join("recipe.json");
                let recipe: Option<Recipe> = fs::read(&recipe_json)
                    .ok()
                    .and_then(|b| serde_json::from_slice(&b).ok());

                if let Some(meta) = metadata {
                    let mut roles: Vec<Role> = meta.files.iter().map(|f| f.role).collect();
                    roles.sort_unstable();
                    roles.dedup();

                    let disk_bytes: u64 = meta.files.iter().map(|f| f.size).sum();

                    let (backend, quant) = if let Some(r) = &recipe {
                        if let Some(v) = r.variants.iter().find(|v| v.key == variant_key) {
                            (v.backend.clone(), v.quant.clone())
                        } else {
                            ("unknown".to_string(), "unknown".to_string())
                        }
                    } else {
                        ("unknown".to_string(), "unknown".to_string())
                    };

                    list.push(InstalledVariantSummary {
                        model: model_name.clone(),
                        variant: variant_key,
                        backend,
                        quant,
                        roles,
                        disk_bytes,
                        manifest_sha256: meta.manifest_sha256,
                        recipe,
                    });
                }
            }
        }

        Ok(list)
    }

    /// Garbage collects unreferenced blobs and stale partial downloads (§5.7).
    ///
    /// # Errors
    /// Returns [`HubError`] if walking blob directories fails.
    pub fn run_gc(&self) -> Result<GcStats, HubError> {
        let mut stats = GcStats::default();

        // 1. Delete blobs with link count == 1
        let blobs_dir = self.config.home.join("blobs").join("sha256");
        if blobs_dir.is_dir()
            && let Ok(entries) = fs::read_dir(&blobs_dir)
        {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file()
                    && let Ok(meta) = fs::metadata(&path)
                {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::MetadataExt;
                        if meta.nlink() == 1 {
                            stats.reclaimed_bytes += meta.len();
                            stats.deleted_blobs += 1;
                            let _ = fs::remove_file(&path);
                        }
                    }
                }
            }
        }

        // 2. Clean up tmp/*.part older than 7 days without held lock
        let tmp_dir = self.config.home.join("tmp");
        if tmp_dir.is_dir() {
            let seven_days = Duration::from_secs(7 * 24 * 3600);
            let now = SystemTime::now();

            if let Ok(entries) = fs::read_dir(&tmp_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(ext) = path.extension()
                        && ext == "part"
                        && let Ok(meta) = fs::metadata(&path)
                        && let Ok(modified) = meta.modified()
                        && now.duration_since(modified).unwrap_or_default() > seven_days
                    {
                        let lock_path = path.with_extension("lock");
                        let can_delete = if lock_path.is_file() {
                            if let Ok(f) =
                                OpenOptions::new().read(true).write(true).open(&lock_path)
                            {
                                f.try_lock().is_ok()
                            } else {
                                false
                            }
                        } else {
                            true
                        };

                        if can_delete {
                            stats.deleted_parts += 1;
                            let _ = fs::remove_file(&path);
                            let _ = fs::remove_file(&lock_path);
                        }
                    }
                }
            }
        }

        Ok(stats)
    }

    /// Reads authentication token from `~/.shiso/auth.json` checking 0600 mode (§4.4, §8.5).
    ///
    /// # Errors
    /// Returns [`HubError::AuthFile`] if permissions are broader than 0600 on Unix.
    // None is not a refusal: no auth.json file exists on disk.
    pub fn read_auth_token(&self) -> Result<Option<HfToken>, HubError> {
        let auth_path = self.config.home.join("auth.json");
        if !auth_path.is_file() {
            return Ok(None);
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let meta = fs::metadata(&auth_path).map_err(|e| HubError::Io {
                path: auth_path.clone(),
                source: e,
            })?;
            let mode = meta.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                return Err(HubError::AuthFile {
                    path: auth_path,
                    reason: format!(
                        "insecure permissions 0{mode:o} on auth.json; must be 0600 (run 'chmod 600 <path>')"
                    ),
                });
            }
        }

        let content = fs::read_to_string(&auth_path).map_err(|e| HubError::Io {
            path: auth_path.clone(),
            source: e,
        })?;

        if let Ok(doc) = serde_json::from_str::<AuthDoc>(&content) {
            let token_str = doc.token.or(doc.hf_token);
            Ok(token_str.map(HfToken::new))
        } else {
            Ok(None)
        }
    }

    /// Writes `token` to `~/.shiso/auth.json` with 0600 permissions.
    ///
    /// # Errors
    /// Returns [`HubError::Io`] on write failure.
    pub fn write_auth_token(&self, token: &HfToken) -> Result<(), HubError> {
        fs::create_dir_all(&self.config.home).map_err(|e| HubError::Io {
            path: self.config.home.clone(),
            source: e,
        })?;

        let auth_path = self.config.home.join("auth.json");
        let json = serde_json::json!({
            "hf_token": token.as_str()
        });
        let bytes = serde_json::to_vec_pretty(&json)?;

        fs::write(&auth_path, bytes).map_err(|e| HubError::Io {
            path: auth_path.clone(),
            source: e,
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&auth_path, fs::Permissions::from_mode(0o600));
        }

        Ok(())
    }
}

fn current_iso8601() -> String {
    use time::format_description::well_known::Rfc3339;
    time::OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "2026-10-03T12:00:00Z".to_string())
}
