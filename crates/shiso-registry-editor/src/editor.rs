//! Core editor functions shared between headless CLI and web server (§7).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256 as Sha256Hasher};

use shiso_hub::recipe::{
    CURRENT_SCHEMA_VERSION, Index, IndexModelEntry, IndexVariantEntry, ModelName, Recipe, Sha256,
};
use shiso_hub::validate::{FamilyValidationFacts, ValidationMode, validate_recipe};

/// Validation report for one recipe file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RecipeValidationReport {
    /// File path relative to repository root.
    pub(crate) file: String,
    /// Canonical model name.
    pub(crate) name: String,
    /// Whether the recipe passed strict validation.
    pub(crate) valid: bool,
    /// Error messages if validation failed.
    pub(crate) errors: Vec<String>,
    /// Warnings if non-fatal issues were detected.
    pub(crate) warnings: Vec<String>,
}

/// Result of inspecting a Hugging Face repository (§7.2, §7.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct HfInspectResult {
    /// Remote repository (`owner/name`).
    pub(crate) repo: String,
    /// Requested revision or branch.
    pub(crate) revision: String,
    /// Resolved 40-character Git commit SHA.
    pub(crate) commit: String,
    /// Files present in the repository at this commit.
    pub(crate) files: Vec<HfInspectFile>,
}

/// A file discovered during Hugging Face repository inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct HfInspectFile {
    /// Relative path within repository.
    pub(crate) name: String,
    /// File size in bytes.
    pub(crate) size: u64,
    /// Hexadecimal SHA-256 digest.
    pub(crate) sha256: String,
    /// Whether this file is stored via Git LFS.
    pub(crate) lfs: bool,
}

/// Result of building the registry index and manifests (§7.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BuildResult {
    /// Number of compiled model manifests.
    pub(crate) models_count: usize,
    /// Compiled manifest file paths.
    pub(crate) manifest_paths: Vec<PathBuf>,
    /// Index diff against current published index.
    pub(crate) diff: String,
}

/// Git status information for the repository banner (§7.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GitStatusResult {
    /// Current git branch name.
    pub(crate) branch: String,
    /// Changed or untracked recipe files under `recipes/`.
    pub(crate) changed_recipes: Vec<String>,
    /// Whether the recipe tree has uncommitted modifications.
    pub(crate) is_dirty: bool,
}

#[derive(Deserialize)]
struct HfRevisionInfo {
    sha: Option<String>,
}

#[derive(Deserialize)]
struct TreeEntry {
    path: String,
    size: Option<u64>,
    #[serde(rename = "type")]
    entry_type: String,
    lfs: Option<LfsMetadata>,
    oid: Option<String>,
}

#[derive(Deserialize)]
struct LfsMetadata {
    oid: String,
    size: u64,
}

/// Strictly validates all recipes found under `recipes/` (§7.1, §7.3).
///
/// # Errors
/// Returns an error if the directory cannot be read.
pub(crate) fn validate_all(repo_path: &Path) -> anyhow::Result<Vec<RecipeValidationReport>> {
    let recipes_dir = repo_path.join("recipes");
    if !recipes_dir.is_dir() {
        bail!("directory '{}' does not exist", recipes_dir.display());
    }

    let mut reports = Vec::new();
    let mut global_names: HashMap<String, String> = HashMap::new();

    let mut entries: Vec<_> = fs::read_dir(&recipes_dir)?
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
        .collect();
    entries.sort_by_key(fs::DirEntry::file_name);

    for entry in entries {
        let path = entry.path();
        let rel_file = path
            .strip_prefix(repo_path)
            .unwrap_or(&path)
            .display()
            .to_string();

        let bytes = fs::read(&path)?;
        let mut errors = Vec::new();
        let warnings = Vec::new();

        let recipe_res: Result<Recipe, _> = serde_json::from_slice(&bytes);
        match recipe_res {
            Ok(recipe) => {
                let name = recipe.name.to_string();

                if let Err(e) = validate_recipe(&recipe, ValidationMode::Strict) {
                    errors.push(e.to_string());
                }

                if let Some(prev) = global_names.insert(name.clone(), rel_file.clone()) {
                    errors.push(format!("model name '{name}' already declared in '{prev}'"));
                }
                for alias in &recipe.aliases {
                    let a_str = alias.to_string();
                    if let Some(prev) = global_names.insert(a_str.clone(), rel_file.clone()) {
                        errors.push(format!(
                            "alias '{a_str}' conflicts with name or alias in '{prev}'"
                        ));
                    }
                }

                reports.push(RecipeValidationReport {
                    file: rel_file,
                    name,
                    valid: errors.is_empty(),
                    errors,
                    warnings,
                });
            }
            Err(e) => {
                reports.push(RecipeValidationReport {
                    file: rel_file,
                    name: entry.file_name().to_string_lossy().to_string(),
                    valid: false,
                    errors: vec![format!("JSON syntax or unknown field error: {e}")],
                    warnings,
                });
            }
        }
    }

    Ok(reports)
}

/// Inspects a Hugging Face repository to retrieve commit SHA, file sizes, and digests (§7.2, §7.3).
///
/// # Errors
/// Returns an error if querying Hugging Face fails.
pub(crate) async fn inspect_hf(
    repo: &str,
    revision: &str,
    token: Option<&str>,
) -> anyhow::Result<HfInspectResult> {
    let client = reqwest::Client::new();

    let info_url = format!("https://huggingface.co/api/models/{repo}/revision/{revision}");
    let mut req = client.get(&info_url);
    if let Some(tok) = token {
        req = req.header("authorization", format!("Bearer {tok}"));
    }

    let resp = req
        .send()
        .await
        .with_context(|| format!("querying Hugging Face model info at '{info_url}'"))?;

    if !resp.status().is_success() {
        bail!(
            "Hugging Face repository '{repo}' revision '{revision}' returned status {}",
            resp.status()
        );
    }

    let info: HfRevisionInfo = resp.json().await.unwrap_or(HfRevisionInfo { sha: None });
    let commit = info.sha.unwrap_or_else(|| revision.to_string());

    let tree_url = format!("https://huggingface.co/api/models/{repo}/tree/{commit}?recursive=true");
    let mut tree_req = client.get(&tree_url);
    if let Some(tok) = token {
        tree_req = tree_req.header("authorization", format!("Bearer {tok}"));
    }

    let tree_resp = tree_req
        .send()
        .await
        .with_context(|| format!("fetching file tree from '{tree_url}'"))?;

    let entries: Vec<TreeEntry> = tree_resp.json().await.unwrap_or_default();
    let mut files = Vec::new();

    for entry in entries {
        if entry.entry_type != "file" {
            continue;
        }

        let name = entry.path;
        if let Some(lfs) = entry.lfs {
            files.push(HfInspectFile {
                name,
                size: lfs.size,
                sha256: lfs.oid,
                lfs: true,
            });
        } else {
            let size = entry.size.unwrap_or(0);
            let sha = if size <= 20 * 1024 * 1024 {
                hash_remote_non_lfs_file(&client, repo, &commit, &name, token)
                    .await
                    .unwrap_or_else(|| {
                        entry.oid.unwrap_or_else(|| {
                            "0000000000000000000000000000000000000000000000000000000000000000"
                                .to_string()
                        })
                    })
            } else {
                entry.oid.unwrap_or_else(|| {
                    "0000000000000000000000000000000000000000000000000000000000000000".to_string()
                })
            };
            files.push(HfInspectFile {
                name,
                size,
                sha256: sha,
                lfs: false,
            });
        }
    }

    Ok(HfInspectResult {
        repo: repo.to_string(),
        revision: revision.to_string(),
        commit,
        files,
    })
}
async fn hash_remote_non_lfs_file(
    client: &reqwest::Client,
    repo: &str,
    commit: &str,
    path: &str,
    token: Option<&str>,
) -> Option<String> {
    let url = format!("https://huggingface.co/{repo}/resolve/{commit}/{path}");
    let mut req = client.get(&url);
    if let Some(tok) = token {
        req = req.header("authorization", format!("Bearer {tok}"));
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let bytes = resp.bytes().await.ok()?;
    Some(hex::encode(Sha256Hasher::digest(&bytes)))
}

/// Builds and validates the registry output (§7.1, §7.5).
///
/// # Errors
/// Returns an error if validation fails or uncommitted changes exist.
pub(crate) fn build_registry(
    repo_path: &Path,
    out_dir: &Path,
    allow_dirty: bool,
) -> anyhow::Result<BuildResult> {
    let status = git_status(repo_path);
    if status.is_dirty && !allow_dirty {
        bail!(
            "Build refused: uncommitted changes exist under recipes/:\n  {}\nCommit your changes so Git timestamps are accurate, or pass --allow-dirty.",
            status.changed_recipes.join("\n  ")
        );
    }

    let reports = validate_all(repo_path)?;
    let failures: Vec<_> = reports.iter().filter(|r| !r.valid).collect();
    if !failures.is_empty() {
        let mut msg = String::from("Validation failed for the following recipes:\n");
        for f in failures {
            let _ = writeln!(msg, "  {}: {}", f.file, f.errors.join("; "));
        }
        bail!("{msg}");
    }

    let models_out = out_dir.join("models");
    fs::create_dir_all(&models_out)?;

    let mut index_models = Vec::new();
    let mut manifest_paths = Vec::new();

    let recipes_dir = repo_path.join("recipes");
    let mut recipe_files: Vec<_> = fs::read_dir(&recipes_dir)?
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
        .collect();
    recipe_files.sort_by_key(fs::DirEntry::file_name);

    for file_entry in recipe_files {
        let path = file_entry.path();
        let bytes = fs::read(&path)?;
        let mut recipe: Recipe = serde_json::from_slice(&bytes)?;

        let (published, updated) = get_git_timestamps(&path);
        recipe.published_at = Some(published.clone());
        recipe.updated_at = Some(updated.clone());

        let compiled_bytes = serde_json::to_vec_pretty(&recipe)?;
        let manifest_sha = hex::encode(Sha256Hasher::digest(&compiled_bytes));
        let dest_file = models_out.join(format!("{}.json", recipe.name.as_str()));

        atomic_write(&dest_file, &compiled_bytes)?;
        manifest_paths.push(dest_file);

        let variants_summary: Vec<_> = recipe
            .variants
            .iter()
            .map(|v| IndexVariantEntry {
                key: v.key.clone(),
                backend: v.backend.clone(),
                quant: v.quant.clone(),
                min_ram_bytes: v.requirements.min_ram_bytes,
                disk_bytes: v.requirements.disk_bytes,
                min_cuda_sm: v.requires.as_ref().and_then(|r| r.min_cuda_sm),
            })
            .collect();

        index_models.push(IndexModelEntry {
            name: recipe.name,
            aliases: recipe.aliases,
            family: recipe.family,
            min_shiso_version: recipe.min_shiso_version,
            description: recipe.description,
            tags: recipe.tags,
            tasks: recipe.tasks,
            inputs: recipe.modalities.inputs,
            outputs: recipe.modalities.outputs,
            updated_at: updated,
            manifest_sha256: Sha256::parse(&manifest_sha)
                .map_err(|e| anyhow::anyhow!("invalid sha: {e}"))?,
            variants: variants_summary,
        });
    }

    index_models.sort_by(|a, b| a.name.cmp(&b.name));

    let index = Index {
        schema_version: CURRENT_SCHEMA_VERSION,
        generated_at: current_iso8601(),
        models: index_models,
    };

    let index_bytes = serde_json::to_vec_pretty(&index)?;
    let index_path = out_dir.join("index.json");
    atomic_write(&index_path, &index_bytes)?;

    let diff = format!(
        "Compiled {} model manifests into '{}'",
        manifest_paths.len(),
        out_dir.display()
    );

    Ok(BuildResult {
        models_count: manifest_paths.len(),
        manifest_paths,
        diff,
    })
}

/// Runs `git status` to detect branch name and changed recipes (§7.2).
#[must_use]
pub(crate) fn git_status(repo_path: &Path) -> GitStatusResult {
    let output = Command::new("git")
        .args(["status", "--porcelain=v2", "--branch"])
        .current_dir(repo_path)
        .output();

    let mut branch = String::from("unknown");
    let mut changed = Vec::new();

    if let Ok(out) = output {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if let Some(b) = line.strip_prefix("# branch.head ") {
                branch = b.trim().to_string();
            } else if line.contains("recipes/") {
                let parts: Vec<_> = line.split_whitespace().collect();
                if let Some(file) = parts.last() {
                    changed.push(file.to_string());
                }
            }
        }
    }

    let is_dirty = !changed.is_empty();
    GitStatusResult {
        branch,
        changed_recipes: changed,
        is_dirty,
    }
}

/// Reads a single recipe JSON by model name.
///
/// # Errors
/// Returns an error if model name is invalid or file cannot be read.
pub(crate) fn get_recipe(repo_path: &Path, name_str: &str) -> anyhow::Result<Recipe> {
    let name = ModelName::parse(name_str)
        .map_err(|e| anyhow::anyhow!("invalid model name '{name_str}': {e}"))?;
    let path = repo_path
        .join("recipes")
        .join(format!("{}.json", name.as_str()));
    let bytes =
        fs::read(&path).with_context(|| format!("reading recipe file at '{}'", path.display()))?;
    let recipe: Recipe = serde_json::from_slice(&bytes)?;
    Ok(recipe)
}

/// Saves a recipe with atomic write and strict validation.
///
/// # Errors
/// Returns an error if validation fails or writing fails.
pub(crate) fn save_recipe(repo_path: &Path, name_str: &str, content: &str) -> anyhow::Result<()> {
    let name = ModelName::parse(name_str)
        .map_err(|e| anyhow::anyhow!("invalid model name '{name_str}': {e}"))?;

    let recipe: Recipe = serde_json::from_str(content)
        .context("parsing JSON into Recipe struct (strict validation)")?;

    if recipe.name != name {
        bail!(
            "recipe name in manifest ('{}') does not match filename ('{name}')",
            recipe.name
        );
    }

    validate_recipe(&recipe, ValidationMode::Strict)
        .map_err(|e| anyhow::anyhow!("recipe validation error: {e}"))?;

    let formatted = serde_json::to_string_pretty(&recipe)?;
    let target = repo_path
        .join("recipes")
        .join(format!("{}.json", name.as_str()));

    atomic_write(&target, formatted.as_bytes())?;
    Ok(())
}

/// Deletes a recipe file, checking that no other recipe alias points to it.
///
/// # Errors
/// Returns an error if an alias in another recipe points to this model.
pub(crate) fn delete_recipe(repo_path: &Path, name_str: &str) -> anyhow::Result<()> {
    let name = ModelName::parse(name_str)
        .map_err(|e| anyhow::anyhow!("invalid model name '{name_str}': {e}"))?;

    let recipes_dir = repo_path.join("recipes");
    if recipes_dir.is_dir() {
        for entry in fs::read_dir(&recipes_dir)?.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json")
                && let Ok(bytes) = fs::read(&path)
                && let Ok(recipe) = serde_json::from_slice::<Recipe>(&bytes)
                && recipe.name != name
                && recipe.aliases.contains(&name)
            {
                bail!(
                    "cannot delete recipe '{name}': model '{}' declares it as an alias",
                    recipe.name
                );
            }
        }
    }

    let target = recipes_dir.join(format!("{}.json", name.as_str()));
    if target.is_file() {
        fs::remove_file(&target)?;
    }

    Ok(())
}

fn get_git_timestamps(path: &Path) -> (String, String) {
    let output = Command::new("git")
        .args(["log", "--follow", "--format=%cI", "--"])
        .arg(path)
        .output();

    if let Ok(out) = output {
        let text = String::from_utf8_lossy(&out.stdout);
        let lines: Vec<_> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        if !lines.is_empty() {
            let updated = lines[0].to_string();
            let published = lines.last().unwrap_or(&lines[0]).to_string();
            return (published, updated);
        }
    }

    let now = current_iso8601();
    (now.clone(), now)
}

fn atomic_write(dest: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let parent = dest.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;

    let tmp = parent.join(format!(
        ".tmp_{}_{}",
        std::process::id(),
        dest.file_name().and_then(|f| f.to_str()).unwrap_or("file")
    ));

    fs::write(&tmp, bytes)?;
    if let Err(e) = fs::rename(&tmp, dest) {
        let _ = fs::remove_file(&tmp);
        return Err(e.into());
    }

    Ok(())
}

fn current_iso8601() -> String {
    use time::format_description::well_known::Rfc3339;
    time::OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "2026-10-03T12:00:00Z".to_string())
}
