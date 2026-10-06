//! Core editor functions shared between headless CLI and web server (§7).

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256 as Sha256Hasher};

use shiso_hub::recipe::{
    CURRENT_SCHEMA_VERSION, HUB_FILE_MAX_BYTES, Index, IndexModelEntry, IndexVariantEntry,
    ModelName, Provider, RESERVED_MODEL_FILES, Recipe, Sha256, hub_file_name_ok,
};
use shiso_hub::validate::{ValidationMode, validate_recipe};

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
    /// Site files written, relative to the output root.
    pub(crate) site_files: Vec<String>,
    /// Models whose page has no card because none is cached.
    pub(crate) missing_cards: Vec<String>,
    /// Stale output removed (pages and manifests of models that are gone).
    pub(crate) pruned: Vec<String>,
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
                errors.extend(check_hosted_files(repo_path, &recipe));

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

/// Where a build writes, given the repository and what the caller asked for.
///
/// The output is the site root: `index.html` and `models/` at its top, the
/// registry under `registry/v1/`. It belongs to the registry repository,
/// wherever the tool was started from: nothing (or an empty field) is the
/// repository itself, a relative path is relative to the repository, and only
/// an absolute path leaves it.
#[must_use]
pub(crate) fn resolve_out(repo_path: &Path, out: Option<&Path>) -> PathBuf {
    match out {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) if !path.as_os_str().is_empty() => repo_path.join(path),
        _ => repo_path.to_path_buf(),
    }
}

/// The registry base directory under a site root.
pub(crate) fn registry_dir(out_root: &Path) -> PathBuf {
    out_root.join("registry").join("v1")
}

/// Where a hub-hosted file lives under a repository or build root: beside the
/// model's manifest, so the static host serves it from the same place.
pub(crate) fn hosted_file_path(root: &Path, model: &str, name: &str) -> PathBuf {
    registry_dir(root).join("models").join(model).join(name)
}

/// Every problem with a recipe's hub-hosted files as committed in `repo`:
/// missing, a different size, a different digest, or over the hub's limit.
pub(crate) fn check_hosted_files(repo: &Path, recipe: &Recipe) -> Vec<String> {
    let mut errors = Vec::new();
    for variant in &recipe.variants {
        for (role, comp) in &variant.components {
            if comp.provider != Provider::Shiso {
                continue;
            }
            for file in &comp.files {
                let path = hosted_file_path(repo, recipe.name.as_str(), &file.name);
                let what = format!(
                    "variant '{}' role '{role}' hub file '{}'",
                    variant.key.as_str(),
                    file.name
                );
                let Ok(bytes) = fs::read(&path) else {
                    errors.push(format!("{what} is missing at {}", path.display()));
                    continue;
                };
                let size = bytes.len() as u64;
                if size > HUB_FILE_MAX_BYTES {
                    errors.push(format!(
                        "{what} is {size} bytes; the hub hosts at most {HUB_FILE_MAX_BYTES}"
                    ));
                }
                if size != file.size {
                    errors.push(format!(
                        "{what} size is {size} on disk, {} in the recipe",
                        file.size
                    ));
                }
                let digest = hex::encode(Sha256Hasher::digest(&bytes));
                if digest != file.sha256.as_str() {
                    errors.push(format!(
                        "{what} sha256 is {digest} on disk, {} in the recipe",
                        file.sha256.as_str()
                    ));
                }
            }
        }
    }
    errors
}

/// A hosted file as written: what the recipe's file entry needs.
#[derive(Debug, Serialize)]
pub(crate) struct HostedFile {
    name: String,
    size: u64,
    sha256: String,
}

/// Writes `bytes` as model `model`'s hosted file `name`, after the same checks
/// the recipe validation applies to it.
pub(crate) fn write_hosted_file(
    repo: &Path,
    model: &str,
    name: &str,
    bytes: &[u8],
) -> anyhow::Result<HostedFile> {
    ModelName::parse(model).map_err(|e| anyhow::anyhow!("model name '{model}': {e}"))?;
    if !hub_file_name_ok(name) {
        bail!(
            "'{name}' cannot be a hosted file name: one plain name, not index.json or benchmarks.json"
        );
    }
    let size = bytes.len() as u64;
    if size > HUB_FILE_MAX_BYTES {
        bail!("'{name}' is {size} bytes; the hub hosts at most {HUB_FILE_MAX_BYTES}");
    }
    let path = hosted_file_path(repo, model, name);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(&path, bytes)?;
    Ok(HostedFile {
        name: name.to_string(),
        size,
        sha256: hex::encode(Sha256Hasher::digest(bytes)),
    })
}

/// Copies every recipe's hosted files from the repository to a build root
/// that is somewhere else; with the default root they are already in place.
fn copy_hosted_files(repo_path: &Path, out_root: &Path, recipes: &[Recipe]) -> anyhow::Result<()> {
    if fs::canonicalize(repo_path).ok() == fs::canonicalize(out_root).ok() {
        return Ok(());
    }
    for recipe in recipes {
        for comp in recipe.variants.iter().flat_map(|v| v.components.values()) {
            if comp.provider != Provider::Shiso {
                continue;
            }
            for file in &comp.files {
                let from = hosted_file_path(repo_path, recipe.name.as_str(), &file.name);
                let to = hosted_file_path(out_root, recipe.name.as_str(), &file.name);
                if let Some(dir) = to.parent() {
                    fs::create_dir_all(dir)?;
                }
                fs::copy(&from, &to)
                    .with_context(|| format!("copying hosted file {}", from.display()))?;
            }
        }
    }
    Ok(())
}

/// A model's manifest path relative to the registry base, as the index
/// carries it.
pub(crate) fn manifest_rel(name: &ModelName) -> String {
    format!("models/{name}/index.json")
}

/// Every recipe under `recipes/`, sorted by file name.
///
/// # Errors
/// When the directory cannot be read or a recipe does not parse.
pub(crate) fn load_recipes(repo_path: &Path) -> anyhow::Result<Vec<(PathBuf, Recipe)>> {
    let recipes_dir = repo_path.join("recipes");
    let mut files: Vec<_> = fs::read_dir(&recipes_dir)
        .with_context(|| format!("reading {}", recipes_dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|path| {
            let recipe: Recipe = serde_json::from_slice(&fs::read(&path)?)
                .with_context(|| format!("parsing {}", path.display()))?;
            Ok((path, recipe))
        })
        .collect()
}

/// Builds and validates the registry output and the site (§7.1, §7.5).
///
/// Cards are read from the `cards/` cache only; refresh it first with
/// [`crate::card::refresh_cache`] to fetch new ones.
///
/// # Errors
/// Returns an error if validation fails or uncommitted changes exist.
pub(crate) fn build_registry(
    repo_path: &Path,
    out_root: &Path,
    allow_dirty: bool,
) -> anyhow::Result<BuildResult> {
    let status = git_status(repo_path);
    if status.is_dirty && !allow_dirty {
        bail!(
            "Build refused: uncommitted changes exist under recipes/ or in hosted files:\n  {}\nCommit your changes so Git timestamps are accurate, or pass --allow-dirty.",
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

    let mut recipes = Vec::new();
    for (path, mut recipe) in load_recipes(repo_path)? {
        let (published, updated) = get_git_timestamps(repo_path, &path)?;
        recipe.published_at = Some(published);
        recipe.updated_at = Some(updated);
        recipes.push(recipe);
    }
    recipes.sort_by(|a, b| a.name.cmp(&b.name));
    let benchmarks = crate::bench::load_all(repo_path, &recipes)?;

    let registry = registry_dir(out_root);
    let manifest_paths = write_registry(&registry, &recipes, &benchmarks)?;
    copy_hosted_files(repo_path, out_root, &recipes)?;
    let (site_files, missing_cards) = write_site(repo_path, out_root, &recipes, &benchmarks)?;

    let names: Vec<&str> = recipes.iter().map(|r| r.name.as_str()).collect();
    let mut pruned = prune_models_dir(&out_root.join("models"), &names, &["index.html"], false)?;
    pruned.extend(prune_models_dir(
        &registry.join("models"),
        &names,
        &["index.json", "benchmarks.json"],
        true,
    )?);

    let diff = format!(
        "Compiled {} model manifests and {} site files into '{}'",
        manifest_paths.len(),
        site_files.len(),
        out_root.display()
    );

    Ok(BuildResult {
        models_count: manifest_paths.len(),
        manifest_paths,
        diff,
        site_files,
        missing_cards,
        pruned,
    })
}

/// Writes the manifests, their benchmark files and `index.json`.
fn write_registry(
    registry: &Path,
    recipes: &[Recipe],
    benchmarks: &BTreeMap<ModelName, Vec<crate::bench::PublishedRecord>>,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut index_models = Vec::new();
    let mut manifest_paths = Vec::new();

    for recipe in recipes {
        let compiled_bytes = serde_json::to_vec_pretty(recipe)?;
        let manifest_sha = hex::encode(Sha256Hasher::digest(&compiled_bytes));
        let rel = manifest_rel(&recipe.name);
        let dest_file = registry.join(&rel);
        atomic_write(&dest_file, &compiled_bytes)?;
        manifest_paths.push(dest_file);

        let bench_file = registry.join(format!("models/{}/benchmarks.json", recipe.name));
        match benchmarks.get(&recipe.name) {
            Some(records) => {
                let body = serde_json::json!({
                    "schema_version": crate::bench::BENCH_SCHEMA_VERSION,
                    "model": recipe.name,
                    "records": records,
                });
                atomic_write(&bench_file, &serde_json::to_vec_pretty(&body)?)?;
            }
            None if bench_file.is_file() => fs::remove_file(&bench_file)?,
            None => {}
        }

        let variants_summary: Vec<_> = recipe
            .variants
            .iter()
            .map(|v| IndexVariantEntry {
                key: v.key.clone(),
                backends: v.backends.clone(),
                quant: v.quant.clone(),
                min_ram_bytes: v.requirements.min_ram_bytes,
                disk_bytes: v.requirements.disk_bytes,
                min_cuda_sm: v.requires.as_ref().and_then(|r| r.min_cuda_sm),
                min_metal_family: v.requires.as_ref().and_then(|r| r.min_metal_family),
            })
            .collect();

        index_models.push(IndexModelEntry {
            name: recipe.name.clone(),
            aliases: recipe.aliases.clone(),
            family: recipe.family.clone(),
            min_shiso_version: recipe.min_shiso_version.clone(),
            description: recipe.description.clone(),
            tags: recipe.tags.clone(),
            tasks: recipe.tasks.clone(),
            inputs: recipe.modalities.inputs.clone(),
            outputs: recipe.modalities.outputs.clone(),
            updated_at: recipe.updated_at.clone().unwrap_or_default(),
            manifest: rel,
            manifest_sha256: Sha256::parse(&manifest_sha)
                .map_err(|e| anyhow::anyhow!("invalid sha: {e}"))?,
            variants: variants_summary,
        });
    }

    let index = Index {
        schema_version: CURRENT_SCHEMA_VERSION,
        generated_at: current_iso8601(),
        models: index_models,
    };
    atomic_write(
        &registry.join("index.json"),
        &serde_json::to_vec_pretty(&index)?,
    )?;
    Ok(manifest_paths)
}

/// Writes the site, returning its files and the models that have no card.
fn write_site(
    repo_path: &Path,
    out_root: &Path,
    recipes: &[Recipe],
    benchmarks: &BTreeMap<ModelName, Vec<crate::bench::PublishedRecord>>,
) -> anyhow::Result<(Vec<String>, Vec<String>)> {
    let mut missing_cards = Vec::new();
    let mut inputs = Vec::new();
    for recipe in recipes {
        let card = match crate::card::CardRef::of(recipe) {
            Some(r) => crate::card::load(repo_path, &r)?,
            None => None,
        };
        if card.is_none() {
            missing_cards.push(recipe.name.to_string());
        }
        inputs.push(crate::site::ModelInput {
            recipe,
            card,
            benchmarks: benchmarks.get(&recipe.name).map_or(&[], Vec::as_slice),
        });
    }
    let site_files = crate::site::write(out_root, &inputs)?;
    Ok((site_files, missing_cards))
}

/// Removes what a build wrote for models that no longer have a recipe.
///
/// Only an entry the build could have written is removed: a directory named
/// like a model that holds nothing but `owned` files, or a `<name>.json` left
/// by the earlier flat manifest layout (`flat_json`, the registry directory
/// only). Anything else is refused, because the output may be a directory the
/// maintainer keeps other files in.
fn prune_models_dir(
    dir: &Path,
    keep: &[&str],
    owned: &[&str],
    flat_json: bool,
) -> anyhow::Result<Vec<String>> {
    let mut removed = Vec::new();
    if !dir.is_dir() {
        return Ok(removed);
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if path.is_file() {
            if !flat_json {
                continue;
            }
            let stem = file_name.strip_suffix(".json");
            if let Some(stem) = stem
                && ModelName::parse(stem).is_ok()
            {
                fs::remove_file(&path)?;
                removed.push(path.display().to_string());
            }
            continue;
        }
        if keep.contains(&file_name.as_str()) || ModelName::parse(&file_name).is_err() {
            continue;
        }
        let mut foreign = Vec::new();
        for inner in fs::read_dir(&path)? {
            let name = inner?.file_name().to_string_lossy().to_string();
            if !owned.contains(&name.as_str()) {
                foreign.push(name);
            }
        }
        if !foreign.is_empty() {
            bail!(
                "{} belongs to no recipe but holds files the build did not write ({}); remove it by hand",
                path.display(),
                foreign.join(", ")
            );
        }
        fs::remove_dir_all(&path)?;
        removed.push(path.display().to_string());
    }
    Ok(removed)
}

/// Whether a `git status --porcelain=v2` line is about a hosted file: a file
/// under a model's registry directory other than the build's own.
fn touches_hosted_file(line: &str) -> bool {
    line.split_whitespace().last().is_some_and(|path| {
        path.strip_prefix("registry/v1/models/")
            .is_some_and(|rest| {
                rest.rsplit('/')
                    .next()
                    .is_some_and(|name| rest.contains('/') && !RESERVED_MODEL_FILES.contains(&name))
            })
    })
}

/// Runs `git status` to detect branch name and changed recipes (§7.2).
#[must_use]
pub(crate) fn git_status(repo_path: &Path) -> GitStatusResult {
    let output = Command::new("git")
        .args([
            "status",
            "--porcelain=v2",
            "--branch",
            "--untracked-files=all",
        ])
        .current_dir(repo_path)
        .output();

    let mut branch = String::from("unknown");
    let mut changed = Vec::new();

    if let Ok(out) = output {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if let Some(b) = line.strip_prefix("# branch.head ") {
                branch = b.trim().to_string();
            } else if line.contains("recipes/") || touches_hosted_file(line) {
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

/// A recipe's first and latest commit times, read from the registry
/// repository's own history.
///
/// Run inside `repo_path`, never the caller's working directory: started from
/// another repository, `git log` there knows nothing of the recipe and every
/// manifest was stamped with the build time. A recipe with no commits yet -
/// only buildable with `--allow-dirty` - is stamped now.
///
/// # Errors
/// When git cannot read the repository's history at all.
fn get_git_timestamps(repo_path: &Path, path: &Path) -> anyhow::Result<(String, String)> {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(repo_path)
            .output()
            .with_context(|| format!("running git in {}", repo_path.display()))
    };
    let tree = git(&["rev-parse", "--is-inside-work-tree"])?;
    if !tree.status.success() {
        bail!(
            "{} is not a git repository; the build reads publication dates from its history: {}",
            repo_path.display(),
            String::from_utf8_lossy(&tree.stderr).trim()
        );
    }
    let now = current_iso8601();
    // A repository with no commit yet has no history to read.
    if !git(&["rev-parse", "--verify", "-q", "HEAD"])?
        .status
        .success()
    {
        return Ok((now.clone(), now));
    }
    let out = Command::new("git")
        .args(["log", "--follow", "--format=%cI", "--"])
        .arg(path)
        .current_dir(repo_path)
        .output()
        .with_context(|| format!("running git in {}", repo_path.display()))?;
    if !out.status.success() {
        bail!(
            "git cannot read the history of {}: {}",
            repo_path.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<_> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if let (Some(updated), Some(published)) = (lines.first(), lines.last()) {
        return Ok(((*published).to_string(), (*updated).to_string()));
    }
    Ok((now.clone(), now))
}

pub(crate) fn atomic_write(dest: &Path, bytes: &[u8]) -> anyhow::Result<()> {
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

pub(crate) fn current_iso8601() -> String {
    use time::format_description::well_known::Rfc3339;
    time::OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "2026-10-03T12:00:00Z".to_string())
}
