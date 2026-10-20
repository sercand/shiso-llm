//! Hub integration for `shiso-server`: model resolution, pull, list, info, verify, rm (§6).

#![expect(
    clippy::print_stdout,
    reason = "CLI command output intended for operator console or NDJSON stream"
)]

use std::fs::File;
use std::io::IsTerminal;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle as IndicatifStyle};
use tokio::sync::mpsc;
use url::Url;

use shiso_hub::config::{HfToken, HubConfig, RegistryEndpoint};
use shiso_hub::download::Downloader;
use shiso_hub::host::Backend;
use shiso_hub::host::HostProfile;
use shiso_hub::progress::ProgressEvent;
use shiso_hub::recipe::{ModelName, ModelRef, Role, Variant};
use shiso_hub::registry::RegistryClient;
use shiso_hub::select::select_variant;
use shiso_hub::store::{BlobStore, InstalledVariantSummary};

use crate::cli::{
    HubArgs, InfoArgs, ListArgs, ModelArgKind, ProgressStyle, PullArgs, RmArgs, ServeArgs,
    VerifyArgs, apply_recipe_defaults, classify_model_arg,
};

/// Builds a `HostProfile` based on compile-time features and runtime RAM probing (§5.2).
#[must_use]
pub fn probe_host_profile() -> HostProfile {
    let backends = vec![
        #[cfg(feature = "metal")]
        Backend::Metal,
        #[cfg(feature = "cuda")]
        Backend::Cuda,
    ];

    #[cfg(feature = "cuda")]
    let cuda = {
        if shiso_cuda::cuda_available() {
            Some(shiso_hub::host::CudaInfo {
                sm: 89,
                vram_bytes: 16 * 1024 * 1024 * 1024,
            })
        } else {
            None
        }
    };
    #[cfg(not(feature = "cuda"))]
    let cuda = None;

    #[cfg(feature = "metal")]
    let metal = shiso_metal::device_facts().map(|facts| shiso_hub::host::MetalInfo {
        name: facts.name,
        family: facts.apple_family,
        working_set_bytes: facts.working_set_bytes,
    });
    #[cfg(not(feature = "metal"))]
    let metal = None;

    HostProfile::probe(backends, cuda, metal)
}

/// Resolves `home` directory from string, expanding `~`.
#[must_use]
pub fn resolve_home(home_str: &str) -> PathBuf {
    if (home_str.starts_with("~/") || home_str == "~")
        && let Some(user_home) = std::env::var_os("HOME")
    {
        let mut path = PathBuf::from(user_home);
        if home_str.len() > 1 {
            path.push(&home_str[2..]);
        }
        return path;
    }
    PathBuf::from(home_str)
}

/// Constructs a [`HubConfig`] from [`HubArgs`].
///
/// # Errors
/// Returns an error if registry URL or Hugging Face endpoint is invalid, or custom registry key is missing.
pub fn build_hub_config(args: &HubArgs) -> anyhow::Result<HubConfig> {
    let home = resolve_home(&args.home);

    // Any URL is accepted without key material; trust rests on HTTPS plus the
    // per-file digests the manifest pins (§4.3).
    let registry = RegistryEndpoint::parse(&args.registry_url)
        .map_err(|e| anyhow::anyhow!("invalid registry URL: {e}"))?;

    let hf_endpoint = Url::parse(&args.hf_endpoint)
        .with_context(|| format!("invalid Hugging Face endpoint URL '{}'", args.hf_endpoint))?;

    let max_parallel_files = NonZeroUsize::new(args.jobs.max(1)).unwrap_or(NonZeroUsize::MIN);

    let mut config = HubConfig {
        home,
        registry,
        hf_endpoint,
        hf_token: args.hf_token.as_ref().map(HfToken::new),
        max_parallel_files,
    };

    if config.hf_token.is_none() {
        let store = BlobStore::new(config.clone());
        if let Ok(Some(tok)) = store.read_auth_token() {
            config.hf_token = Some(tok);
        }
    }

    Ok(config)
}

/// Resolves a positional model argument on `ServeArgs` (§6.6).
///
/// Returns an open shared lock file guard if a registry model was resolved.
///
/// # Errors
/// Returns an error if component flags conflict with a model reference, the model is not installed,
/// or opening the variant fails.
// None is not a refusal: a direct filesystem path was provided or no model was specified, so no shared lock was taken.
pub fn resolve_serve_model(
    args: &mut ServeArgs,
    serve_matches: Option<&clap::ArgMatches>,
) -> anyhow::Result<Option<File>> {
    let Some(model_ref_str) = &args.model_ref else {
        return Ok(None);
    };

    match classify_model_arg(model_ref_str) {
        ModelArgKind::Path(path) => {
            args.model = Some(path);
            Ok(None)
        }
        ModelArgKind::RegistryRef(ref_str) => {
            check_component_flag_conflicts(args, &ref_str)?;

            let parsed_ref = ModelRef::parse(&ref_str)
                .map_err(|e| anyhow::anyhow!("invalid model reference '{ref_str}': {e}"))?;

            let default_home = HubConfig::default_home();
            let config = HubConfig::with_defaults(default_home);
            let store = BlobStore::new(config);

            let installed = store.list_installed()?;
            let chosen = find_installed_serve_variant(&installed, &parsed_ref, &ref_str)?;

            let variant_key = &chosen.variant;
            let vdir = store.variant_dir(&parsed_ref.model, variant_key);

            let lock_file = store.open_variant_lock(&parsed_ref.model, variant_key)?;
            lock_file
                .lock_shared()
                .with_context(|| format!("locking variant {variant_key}"))?;

            let recipe = chosen
                .recipe
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("installed recipe.json missing"))?;
            let variant = recipe
                .variants
                .iter()
                .find(|v| &v.key == variant_key)
                .ok_or_else(|| anyhow::anyhow!("variant not in recipe"))?;

            populate_serve_args_from_variant(args, &vdir, variant, &recipe.name)?;

            if let Some(defaults) = &variant.server_defaults {
                apply_recipe_defaults(args, serve_matches, defaults);
            }

            Ok(Some(lock_file))
        }
    }
}

fn check_component_flag_conflicts(args: &ServeArgs, ref_str: &str) -> anyhow::Result<()> {
    if args.config.is_some()
        || args.ngram.is_some()
        || args.vision.is_some()
        || args.draft_model.is_some()
        || args.draft_vocab.is_some()
        || args.asr_model.is_some()
        || args.asr_mmproj.is_some()
        || args.diarization_model.is_some()
        || args.image_model.is_some()
        || args.image_assets_dir.is_some()
        || args.image_text_encoder.is_some()
        || args.image_vae.is_some()
        || args.image_tokenizer.is_some()
    {
        bail!("cannot pass component flags together with model reference '{ref_str}'");
    }
    Ok(())
}

fn find_installed_serve_variant<'a>(
    installed: &'a [InstalledVariantSummary],
    parsed_ref: &ModelRef,
    ref_str: &str,
) -> anyhow::Result<&'a InstalledVariantSummary> {
    let matching_variants: Vec<_> = installed
        .iter()
        .filter(|iv| iv.model == parsed_ref.model)
        .collect();

    if matching_variants.is_empty() {
        bail!(
            "Error: model '{}' is not installed. Run 'shiso pull {}' first.",
            parsed_ref.model,
            ref_str
        );
    }

    if let Some(target_key) = &parsed_ref.variant {
        matching_variants
            .into_iter()
            .find(|iv| &iv.variant == target_key)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Error: variant '{target_key}' for model '{}' is not installed.",
                    parsed_ref.model
                )
            })
    } else {
        let host = probe_host_profile();
        let recipe = matching_variants
            .first()
            .and_then(|iv| iv.recipe.as_ref())
            .ok_or_else(|| anyhow::anyhow!("could not load recipe for installed model"))?;

        let selected = select_variant(recipe, &host, None)
            .map_err(|e| anyhow::anyhow!("no compatible installed variant for this host: {e}"))?;

        matching_variants
            .into_iter()
            .find(|iv| iv.variant == selected.variant.key)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Error: variant '{}' is not installed.",
                    selected.variant.key
                )
            })
    }
}

fn populate_serve_args_from_variant(
    args: &mut ServeArgs,
    vdir: &Path,
    variant: &Variant,
    recipe_name: &ModelName,
) -> anyhow::Result<()> {
    // A role is supplied by its own component or by a file another component
    // binds to it; `role_paths` folds both into one answer per flag.
    let roles = variant
        .role_paths()
        .map_err(|e| anyhow::anyhow!("installed recipe is inconsistent: {e}"))?;
    for (role, place) in roles {
        let role_dir = vdir.join(place.dir.as_str());
        let target_path = match place.file {
            Some(file) => role_dir.join(file),
            None => role_dir,
        };

        match role {
            Role::Base => args.model = Some(target_path),
            Role::Draft => args.draft_model = Some(target_path),
            Role::DraftVocab => args.draft_vocab = Some(target_path),
            Role::Vision => args.vision = Some(target_path),
            Role::Asr => args.asr_model = Some(target_path),
            Role::AsrMmproj => args.asr_mmproj = Some(target_path),
            Role::Diarization => args.diarization_model = Some(target_path),
            Role::Config => args.config = Some(target_path),
            Role::Ngram => args.ngram = Some(target_path),
            Role::Image => args.image_model = Some(target_path),
            Role::ImageAssets => args.image_assets_dir = Some(target_path),
            Role::ImageTextEncoder => args.image_text_encoder = Some(target_path),
            Role::ImageVae => args.image_vae = Some(target_path),
            Role::ImageTokenizer => args.image_tokenizer = Some(target_path),
        }
    }

    if args.image_model.is_some() && args.image_model_id.is_none() {
        args.image_model_id = Some(recipe_name.to_string());
    }

    if args.model_name.is_none() {
        args.model_name = Some(recipe_name.to_string());
    }

    Ok(())
}

/// The backends a variant names, as `metal/cuda`.
fn backends_label(backends: &[Backend]) -> String {
    if backends.is_empty() {
        return "unknown".to_string();
    }
    let names: Vec<&str> = backends.iter().map(|b| b.as_str()).collect();
    names.join("/")
}

/// Executes the `shiso pull <ref>` command (§6.1).
///
/// # Errors
/// Returns an error if resolution, download, or installation fails.
pub async fn run_pull(args: PullArgs) -> anyhow::Result<()> {
    let config = build_hub_config(&args.hub)?;
    let parsed_ref = ModelRef::parse(&args.model_ref)
        .map_err(|e| anyhow::anyhow!("invalid model reference '{}': {e}", args.model_ref))?;

    let client = RegistryClient::new(config.clone());
    let downloader = Downloader::new(config.clone());
    let store = BlobStore::new(config);

    println!("Fetching registry index...");
    let index = client.fetch_index().await?;

    let model_entry = client
        .resolve_model(&index, parsed_ref.model.as_str())
        .ok_or_else(|| anyhow::anyhow!("model '{}' not found in registry", parsed_ref.model))?;

    let current_version = semver::Version::parse(env!("CARGO_PKG_VERSION"))
        .unwrap_or_else(|_| semver::Version::new(0, 1, 0));
    let recipe = client.fetch_manifest(model_entry, &current_version).await?;

    let host = probe_host_profile();
    let selected = select_variant(&recipe, &host, parsed_ref.variant.as_ref())?;

    println!("{}", selected.explanation);
    if let Some(warn) = selected.warning {
        println!("{warn}");
    }

    if store.is_up_to_date(
        &recipe.name,
        &selected.variant.key,
        &model_entry.manifest_sha256,
        selected.variant,
    ) {
        println!(
            "Model '{}:{}' is already up to date.",
            recipe.name, selected.variant.key
        );
        return Ok(());
    }

    let (tx, mut rx) = mpsc::channel::<ProgressEvent>(64);
    let progress_style = args.hub.progress;
    let is_tty = std::io::stdout().is_terminal();

    let render_handle = tokio::spawn(async move {
        match progress_style {
            ProgressStyle::Json => {
                while let Some(event) = rx.recv().await {
                    if let Ok(json) = serde_json::to_string(&event) {
                        println!("{json}");
                    }
                }
            }
            ProgressStyle::Bars | ProgressStyle::Auto if is_tty => {
                render_bars(&mut rx).await;
            }
            _ => {
                while let Some(event) = rx.recv().await {
                    match event {
                        ProgressEvent::Resolved { total, .. } => {
                            println!("Downloading {total} bytes across components...");
                        }
                        ProgressEvent::FileStarted {
                            role, file, total, ..
                        } => {
                            println!("  [{role}] {file} started ({total} bytes)");
                        }
                        ProgressEvent::FileVerified { role, file, .. } => {
                            println!("  [{role}] {file} verified");
                        }
                        ProgressEvent::Installed { model, variant } => {
                            println!("Installed {model}:{variant}");
                        }
                        ProgressEvent::Error { message, .. } => {
                            println!("Error: {message}");
                        }
                        ProgressEvent::FileProgress { .. } => {}
                    }
                }
            }
        }
    });

    downloader
        .download_variant(recipe.name.as_ref(), selected.variant, Some(&tx))
        .await?;

    store.install_variant(&recipe, selected.variant, &model_entry.manifest_sha256)?;

    let _ = tx
        .send(ProgressEvent::Installed {
            model: recipe.name.to_string(),
            variant: selected.variant.key.to_string(),
        })
        .await;
    drop(tx);
    let _ = render_handle.await;

    println!(
        "Successfully pulled and installed '{}:{}'.",
        recipe.name, selected.variant.key
    );
    Ok(())
}

async fn render_bars(rx: &mut mpsc::Receiver<ProgressEvent>) {
    let mp = MultiProgress::new();
    let style = IndicatifStyle::default_bar()
        .template(
            "{prefix:.bold} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
        )
        .unwrap_or_else(|_| IndicatifStyle::default_bar());

    let mut bars = std::collections::HashMap::new();

    while let Some(event) = rx.recv().await {
        match event {
            ProgressEvent::FileStarted {
                role, file, total, ..
            } => {
                let pb = mp.add(ProgressBar::new(total));
                pb.set_style(style.clone());
                pb.set_prefix(format!("{role}/{file}"));
                bars.insert(file, pb);
            }
            ProgressEvent::FileProgress { file, bytes, .. } => {
                if let Some(pb) = bars.get(&file) {
                    pb.set_position(bytes);
                }
            }
            ProgressEvent::FileVerified { file, .. } => {
                if let Some(pb) = bars.get(&file) {
                    pb.finish_with_message("verified");
                }
            }
            _ => {}
        }
    }
}

/// Executes the `shiso list [--remote] [--task <t>] [--tag <t>]` command (§6.2).
///
/// # Errors
/// Returns an error if querying registry or reading local storage fails.
pub async fn run_list(args: ListArgs) -> anyhow::Result<()> {
    let config = build_hub_config(&args.hub)?;

    if args.remote {
        let client = RegistryClient::new(config.clone());
        let store = BlobStore::new(config);
        let host = probe_host_profile();

        println!("Querying remote registry...");
        let index = client.fetch_index().await?;
        let installed = store.list_installed().unwrap_or_default();

        println!(
            "{:<24} {:<12} {:<10} {:<8} {:<10} DESCRIPTION",
            "MODEL", "VARIANT", "BACKEND", "QUANT", "STATUS"
        );
        println!("{}", "-".repeat(90));

        for model in index.models {
            if let Some(filter_task) = &args.task
                && !model.tasks.iter().any(|t| t.as_str() == filter_task)
            {
                continue;
            }
            if let Some(filter_tag) = &args.tag
                && !model.tags.iter().any(|t| t == filter_tag)
            {
                continue;
            }

            for v in model.variants {
                let is_inst = installed
                    .iter()
                    .any(|iv| iv.model == model.name && iv.variant == v.key);
                let fits_host = v.backends.iter().any(|b| host.backends.contains(b));

                let status = match (is_inst, fits_host) {
                    (true, _) => "installed",
                    (false, true) => "available",
                    (false, false) => "unsupported",
                };

                println!(
                    "{:<24} {:<12} {:<10} {:<8} {:<10} {}",
                    model.name.as_str(),
                    v.key.as_str(),
                    backends_label(&v.backends),
                    v.quant,
                    status,
                    model.description
                );
            }
        }
    } else {
        let store = BlobStore::new(config);
        let list = store.list_installed()?;

        if list.is_empty() {
            println!("No models installed. Run 'shiso pull <model>' to install one.");
            return Ok(());
        }

        println!(
            "{:<32} {:<10} {:<10} {:<12} SIZE",
            "MODEL:VARIANT", "BACKEND", "QUANT", "ROLES"
        );
        println!("{}", "-".repeat(75));

        for item in list {
            if let Some(filter_task) = &args.task
                && let Some(recipe) = &item.recipe
                && !recipe.tasks.iter().any(|t| t.as_str() == filter_task)
            {
                continue;
            }
            if let Some(filter_tag) = &args.tag
                && let Some(recipe) = &item.recipe
                && !recipe.tags.iter().any(|t| t == filter_tag)
            {
                continue;
            }

            let roles_str: Vec<_> = item.roles.iter().map(|r| r.as_str()).collect();
            let size_mb = item.disk_bytes / (1024 * 1024);

            println!(
                "{:<32} {:<10} {:<10} {:<12} {size_mb} MB",
                format!("{}:{}", item.model, item.variant),
                backends_label(&item.backends),
                item.quant,
                roles_str.join(",")
            );
        }
    }

    Ok(())
}

/// Executes the `shiso info <ref>` command (§6.3).
///
/// # Errors
/// Returns an error if recipe cannot be found.
pub async fn run_info(args: InfoArgs) -> anyhow::Result<()> {
    let config = build_hub_config(&args.hub)?;
    let parsed_ref = ModelRef::parse(&args.model_ref)
        .map_err(|e| anyhow::anyhow!("invalid model reference '{}': {e}", args.model_ref))?;

    let client = RegistryClient::new(config);

    let recipe = if let Some(cached) = client.read_cached_recipe(&parsed_ref.model) {
        println!("(Showing cached recipe)");
        cached
    } else {
        let index = client.fetch_index().await?;
        let entry = client
            .resolve_model(&index, parsed_ref.model.as_str())
            .ok_or_else(|| anyhow::anyhow!("model '{}' not found in registry", parsed_ref.model))?;
        let current_version = semver::Version::parse(env!("CARGO_PKG_VERSION"))
            .unwrap_or_else(|_| semver::Version::new(0, 1, 0));
        client.fetch_manifest(entry, &current_version).await?
    };

    println!("Model:       {}", recipe.name);
    let aliases: Vec<_> = recipe.aliases.iter().map(ModelName::as_str).collect();
    if !aliases.is_empty() {
        println!("Aliases:     {}", aliases.join(", "));
    }
    println!("Family:      {}", recipe.family);
    println!("Description: {}", recipe.description);
    println!("Min Shiso:   {}", recipe.min_shiso_version);
    let tasks: Vec<_> = recipe.tasks.iter().map(|t| t.as_str()).collect();
    println!("Tasks:       {}", tasks.join(", "));
    println!("Tags:        {}", recipe.tags.join(", "));

    let host = probe_host_profile();
    if let Some(metal) = &host.metal {
        println!(
            "Host:        {} (Apple GPU family {} = {}, working set {} MB)",
            metal.name,
            metal.family,
            shiso_hub::select::apple_family_label(metal.family),
            metal.working_set_bytes / (1024 * 1024)
        );
    }
    let best_variant = select_variant(&recipe, &host, None)
        .ok()
        .map(|sv| sv.variant.key.clone());

    println!("\nVariants:");
    for v in &recipe.variants {
        let is_selected = best_variant.as_ref() == Some(&v.key);
        let marker = if is_selected {
            "* [recommended for this host]"
        } else {
            ""
        };
        println!("  - Key:       {} {marker}", v.key.as_str());
        println!("    Backends:  {}", backends_label(&v.backends));
        println!("    Quant:     {}", v.quant);
        if let Some(sm) = v.requires.as_ref().and_then(|r| r.min_cuda_sm) {
            println!("    Min CUDA:  sm{sm}");
        }
        if let Some(family) = v.requires.as_ref().and_then(|r| r.min_metal_family) {
            println!(
                "    Min Metal: Apple GPU family {family} ({})",
                shiso_hub::select::apple_family_label(family)
            );
        }
        println!(
            "    Min RAM:   {} MB",
            v.requirements.min_ram_bytes / (1024 * 1024)
        );
        println!(
            "    Disk Size: {} MB",
            v.requirements.disk_bytes / (1024 * 1024)
        );

        println!("    Components:");
        for (role, comp) in &v.components {
            match comp.hf_source() {
                Some((repo, rev)) => println!("      [{}] {repo}@{rev}", role.as_str()),
                None => println!(
                    "      [{}] hub:{}",
                    role.as_str(),
                    comp.files
                        .iter()
                        .map(|f| f.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
            for f in &comp.files {
                match f.binds {
                    Some(bound) => println!(
                        "        - {} ({} bytes) -> {}",
                        f.name,
                        f.size,
                        bound.flag_name()
                    ),
                    None => println!("        - {} ({} bytes)", f.name, f.size),
                }
            }
        }
    }

    Ok(())
}

/// Executes the `shiso verify <ref> [--repair]` command (§6.4).
///
/// # Errors
/// Returns an error if verification fails or files cannot be repaired.
pub async fn run_verify(args: VerifyArgs) -> anyhow::Result<()> {
    let config = build_hub_config(&args.hub)?;
    let parsed_ref = ModelRef::parse(&args.model_ref)
        .map_err(|e| anyhow::anyhow!("invalid model reference '{}': {e}", args.model_ref))?;

    let store = BlobStore::new(config.clone());
    let installed = store.list_installed()?;

    let target_key = if let Some(k) = &parsed_ref.variant {
        k.clone()
    } else {
        let matching: Vec<_> = installed
            .iter()
            .filter(|iv| iv.model == parsed_ref.model)
            .collect();
        if matching.is_empty() {
            bail!("model '{}' is not installed", parsed_ref.model);
        }
        matching[0].variant.clone()
    };

    println!("Verifying '{}:{}'...", parsed_ref.model, target_key);
    let results = store.verify_variant(&parsed_ref.model, &target_key)?;

    let mut any_failed = false;
    for (file, ok) in &results {
        if *ok {
            println!("  [OK]     {file}");
        } else {
            println!("  [FAILED] {file}");
            any_failed = true;
        }
    }

    if any_failed {
        if args.repair {
            println!("\nRepairing corrupt or missing files...");
            let client = RegistryClient::new(config.clone());
            let downloader = Downloader::new(config.clone());

            let index = client.fetch_index().await?;
            let entry = client
                .resolve_model(&index, parsed_ref.model.as_str())
                .ok_or_else(|| anyhow::anyhow!("model not found in registry"))?;
            let current_version = semver::Version::parse(env!("CARGO_PKG_VERSION"))
                .unwrap_or_else(|_| semver::Version::new(0, 1, 0));
            let recipe = client.fetch_manifest(entry, &current_version).await?;

            let variant = recipe
                .variants
                .iter()
                .find(|v| v.key == target_key)
                .ok_or_else(|| anyhow::anyhow!("variant not found in recipe"))?;

            downloader
                .download_variant(recipe.name.as_ref(), variant, None)
                .await?;
            store.install_variant(&recipe, variant, &entry.manifest_sha256)?;
            println!("Repair complete.");
        } else {
            bail!("Verification failed. Run with '--repair' to re-download corrupt files.");
        }
    } else {
        println!("All files verified successfully.");
    }

    Ok(())
}

/// Executes the `shiso rm <ref> [--force]` command (§6.5).
///
/// # Errors
/// Returns an error if removing the variant fails.
pub fn run_rm(args: &RmArgs) -> anyhow::Result<()> {
    let config = build_hub_config(&args.hub)?;
    let parsed_ref = ModelRef::parse(&args.model_ref)
        .map_err(|e| anyhow::anyhow!("invalid model reference '{}': {e}", args.model_ref))?;

    let store = BlobStore::new(config);

    if let Some(key) = &parsed_ref.variant {
        store
            .remove_variant(&parsed_ref.model, key, args.force)
            .with_context(|| format!("removing variant {key}"))?;
        println!("Removed variant '{}:{}'.", parsed_ref.model, key);
    } else {
        let count = store
            .remove_model(&parsed_ref.model, args.force)
            .with_context(|| format!("removing model {}", parsed_ref.model))?;
        println!(
            "Removed {count} variant(s) of model '{}'.",
            parsed_ref.model
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use clap::{CommandFactory as _, FromArgMatches as _};
    use shiso_hub::recipe::{Recipe, Role};
    use shiso_hub::validate::{ValidationMode, validate_recipe};

    use super::populate_serve_args_from_variant;
    use crate::cli::{Cli, ServeArgs, apply_recipe_defaults};

    /// The recipe the registry fixture carries for the sparse family: the
    /// published trellis export as one directory, its tower file bound to
    /// `--vision`, its draft head inside the directory.
    const QWEN38: &str = include_str!("../../shiso-registry/tests/fixtures/qwen3.8-flash.json");

    /// The CUDA variant resolves to the directory as `--model`, the tower
    /// and n-gram table files inside it as `--vision` and `--ngram`, and the
    /// tuned knobs as recipe defaults.
    #[test]
    fn exl3_variant_resolves_to_directory_and_tower() {
        let recipe: Recipe = serde_json::from_str(QWEN38).expect("fixture parses");
        validate_recipe(&recipe, ValidationMode::Strict).expect("fixture validates");
        let variant = recipe
            .variants
            .iter()
            .find(|v| v.key.as_str() == "cuda-exl3-405")
            .expect("cuda variant");

        let matches =
            Cli::command().get_matches_from(["shiso", "serve", "--ctx", "8192", "--port", "1"]);
        let serve = matches.subcommand_matches("serve").expect("serve");
        let mut args = ServeArgs::from_arg_matches(serve).expect("args");
        let vdir = Path::new("/hub/models/qwen3.8-flash/cuda-exl3-405");
        populate_serve_args_from_variant(&mut args, vdir, variant, &recipe.name)
            .expect("populates");
        apply_recipe_defaults(
            &mut args,
            Some(serve),
            variant.server_defaults.as_ref().expect("defaults"),
        );

        let base = vdir.join(Role::Base.as_str());
        assert_eq!(args.model, Some(base.clone()));
        assert_eq!(args.vision, Some(base.join("vision_k6.safetensors")));
        assert_eq!(args.ngram, Some(base.join("ngram_embedding.safetensors")));
        // The head is inside the directory: no flag names it.
        assert_eq!(args.draft_model, None);
        assert_eq!(args.config, None);
        assert_eq!(args.model_name.as_deref(), Some("qwen3.8-flash"));

        assert_eq!(args.ctx, 8192, "an explicit flag beats the recipe");
        assert_eq!(args.spec_k, 4);
        assert_eq!(args.spec_lookup, 8);
        assert!(args.capture_verify && args.capture_batch);
        assert_eq!(args.gpu_memory_utilization, Some(0.88));
        assert_eq!(args.solo_prefill_chunk, 4096);
    }

    /// The re-quantized export of the same family: its shards carry no tower
    /// file, so the base checkpoint's tower comes from another repository as a
    /// component of its own.
    const QWEN38_UNCENSORED: &str =
        include_str!("../../shiso-registry/tests/fixtures/qwen3.8-flash-uncensored.json");

    /// A tower supplied by a component of its own resolves to that
    /// component's directory and entry, while the n-gram table stays the file
    /// bound inside the base directory.
    #[test]
    fn a_tower_from_another_repository_resolves_to_its_component() {
        let recipe: Recipe = serde_json::from_str(QWEN38_UNCENSORED).expect("fixture parses");
        validate_recipe(&recipe, ValidationMode::Strict).expect("fixture validates");
        let variant = recipe
            .variants
            .iter()
            .find(|v| v.key.as_str() == "exl3-405")
            .expect("cuda variant");
        let matches = Cli::command().get_matches_from(["shiso", "serve", "--port", "1"]);
        let serve = matches.subcommand_matches("serve").expect("serve");
        let mut args = ServeArgs::from_arg_matches(serve).expect("args");
        let vdir = Path::new("/hub/models/qwen3.8-flash-uncensored/exl3-405");
        populate_serve_args_from_variant(&mut args, vdir, variant, &recipe.name)
            .expect("populates");

        let base = vdir.join(Role::Base.as_str());
        assert_eq!(args.model, Some(base.clone()));
        assert_eq!(
            args.vision,
            Some(
                vdir.join(Role::Vision.as_str())
                    .join("vision_k6.safetensors")
            )
        );
        assert_eq!(args.ngram, Some(base.join("ngram_embedding.safetensors")));
        assert_eq!(args.draft_model, None);
        assert_eq!(args.model_name.as_deref(), Some("qwen3.8-flash-uncensored"));
    }

    /// The MLX variant is a directory checkpoint with no entry file, and the
    /// residency choice is the server's own `auto`.
    #[test]
    fn mlx_variant_points_at_its_directory() {
        let recipe: Recipe = serde_json::from_str(QWEN38).expect("fixture parses");
        let variant = recipe
            .variants
            .iter()
            .find(|v| v.key.as_str() == "metal-mlx4")
            .expect("metal variant");
        let matches = Cli::command().get_matches_from(["shiso", "serve"]);
        let serve = matches.subcommand_matches("serve").expect("serve");
        let mut args = ServeArgs::from_arg_matches(serve).expect("args");
        // A real directory: the config is looked for inside a checkpoint
        // directory and beside a checkpoint file, decided on the filesystem.
        let home = tempfile::tempdir().expect("tempdir");
        let vdir = home.path();
        std::fs::create_dir_all(vdir.join("base")).expect("base dir");
        populate_serve_args_from_variant(&mut args, vdir, variant, &recipe.name)
            .expect("populates");
        assert_eq!(args.model, Some(vdir.join("base")));
        assert_eq!(args.config, None);
        assert_eq!(args.config_path(), vdir.join("base").join("config.json"));
        assert_eq!(args.draft_model, None);
    }

    /// A hub-hosted draft vocabulary reaches `--draft-vocab` from its own
    /// component directory, beside a base that speculates by itself.
    #[test]
    fn a_hub_draft_vocabulary_resolves_to_its_file() {
        let recipe: Recipe = serde_json::from_str(
            r#"{
              "schema_version": 1, "name": "q-test", "family": "qwen4exp",
              "min_shiso_version": "0.2.0", "description": "x",
              "tasks": ["chat"], "features": {},
              "modalities": {"inputs": ["text"], "outputs": ["text"]},
              "variants": [{
                "key": "exl3", "backends": ["cuda"], "quant": "exl3",
                "requirements": {"min_ram_bytes": 1, "disk_bytes": 20},
                "server_defaults": {"spec_k": 4},
                "components": {
                  "base": {"provider": "huggingface", "repo": "o/q",
                    "revision": "3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c",
                    "files": [{"name": "model.safetensors", "size": 10,
                      "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"}]},
                  "draft_vocab": {"provider": "shiso", "entry": "draft_vocab_counts.txt",
                    "files": [{"name": "draft_vocab_counts.txt", "size": 10,
                      "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"}]}
                }
              }]
            }"#,
        )
        .expect("recipe parses");
        validate_recipe(&recipe, ValidationMode::Strict).expect("recipe validates");
        let matches = Cli::command().get_matches_from(["shiso", "serve"]);
        let serve = matches.subcommand_matches("serve").expect("serve");
        let mut args = ServeArgs::from_arg_matches(serve).expect("args");
        let vdir = Path::new("/hub/models/q-test/exl3");
        populate_serve_args_from_variant(&mut args, vdir, &recipe.variants[0], &recipe.name)
            .expect("populates");
        assert_eq!(
            args.draft_vocab,
            Some(vdir.join("draft_vocab").join("draft_vocab_counts.txt"))
        );
    }

    /// Files bound to `image_text_encoder`, `image_vae` and
    /// `image_tokenizer` reach their own flags as paths inside the component
    /// that downloads them.
    #[test]
    fn image_variant_routes_bound_text_encoder_vae_and_tokenizer() {
        let recipe: Recipe = serde_json::from_str(
            r#"{
              "schema_version": 1, "name": "qwen-test", "family": "flux2",
              "min_shiso_version": "0.1.0", "description": "x",
              "tasks": ["image_generation"], "features": {},
              "modalities": {"inputs": ["text"], "outputs": ["image"]},
              "variants": [{
                "key": "fp8", "backends": ["metal"], "quant": "fp8",
                "requirements": {"min_ram_bytes": 1, "disk_bytes": 40},
                "components": {
                  "image": {"provider": "huggingface", "repo": "o/split",
                    "revision": "3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c",
                    "entry": "dit.safetensors",
                    "files": [
                      {"name": "dit.safetensors", "size": 10,
                       "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"},
                      {"name": "te.safetensors", "size": 10, "binds": "image_text_encoder",
                       "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"},
                      {"name": "vae.safetensors", "size": 10, "binds": "image_vae",
                       "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"},
                      {"name": "tokenizer.json", "size": 10, "binds": "image_tokenizer",
                       "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"}]}
                }
              }]
            }"#,
        )
        .expect("recipe parses");
        validate_recipe(&recipe, ValidationMode::Strict).expect("recipe validates");
        let matches = Cli::command().get_matches_from(["shiso", "serve"]);
        let serve = matches.subcommand_matches("serve").expect("serve");
        let mut args = ServeArgs::from_arg_matches(serve).expect("args");
        let vdir = Path::new("/hub/models/qwen-test/fp8");
        populate_serve_args_from_variant(&mut args, vdir, &recipe.variants[0], &recipe.name)
            .expect("populates");
        assert_eq!(
            args.image_model,
            Some(vdir.join("image").join("dit.safetensors"))
        );
        assert_eq!(
            args.image_text_encoder,
            Some(vdir.join("image").join("te.safetensors"))
        );
        assert_eq!(
            args.image_vae,
            Some(vdir.join("image").join("vae.safetensors"))
        );
        assert_eq!(
            args.image_tokenizer,
            Some(vdir.join("image").join("tokenizer.json"))
        );
        assert_eq!(args.image_assets_dir, None);
    }

    /// An image variant hands its checkpoint file to `--image-model`, its
    /// asset tree to `--image-assets-dir`, and answers to the recipe's name.
    #[test]
    fn image_variant_resolves_to_model_and_assets() {
        let recipe: Recipe = serde_json::from_str(
            r#"{
              "schema_version": 1, "name": "flux-test", "family": "flux2",
              "min_shiso_version": "0.1.0", "description": "x",
              "tasks": ["image_generation"], "features": {},
              "modalities": {"inputs": ["text"], "outputs": ["image"]},
              "variants": [{
                "key": "fp8", "backends": ["metal"], "quant": "fp8",
                "requirements": {"min_ram_bytes": 1, "disk_bytes": 20},
                "components": {
                  "image": {"provider": "huggingface", "repo": "o/fp8",
                    "revision": "3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c",
                    "entry": "model.safetensors",
                    "files": [{"name": "model.safetensors", "size": 10,
                      "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"}]},
                  "image_assets": {"provider": "huggingface", "repo": "o/bf16",
                    "revision": "3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c",
                    "files": [{"name": "vae/config.json", "size": 10,
                      "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"}]}
                }
              }]
            }"#,
        )
        .expect("recipe parses");
        validate_recipe(&recipe, ValidationMode::Strict).expect("recipe validates");
        let matches = Cli::command().get_matches_from(["shiso", "serve"]);
        let serve = matches.subcommand_matches("serve").expect("serve");
        let mut args = ServeArgs::from_arg_matches(serve).expect("args");
        let vdir = Path::new("/hub/models/flux-test/fp8");
        populate_serve_args_from_variant(&mut args, vdir, &recipe.variants[0], &recipe.name)
            .expect("populates");
        assert_eq!(
            args.image_model,
            Some(vdir.join("image").join("model.safetensors"))
        );
        assert_eq!(args.image_assets_dir, Some(vdir.join("image_assets")));
        assert_eq!(args.image_model_id.as_deref(), Some("flux-test"));
        assert_eq!(args.model, None);
    }
}
