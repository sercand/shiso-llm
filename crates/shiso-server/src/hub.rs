//! Hub integration for `shiso-server`: model resolution, pull, list, info, verify, rm (§6).

#![expect(
    clippy::print_stdout,
    reason = "CLI command output intended for operator console or NDJSON stream"
)]

use std::io::{IsTerminal, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Context, bail};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle as IndicatifStyle};
use url::Url;

use shiso_hub::config::{HfToken, HubConfig, RegistryEndpoint};
use shiso_hub::download::Downloader;
use shiso_hub::host::{Backend, HostProfile};
use shiso_hub::progress::ProgressEvent;
use shiso_hub::recipe::{ModelName, ModelRef};
use shiso_hub::registry::RegistryClient;
use shiso_hub::select::select_variant;
use shiso_hub::store::BlobStore;
use shiso_runtime::hub::{InstalledModel, PullEvent, PullPolicy, pull_observed_on};

use crate::cli::{
    ExplicitlySet, HubArgs, InfoArgs, ListArgs, ProgressStyle, PullArgs, RmArgs, ServeArgs,
    VerifyArgs,
};

pub use shiso_runtime::hub::probe_host_profile;

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

/// Resolves a positional model argument on `ServeArgs` (§6.6): a path becomes
/// the model, a registry reference is looked up in the model home and fills the
/// path options and the recipe defaults the command line left alone.
///
/// Returns the installed model, whose shared variant lock the caller holds
/// while serving, or `None` when a path (or nothing) was given.
///
/// # Errors
/// Returns an error if component flags conflict with a model reference, the model is not installed,
/// or opening the variant fails.
pub fn resolve_serve_model(
    args: &mut ServeArgs,
    explicit: Option<&ExplicitlySet>,
) -> anyhow::Result<Option<InstalledModel>> {
    let Some(model_ref) = args.model_ref.clone() else {
        return Ok(None);
    };
    let config = HubConfig::with_defaults(HubConfig::default_home());
    let none = ExplicitlySet::default();
    let installed = shiso_runtime::hub::resolve_serve_model(
        &model_ref,
        &mut args.options,
        explicit.unwrap_or(&none),
        &config,
    )?;
    Ok(installed)
}

/// The backends a variant names, as `metal/cuda`.
fn backends_label(backends: &[Backend]) -> String {
    if backends.is_empty() {
        return "unknown".to_string();
    }
    let names: Vec<&str> = backends.iter().map(|b| b.as_str()).collect();
    names.join("/")
}

/// Prints a pull as `shiso pull` shows it: the steps as lines, the bytes as
/// bars, a plain log, or one JSON object per download event (§6.1).
struct PullRenderer<'a> {
    style: ProgressStyle,
    is_tty: bool,
    out: &'a mut dyn Write,
    bars: Option<Bars>,
    /// The model and variant that was installed or found up to date, and
    /// whether it was installed now.
    done: Option<(String, String, bool)>,
}

impl<'a> PullRenderer<'a> {
    fn new(style: ProgressStyle, is_tty: bool, out: &'a mut dyn Write) -> Self {
        Self {
            style,
            is_tty,
            out,
            bars: None,
            done: None,
        }
    }

    fn bars_wanted(&self) -> bool {
        matches!(self.style, ProgressStyle::Bars | ProgressStyle::Auto) && self.is_tty
    }

    /// One line on the output. A closed pipe is the reader's business, as it
    /// is for `println!` in the other commands.
    fn line(&mut self, text: &str) {
        let _ = writeln!(self.out, "{text}");
    }

    fn event(&mut self, event: &PullEvent) {
        match event {
            PullEvent::FetchingIndex => self.line("Fetching registry index..."),
            PullEvent::Selected {
                explanation,
                warning,
            } => {
                self.line(explanation);
                if let Some(warn) = warning {
                    self.line(warn);
                }
            }
            PullEvent::UpToDate { model, variant } => {
                self.line(&format!("Model '{model}:{variant}' is already up to date."));
                self.done = Some((model.clone(), variant.clone(), false));
            }
            PullEvent::Progress(event) => self.progress(event),
            PullEvent::Tick => {}
        }
    }

    fn progress(&mut self, event: &ProgressEvent) {
        if let ProgressEvent::Installed { model, variant } = event {
            self.done = Some((model.clone(), variant.clone(), true));
        }
        if matches!(self.style, ProgressStyle::Json) {
            if let Ok(json) = serde_json::to_string(event) {
                self.line(&json);
            }
        } else if self.bars_wanted() {
            self.bars.get_or_insert_with(Bars::new).event(event);
        } else {
            match event {
                ProgressEvent::Resolved { total, .. } => {
                    self.line(&format!("Downloading {total} bytes across components..."));
                }
                ProgressEvent::FileStarted {
                    role, file, total, ..
                } => {
                    self.line(&format!("  [{role}] {file} started ({total} bytes)"));
                }
                ProgressEvent::FileVerified { role, file, .. } => {
                    self.line(&format!("  [{role}] {file} verified"));
                }
                ProgressEvent::Installed { model, variant } => {
                    self.line(&format!("Installed {model}:{variant}"));
                }
                ProgressEvent::Error { message, .. } => {
                    self.line(&format!("Error: {message}"));
                }
                ProgressEvent::FileProgress { .. } => {}
            }
        }
    }

    /// The closing line, after the bars have ended.
    fn finish(mut self) {
        let done = self.done.take();
        self.bars = None;
        if let Some((model, variant, true)) = done {
            self.line(&format!(
                "Successfully pulled and installed '{model}:{variant}'."
            ));
        }
    }
}

/// Pulls `model_ref` for `host`, rendering to `out` as `shiso pull` does.
/// Blocks; see `shiso_runtime::hub::pull_observed_on`.
fn pull_rendered(
    model_ref: &str,
    config: &HubConfig,
    host: &HostProfile,
    style: ProgressStyle,
    is_tty: bool,
    out: &mut dyn Write,
) -> anyhow::Result<()> {
    let mut renderer = PullRenderer::new(style, is_tty, out);
    let installed = pull_observed_on(
        host,
        model_ref,
        config,
        PullPolicy::CheckRegistry,
        &mut |event| {
            renderer.event(event);
            true
        },
    )?;
    drop(installed);
    renderer.finish();
    Ok(())
}

/// Executes the `shiso pull <ref>` command (§6.1).
///
/// The pull itself is `shiso_runtime::hub::pull_observed`, the one embedders
/// call too; this only prints what it reports.
///
/// # Errors
/// Returns an error if resolution, download, or installation fails.
pub async fn run_pull(args: PullArgs) -> anyhow::Result<()> {
    let config = build_hub_config(&args.hub)?;
    let model_ref = args.model_ref;
    let style = args.hub.progress;
    let is_tty = std::io::stdout().is_terminal();
    // The pull blocks on a runtime of its own, so it runs on a thread that may.
    tokio::task::spawn_blocking(move || {
        pull_rendered(
            &model_ref,
            &config,
            &probe_host_profile(),
            style,
            is_tty,
            &mut std::io::stdout(),
        )
    })
    .await
    .context("the pull task ended abnormally")?
}

/// One progress bar per file, as an interactive terminal shows a pull.
struct Bars {
    multi: MultiProgress,
    style: IndicatifStyle,
    by_file: std::collections::HashMap<String, ProgressBar>,
}

impl Bars {
    fn new() -> Self {
        let style = IndicatifStyle::default_bar()
            .template(
                "{prefix:.bold} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
            )
            .unwrap_or_else(|_| IndicatifStyle::default_bar());
        Self {
            multi: MultiProgress::new(),
            style,
            by_file: std::collections::HashMap::new(),
        }
    }

    fn event(&mut self, event: &ProgressEvent) {
        match event {
            ProgressEvent::FileStarted {
                role, file, total, ..
            } => {
                let pb = self.multi.add(ProgressBar::new(*total));
                pb.set_style(self.style.clone());
                pb.set_prefix(format!("{role}/{file}"));
                self.by_file.insert(file.clone(), pb);
            }
            ProgressEvent::FileProgress { file, bytes, .. } => {
                if let Some(pb) = self.by_file.get(file) {
                    pb.set_position(*bytes);
                }
            }
            ProgressEvent::FileVerified { file, .. } => {
                if let Some(pb) = self.by_file.get(file) {
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
    use std::num::NonZeroUsize;

    use serde_json::json;
    use sha2::{Digest as _, Sha256};
    use shiso_hub::config::RegistryEndpoint;
    use shiso_hub::host::MetalInfo;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    fn digest(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    /// A registry and file host in one mock server, serving one tiny model.
    struct Registry {
        rt: tokio::runtime::Runtime,
        server: MockServer,
        home: tempfile::TempDir,
    }

    impl Registry {
        fn new() -> Self {
            let rt = tokio::runtime::Runtime::new().expect("runtime");
            let server = rt.block_on(MockServer::start());
            let files: [(&str, &str, Vec<u8>); 2] = [
                ("base", "mini.gguf", vec![3u8; 2048]),
                (
                    "config",
                    "config.json",
                    br#"{"model_type":"lfm2"}"#.to_vec(),
                ),
            ];
            let component = |role: &str| {
                let entries: Vec<_> = files
                    .iter()
                    .filter(|(r, ..)| *r == role)
                    .map(|(_, name, bytes)| {
                        json!({"name": name, "size": bytes.len(), "sha256": digest(bytes)})
                    })
                    .collect();
                json!({
                    "provider": "huggingface",
                    "repo": format!("acme/{role}"),
                    "revision": REVISION,
                    "entry": if role == "base" { "mini.gguf" } else { "config.json" },
                    "files": entries,
                })
            };
            let manifest = json!({
                "schema_version": 1, "name": "mini", "aliases": [], "family": "lfm2",
                "min_shiso_version": "0.1.0", "description": "fixture", "tags": [],
                "tasks": ["decision"],
                "modalities": {"inputs": ["text"], "outputs": ["text"]},
                "features": {
                    "tools": false, "reasoning": false, "reasoning_efforts": [],
                    "default_reasoning_effort": null, "structured_output": false
                },
                "variants": [{
                    "key": "q8_0", "backends": ["metal", "cuda"], "quant": "Q8_0",
                    "requirements": {"min_ram_bytes": 1, "disk_bytes": 2069},
                    "components": {"base": component("base"), "config": component("config")},
                }],
            })
            .to_string()
            .into_bytes();
            let index = json!({
                "schema_version": 1, "generated_at": "2026-10-05T00:00:00Z",
                "models": [{
                    "name": "mini", "family": "lfm2", "min_shiso_version": "0.1.0",
                    "description": "fixture", "tasks": ["decision"],
                    "updated_at": "2026-10-05T00:00:00Z",
                    "manifest": "models/mini/index.json",
                    "manifest_sha256": digest(&manifest), "variants": [],
                }],
            })
            .to_string()
            .into_bytes();
            let serve = |route: &str, body: Vec<u8>| {
                let mock = Mock::given(method("GET"))
                    .and(path(route.to_owned()))
                    .respond_with(ResponseTemplate::new(200).set_body_bytes(body));
                rt.block_on(mock.mount(&server));
            };
            serve("/v1/index.json", index);
            serve("/v1/models/mini/index.json", manifest);
            for (role, name, bytes) in files {
                serve(&format!("/acme/{role}/resolve/{REVISION}/{name}"), bytes);
            }
            Self {
                rt,
                server,
                home: tempfile::tempdir().expect("tempdir"),
            }
        }

        fn config(&self) -> HubConfig {
            let uri = self.server.uri();
            HubConfig {
                home: self.home.path().to_path_buf(),
                registry: RegistryEndpoint {
                    url: Url::parse(&format!("{uri}/v1")).expect("url"),
                },
                hf_endpoint: Url::parse(&uri).expect("url"),
                hf_token: None,
                max_parallel_files: NonZeroUsize::MIN,
            }
        }

        /// What `shiso pull mini` prints, on a host that runs the fixture.
        fn pull(&self, style: ProgressStyle) -> Vec<String> {
            let host = HostProfile::probe(
                vec![Backend::Metal],
                None,
                Some(MetalInfo {
                    name: "test device".to_owned(),
                    family: 9,
                    working_set_bytes: 64 << 30,
                }),
            );
            let mut out = Vec::new();
            pull_rendered("mini", &self.config(), &host, style, false, &mut out)
                .unwrap_or_else(|e| panic!("{e}"));
            String::from_utf8(out)
                .expect("utf8")
                .lines()
                .map(str::to_owned)
                .collect()
        }
    }

    #[test]
    fn a_fresh_pull_prints_the_steps_then_the_closing_line() {
        let fx = Registry::new();
        let lines = fx.pull(ProgressStyle::Auto);
        assert_eq!(lines[0], "Fetching registry index...");
        assert!(lines[1].starts_with("selected variant 'q8_0'"), "{lines:?}");
        assert_eq!(lines[2], "Downloading 2069 bytes across components...");
        let mut started: Vec<_> = lines
            .iter()
            .filter(|l| l.contains("started ("))
            .cloned()
            .collect();
        started.sort();
        assert_eq!(
            started,
            [
                "  [base] mini.gguf started (2048 bytes)",
                "  [config] config.json started (21 bytes)"
            ]
        );
        assert!(lines.contains(&"  [base] mini.gguf verified".to_owned()));
        assert!(lines.contains(&"  [config] config.json verified".to_owned()));
        let n = lines.len();
        assert_eq!(lines[n - 2], "Installed mini:q8_0");
        assert_eq!(
            lines[n - 1],
            "Successfully pulled and installed 'mini:q8_0'."
        );
        assert_eq!(n, 9, "{lines:?}");
    }

    #[test]
    fn pulling_what_is_installed_says_it_is_up_to_date_and_nothing_more() {
        let fx = Registry::new();
        fx.pull(ProgressStyle::Auto);
        let lines = fx.pull(ProgressStyle::Auto);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert_eq!(lines[0], "Fetching registry index...");
        assert!(lines[1].starts_with("selected variant 'q8_0'"));
        assert_eq!(lines[2], "Model 'mini:q8_0' is already up to date.");
        drop(fx.rt);
    }

    #[test]
    fn json_progress_prints_one_object_per_download_event() {
        let fx = Registry::new();
        let lines = fx.pull(ProgressStyle::Json);
        assert_eq!(lines[0], "Fetching registry index...");
        let events: Vec<serde_json::Value> = lines[2..lines.len() - 1]
            .iter()
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{l}: {e}")))
            .collect();
        assert_eq!(events[0]["event"], "resolved");
        assert_eq!(events[0]["total"], 2069);
        assert_eq!(
            events.last().map(|e| e["event"].clone()),
            Some("installed".into())
        );
        assert_eq!(
            lines.last().map(String::as_str),
            Some("Successfully pulled and installed 'mini:q8_0'.")
        );
    }
}
