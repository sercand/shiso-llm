//! Throughput records: what a variant measured on a named device.
//!
//! A record is one file, `benchmarks/<model>/<variant>/<device>.json`, written
//! by `shiso-registry bench` on the device itself and committed to the
//! registry repository. The build publishes them next to the manifest
//! (`models/<name>/benchmarks.json`) and on the model page, never inside the
//! manifest: a new measurement must not change the manifest digest, or every
//! host that installed the model would see an update that changes no file.
//!
//! The bench drives a running `shiso serve` over HTTP instead of loading the
//! model itself. That keeps this tool free of every backend and model family,
//! and it measures the configuration a user actually gets: the recipe's server
//! defaults, the release build, the real scheduler.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256 as Sha256Hasher};
use shiso_hub::Backend;
use shiso_hub::recipe::{
    InstalledMetadata, ModelName, Recipe, Role, Sha256, Task, Variant, VariantKey,
};

mod chat;
mod client;
mod decision;
mod image;
mod prompt;
mod transcription;

pub(crate) use chat::ChatResult;
pub(crate) use decision::DecisionResult;
pub(crate) use image::ImageResult;
pub(crate) use transcription::TranscriptionResult;

/// Record format this build reads and writes.
pub(crate) const BENCH_SCHEMA_VERSION: u32 = 1;

/// A device name such as `m1-max` or `gb10`: lowercase letters, digits and
/// dashes, starting with a letter or digit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(crate) struct DeviceSlug(String);

impl DeviceSlug {
    pub(crate) fn parse(s: &str) -> anyhow::Result<Self> {
        let ok = !s.is_empty()
            && s.len() <= 48
            && s.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit())
            && s.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        if !ok {
            bail!("device '{s}' must be lowercase letters, digits and '-' (e.g. m1-max, gb10)");
        }
        Ok(Self(s.to_string()))
    }

    /// The slug for a probed chip name: `Apple M1 Max` is `m1-max`,
    /// `NVIDIA GB10` is `gb10`.
    pub(crate) fn from_chip(chip: &str) -> anyhow::Result<Self> {
        let lower = chip.to_ascii_lowercase();
        let trimmed = ["apple ", "nvidia ", "geforce "]
            .iter()
            .fold(lower.as_str(), |s, p| s.strip_prefix(p).unwrap_or(s));
        let slug: String = trimmed
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join("-");
        Self::parse(&slug)
            .with_context(|| format!("deriving a device name from '{chip}'; pass --device"))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for DeviceSlug {
    type Error = anyhow::Error;
    fn try_from(s: String) -> anyhow::Result<Self> {
        Self::parse(&s)
    }
}

impl From<DeviceSlug> for String {
    fn from(d: DeviceSlug) -> Self {
        d.0
    }
}

impl fmt::Display for DeviceSlug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One measurement, of the kind the model's task calls for.
///
/// Throughput means something different per task: tokens a second for a chat
/// model, a decision's latency for a decision head, seconds an image for an
/// image model, a real-time factor for transcription. Each is its own kind
/// rather than one set of fields half of which a model leaves empty, so a
/// page never compares a chat model's decode rate with an image model's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BenchResult {
    /// A chat completion at one prompt length.
    Chat(ChatResult),
    /// One decision at one prompt length.
    Decision(DecisionResult),
    /// One image at one size.
    Image(ImageResult),
    /// One clip transcribed.
    Transcription(TranscriptionResult),
}

impl BenchResult {
    /// The kind of bench that produced this result.
    pub(crate) const fn kind(&self) -> BenchKind {
        match self {
            Self::Chat(_) => BenchKind::Chat,
            Self::Decision(_) => BenchKind::Decision,
            Self::Image(_) => BenchKind::Image,
            Self::Transcription(_) => BenchKind::Transcription,
        }
    }
}

/// What a bench measures, chosen from the task a recipe serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum BenchKind {
    /// Prefill and decode tokens a second.
    Chat,
    /// Milliseconds a decision.
    Decision,
    /// Seconds an image.
    Image,
    /// Real-time multiple and time to first text.
    Transcription,
}

impl BenchKind {
    /// The bench for a task, if there is one. Diarization and embeddings
    /// have none yet.
    pub(crate) const fn for_task(task: Task) -> Option<Self> {
        match task {
            Task::Chat => Some(Self::Chat),
            Task::Decision => Some(Self::Decision),
            Task::ImageGeneration => Some(Self::Image),
            Task::Transcription => Some(Self::Transcription),
            Task::Diarization | Task::Embedding | Task::Classification => None,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Decision => "decision",
            Self::Image => "image",
            Self::Transcription => "transcription",
        }
    }
}

impl fmt::Display for BenchKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One device's measurement of one variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BenchRecord {
    /// Record format version.
    pub(crate) schema_version: u32,
    /// Model measured.
    pub(crate) model: ModelName,
    /// Variant measured.
    pub(crate) variant: VariantKey,
    /// Digest of the variant's file list when it was measured; see
    /// [`variant_fingerprint`].
    pub(crate) variant_fingerprint: Sha256,
    /// Device name (`m1-max`, `gb10`, ...).
    pub(crate) device: DeviceSlug,
    /// The chip as the host reports it (`Apple M1 Max`).
    pub(crate) chip: String,
    /// Backend that ran the variant.
    pub(crate) backend: Backend,
    /// Host memory.
    pub(crate) ram_bytes: u64,
    /// Operating system and version.
    pub(crate) os: String,
    /// `shiso` build that served the variant.
    pub(crate) shiso_version: String,
    /// RFC 3339 time of the measurement.
    pub(crate) measured_at: String,
    /// Measured runs per setting, after one discarded warm-up.
    pub(crate) runs: u32,
    /// Context window the server ran with, for a language model; the same
    /// weights decode slower under a larger cache, so a rate is read
    /// against it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) context_length: Option<u64>,
    /// One entry per setting (prompt length, image size, clip), all of one
    /// kind.
    pub(crate) results: Vec<BenchResult>,
    /// Anything a reader needs to know (a non-default flag, thermals).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) notes: Option<String>,
}

/// A digest over every file a variant installs, by role, name and content.
///
/// Records carry it so the build can tell a measurement of today's weights
/// from one of weights the recipe has since replaced. The manifest digest
/// cannot do that: it also changes when only a description does.
pub(crate) fn variant_fingerprint<'a>(
    files: impl IntoIterator<Item = (Role, &'a str, &'a Sha256)>,
) -> Sha256 {
    let mut lines: Vec<String> = files
        .into_iter()
        .map(|(role, name, sha)| format!("{role}\t{name}\t{sha}\n"))
        .collect();
    lines.sort();
    let digest = Sha256Hasher::digest(lines.concat().as_bytes());
    Sha256::parse(&hex::encode(digest)).expect("a SHA-256 digest is 64 hex characters")
}

/// [`variant_fingerprint`] of a recipe's variant.
pub(crate) fn recipe_fingerprint(variant: &Variant) -> Sha256 {
    variant_fingerprint(variant.components.iter().flat_map(|(role, comp)| {
        comp.files
            .iter()
            .map(move |f| (*role, f.name.as_str(), &f.sha256))
    }))
}

/// A record as the build publishes it.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PublishedRecord {
    #[serde(flatten)]
    pub(crate) record: BenchRecord,
    /// The variant's files have changed since this was measured.
    pub(crate) stale: bool,
}

/// Reads every record under `<repo>/benchmarks/`, grouped by model.
///
/// # Errors
/// When a record does not parse, sits at a path that disagrees with its
/// contents, or names a model or variant no recipe has. A record of a
/// removed variant is an error rather than skipped: it is a file the
/// maintainer should delete, and a skip would hide it forever.
pub(crate) fn load_all(
    repo_path: &Path,
    recipes: &[Recipe],
) -> anyhow::Result<BTreeMap<ModelName, Vec<PublishedRecord>>> {
    let root = repo_path.join("benchmarks");
    let mut out: BTreeMap<ModelName, Vec<PublishedRecord>> = BTreeMap::new();
    if !root.is_dir() {
        return Ok(out);
    }
    for path in json_files_under(&root)? {
        let rel = path
            .strip_prefix(repo_path)
            .unwrap_or(&path)
            .display()
            .to_string();
        let record: BenchRecord = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("parsing benchmark record {rel}"))?;
        if record.schema_version != BENCH_SCHEMA_VERSION {
            bail!(
                "{rel}: schema_version {} is not {BENCH_SCHEMA_VERSION}",
                record.schema_version
            );
        }
        let expected = root
            .join(record.model.as_str())
            .join(record.variant.as_str())
            .join(format!("{}.json", record.device));
        if expected != path {
            bail!(
                "{rel}: a record of {}:{} on {} belongs at {}",
                record.model,
                record.variant,
                record.device,
                expected
                    .strip_prefix(repo_path)
                    .unwrap_or(&expected)
                    .display()
            );
        }
        let recipe = recipes
            .iter()
            .find(|r| r.name == record.model)
            .with_context(|| format!("{rel}: no recipe is named '{}'", record.model))?;
        let variant = recipe
            .variants
            .iter()
            .find(|v| v.key == record.variant)
            .with_context(|| {
                format!(
                    "{rel}: '{}' has no variant '{}'; delete the record",
                    record.model, record.variant
                )
            })?;
        let Some(first) = record.results.first() else {
            bail!("{rel}: a record needs at least one result");
        };
        if record.results.iter().any(|r| r.kind() != first.kind()) {
            bail!("{rel}: a record holds results of one kind only");
        }
        let stale = recipe_fingerprint(variant) != record.variant_fingerprint;
        out.entry(record.model.clone())
            .or_default()
            .push(PublishedRecord { record, stale });
    }
    for records in out.values_mut() {
        records.sort_by(|a, b| {
            (a.record.variant.as_str(), a.record.device.as_str())
                .cmp(&(b.record.variant.as_str(), b.record.device.as_str()))
        });
    }
    Ok(out)
}

fn json_files_under(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in fs::read_dir(&d).with_context(|| format!("reading {}", d.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "json") {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}

/// Progress lines of a running bench, shared with the editor so its page can
/// show them while the measurement runs.
#[derive(Debug, Clone, Default)]
pub(crate) struct BenchLog(Arc<Mutex<Vec<String>>>);

impl BenchLog {
    pub(crate) fn push(&self, line: impl Into<String>) {
        if let Ok(mut lines) = self.0.lock() {
            lines.push(line.into());
        }
    }

    pub(crate) fn lines(&self) -> Vec<String> {
        self.0.lock().map(|l| l.clone()).unwrap_or_default()
    }
}

/// What to measure, and where.
#[derive(Debug, Clone)]
pub(crate) struct BenchDesc {
    /// Registry repository; the record is written under its `benchmarks/`.
    pub(crate) repo: PathBuf,
    /// Model served.
    pub(crate) model: ModelName,
    /// Variant served; `None` takes the only installed one.
    pub(crate) variant: Option<VariantKey>,
    /// `shiso serve` base URL.
    pub(crate) server_url: String,
    /// Device name; `None` derives it from the chip.
    pub(crate) device: Option<DeviceSlug>,
    /// The task to measure; `None` takes the recipe's only measurable one.
    pub(crate) task: Option<Task>,
    /// Target prompt lengths (chat, decision).
    pub(crate) prompt_tokens: Vec<u64>,
    /// Tokens generated per run (chat).
    pub(crate) decode_tokens: u64,
    /// Image sizes (`WxH`); empty measures the server default.
    pub(crate) image_sizes: Vec<String>,
    /// WAV clip to transcribe; `None` synthesizes one on macOS.
    pub(crate) audio: Option<PathBuf>,
    /// Measured runs per prompt length.
    pub(crate) runs: u32,
    /// `shiso` home holding the installed variant.
    pub(crate) home: PathBuf,
    /// Version of the serving build; `None` asks `shiso --version`.
    pub(crate) shiso_version: Option<String>,
    /// Free-form note stored with the record.
    pub(crate) notes: Option<String>,
    /// Where progress goes besides the trace log.
    pub(crate) log: Option<BenchLog>,
}

fn say(log: Option<&BenchLog>, line: String) {
    tracing::info!("{line}");
    if let Some(log) = log {
        log.push(line);
    }
}

fn run_label(run: u32) -> String {
    if run == 0 {
        "warm-up".to_string()
    } else {
        format!("run {run}")
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    match n {
        0 => 0.0,
        _ if n % 2 == 1 => v[n / 2],
        _ => f64::midpoint(v[n / 2 - 1], v[n / 2]),
    }
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Measures a served variant and writes its record.
///
/// # Errors
/// When the installed variant does not match the recipe in the repository,
/// the server is unreachable or does not report its timings, or a request
/// fails. A refused measurement writes nothing.
pub(crate) async fn run(desc: BenchDesc) -> anyhow::Result<PathBuf> {
    if desc.runs == 0 {
        bail!("bench needs --runs >= 1");
    }
    let log = desc.log.as_ref();
    let Target {
        model,
        key,
        backends,
        fingerprint,
        kind,
    } = Target::resolve(&desc)?;
    if matches!(kind, BenchKind::Chat | BenchKind::Decision) {
        if desc.prompt_tokens.is_empty() {
            bail!("a {kind} bench needs at least one prompt size");
        }
        if let Some(p) = desc.prompt_tokens.iter().find(|p| **p < 64) {
            bail!("prompt size {p} is too short to time a prefill; use 64 or more");
        }
    }

    let host = probe_host()?;
    let device = match desc.device.clone() {
        Some(d) => d,
        None => DeviceSlug::from_chip(&host.chip)?,
    };
    let backend = host.backend;
    if !backends.contains(&backend) {
        bail!("variant '{key}' does not list backend {backend}, which this host runs");
    }
    let shiso_version = match desc.shiso_version.clone() {
        Some(v) => v,
        None => shiso_version()?,
    };

    let server = client::Server::connect(&desc.server_url, kind).await?;
    say(
        log,
        format!(
            "Measuring {kind} for {model}:{key} served as '{}' on {device} ({}, {backend})",
            server.model_id, host.chip
        ),
    );
    let results = match kind {
        BenchKind::Chat => chat::run(&server, &desc, log).await?,
        BenchKind::Decision => decision::run(&server, &desc, log).await?,
        BenchKind::Image => image::run(&server, &desc, log).await?,
        BenchKind::Transcription => transcription::run(&server, &desc, log).await?,
    };

    let record = BenchRecord {
        schema_version: BENCH_SCHEMA_VERSION,
        model: model.clone(),
        variant: key.clone(),
        variant_fingerprint: fingerprint,
        device: device.clone(),
        chip: host.chip,
        backend,
        ram_bytes: shiso_hub::probe_ram(),
        os: host.os,
        shiso_version,
        measured_at: crate::editor::current_iso8601(),
        runs: desc.runs,
        context_length: (server.context_length > 0).then_some(server.context_length),
        results,
        notes: desc.notes,
    };
    let path = desc
        .repo
        .join("benchmarks")
        .join(model.as_str())
        .join(key.as_str())
        .join(format!("{device}.json"));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(&record)?;
    bytes.push(b'\n');
    fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    say(log, format!("Wrote {}", path.display()));
    Ok(path)
}

/// The recipe variant a bench measures, checked against what is installed.
struct Target {
    model: ModelName,
    key: VariantKey,
    backends: Vec<Backend>,
    fingerprint: Sha256,
    kind: BenchKind,
}

impl Target {
    /// Resolves the model (an alias measures its canonical model, and the
    /// record is filed under it) and refuses a task the bench cannot measure
    /// or an install whose files differ from the recipe.
    fn resolve(desc: &BenchDesc) -> anyhow::Result<Self> {
        let recipe = crate::editor::load_recipes(&desc.repo)?
            .into_iter()
            .map(|(_, r)| r)
            .find(|r| r.name == desc.model || r.aliases.contains(&desc.model))
            .with_context(|| {
                format!(
                    "no recipe in {} is named '{}'",
                    desc.repo.display(),
                    desc.model
                )
            })?;
        let model = recipe.name.clone();
        let kind = Self::kind(&recipe, desc.task)?;
        let installed = InstalledVariant::find(&desc.home, &model, desc.variant.as_ref())?;
        let variant = recipe
            .variants
            .iter()
            .find(|v| v.key == installed.key)
            .with_context(|| format!("the recipe has no variant '{}'", installed.key))?;
        let fingerprint = recipe_fingerprint(variant);
        if fingerprint != installed.fingerprint {
            bail!(
                "the installed {model}:{} has different files from the recipe in {}; run `shiso pull {model}:{}` first",
                installed.key,
                desc.repo.display(),
                installed.key
            );
        }

        Ok(Self {
            model,
            key: installed.key,
            backends: variant.backends.clone(),
            fingerprint,
            kind,
        })
    }

    /// The bench for the task asked for, or for the recipe's only task that
    /// has one.
    fn kind(recipe: &Recipe, asked: Option<Task>) -> anyhow::Result<BenchKind> {
        let model = &recipe.name;
        if let Some(task) = asked {
            if !recipe.tasks.contains(&task) {
                bail!("'{model}' does not serve {task}");
            }
            return BenchKind::for_task(task)
                .with_context(|| format!("there is no {task} bench yet"));
        }
        let kinds: Vec<BenchKind> = recipe
            .tasks
            .iter()
            .filter_map(|t| BenchKind::for_task(*t))
            .collect();
        match kinds.as_slice() {
            [one] => Ok(*one),
            [] => {
                let tasks: Vec<_> = recipe.tasks.iter().map(|t| t.as_str()).collect();
                bail!(
                    "'{model}' serves {}, which no bench measures yet",
                    tasks.join(", ")
                )
            }
            _ => bail!("'{model}' serves several tasks; choose one with --task"),
        }
    }
}

struct InstalledVariant {
    key: VariantKey,
    fingerprint: Sha256,
}

impl InstalledVariant {
    /// The installed variant to measure: the one asked for, or the only one.
    fn find(home: &Path, model: &ModelName, key: Option<&VariantKey>) -> anyhow::Result<Self> {
        let model_dir = home.join("models").join(model.as_str());
        let key = if let Some(k) = key {
            k.clone()
        } else {
            let mut keys = Vec::new();
            if model_dir.is_dir() {
                for entry in fs::read_dir(&model_dir)? {
                    let entry = entry?;
                    let name = entry.file_name().to_string_lossy().to_string();
                    if entry.path().join("installed.json").is_file()
                        && let Ok(k) = VariantKey::parse(&name)
                    {
                        keys.push(k);
                    }
                }
            }
            match keys.len() {
                1 => keys.remove(0),
                0 => bail!("'{model}' is not installed under {}", home.display()),
                _ => {
                    bail!("'{model}' has several variants installed; name one as {model}:<key>")
                }
            }
        };
        let path = model_dir.join(key.as_str()).join("installed.json");
        let meta: InstalledMetadata = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))?;
        let fingerprint = variant_fingerprint(
            meta.files
                .iter()
                .map(|f| (f.role, f.name.as_str(), &f.sha256)),
        );
        Ok(Self { key, fingerprint })
    }
}

struct HostFacts {
    chip: String,
    os: String,
    backend: Backend,
}

fn command_line(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status.success().then(|| {
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string()
    })
}

fn probe_host() -> anyhow::Result<HostFacts> {
    if cfg!(target_os = "macos") {
        let chip = command_line("sysctl", &["-n", "machdep.cpu.brand_string"])
            .filter(|c| !c.is_empty())
            .context("sysctl did not report the chip")?;
        let version = command_line("sw_vers", &["-productVersion"]).unwrap_or_default();
        Ok(HostFacts {
            chip,
            os: format!("macOS {version}").trim().to_string(),
            backend: Backend::Metal,
        })
    } else {
        let chip = command_line("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"])
            .filter(|c| !c.is_empty())
            .context("nvidia-smi did not report a GPU; this host has no backend to measure")?;
        let os = fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|t| {
                t.lines().find_map(|l| {
                    l.strip_prefix("PRETTY_NAME=")
                        .map(|v| v.trim_matches('"').to_string())
                })
            })
            .unwrap_or_else(|| std::env::consts::OS.to_string());
        Ok(HostFacts {
            chip,
            os,
            backend: Backend::Cuda,
        })
    }
}

fn shiso_version() -> anyhow::Result<String> {
    let line = command_line("shiso", &["--version"])
        .context("`shiso --version` did not run; pass --shiso-version")?;
    // `shiso 0.1.0` -> `0.1.0`
    Ok(line.rsplit(' ').next().unwrap_or(line.as_str()).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_names_come_from_the_chip() {
        assert_eq!(
            DeviceSlug::from_chip("Apple M1 Max")
                .expect("valid")
                .as_str(),
            "m1-max"
        );
        assert_eq!(
            DeviceSlug::from_chip("Apple M5 Ultra")
                .expect("valid")
                .as_str(),
            "m5-ultra"
        );
        assert_eq!(
            DeviceSlug::from_chip("NVIDIA GB10")
                .expect("valid")
                .as_str(),
            "gb10"
        );
        assert!(DeviceSlug::parse("M1 Max").is_err());
        assert!(DeviceSlug::parse("../x").is_err());
    }

    #[test]
    fn the_fingerprint_ignores_file_order() {
        let a = Sha256::parse(&"a".repeat(64)).expect("valid");
        let b = Sha256::parse(&"b".repeat(64)).expect("valid");
        let one = variant_fingerprint([(Role::Base, "m.gguf", &a), (Role::Vision, "v.gguf", &b)]);
        let two = variant_fingerprint([(Role::Vision, "v.gguf", &b), (Role::Base, "m.gguf", &a)]);
        assert_eq!(one, two);
        let other = variant_fingerprint([(Role::Base, "m.gguf", &b), (Role::Vision, "v.gguf", &a)]);
        assert_ne!(one, other);
    }

    /// Every result names its kind, and a chat result with a field it does
    /// not have is refused rather than silently dropped.
    #[test]
    fn results_are_tagged_by_kind() {
        let chat = BenchResult::Chat(ChatResult {
            prompt_tokens: 512,
            decode_tokens: 128,
            prefill_tokens_per_second: 300.0,
            decode_tokens_per_second: 28.5,
            time_to_first_token_ms: 1700.0,
        });
        let json = serde_json::to_value(&chat).expect("serializes");
        assert_eq!(json["kind"], "chat");
        let back: BenchResult = serde_json::from_value(json.clone()).expect("round trip");
        assert_eq!(back, chat);
        let mut extra = json;
        extra["seconds_per_image"] = serde_json::json!(1.0);
        assert!(serde_json::from_value::<BenchResult>(extra).is_err());
    }

    #[test]
    fn every_kind_round_trips_under_its_own_tag() {
        let all = [
            BenchResult::Decision(DecisionResult {
                prompt_tokens: 512,
                latency_ms: 120.0,
                prompt_tokens_per_second: 4266.67,
            }),
            BenchResult::Image(ImageResult {
                width: 1024,
                height: 1024,
                steps: 4,
                seconds_per_image: 3.5,
            }),
            BenchResult::Transcription(TranscriptionResult {
                clip: "synthesized speech".to_string(),
                audio_seconds: 58.2,
                wall_seconds: 2.1,
                realtime_multiple: 27.71,
                first_text_ms: 640.0,
            }),
        ];
        for (r, tag) in all.iter().zip(["decision", "image", "transcription"]) {
            let json = serde_json::to_value(r).expect("serializes");
            assert_eq!(json["kind"], tag);
            assert_eq!(
                &serde_json::from_value::<BenchResult>(json).expect("parses"),
                r
            );
        }
    }

    /// Drives the transcription runner against a server already serving a
    /// recognizer. The variant checks and the record file are not part of
    /// it: the point is the request, the stream parse and the clip length.
    #[tokio::test]
    #[ignore = "needs `shiso serve --asr-model`; set SHISO_TEST_BENCH_ASR_URL"]
    #[expect(
        clippy::print_stdout,
        reason = "the skip and the result are the test's report"
    )]
    async fn transcription_runner_against_a_live_server() {
        let Ok(url) = std::env::var("SHISO_TEST_BENCH_ASR_URL") else {
            println!("skipped: SHISO_TEST_BENCH_ASR_URL is not set");
            return;
        };
        let server = client::Server::connect(&url, BenchKind::Transcription)
            .await
            .expect("the server lists a recognizer");
        let desc = BenchDesc {
            repo: PathBuf::new(),
            model: ModelName::parse("asr").expect("valid"),
            variant: None,
            server_url: url,
            device: None,
            task: None,
            prompt_tokens: Vec::new(),
            decode_tokens: 0,
            image_sizes: Vec::new(),
            audio: std::env::var_os("SHISO_TEST_BENCH_AUDIO").map(PathBuf::from),
            runs: 1,
            home: PathBuf::new(),
            shiso_version: None,
            notes: None,
            log: None,
        };
        let results = transcription::run(&server, &desc, None)
            .await
            .expect("the runner measures");
        let [BenchResult::Transcription(r)] = results.as_slice() else {
            panic!("one transcription result, got {results:?}");
        };
        assert!(r.audio_seconds > 1.0 && r.realtime_multiple > 0.0, "{r:?}");
        assert!(r.first_text_ms <= r.wall_seconds * 1000.0, "{r:?}");
        println!("{r:?}");
    }

    #[test]
    fn the_bench_follows_the_recipe_task() {
        let recipe = |tasks: Vec<Task>| -> Recipe {
            let mut r: Recipe =
                serde_json::from_str(include_str!("../../tests/fixtures/qwen3.5-9b.json"))
                    .expect("fixture parses");
            r.tasks = tasks;
            r
        };
        assert_eq!(
            Target::kind(&recipe(vec![Task::Decision]), None).expect("one task"),
            BenchKind::Decision
        );
        assert_eq!(
            Target::kind(&recipe(vec![Task::ImageGeneration]), None).expect("one task"),
            BenchKind::Image
        );
        // A task no bench measures is refused, and so is a guess between two.
        assert!(Target::kind(&recipe(vec![Task::Embedding]), None).is_err());
        let both = recipe(vec![Task::Chat, Task::Transcription]);
        assert!(Target::kind(&both, None).is_err());
        assert_eq!(
            Target::kind(&both, Some(Task::Transcription)).expect("asked"),
            BenchKind::Transcription
        );
        assert!(Target::kind(&both, Some(Task::Decision)).is_err());
    }
}
