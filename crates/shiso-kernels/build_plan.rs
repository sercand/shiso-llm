//! Kernel manifests and compile-job planning.
//!
//! Every item here is a pure function of its arguments; no compiler process is
//! spawned from this file. That split is not stylistic. A build script's own
//! `#[cfg(test)]` modules never run under `cargo test` — cargo compiles the
//! script, runs it, and throws the binary away — so the only way to test build
//! logic is to put it in a file an integration test can pull in with `#[path]`.
//! Which in turn means nothing in this file may reach back into the build
//! script through `super::`: it has to stand on its own.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Keys accepted inside `[build]`. Anything else is a build error.
const BUILD_KEYS: &[&str] = &[
    "std",
    "base_flags",
    "extra_nvcc_flags",
    "defines",
    "includes",
    "arch",
    "outputs",
    "sources",
];

/// Tables accepted at the top level of a manifest.
const TOP_KEYS: &[&str] = &["build", "modules"];

/// Architecture spellings that select the block-scaled FP4 kernel family.
///
/// These are family-specific targets: the plain `sm_120`/`sm_121` forms reject
/// the block-scaled MMA instructions the FP4 path emits, so the define is tied
/// to the `a` suffix rather than to a version comparison.
const MXF4_ARCHES: &[&str] = &["sm_120a", "sm_121a", "compute_120a", "compute_121a"];

/// The define that switches the FP4 kernel family on.
const MXF4_DEFINE: &str = "SHISO_CUDA_HAVE_MXF4=1";

/// The `[build]` section of a manifest, after merging the directory over the
/// tree defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuildConfig {
    /// C++ dialect, emitted as `-std=<value>`.
    pub std: String,
    /// Optimisation and debug flags applied to every unit.
    pub base_flags: Vec<String>,
    /// Flags appended after everything else this module composes.
    pub extra_nvcc_flags: Vec<String>,
    /// Preprocessor defines, emitted as `-D<value>`.
    pub defines: Vec<String>,
    /// Include directories relative to the kernel root, emitted as `-I`.
    pub includes: Vec<String>,
    /// Device architectures; the first is the one a plain lookup resolves.
    pub arch: Vec<String>,
    /// Requested output kinds; `ptx` is the only one the lane implements.
    pub outputs: Vec<String>,
    /// Translation units to compile. Every other file is include-only.
    pub sources: Vec<String>,
}

/// One kernel directory and its merged configuration.
#[derive(Debug, Clone)]
pub(crate) struct KernelDir {
    /// Path relative to the kernel root, slash-separated (`cuda/mmq`).
    pub rel: String,
    /// Merged `[build]` section.
    pub build: BuildConfig,
    /// Stem to module-name overrides.
    pub modules: BTreeMap<String, String>,
}

impl KernelDir {
    /// Registry key for a source stem: the override if one exists, else the
    /// stem itself.
    pub(crate) fn module_of(&self, stem: &str) -> String {
        self.modules
            .get(stem)
            .cloned()
            .unwrap_or_else(|| stem.to_string())
    }
}

/// Everything the build script learned from the manifest tree.
#[derive(Debug)]
pub(crate) struct LanePlan {
    /// Absolute path of the kernel root that holds the top-level manifest.
    pub root: PathBuf,
    /// Every manifest read, so the build script can ask cargo to watch them.
    pub manifests: Vec<PathBuf>,
    /// Kernel directories, in a stable order.
    pub dirs: Vec<KernelDir>,
}

/// One compiler invocation: a source, an architecture, and the flags that go
/// with them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KernelJob {
    /// Owning directory, relative to the kernel root.
    pub dir: String,
    /// Source file stem.
    pub stem: String,
    /// Registry key this job's output is registered under.
    pub module: String,
    /// Absolute path of the source.
    pub source: PathBuf,
    /// Architecture this invocation targets.
    pub arch: String,
    /// Flags between the architecture and the source, in argv order.
    pub flags: Vec<String>,
    /// Absolute path of the output artefact.
    pub out: PathBuf,
}

/// The work the nvcc lane has to do, split into invocations and copies.
#[derive(Debug, Default)]
pub(crate) struct JobPlan {
    /// Unique invocations, one per distinct source/architecture/flag triple.
    pub compile: Vec<KernelJob>,
    /// Outputs that are byte-identical to an invocation already in `compile`,
    /// as `(already produced, wanted)`.
    pub copy: Vec<(PathBuf, PathBuf)>,
    /// Registry entries as `(module, artefact path)`, first architecture wins.
    pub registry: Vec<(String, PathBuf)>,
}

/// Which lane this build should run, decided before any file is touched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Lane {
    /// Generate an empty registry. The string is the reason, for the build log.
    Skip(String),
    /// Compile to PTX with nvcc. The only lane there is, and the only one whose
    /// output ever ships.
    Nvidia,
}

/// Decide the lane from the environment.
///
/// The precedence is fixed and the order of the checks is the specification:
/// `SHISO_SKIP_BUILD` wins over everything, because a checkout with no device
/// compiler at all still has to build; a real compile then has to be asked for
/// explicitly with `SHISO_CUDA_COMPILE=1`, so that neither `cargo build` nor a
/// test run ever discovers it needs a toolchain it does not have. Only once
/// both of those are past does the implementation choice matter.
///
/// `SHISO_CUDA_IMPL` has exactly one accepted value today. The knob is kept
/// rather than dropped because it is what a second backend would be selected
/// by, and because a spelling this script no longer understands has to be
/// refused out loud rather than silently compiled as nvidia.
pub(crate) fn select_lane(env: &dyn Fn(&str) -> Option<String>) -> Lane {
    if matches!(env("SHISO_SKIP_BUILD").as_deref(), Some("1" | "true")) {
        return Lane::Skip("SHISO_SKIP_BUILD is set".to_string());
    }
    if env("SHISO_CUDA_COMPILE").as_deref() != Some("1") {
        return Lane::Skip("SHISO_CUDA_COMPILE=1 is required for a real compile".to_string());
    }
    match env("SHISO_CUDA_IMPL").as_deref() {
        None | Some("nvidia") => Lane::Nvidia,
        Some(other) => Lane::Skip(format!(
            "SHISO_CUDA_IMPL={other} is not a known implementation (nvidia)"
        )),
    }
}

// ── Manifest parsing ────────────────────────────────────────────────────────

/// Read the top-level manifest and every per-directory manifest below it.
///
/// Directories are discovered by walking for `KERNEL.toml` rather than listed
/// anywhere, so adding a kernel family is one file in one place. The walk is
/// sorted, so the job order — and therefore which duplicate becomes the copy —
/// does not depend on the filesystem's iteration order.
pub(crate) fn parse_manifests(root: &Path) -> Result<LanePlan, String> {
    let root_manifest = root.join("KERNEL.toml");
    let (defaults_build, defaults_modules) = read_manifest(&root_manifest)?;
    let mut manifests = vec![root_manifest];
    let mut dirs = Vec::new();
    let mut seen_modules: BTreeMap<String, String> = BTreeMap::new();

    for dir_manifest in find_manifests(root)? {
        let (build, modules) = read_manifest(&dir_manifest)?;
        let rel = dir_manifest
            .parent()
            .and_then(|p| p.strip_prefix(root).ok())
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .ok_or_else(|| format!("{} is not under {}", dir_manifest.display(), root.display()))?;

        let mut merged_modules = defaults_modules.clone();
        merged_modules.extend(modules);
        let dir = KernelDir {
            build: merge_build(&defaults_build, &build, &dir_manifest)?,
            modules: merged_modules,
            rel,
        };
        for source in &dir.build.sources {
            let module = dir.module_of(stem_of(source));
            if let Some(owner) = seen_modules.insert(module.clone(), dir.rel.clone()) {
                return Err(format!(
                    "module name {module:?} is claimed by both {owner} and {}; a module name is \
                     the registry key and must be unique",
                    dir.rel
                ));
            }
        }
        dirs.push(dir);
        manifests.push(dir_manifest);
    }
    Ok(LanePlan {
        root: root.to_path_buf(),
        manifests,
        dirs,
    })
}

/// Every `KERNEL.toml` strictly below `root`, sorted.
fn find_manifests(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut found = Vec::new();
    let mut queue = vec![root.to_path_buf()];
    while let Some(dir) = queue.pop() {
        let entries =
            std::fs::read_dir(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
            let path = entry.path();
            if path.is_dir() {
                queue.push(path);
            } else if path.file_name().is_some_and(|n| n == "KERNEL.toml") && dir != root {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}

/// Parse one manifest into its two tables, rejecting unknown keys.
type RawManifest = (toml::Table, BTreeMap<String, String>);
fn read_manifest(path: &Path) -> Result<RawManifest, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let table: toml::Table = text
        .parse()
        .map_err(|e| format!("{} is not valid TOML: {e}", path.display()))?;
    reject_unknown(&table, TOP_KEYS, path, "top level")?;

    let build = sub_table(&table, "build", path)?;
    reject_unknown(&build, BUILD_KEYS, path, "[build]")?;

    let mut modules = BTreeMap::new();
    for (stem, value) in sub_table(&table, "modules", path)? {
        let name = value
            .as_str()
            .ok_or_else(|| format!("{}: [modules].{stem} must be a string", path.display()))?;
        modules.insert(stem, name.to_string());
    }
    Ok((build, modules))
}

/// Fail on any key outside `known`, naming the file and the section.
fn reject_unknown(
    table: &toml::Table,
    known: &[&str],
    path: &Path,
    section: &str,
) -> Result<(), String> {
    let unknown: Vec<&str> = table
        .keys()
        .map(String::as_str)
        .filter(|k| !known.contains(k))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{}: unknown {section} key(s) {unknown:?}; accepted keys are {known:?}",
        path.display()
    ))
}

fn sub_table(table: &toml::Table, key: &str, path: &Path) -> Result<toml::Table, String> {
    match table.get(key) {
        None => Ok(toml::Table::new()),
        Some(toml::Value::Table(t)) => Ok(t.clone()),
        Some(_) => Err(format!("{}: [{key}] must be a table", path.display())),
    }
}

fn string_key(table: &toml::Table, key: &str, path: &Path) -> Result<Option<String>, String> {
    match table.get(key) {
        None => Ok(None),
        Some(toml::Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(format!("{}: {key} must be a string", path.display())),
    }
}

fn list_key(table: &toml::Table, key: &str, path: &Path) -> Result<Option<Vec<String>>, String> {
    let Some(value) = table.get(key) else {
        return Ok(None);
    };
    let array = value
        .as_array()
        .ok_or_else(|| format!("{}: {key} must be an array of strings", path.display()))?;
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        out.push(
            item.as_str()
                .ok_or_else(|| format!("{}: {key} must be an array of strings", path.display()))?
                .to_string(),
        );
    }
    Ok(Some(out))
}

/// Merge a directory's `[build]` over the tree defaults. A key the directory
/// sets replaces the default outright, including lists.
fn merge_build(
    defaults: &toml::Table,
    dir: &toml::Table,
    path: &Path,
) -> Result<BuildConfig, String> {
    let pick = |key: &str| -> Result<Option<String>, String> {
        Ok(string_key(dir, key, path)?.or(string_key(defaults, key, path)?))
    };
    let list = |key: &str| -> Result<Vec<String>, String> {
        Ok(list_key(dir, key, path)?
            .or(list_key(defaults, key, path)?)
            .unwrap_or_default())
    };
    Ok(BuildConfig {
        std: pick("std")?.unwrap_or_else(|| "c++17".to_string()),
        base_flags: list("base_flags")?,
        extra_nvcc_flags: list("extra_nvcc_flags")?,
        defines: list("defines")?,
        includes: list("includes")?,
        arch: list("arch")?,
        outputs: list("outputs")?,
        sources: list("sources")?,
    })
}

/// The file stem of a source name as written in a manifest.
pub(crate) fn stem_of(source: &str) -> &str {
    source.rsplit('/').next().map_or(source, |name| {
        name.split_once('.').map_or(name, |(stem, _)| stem)
    })
}

// ── Flag composition ────────────────────────────────────────────────────────

/// Everything between the architecture and the source in an nvcc argv.
///
/// The order is fixed and pinned by a test rather than left to whatever the
/// loop happens to produce, because these flags end up in the PTX header and a
/// reordering would look like a kernel change to anything diffing two builds.
pub(crate) fn compose_nvcc_flags(
    dir: &KernelDir,
    root: &Path,
    extra_defines: &[String],
) -> Vec<String> {
    let mut flags = dir.build.base_flags.clone();
    flags.push(format!("-std={}", dir.build.std));
    for include in &dir.build.includes {
        flags.push(format!("-I{}", root.join(include).display()));
    }
    for define in dir.build.defines.iter().chain(extra_defines) {
        flags.push(format!("-D{define}"));
    }
    flags.extend(dir.build.extra_nvcc_flags.iter().cloned());
    flags
}

/// The complete nvcc argument list for one job.
pub(crate) fn compose_nvcc_argv(
    source: &Path,
    arch: &str,
    flags: &[String],
    out: &Path,
) -> Vec<String> {
    let mut argv = vec!["--ptx".to_string(), format!("-arch={arch}")];
    argv.extend(flags.iter().cloned());
    argv.push(source.display().to_string());
    argv.push("-o".to_string());
    argv.push(out.display().to_string());
    argv
}

// ── Job planning ────────────────────────────────────────────────────────────

/// The identity of a compiler invocation.
///
/// The path is canonicalised, because two directories reach a shared source by
/// different relative routes and the textual paths would then differ for a file
/// that is one file. When the path does not resolve — the source is missing, and
/// the caller is about to say so — the literal path stands in, which keeps this
/// function total.
///
/// Flags are sorted, because two directories that reach the same flag set by
/// different routes produce byte-identical output and compiling it twice is
/// pure wall-clock. Sorting is safe only because every flag this module
/// composes into an nvcc invocation is order-independent to the compiler; a
/// flag that took a separate value argument would have to be keyed as a pair
/// instead, and none of the ones composed here does.
pub(crate) fn job_key(source: &Path, arch: &str, flags: &[String]) -> String {
    let canonical = source
        .canonicalize()
        .unwrap_or_else(|_| source.to_path_buf());
    let mut sorted: Vec<&str> = flags.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    format!(
        "{}\u{1f}{arch}\u{1f}{}",
        canonical.display(),
        sorted.join("\u{1f}")
    )
}

/// Plan the nvcc lane: one invocation per distinct job key, a copy for every
/// repeat, and a registry entry per module.
pub(crate) fn plan_nvcc_jobs(
    plan: &LanePlan,
    out_root: &Path,
    arch_override: Option<&[String]>,
) -> Result<JobPlan, String> {
    let extra_defines = mxf4_defines(arch_override);
    let mut jobs = JobPlan::default();
    let mut seen: BTreeMap<String, PathBuf> = BTreeMap::new();

    for dir in &plan.dirs {
        for kind in &dir.build.outputs {
            if kind != "ptx" {
                return Err(format!(
                    "{}: outputs = {kind:?} is accepted by the schema but not implemented by the \
                     compile lane; remove it rather than let the build emit PTX under a cubin name",
                    dir.rel
                ));
            }
        }
        let arches: &[String] = arch_override.unwrap_or(&dir.build.arch);
        if arches.is_empty() {
            return Err(format!("{}: no architecture to build for", dir.rel));
        }
        let flags = compose_nvcc_flags(dir, &plan.root, &extra_defines);
        for source in &dir.build.sources {
            let stem = stem_of(source).to_string();
            let module = dir.module_of(&stem);
            let path = plan.root.join(&dir.rel).join(source);
            for (idx, arch) in arches.iter().enumerate() {
                let out = out_root
                    .join("ptx")
                    .join(&dir.rel)
                    .join(format!("{stem}__{arch}.ptx"));
                if idx == 0 {
                    jobs.registry.push((module.clone(), out.clone()));
                }
                let key = job_key(&path, arch, &flags);
                if let Some(done) = seen.get(&key) {
                    jobs.copy.push((done.clone(), out));
                } else {
                    seen.insert(key, out.clone());
                    jobs.compile.push(KernelJob {
                        dir: dir.rel.clone(),
                        stem: stem.clone(),
                        module: module.clone(),
                        source: path.clone(),
                        arch: arch.clone(),
                        flags: flags.clone(),
                        out,
                    });
                }
            }
        }
    }
    Ok(jobs)
}

/// The extra defines an architecture override implies.
fn mxf4_defines(arch_override: Option<&[String]>) -> Vec<String> {
    let selects_mxf4 = arch_override.is_some_and(|list| {
        list.iter()
            .any(|arch| MXF4_ARCHES.contains(&arch.trim().trim_start_matches("arch=")))
    });
    if selects_mxf4 {
        vec![MXF4_DEFINE.to_string()]
    } else {
        Vec::new()
    }
}
