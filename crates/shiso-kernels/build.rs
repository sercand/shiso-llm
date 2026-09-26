//! Compiles the CUDA kernel tree and generates the module registry the library
//! embeds.
//!
//! The default on every machine is to compile nothing. A device compiler is a
//! multi-gigabyte install that exists on exactly one box in this project, so a
//! build that needed one would make `cargo test` unrunnable everywhere else.
//! Instead the script writes an empty registry whose accessors compile and
//! answer with empty slices, and a real compile is asked for explicitly:
//!
//! - `SHISO_SKIP_BUILD=1` — never compile. Wins over everything below, so a
//!   checkout with no toolchain at all still builds and tests.
//! - `SHISO_CUDA_COMPILE=1` — required before any compiler is invoked.
//! - `SHISO_CUDA_IMPL=nvidia` — which compiler. The only accepted value; the
//!   knob is what a second backend would be selected by.
//! - `SHISO_CUDA_ARCH` — comma-separated architectures, replacing what the
//!   manifests ask for.
#[path = "build_codegen.rs"]
mod build_codegen;
#[path = "build_plan.rs"]
mod build_plan;
#[path = "build_target.rs"]
mod build_target;

use std::path::{Path, PathBuf};

use build_codegen::{content_hash, write_registry, write_stub_registry};
use build_plan::{Lane, LanePlan, select_lane};
use build_target::{CompileTarget, NvccTarget};

/// Every selector the script reads. Each one is declared to cargo, because a
/// cached build would otherwise keep a registry built under a previous
/// selection: the inputs cargo watches by default are files, and none of these
/// is a file.
const ENV_KNOBS: &[&str] = &[
    "SHISO_SKIP_BUILD",
    "SHISO_CUDA_COMPILE",
    "SHISO_CUDA_IMPL",
    "SHISO_CUDA_ARCH",
    "CUDA_HOME",
];

fn main() {
    for knob in ENV_KNOBS {
        println!("cargo:rerun-if-env-changed={knob}");
    }
    let out_dir = PathBuf::from(env("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let manifest_dir = PathBuf::from(env("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let kernels_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("the crate sits two directories below the workspace root")
        .join("kernels");
    let target_os = env("CARGO_CFG_TARGET_OS").unwrap_or_default();

    let lane = select_lane(&env);
    let combined = match lane {
        Lane::Skip(reason) => {
            println!("cargo:warning=shiso-kernels: no kernels compiled ({reason})");
            stub(&out_dir)
        }
        Lane::Nvidia => run_nvidia(&kernels_root, &out_dir, &target_os),
    };

    // The library reads both of these with `env!`. `SHISO_KERNELS_OUT_DIR` is
    // how the generated registry finds its artefacts; the hash is what makes
    // cargo rebuild the library when the registry changes, which it otherwise
    // cannot see through a build-script-generated `include!`.
    println!(
        "cargo:rustc-env=SHISO_KERNELS_OUT_DIR={}",
        out_dir.display()
    );
    println!(
        "cargo:rustc-env=SHISO_KERNEL_SET_HASH={}",
        content_hash(&combined)
    );
}

/// Read an environment variable, as a function so the lane selector can be
/// tested against a table instead of the real environment.
fn env(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// Write an empty registry, panicking if even that fails.
fn stub(out_dir: &Path) -> String {
    write_stub_registry(out_dir).unwrap_or_else(|e| panic!("shiso-kernels: {e}"))
}

/// Parse the manifest tree and ask cargo to watch every file it named.
fn load_plan(kernels_root: &Path) -> LanePlan {
    let plan =
        build_plan::parse_manifests(kernels_root).unwrap_or_else(|e| panic!("shiso-kernels: {e}"));
    for manifest in &plan.manifests {
        println!("cargo:rerun-if-changed={}", manifest.display());
    }
    // The root too, which cargo scans recursively: the directories below are
    // the ones this run found, and a kernel directory added later is a new
    // module no watch on an existing directory would notice - the build then
    // keeps embedding a registry that lacks it and every launch into it fails
    // at run time with "this build embeds no kernel module".
    println!("cargo:rerun-if-changed={}", plan.root.display());
    // Watching each directory rather than each source catches a changed header
    // too, and every file in these directories is either a unit or a header one
    // of them includes.
    for dir in &plan.dirs {
        println!(
            "cargo:rerun-if-changed={}",
            plan.root.join(&dir.rel).display()
        );
    }
    plan
}

/// Fail unless every source a manifest names exists on disk.
///
/// A manifest naming a file that is not there is a manifest bug, and a lane
/// that shrugged at it would quietly register nothing and report success.
fn require_sources(paths: impl IntoIterator<Item = PathBuf>) {
    let missing: Vec<String> = paths
        .into_iter()
        .filter(|path| !path.is_file())
        .map(|path| path.display().to_string())
        .collect();
    assert!(
        missing.is_empty(),
        "shiso-kernels: {} manifest source(s) do not exist on disk: {missing:?}",
        missing.len()
    );
}

/// The architecture list `SHISO_CUDA_ARCH` asks for, if it asks for one.
fn arch_override() -> Option<Vec<String>> {
    let raw = env("SHISO_CUDA_ARCH")?;
    let list: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    (!list.is_empty()).then_some(list)
}

/// Compile every unit to PTX and generate the shipping registry.
///
/// A compile failure is fatal here, without exception. A binary that linked
/// with a module missing from its registry would fail at the first launch of
/// that kernel, in production, with a message about a null module rather than
/// about the compile that never happened.
fn run_nvidia(kernels_root: &Path, out_dir: &Path, target_os: &str) -> String {
    let Some(nvcc) = NvccTarget::find() else {
        assert!(
            target_os == "macos",
            "shiso-kernels: SHISO_CUDA_IMPL=nvidia and SHISO_CUDA_COMPILE=1 were set, but no nvcc \
             was found. Set CUDA_HOME, or put nvcc on PATH."
        );
        // On macOS this is the expected outcome: the implementation was asked
        // for by name on a platform that has no toolkit. Degrade rather than
        // fail, so a Mac can still type-check the crate under any selection.
        println!(
            "cargo:warning=shiso-kernels: no nvcc on this machine; generating an empty registry"
        );
        return stub(out_dir);
    };

    let plan = load_plan(kernels_root);
    let jobs = build_plan::plan_nvcc_jobs(&plan, out_dir, arch_override().as_deref())
        .unwrap_or_else(|e| panic!("shiso-kernels: {e}"));
    require_sources(jobs.compile.iter().map(|job| job.source.clone()));
    // The planner names the artefacts and the target produces them; if a second
    // backend is ever wired to this planner, this is where the mismatch stops
    // rather than three steps later in a registry full of misnamed blobs.
    let extension = nvcc.output_extension();
    assert!(
        jobs.compile
            .iter()
            .all(|job| job.out.extension().is_some_and(|e| e == extension)),
        "shiso-kernels: the job planner named artefacts this compiler does not produce (.{extension})"
    );

    compile_all(&nvcc, &jobs.compile);
    for (from, to) in &jobs.copy {
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                panic!("shiso-kernels: cannot create {}: {e}", parent.display())
            });
        }
        std::fs::copy(from, to).unwrap_or_else(|e| {
            panic!(
                "shiso-kernels: cannot copy {} to {}: {e}",
                from.display(),
                to.display()
            )
        });
    }
    let total = jobs.compile.len() + jobs.copy.len();
    println!(
        "cargo:warning=shiso-kernels: {} nvcc invocation(s) for {total} artefact(s) ({} reused)",
        jobs.compile.len(),
        jobs.copy.len(),
    );
    write_registry(out_dir, &jobs.registry).unwrap_or_else(|e| panic!("shiso-kernels: {e}"))
}

/// Run the compile jobs across as many threads as there is work for.
fn compile_all(nvcc: &NvccTarget, jobs: &[build_plan::KernelJob]) {
    if jobs.is_empty() {
        return;
    }
    let workers = std::thread::available_parallelism()
        .map_or(4, std::num::NonZeroUsize::get)
        .min(jobs.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let errors = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let (next, errors, target) = (&next, &errors, nvcc);
            scope.spawn(move || {
                loop {
                    let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(job) = jobs.get(idx) else { break };
                    if let Err(e) = target.compile(&job.source, &job.arch, &job.flags, &job.out) {
                        errors.lock().expect("compile error list").push(e);
                    }
                }
            });
        }
    });
    let errors = errors.into_inner().expect("compile error list");
    assert!(
        errors.is_empty(),
        "shiso-kernels: {} kernel(s) failed to compile:\n{}",
        errors.len(),
        errors.join("\n")
    );
}
