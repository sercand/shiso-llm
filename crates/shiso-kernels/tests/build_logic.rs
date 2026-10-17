//! Tests for the build script's logic.
//!
//! A build script's own `#[cfg(test)]` modules never run: cargo compiles the
//! script, runs it, and discards the binary. So the logic lives in files an
//! integration test can pull in directly, and this is where it gets exercised.
//! Everything here runs on a machine with no device compiler of any kind —
//! fixtures in temporary directories, and nothing that spawns a compiler.

#[path = "../build_codegen.rs"]
mod build_codegen;
#[path = "../build_plan.rs"]
mod build_plan;

use std::path::{Path, PathBuf};

use build_codegen::{HEADER, content_hash, generate_registry, write_stub_registry};
use build_plan::{
    Lane, compose_nvcc_argv, compose_nvcc_flags, job_key, parse_manifests, plan_nvcc_jobs,
    select_lane,
};

/// Write a two-directory kernel tree into `root` and return it.
///
/// `alpha` takes the tree defaults; `beta` overrides the architecture and adds
/// its own flags, which is the shape every real manifest pair has.
fn fixture(root: &Path) -> PathBuf {
    write(
        &root.join("KERNEL.toml"),
        r#"
[build]
std = "c++17"
base_flags = ["-O3", "--use_fast_math"]
arch = ["compute_75"]
outputs = ["ptx"]
"#,
    );
    write(
        &root.join("cuda/alpha/KERNEL.toml"),
        r#"
[build]
sources = ["one.cu", "two.cu"]
includes = ["cuda/alpha"]
[modules]
one = "alpha_one"
"#,
    );
    write(
        &root.join("cuda/beta/KERNEL.toml"),
        r#"
[build]
sources = ["three.cu"]
arch = ["compute_80"]
defines = ["BETA=1"]
extra_nvcc_flags = ["--fmad=false"]
"#,
    );
    for unit in [
        "cuda/alpha/one.cu",
        "cuda/alpha/two.cu",
        "cuda/beta/three.cu",
    ] {
        write(&root.join(unit), "__global__ void k() {}\n");
    }
    root.to_path_buf()
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().expect("fixture paths have a parent"))
        .expect("create fixture directory");
    std::fs::write(path, text).expect("write fixture file");
}

// ── Manifest parsing and merging ────────────────────────────────────────────

#[test]
fn defaults_flow_down_and_directory_keys_win() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = fixture(tmp.path());
    let plan = parse_manifests(&root).expect("fixture parses");

    assert_eq!(plan.dirs.len(), 2, "both directories are discovered");
    let alpha = &plan.dirs[0];
    let beta = &plan.dirs[1];
    assert_eq!(alpha.rel, "cuda/alpha");
    assert_eq!(beta.rel, "cuda/beta");

    // Inherited.
    assert_eq!(alpha.build.std, "c++17");
    assert_eq!(alpha.build.base_flags, ["-O3", "--use_fast_math"]);
    assert_eq!(alpha.build.arch, ["compute_75"]);
    assert_eq!(beta.build.base_flags, ["-O3", "--use_fast_math"]);

    // Overridden, and the override replaces rather than extends.
    assert_eq!(beta.build.arch, ["compute_80"]);
    assert_eq!(beta.build.defines, ["BETA=1"]);
    assert!(alpha.build.defines.is_empty());

    // `[modules]` maps a stem; an unlisted stem is its own module name.
    assert_eq!(alpha.module_of("one"), "alpha_one");
    assert_eq!(alpha.module_of("two"), "two");

    // `sources` semantics.
    assert_eq!(alpha.build.sources, ["one.cu", "two.cu"]);
    assert_eq!(beta.build.sources, ["three.cu"]);
}

#[test]
fn an_unknown_key_is_a_build_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = fixture(tmp.path());
    write(
        &root.join("cuda/beta/KERNEL.toml"),
        "[build]\nsources = [\"three.cu\"]\nextra_nvc_flags = []\n",
    );
    let err = parse_manifests(&root).expect_err("a typo must not be silently dropped");
    assert!(err.contains("extra_nvc_flags"), "{err}");
    assert!(err.contains("unknown [build] key"), "{err}");
}

#[test]
fn a_module_name_may_not_be_claimed_twice() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = fixture(tmp.path());
    write(
        &root.join("cuda/beta/KERNEL.toml"),
        "[build]\nsources = [\"three.cu\"]\n[modules]\nthree = \"alpha_one\"\n",
    );
    let err = parse_manifests(&root).expect_err("a duplicate registry key is an error");
    assert!(err.contains("alpha_one"), "{err}");
}

#[test]
fn the_real_manifest_tree_parses() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the crate sits two directories below the workspace root")
        .join("kernels");
    let plan = parse_manifests(&root).expect("the shipped manifests parse");
    assert!(
        plan.manifests.contains(&root.join("KERNEL.toml")),
        "the tree-default manifest is watched for changes too"
    );
    assert_eq!(
        plan.manifests.len(),
        plan.dirs.len() + 1,
        "one manifest per directory, plus the tree default"
    );
    let modules: Vec<String> = plan
        .dirs
        .iter()
        .flat_map(|dir| {
            dir.build
                .sources
                .iter()
                .map(|s| dir.module_of(build_plan::stem_of(s)))
        })
        .collect();
    // The whole registry, exactly. These strings become identifiers a caller
    // writes at a launch site, so they are pinned as a set rather than spot
    // checked: a name silently gained or renamed is a caller that stops
    // resolving, and the lane that would notice needs a device compiler.
    let mut sorted = modules.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        [
            "audio",
            "dense",
            "exl3",
            "generic",
            "ggml_stubs",
            "mmid",
            "mmq",
            "mmq_d2r",
            "mmvq",
            "q4e_embed",
            "q4e_exact",
            "q4e_gdn",
            "q4e_gdn_tc",
            "q4e_hc",
            "q4e_idx",
            "q4e_kv8",
            "q4e_matvec",
            "q4e_misc",
            "q4e_modelopt",
            "q4e_moe",
            "q4e_mxfp8",
            "q4e_ple",
            "q4e_qsa",
            "q4e_qsa_sparse",
            "quantize",
            "qwen35",
            "qwen35moe",
            "repack",
            "sample",
            "ternary",
            "vision",
            "vision_attn",
            "vision_embed",
            "vision_norm"
        ],
        "the shipped registry changed"
    );

    // The two values the shipped manifests exist to state. They are pinned here
    // because the compile lane that would notice them being wrong only runs on a
    // machine with a device compiler.
    let trellis = plan
        .dirs
        .iter()
        .find(|dir| dir.rel == "cuda/exl3")
        .expect("the trellis directory is in the tree");
    assert_eq!(
        trellis.build.arch,
        ["compute_80"],
        "these kernels have no Turing form and must not take the tree default"
    );
    let quantized = plan
        .dirs
        .iter()
        .find(|dir| dir.rel == "cuda/mmq")
        .expect("the quantised matmul directory is in the tree");
    assert_eq!(quantized.build.arch, ["compute_75"], "inherits the default");
    assert_eq!(quantized.build.includes, ["cuda/mmq"]);
    assert_eq!(quantized.build.sources.len(), 7);
}

// ── Dedup keying ────────────────────────────────────────────────────────────

#[test]
fn an_identical_job_compiles_once_and_is_copied_once() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path().join("kernels");
    // Two directories naming the same source, with the same flags and arch.
    write(
        &root.join("KERNEL.toml"),
        "[build]\nstd = \"c++17\"\narch = [\"compute_75\"]\noutputs = [\"ptx\"]\n",
    );
    for (dir, module) in [("cuda/a", "a_shared"), ("cuda/b", "b_shared")] {
        write(
            &root.join(dir).join("KERNEL.toml"),
            &format!("[build]\nsources = [\"../shared.cu\"]\n[modules]\nshared = \"{module}\"\n"),
        );
    }
    write(&root.join("cuda/shared.cu"), "__global__ void k() {}\n");

    let plan = parse_manifests(&root).expect("parses");
    let jobs = plan_nvcc_jobs(&plan, Path::new("/out"), None).expect("plans");
    assert_eq!(jobs.compile.len(), 1, "the identical triple compiles once");
    assert_eq!(jobs.copy.len(), 1, "the repeat becomes a copy");
    assert_eq!(jobs.registry.len(), 2, "both modules are still registered");

    // One extra define splits the key.
    write(
        &root.join("cuda/b/KERNEL.toml"),
        "[build]\nsources = [\"../shared.cu\"]\ndefines = [\"B=1\"]\n[modules]\nshared = \"b_shared\"\n",
    );
    let plan = parse_manifests(&root).expect("parses");
    let jobs = plan_nvcc_jobs(&plan, Path::new("/out"), None).expect("plans");
    assert_eq!(
        jobs.compile.len(),
        2,
        "a different define is a different job"
    );
    assert!(jobs.copy.is_empty());
}

#[test]
fn the_job_key_ignores_flag_order_but_not_flag_content() {
    let src = Path::new("/k/x.cu");
    let a = ["-O3".to_string(), "-DX=1".to_string()];
    let b = ["-DX=1".to_string(), "-O3".to_string()];
    let c = ["-DX=2".to_string(), "-O3".to_string()];
    assert_eq!(
        job_key(src, "compute_75", &a),
        job_key(src, "compute_75", &b)
    );
    assert_ne!(
        job_key(src, "compute_75", &a),
        job_key(src, "compute_75", &c)
    );
    assert_ne!(
        job_key(src, "compute_75", &a),
        job_key(src, "compute_80", &a)
    );
}

// ── Flag composition ────────────────────────────────────────────────────────

#[test]
fn the_nvcc_argv_order_is_pinned() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = fixture(tmp.path());
    let plan = parse_manifests(&root).expect("parses");
    let beta = &plan.dirs[1];
    let flags = compose_nvcc_flags(beta, &plan.root, &[]);
    let argv = compose_nvcc_argv(
        Path::new("/k/cuda/beta/three.cu"),
        "compute_80",
        &flags,
        Path::new("/out/three.ptx"),
    );
    assert_eq!(
        argv,
        [
            "--ptx",
            "-arch=compute_80",
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-DBETA=1",
            "--fmad=false",
            "/k/cuda/beta/three.cu",
            "-o",
            "/out/three.ptx",
        ]
    );
}

#[test]
fn include_directories_resolve_against_the_kernel_root() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = fixture(tmp.path());
    let plan = parse_manifests(&root).expect("parses");
    let flags = compose_nvcc_flags(&plan.dirs[0], &plan.root, &[]);
    assert!(
        flags.contains(&format!("-I{}", root.join("cuda/alpha").display())),
        "{flags:?}"
    );
}

#[test]
fn a_family_specific_architecture_adds_the_fp4_define() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = fixture(tmp.path());
    let plan = parse_manifests(&root).expect("parses");

    let generic =
        plan_nvcc_jobs(&plan, Path::new("/out"), Some(&["sm_120".to_string()])).expect("plans");
    assert!(
        !generic.compile[0]
            .flags
            .iter()
            .any(|f| f.contains("HAVE_MXF4")),
        "plain sm_120 rejects the block-scaled instructions, so it must not select them"
    );

    let family =
        plan_nvcc_jobs(&plan, Path::new("/out"), Some(&["sm_120a".to_string()])).expect("plans");
    for job in &family.compile {
        assert!(
            job.flags.contains(&"-DSHISO_CUDA_HAVE_MXF4=1".to_string()),
            "every job gets the define, not just the one that named the arch: {:?}",
            job.flags
        );
        assert_eq!(
            job.arch, "sm_120a",
            "the override replaces the manifest list"
        );
    }
}

// ── Lane selection and the stub path ────────────────────────────────────────

#[test]
fn skip_wins_over_every_other_selector() {
    let env = |key: &str| -> Option<String> {
        match key {
            "SHISO_SKIP_BUILD" | "SHISO_CUDA_COMPILE" => Some("1".to_string()),
            "SHISO_CUDA_IMPL" => Some("nvidia".to_string()),
            _ => None,
        }
    };
    assert!(matches!(select_lane(&env), Lane::Skip(_)));
}

#[test]
fn a_compile_must_be_asked_for_and_an_unknown_implementation_is_refused() {
    let nothing = |_: &str| None;
    assert!(matches!(select_lane(&nothing), Lane::Skip(_)));

    let compile = |key: &str| (key == "SHISO_CUDA_COMPILE").then(|| "1".to_string());
    assert_eq!(select_lane(&compile), Lane::Nvidia);

    let explicit = |key: &str| match key {
        "SHISO_CUDA_COMPILE" => Some("1".to_string()),
        "SHISO_CUDA_IMPL" => Some("nvidia".to_string()),
        _ => None,
    };
    assert_eq!(select_lane(&explicit), Lane::Nvidia);

    // A spelling this script no longer understands must not be compiled as
    // nvidia by default; it has to be refused and said out loud.
    let gone = |key: &str| match key {
        "SHISO_CUDA_COMPILE" => Some("1".to_string()),
        "SHISO_CUDA_IMPL" => Some("cumetal".to_string()),
        _ => None,
    };
    let Lane::Skip(reason) = select_lane(&gone) else {
        panic!("an unknown implementation must not select a compile lane");
    };
    assert!(reason.contains("cumetal"), "{reason}");
}

#[test]
fn the_stub_path_writes_an_empty_registry() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let generated = write_stub_registry(tmp.path()).expect("the stub is written");
    let text = std::fs::read_to_string(tmp.path().join("target_ptx.rs")).expect("stub exists");
    assert!(text.starts_with(HEADER), "the header line is missing");
    assert!(!text.contains("include_bytes!"), "the stub embeds nothing");
    // The accessors are still generated, and an empty table means every lookup
    // resolves to the empty slice rather than to a panic.
    assert!(generated.contains("pub fn ptx(module: &str) -> &'static [u8]"));
    assert!(generated.contains("static PTX_MODULES: &[(&str, &[u8])] = &[\n];"));
    assert!(generated.contains("found.map_or(&[], |&(_, bytes)| bytes)"));
}

/// The registry, populated, must be valid Rust.
///
/// The empty path is exercised by every build on every machine; the populated
/// path is only ever taken where a device compiler exists, which is one box
/// nobody develops on. A generator bug on that branch — a stray brace, a
/// constant name that is not an identifier — would therefore surface for the
/// first time on the Linux box, at the end of a long kernel compile. Handing
/// the generated text to `rustc` here costs a fraction of a second and moves
/// that discovery to this machine. Real files back the `include_bytes!` calls,
/// so the embedding is checked too, not just the syntax.
#[test]
fn a_populated_registry_is_valid_rust() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut entries = Vec::new();
    for (module, bytes) in [("exl3", &b".version 8.0"[..]), ("mmq_d2r", &b"MTLB"[..])] {
        let artefact = tmp.path().join(format!("{module}.bin"));
        std::fs::write(&artefact, bytes).expect("write artefact");
        entries.push((module.to_string(), artefact));
    }
    let source = tmp.path().join("registry.rs");
    std::fs::write(&source, generate_registry(&entries)).expect("write generated registry");

    let out = std::process::Command::new(std::env::var_os("RUSTC").unwrap_or("rustc".into()))
        .args([
            "--edition",
            "2024",
            "--crate-type",
            "lib",
            "--emit",
            "metadata",
        ])
        .arg("-o")
        .arg(tmp.path().join("registry.rmeta"))
        .arg(&source)
        .output()
        .expect("run rustc");
    assert!(
        out.status.success(),
        "the generated registry does not compile:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ── Codegen shape ───────────────────────────────────────────────────────────

#[test]
fn every_module_appears_in_the_generated_lookup_table() {
    let entries = vec![
        ("exl3".to_string(), PathBuf::from("/out/ptx/exl3.ptx")),
        ("mmq_d2r".to_string(), PathBuf::from("/out/ptx/mmq_d2r.ptx")),
    ];
    let generated = generate_registry(&entries);
    assert!(generated.starts_with(HEADER));
    assert!(generated.contains("const EXL3_PTX: &[u8] = include_bytes!(\"/out/ptx/exl3.ptx\")"));
    assert!(generated.contains("(\"exl3\", EXL3_PTX)"));
    assert!(generated.contains("(\"mmq_d2r\", MMQ_D2R_PTX)"));
    assert!(generated.contains("#[must_use]"));
    assert!(generated.contains("pub fn ptx_modules() -> Vec<(&'static str, &'static [u8])>"));
}

#[test]
fn the_hash_changes_only_when_the_content_does() {
    let a = generate_registry(&[("x".to_string(), PathBuf::from("/o/x.ptx"))]);
    let b = generate_registry(&[("x".to_string(), PathBuf::from("/o/x.ptx"))]);
    let c = generate_registry(&[("y".to_string(), PathBuf::from("/o/y.ptx"))]);
    assert_eq!(content_hash(&a), content_hash(&b));
    assert_ne!(content_hash(&a), content_hash(&c));
    assert_eq!(content_hash(&a).len(), 12);
}

// ── Linux, with a real compiler ─────────────────────────────────────────────

/// The real nvcc lane, which only exists on the Linux box.
///
/// Ignored by default and run by hand there, because it needs a CUDA toolkit
/// and the compiled kernel sources on disk. What it checks is that every unit
/// produced a non-empty PTX file carrying a `.version` directive — the cheapest
/// sniff that distinguishes loadable PTX from a file nvcc created and then
/// failed to fill.
#[test]
#[ignore = "requires nvcc and a CUDA toolkit; run on the Linux box"]
#[cfg(target_os = "linux")]
fn every_compiled_module_is_loadable_ptx() {
    let modules = shiso_kernels::ptx_modules();
    assert!(
        !modules.is_empty(),
        "a real compile registers at least one module"
    );
    for (name, bytes) in modules {
        assert!(!bytes.is_empty(), "{name} registered zero bytes");
        let text = std::str::from_utf8(bytes).unwrap_or_else(|e| panic!("{name} is not text: {e}"));
        assert!(
            text.contains(".version"),
            "{name} carries no PTX version directive"
        );
    }
}
