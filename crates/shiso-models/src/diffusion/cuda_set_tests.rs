//! The CUDA `diffusion` module carries every entry the driver launches.
//!
//! Host-only: no device, no compiler. The driver refuses a backend whose module
//! lacks any entry in `REQUIRED_KERNELS` (`Ops::new`), so on a CUDA machine a
//! missing entry means FLUX.3 and the other `DiT` families do not load at all.
//! That is the failure this file moves to `cargo test` on any machine, by
//! reading the sources the CUDA build compiles (`kernels/cuda/diffusion`) and
//! the name table beside them.
//!
//! What this does NOT show: that the sources compile, or that an entry computes
//! the right numbers. Both need nvcc and a device (`SHISO_CUDA_COMPILE=1`, then
//! `cargo test -p shiso-cuda --test diffusion_kernels -- --ignored`).

use std::collections::BTreeSet;
use std::path::PathBuf;

use super::ops::{MODULE, REQUIRED_KERNELS};

/// `kernels/cuda/diffusion`, from this crate's manifest directory.
fn kernel_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels/cuda/diffusion")
}

/// Drops `/* */` and `//` comments, so a name that only a comment mentions does
/// not count as an entry.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(at) = rest.find(['/']) {
        out.push_str(&rest[..at]);
        let tail = &rest[at..];
        if let Some(body) = tail.strip_prefix("/*") {
            rest = body.split_once("*/").map_or("", |(_, after)| after);
        } else if let Some(body) = tail.strip_prefix("//") {
            rest = body.split_once('\n').map_or("", |(_, after)| after);
        } else {
            out.push('/');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Every `extern "C" __global__ void kernel_dit_*(` the module's sources
/// define, whitespace-normalized so a wrapped signature still reads.
fn defined_entries() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let dir = kernel_dir();
    for entry in std::fs::read_dir(&dir).expect("the CUDA diffusion directory exists") {
        let path = entry.expect("directory entry").path();
        let ext = path.extension().and_then(|e| e.to_str());
        if !matches!(ext, Some("cu" | "cuh")) {
            continue;
        }
        let text = strip_comments(&std::fs::read_to_string(&path).expect("source is readable"));
        let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let marker = "extern \"C\" __global__ void ";
        let mut rest = flat.as_str();
        while let Some(at) = rest.find(marker) {
            rest = &rest[at + marker.len()..];
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if name.starts_with("kernel_dit_") {
                found.insert(name);
            }
        }
    }
    found
}

#[test]
fn required_kernels_all_present_on_cuda() {
    let required: BTreeSet<String> = REQUIRED_KERNELS.iter().map(|s| (*s).to_owned()).collect();
    assert_eq!(
        required.len(),
        REQUIRED_KERNELS.len(),
        "REQUIRED_KERNELS lists a name twice"
    );
    let defined = defined_entries();
    let missing: Vec<&String> = required.difference(&defined).collect();
    assert!(
        missing.is_empty(),
        "kernels/cuda/diffusion defines no extern \"C\" entry for {missing:?}; the driver would \
         refuse a CUDA backend at load"
    );
    let orphans: Vec<&String> = defined.difference(&required).collect();
    assert!(
        orphans.is_empty(),
        "kernels/cuda/diffusion defines {orphans:?}, which REQUIRED_KERNELS does not list: add \
         them there (and to the Metal twin) or remove them"
    );
}

#[test]
fn the_cuda_name_table_and_manifest_agree_with_the_driver() {
    let dir = kernel_dir();
    let names = std::fs::read_to_string(dir.join("KERNEL_NAMES.md")).expect("name table");
    let listed: BTreeSet<&str> = names
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("kernel_dit_"))
        .collect();
    let missing: Vec<&&str> = REQUIRED_KERNELS
        .iter()
        .filter(|name| !listed.contains(**name))
        .collect();
    assert!(
        missing.is_empty(),
        "KERNEL_NAMES.md does not list {missing:?}"
    );

    // One translation unit under the module name the driver launches through:
    // a second source would be a second image, and the entries a different
    // image than the one `Ops::new` probes.
    let manifest = std::fs::read_to_string(dir.join("KERNEL.toml")).expect("manifest");
    assert!(
        manifest.contains("sources = [\"diffusion.cu\"]"),
        "the diffusion module must be exactly one translation unit"
    );
    assert!(
        manifest.contains(&format!("diffusion = \"{MODULE}\"")),
        "the manifest's module name must be the driver's `{MODULE}`"
    );
}

#[test]
fn every_piece_the_unit_includes_exists_and_every_source_is_included() {
    let dir = kernel_dir();
    let unit = std::fs::read_to_string(dir.join("diffusion.cu")).expect("unit");
    let included: BTreeSet<String> = strip_comments(&unit)
        .lines()
        .filter_map(|l| l.trim().strip_prefix("#include \""))
        .filter_map(|l| l.strip_suffix('"'))
        .map(str::to_owned)
        .collect();
    for piece in &included {
        assert!(
            dir.join(piece).is_file(),
            "diffusion.cu includes missing {piece}"
        );
    }
    for entry in std::fs::read_dir(&dir).expect("directory") {
        let path = entry.expect("entry").path();
        let ext = path.extension().and_then(|e| e.to_str());
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if matches!(ext, Some("cu" | "cuh")) && name != "diffusion.cu" {
            assert!(
                included.contains(name),
                "{name} is in the directory but not included by diffusion.cu, so it is never built"
            );
        }
    }
}
