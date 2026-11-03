//! Every expert entry `crate::moe` can name is defined by a source that ships.
//!
//! [`shiso_engine::moe::Moe::new`] checks `expert_entry`'s **name table**, not a
//! compiled image, so a combination that gains a row there before it gains a
//! kernel is accepted while the checkpoint is being read and refused at the
//! fifth launch of a decode, by the driver, in a message about a function that
//! could not be resolved. This binary turns that into a build failure.
//!
//! # Why this is a new test and not an extension of the sparse one
//!
//! `tests/sparse_kernels.rs` already has
//! `every_entry_the_pass_can_dispatch_is_in_the_kernel_set_s_own_table`, and an
//! earlier revision of this family's plan said to extend it. Neither half of
//! that test can see these entries:
//!
//!   * it walks the **sparse** family's plan, and `shiso_engine::sparse::moe`
//!     never dispatches `shiso_engine::moe`'s entries -- the two families share
//!     every kernel that does not touch a weight and no host code at all;
//!   * it checks names against `kernels/cuda/q4e/KERNEL_NAMES.md`, a table of
//!     the CUDA q4e set, which no Metal entry would ever appear in.
//!
//! So this file walks `expert_entry`'s own domain instead -- every
//! `(stage, codec, dtype)` triple, rather than a list somebody copied out of it
//! -- and looks for each name in the sources the two backends actually compile.
//! A seventh combination added to `expert_entry` is covered the day it lands.
//!
//! # It is deliberately backend-blind
//!
//! A name satisfies this file if **either** backend defines it, so
//! `..._nvfp4_f32` -- which exists in `kernels/cuda/qwen35moe/experts.cuh` and
//! in no `.metal` file -- passes on a Metal-only build, and an NVFP4 checkpoint
//! there still clears `Moe::new` and dies at the launch. That is not an
//! oversight: `expert_entry` is backend-blind too, so a per-backend check here
//! would be asserting something the function it grades does not claim. Making
//! the pair (backend, entry) a checked thing is a change to `Moe::new`, and
//! this file would follow it rather than lead it.

use std::path::{Path, PathBuf};

use shiso_engine::moe::{ActDtype, ExpertCodec, ExpertStageKind, expert_entry};

/// Every stage an entry can run.
const STAGES: [ExpertStageKind; 2] = [ExpertStageKind::GateUp, ExpertStageKind::Down];
/// Every codec an expert stack can be stored in.
const CODECS: [ExpertCodec; 2] = [ExpertCodec::Nvfp4, ExpertCodec::Q4Affine];
/// Every float the routed activations can cross a launch boundary in.
const DTYPES: [ActDtype; 2] = [ActDtype::F32, ActDtype::Bf16];

/// The workspace directory holding this family's sources for one backend.
fn kernel_dir(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

/// Every source of one directory with any of `extensions`, concatenated.
///
/// Reading the whole directory rather than naming the files is deliberate: a
/// kernel moved into a new file beside the others must not make this test stop
/// looking at it.
///
/// # Panics
///
/// When the directory is not readable or holds no matching source. Both mean
/// this test is checking nothing, which is worse than failing.
fn sources(relative: &str, extensions: &[&str]) -> String {
    let dir = kernel_dir(relative);
    let entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("{} is readable: {err}", dir.display()));
    let mut text = String::new();
    for entry in entries {
        let path = entry
            .unwrap_or_else(|err| panic!("{} lists: {err}", dir.display()))
            .path();
        let matches = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| extensions.contains(&e));
        if !matches {
            continue;
        }
        text.push_str(
            &std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("{} is readable: {err}", path.display())),
        );
        text.push('\n');
    }
    assert!(
        !text.is_empty(),
        "{} holds no {extensions:?} source, so this test grades nothing",
        dir.display()
    );
    text
}

/// Whether `name` is *defined* -- not merely mentioned -- by `text`.
///
/// The spelling differs by backend and both are the definition rather than a
/// reference to one: CUDA writes `__global__ void <name>(` and MSL writes
/// `kernel void <name>(`. A comment naming the entry does not satisfy either,
/// which is the point: the question is whether the image will hold it.
fn defines(text: &str, name: &str) -> bool {
    text.contains(&format!("__global__ void {name}("))
        || text.contains(&format!("kernel void {name}("))
}

#[test]
fn every_name_expert_entry_can_return_is_defined_by_a_source_that_ships() {
    let cuda = sources("kernels/cuda/qwen35moe", &["cuh", "cu"]);
    let metal = sources("kernels/metal/qwen35moe", &["metal"]);

    let mut found = 0usize;
    for stage in STAGES {
        for codec in CODECS {
            for dtype in DTYPES {
                let Some(name) = expert_entry(stage, codec, dtype) else {
                    continue;
                };
                assert!(
                    defines(&cuda, name) || defines(&metal, name),
                    "{name} is what expert_entry returns for {stage:?}/{}/{}, and no \
                     qwen35moe source defines it -- a launch of it would fail in the \
                     driver rather than here",
                    codec.name(),
                    dtype.name(),
                );
                found += 1;
            }
        }
    }
    // A floor, so that an `expert_entry` that started returning `None` for
    // everything would fail here rather than pass vacuously.
    assert_eq!(
        found, 6,
        "expert_entry named {found} entries, not the six this family builds"
    );
}

#[test]
fn no_two_combinations_share_an_entry_name() {
    // Metal resolves an entry by name out of ONE flat library and ignores the
    // module string a `KernelRef` carries, so two combinations that spelled
    // themselves alike would silently run the same kernel -- an NVFP4 stack
    // through the affine body, say, which finishes and writes plausible
    // numbers.
    let mut seen: Vec<&'static str> = Vec::new();
    for stage in STAGES {
        for codec in CODECS {
            for dtype in DTYPES {
                if let Some(name) = expert_entry(stage, codec, dtype) {
                    assert!(
                        !seen.contains(&name),
                        "{name} is returned for more than one (stage, codec, dtype)"
                    );
                    seen.push(name);
                }
            }
        }
    }
}
