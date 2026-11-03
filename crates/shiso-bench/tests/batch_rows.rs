//! A decode batch gives each session the logits it gets alone.
//!
//! The scheduler packs every live session into one decode step, so a session's
//! answer must not depend on how many others share the step: the same prompt
//! sampled greedily must say the same thing whether the server is idle or busy.
//! This holds only because every projection a decode row passes through runs
//! one arithmetic chain at every batch width, and this is what checks it on a
//! real checkpoint, byte for byte, at every width up to the decode batch.
//!
//! ```text
//! SHISO_TEST_MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf \
//! SHISO_TEST_MODEL_MLX4=/path/to/Qwen3.5-4B-MLX-4bit \
//!   cargo test -p shiso-bench --release --test batch_rows -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use shiso_bench::decode::{Rig, first_batch_divergence};

/// Steps each width decodes.
const STEPS: u32 = 6;

/// The checkpoint `var` names, or a note on stderr saying why the case did not run.
#[expect(
    clippy::print_stderr,
    reason = "a skipped device case has to say so; it reads exactly like a pass"
)]
fn checkpoint(var: &str) -> Option<PathBuf> {
    let Some(path) = std::env::var_os(var).map(PathBuf::from) else {
        eprintln!("SKIP: {var} names no checkpoint");
        return None;
    };
    if !path.exists() {
        eprintln!("SKIP: {var} names {}, which is not there", path.display());
        return None;
    }
    Some(path)
}

/// Every width from two to eight rows agrees with its rows decoded alone.
fn every_width_matches_alone(path: &Path) {
    // A directory is a snapshot the profiler opens the same way; a file is a
    // GGUF container read against the recorded shape profile.
    let mut rig = if path.is_dir() {
        Rig::open_qwen35moe(path)
    } else {
        Rig::open(path)
    }
    .expect("the checkpoint opens");
    rig.set_prompt_len(48);
    for rows in 2..=8 {
        let first = first_batch_divergence(&mut rig, rows, STEPS).expect("the decode runs");
        assert_eq!(
            first,
            None,
            "{}: a batch of {rows} gave (step, row) {first:?} logits other than the row alone",
            path.display()
        );
    }
}

/// A `Q4_K_M` container: `q4_K`, `q5_K` and `q6_K` projections and a `q6_K` head.
#[test]
#[ignore = "requires a Metal device and a Qwen 3.5 GGUF checkpoint"]
fn a_k_quant_batch_gives_each_row_its_solo_logits() {
    let Some(path) = checkpoint("SHISO_TEST_MODEL") else {
        return;
    };
    every_width_matches_alone(&path);
}

/// An MLX 4-bit directory: the row-exact affine family and its head.
#[test]
#[ignore = "requires a Metal device and a Qwen 3.5 MLX 4-bit checkpoint"]
fn an_affine_batch_gives_each_row_its_solo_logits() {
    let Some(path) = checkpoint("SHISO_TEST_MODEL_MLX4") else {
        return;
    };
    every_width_matches_alone(&path);
}
