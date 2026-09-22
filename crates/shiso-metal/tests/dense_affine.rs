//! Metal's run of the shared `dense_affine` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! One case is not ignore-gated: the pipelines have to resolve, and a missing or
//! misnamed entry point should fail the default gate rather than hide behind the
//! GPU gate. The rest are `#[ignore]`d: `cargo test -p shiso-metal --features
//! metal --test dense_affine -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::affine;
use shiso_metal::MetalBackend;

/// Opens the device, or says on stderr why the cases did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<Gpu> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the GPU cases");
        return None;
    }
    let metal = MetalBackend::new().expect("the embedded kernel library compiles");
    let device: Arc<dyn Backend> = metal.clone();
    Some(Gpu::new(device, Box::new(metal)))
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "a machine with no Metal device skips rather than fails, and the reason belongs on stderr"
)]
fn every_dense_affine_entry_point_resolves_as_a_pipeline() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the pipeline check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    for entry in affine::ENTRIES {
        backend
            .prepare(KernelRef::new("dense", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn qmv_4_matches_the_host_dequant_matmul() {
    let Some(gpu) = open() else { return };
    affine::qmv_4_matches_the_host_dequant_matmul(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn qmv_8_matches_the_host_dequant_matmul() {
    let Some(gpu) = open() else { return };
    affine::qmv_8_matches_the_host_dequant_matmul(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn qmm_4_matches_at_n_tok_64() {
    let Some(gpu) = open() else { return };
    affine::qmm_4_matches_at_n_tok_64(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn qmm_8_matches_at_n_tok_64() {
    let Some(gpu) = open() else { return };
    affine::qmm_8_matches_at_n_tok_64(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn a_row_count_that_is_not_a_tile_multiple_is_handled() {
    let Some(gpu) = open() else { return };
    affine::a_row_count_that_is_not_a_tile_multiple_is_handled(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn qmm_4_matches_at_a_chunk_that_is_not_a_whole_band() {
    let Some(gpu) = open() else { return };
    affine::qmm_4_matches_at_a_chunk_that_is_not_a_whole_band(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn qmm_8_matches_at_a_chunk_that_is_not_a_whole_band() {
    let Some(gpu) = open() else { return };
    affine::qmm_8_matches_at_a_chunk_that_is_not_a_whole_band(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn qmm_4_matches_at_a_chunk_shorter_than_one_band() {
    let Some(gpu) = open() else { return };
    affine::qmm_4_matches_at_a_chunk_shorter_than_one_band(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_group_bias_is_applied() {
    let Some(gpu) = open() else { return };
    affine::the_group_bias_is_applied(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_affine_gather_dequantizes_the_rows_the_ids_name() {
    let Some(gpu) = open() else { return };
    affine::the_affine_gather_dequantizes_the_rows_the_ids_name(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_affine_gather_reads_one_row_per_token_and_repeats_a_repeated_id() {
    let Some(gpu) = open() else { return };
    affine::the_affine_gather_reads_one_row_per_token_and_repeats_a_repeated_id(&gpu);
}
