//! Metal's run of the shared `qwen4exp` GGML block cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --features metal --test kernels q4e_ggml:: -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_kerneltest::Gpu;
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

use shiso_kerneltest::qwen4exp::ggml as cases;

#[test]
#[ignore = "requires a Metal device"]
fn the_dense_projection_matches_the_host_in_every_format() {
    let Some(gpu) = open() else { return };
    cases::the_dense_projection_matches_the_host_in_every_format(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_routed_stages_match_the_host_in_every_format() {
    let Some(gpu) = open() else { return };
    cases::the_routed_stages_match_the_host_in_every_format(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_vocabulary_gather_decodes_rows_exactly() {
    let Some(gpu) = open() else { return };
    cases::the_vocabulary_gather_decodes_rows_exactly(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_ngram_row_decode_places_iq4_nl_rows_exactly() {
    let Some(gpu) = open() else { return };
    cases::the_ngram_row_decode_places_iq4_nl_rows_exactly(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_lane_step_entries_match_the_host_in_every_block_format() {
    let Some(gpu) = open() else { return };
    cases::the_lane_step_entries_match_the_host_in_every_block_format(&gpu);
}
