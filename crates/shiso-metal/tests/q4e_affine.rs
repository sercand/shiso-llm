//! Metal's run of the shared `qwen4exp` affine cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --test kernels q4e_affine:: -- --ignored`.
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

use shiso_kerneltest::qwen4exp::affine as cases;

#[test]
#[ignore = "requires a Metal device"]
fn the_one_row_projection_matches_the_host() {
    let Some(gpu) = open() else { return };
    cases::the_one_row_projection_matches_the_host(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_multi_row_projection_matches_the_host() {
    let Some(gpu) = open() else { return };
    cases::the_multi_row_projection_matches_the_host(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_prompt_projection_matches_the_host_at_every_width() {
    let Some(gpu) = open() else { return };
    cases::the_prompt_projection_matches_the_host_at_every_width(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_prompt_expert_stages_match_the_host_at_every_width() {
    let Some(gpu) = open() else { return };
    cases::the_prompt_expert_stages_match_the_host_at_every_width(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_prompt_brain_float_projection_matches_the_host_at_every_width() {
    let Some(gpu) = open() else { return };
    cases::the_prompt_brain_float_projection_matches_the_host_at_every_width(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_affine_gather_matches_the_host() {
    let Some(gpu) = open() else { return };
    cases::the_affine_gather_matches_the_host(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_expert_stages_follow_the_slot_map() {
    let Some(gpu) = open() else { return };
    cases::the_expert_stages_follow_the_slot_map(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_ngram_rows_decode_from_the_affine_row() {
    let Some(gpu) = open() else { return };
    cases::the_ngram_rows_decode_from_the_affine_row(&gpu);
}
