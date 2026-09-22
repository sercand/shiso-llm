//! Metal's run of the shared `q35_gdn` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --features metal --test q35_gdn --
//! --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen35::gdn as cases;
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
#[ignore = "requires a Metal device"]
fn conv_resumes_from_the_window_a_previous_call_left() {
    let Some(gpu) = open() else { return };
    cases::conv_resumes_from_the_window_a_previous_call_left(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn l2norm_normalizes_each_head_row_in_place() {
    let Some(gpu) = open() else { return };
    cases::l2norm_normalizes_each_head_row_in_place(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn gates_apply_the_softplus_guard_and_the_sigmoid() {
    let Some(gpu) = open() else { return };
    cases::gates_apply_the_softplus_guard_and_the_sigmoid(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn out_gate_normalizes_one_head_and_scales_it_by_the_gate() {
    let Some(gpu) = open() else { return };
    cases::out_gate_normalizes_one_head_and_scales_it_by_the_gate(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn recurrent_matches_a_naive_delta_rule() {
    let Some(gpu) = open() else { return };
    cases::recurrent_matches_a_naive_delta_rule(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn every_viable_recurrent_variant_matches_the_oracle() {
    let Some(gpu) = open() else { return };
    cases::every_viable_recurrent_variant_matches_the_oracle(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn recurrent_variants_are_bit_identical() {
    let Some(gpu) = open() else { return };
    cases::recurrent_variants_are_bit_identical(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn chunk_variants_are_bit_identical() {
    let Some(gpu) = open() else { return };
    cases::chunk_variants_are_bit_identical(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn every_viable_chunk_variant_matches_the_recurrence() {
    let Some(gpu) = open() else { return };
    cases::every_viable_chunk_variant_matches_the_recurrence(&gpu);
}
