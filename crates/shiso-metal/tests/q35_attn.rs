//! Metal's run of the shared `q35_attn` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --features metal --test kernels q35_attn:: --
//! --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen35::attn as cases;
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
fn fixture_values_survive_the_f16_cache() {
    cases::fixture_values_survive_the_f16_cache();
}

#[test]
#[ignore = "requires a Metal device"]
fn decode_matches_a_naive_masked_softmax() {
    let Some(gpu) = open() else { return };
    cases::decode_matches_a_naive_masked_softmax(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn prefill_matches_token_by_token_decode() {
    let Some(gpu) = open() else { return };
    cases::prefill_matches_token_by_token_decode(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn token_zero_returns_only_its_own_value_row() {
    let Some(gpu) = open() else { return };
    cases::token_zero_returns_only_its_own_value_row(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn gate_mul_applies_the_sigmoid_gate() {
    let Some(gpu) = open() else { return };
    cases::gate_mul_applies_the_sigmoid_gate(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn two_rows_in_two_pages_match_the_rows_run_alone() {
    let Some(gpu) = open() else { return };
    cases::two_rows_in_two_pages_match_the_rows_run_alone(&gpu);
}
