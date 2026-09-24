//! Metal's run of the shared `qwen4exp` matvec cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --test q4e_matvec -- --ignored`.
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

use shiso_kerneltest::qwen4exp::matvec;

#[test]
#[ignore = "requires a Metal device"]
fn the_bf16_rows_projection_matches_at_the_real_shapes() {
    let Some(gpu) = open() else { return };
    matvec::the_bf16_rows_projection_matches_at_the_real_shapes(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn every_bf16_rows_instantiation_matches() {
    let Some(gpu) = open() else { return };
    matvec::every_bf16_rows_instantiation_matches(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_f16_rows_projection_splits_and_folds_at_the_real_shapes() {
    let Some(gpu) = open() else { return };
    matvec::the_f16_rows_projection_splits_and_folds_at_the_real_shapes(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn every_f32_io_f16_rows_instantiation_matches() {
    let Some(gpu) = open() else { return };
    matvec::every_f32_io_f16_rows_instantiation_matches(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn every_half_input_f16_rows_instantiation_matches() {
    let Some(gpu) = open() else { return };
    matvec::every_half_input_f16_rows_instantiation_matches(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn every_half_output_f16_rows_instantiation_matches() {
    let Some(gpu) = open() else { return };
    matvec::every_half_output_f16_rows_instantiation_matches(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_half_combine_rounds_the_folded_sum() {
    let Some(gpu) = open() else { return };
    matvec::the_half_combine_rounds_the_folded_sum(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_f32_matvec_splits_and_folds() {
    let Some(gpu) = open() else { return };
    matvec::the_f32_matvec_splits_and_folds(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_width_conversions_round_and_widen_exactly() {
    let Some(gpu) = open() else { return };
    matvec::the_width_conversions_round_and_widen_exactly(&gpu);
}
