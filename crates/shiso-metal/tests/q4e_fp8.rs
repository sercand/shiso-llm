//! Metal's run of the shared `qwen4exp` FP8 cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --test kernels q4e_fp8:: -- --ignored`.
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

use shiso_kerneltest::qwen4exp::fp8 as cases;

#[test]
#[ignore = "requires a Metal device"]
fn the_fp8_expert_stages_follow_the_slot_map() {
    let Some(gpu) = open() else { return };
    cases::the_fp8_expert_stages_follow_the_slot_map(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_fp8_ngram_rows_decode_under_the_table_scale() {
    let Some(gpu) = open() else { return };
    cases::the_fp8_ngram_rows_decode_under_the_table_scale(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_brain_float_gather_widens_exactly() {
    let Some(gpu) = open() else { return };
    cases::the_brain_float_gather_widens_exactly(&gpu);
}
