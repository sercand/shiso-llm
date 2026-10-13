//! Metal's run of the shared `qwen4exp` ple cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --test kernels q4e_ple:: -- --ignored`.
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

use shiso_kerneltest::qwen4exp::ple;

#[test]
#[ignore = "requires a Metal device"]
fn the_gated_value_takes_a_signed_square_root_before_the_sigmoid() {
    let Some(gpu) = open() else { return };
    ple::the_gated_value_takes_a_signed_square_root_before_the_sigmoid(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_convolution_runs_in_place_across_its_carried_history() {
    let Some(gpu) = open() else { return };
    ple::the_convolution_runs_in_place_across_its_carried_history(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_convolution_records_a_window_per_token_up_to_its_capacity() {
    let Some(gpu) = open() else { return };
    ple::the_convolution_records_a_window_per_token_up_to_its_capacity(&gpu);
}
