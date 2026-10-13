//! Metal's run of the shared `dense_misc` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! One case is not ignore-gated: the pipelines have to resolve, and a missing or
//! misnamed entry point should fail the default gate rather than hide behind the
//! GPU gate. The rest are `#[ignore]`d: `cargo test -p shiso-metal --features
//! metal --test kernels dense_misc:: -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::misc;
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
fn every_dense_entry_point_resolves_as_a_pipeline() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the pipeline check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    for entry in misc::ENTRIES {
        backend
            .prepare(KernelRef::new("dense", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn the_gated_gelu_is_the_tanh_approximation_times_the_up_row() {
    let Some(gpu) = open() else { return };
    misc::the_gated_gelu_is_the_tanh_approximation_times_the_up_row(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_gated_gelu_saturates_on_a_large_gate_instead_of_overflowing_to_nan() {
    let Some(gpu) = open() else { return };
    misc::the_gated_gelu_saturates_on_a_large_gate_instead_of_overflowing_to_nan(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_scaled_close_scales_the_sum_before_it_stores_and_before_it_reduces() {
    let Some(gpu) = open() else { return };
    misc::the_scaled_close_scales_the_sum_before_it_stores_and_before_it_reduces(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_softcap_squashes_a_row_into_the_cap() {
    let Some(gpu) = open() else { return };
    misc::the_softcap_squashes_a_row_into_the_cap(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_row_scale_multiplies_in_place_and_stops_at_its_bound() {
    let Some(gpu) = open() else { return };
    misc::the_row_scale_multiplies_in_place_and_stops_at_its_bound(&gpu);
}
