//! Metal's run of the shared `dense_ple` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! One case is not ignore-gated: the pipelines have to resolve, and a missing or
//! misnamed entry point should fail the default gate rather than hide behind the
//! GPU gate. The rest are `#[ignore]`d: `cargo test -p shiso-metal --features
//! metal --test dense_ple -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::ple;
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
fn every_per_layer_entry_point_resolves_as_a_pipeline() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the pipeline check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    for entry in ple::ENTRIES {
        backend
            .prepare(KernelRef::new("dense", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn a_resident_table_mixes_the_normalized_projection_with_the_row_its_id_names() {
    let Some(gpu) = open() else { return };
    ple::a_resident_table_mixes_the_normalized_projection_with_the_row_its_id_names(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn a_streamed_table_reads_the_row_that_was_staged_for_each_token() {
    let Some(gpu) = open() else { return };
    ple::a_streamed_table_reads_the_row_that_was_staged_for_each_token(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_two_slots_of_one_token_are_two_different_rows_of_the_table() {
    let Some(gpu) = open() else { return };
    ple::the_two_slots_of_one_token_are_two_different_rows_of_the_table(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_affine_prepare_mixes_the_projection_with_the_row_its_id_names() {
    let Some(gpu) = open() else { return };
    ple::the_affine_prepare_mixes_the_projection_with_the_row_its_id_names(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_affine_prepare_reads_the_row_that_was_staged_for_each_token() {
    let Some(gpu) = open() else { return };
    ple::the_affine_prepare_reads_the_row_that_was_staged_for_each_token(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_two_slots_of_one_affine_row_are_two_different_spans() {
    let Some(gpu) = open() else { return };
    ple::the_two_slots_of_one_affine_row_are_two_different_spans(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_gate_is_the_activation_times_the_named_slots_row() {
    let Some(gpu) = open() else { return };
    ple::the_gate_is_the_activation_times_the_named_slots_row(&gpu);
}
