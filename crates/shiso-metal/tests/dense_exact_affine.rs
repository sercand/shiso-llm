//! Metal's run of the shared row-exact affine cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run. The pipeline check is not ignore-gated, so a missing
//! entry point fails the default gate. The rest run with
//! `cargo test -p shiso-metal --features metal --test dense_exact_affine -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::exact_affine;
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
fn every_exact_affine_entry_point_resolves_as_a_pipeline() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the pipeline check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    for entry in exact_affine::ENTRIES {
        backend
            .prepare(KernelRef::new("dense", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn values_match_the_host_4bit() {
    let Some(gpu) = open() else { return };
    exact_affine::values_match_the_host_4bit(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn values_match_the_host_8bit() {
    let Some(gpu) = open() else { return };
    exact_affine::values_match_the_host_8bit(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn rows_are_exact_at_small_shapes() {
    let Some(gpu) = open() else { return };
    exact_affine::rows_are_exact_at_small_shapes(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn rows_are_exact_at_model_shapes() {
    let Some(gpu) = open() else { return };
    exact_affine::rows_are_exact_at_model_shapes(&gpu);
}
