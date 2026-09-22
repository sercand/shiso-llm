//! Metal's run of the shared `dense_vision` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! One case is not ignore-gated: the pipelines have to resolve, and a missing or
//! misnamed entry point should fail the default gate rather than hide behind the
//! GPU gate. The rest are `#[ignore]`d: `cargo test -p shiso-metal --features
//! metal --test dense_vision -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::vision;
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
fn every_vision_dense_entry_point_resolves_as_a_pipeline() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the pipeline check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    for entry in vision::ENTRIES {
        backend
            .prepare(KernelRef::new("dense", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
    backend
        .prepare(KernelRef::new("vision", "kernel_vit_attention"))
        .expect("the attention entry resolves");
}

#[test]
#[ignore = "requires a Metal device"]
fn rope2d_rotates_each_part_by_its_own_axis() {
    let Some(gpu) = open() else { return };
    vision::rope2d_rotates_each_part_by_its_own_axis(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn avgpool3_averages_nine_patches_and_scales() {
    let Some(gpu) = open() else { return };
    vision::avgpool3_averages_nine_patches_and_scales(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn clamp_is_elementwise() {
    let Some(gpu) = open() else { return };
    vision::clamp_is_elementwise(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn vit_attention_takes_its_scale() {
    let Some(gpu) = open() else { return };
    vision::vit_attention_takes_its_scale(&gpu);
}
