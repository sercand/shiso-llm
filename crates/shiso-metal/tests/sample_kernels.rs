//! Metal's run of the shared logits-stage cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --test kernels sample_kernels:: -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_kerneltest::Gpu;
use shiso_kerneltest::sample;
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
fn the_suppress_stage_masks_every_listed_id_in_every_row() {
    let Some(gpu) = open() else { return };
    sample::the_suppress_stage_masks_every_listed_id_in_every_row(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn an_empty_list_leaves_the_rows_alone() {
    let Some(gpu) = open() else { return };
    sample::an_empty_list_leaves_the_rows_alone(&gpu);
}
