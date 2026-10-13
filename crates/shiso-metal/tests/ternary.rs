//! Metal's run of the shared `ternary` cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run. The pipeline check is not ignore-gated, so a missing
//! entry fails the default gate. The rest: `cargo test -p shiso-metal
//! --features metal --test kernels ternary:: -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::ternary::{self, matmul, rotate};
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
fn every_ternary_entry_point_resolves_as_a_pipeline() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the pipeline check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    for entry in ternary::ENTRIES {
        backend
            .prepare(KernelRef::new("ternary", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
    for entry in matmul::entries() {
        let entry: &'static str = Box::leak(entry.into_boxed_str());
        backend
            .prepare(KernelRef::new("ternary", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn rotate_forward() {
    let Some(gpu) = open() else { return };
    rotate::forward_matches_the_dense_transform(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn rotate_inverse() {
    let Some(gpu) = open() else { return };
    rotate::inverse_matches_the_dense_transform(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn rotate_permuted_heads() {
    let Some(gpu) = open() else { return };
    rotate::permuted_heads_read_tiled_as_grouped(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn rotate_round_trip() {
    let Some(gpu) = open() else { return };
    rotate::forward_then_inverse_is_the_identity(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn mv_all_formats() {
    let Some(gpu) = open() else { return };
    matmul::mv_matches_the_host_matmul(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn mv_pq2_equals_mlx2() {
    let Some(gpu) = open() else { return };
    matmul::pq2_and_mlx2_agree_bitwise(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn get_rows_all_formats() {
    let Some(gpu) = open() else { return };
    matmul::get_rows_matches_the_host_decode(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn mm_all_formats() {
    let Some(gpu) = open() else { return };
    matmul::mm_matches_the_host_matmul(&gpu);
}
