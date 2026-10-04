//! CUDA's run of the shared `ternary` cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run. GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p
//! shiso-cuda --test ternary -- --ignored`
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::ternary::{matmul, rotate};

/// Opens device 0, or says on stderr why the cases did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<Gpu> {
    match CudaBackend::new(0, CudaOptions::default()) {
        Ok(cuda) => {
            let device: Arc<dyn Backend> = cuda.clone();
            Some(Gpu::new(device, Box::new(cuda)))
        }
        Err(err) => {
            eprintln!("note: no CUDA device on this machine, skipping the GPU cases: {err}");
            None
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn rotate_forward() {
    let Some(gpu) = open() else { return };
    rotate::forward_matches_the_dense_transform(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn rotate_inverse() {
    let Some(gpu) = open() else { return };
    rotate::inverse_matches_the_dense_transform(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn rotate_permuted_heads() {
    let Some(gpu) = open() else { return };
    rotate::permuted_heads_read_tiled_as_grouped(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn rotate_round_trip() {
    let Some(gpu) = open() else { return };
    rotate::forward_then_inverse_is_the_identity(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn mv_all_formats() {
    let Some(gpu) = open() else { return };
    matmul::mv_matches_the_host_matmul(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn mv_pq2_equals_mlx2() {
    let Some(gpu) = open() else { return };
    matmul::pq2_and_mlx2_agree_bitwise(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn get_rows_all_formats() {
    let Some(gpu) = open() else { return };
    matmul::get_rows_matches_the_host_decode(&gpu);
}
