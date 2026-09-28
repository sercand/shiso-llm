//! CUDA's run of the shared row-exact affine cases.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! dense_exact_affine -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::exact_affine;

/// Opens device 0, or says on stderr why the cases did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<(Gpu, Arc<CudaBackend>)> {
    match CudaBackend::new(0, CudaOptions::default()) {
        Ok(cuda) => {
            let device: Arc<dyn Backend> = cuda.clone();
            Some((Gpu::new(device, Box::new(cuda.clone())), cuda))
        }
        Err(err) => {
            eprintln!("note: no CUDA device on this machine, skipping the GPU cases: {err}");
            None
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_exact_affine_entry_point_resolves() {
    let Some((_, cuda)) = open() else { return };
    for entry in exact_affine::ENTRIES {
        cuda.prepare(KernelRef::new("dense", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn values_match_the_host_4bit() {
    let Some((gpu, _)) = open() else { return };
    exact_affine::values_match_the_host_4bit(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn values_match_the_host_8bit() {
    let Some((gpu, _)) = open() else { return };
    exact_affine::values_match_the_host_8bit(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn rows_are_exact_at_small_shapes() {
    let Some((gpu, _)) = open() else { return };
    exact_affine::rows_are_exact_at_small_shapes(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn rows_are_exact_at_model_shapes() {
    let Some((gpu, _)) = open() else { return };
    exact_affine::rows_are_exact_at_model_shapes(&gpu);
}
