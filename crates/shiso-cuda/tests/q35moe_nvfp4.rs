//! CUDA's run of the shared `qwen35moe` NVFP4 cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! q35moe_nvfp4 -- --ignored --nocapture`. `--nocapture` is worth typing:
//! every case prints the worst divergence it measured beside the budget it was
//! held to, and that number is the point of the case.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen35moe::nvfp4 as cases;

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
fn nvfp4_matvec_matches_the_host_unpacker() {
    let Some(gpu) = open() else { return };
    cases::nvfp4_matvec_matches_the_host_unpacker(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn nvfp4_gemm_matches_the_host_unpacker() {
    let Some(gpu) = open() else { return };
    cases::nvfp4_gemm_matches_the_host_unpacker(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_per_tensor_scale_is_visible_in_these_fixtures() {
    let Some(gpu) = open() else { return };
    cases::the_per_tensor_scale_is_visible_in_these_fixtures(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_matvec_and_the_gemm_agree_on_the_same_weights() {
    let Some(gpu) = open() else { return };
    cases::the_matvec_and_the_gemm_agree_on_the_same_weights(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_e2m1_code_decodes_the_way_the_host_table_does() {
    let Some(gpu) = open() else { return };
    cases::every_e2m1_code_decodes_the_way_the_host_table_does(&gpu);
}
