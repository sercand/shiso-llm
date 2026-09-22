//! CUDA's run of the shared `qwen35moe` FP8 cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q35moe_fp8
//! -- --ignored --nocapture`. `--nocapture` is worth typing: every case here
//! prints the worst divergence it measured beside the budget it was held to,
//! and that number is the point of the case.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen35moe::fp8 as cases;

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
fn every_e4m3_code_decodes_the_way_the_host_decodes_it() {
    let Some(gpu) = open() else { return };
    cases::every_e4m3_code_decodes_the_way_the_host_decodes_it(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn fp8_matvec_matches_the_host_codec() {
    let Some(gpu) = open() else { return };
    cases::fp8_matvec_matches_the_host_codec(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn fp8_gemm_matches_the_host_codec() {
    let Some(gpu) = open() else { return };
    cases::fp8_gemm_matches_the_host_codec(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_matvec_and_the_gemm_agree_on_the_same_weights() {
    let Some(gpu) = open() else { return };
    cases::the_matvec_and_the_gemm_agree_on_the_same_weights(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn fp8_matvec_matches_the_host_codec_at_an_unaligned_width() {
    let Some(gpu) = open() else { return };
    cases::fp8_matvec_matches_the_host_codec_at_an_unaligned_width(&gpu);
}
