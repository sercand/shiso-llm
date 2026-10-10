//! CUDA's run of the shared `qwen4exp` cases for the `q4e_exact` module.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run. The tensor-core attention cases are CUDA's alone:
//! no other backend ports that kernel.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q4e_exact
//! -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::{gdn, qsa};

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
fn the_tensor_core_attention_matches_fp32_weights() {
    let Some(gpu) = open() else { return };
    qsa::the_tensor_core_attention_matches_fp32_weights(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_dense_tensor_core_attention_matches_fp32_weights() {
    let Some(gpu) = open() else { return };
    qsa::the_dense_tensor_core_attention_matches_fp32_weights(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_ported_tensor_core_attention_misses_that_budget() {
    let Some(gpu) = open() else { return };
    qsa::the_ported_tensor_core_attention_misses_that_budget(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn step_decode_matches_the_chain() {
    let Some(gpu) = open() else { return };
    gdn::step_decode_matches_the_chain(&gpu);
}
