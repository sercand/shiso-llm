//! CUDA's run of the shared `q35_gemv` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q35_gemv --
//! --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen35::gemv as cases;

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
fn q5_k_matvec_matches_the_oracle_dot() {
    let Some(gpu) = open() else { return };
    cases::q5_k_matvec_matches_the_oracle_dot(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn q6_k_matvec_matches_the_oracle_dot() {
    let Some(gpu) = open() else { return };
    cases::q6_k_matvec_matches_the_oracle_dot(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn q6_k_gather_matches_the_dense_unpack() {
    let Some(gpu) = open() else { return };
    cases::q6_k_gather_matches_the_dense_unpack(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn q4_k_prefill_gemm_matches_a_triple_loop() {
    let Some(gpu) = open() else { return };
    cases::q4_k_prefill_gemm_matches_a_triple_loop(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn q4_k_matvec_matches_the_oracle_dot() {
    let Some(gpu) = open() else { return };
    cases::q4_k_matvec_matches_the_oracle_dot(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_dequantizers_produce_the_same_bits_they_always_have() {
    let Some(gpu) = open() else { return };
    cases::the_dequantizers_produce_the_same_bits_they_always_have(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn k_quant_matvec_batches_match_the_oracle_dot() {
    let Some(gpu) = open() else { return };
    cases::k_quant_matvec_batches_match_the_oracle_dot(&gpu);
}
