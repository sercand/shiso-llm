//! CUDA's run of the shared `q35_attn` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q35_attn --
//! --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen35::attn as cases;

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
fn fixture_values_survive_the_f16_cache() {
    cases::fixture_values_survive_the_f16_cache();
}

#[test]
#[ignore = "requires a CUDA device"]
fn decode_matches_a_naive_masked_softmax() {
    let Some(gpu) = open() else { return };
    cases::decode_matches_a_naive_masked_softmax(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn prefill_matches_token_by_token_decode() {
    let Some(gpu) = open() else { return };
    cases::prefill_matches_token_by_token_decode(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn token_zero_returns_only_its_own_value_row() {
    let Some(gpu) = open() else { return };
    cases::token_zero_returns_only_its_own_value_row(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn gate_mul_applies_the_sigmoid_gate() {
    let Some(gpu) = open() else { return };
    cases::gate_mul_applies_the_sigmoid_gate(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn two_rows_in_two_pages_match_the_rows_run_alone() {
    let Some(gpu) = open() else { return };
    cases::two_rows_in_two_pages_match_the_rows_run_alone(&gpu);
}
