//! CUDA's run of the shared `qwen35moe` routed-block cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! q35moe_experts -- --ignored --nocapture`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen35moe::experts as cases;

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
fn the_routed_block_matches_the_host_reference() {
    let Some(gpu) = open() else { return };
    cases::the_routed_block_matches_the_host_reference(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_routed_block_at_one_token_matches_the_same_reference() {
    let Some(gpu) = open() else { return };
    cases::the_routed_block_at_one_token_matches_the_same_reference(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_first_row_of_a_chunk_is_what_that_token_answers_alone() {
    let Some(gpu) = open() else { return };
    cases::the_first_row_of_a_chunk_is_what_that_token_answers_alone(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_shared_expert_s_gate_is_one_scalar_per_token() {
    let Some(gpu) = open() else { return };
    cases::the_shared_expert_s_gate_is_one_scalar_per_token(&gpu);
}
