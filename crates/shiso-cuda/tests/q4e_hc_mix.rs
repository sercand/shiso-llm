//! CUDA's run of the shared `qwen4exp` fused stream-mix cases: the pair
//! launch and the fused expansion and collapse, against the launches they
//! replace.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! q4e_hc_mix -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::hc_mix;

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
fn the_pair_gemm_is_its_two_gemms_bit_for_bit() {
    let Some(gpu) = open() else { return };
    hc_mix::the_pair_gemm_is_its_two_gemms_bit_for_bit(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_fused_collapse_is_the_expansion_and_the_collapse_bit_for_bit() {
    let Some(gpu) = open() else { return };
    hc_mix::the_fused_collapse_is_the_expansion_and_the_collapse_bit_for_bit(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_scaled_combine_and_norming_reads_are_the_stored_normed_streams() {
    let Some(gpu) = open() else { return };
    hc_mix::the_scaled_combine_and_norming_reads_are_the_stored_normed_streams(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_step_pair_is_its_two_steps_bit_for_bit() {
    let Some(gpu) = open() else { return };
    hc_mix::the_step_pair_is_its_two_steps_bit_for_bit(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_fused_mix_step_is_the_mix_and_a_verify_row_is_the_step() {
    let Some(gpu) = open() else { return };
    hc_mix::the_fused_mix_step_is_the_mix_and_a_verify_row_is_the_step(&gpu);
}
