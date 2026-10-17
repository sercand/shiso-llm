//! CUDA's run of the shared `qwen4exp` MXFP8 cases: the dense projections of
//! the `q4e_mxfp8` module, graded against the brain-float tier and the host
//! decode.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! q4e_mxfp8 -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::mxfp8;

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
fn every_code_decodes_exactly_at_every_scale() {
    let Some(gpu) = open() else { return };
    mxfp8::every_code_decodes_exactly_at_every_scale(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_step_kernels_are_the_brain_float_tier_bit_for_bit() {
    let Some(gpu) = open() else { return };
    mxfp8::the_step_kernels_are_the_brain_float_tier_bit_for_bit(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_verify_row_reproduces_the_single_row_step() {
    let Some(gpu) = open() else { return };
    mxfp8::a_verify_row_reproduces_the_single_row_step(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
#[expect(
    clippy::print_stderr,
    reason = "the measured error is the number the case's tolerance is justified by"
)]
fn the_prompt_kernel_holds_its_rows_to_every_width() {
    let Some(gpu) = open() else { return };
    let worst = mxfp8::the_prompt_kernel_holds_its_rows_to_every_width(&gpu);
    eprintln!("mxfp8 prompt kernel: worst error {worst:e} of the absolute sum");
}
