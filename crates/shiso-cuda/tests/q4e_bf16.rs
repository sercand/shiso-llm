//! CUDA's run of the shared `qwen4exp` brain-float cases: the dense
//! projections of the `q4e_bf16` module, graded against the host product and
//! against themselves across pass widths.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! q4e_bf16 -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::bf16;

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
#[expect(
    clippy::print_stderr,
    reason = "the measured error is the number the case's tolerance is justified by"
)]
fn a_verify_row_reproduces_the_single_row_step() {
    let Some(gpu) = open() else { return };
    let worst = bf16::a_verify_row_reproduces_the_single_row_step(&gpu);
    eprintln!("bf16 step kernels: worst error {worst:e} of the absolute sum");
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_step_row_count_matches() {
    let Some(gpu) = open() else { return };
    bf16::every_step_row_count_matches(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
#[expect(
    clippy::print_stderr,
    reason = "the measured error is the number the case's tolerance is justified by"
)]
fn the_prompt_gemms_hold_their_rows_to_every_width() {
    let Some(gpu) = open() else { return };
    let (plain, split) = bf16::the_prompt_gemms_hold_their_rows_to_every_width(&gpu);
    eprintln!(
        "bf16 prompt GEMMs: worst error {plain:e} (rounded activations), {split:e} (split \
         activations) of the absolute sum"
    );
}
