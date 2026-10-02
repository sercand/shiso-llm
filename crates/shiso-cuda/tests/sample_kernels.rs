//! CUDA's run of the shared logits-stage cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! sample_kernels -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::sample;

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
fn the_suppress_stage_masks_every_listed_id_in_every_row() {
    let Some(gpu) = open() else { return };
    sample::the_suppress_stage_masks_every_listed_id_in_every_row(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn an_empty_list_leaves_the_rows_alone() {
    let Some(gpu) = open() else { return };
    sample::an_empty_list_leaves_the_rows_alone(&gpu);
}
