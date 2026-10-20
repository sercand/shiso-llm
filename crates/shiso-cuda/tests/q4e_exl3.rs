//! CUDA's run of the shared `qwen4exp` trellis cases: the decode-shaped dense
//! projection at every built width, graded against the expansion.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q4e_exl3
//! -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::exl3;

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
fn four_bits() {
    let Some(gpu) = open() else { return };
    exl3::the_decode_projection_is_the_expanded_weight(&gpu, 4);
}

#[test]
#[ignore = "requires a CUDA device"]
fn five_bits() {
    let Some(gpu) = open() else { return };
    exl3::the_decode_projection_is_the_expanded_weight(&gpu, 5);
}

#[test]
#[ignore = "requires a CUDA device"]
fn six_bits() {
    let Some(gpu) = open() else { return };
    exl3::the_decode_projection_is_the_expanded_weight(&gpu, 6);
}

#[test]
#[ignore = "requires a CUDA device"]
fn eight_bits() {
    let Some(gpu) = open() else { return };
    exl3::the_decode_projection_is_the_expanded_weight(&gpu, 8);
}
