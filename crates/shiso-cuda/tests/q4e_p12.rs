//! CUDA's run of the twelve-bit expansion case.
//!
//! The case lives in `shiso-kerneltest`; this file opens a device or says why
//! the case did not run.
//!
//! GPU case: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q4e_p12 --
//! --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::p12;

/// Opens device 0, or says on stderr why the case did not run.
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
fn the_expansion_restores_the_packed_halves() {
    let Some(gpu) = open() else { return };
    p12::the_expansion_restores_the_packed_halves(&gpu);
}
