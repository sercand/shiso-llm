//! CUDA's run of the shared `qwen4exp` indexer cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run. They grade both block scorers over a page table whose
//! pages are out of order, which is the table a long-running process hands a
//! session and a fresh one never does.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q4e_idx --
//! --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::indexer;

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
fn the_scalar_scorer_follows_a_scattered_page_table() {
    let Some(gpu) = open() else { return };
    indexer::the_scalar_scorer_follows_a_scattered_page_table(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_tensor_core_scorer_follows_a_scattered_page_table() {
    let Some(gpu) = open() else { return };
    indexer::the_tensor_core_scorer_follows_a_scattered_page_table(&gpu);
}
