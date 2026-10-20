//! CUDA's run of the shared `qwen4exp` fused route case: the router, the
//! routed selection and the shared expert as one launch, against the launches
//! it stands in for.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q4e_ffn
//! -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::ffn;

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
fn the_fused_route_is_the_launches_it_replaces_bit_for_bit() {
    let Some(gpu) = open() else { return };
    ffn::the_fused_route_is_the_launches_it_replaces_bit_for_bit(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_fused_close_is_the_combine_and_the_shared_add() {
    let Some(gpu) = open() else { return };
    ffn::the_fused_close_is_the_combine_and_the_shared_add(&gpu);
}
