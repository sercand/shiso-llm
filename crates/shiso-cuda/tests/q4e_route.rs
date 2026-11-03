//! CUDA's run of the shared `qwen4exp` routing cases: the one-launch routed
//! selection and sort against the four launches it stands in for.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q4e_route
//! -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::route;

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
fn the_fused_route_sort_is_the_four_launch_route_and_sort() {
    let Some(gpu) = open() else { return };
    route::the_fused_route_sort_is_the_four_launch_route_and_sort(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_tiny_route_sort_is_the_four_launch_route_and_sort() {
    let Some(gpu) = open() else { return };
    route::the_tiny_route_sort_is_the_four_launch_route_and_sort(&gpu);
}
