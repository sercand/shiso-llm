//! CUDA's run of the shared `dense_vision` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! dense_vision -- --ignored`. Without `SHISO_CUDA_COMPILE=1` the kernel
//! crate embeds an empty registry and the refusal says so, which is a different
//! message from the one a real build gives.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::vision;

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
fn rope2d_rotates_each_part_by_its_own_axis() {
    let Some(gpu) = open() else { return };
    vision::rope2d_rotates_each_part_by_its_own_axis(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn avgpool3_averages_nine_patches_and_scales() {
    let Some(gpu) = open() else { return };
    vision::avgpool3_averages_nine_patches_and_scales(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn clamp_is_elementwise() {
    let Some(gpu) = open() else { return };
    vision::clamp_is_elementwise(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn vit_attention_takes_its_scale() {
    let Some(gpu) = open() else { return };
    vision::vit_attention_takes_its_scale(&gpu);
}
