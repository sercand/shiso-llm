//! CUDA's run of the shared `qwen4exp` `modelopt` cases: the NVFP4 routed
//! expert stages and the FP8 n-gram row decode, both in the `q4e_modelopt`
//! module.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! q4e_modelopt -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::{fp8, nvfp4};

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
fn the_nvfp4_expert_stages_follow_the_slot_map() {
    let Some(gpu) = open() else { return };
    nvfp4::the_nvfp4_expert_stages_follow_the_slot_map(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_fp8_ngram_rows_decode_under_the_table_scale() {
    let Some(gpu) = open() else { return };
    fp8::the_fp8_ngram_rows_decode_under_the_table_scale(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_fp8_expert_stages_follow_the_slot_map() {
    let Some(gpu) = open() else { return };
    fp8::the_fp8_expert_stages_follow_the_slot_map(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_nvfp4_prompt_stages_follow_the_slot_map() {
    let Some(gpu) = open() else { return };
    nvfp4::the_nvfp4_prompt_stages_follow_the_slot_map(&gpu);
}
