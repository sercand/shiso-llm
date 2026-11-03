//! CUDA's run of the shared `qwen4exp` cases for the `q4e_gdn_fused` module.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run. The fused prompt pair is CUDA's alone: no other
//! backend carries the tensor-core chain it reproduces.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --release --test
//! q4e_gdn_fused -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::gdn_fused;

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
fn the_fused_prompt_pair_gives_the_chain_s_bits() {
    let Some(gpu) = open() else { return };
    gdn_fused::fused_prompt_matches_the_chain(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_fused_prompt_pair_is_chunk_width_invariant() {
    let Some(gpu) = open() else { return };
    gdn_fused::fused_prompt_is_chunk_width_invariant(&gpu);
}
