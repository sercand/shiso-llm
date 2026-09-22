//! CUDA's run of the shared `q35_norm_rope` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q35_norm_rope --
//! --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen35::norm_rope as cases;

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
fn f16_decoder_matches_known_patterns() {
    cases::f16_decoder_matches_known_patterns();
}

#[test]
#[ignore = "requires a CUDA device"]
fn q_norm_rope_matches_the_unfused_oracle() {
    let Some(gpu) = open() else { return };
    cases::q_norm_rope_matches_the_unfused_oracle(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn k_norm_rope_writes_the_f16_cache_at_its_position_slot() {
    let Some(gpu) = open() else { return };
    cases::k_norm_rope_writes_the_f16_cache_at_its_position_slot(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_multimodal_query_rotation_deals_its_pairs_among_the_three_streams() {
    let Some(gpu) = open() else { return };
    cases::the_multimodal_query_rotation_deals_its_pairs_among_the_three_streams(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn three_agreeing_streams_reproduce_the_one_dimensional_rotation_exactly() {
    let Some(gpu) = open() else { return };
    cases::three_agreeing_streams_reproduce_the_one_dimensional_rotation_exactly(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_multimodal_key_path_separates_the_cache_row_from_the_angle() {
    let Some(gpu) = open() else { return };
    cases::the_multimodal_key_path_separates_the_cache_row_from_the_angle(&gpu);
}
