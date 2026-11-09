//! CUDA's run of the shared `qwen4exp` GGML block cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run. They grade the dense projection, the routed stages
//! through a slot map, the vocabulary gather and the `IQ4_NL` n-gram row decode
//! in every block format against a host decode of the same bytes.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q4e_ggml --
//! --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::ggml as cases;

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
fn the_dense_projection_matches_the_host_in_every_format() {
    let Some(gpu) = open() else { return };
    cases::the_dense_projection_matches_the_host_in_every_format(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_routed_stages_match_the_host_in_every_format() {
    let Some(gpu) = open() else { return };
    cases::the_routed_stages_match_the_host_in_every_format(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_vocabulary_gather_decodes_rows_exactly() {
    let Some(gpu) = open() else { return };
    cases::the_vocabulary_gather_decodes_rows_exactly(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_ngram_row_decode_places_iq4_nl_rows_exactly() {
    let Some(gpu) = open() else { return };
    cases::the_ngram_row_decode_places_iq4_nl_rows_exactly(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_prompt_projection_matches_the_host_in_every_block_format() {
    let Some(gpu) = open() else { return };
    cases::the_prompt_projection_matches_the_host_in_every_block_format(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_prompt_routed_stages_match_the_host_in_every_block_format() {
    let Some(gpu) = open() else { return };
    cases::the_prompt_routed_stages_match_the_host_in_every_block_format(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_lane_step_entries_match_the_host_in_every_block_format() {
    let Some(gpu) = open() else { return };
    cases::the_lane_step_entries_match_the_host_in_every_block_format(&gpu);
}
