//! CUDA's run of the shared `dense_affine` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! These fail until the `dense` module exists, and they are supposed to. A
//! refusal naming the missing module is the baseline every kernel task in this
//! phase moves off; a *pass* here would mean the cases asserted nothing.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test dense_affine
//! -- --ignored`
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::affine;

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
fn qmv_4_matches_the_host_dequant_matmul() {
    let Some(gpu) = open() else { return };
    affine::qmv_4_matches_the_host_dequant_matmul(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn qmv_8_matches_the_host_dequant_matmul() {
    let Some(gpu) = open() else { return };
    affine::qmv_8_matches_the_host_dequant_matmul(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn qmm_4_matches_at_n_tok_64() {
    let Some(gpu) = open() else { return };
    affine::qmm_4_matches_at_n_tok_64(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn qmm_8_matches_at_n_tok_64() {
    let Some(gpu) = open() else { return };
    affine::qmm_8_matches_at_n_tok_64(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_row_count_that_is_not_a_tile_multiple_is_handled() {
    let Some(gpu) = open() else { return };
    affine::a_row_count_that_is_not_a_tile_multiple_is_handled(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn qmm_4_matches_at_a_chunk_that_is_not_a_whole_band() {
    let Some(gpu) = open() else { return };
    affine::qmm_4_matches_at_a_chunk_that_is_not_a_whole_band(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn qmm_8_matches_at_a_chunk_that_is_not_a_whole_band() {
    let Some(gpu) = open() else { return };
    affine::qmm_8_matches_at_a_chunk_that_is_not_a_whole_band(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn qmm_4_matches_at_a_chunk_shorter_than_one_band() {
    let Some(gpu) = open() else { return };
    affine::qmm_4_matches_at_a_chunk_shorter_than_one_band(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_group_bias_is_applied() {
    let Some(gpu) = open() else { return };
    affine::the_group_bias_is_applied(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_affine_gather_dequantizes_the_rows_the_ids_name() {
    let Some(gpu) = open() else { return };
    affine::the_affine_gather_dequantizes_the_rows_the_ids_name(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_affine_gather_reads_one_row_per_token_and_repeats_a_repeated_id() {
    let Some(gpu) = open() else { return };
    affine::the_affine_gather_reads_one_row_per_token_and_repeats_a_repeated_id(&gpu);
}
