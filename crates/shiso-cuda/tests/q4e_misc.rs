//! CUDA's run of the shared `qwen4exp` misc cases: the n-gram add, the greedy
//! pick and the row gathers of the `q4e_misc` and `q4e_embed` modules.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q4e_misc
//! -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::misc;

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
fn the_ngram_add_sums_both_branches_into_the_residual() {
    let Some(gpu) = open() else { return };
    misc::the_ngram_add_sums_both_branches_into_the_residual(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_greedy_pick_takes_the_lowest_index_of_the_maximum() {
    let Some(gpu) = open() else { return };
    misc::the_greedy_pick_takes_the_lowest_index_of_the_maximum(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_greedy_pick_reads_an_odd_width_one_float_at_a_time() {
    let Some(gpu) = open() else { return };
    misc::the_greedy_pick_reads_an_odd_width_one_float_at_a_time(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_half_gather_reads_one_row_per_token() {
    let Some(gpu) = open() else { return };
    misc::the_half_gather_reads_one_row_per_token(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_half_gather_fans_a_row_out_across_streams() {
    let Some(gpu) = open() else { return };
    misc::the_half_gather_fans_a_row_out_across_streams(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_single_token_gather_tiles_its_row() {
    let Some(gpu) = open() else { return };
    misc::the_single_token_gather_tiles_its_row(&gpu);
}
