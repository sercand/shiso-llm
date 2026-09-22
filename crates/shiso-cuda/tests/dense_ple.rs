//! CUDA's run of the shared `dense_ple` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! These fail until the `dense` module exists, and they are supposed to. A
//! refusal naming the missing module is the baseline every kernel task in this
//! phase moves off; a *pass* here would mean the cases asserted nothing.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test dense_ple
//! -- --ignored`
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::ple;

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
fn a_resident_table_mixes_the_normalized_projection_with_the_row_its_id_names() {
    let Some(gpu) = open() else { return };
    ple::a_resident_table_mixes_the_normalized_projection_with_the_row_its_id_names(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_streamed_table_reads_the_row_that_was_staged_for_each_token() {
    let Some(gpu) = open() else { return };
    ple::a_streamed_table_reads_the_row_that_was_staged_for_each_token(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_two_slots_of_one_token_are_two_different_rows_of_the_table() {
    let Some(gpu) = open() else { return };
    ple::the_two_slots_of_one_token_are_two_different_rows_of_the_table(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_affine_prepare_mixes_the_projection_with_the_row_its_id_names() {
    let Some(gpu) = open() else { return };
    ple::the_affine_prepare_mixes_the_projection_with_the_row_its_id_names(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_affine_prepare_reads_the_row_that_was_staged_for_each_token() {
    let Some(gpu) = open() else { return };
    ple::the_affine_prepare_reads_the_row_that_was_staged_for_each_token(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_two_slots_of_one_affine_row_are_two_different_spans() {
    let Some(gpu) = open() else { return };
    ple::the_two_slots_of_one_affine_row_are_two_different_spans(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_gate_is_the_activation_times_the_named_slots_row() {
    let Some(gpu) = open() else { return };
    ple::the_gate_is_the_activation_times_the_named_slots_row(&gpu);
}
