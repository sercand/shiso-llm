//! CUDA's run of the shared `qwen4exp` cases for the `q4e_kv8` module: the
//! FP8 key/value cache's store and readers.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run. Each reader case runs the f16 kernel beside its FP8
//! twin, so the f16 modules have to be built too.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q4e_kv8
//! -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::kv8;

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
fn the_store_writes_the_e4m3_code_of_each_value() {
    let Some(gpu) = open() else { return };
    kv8::the_store_writes_the_e4m3_code_of_each_value(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_batched_store_writes_the_e4m3_code_of_each_value() {
    let Some(gpu) = open() else { return };
    kv8::the_batched_store_writes_the_e4m3_code_of_each_value(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_code_decodes_to_its_value() {
    let Some(gpu) = open() else { return };
    kv8::every_code_decodes_to_its_value(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_dense_reader_matches_f16_at_decode() {
    let Some(gpu) = open() else { return };
    kv8::the_dense_reader_matches_f16_at_decode(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_dense_reader_matches_f16_over_a_chunk() {
    let Some(gpu) = open() else { return };
    kv8::the_dense_reader_matches_f16_over_a_chunk(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_batched_reader_matches_f16() {
    let Some(gpu) = open() else { return };
    kv8::the_batched_reader_matches_f16(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_tiled_reader_matches_f16() {
    let Some(gpu) = open() else { return };
    kv8::the_tiled_reader_matches_f16(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_split_reader_matches_f16() {
    let Some(gpu) = open() else { return };
    kv8::the_split_reader_matches_f16(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_gathered_reader_matches_f16() {
    let Some(gpu) = open() else { return };
    kv8::the_gathered_reader_matches_f16(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_tensor_core_reader_matches_f16() {
    let Some(gpu) = open() else { return };
    kv8::the_tensor_core_reader_matches_f16(&gpu);
}
