//! CUDA's run of the shared block drafter cases.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! dense_block_draft -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::block_draft;

/// Opens device 0, or says on stderr why the cases did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<(Gpu, Arc<CudaBackend>)> {
    match CudaBackend::new(0, CudaOptions::default()) {
        Ok(cuda) => {
            let device: Arc<dyn Backend> = cuda.clone();
            Some((Gpu::new(device, Box::new(cuda.clone())), cuda))
        }
        Err(err) => {
            eprintln!("note: no CUDA device on this machine, skipping the GPU cases: {err}");
            None
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_block_draft_entry_point_resolves() {
    let Some((_, cuda)) = open() else { return };
    for entry in block_draft::ENTRIES {
        cuda.prepare(KernelRef::new("dense", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn copy_rows_lands_each_row_in_its_slot() {
    let Some((gpu, _)) = open() else { return };
    block_draft::copy_rows_lands_each_row_in_its_slot(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn dyn_conv_matches_the_host_rule() {
    let Some((gpu, _)) = open() else { return };
    block_draft::dyn_conv_matches_the_host_rule(&gpu);
}
