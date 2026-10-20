//! CUDA's run of the shared `qwen4exp` raw key ring cases for the
//! `q4e_idxring` module: the pooled pass over a per-session ring, graded bit
//! for bit against the raw key plane it replaced, run side by side.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q4e_idxring
//! -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::idxring;

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
fn chunked_prefill_at_any_boundary_pools_as_the_raw_plane() {
    let Some(gpu) = open() else { return };
    idxring::chunked_prefill_at_any_boundary_pools_as_the_raw_plane(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn decode_row_by_row_pools_as_the_raw_plane() {
    let Some(gpu) = open() else { return };
    idxring::decode_row_by_row_pools_as_the_raw_plane(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn verify_rollbacks_pool_as_the_raw_plane() {
    let Some(gpu) = open() else { return };
    idxring::verify_rollbacks_pool_as_the_raw_plane(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn draft_chain_and_catch_up_pool_as_the_raw_plane() {
    let Some(gpu) = open() else { return };
    idxring::draft_chain_and_catch_up_pool_as_the_raw_plane(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn resume_from_a_canonical_ring_pools_as_the_raw_plane() {
    let Some(gpu) = open() else { return };
    idxring::resume_from_a_canonical_ring_pools_as_the_raw_plane(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_tap_copies_the_frontier_blocks_raw_keys() {
    let Some(gpu) = open() else { return };
    idxring::the_tap_copies_the_frontier_blocks_raw_keys(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn interleaved_sessions_keep_their_own_rings() {
    let Some(gpu) = open() else { return };
    idxring::interleaved_sessions_keep_their_own_rings(&gpu);
}
