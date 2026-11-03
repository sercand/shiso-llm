//! Metal's run of the shared `qwen4exp` raw key ring cases: the pooled pass
//! over a per-session ring, graded bit for bit against the raw key plane it
//! replaced, run side by side.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --features metal --test kernels
//! q4e_idxring:: -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::idxring;
use shiso_metal::MetalBackend;

/// Opens the device, or says on stderr why the cases did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<Gpu> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the GPU cases");
        return None;
    }
    let metal = MetalBackend::new().expect("the embedded kernel library compiles");
    let device: Arc<dyn Backend> = metal.clone();
    Some(Gpu::new(device, Box::new(metal)))
}

#[test]
#[ignore = "requires a Metal device"]
fn chunked_prefill_at_any_boundary_pools_as_the_raw_plane() {
    let Some(gpu) = open() else { return };
    idxring::chunked_prefill_at_any_boundary_pools_as_the_raw_plane(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn decode_row_by_row_pools_as_the_raw_plane() {
    let Some(gpu) = open() else { return };
    idxring::decode_row_by_row_pools_as_the_raw_plane(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn verify_rollbacks_pool_as_the_raw_plane() {
    let Some(gpu) = open() else { return };
    idxring::verify_rollbacks_pool_as_the_raw_plane(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn draft_chain_and_catch_up_pool_as_the_raw_plane() {
    let Some(gpu) = open() else { return };
    idxring::draft_chain_and_catch_up_pool_as_the_raw_plane(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn resume_from_a_canonical_ring_pools_as_the_raw_plane() {
    let Some(gpu) = open() else { return };
    idxring::resume_from_a_canonical_ring_pools_as_the_raw_plane(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_tap_copies_the_frontier_blocks_raw_keys() {
    let Some(gpu) = open() else { return };
    idxring::the_tap_copies_the_frontier_blocks_raw_keys(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn interleaved_sessions_keep_their_own_rings() {
    let Some(gpu) = open() else { return };
    idxring::interleaved_sessions_keep_their_own_rings(&gpu);
}
