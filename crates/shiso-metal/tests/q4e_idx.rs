//! Metal's run of the shared `qwen4exp` idx cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --test q4e_idx -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_kerneltest::Gpu;
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

use shiso_kerneltest::qwen4exp::indexer;

#[test]
#[ignore = "requires a Metal device"]
fn the_scalar_scorer_follows_a_scattered_page_table() {
    let Some(gpu) = open() else { return };
    indexer::the_scalar_scorer_follows_a_scattered_page_table(&gpu);
}

use shiso_kerneltest::qwen4exp::idx;

#[test]
#[ignore = "requires a Metal device"]
fn the_key_store_follows_a_scattered_page_table() {
    let Some(gpu) = open() else { return };
    idx::the_key_store_follows_a_scattered_page_table(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_pooled_pass_covers_the_blocks_a_chunk_completes() {
    let Some(gpu) = open() else { return };
    idx::the_pooled_pass_covers_the_blocks_a_chunk_completes(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_query_pass_norms_and_rotates_every_head() {
    let Some(gpu) = open() else { return };
    idx::the_query_pass_norms_and_rotates_every_head(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_scalar_scorer_covers_a_partial_second_query_tile() {
    let Some(gpu) = open() else { return };
    idx::the_scalar_scorer_covers_a_partial_second_query_tile(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_selection_keeps_the_best_blocks_in_index_order() {
    let Some(gpu) = open() else { return };
    idx::the_selection_keeps_the_best_blocks_in_index_order(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_expansion_appends_every_tail_length() {
    let Some(gpu) = open() else { return };
    idx::the_expansion_appends_every_tail_length(&gpu);
}
