//! Metal's run of the shared `qwen4exp` misc cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --test q4e_misc -- --ignored`.
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

use shiso_kerneltest::qwen4exp::misc;

#[test]
#[ignore = "requires a Metal device"]
fn the_ngram_add_sums_both_branches_into_the_residual() {
    let Some(gpu) = open() else { return };
    misc::the_ngram_add_sums_both_branches_into_the_residual(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_greedy_pick_takes_the_lowest_index_of_the_maximum() {
    let Some(gpu) = open() else { return };
    misc::the_greedy_pick_takes_the_lowest_index_of_the_maximum(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_half_gather_reads_one_row_per_token() {
    let Some(gpu) = open() else { return };
    misc::the_half_gather_reads_one_row_per_token(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_half_gather_fans_a_row_out_across_streams() {
    let Some(gpu) = open() else { return };
    misc::the_half_gather_fans_a_row_out_across_streams(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_single_token_gather_tiles_its_row() {
    let Some(gpu) = open() else { return };
    misc::the_single_token_gather_tiles_its_row(&gpu);
}
