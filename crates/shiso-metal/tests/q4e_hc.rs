//! Metal's run of the shared `qwen4exp` hc cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --test kernels q4e_hc:: -- --ignored`.
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

use shiso_kerneltest::qwen4exp::hc;

#[test]
#[ignore = "requires a Metal device"]
fn the_residual_broadcast_tiles_the_embedding_across_streams() {
    let Some(gpu) = open() else { return };
    hc::the_residual_broadcast_tiles_the_embedding_across_streams(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_draft_broadcast_adds_the_per_stream_term() {
    let Some(gpu) = open() else { return };
    hc::the_draft_broadcast_adds_the_per_stream_term(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_stream_norm_normalizes_each_stream_on_its_own() {
    let Some(gpu) = open() else { return };
    hc::the_stream_norm_normalizes_each_stream_on_its_own(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_half_stream_norm_normalizes_each_stream_on_its_own() {
    let Some(gpu) = open() else { return };
    hc::the_half_stream_norm_normalizes_each_stream_on_its_own(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_draft_norm_normalizes_all_streams_as_one_group() {
    let Some(gpu) = open() else { return };
    hc::the_draft_norm_normalizes_all_streams_as_one_group(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_mix_activation_scales_inside_the_silu() {
    let Some(gpu) = open() else { return };
    hc::the_mix_activation_scales_inside_the_silu(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_collapse_takes_the_mean_of_the_gated_streams() {
    let Some(gpu) = open() else { return };
    hc::the_collapse_takes_the_mean_of_the_gated_streams(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_half_collapses_agree_and_round_the_block_input_once() {
    let Some(gpu) = open() else { return };
    hc::the_half_collapses_agree_and_round_the_block_input_once(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_combine_updates_and_norms_per_stream_block() {
    let Some(gpu) = open() else { return };
    hc::the_combine_updates_and_norms_per_stream_block(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_combine_updates_and_norms_per_token_block() {
    let Some(gpu) = open() else { return };
    hc::the_combine_updates_and_norms_per_token_block(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_half_combine_updates_and_norms() {
    let Some(gpu) = open() else { return };
    hc::the_half_combine_updates_and_norms(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_unchained_combine_only_updates_the_residual() {
    let Some(gpu) = open() else { return };
    hc::the_unchained_combine_only_updates_the_residual(&gpu);
}
