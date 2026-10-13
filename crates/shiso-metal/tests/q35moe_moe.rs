//! Metal's run of the shared `qwen35moe` cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --features metal --test kernels q35moe_moe:: --
//! --ignored --test-threads=1`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen35moe::experts as block;
use shiso_kerneltest::qwen35moe::q4affine as cases;
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
fn the_affine_tile_matches_the_host_product() {
    let Some(gpu) = open() else { return };
    cases::the_affine_tile_matches_the_host_product(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_router_picks_what_the_host_reference_picks() {
    let Some(gpu) = open() else { return };
    cases::the_router_picks_what_the_host_reference_picks(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_activation_matches_the_host_silu_product() {
    let Some(gpu) = open() else { return };
    cases::the_activation_matches_the_host_silu_product(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_combine_matches_the_host_weighted_sum() {
    let Some(gpu) = open() else { return };
    cases::the_combine_matches_the_host_weighted_sum(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_shared_add_matches_the_host_gated_add() {
    let Some(gpu) = open() else { return };
    cases::the_shared_add_matches_the_host_gated_add(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_sort_histogram_counts_every_column() {
    let Some(gpu) = open() else { return };
    cases::the_sort_histogram_counts_every_column(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_sort_scan_is_the_exclusive_prefix_of_the_counts() {
    let Some(gpu) = open() else { return };
    cases::the_sort_scan_is_the_exclusive_prefix_of_the_counts(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_sort_scatter_fills_every_expert_span() {
    let Some(gpu) = open() else { return };
    cases::the_sort_scatter_fills_every_expert_span(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_gate_up_stage_matches_the_host_reference() {
    let Some(gpu) = open() else { return };
    cases::the_gate_up_stage_matches_the_host_reference(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_down_stage_matches_the_host_reference() {
    let Some(gpu) = open() else { return };
    cases::the_down_stage_matches_the_host_reference(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_gate_up_stage_in_bf16_matches_the_host_reference() {
    let Some(gpu) = open() else { return };
    cases::the_gate_up_stage_in_bf16_matches_the_host_reference(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_down_stage_in_bf16_matches_the_host_reference() {
    let Some(gpu) = open() else { return };
    cases::the_down_stage_in_bf16_matches_the_host_reference(&gpu);
}

// The whole routed block, over the q4-affine codec, against a host reference
// that does the same block in plain Rust.
//
// The cases above grade the entries one at a time, and not one of them can see
// a seam: a routed column is `token * used + slot`, gate and up read activation
// row `col / used` while down reads its own row `col`, down writes column-major
// over (token, slot), and the per-expert parameters are indexed by expert id.
// The four below launch all eight stages in order and are the only cases in
// this file that grade those four conventions.
//
// They are `shiso-kerneltest::qwen35moe::experts`' own cases - the same four
// that `shiso-cuda/tests/q35moe_experts.rs` runs over NVFP4. The host reference
// is one implementation for both codecs; only the two expert entries and the
// widths their packers allow differ.

#[test]
#[ignore = "requires a Metal device"]
fn the_routed_block_in_q4_affine_matches_the_host_reference() {
    let Some(gpu) = open() else { return };
    block::the_routed_block_in_q4_affine_matches_the_host_reference(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_routed_block_in_q4_affine_at_one_token_matches_the_same_reference() {
    let Some(gpu) = open() else { return };
    block::the_routed_block_in_q4_affine_at_one_token_matches_the_same_reference(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_first_row_of_a_q4_affine_chunk_is_what_that_token_answers_alone() {
    let Some(gpu) = open() else { return };
    block::the_first_row_of_a_q4_affine_chunk_is_what_that_token_answers_alone(&gpu);
}

// The shared expert's gate has no codec dimension: its fixture is three float
// buffers and no packed weight at all. It runs once here, as the fourth of the
// block cases, because the property it grades - that the gate is one scalar per
// token and not a vector per channel - is a property of this backend's
// `q4e_shared_add_kernel` and is not graded by the codec cases above.
#[test]
#[ignore = "requires a Metal device"]
fn the_shared_expert_s_gate_is_one_scalar_per_token() {
    let Some(gpu) = open() else { return };
    block::the_shared_expert_s_gate_is_one_scalar_per_token(&gpu);
}
