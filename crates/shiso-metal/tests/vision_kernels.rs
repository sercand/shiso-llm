//! Metal's run of the shared `vision` tower cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! One case is not ignore-gated: the pipelines have to resolve, and a missing or
//! misnamed entry point should fail the default gate rather than hide behind the
//! GPU gate. The rest are `#[ignore]`d: `cargo test -p shiso-metal --features
//! metal --test vision_kernels -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::vision::tower;
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
#[expect(
    clippy::print_stderr,
    reason = "a machine with no Metal device skips rather than fails, and the reason belongs on stderr"
)]
fn every_vision_entry_point_resolves_as_a_pipeline() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the pipeline check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    for entry in tower::ENTRIES.iter().chain(&tower::METAL_ENTRIES) {
        backend
            .prepare(KernelRef::new("vision", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn the_blocked_gemm_agrees_with_a_row_times_column_loop() {
    let Some(gpu) = open() else { return };
    tower::the_blocked_gemm_agrees_with_a_row_times_column_loop(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_normalization_subtracts_the_mean_as_well_as_scaling_by_the_deviation() {
    let Some(gpu) = open() else { return };
    tower::the_normalization_subtracts_the_mean_as_well_as_scaling_by_the_deviation(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_normalization_is_safe_to_run_over_its_own_input() {
    let Some(gpu) = open() else { return };
    tower::the_normalization_is_safe_to_run_over_its_own_input(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn each_activation_matches_its_own_formula_and_not_the_other_one() {
    let Some(gpu) = open() else { return };
    tower::each_activation_matches_its_own_formula_and_not_the_other_one(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_activations_stay_finite_over_the_range_a_feed_forward_row_reaches() {
    let Some(gpu) = open() else { return };
    tower::the_activations_stay_finite_over_the_range_a_feed_forward_row_reaches(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_add_repeats_a_shorter_period_and_leaves_an_equal_one_elementwise() {
    let Some(gpu) = open() else { return };
    tower::the_add_repeats_a_shorter_period_and_leaves_an_equal_one_elementwise(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_rotation_splits_the_head_between_the_row_axis_and_the_column_axis() {
    let Some(gpu) = open() else { return };
    tower::the_rotation_splits_the_head_between_the_row_axis_and_the_column_axis(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_rotation_is_the_identity_at_the_grid_origin_and_leaves_the_values_alone() {
    let Some(gpu) = open() else { return };
    tower::the_rotation_is_the_identity_at_the_grid_origin_and_leaves_the_values_alone(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_attention_agrees_with_a_plain_softmax_over_every_patch() {
    let Some(gpu) = open() else { return };
    tower::the_attention_agrees_with_a_plain_softmax_over_every_patch(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn a_query_never_attends_across_a_temporal_group_boundary() {
    let Some(gpu) = open() else { return };
    tower::a_query_never_attends_across_a_temporal_group_boundary(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_spatial_merge_gathers_the_block_the_grid_names() {
    let Some(gpu) = open() else { return };
    tower::the_spatial_merge_gathers_the_block_the_grid_names(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_splice_replaces_exactly_the_rows_the_map_names() {
    let Some(gpu) = open() else { return };
    tower::the_splice_replaces_exactly_the_rows_the_map_names(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn a_source_row_past_the_block_leaves_the_embedding_alone() {
    let Some(gpu) = open() else { return };
    tower::a_source_row_past_the_block_leaves_the_embedding_alone(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_half_precision_gemm_agrees_with_a_row_times_column_loop() {
    let Some(gpu) = open() else { return };
    tower::the_half_precision_gemm_agrees_with_a_row_times_column_loop(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_half_precision_normalization_agrees_with_the_same_oracle() {
    let Some(gpu) = open() else { return };
    tower::the_half_precision_normalization_agrees_with_the_same_oracle(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_tiled_gemm_agrees_with_a_row_times_column_loop() {
    let Some(gpu) = open() else { return };
    tower::the_tiled_gemm_agrees_with_a_row_times_column_loop(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_half_precision_tiled_gemm_agrees_with_a_row_times_column_loop() {
    let Some(gpu) = open() else { return };
    tower::the_half_precision_tiled_gemm_agrees_with_a_row_times_column_loop(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_flash_attention_agrees_with_a_plain_softmax() {
    let Some(gpu) = open() else { return };
    tower::the_flash_attention_agrees_with_a_plain_softmax(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn a_flash_query_never_attends_across_a_group() {
    let Some(gpu) = open() else { return };
    tower::a_flash_query_never_attends_across_a_group(&gpu);
}
