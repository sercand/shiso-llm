//! CUDA's run of the shared `vision` tower cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! vision_kernels -- --ignored`. Without `SHISO_CUDA_COMPILE=1` the kernel
//! crate embeds an empty registry and the refusal says so, which is a different
//! message from the one a real build gives.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::vision::tower;

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
fn the_blocked_gemm_agrees_with_a_row_times_column_loop() {
    let Some(gpu) = open() else { return };
    tower::the_blocked_gemm_agrees_with_a_row_times_column_loop(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_normalization_subtracts_the_mean_as_well_as_scaling_by_the_deviation() {
    let Some(gpu) = open() else { return };
    tower::the_normalization_subtracts_the_mean_as_well_as_scaling_by_the_deviation(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_normalization_is_safe_to_run_over_its_own_input() {
    let Some(gpu) = open() else { return };
    tower::the_normalization_is_safe_to_run_over_its_own_input(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn each_activation_matches_its_own_formula_and_not_the_other_one() {
    let Some(gpu) = open() else { return };
    tower::each_activation_matches_its_own_formula_and_not_the_other_one(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_activations_stay_finite_over_the_range_a_feed_forward_row_reaches() {
    let Some(gpu) = open() else { return };
    tower::the_activations_stay_finite_over_the_range_a_feed_forward_row_reaches(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_add_repeats_a_shorter_period_and_leaves_an_equal_one_elementwise() {
    let Some(gpu) = open() else { return };
    tower::the_add_repeats_a_shorter_period_and_leaves_an_equal_one_elementwise(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_rotation_splits_the_head_between_the_row_axis_and_the_column_axis() {
    let Some(gpu) = open() else { return };
    tower::the_rotation_splits_the_head_between_the_row_axis_and_the_column_axis(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_rotation_is_the_identity_at_the_grid_origin_and_leaves_the_values_alone() {
    let Some(gpu) = open() else { return };
    tower::the_rotation_is_the_identity_at_the_grid_origin_and_leaves_the_values_alone(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_attention_agrees_with_a_plain_softmax_over_every_patch() {
    let Some(gpu) = open() else { return };
    tower::the_attention_agrees_with_a_plain_softmax_over_every_patch(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_query_never_attends_across_a_temporal_group_boundary() {
    let Some(gpu) = open() else { return };
    tower::a_query_never_attends_across_a_temporal_group_boundary(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_spatial_merge_gathers_the_block_the_grid_names() {
    let Some(gpu) = open() else { return };
    tower::the_spatial_merge_gathers_the_block_the_grid_names(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_splice_replaces_exactly_the_rows_the_map_names() {
    let Some(gpu) = open() else { return };
    tower::the_splice_replaces_exactly_the_rows_the_map_names(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_source_row_past_the_block_leaves_the_embedding_alone() {
    let Some(gpu) = open() else { return };
    tower::a_source_row_past_the_block_leaves_the_embedding_alone(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_operand_split_sums_back_to_its_input() {
    let Some(gpu) = open() else { return };
    tower::the_operand_split_sums_back_to_its_input(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_brain_float_bias_repeats_with_its_period() {
    let Some(gpu) = open() else { return };
    tower::the_brain_float_bias_repeats_with_its_period(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_tensor_core_attention_agrees_with_a_plain_softmax() {
    let Some(gpu) = open() else { return };
    tower::the_tensor_core_attention_agrees_with_a_plain_softmax(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_tensor_core_query_never_attends_across_a_group() {
    let Some(gpu) = open() else { return };
    tower::a_tensor_core_query_never_attends_across_a_group(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_half_precision_gemm_agrees_with_a_row_times_column_loop() {
    let Some(gpu) = open() else { return };
    tower::the_half_precision_gemm_agrees_with_a_row_times_column_loop(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_half_precision_normalization_agrees_with_the_same_oracle() {
    let Some(gpu) = open() else { return };
    tower::the_half_precision_normalization_agrees_with_the_same_oracle(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_half_precision_operand_split_sums_back_to_its_input() {
    let Some(gpu) = open() else { return };
    tower::the_half_precision_operand_split_sums_back_to_its_input(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_half_precision_bias_repeats_with_its_period() {
    let Some(gpu) = open() else { return };
    tower::the_half_precision_bias_repeats_with_its_period(&gpu);
}
