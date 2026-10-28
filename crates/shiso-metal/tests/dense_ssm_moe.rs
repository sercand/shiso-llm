//! Metal's run of the shared state-space, short-convolution and routed
//! feed-forward cases.
//!
//! The pipeline check is not ignore-gated; the rest run with
//! `cargo test -p shiso-metal --features metal --test kernels dense_ssm_moe:: -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::{routed, short_conv, ssm};
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
fn every_entry_point_resolves_as_a_pipeline() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the pipeline check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    for entry in ssm::ENTRIES
        .iter()
        .chain(routed::ENTRIES.iter())
        .chain(short_conv::ENTRIES.iter())
    {
        backend
            .prepare(KernelRef::new("dense", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn ssm_conv_carries_its_window_across_chunks() {
    let Some(gpu) = open() else { return };
    ssm::ssm_conv_carries_its_window_across_chunks(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn ssm_scan_carries_its_state_across_chunks() {
    let Some(gpu) = open() else { return };
    ssm::ssm_scan_carries_its_state_across_chunks(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn short_conv_carries_its_window_across_chunks() {
    let Some(gpu) = open() else { return };
    short_conv::short_conv_carries_its_window_across_chunks(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn ssm_gate_norm_matches_the_host_rule() {
    let Some(gpu) = open() else { return };
    ssm::ssm_gate_norm_matches_the_host_rule(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn route_sigmoid_chooses_by_biased_score() {
    let Some(gpu) = open() else { return };
    routed::route_sigmoid_chooses_by_biased_score(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn experts_affine4_matches_the_host_product() {
    let Some(gpu) = open() else { return };
    routed::experts_affine4_matches_the_host_product(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn iq4nl_expert_stage_matches_host_dequant() {
    let Some(gpu) = open() else { return };
    routed::iq4nl_expert_stage_matches_host_dequant(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn iq4nl_expert_stage_reads_through_the_slot_map() {
    let Some(gpu) = open() else { return };
    routed::iq4nl_expert_stage_reads_through_the_slot_map(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn iq4nl_expert_stage_relu2_is_fused() {
    let Some(gpu) = open() else { return };
    routed::iq4nl_expert_stage_relu2_is_fused(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn iq4nl_expert_gemm_matches_host_dequant() {
    let Some(gpu) = open() else { return };
    routed::iq4nl_expert_gemm_matches_host_dequant(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn iq4nl_expert_gemm_reads_through_the_slot_map() {
    let Some(gpu) = open() else { return };
    routed::iq4nl_expert_gemm_reads_through_the_slot_map(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn iq4nl_expert_gemm_column_is_independent_of_its_neighbours() {
    let Some(gpu) = open() else { return };
    routed::iq4nl_expert_gemm_column_is_independent_of_its_neighbours(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn relu2_squares_the_rectified_value() {
    let Some(gpu) = open() else { return };
    routed::relu2_squares_the_rectified_value(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn kv_store_writes_the_paged_rows() {
    let Some(gpu) = open() else { return };
    routed::kv_store_writes_the_paged_rows(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn route_sort_groups_columns_by_expert() {
    let Some(gpu) = open() else { return };
    routed::route_sort_groups_columns_by_expert(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn combine_add_folds_the_shared_row_in() {
    let Some(gpu) = open() else { return };
    routed::combine_add_folds_the_shared_row_in(&gpu);
}
