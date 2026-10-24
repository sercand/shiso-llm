//! CUDA's run of the shared `generic` and `qwen35` module cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! These grade the entries a Gemma 4 pass dispatches outside the `dense`
//! module - the two fused norms, the residual add, the `q6_K` vocabulary
//! gather, the `q6_K` output-head matvec at all fifteen instantiations, and the
//! two halves of the per-layer model projection. None of them had a case
//! anywhere in the tree before, on either backend. The two `q8_0` cases at the
//! end are a qwen35 pass's rather than a Gemma 4 one's: that container keeps
//! its two gate projections at that codec, and neither product existed on this
//! backend until Phase 8.
//!
//! The staging case at the end grades all four `kernel_dequant_plane_*`
//! entries against the staged plane itself. Those had no case for any dtype
//! anywhere in the tree, which is how an `f16` plane that was correct only in
//! its first four-element chunk shipped and took Gemma 4 from 64/64 to 0/64.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test generic --
//! --ignored`. Without `SHISO_CUDA_COMPILE=1` the kernel crate embeds an empty
//! registry and the refusal says so, which is a different message from the one
//! a real build gives.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::generic::{glu, norm, plane, product, rows_q8};

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
fn the_plain_norm_scales_the_row_and_multiplies_by_the_weight() {
    let Some(gpu) = open() else { return };
    norm::the_plain_norm_scales_the_row_and_multiplies_by_the_weight(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_fused_close_writes_both_the_sum_and_the_normalized_row() {
    let Some(gpu) = open() else { return };
    norm::the_fused_close_writes_both_the_sum_and_the_normalized_row(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_residual_add_writes_its_own_destination_and_stops_at_its_bound() {
    let Some(gpu) = open() else { return };
    norm::the_residual_add_writes_its_own_destination_and_stops_at_its_bound(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_q6_k_gather_dequantizes_the_rows_the_ids_name() {
    let Some(gpu) = open() else { return };
    product::the_q6_k_gather_dequantizes_the_rows_the_ids_name(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_q6_k_matvec_instantiation_agrees_with_the_oracle() {
    let Some(gpu) = open() else { return };
    product::every_q6_k_matvec_instantiation_agrees_with_the_oracle(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_f16_prefill_gemm_matches_a_triple_loop() {
    let Some(gpu) = open() else { return };
    product::the_f16_prefill_gemm_matches_a_triple_loop(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_f16_decode_matvec_matches_the_oracle_dot() {
    let Some(gpu) = open() else { return };
    product::the_f16_decode_matvec_matches_the_oracle_dot(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_q8_0_prefill_gemm_matches_a_triple_loop() {
    let Some(gpu) = open() else { return };
    product::the_q8_0_prefill_gemm_matches_a_triple_loop(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_q8_0_decode_matvec_matches_the_oracle_dot() {
    let Some(gpu) = open() else { return };
    product::the_q8_0_decode_matvec_matches_the_oracle_dot(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_gated_silu_is_the_sigmoid_weighted_gate_times_the_up_row() {
    let Some(gpu) = open() else { return };
    glu::the_gated_silu_is_the_sigmoid_weighted_gate_times_the_up_row(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_clamp_is_one_sided_on_the_gate_and_two_sided_on_the_up_row() {
    let Some(gpu) = open() else { return };
    glu::the_clamp_is_one_sided_on_the_gate_and_two_sided_on_the_up_row(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_f32_decode_matvec_matches_the_oracle_dot() {
    let Some(gpu) = open() else { return };
    product::the_f32_decode_matvec_matches_the_oracle_dot(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_f16_gather_matches_the_table() {
    let Some(gpu) = open() else { return };
    product::the_f16_gather_matches_the_table(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_staged_plane_matches_the_weight_at_every_codec() {
    let Some(gpu) = open() else { return };
    plane::the_staged_plane_matches_the_weight_at_every_codec(&gpu);
}

// The `q8_0` gather has a CUDA entry (`kernel_get_rows_q8_0_f32`, instantiated
// in `kernels/cuda/generic/get_rows_block.cuh`). The extended matvec case
// beside it in `rows_q8` has no wrapper here on purpose: it is Metal-only,
// because CUDA specializes its matvecs by entry name and has no `q8_0`
// extended table to grade.
#[test]
#[ignore = "requires a CUDA device"]
fn the_q8_0_gather_dequantizes_the_rows_the_ids_name() {
    let Some(gpu) = open() else { return };
    rows_q8::the_q8_0_gather_dequantizes_the_rows_the_ids_name(&gpu);
}
