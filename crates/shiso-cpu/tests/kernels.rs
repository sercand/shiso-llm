//! The CPU device's run of the shared kernel cases.
//!
//! The cases live in `shiso-kerneltest`, so this file is the backend's half
//! only: open the device and name the cases it supports. Unlike the GPU
//! backends' cases these are not `#[ignore]`d: the CPU is always present.
//! Later kernel tasks add their cases to the `cpu_cases!` lists below.

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cpu::{CpuBackend, CpuOptions};
use shiso_kerneltest::Gpu;

/// Opens the CPU device. The keepalive is the backend itself.
fn open() -> Gpu {
    let cpu = CpuBackend::new(CpuOptions::default()).expect("the CPU device always opens");
    let device: Arc<dyn Backend> = cpu.clone();
    Gpu::new(device, Box::new(cpu))
}

/// One `#[test]` per case, named after it, each on a fresh device. A case is
/// written `module::file::case`, its path under `shiso_kerneltest`.
macro_rules! cpu_cases {
    ($($m:ident :: $f:ident :: $case:ident),+ $(,)?) => {
        $(
            #[test]
            fn $case() {
                shiso_kerneltest::$m::$f::$case(&open());
            }
        )+
    };
}

cpu_cases!(
    generic::norm::the_plain_norm_scales_the_row_and_multiplies_by_the_weight,
    generic::norm::the_fused_close_writes_both_the_sum_and_the_normalized_row,
    generic::glu::the_gated_silu_is_the_sigmoid_weighted_gate_times_the_up_row,
    generic::glu::the_clamp_is_one_sided_on_the_gate_and_two_sided_on_the_up_row,
    generic::rows_q8::the_q8_0_gather_dequantizes_the_rows_the_ids_name,
    generic::rows_q8::the_q8_0_ext_matvec_matches_the_oracle_dot_at_every_row_count,
    generic::product::the_q8_0_decode_matvec_matches_the_oracle_dot,
    generic::product::the_q8_0_prefill_gemm_matches_a_triple_loop,
    dense::attn::rope_without_a_norm_rotates_the_raw_head,
    dense::attn::the_rope_only_store_keeps_the_value_raw,
    dense::attn::window_and_bound_select_the_keys,
    dense::attn::scale_is_an_argument,
    dense::attn::an_empty_span_contributes_zeros_rather_than_nan,
    audio::cases::every_unary_op_matches_its_formula,
    audio::cases::unary_runs_in_place,
    audio::cases::axpy_adds_a_scaled_copy,
    audio::cases::layer_norm_matches_with_and_without_bias,
    audio::cases::layer_norm_keeps_precision_on_a_large_mean_small_variance_row,
    audio::cases::window_attention_full_bidirectional_matches,
    audio::cases::a_query_with_no_visible_key_yields_zeros,
    audio::cases::q8_0w_gemm_matches_for_ragged_shapes,
    audio::cases::q8_0w_split_k_gemm_matches_the_whole_product,
    audio::cases::q8_0w_tall_gemm_matches_the_tiled_one,
    audio::cases::q8_0w_gemv_matches,
);
