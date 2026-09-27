//! Metal's run of the shared `audio` encoder cases.
//!
//! The cases live in `shiso-kerneltest`, so this file is the backend's half
//! only: open a device, or say why the cases did not run.
//!
//! One case is not ignore-gated: the pipelines have to resolve, and a missing or
//! misnamed entry point should fail the default gate rather than hide behind the
//! GPU gate. The rest are `#[ignore]`d: `cargo test -p shiso-metal --features
//! metal --test audio_kernels -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::audio::cases;
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

/// One ignored wrapper per shared case.
macro_rules! device_cases {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            #[ignore = "requires a Metal device"]
            fn $name() {
                let Some(gpu) = open() else { return };
                cases::$name(&gpu);
            }
        )*
    };
}

device_cases!(
    every_unary_op_matches_its_formula,
    unary_runs_in_place,
    axpy_adds_a_scaled_copy,
    glu_gates_the_first_half_with_the_second,
    silu_mul_gates_the_up_rows,
    layer_norm_matches_with_and_without_bias,
    layer_norm_keeps_precision_on_a_large_mean_small_variance_row,
    rms_norm_matches_with_and_without_weight,
    gemm_matches_for_every_weight_type_and_ragged_shape,
    split_k_gemm_matches_the_whole_product,
    gemv_matches_for_every_weight_type,
    tall_gemm_matches_for_every_weight_type,
    conv2d_matches_causal_symmetric_full_and_depthwise,
    conv2d_keeps_batched_planes_apart,
    im2col_lays_out_the_patches_of_every_plane,
    dwconv1d_sums_taps_over_pre_padded_rows,
    unfold_stacks_neighbouring_rows_with_zero_edges,
    transpose_swaps_the_inner_two_axes,
    window_attention_chunked_relpos_matches,
    window_attention_sliding_softcapped_matches,
    window_attention_full_bidirectional_matches,
    a_query_with_no_visible_key_yields_zeros,
    rope_half_rotates_pairs_across_the_halves,
    lstm_cell_matches_the_gate_equations,
    bias_rows_adds_one_row_to_every_row,
    argmax_rows_keeps_the_lowest_index_of_the_largest,
    greedy_step_follows_the_search_rules,
    greedy_window_adds_the_prediction_past_the_frame,
    embed_row_takes_the_last_tokens_row,
    gated_lstm_cell_keeps_the_state_only_on_emit,
    keep_copies_only_on_emit,
    causal_attention_matches_for_gqa_and_a_cache_offset,
    decode_attention_reads_its_length_from_the_state,
    kv_append_writes_at_the_state_position,
    decode_step_stops_on_eos_and_goes_quiet,
    embed_token_takes_the_state_token_row,
    rope_half_state_takes_its_position_from_the_state,
);

#[test]
#[expect(
    clippy::print_stderr,
    reason = "a machine with no Metal device skips rather than fails, and the reason belongs on stderr"
)]
fn every_audio_entry_point_resolves_as_a_pipeline() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the pipeline check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    for entry in cases::ENTRIES {
        backend
            .prepare(KernelRef::new("audio", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}
