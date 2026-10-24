//! The CPU device's kernel table.
//!
//! One row per `(module, entry)`: the group code and how the dispatcher must
//! treat its arguments. Later kernels add their rows here; a launch that names
//! an entry without a row is `Unsupported`, never a silent no-op.

mod args;
mod audio;
mod dense;
mod generic;
mod qwen35;

use crate::dispatch::Table;

/// Every kernel the CPU device runs.
pub(crate) static TABLE: &Table = &[
    (
        "generic",
        "kernel_get_rows_q8_0_f32",
        generic::get_rows::Q8_0,
    ),
    (
        "generic",
        "kernel_rms_norm_mul_f32_4",
        generic::norm::RMS_NORM_MUL,
    ),
    (
        "generic",
        "kernel_add_rms_norm_mul_f32_4",
        generic::norm::ADD_RMS_NORM_MUL,
    ),
    (
        "generic",
        "kernel_swiglu_flat_f32",
        generic::glu::SWIGLU_FLAT,
    ),
    (
        "generic",
        "kernel_mul_mv_q8_0_f32",
        generic::mul_mv_q8_0::Q8_0,
    ),
    (
        "generic",
        "kernel_mul_mv_ext_q8_0_f32_r1_2",
        generic::mul_mv_ext::R1_2,
    ),
    (
        "generic",
        "kernel_mul_mv_ext_q8_0_f32_r1_3",
        generic::mul_mv_ext::R1_3,
    ),
    (
        "generic",
        "kernel_mul_mv_ext_q8_0_f32_r1_4",
        generic::mul_mv_ext::R1_4,
    ),
    (
        "generic",
        "kernel_mul_mv_ext_q8_0_f32_r1_5",
        generic::mul_mv_ext::R1_5,
    ),
    ("dense", "kernel_dense_q_rope", dense::norm_rope::Q_ROPE),
    (
        "dense",
        "kernel_dense_kv_rope_store",
        dense::norm_rope::KV_ROPE_STORE,
    ),
    ("dense", "kernel_dense_attn_decode", dense::attn::DECODE),
    ("dense", "kernel_dense_attn_prefill", dense::attn::PREFILL),
    ("qwen35", "kernel_mul_mm_q8_0_f32", qwen35::gemm::Q8_0),
    ("audio", "kernel_audio_gemm_q8_0w", audio::gemm::GEMM),
    ("audio", "kernel_audio_gemm64_q8_0w", audio::gemm::GEMM64),
    ("audio", "kernel_audio_gemv_q8_0w", audio::gemm::GEMV),
    (
        "audio",
        "kernel_audio_split_sum_f32",
        audio::gemm::SPLIT_SUM,
    ),
    (
        "audio",
        "kernel_audio_layer_norm_f32",
        audio::norm::LAYER_NORM,
    ),
    (
        "audio",
        "kernel_audio_window_attn_f32",
        audio::attn::WINDOW_ATTN,
    ),
    ("audio", "kernel_audio_unary_f32", audio::elementwise::UNARY),
    ("audio", "kernel_audio_axpy_f32", audio::elementwise::AXPY),
];
