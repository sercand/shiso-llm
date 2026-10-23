//! The CPU device's kernel table.
//!
//! One row per `(module, entry)`: the group code and how the dispatcher must
//! treat its arguments. Later kernels add their rows here; a launch that names
//! an entry without a row is `Unsupported`, never a silent no-op.

mod args;
mod generic;

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
];
