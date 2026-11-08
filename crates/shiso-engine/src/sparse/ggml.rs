//! Weights kept in their GGUF blocks: which formats the block entries read,
//! the entry each stage takes for a format, and the launch shape they share.
//!
//! One simdgroup computes one output row, `ROWS` rows a threadgroup, and a
//! pass carries `COLS` activation rows (or routed columns) in registers, so
//! the grid's second axis is the row count over `COLS`. The kernels are in
//! `q4e/ggml.metal`; this is the host side of their contract.

use shiso_core::{DType, Extent3};

use crate::sparse::kernels::{Entry, Launch, MOD_GGML};

/// Output rows one threadgroup covers.
pub const ROWS: u32 = 4;
/// Activation rows (or routed columns) one pass carries.
pub const COLS: u32 = 8;
/// Threads a threadgroup: one simdgroup an output row.
pub const THREADS: u32 = 32 * ROWS;
/// Weights one sub-block decode yields; every contraction is a multiple.
pub const SUB_BLOCK: u32 = 32;

/// Which of a format's entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// A dense projection over any row count.
    Matvec,
    /// The routed gate and up projections.
    GateUp,
    /// The routed down projection.
    Down,
    /// The vocabulary row gather.
    Embed,
}

/// Whether `dtype` is a block format these entries read.
///
/// Single precision has entries too, for the multi-row projections its own
/// tier does not take, but it is not a block format.
#[must_use]
pub const fn is_block(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Q8_0
            | DType::Q6_K
            | DType::IQ3_XXS
            | DType::IQ4_NL
            | DType::IQ2_S
            | DType::IQ4_XS
            | DType::Q2_0
    )
}

/// The entry `stage` takes for `dtype`, or `None` for a format with no
/// instantiation.
#[must_use]
pub const fn entry(dtype: DType, stage: Stage) -> Option<Entry> {
    macro_rules! names {
        ($n:literal) => {
            match stage {
                Stage::Matvec => concat!("q4e_g_matvec_", $n, "_kernel"),
                Stage::GateUp => concat!("q4e_g_experts_gate_up_", $n, "_kernel"),
                Stage::Down => concat!("q4e_g_experts_down_", $n, "_kernel"),
                Stage::Embed => concat!("q4e_g_embed_", $n, "_kernel"),
            }
        };
    }
    let name = match dtype {
        DType::F32 => names!("f32"),
        DType::Q8_0 => names!("q8_0"),
        DType::Q6_K => names!("q6_k"),
        DType::IQ3_XXS => names!("iq3_xxs"),
        DType::IQ4_NL => names!("iq4_nl"),
        DType::IQ2_S => names!("iq2_s"),
        DType::IQ4_XS => names!("iq4_xs"),
        DType::Q2_0 => names!("q2_0"),
        _ => return None,
    };
    Some(Entry::plain(MOD_GGML, name))
}

/// The launch over `out_dim` output rows and `second` passes' worth of the
/// second axis: activation rows for a projection, active experts for a
/// routed stage.
#[must_use]
pub const fn launch(out_dim: u32, second: u32) -> Launch {
    Launch::new(Extent3::new(out_dim.div_ceil(ROWS), second, 1), THREADS, 0)
}

/// The projection launch over `rows` activation rows.
#[must_use]
pub const fn rows_launch(out_dim: u32, rows: u32) -> Launch {
    launch(out_dim, rows.div_ceil(COLS))
}

/// The gather launch: one threadgroup of one simdgroup a token.
#[must_use]
pub const fn embed_launch(n_tokens: u32) -> Launch {
    Launch::new(Extent3::new(n_tokens, 1, 1), SUB_BLOCK, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_block_format_has_all_four_entries_and_others_none() {
        for dtype in [
            DType::Q8_0,
            DType::Q6_K,
            DType::IQ3_XXS,
            DType::IQ4_NL,
            DType::IQ2_S,
            DType::IQ4_XS,
            DType::Q2_0,
        ] {
            assert!(is_block(dtype), "{dtype:?}");
            for stage in [Stage::Matvec, Stage::GateUp, Stage::Down, Stage::Embed] {
                assert!(entry(dtype, stage).is_some(), "{dtype:?} {stage:?}");
            }
        }
        assert!(!is_block(DType::F32));
        assert!(entry(DType::Q4_K, Stage::Matvec).is_none());
    }

    #[test]
    fn a_projection_grid_covers_partial_row_blocks() {
        let l = rows_launch(37, 11);
        assert_eq!((l.grid.x, l.grid.y), (10, 2));
    }
}
