//! Weights kept in their GGUF blocks: which formats the block entries read,
//! the entry each stage takes for a format, and the launch shape they share.
//!
//! One simdgroup computes one output row, `ROWS` rows a threadgroup, and a
//! pass carries `COLS` activation rows (or routed columns) in registers, so
//! the grid's second axis is the row count over `COLS`. The kernels are in
//! `q4e/ggml.metal` and its CUDA twin `q4e/ggml.cu`; this is the host side of
//! their contract.
//!
//! The CUDA set also has prompt entries on tensor cores ([`Stage::Gemm`],
//! [`Stage::GemmGateUp`], [`Stage::GemmDown`]): the step entries' argument
//! lists, tiles on tensor cores, activations rounded to halves and the
//! weights exact. Their sums are not the step entries' sums, so only a
//! prompt pass, which is held to every prompt width rather than to a step,
//! takes them.

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
/// Output rows a dense prompt block covers (`Q4E_GG_DENSE_BM`).
pub const GEMM_DENSE_ROWS: u32 = 128;
/// Columns a dense prompt block covers (`Q4E_GG_DENSE_BN`).
pub const GEMM_DENSE_COLS: u32 = 128;
/// Output rows a gate/up prompt block covers, half of each matrix
/// (`Q4E_GG_GATE_UP_BM`).
pub const GEMM_GATE_UP_ROWS: u32 = 256;
/// Output rows a down prompt block covers (`Q4E_GG_DOWN_BM`).
pub const GEMM_DOWN_ROWS: u32 = 128;
/// Columns a routed prompt block covers at a time (`Q4E_GG_EXPERT_BN`).
pub const GEMM_EXPERT_COLS: u32 = 64;

/// Dynamic shared memory a prompt block of `rows x cols` takes
/// (`Q4E_GG_SMEM`): two buffers of 40-half rows of weights and activations,
/// two of the run scales.
#[must_use]
pub const fn gemm_smem(rows: u32, cols: u32) -> u32 {
    2 * (rows + cols) * 40 * 2 + 2 * rows * 2 * 4
}

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
    /// A prompt pass's dense projection on tensor cores.
    Gemm,
    /// A prompt pass's routed gate and up projections on tensor cores.
    GemmGateUp,
    /// A prompt pass's routed down projection on tensor cores.
    GemmDown,
    /// A step pass's dense projection, `lpr` lanes a row.
    StepMatvec,
    /// [`Stage::StepMatvec`]'s sums with four rows a unit, so a pass of
    /// several activation rows reads each once for four output rows.
    StepMatvec4,
    /// A step pass's routed gate and up projections, `lpr` lanes a row.
    StepGateUp,
    /// A step pass's routed down projection, `lpr` lanes a row.
    StepDown,
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
/// instantiation. Single precision has no prompt entries.
#[must_use]
pub const fn entry(dtype: DType, stage: Stage) -> Option<Entry> {
    // Single precision has no routed entries past the Metal-shared ones and
    // no prompt entries: its prompt rows go to the BLAS tier.
    if matches!(dtype, DType::F32)
        && matches!(
            stage,
            Stage::Gemm | Stage::GemmGateUp | Stage::GemmDown | Stage::StepGateUp | Stage::StepDown
        )
    {
        return None;
    }
    macro_rules! names {
        ($n:literal) => {
            match stage {
                Stage::Matvec => concat!("q4e_g_matvec_", $n, "_kernel"),
                Stage::GateUp => concat!("q4e_g_experts_gate_up_", $n, "_kernel"),
                Stage::Down => concat!("q4e_g_experts_down_", $n, "_kernel"),
                Stage::Embed => concat!("q4e_g_embed_", $n, "_kernel"),
                Stage::Gemm => concat!("q4e_g_gemm_", $n, "_kernel"),
                Stage::GemmGateUp => concat!("q4e_g_experts_gemm_gate_up_", $n, "_kernel"),
                Stage::GemmDown => concat!("q4e_g_experts_gemm_down_", $n, "_kernel"),
                Stage::StepMatvec => concat!("q4e_g_step_matvec_", $n, "_kernel"),
                Stage::StepMatvec4 => concat!("q4e_g_step_matvec4_", $n, "_kernel"),
                Stage::StepGateUp => concat!("q4e_g_step_gate_up_", $n, "_kernel"),
                Stage::StepDown => concat!("q4e_g_step_down_", $n, "_kernel"),
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

/// The dense prompt launch over `rows` activation rows: two threads an
/// output row.
#[must_use]
pub const fn gemm_rows_launch(out_dim: u32, rows: u32) -> Launch {
    Launch::new(
        Extent3::new(
            out_dim.div_ceil(GEMM_DENSE_ROWS),
            rows.div_ceil(GEMM_DENSE_COLS),
            1,
        ),
        2 * GEMM_DENSE_ROWS,
        gemm_smem(GEMM_DENSE_ROWS, GEMM_DENSE_COLS),
    )
}

/// Dynamic shared memory a routed prompt block takes (`Q4E_GG_SMEM_H`): its
/// half activations arrive by asynchronous copy into three buffers.
#[must_use]
pub const fn gemm_smem_half(rows: u32, cols: u32) -> u32 {
    (2 * rows + 3 * cols) * 40 * 2 + 2 * rows * 2 * 4
}

/// The routed gate/up prompt launch over `groups` active experts; a block
/// covers half its rows of each matrix.
#[must_use]
pub const fn gemm_gate_up_launch(out_dim: u32, groups: u32) -> Launch {
    Launch::new(
        Extent3::new(out_dim.div_ceil(GEMM_GATE_UP_ROWS / 2), groups, 1),
        2 * GEMM_GATE_UP_ROWS,
        gemm_smem_half(GEMM_GATE_UP_ROWS, GEMM_EXPERT_COLS),
    )
}

/// The routed down prompt launch over `groups` active experts.
#[must_use]
pub const fn gemm_down_launch(out_dim: u32, groups: u32) -> Launch {
    Launch::new(
        Extent3::new(out_dim.div_ceil(GEMM_DOWN_ROWS), groups, 1),
        2 * GEMM_DOWN_ROWS,
        gemm_smem_half(GEMM_DOWN_ROWS, GEMM_EXPERT_COLS),
    )
}

/// Warps a step block runs (`Q4E_GS_WARPS`).
pub const STEP_WARPS: u32 = 4;
/// Activation rows a step pass carries (`Q4E_GS_COLS`).
pub const STEP_COLS: u32 = 8;

/// Which lanes-a-row step entries a device's kernel set has. The two take the
/// same arguments and launches; what differs is what a lane is fast at, and
/// so how many of them the host gives a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepTier {
    /// Lanes that each read 16-weight runs through aligned words and are
    /// bound by memory: many lanes a row.
    Runs,
    /// Lanes that each decode 32-weight sub-blocks and are bound by how many
    /// threadgroups a launch starts: few lanes a row, many rows a
    /// threadgroup.
    Packed,
}

/// Sub-blocks a lane of a [`StepTier::Packed`] row walks, at least.
///
/// Measured on an M1 Max over a plain decode step of the published sparse
/// checkpoint in `UD-IQ3_XXS`: 2.9 ms of device time a layer at eight, 3.2 at
/// four, and 3.7 with four rows a lane on top of eight.
const PACKED_SUB_BLOCKS: u32 = 8;

/// Rows a [`StepTier::Packed`] routed unit computes against one read of the
/// activations (`Q4E_GS_UNIT_ROWS`): eight of down, or four of gate and the
/// same four of up.
pub const PACKED_UNIT_ROWS: u32 = 8;

/// Sub-blocks of a row a routed unit's lanes share out, at least, per lane
/// of [`step_expert_lanes`]: 2560 wide gives a unit sixteen lanes of five
/// sub-blocks, 640 wide sixteen lanes of one or two.
const PACKED_UNIT_SUB_BLOCKS: u32 = 10;

/// Lanes a [`StepTier::Packed`] row takes: the largest power of two that
/// still leaves each lane [`PACKED_SUB_BLOCKS`] sub-blocks, a simdgroup at
/// most. From the contraction width alone, so every pass width sums a row
/// alike.
#[must_use]
pub const fn packed_lanes(in_dim: u32) -> u32 {
    let per = in_dim / SUB_BLOCK / PACKED_SUB_BLOCKS;
    let mut lanes = 1;
    while lanes * 2 <= per && lanes < 32 {
        lanes *= 2;
    }
    lanes
}

/// Lanes a dense step projection gives one output row, from the contraction
/// width alone, so every pass width sums a row alike: a warp up to 32 runs
/// of 16 weights, a whole block past that.
///
/// Measured on a GB10 streaming the weight (one row, against the one-warp
/// entries): `Q6_K` 10240 x 2560 188 -> 207 GB/s, 2560 x 6144 140-175 -> 192,
/// the 248k-row head 191 -> 203; `Q8_0` 320 x 10240 157 -> 203 at a block a
/// row, 10240 x 320 124 -> 181 at a warp.
#[must_use]
pub const fn step_lanes(in_dim: u32, tier: StepTier) -> u32 {
    if matches!(tier, StepTier::Packed) {
        packed_lanes(in_dim)
    } else if in_dim / 16 <= 32 {
        32
    } else {
        32 * STEP_WARPS
    }
}

/// Lanes a routed step stage gives one output row, from the contraction
/// width alone: 16 for the 640-wide down projection, a warp for gate/up.
///
/// Measured on a GB10 over ten experts (one column each), against the
/// one-warp entries: `IQ2_S` gate/up 183 -> 83 us, `IQ4_NL` down 95 -> 54 us.
#[must_use]
pub const fn step_expert_lanes(in_dim: u32, tier: StepTier) -> u32 {
    if matches!(tier, StepTier::Packed) {
        // A packed routed unit is `PACKED_UNIT_ROWS` rows over that many
        // times these lanes, a simdgroup at most.
        let per = in_dim / SUB_BLOCK / PACKED_UNIT_SUB_BLOCKS;
        let mut lanes = 1;
        while lanes * 2 <= per && lanes * 2 * PACKED_UNIT_ROWS <= 32 {
            lanes *= 2;
        }
        lanes
    } else if in_dim / 16 <= 64 {
        16
    } else {
        32
    }
}

/// Activation rows past which a dense step projection takes
/// [`Stage::StepMatvec4`]. The two sum a row identically, so the choice moves
/// time, not bits. Measured on a GB10 (`Q6_K` 10240 x 2560): four rows
/// 150 -> 109 us, eight 252 -> 152 us; one row 107 -> 110 us.
pub const STEP_FOUR_ROWS_PAST: u32 = 2;

/// Output rows a unit of a dense step entry covers at `rows` activation
/// rows, and the stage that covers them.
#[must_use]
pub const fn step_stage(rows: u32) -> (Stage, u32) {
    if rows > STEP_FOUR_ROWS_PAST {
        (Stage::StepMatvec4, 4)
    } else {
        (Stage::StepMatvec, 1)
    }
}

/// The dense step launch over `rows` activation rows at `lpr` lanes a row.
#[must_use]
pub const fn step_launch(out_dim: u32, rows: u32, lpr: u32) -> Launch {
    let per_block = STEP_WARPS * 32 / lpr * step_stage(rows).1;
    Launch::new(
        Extent3::new(out_dim.div_ceil(per_block), rows.div_ceil(STEP_COLS), 1),
        32 * STEP_WARPS,
        0,
    )
}

/// The routed gate/up step launch: a warp's rows alternate the two matrices.
#[must_use]
pub const fn step_gate_up_launch(out_dim: u32, groups: u32, lpr: u32) -> Launch {
    let per_block = STEP_WARPS * 16 / lpr;
    Launch::new(
        Extent3::new(out_dim.div_ceil(per_block), groups, 1),
        32 * STEP_WARPS,
        0,
    )
}

/// The routed down step launch.
#[must_use]
pub const fn step_down_launch(out_dim: u32, groups: u32, lpr: u32) -> Launch {
    let per_block = STEP_WARPS * 32 / lpr;
    Launch::new(
        Extent3::new(out_dim.div_ceil(per_block), groups, 1),
        32 * STEP_WARPS,
        0,
    )
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
            for stage in [
                Stage::Matvec,
                Stage::GateUp,
                Stage::Down,
                Stage::Embed,
                Stage::Gemm,
                Stage::GemmGateUp,
                Stage::GemmDown,
                Stage::StepMatvec,
                Stage::StepMatvec4,
                Stage::StepGateUp,
                Stage::StepDown,
            ] {
                assert!(entry(dtype, stage).is_some(), "{dtype:?} {stage:?}");
            }
        }
        assert!(!is_block(DType::F32));
        assert!(entry(DType::F32, Stage::Matvec).is_some());
        assert!(entry(DType::F32, Stage::Gemm).is_none());
        assert!(entry(DType::F32, Stage::StepMatvec).is_some());
        assert!(entry(DType::F32, Stage::StepGateUp).is_none());
        assert!(entry(DType::Q4_K, Stage::Matvec).is_none());
    }

    #[test]
    fn a_projection_grid_covers_partial_row_blocks() {
        let l = rows_launch(37, 11);
        assert_eq!((l.grid.x, l.grid.y), (10, 2));
    }
}
