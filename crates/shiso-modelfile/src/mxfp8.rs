//! MXFP8: `e4m3` weights under one power-of-two scale per 32, decoded to `f32`.
//!
//! The OCP microscaling format `modelopt` writes for `quant_algo: "MXFP8"`
//! targets. A matrix of `rows x cols` weights is two arrays:
//!
//! - `weight`, `F8_E4M3 [rows, cols]`, one `e4m3` byte per weight;
//! - `weight_scale`, `U8 [rows, cols/32]`, one `ue8m0` byte per group of
//!   [`GROUP`] consecutive weights along a row: an exponent with bias 127 and
//!   no mantissa, so a scale is exactly `2^(byte - 127)`.
//!
//! ```text
//! out_ij = e4m3(weight[i][j]) * 2^(scale[i][j / 32] - 127)
//! ```
//!
//! # Exact, and that is why it is the reference and not the serving path
//!
//! Multiplying by a power of two only moves an exponent, so every decoded
//! value is exactly representable in `bf16` (eight significant bits hold an
//! `e4m3`'s four) as long as the exponent stays in range, which it does for
//! every scale the checkpoint carries. The GPU kernels serve these matrices as
//! stored; this decode is the oracle their outputs are compared with, bit for
//! bit, after both go through the same 16-bit matmul.
//!
//! # The one reserved code
//!
//! Byte `0xFF` is NaN in `ue8m0`, a whole group of NaNs. The writer never
//! emits it for a weight scale, so it is refused at decode rather than
//! propagated into a matrix as 32 NaNs.

use crate::fp8::e4m3;
use crate::reader::{GgufError, GgufResult};

/// Weights sharing one `ue8m0` scale byte.
pub const GROUP: usize = 32;

/// Exponent bias of a `ue8m0` byte.
pub const BIAS: i32 = 127;

/// The reserved NaN code of a `ue8m0` byte.
pub const NAN_SCALE: u8 = 0xFF;

/// Decodes one `ue8m0` scale byte to the power of two it stands for.
///
/// `None` for [`NAN_SCALE`]. Every other byte is a normal `f32` except 0,
/// which is `2^-127` - an `f32` subnormal, still exact.
#[must_use]
pub fn ue8m0(byte: u8) -> Option<f32> {
    if byte == NAN_SCALE {
        return None;
    }
    let exponent = i32::from(byte) - BIAS;
    Some(if exponent >= -126 {
        // Normal: the byte is the f32 exponent field itself.
        f32::from_bits(u32::from(byte) << 23)
    } else {
        // 2^-127 is the one subnormal: mantissa bit 22 alone.
        f32::from_bits(1 << 22)
    })
}

/// How an MXFP8 matrix is laid out: its logical shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mxfp8Layout {
    /// Rows of the matrix, the slowest-varying axis.
    pub rows: usize,
    /// Weights per row; the scale array has `cols / 32` bytes a row.
    pub cols: usize,
}

impl Mxfp8Layout {
    /// Scale bytes per row, `None` when [`GROUP`] does not divide a row.
    #[must_use]
    pub fn groups_per_row(&self) -> Option<usize> {
        self.cols.is_multiple_of(GROUP).then_some(self.cols / GROUP)
    }
}

/// Decodes an MXFP8 matrix into `out`, row-major `[rows, cols]`.
///
/// `out` is cleared first and holds exactly `rows * cols` values on return.
///
/// # Errors
///
/// [`GgufError::LayoutMismatch`] when the row is not whole groups of 32, when
/// either array is not the length the layout implies, or when a scale byte is
/// the reserved NaN code.
pub fn dequant_mxfp8(
    tensor: &str,
    layout: Mxfp8Layout,
    weights: &[u8],
    scales: &[u8],
    out: &mut Vec<f32>,
) -> GgufResult<()> {
    let mismatch = |want: String, got: String| GgufError::LayoutMismatch {
        tensor: tensor.into(),
        want: want.into_boxed_str(),
        got: got.into_boxed_str(),
    };
    let groups = layout.groups_per_row().ok_or_else(|| {
        mismatch(
            "rows in whole groups of 32".into(),
            format!("{} weights per row", layout.cols),
        )
    })?;
    let elements = layout.rows.checked_mul(layout.cols);
    if elements != Some(weights.len()) {
        return Err(mismatch(
            format!("{} x {} weight bytes", layout.rows, layout.cols),
            format!("{} bytes", weights.len()),
        ));
    }
    if layout.rows.checked_mul(groups) != Some(scales.len()) {
        return Err(mismatch(
            format!("{} x {groups} scale bytes", layout.rows),
            format!("{} bytes", scales.len()),
        ));
    }
    out.clear();
    out.reserve(weights.len());
    for (group, &scale) in weights.chunks_exact(GROUP).zip(scales) {
        let scale = ue8m0(scale).ok_or_else(|| {
            mismatch(
                "finite ue8m0 scales".into(),
                "the reserved NaN scale 0xFF".into(),
            )
        })?;
        out.extend(group.iter().map(|&code| e4m3(code) * scale));
    }
    Ok(())
}

#[cfg(test)]
#[path = "mxfp8_tests.rs"]
mod tests;
