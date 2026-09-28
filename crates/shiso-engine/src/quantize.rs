//! Affine quantization at load, into the packed format the affine matmuls read.
//!
//! A checkpoint that ships a projection in `bfloat16` - a draft model, say -
//! would otherwise run it through a dense `f32` matmul that reads four times the
//! bytes. Quantizing it once at load to 4-bit affine in groups of 64 puts it on
//! the same row-exact matmul as every packed weight, at an eighth of the bytes.
//!
//! The rule is the one the MLX-format exports were written with, so a weight
//! quantized here and the same weight quantized by that export's converter agree
//! value for value up to the converter's own float rounding. Per group of 64:
//!
//! - `w_max = max(0, max w)` and `w_min = min w` - the maximum starts at zero,
//!   not at minus infinity, as the converter's does;
//! - `scale = max((w_max - w_min) / (2^bits - 1), 1e-7)`, negated when the
//!   group's larger magnitude is its maximum;
//! - the edge (the extreme of larger magnitude) is snapped to a whole number of
//!   scales, `scale = edge / round(edge / scale)`, and becomes the bias; an edge
//!   that rounds to zero steps leaves the scale alone and a zero bias;
//! - `q = min(round((w - bias) / scale), 2^bits - 1)`, packed low bits first;
//! - scale and bias are stored as `bfloat16`, while `q` was computed against
//!   the unrounded ones.

use half::bf16;
use shiso_core::error::{CoreError, CoreResult};

/// Weights sharing one scale and one bias.
pub const GROUP: usize = 64;

/// A packed affine weight: the three arrays [`crate::projection::Weight::import_affine_slices`]
/// takes, little-endian.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffinePacked {
    /// Packed words, `rows * cols * bits / 32` of them.
    pub words: Vec<u8>,
    /// One `bfloat16` scale per group.
    pub scales: Vec<u8>,
    /// One `bfloat16` bias per group.
    pub biases: Vec<u8>,
}

/// Quantizes a row-major `[rows, cols]` matrix to `bits`-bit affine, groups of
/// [`GROUP`] along each row.
///
/// # Errors
///
/// [`CoreError::InvalidValue`] for a bit width other than 4 or 8, and
/// [`CoreError::ShapeMismatch`] when `cols` is not a whole number of groups or
/// `values` is not `rows * cols` long.
pub fn quantize_affine(
    values: &[f32],
    rows: usize,
    cols: usize,
    bits: u32,
) -> CoreResult<AffinePacked> {
    if !matches!(bits, 4 | 8) {
        return Err(CoreError::InvalidValue {
            key: "affine bits",
            value: format!("{bits}, where only 4 and 8 have a matmul"),
        });
    }
    if !cols.is_multiple_of(GROUP) {
        return Err(CoreError::ShapeMismatch {
            field: "affine columns",
            expected: u64::try_from(cols.div_ceil(GROUP) * GROUP).unwrap_or(u64::MAX),
            got: u64::try_from(cols).unwrap_or(u64::MAX),
        });
    }
    let want = rows.saturating_mul(cols);
    if values.len() != want {
        return Err(CoreError::ShapeMismatch {
            field: "affine values",
            expected: u64::try_from(want).unwrap_or(u64::MAX),
            got: u64::try_from(values.len()).unwrap_or(u64::MAX),
        });
    }
    let n_bins = f32::from(u8::try_from((1u32 << bits) - 1).unwrap_or(u8::MAX));
    let per_word = 32 / bits as usize;
    let groups = values.len() / GROUP;
    let mut words = Vec::with_capacity(values.len() / per_word * 4);
    let mut scales = Vec::with_capacity(groups * 2);
    let mut biases = Vec::with_capacity(groups * 2);
    for group in values.chunks_exact(GROUP) {
        let w_min = group.iter().copied().fold(f32::MAX, f32::min);
        let w_max = group.iter().copied().fold(0.0f32, f32::max);
        let mut scale = ((w_max - w_min) / n_bins).max(1e-7);
        let side = w_min.abs() > w_max.abs();
        if !side {
            scale = -scale;
        }
        let edge = if side { w_min } else { w_max };
        let q0 = (edge / scale).round();
        let at_zero = q0 == 0.0;
        if !at_zero {
            scale = edge / q0;
        }
        let offset = if at_zero { 0.0 } else { edge };
        scales.extend_from_slice(&bf16::from_f32(scale).to_bits().to_le_bytes());
        biases.extend_from_slice(&bf16::from_f32(offset).to_bits().to_le_bytes());
        for packed in group.chunks_exact(per_word) {
            let mut word = 0u32;
            for (i, &w) in packed.iter().enumerate() {
                // The quotient is non-negative and at most `n_bins` after the
                // clamp, so the narrowing below loses nothing.
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a clamped, rounded, non-negative quantization level"
                )]
                let q = ((w - offset) / scale).round().clamp(0.0, n_bins) as u32;
                word |= q << (bits as usize * i);
            }
            words.extend_from_slice(&word.to_le_bytes());
        }
    }
    Ok(AffinePacked {
        words,
        scales,
        biases,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dequant(p: &AffinePacked, cols: usize, bits: u32) -> Vec<f32> {
        let per_word = 32 / bits as usize;
        let mask = (1u32 << bits) - 1;
        let words: Vec<u32> = p
            .words
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let scale_at = |i: usize| {
            f32::from(bf16::from_bits(u16::from_le_bytes([
                p.scales[2 * i],
                p.scales[2 * i + 1],
            ])))
        };
        let offset_at = |i: usize| {
            f32::from(bf16::from_bits(u16::from_le_bytes([
                p.biases[2 * i],
                p.biases[2 * i + 1],
            ])))
        };
        let _ = cols;
        (0..words.len() * per_word)
            .map(|k| {
                let q = (words[k / per_word] >> (bits as usize * (k % per_word))) & mask;
                #[expect(clippy::cast_precision_loss, reason = "a quantization level")]
                let q = q as f32;
                q * scale_at(k / GROUP) + offset_at(k / GROUP)
            })
            .collect()
    }

    #[test]
    fn a_round_trip_is_within_one_step() {
        #[expect(clippy::cast_precision_loss, reason = "a small index")]
        let values: Vec<f32> = (0..4 * 128)
            .map(|i| ((i as f32) * 0.37).sin() * 0.05)
            .collect();
        for bits in [4, 8] {
            let packed = quantize_affine(&values, 4, 128, bits).expect("quantizes");
            let back = dequant(&packed, 128, bits);
            for (group, (v, b)) in values.chunks(GROUP).zip(back.chunks(GROUP)).enumerate() {
                // The step is the stored scale: snapping the edge to a whole
                // number of steps can make it wider than (max - min) / bins.
                let step = f32::from(bf16::from_bits(u16::from_le_bytes([
                    packed.scales[2 * group],
                    packed.scales[2 * group + 1],
                ])))
                .abs();
                for (x, y) in v.iter().zip(b) {
                    // One step: the snapped edge can leave the far end of the
                    // group up to a step outside the representable range,
                    // where the clamp holds it - as the converter's rule does.
                    // Plus the bfloat16 rounding of scale and bias.
                    assert!(
                        (x - y).abs() <= step + (x.abs() + step) * 1e-2,
                        "group {group} at {bits} bits: {x} came back as {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn an_all_positive_group_keeps_its_edge_exactly() {
        let values = vec![0.25f32; GROUP];
        let packed = quantize_affine(&values, 1, GROUP, 4).expect("quantizes");
        let back = dequant(&packed, GROUP, 4);
        assert!(
            back.iter().all(|&v| v.to_bits() == 0.25f32.to_bits()),
            "{back:?}"
        );
    }

    #[test]
    fn a_width_that_is_not_whole_groups_is_refused() {
        assert!(quantize_affine(&[0.0; 96], 1, 96, 4).is_err());
        assert!(quantize_affine(&[0.0; 64], 1, 64, 3).is_err());
    }
}
