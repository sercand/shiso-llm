//! NVFP4: four-bit floats under two levels of scale, decoded to `f32`.
//!
//! The weight format `modelopt` writes for `W4A16_NVFP4` targets, and the first
//! codec in this tree whose values are floats rather than integers. A matrix is
//! three arrays that must be read together:
//!
//! - `weight`, a row-major `u8` array whose bytes each pack **two** consecutive
//!   weights of one row, the lower-numbered weight in the *low* nibble;
//! - `weight_scale`, a row-major `u8` array of `e4m3` bytes, one per group of
//!   [`GROUP`] consecutive weights along the same row;
//! - `weight_scale_2`, a single `f32` for the whole tensor.
//!
//! Decoding one weight is therefore
//!
//! ```text
//! out_ij = e2m1(nibble) * (e4m3(scale[i][j / GROUP]) * global)
//! ```
//!
//! # Three levels, not two, and the third is the one that is easy to miss
//!
//! A description of NVFP4 as "E2M1 values under per-group E4M3 scales" is one
//! factor short. The E4M3 byte is not the weight's scale: it is the weight's
//! scale *relative to a per-tensor `f32`*, and the quantizer picks the pair so
//! that the tensor's largest magnitude lands on the format's own ceiling. That
//! is checkable rather than assumed, and it was checked against this
//! checkpoint's own bytes before a line of this module was written: for
//! `model.language_model.layers.0.mlp.experts.0.gate_proj`, the largest
//! magnitude over the decoded tensor is 0.08544922 and
//! `6 * 448 * weight_scale_2` is 0.085449215 - the product of `e2m1`'s maximum
//! and `e4m3`'s, to the last digit an `f32` carries. Dropping the third level
//! would scale every weight of that tensor by about thirty-one thousand, and
//! **dropping it silently is the interesting case**: a matmul over uniformly
//! rescaled weights produces a residual with the same shape and the wrong
//! magnitude, which the next normalization very nearly removes.
//!
//! # The `e4m3` decode is [`crate::fp8`]'s, and deliberately not a second copy
//!
//! This checkpoint uses `e4m3` for two unrelated things: an FP8 target's
//! weights, and an NVFP4 target's per-group scales. The bytes mean different
//! quantities and are read by different stages, but they are one format, and
//! two implementations of one format are two places for a subnormal boundary
//! to be off by one. So the law lives in [`crate::fp8`] and this module calls
//! it, and both are graded on all 256 codes against the same reference.
//!
//! # What is a convention here and what is arithmetic
//!
//! Everything above except one thing is forced. The `e2m1` value table is the
//! format's - `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` by the three magnitude bits, with
//! bit three the sign - and `e4m3` is IEEE-shaped with a bias of 7 and no
//! infinities. The one free choice is **which nibble of a byte is the earlier
//! weight**, and the answer is the low one. That is not derivable from this
//! checkpoint: a reversed nibble order permutes weights within a byte, both
//! members of every pair stay inside the same group of sixteen, so every group
//! scale, every magnitude histogram and every column norm is unchanged. It is
//! taken from the reference this box carries - `vllm`'s
//! `vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`,
//! whose dequantizer reads `raw_bytes & 0x0F` and `raw_bytes >> 4` and then
//! interleaves them low-first - and it is pinned here by a hand-written byte,
//! because a test over real weights cannot see it.
//!
//! # The decode is exact, so its unit cases compare on bits
//!
//! There is no tolerance in this module and that is a measurement rather than
//! an omission. Every step is a single IEEE `f32` operation: the two scales
//! multiply once, the code multiplies once, and nothing is fused or widened.
//! The cases below therefore compare against the reference **bit for bit**,
//! and the tolerance question belongs where the arithmetic stops being exact -
//! a device kernel that folds the scale into an accumulator, which is a
//! different measurement on different data.

use crate::fp8::e4m3;
use crate::reader::{GgufError, GgufResult};

/// Weights sharing one `e4m3` scale byte.
pub const GROUP: usize = 16;

/// Weights packed into one storage byte.
pub const PER_BYTE: usize = 2;

/// Largest magnitude an `e2m1` code can carry.
///
/// Public because it is half of the relation that identifies the per-tensor
/// scale: a correctly quantized tensor's largest magnitude is
/// `E2M1_MAX * E4M3_MAX * weight_scale_2`.
pub const E2M1_MAX: f32 = 6.0;

/// Largest finite magnitude an `e4m3` scale byte can carry.
///
/// Re-exported from [`crate::fp8`] rather than restated: this checkpoint uses
/// `e4m3` for two unrelated things - an FP8 target's weights and an NVFP4
/// target's per-group scales - and two constants for one format's ceiling are
/// two places to edit it wrong.
pub use crate::fp8::E4M3_MAX;

/// Subject named by the layout errors raised from this module.
const SUBJECT: &str = "nvfp4 block";

/// The eight magnitudes the three low bits of an `e2m1` code select.
///
/// Written out rather than computed. The format's two exponent bits and one
/// mantissa bit give `{0, 0.5}` for the subnormal exponent and
/// `{1, 1.5, 2, 3, 4, 6}` above it, which is a rule with two special cases at
/// the bottom; a table of eight is the whole of it and cannot be off by one.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decodes one four-bit code, sign bit included.
///
/// A negative zero comes back for code `0x8`, which is what the format stores
/// and what every reference decoder produces. It compares equal to positive
/// zero and multiplies like it, so nothing downstream distinguishes them - but
/// a decoder that folded it to `+0.0` would stop being bit-comparable against
/// the reference, which is the only thing that grades this module.
#[must_use]
pub fn e2m1(code: u8) -> f32 {
    let magnitude = E2M1[(code & 0x07) as usize];
    if code & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// How a packed NVFP4 matrix is laid out.
///
/// `out_rows` and `in_dim` describe the *logical* matrix, not the packed one:
/// `in_dim` counts weights, so the packed row holds `in_dim / 2` bytes and the
/// scale row holds `in_dim / 16` bytes. They travel as one named struct rather
/// than as two same-typed arguments a caller can transpose.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Nvfp4Layout {
    /// Rows of the logical matrix, the slowest-varying axis.
    pub out_rows: usize,
    /// Weights per row of the logical matrix, the fastest-varying axis.
    pub in_dim: usize,
    /// The whole tensor's scale, which the group scales are relative to.
    pub global: f32,
}

impl Nvfp4Layout {
    /// Describes a packed matrix.
    ///
    /// The argument order matches the checkpoint's own shape field: outer axis,
    /// inner axis, then the quantization parameter.
    #[must_use]
    pub const fn new(out_rows: usize, in_dim: usize, global: f32) -> Self {
        Self {
            out_rows,
            in_dim,
            global,
        }
    }

    /// Logical weights, `None` when the product overflows.
    #[must_use]
    pub const fn elements(&self) -> Option<usize> {
        self.out_rows.checked_mul(self.in_dim)
    }

    /// Groups along one row, `None` when the group size does not divide it.
    #[must_use]
    pub fn groups_per_row(&self) -> Option<usize> {
        if self.in_dim.is_multiple_of(GROUP) {
            Some(self.in_dim / GROUP)
        } else {
            None
        }
    }
}

/// Decodes a packed NVFP4 matrix into `out`, row-major `[out_rows, in_dim]`.
///
/// `out` is cleared first and holds exactly `out_rows * in_dim` values when
/// this returns, so a caller can hand the same buffer to every tensor of a
/// checkpoint without reallocating per tensor.
///
/// `scales` is the raw `e4m3` byte array as the container stores it, not
/// widened floats: widening it outside this module would make the caller
/// responsible for a decode that has two special cases, and the point of this
/// module is that nobody else has to know them.
///
/// # Errors
///
/// [`GgufError::LayoutMismatch`] when [`GROUP`] does not divide `in_dim`, or
/// when either array is not exactly the length the layout implies. Both are
/// checked before a byte is read, so a short array is a load error rather than
/// a matrix that is correct down to the row it ran out at.
pub fn dequant_nvfp4(
    layout: Nvfp4Layout,
    weights: &[u8],
    scales: &[u8],
    out: &mut Vec<f32>,
) -> GgufResult<()> {
    let groups_per_row = layout.groups_per_row().ok_or_else(|| {
        layout_error(
            "an inner dimension in whole groups of 16",
            format!("{} weights per row", layout.in_dim),
        )
    })?;
    let bytes_per_row = layout.in_dim / PER_BYTE;
    let want_weights = layout
        .out_rows
        .checked_mul(bytes_per_row)
        .ok_or_else(|| layout_error("a matrix that fits in memory", "an overflowing row count"))?;
    let want_scales = layout
        .out_rows
        .checked_mul(groups_per_row)
        .ok_or_else(|| layout_error("a matrix that fits in memory", "an overflowing row count"))?;
    if weights.len() != want_weights {
        return Err(layout_error(
            format!("{want_weights} packed bytes"),
            format!("{} bytes", weights.len()),
        ));
    }
    if scales.len() != want_scales {
        return Err(layout_error(
            format!("{want_scales} scale bytes"),
            format!("{} bytes", scales.len()),
        ));
    }

    let elements = layout.elements().ok_or_else(|| {
        layout_error(
            "a matrix that fits in memory",
            "an overflowing element count",
        )
    })?;
    out.clear();
    out.reserve(elements);
    for row in 0..layout.out_rows {
        let packed = &weights[row * bytes_per_row..(row + 1) * bytes_per_row];
        let row_scales = &scales[row * groups_per_row..(row + 1) * groups_per_row];
        for (group, scale) in row_scales.iter().enumerate() {
            // The two scales multiply ONCE, before the code, and the product is
            // reused across the group. Folding the global into each weight
            // instead is a second rounding per weight and stops this agreeing
            // with the reference on the last bit.
            let scale = e4m3(*scale) * layout.global;
            let bytes = &packed[group * (GROUP / PER_BYTE)..(group + 1) * (GROUP / PER_BYTE)];
            for byte in bytes {
                out.push(e2m1(byte & 0x0F) * scale);
                out.push(e2m1(byte >> 4) * scale);
            }
        }
    }
    Ok(())
}

/// One layout rejection, named for this format.
fn layout_error(want: impl Into<String>, got: impl Into<String>) -> GgufError {
    GgufError::LayoutMismatch {
        tensor: Box::from(SUBJECT),
        want: want.into().into_boxed_str(),
        got: got.into().into_boxed_str(),
    }
}

#[cfg(test)]
#[expect(
    clippy::float_cmp,
    reason = "this decode is exact, which is the property these cases exist to \
              hold: every value is a table entry or a single IEEE product, so a \
              comparison with a tolerance would stop catching the one-bit \
              divergences a fused or reordered decode introduces"
)]
mod tests {
    use super::*;

    include!("nvfp4_fixtures.rs");

    /// Bit-for-bit, because this decode is exact: see the module doc.
    fn assert_bits(got: &[f32], want: &[u32], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: element count");
        for (i, (value, bits)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                value.to_bits(),
                *bits,
                "{label}: element {i} decoded to {value} (0x{:08x}), reference 0x{bits:08x}",
                value.to_bits(),
            );
        }
    }

    #[test]
    fn the_code_table_is_the_formats_eight_magnitudes_and_their_negatives() {
        let want = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, value) in want.iter().enumerate() {
            let code = u8::try_from(code).expect("eight codes");
            assert_eq!(e2m1(code), *value, "code {code:#x}");
            assert_eq!(e2m1(code | 0x08), -*value, "code {:#x}", code | 0x08);
        }
        // The sign of zero survives, which is what keeps this bit-comparable
        // against the reference decoders.
        assert!(e2m1(0x08).is_sign_negative());
    }

    #[test]
    fn the_low_nibble_of_a_byte_is_the_earlier_weight() {
        // The ONE convention in this format that the checkpoint cannot settle:
        // reversing the nibbles permutes weights inside a byte and inside a
        // group, so every scale, every histogram and every column norm is
        // unchanged. So it is pinned on a hand-written byte instead.
        //
        // 0x21 is code 1 (=0.5) in the low nibble and code 2 (=1.0) in the
        // high one, under a scale of exactly one.
        let mut out = Vec::new();
        dequant_nvfp4(
            Nvfp4Layout::new(1, 16, 1.0),
            &[0x21, 0, 0, 0, 0, 0, 0, 0],
            &[0x38],
            &mut out,
        )
        .expect("a one-group row");
        assert_eq!(
            e4m3(0x38),
            1.0,
            "0x38 is one, so the scale is out of the way"
        );
        assert_eq!(out[0], 0.5, "the low nibble is weight zero");
        assert_eq!(out[1], 1.0, "the high nibble is weight one");
    }

    #[test]
    fn the_scale_byte_decodes_at_both_ends_of_its_range() {
        // The two special cases: the subnormal floor this checkpoint really
        // uses, and the finite ceiling that is half of the relation
        // identifying the per-tensor scale.
        assert_eq!(e4m3(0x00), 0.0);
        assert_eq!(e4m3(0x01), 2.0_f32.powi(-9), "the subnormal floor");
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0x7e), E4M3_MAX, "the finite ceiling");
        assert_eq!(e4m3(0xb8), -1.0);
        assert!(e4m3(0x7f).is_nan(), "there is no infinity in this format");
    }

    #[test]
    fn a_tensors_largest_magnitude_is_the_two_ceilings_times_its_global_scale() {
        // The relation that identifies the third level, checked on the
        // checkpoint's own numbers rather than asserted. `experts.0.gate_proj`
        // reaches its ceiling within the four groups this fixture carries.
        let mut out = Vec::new();
        dequant_nvfp4(
            Nvfp4Layout::new(1, 64, EXPERT_G2),
            &EXPERT_BYTES,
            &EXPERT_SCALES,
            &mut out,
        )
        .expect("a four-group row");
        let ceiling = E2M1_MAX * E4M3_MAX * EXPERT_G2;
        let amax = out.iter().fold(0.0_f32, |acc, v| acc.max(v.abs()));
        assert!(
            amax <= ceiling,
            "no weight may pass the format's ceiling: {amax} > {ceiling}",
        );
    }

    #[test]
    fn the_lm_head_rows_decode_to_the_reference_bit_for_bit() {
        let mut out = Vec::new();
        dequant_nvfp4(
            Nvfp4Layout::new(2, 48, LM_HEAD_G2),
            &LM_HEAD_BYTES,
            &LM_HEAD_SCALES,
            &mut out,
        )
        .expect("two three-group rows");
        assert_bits(&out, &LM_HEAD_EXPECT, "lm_head");
    }

    #[test]
    fn a_routed_expert_row_decodes_to_the_reference_bit_for_bit() {
        let mut out = Vec::new();
        dequant_nvfp4(
            Nvfp4Layout::new(1, 64, EXPERT_G2),
            &EXPERT_BYTES,
            &EXPERT_SCALES,
            &mut out,
        )
        .expect("a four-group row");
        assert_bits(&out, &EXPERT_EXPECT, "experts.0.gate_proj");
    }

    #[test]
    fn a_shared_expert_down_projection_decodes_to_the_reference_bit_for_bit() {
        // A third tensor because the down projection is the one whose group
        // count per row differs from the other two - 512 inputs in 32 groups
        // against 2048 in 128 - and a stride taken from the wrong axis would
        // pass on a square-ish fixture.
        let mut out = Vec::new();
        dequant_nvfp4(
            Nvfp4Layout::new(1, 64, SHARED_DOWN_G2),
            &SHARED_DOWN_BYTES,
            &SHARED_DOWN_SCALES,
            &mut out,
        )
        .expect("a four-group row");
        assert_bits(&out, &SHARED_DOWN_EXPECT, "shared_expert.down_proj");
    }

    #[test]
    fn the_second_row_reads_its_own_scales_and_not_the_first_rows() {
        // Two rows of the same bytes under different scales. A decoder that
        // strided the scale array by the byte count rather than by the group
        // count would produce a matrix that is right in its first row, wrong
        // in every later one, and plausible throughout.
        let bytes = [0x21_u8, 0, 0, 0, 0, 0, 0, 0, 0x21, 0, 0, 0, 0, 0, 0, 0];
        let mut out = Vec::new();
        dequant_nvfp4(
            Nvfp4Layout::new(2, 16, 1.0),
            &bytes,
            &[0x38, 0x40],
            &mut out,
        )
        .expect("two one-group rows");
        assert_eq!(out[0], 0.5, "row zero at scale one");
        assert_eq!(out[16], 1.0, "row one at scale two");
    }

    #[test]
    fn a_row_that_is_not_a_whole_number_of_groups_is_refused_before_any_read() {
        let mut out = Vec::new();
        let err = dequant_nvfp4(Nvfp4Layout::new(1, 24, 1.0), &[0; 12], &[0; 1], &mut out)
            .expect_err("24 is not a multiple of 16");
        assert!(matches!(err, GgufError::LayoutMismatch { .. }), "{err:?}");
        assert!(out.is_empty(), "nothing is written before the check");
    }

    #[test]
    fn a_short_array_is_a_load_error_rather_than_a_truncated_matrix() {
        let mut out = Vec::new();
        for (weights, scales) in [(&[0_u8; 7][..], &[0_u8; 1][..]), (&[0_u8; 8][..], &[][..])] {
            let err = dequant_nvfp4(Nvfp4Layout::new(1, 16, 1.0), weights, scales, &mut out)
                .expect_err("one group is eight bytes under one scale");
            assert!(matches!(err, GgufError::LayoutMismatch { .. }), "{err:?}");
        }
    }

    #[test]
    fn the_buffer_is_replaced_rather_than_appended_to() {
        let mut out = vec![7.0; 3];
        dequant_nvfp4(Nvfp4Layout::new(1, 16, 1.0), &[0; 8], &[0x38], &mut out)
            .expect("a one-group row");
        assert_eq!(
            out.len(),
            16,
            "one tensor's worth, not the previous one's too"
        );
    }
}
