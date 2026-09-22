//! Pins the affine group-quantization decode, with no model on disk.
//!
//! Every number here is hand-written, and that is the point. The decode has
//! three properties that a plausible-looking wrong implementation also has -
//! it produces a matrix of the right shape, in the right range, with the right
//! statistics - so the only way to catch a reversed field order or an off-by-one
//! group boundary is to write down the answer in advance.

use half::f16;
use shiso_modelfile::{GgufError, MlxAffineLayout, dequant_affine};

/// `1.0` as a little-endian `bf16`.
const ONE: [u8; 2] = [0x80, 0x3F];
/// `2.0` as a little-endian `bf16`.
const TWO: [u8; 2] = [0x00, 0x40];
/// `0.0` as a little-endian `bf16`.
const ZERO: [u8; 2] = [0x00, 0x00];
/// `2.5` as a little-endian `bf16`.
const TWO_AND_A_HALF: [u8; 2] = [0x20, 0x40];

/// The word `0x89AB_CDEF`, little endian, which every field-order test uses.
const WORD: [u8; 4] = [0xEF, 0xCD, 0xAB, 0x89];

/// Decodes one matrix, failing the test with the loader's own message.
fn decode(w: &[u8], scales: &[u8], biases: &[u8], layout: MlxAffineLayout) -> Vec<f32> {
    let mut out = Vec::new();
    dequant_affine(w, scales, biases, layout, &mut out).expect("the layout is a valid one");
    out.iter().copied().map(f16::to_f32).collect()
}

/// Flattens a list of two-byte constants into one array.
fn bytes(parts: &[[u8; 2]]) -> Vec<u8> {
    parts.iter().flatten().copied().collect()
}

#[test]
fn four_bit_fields_are_packed_low_bits_first() {
    // 0x89ABCDEF read as eight nibbles, lowest first, is F E D C B A 9 8.
    // Reading it the other way round would give 8 9 A B C D E F, which is just
    // as plausible a matrix and completely wrong.
    let values = decode(
        &WORD,
        &bytes(&[ONE]),
        &bytes(&[ZERO]),
        MlxAffineLayout::new(1, 8, 8, 4),
    );
    assert_eq!(values, vec![15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0]);
}

#[test]
fn eight_bit_fields_are_packed_low_bits_first() {
    // The same word as four bytes is the little-endian byte order it was
    // written in, which is the same statement one field width up.
    let values = decode(
        &WORD,
        &bytes(&[ONE]),
        &bytes(&[ZERO]),
        MlxAffineLayout::new(1, 4, 4, 8),
    );
    assert_eq!(values, vec![239.0, 205.0, 171.0, 137.0]);
}

#[test]
fn unit_scale_and_zero_bias_decode_to_exact_integers() {
    // Every field value up to 255 is exactly representable in f16, so a unit
    // scale must round-trip with no error at all - not "within a tolerance".
    let packed: Vec<u8> = (0_u8..=255).collect();
    let scales = bytes(&[ONE; 4]);
    let biases = bytes(&[ZERO; 4]);
    let values = decode(
        &packed,
        &scales,
        &biases,
        MlxAffineLayout::new(1, 256, 64, 8),
    );
    let want: Vec<f32> = (0..256)
        .map(|q| f32::from(u8::try_from(q).unwrap_or(0)))
        .collect();
    assert_eq!(values, want);
}

#[test]
fn a_zero_scale_leaves_only_the_bias() {
    // A zero scale is not a degenerate case to reject: it is how a constant
    // group is stored, and every weight in it must come out as the bias.
    let values = decode(
        &WORD,
        &bytes(&[ZERO]),
        &bytes(&[TWO_AND_A_HALF]),
        MlxAffineLayout::new(1, 8, 8, 4),
    );
    assert_eq!(values, vec![2.5; 8]);
}

#[test]
fn the_group_boundary_falls_between_the_right_two_weights() {
    // Two groups of four 8-bit weights. Weight 3 is the last one that uses the
    // first group's scale and bias; weight 4 is the first that uses the
    // second's. An off-by-one here is invisible in the output's range.
    let packed: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    let values = decode(
        &packed,
        &bytes(&[ONE, TWO]),
        &bytes(&[ZERO, ONE]),
        MlxAffineLayout::new(1, 8, 4, 8),
    );
    assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 9.0, 11.0, 13.0, 15.0]);
}

#[test]
fn rows_do_not_share_scales() {
    // Two rows of eight 4-bit weights, one group each. The second row's scale
    // must not reach the first row's weights.
    let packed: Vec<u8> = WORD.iter().chain(WORD.iter()).copied().collect();
    let values = decode(
        &packed,
        &bytes(&[ONE, TWO]),
        &bytes(&[ZERO, ZERO]),
        MlxAffineLayout::new(2, 8, 8, 4),
    );
    assert_eq!(
        &values[..8],
        &[15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0]
    );
    assert_eq!(
        &values[8..],
        &[30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0]
    );
}

#[test]
fn a_field_width_the_format_does_not_define_is_rejected() {
    let mut out = Vec::new();
    let err = dequant_affine(
        &WORD,
        &bytes(&[ONE]),
        &bytes(&[ZERO]),
        MlxAffineLayout::new(1, 8, 8, 3),
        &mut out,
    )
    .expect_err("three bits per weight is not a format");
    assert!(
        matches!(err, GgufError::LayoutMismatch { ref got, .. } if got.contains('3')),
        "expected a layout mismatch naming the width, got {err}"
    );
}

#[test]
fn a_group_size_that_does_not_divide_the_row_is_rejected() {
    let mut out = Vec::new();
    let err = dequant_affine(
        &WORD,
        &bytes(&[ONE]),
        &bytes(&[ZERO]),
        MlxAffineLayout::new(1, 8, 3, 4),
        &mut out,
    )
    .expect_err("three does not divide eight");
    assert!(matches!(err, GgufError::LayoutMismatch { .. }), "got {err}");
}

#[test]
fn a_group_smaller_than_a_packed_word_is_rejected() {
    // Four 4-bit weights per group would put two groups in one word, which the
    // format does not describe and the loop is not written for.
    let mut out = Vec::new();
    let err = dequant_affine(
        &WORD,
        &bytes(&[ONE, ONE]),
        &bytes(&[ZERO, ZERO]),
        MlxAffineLayout::new(1, 8, 4, 4),
        &mut out,
    )
    .expect_err("a group must be a whole number of words");
    assert!(matches!(err, GgufError::LayoutMismatch { .. }), "got {err}");
}

#[test]
fn a_short_array_is_a_load_error_rather_than_a_short_matrix() {
    let mut out = Vec::new();
    let err = dequant_affine(
        &WORD[..3],
        &bytes(&[ONE]),
        &bytes(&[ZERO]),
        MlxAffineLayout::new(1, 8, 8, 4),
        &mut out,
    )
    .expect_err("three bytes cannot hold eight nibbles");
    assert!(matches!(err, GgufError::LayoutMismatch { .. }), "got {err}");

    let err = dequant_affine(
        &WORD,
        &bytes(&[ONE]),
        &[],
        MlxAffineLayout::new(1, 8, 8, 4),
        &mut out,
    )
    .expect_err("a group needs a bias");
    assert!(matches!(err, GgufError::LayoutMismatch { .. }), "got {err}");
}

#[test]
fn the_output_buffer_is_reused_rather_than_appended_to() {
    // A streaming loader hands the same buffer to every tensor in a model; a
    // decode that appended would grow it without bound and shift every tensor
    // after the first.
    let mut out = vec![f16::from_f32(7.0); 3];
    let layout = MlxAffineLayout::new(1, 8, 8, 4);
    dequant_affine(&WORD, &bytes(&[ONE]), &bytes(&[ZERO]), layout, &mut out)
        .expect("the layout is a valid one");
    assert_eq!(out.len(), 8);
    dequant_affine(&WORD, &bytes(&[ONE]), &bytes(&[ZERO]), layout, &mut out)
        .expect("the layout is a valid one");
    assert_eq!(out.len(), 8);
}
