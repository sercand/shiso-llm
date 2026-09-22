//! The NVFP4 codec, graded bit for bit against the checkpoint's own bytes.

#![expect(
    clippy::float_cmp,
    reason = "this decode is exact, which is the property these cases exist to \
              hold: every value is a table entry or a single IEEE product, so a \
              comparison with a tolerance would stop catching the one-bit \
              divergences a fused or reordered decode introduces"
)]

use super::*;

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

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
//
// Vectors taken from the checkpoint's own bytes, and the reference decode of
// them, for the cases above.
//
// Generated once on 2026-09-20 from
// `Qwen3.6-35B-A3B-NVFP4` snapshot
// `9660379a2f2c429c465eeed2f3a0f2433fc4381e`, whose quantizer is `modelopt`
// 0.45.0. The expected values are `torch` 2.14's own `float8_e4m3fn` decode of
// the scale bytes, times the format's code table, in the operation order this
// module's decoder uses - so they are an independent decode of the same bytes,
// not a transcript of this module's output.
//
// Each set is a corner of one real tensor: the packed bytes, the `e4m3` scale
// bytes, the per-tensor `f32` as its bit pattern, and the reference values as
// bit patterns, because the comparison is exact.

// LM_HEAD: rows 0..2 of `lm_head`, first 3 groups
const LM_HEAD_G2: f32 = f32::from_bits(0x38e9_2493); // 0.00011117118265246972
const LM_HEAD_BYTES: [u8; 48] = [
    0x8a, 0xa0, 0x50, 0x3c, 0xef, 0xa5, 0x19, 0x35, 0x2e, 0x3a, 0xa9, 0xe0, 0x31, 0x6e, 0xc9, 0xf8,
    0x62, 0xd6, 0xaf, 0x66, 0x38, 0xec, 0xdb, 0x7b, 0x1b, 0xc5, 0x5c, 0xc3, 0xc2, 0xc7, 0x6a, 0x51,
    0x71, 0x29, 0x69, 0x90, 0xbc, 0xd1, 0x12, 0xcb, 0xba, 0x55, 0x3c, 0x3e, 0x2c, 0xfb, 0x6d, 0x4a,
];
const LM_HEAD_SCALES: [u8; 6] = [0x68, 0x65, 0x68, 0x6c, 0x6b, 0x68];
const LM_HEAD_EXPECT: [u32; 96] = [
    0xbbe9_2493,
    0x8000_0000,
    0x0000_0000,
    0xbbe9_2493,
    0x0000_0000,
    0x3cae_db6e,
    0xbc69_2493,
    0x3c2e_db6e,
    0xbd2e_db6e,
    0xbce9_2493,
    0x3cae_db6e,
    0xbbe9_2493,
    0xbb69_2493,
    0x3b69_2493,
    0x3cae_db6e,
    0x3c2e_db6e,
    0xbcbd_6db7,
    0x3bbd_6db7,
    0xbbbd_6db7,
    0x3c0e_1249,
    0xbb3d_6db7,
    0xbbbd_6db7,
    0x0000_0000,
    0xbcbd_6db7,
    0x3b3d_6db7,
    0x3c0e_1249,
    0xbcbd_6db7,
    0x3cbd_6db7,
    0xbb3d_6db7,
    0xbc3d_6db7,
    0x8000_0000,
    0xbd0e_1249,
    0x3be9_2493,
    0x3ce9_2493,
    0x3ce9_2493,
    0xbcae_db6e,
    0xbd2e_db6e,
    0xbbe9_2493,
    0x3ce9_2493,
    0x3ce9_2493,
    0x8000_0000,
    0x3c2e_db6e,
    0xbc69_2493,
    0xbce9_2493,
    0xbc2e_db6e,
    0xbcae_db6e,
    0xbc2e_db6e,
    0x3d2e_db6e,
    0xbc83_2492,
    0x3bae_db6e,
    0x3d03_2492,
    0xbcae_db6e,
    0xbcae_db6e,
    0x3d03_2492,
    0x3c83_2492,
    0xbcae_db6e,
    0x3c2e_db6e,
    0xbcae_db6e,
    0x3d83_2492,
    0xbcae_db6e,
    0xbc2e_db6e,
    0x3d2e_db6e,
    0x3bae_db6e,
    0x3d03_2492,
    0x3ba0_4925,
    0x3d70_6db8,
    0xbba0_4925,
    0x3c20_4925,
    0xbba0_4925,
    0x3d20_4925,
    0x0000_0000,
    0xbba0_4925,
    0xbca0_4925,
    0xbc70_6db8,
    0x3ba0_4925,
    0xbcf0_6db8,
    0x3c20_4925,
    0x3ba0_4925,
    0xbc70_6db8,
    0xbca0_4925,
    0xbbe9_2493,
    0xbc2e_db6e,
    0x3cae_db6e,
    0x3cae_db6e,
    0xbc69_2493,
    0x3c2e_db6e,
    0xbce9_2493,
    0x3c2e_db6e,
    0xbc69_2493,
    0x3be9_2493,
    0xbc2e_db6e,
    0xbd2e_db6e,
    0xbcae_db6e,
    0x3ce9_2493,
    0xbbe9_2493,
    0x3c69_2493,
];
//   nonzero 89 of 96, amax 0.0640345961

// EXPERT: rows 0..1 of `model.language_model.layers.0.mlp.experts.0.gate_proj`, first 4 groups
const EXPERT_G2: f32 = f32::from_bits(0x3805_5555); // 3.178914266754873e-05
const EXPERT_BYTES: [u8; 32] = [
    0x52, 0x47, 0x3a, 0x3c, 0x2a, 0xb1, 0x43, 0x90, 0x83, 0x71, 0xe4, 0xbd, 0x61, 0xbe, 0xa7, 0xde,
    0xc6, 0x6f, 0x49, 0x5e, 0x78, 0x1a, 0xd5, 0xbd, 0x1d, 0xbe, 0xfa, 0x31, 0x47, 0xa1, 0x49, 0xc8,
];
const EXPERT_SCALES: [u8; 4] = [0x71, 0x73, 0x6c, 0x71];
const EXPERT_EXPECT: [u32; 64] = [
    0x3b96_0000,
    0x3c61_0000,
    0x3ce1_0000,
    0x3c16_0000,
    0xbb96_0000,
    0x3be1_0000,
    0xbc16_0000,
    0x3be1_0000,
    0xbb96_0000,
    0x3b96_0000,
    0x3b16_0000,
    0xbbe1_0000,
    0x3be1_0000,
    0x3c16_0000,
    0x0000_0000,
    0xbb16_0000,
    0x3c09_8000,
    0x8000_0000,
    0x3b37_5555,
    0x3d09_8000,
    0x3c37_5555,
    0xbcb7_5555,
    0xbc89_8000,
    0xbc09_8000,
    0x3b37_5555,
    0x3cb7_5555,
    0xbcb7_5555,
    0xbc09_8000,
    0x3d09_8000,
    0xbbb7_5555,
    0xbcb7_5555,
    0xbc89_8000,
    0x3c48_0000,
    0xbbc8_0000,
    0xbc96_0000,
    0x3c48_0000,
    0xbac8_0000,
    0x3bc8_0000,
    0xbc48_0000,
    0x3c16_0000,
    0x8000_0000,
    0x3c96_0000,
    0xbb48_0000,
    0x3ac8_0000,
    0x3c16_0000,
    0xbc16_0000,
    0xbc16_0000,
    0xbb96_0000,
    0xbc61_0000,
    0x3b16_0000,
    0xbc96_0000,
    0xbbe1_0000,
    0xbb96_0000,
    0xbce1_0000,
    0x3b16_0000,
    0x3be1_0000,
    0x3ce1_0000,
    0x3c16_0000,
    0x3b16_0000,
    0xbb96_0000,
    0xbb16_0000,
    0x3c16_0000,
    0x8000_0000,
    0xbc16_0000,
];
//   nonzero 60 of 64, amax 0.0335693359

// SHARED_DOWN: rows 0..1 of `model.language_model.layers.0.mlp.shared_expert.down_proj`, first 4 groups
const SHARED_DOWN_G2: f32 = f32::from_bits(0x38f3_cf3d); // 0.00011625744082266465
const SHARED_DOWN_BYTES: [u8; 32] = [
    0xee, 0xa3, 0x58, 0x4d, 0x89, 0x44, 0x44, 0xfd, 0xce, 0x44, 0x60, 0x49, 0xae, 0x5f, 0x7b, 0x29,
    0xf9, 0xba, 0xdb, 0xc7, 0xbe, 0x0c, 0x19, 0xf9, 0xf2, 0xe1, 0x2a, 0xf4, 0xd5, 0xdf, 0x66, 0xf9,
];
const SHARED_DOWN_SCALES: [u8; 4] = [0x61, 0x60, 0x61, 0x5c];
const SHARED_DOWN_EXPECT: [u32; 64] = [
    0xbc89_2492,
    0xbc89_2492,
    0x3bcd_b6db,
    0xbb89_2492,
    0x8000_0000,
    0x3c4d_b6db,
    0xbc4d_b6db,
    0x3c09_2492,
    0xbb09_2492,
    0x8000_0000,
    0x3c09_2492,
    0x3c09_2492,
    0x3c09_2492,
    0x3c09_2492,
    0xbc4d_b6db,
    0xbccd_b6db,
    0xbc73_cf3d,
    0xbbf3_cf3d,
    0x3bf3_cf3d,
    0x3bf3_cf3d,
    0x0000_0000,
    0x3c73_cf3d,
    0xbaf3_cf3d,
    0x3bf3_cf3d,
    0xbc73_cf3d,
    0xbb73_cf3d,
    0xbcb6_db6e,
    0x3c36_db6e,
    0xbbb6_db6e,
    0x3cb6_db6e,
    0xbaf3_cf3d,
    0x3b73_cf3d,
    0xbb09_2492,
    0xbccd_b6db,
    0xbb89_2492,
    0xbbcd_b6db,
    0xbbcd_b6db,
    0xbc4d_b6db,
    0x3ccd_b6db,
    0xbc09_2492,
    0xbc89_2492,
    0xbbcd_b6db,
    0xbc09_2492,
    0x0000_0000,
    0xbb09_2492,
    0x3b09_2492,
    0xbb09_2492,
    0xbccd_b6db,
    0x3b36_db6e,
    0xbc89_2492,
    0x3ab6_db6e,
    0xbc36_db6e,
    0xbb36_db6e,
    0x3b36_db6e,
    0x3bb6_db6e,
    0xbc89_2492,
    0x3c09_2492,
    0xbc09_2492,
    0xbc89_2492,
    0xbc09_2492,
    0x3c36_db6e,
    0x3c36_db6e,
    0xbab6_db6e,
    0xbc89_2492,
];
//   nonzero 60 of 64, amax 0.0251116063
