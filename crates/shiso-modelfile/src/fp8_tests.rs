//! The FP8 codec, graded bit for bit against `torch` 2.14.
//!
//! No tolerance anywhere in this file, and that is a measurement rather than an
//! omission: every value the decoder produces is a table entry times one IEEE
//! `f32`, with nothing fused and nothing widened, so a comparison with a
//! tolerance would stop catching the one-bit divergences a reordered decode
//! introduces. The tolerance question belongs to a device kernel that
//! accumulates in a tensor core, which is a different measurement on different
//! data.

#![expect(
    clippy::float_cmp,
    reason = "this decode is exact; see the module note above"
)]

use super::{E4M3_MAX, Fp8Layout, dequant_fp8, e4m3, quantize_activations, to_e4m3};
use crate::reader::GgufError;

/// Bit-for-bit, because this decode is exact.
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
fn the_two_nan_encodings_are_the_only_codes_that_are_not_finite() {
    let nans: Vec<u8> = (0..=u8::MAX).filter(|code| e4m3(*code).is_nan()).collect();
    assert_eq!(
        nans,
        vec![0x7f, 0xff],
        "there is no infinity in this format"
    );
}

#[test]
fn the_range_reaches_its_documented_floor_and_ceiling() {
    assert_eq!(e4m3(0x00), 0.0);
    assert_eq!(e4m3(0x01), 2.0_f32.powi(-9), "the subnormal floor");
    assert_eq!(e4m3(0x38), 1.0);
    assert_eq!(e4m3(0x7e), E4M3_MAX, "the finite ceiling");
    assert_eq!(e4m3(0xb8), -1.0);
    // The sign of zero survives, which is what keeps this bit-comparable
    // against the reference decoders.
    assert!(e4m3(0x80).is_sign_negative());
}

#[test]
fn the_recurrent_projections_decode_to_the_reference_bit_for_bit() {
    let mut out = Vec::new();
    dequant_fp8(
        Fp8Layout::weights_only(2, 16, QKV_SCALE),
        &QKV_BYTES,
        &mut out,
    )
    .expect("two sixteen-value rows");
    assert_bits(&out, &QKV_EXPECT, "linear_attn.in_proj_qkv");

    dequant_fp8(
        Fp8Layout::weights_only(2, 16, OUT_SCALE),
        &OUT_BYTES,
        &mut out,
    )
    .expect("two sixteen-value rows");
    assert_bits(&out, &OUT_EXPECT, "linear_attn.out_proj");
}

#[test]
fn an_attention_projection_decodes_to_the_reference_bit_for_bit() {
    // A third tensor from a different slot kind, because the two above are both
    // the recurrent block's and a scale read off the wrong target would pass on
    // either of them alone.
    let mut out = Vec::new();
    dequant_fp8(
        Fp8Layout::weights_only(2, 16, QPROJ_SCALE),
        &QPROJ_BYTES,
        &mut out,
    )
    .expect("two sixteen-value rows");
    assert_bits(&out, &QPROJ_EXPECT, "self_attn.q_proj");
}

#[test]
fn the_second_row_is_decoded_under_the_same_scale_as_the_first() {
    // Unlike NVFP4 there is one scale for the whole tensor, so the row stride
    // is the only thing a decoder can get wrong here - and it would produce a
    // matrix that is right in its first row and a permutation everywhere else.
    // The fixture's two rows are genuinely different bytes, which is what makes
    // the assertion above cover this at all; this pins the element count.
    let mut out = Vec::new();
    dequant_fp8(
        Fp8Layout::weights_only(2, 16, QKV_SCALE),
        &QKV_BYTES,
        &mut out,
    )
    .expect("two rows");
    assert_eq!(out.len(), 32);
    assert_ne!(
        &out[..16],
        &out[16..],
        "the fixture's two rows must differ or this case proves nothing"
    );
}

#[test]
fn a_short_array_is_a_load_error_rather_than_a_truncated_matrix() {
    let mut out = Vec::new();
    let err = dequant_fp8(Fp8Layout::weights_only(2, 16, 1.0), &[0; 31], &mut out)
        .expect_err("thirty-two bytes, not thirty-one");
    assert!(matches!(err, GgufError::LayoutMismatch { .. }), "{err:?}");
    assert!(out.is_empty(), "nothing is written before the check");
}

#[test]
fn the_buffer_is_replaced_rather_than_appended_to() {
    let mut out = vec![7.0; 3];
    dequant_fp8(Fp8Layout::weights_only(1, 4, 1.0), &[0; 4], &mut out).expect("one row");
    assert_eq!(
        out.len(),
        4,
        "one tensor's worth, not the previous one's too"
    );
}

#[test]
fn the_encode_round_trips_every_representable_magnitude() {
    // A code that decodes to `v` must encode back to itself. This is the
    // property that makes the activation quantizer and the weight decoder two
    // halves of one format rather than two functions that agree approximately.
    for code in 0..=u8::MAX {
        let value = e4m3(code);
        if value.is_nan() {
            continue;
        }
        let back = to_e4m3(value);
        assert_eq!(
            e4m3(back),
            value,
            "code {code:#04x} decoded to {value} and encoded back to {back:#04x}",
        );
    }
}

#[test]
fn a_tie_between_two_codes_goes_to_the_even_one() {
    // The one decision in the encode, and the one that has to match the
    // device's own conversion instruction: three mantissa bits leave exact
    // midpoints between neighbours, and a host oracle that rounded them away
    // from the device would disagree on a bit nobody could attribute.
    //
    // 1.0 is `0x38` and the next code up is `0x39` = 1.125, so 1.0625 is the
    // midpoint and rounds to the even code, which is `0x38`.
    assert_eq!(e4m3(0x38), 1.0);
    assert_eq!(e4m3(0x39), 1.125);
    assert_eq!(to_e4m3(1.0625), 0x38, "the midpoint goes to the even code");

    // And one step further, where the even code is the upper one: `0x39` is
    // 1.125 and `0x3a` is 1.25, so 1.1875 rounds up to `0x3a`.
    assert_eq!(e4m3(0x3a), 1.25);
    assert_eq!(to_e4m3(1.1875), 0x3a, "the midpoint goes to the even code");
}

#[test]
fn a_magnitude_past_the_ceiling_saturates_rather_than_becoming_a_nan() {
    // The next encoding above 448 IS the NaN one, so "round to nearest" at the
    // top of the range would turn a finite activation into a NaN that poisons
    // a whole row.
    assert_eq!(to_e4m3(1.0e9), 0x7e);
    assert_eq!(to_e4m3(-1.0e9), 0xfe);
    assert_eq!(e4m3(to_e4m3(1.0e9)), E4M3_MAX);
    assert!(e4m3(to_e4m3(f32::NAN)).is_nan());
}

#[test]
fn the_sign_of_zero_survives_the_encode() {
    assert_eq!(to_e4m3(0.0), 0x00);
    assert_eq!(to_e4m3(-0.0), 0x80);
}

#[test]
fn a_value_below_the_subnormal_floor_rounds_to_zero_or_to_the_floor() {
    let floor = e4m3(0x01);
    assert_eq!(to_e4m3(floor), 0x01);
    assert_eq!(to_e4m3(floor * 0.6), 0x01, "nearer the floor than zero");
    assert_eq!(to_e4m3(floor * 0.4), 0x00, "nearer zero than the floor");
    // The midpoint, where the even code is zero.
    assert_eq!(to_e4m3(floor * 0.5), 0x00);
}

#[test]
fn quantizing_activations_divides_by_the_scale_before_rounding() {
    let mut out = Vec::new();
    quantize_activations(&[2.0, -2.0, 0.0], 2.0, &mut out).expect("a positive scale");
    assert_eq!(out, vec![0x38, 0xb8, 0x00]);
    // And the round trip is the activation the kernel will actually see.
    assert_eq!(e4m3(out[0]) * 2.0, 2.0);
}

#[test]
fn a_scale_that_is_not_positive_and_finite_is_refused() {
    // Not a rounding problem: a zero scale maps every activation onto one code
    // and a negative one flips every sign, and both produce a plausible row
    // that nothing downstream can notice.
    let mut out = Vec::new();
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let err =
            quantize_activations(&[1.0], scale, &mut out).expect_err("a scale that cannot divide");
        assert!(matches!(err, GgufError::LayoutMismatch { .. }), "{err:?}");
    }
}

#[test]
fn the_product_scale_exists_only_when_both_sides_were_quantized() {
    let weights_only = Fp8Layout::weights_only(4, 8, 0.5);
    assert_eq!(weights_only.product_scale(), None);
    // `None` is a different launch, not a scale of one: the weights-only path
    // decodes and multiplies by floats, and the weight scale is already folded
    // into the decoded matrix.
    let both = weights_only.with_activations(0.25);
    assert_eq!(both.product_scale(), Some(0.125));
}

#[test]
fn a_quantized_product_agrees_with_the_dequantized_one_within_the_codec() {
    // The relation the device path rests on: accumulating quantized operands
    // and scaling once at the end is the same arithmetic as decoding both and
    // accumulating, up to the order of the multiplies. Graded here on the host
    // against ITSELF, which is all a host can say - the interesting version of
    // this is the kernel oracle, where the accumulation is a tensor core's.
    let weight_scale = 0.01_f32;
    let input_scale = 0.05_f32;
    let codes: Vec<u8> = (0x30..0x40).collect();
    let activations: Vec<f32> = (0..u16::try_from(codes.len()).expect("a short row"))
        .map(|i| f32::from(i).mul_add(0.37, -2.0))
        .collect();

    let mut weights = Vec::new();
    dequant_fp8(
        Fp8Layout::weights_only(1, codes.len(), weight_scale),
        &codes,
        &mut weights,
    )
    .expect("one row");
    let mut quantized = Vec::new();
    quantize_activations(&activations, input_scale, &mut quantized).expect("a positive scale");

    // The reference product uses the decoded weights and the activations as
    // the kernel will actually see them - quantized and decoded again - so the
    // only difference between the two sides is where the scales are applied.
    let reference: f32 = weights
        .iter()
        .zip(&quantized)
        .map(|(weight, code)| weight * (e4m3(*code) * input_scale))
        .sum();
    let scaled: f32 = codes
        .iter()
        .zip(&quantized)
        .map(|(weight, code)| e4m3(*weight) * e4m3(*code))
        .sum::<f32>()
        * (weight_scale * input_scale);

    // Not bit-exact, and the difference is the point: the two orders round
    // differently, and how much is the number a kernel's tolerance has to
    // clear. Measured here at 2.4e-7 relative on this fixture.
    let relative = (reference - scaled).abs() / reference.abs();
    assert!(relative < 1e-6, "relative {relative:e}");
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
//
// QKV: rows 0..2 of `model.language_model.layers.0.linear_attn.in_proj_qkv`, first 16 columns of [8192, 2048]
const QKV_SCALE: f32 = f32::from_bits(0x3a61_2493); // 0.0008588518830947578
const QKV_BYTES: [u8; 32] = [
    0xda, 0x57, 0xbd, 0x46, 0xdb, 0x4a, 0x62, 0x48, 0xdf, 0x61, 0xc9, 0x58, 0xc9, 0xda, 0x5f, 0x58,
    0xdd, 0x63, 0x4a, 0xde, 0xe1, 0x58, 0x29, 0xc1, 0x5a, 0x65, 0xdc, 0xd2, 0xe0, 0x3c, 0x5c, 0xaa,
];
const QKV_EXPECT: [u32; 32] = [
    0xbc8c_b6dc,
    0x3c53_124a,
    0xbab6_edb7,
    0x3b45_0001,
    0xbc9a_c925,
    0x3b8c_b6dc,
    0x3d0c_b6dc,
    0x3b61_2493,
    0xbcd3_124a,
    0x3cfd_4925,
    0xbb7d_4925,
    0x3c61_2493,
    0xbb7d_4925,
    0xbc8c_b6dc,
    0x3cd3_124a,
    0x3c61_2493,
    0xbcb6_edb7,
    0x3d1a_c925,
    0x3b8c_b6dc,
    0xbcc5_0001,
    0xbcfd_4925,
    0x3c61_2493,
    0x397d_4925,
    0xbafd_4925,
    0x3c8c_b6dc,
    0x3d36_edb7,
    0xbca8_db6e,
    0xbc0c_b6dc,
    0xbce1_2493,
    0x3aa8_db6e,
    0x3ca8_db6e,
    0xb98c_b6dc,
];
//   nonzero 32 of 32, amax 0.0446602963

// OUT: rows 0..2 of `model.language_model.layers.0.linear_attn.out_proj`, first 16 columns of [2048, 4096]
const OUT_SCALE: f32 = f32::from_bits(0x3a79_2493); // 0.0009504046174697578
const OUT_BYTES: [u8; 32] = [
    0x4a, 0xd6, 0xb1, 0xc6, 0x55, 0x4a, 0x57, 0xb2, 0x52, 0xcd, 0xd6, 0xce, 0x4a, 0x52, 0xca, 0xd9,
    0x57, 0x49, 0xba, 0xd7, 0xd8, 0xd5, 0xdb, 0x51, 0xc4, 0x49, 0xd5, 0xda, 0xcc, 0xd3, 0xd0, 0x4c,
];
const OUT_EXPECT: [u32; 32] = [
    0x3b9b_b6dc,
    0xbc5a_0001,
    0xba0c_2493,
    0xbb5a_0001,
    0x3c4a_6db7,
    0x3b9b_b6dc,
    0x3c69_924a,
    0xba1b_b6dc,
    0x3c1b_b6dc,
    0xbbca_6db7,
    0xbc5a_0001,
    0xbbda_0001,
    0x3b9b_b6dc,
    0x3c1b_b6dc,
    0xbb9b_b6dc,
    0xbc8c_2493,
    0x3c69_924a,
    0x3b8c_2493,
    0xba9b_b6dc,
    0xbc69_924a,
    0xbc79_2493,
    0xbc4a_6db7,
    0xbcab_4925,
    0x3c0c_2493,
    0xbb3a_db6e,
    0x3b8c_2493,
    0xbc4a_6db7,
    0xbc9b_b6dc,
    0xbbba_db6e,
    0xbc2b_4925,
    0xbbf9_2493,
    0x3bba_db6e,
];
//   nonzero 32 of 32, amax 0.0209089015

// QPROJ: rows 0..2 of `model.language_model.layers.3.self_attn.q_proj`, first 16 columns of [8192, 2048]
const QPROJ_SCALE: f32 = f32::from_bits(0x3a46_db6e); // 0.0007585798157379031
const QPROJ_BYTES: [u8; 32] = [
    0x52, 0x32, 0x5b, 0xce, 0x50, 0x4b, 0x58, 0xdb, 0xce, 0x57, 0x51, 0x5a, 0x35, 0xc8, 0xd1, 0x58,
    0x5b, 0x4b, 0xd2, 0xd6, 0xda, 0x61, 0x5b, 0x57, 0xca, 0x42, 0xd1, 0xd9, 0xd7, 0x58, 0x4c, 0x5e,
];
const QPROJ_EXPECT: [u32; 32] = [
    0x3bf8_924a,
    0x39f8_924a,
    0x3c88_b6dc,
    0xbbae_0000,
    0x3bc6_db6e,
    0x3b88_b6dc,
    0x3c46_db6e,
    0xbc88_b6dc,
    0xbbae_0000,
    0x3c3a_6db7,
    0x3bdf_b6dc,
    0x3c78_924a,
    0x3a21_9249,
    0xbb46_db6e,
    0xbbdf_b6dc,
    0x3c46_db6e,
    0x3c88_b6dc,
    0x3b88_b6dc,
    0xbbf8_924a,
    0xbc2e_0000,
    0xbc78_924a,
    0x3cdf_b6dc,
    0x3c88_b6dc,
    0x3c3a_6db7,
    0xbb78_924a,
    0x3af8_924a,
    0xbbdf_b6dc,
    0xbc5f_b6dc,
    0xbc3a_6db7,
    0x3c46_db6e,
    0x3b95_2492,
    0x3cae_0000,
];
//   nonzero 32 of 32, amax 0.0273088738
