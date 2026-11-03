//! The MXFP8 decode, exact, so compared on bits.

#![expect(
    clippy::float_cmp,
    reason = "a power-of-two scale only moves an exponent, so every value here \
              is exact and a tolerance would hide a wrong scale bias"
)]

use super::*;

#[test]
fn a_scale_byte_is_two_to_its_value_minus_127() {
    assert_eq!(ue8m0(127), Some(1.0));
    assert_eq!(ue8m0(128), Some(2.0));
    assert_eq!(ue8m0(120), Some(1.0 / 128.0));
    assert_eq!(ue8m0(254), Some(2f32.powi(127)));
    assert_eq!(ue8m0(1), Some(2f32.powi(-126)));
    assert_eq!(
        ue8m0(0).map(f32::to_bits),
        Some(1 << 22),
        "2^-127 is the one subnormal"
    );
    assert_eq!(ue8m0(NAN_SCALE), None);
}

#[test]
fn every_weight_is_its_code_times_its_groups_scale() {
    // Two rows of 64: four groups with scales 2^0, 2^1, 2^-3, 2^4.
    let layout = Mxfp8Layout { rows: 2, cols: 64 };
    let weights: Vec<u8> = (0..128)
        .map(|i| u8::try_from(i % 0x7F).unwrap_or(0))
        .collect();
    let scales = [127, 128, 124, 131];
    let mut out = Vec::new();
    dequant_mxfp8("t", layout, &weights, &scales, &mut out).expect("decodes");
    for (i, value) in out.iter().enumerate() {
        let scale = ue8m0(scales[i / GROUP]).expect("finite");
        assert_eq!(
            value.to_bits(),
            (e4m3(weights[i]) * scale).to_bits(),
            "weight {i}"
        );
    }
    assert_eq!(out[33], e4m3(33) * 2.0);
}

#[test]
fn a_nan_scale_and_a_short_array_are_refused() {
    let layout = Mxfp8Layout { rows: 1, cols: 32 };
    let mut out = Vec::new();
    let err = dequant_mxfp8("t", layout, &[0; 32], &[NAN_SCALE], &mut out);
    assert!(
        matches!(err, Err(GgufError::LayoutMismatch { .. })),
        "{err:?}"
    );
    let err = dequant_mxfp8("t", layout, &[0; 31], &[127], &mut out);
    assert!(
        matches!(err, Err(GgufError::LayoutMismatch { .. })),
        "{err:?}"
    );
    let ragged = Mxfp8Layout { rows: 1, cols: 48 };
    let err = dequant_mxfp8("t", ragged, &[0; 48], &[127, 127], &mut out);
    assert!(
        matches!(err, Err(GgufError::LayoutMismatch { .. })),
        "{err:?}"
    );
}
