//! The NVFP4 row codec, graded against the container crate's own `e2m1` and
//! `e4m3` decoders over every code and every scale byte.
#![expect(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "the decode is exact up to one brain-float rounding, so values compare by bits; \
              fixture indices are below sixteen and the row width is a small literal"
)]

use shiso_core::ids::TokenId;
use shiso_modelfile::fp8::e4m3;
use shiso_modelfile::nvfp4::e2m1;
use shiso_ple::codec::{self, round_to_bf16};
use shiso_ple::{PleError, PleParams, PleParamsDesc, PleRowType};

/// The checkpoint's table scale, as its header stores it.
const GLOBAL: f32 = 3.324_236e-5;
/// The row width the checkpoint ships.
const HEAD_DIM: usize = 160;

fn params(global: f32) -> PleParams {
    PleParams::new(PleParamsDesc {
        row_type: PleRowType::Nvfp4 {
            global_bits: global.to_bits(),
        },
        ngram_size: 3,
        heads_per_ngram: 1,
        head_dim: HEAD_DIM as u32,
        eos_token: TokenId::new(7),
        multipliers: vec![3, 5, 7],
        head_offsets: vec![0, 11],
        head_vocab: vec![11, 13],
        head_bias: None,
    })
    .expect("valid parameters")
}

/// A row of codes `0..16` in every group, under one scale byte for all ten.
fn row_under(scale: u8) -> Vec<u8> {
    let mut row: Vec<u8> = (0..HEAD_DIM / 2)
        .map(|j| {
            let even = (2 * j % 16) as u8;
            even | ((even + 1) << 4)
        })
        .collect();
    row.extend(std::iter::repeat_n(scale, HEAD_DIM / 16));
    row
}

#[test]
fn a_row_is_its_codes_then_its_group_scales() {
    assert_eq!(params(GLOBAL).row_bytes(), 80 + 10);
}

#[test]
fn every_code_under_every_scale_byte_decodes_as_the_reference_orders_it() {
    let p = params(GLOBAL);
    let mut out = vec![0f32; HEAD_DIM];
    for scale in 0..=255u8 {
        codec::dequant_row(&p, &row_under(scale), 0, &mut out).expect("one whole row");
        for (i, &got) in out.iter().enumerate() {
            let code = (i % 16) as u8;
            let exact = e2m1(code) * e4m3(scale);
            let want = round_to_bf16(exact * GLOBAL);
            if want.is_nan() {
                assert!(got.is_nan(), "scale {scale:#04x} code {code}: {got}");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "scale {scale:#04x} code {code}: {got} vs {want}"
                );
                assert_eq!(got.to_bits() & 0xFFFF, 0, "a brain-float value");
            }
        }
    }
}

#[test]
fn the_low_nibble_is_the_earlier_weight_and_code_eight_is_negative_zero() {
    let p = params(0.5);
    let mut row = vec![0u8; 90];
    row[0] = 0x83; // weight 0: 1.5, weight 1: -0
    row[1] = 0xF7; // weight 2: 6, weight 3: -6
    row[80] = 0x38; // group 0's scale: 1.0
    let mut out = vec![1f32; HEAD_DIM];
    codec::dequant_row(&p, &row, 0, &mut out).expect("one whole row");
    assert_eq!(out[0], 0.75);
    assert_eq!(out[1].to_bits(), (-0.0f32).to_bits());
    assert_eq!(out[2], 3.0);
    assert_eq!(out[3], -3.0);
    // Group 1 has a zero scale, so its weights are zero whatever their codes.
    assert!(out[16..32].iter().all(|v| *v == 0.0));
}

#[test]
fn the_brain_float_rounding_breaks_ties_to_even() {
    // 1 + 2^-8 sits halfway between 1 and the next brain float up, 1 + 2^-7.
    assert_eq!(round_to_bf16(1.0 + 2f32.powi(-8)), 1.0);
    // 1 + 3 * 2^-8 sits halfway between 1 + 2^-7 and 1 + 2^-6: the even one.
    assert_eq!(
        round_to_bf16(1.0 + 3.0 * 2f32.powi(-8)),
        1.0 + 2f32.powi(-6)
    );
    assert_eq!(
        round_to_bf16(1.0 + 2f32.powi(-8) + 2f32.powi(-20)),
        1.0 + 2f32.powi(-7)
    );
    assert!(round_to_bf16(f32::NAN).is_nan());
    assert_eq!(round_to_bf16(f32::INFINITY), f32::INFINITY);
}

#[test]
fn a_row_short_of_its_scales_is_refused() {
    let p = params(GLOBAL);
    let mut out = vec![0f32; HEAD_DIM];
    let err = codec::dequant_row(&p, &[0u8; 80], 0, &mut out).expect_err("no scales");
    assert!(
        matches!(err, PleError::BufferTooSmall { want: 90, .. }),
        "{err}"
    );
}
