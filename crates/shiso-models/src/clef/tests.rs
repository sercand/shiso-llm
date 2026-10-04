#![expect(clippy::float_cmp, reason = "dequantized values are exact products")]

use super::host::{HostNorm, dequant_row, l2_normalized, mean_rows, softmax};
use shiso_core::DType;

#[test]
fn layer_norm_matches_the_biased_variance() {
    let norm = HostNorm {
        weight: vec![1.0, 2.0, 1.0, 1.0],
        bias: vec![0.0, 0.0, 0.5, 0.0],
        eps: 0.0,
    };
    let y = norm.apply(&[1.0, 2.0, 3.0, 4.0]);
    // mean 2.5, variance 1.25
    let inv = 1.0 / 1.25f32.sqrt();
    let want = [-1.5 * inv, -0.5 * inv * 2.0, 0.5 * inv + 0.5, 1.5 * inv];
    for (a, b) in y.iter().zip(want) {
        assert!((a - b).abs() < 1e-6, "{y:?}");
    }
}

#[test]
fn pooling_and_normalizing() {
    let rows = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(mean_rows(&rows, 2, 1..3), vec![4.0, 5.0]);
    let unit = l2_normalized(&[3.0, 4.0], 1e-12);
    assert!((unit[0] - 0.6).abs() < 1e-6 && (unit[1] - 0.8).abs() < 1e-6);
    assert_eq!(l2_normalized(&[0.0, 0.0], 1e-12), vec![0.0, 0.0]);
    let p = softmax(&[0.0, 0.0f32.ln_1p()]);
    assert!((p[0] - 0.5).abs() < 1e-6);
}

#[test]
fn q6_k_rows_dequantize_as_ggml_lays_them_out() {
    // One super-block: every low nibble 1, every high pair 2 (so q = 33 - 32 =
    // 1 ... ), scales 0..16, d = 0.5.
    let mut block = vec![0u8; 210];
    for b in &mut block[..128] {
        *b = 0x21; // low nibble 1, high nibble 2
    }
    for b in &mut block[128..192] {
        *b = 0b0101_0101; // every 2-bit pair is 1
    }
    for (i, b) in block[192..208].iter_mut().enumerate() {
        *b = u8::try_from(i).expect("sixteen scales");
    }
    block[208..].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    let row = dequant_row(&block, DType::Q6_K, 256, 0).expect("row");
    // low nibble 1 with high pair 1: (1 | 16) - 32 = -15; high nibble 2: (2 | 16) - 32 = -14.
    assert_eq!(row[0], 0.5 * 0.0 * -15.0);
    assert_eq!(row[16], 0.5 * 1.0 * -15.0);
    assert_eq!(row[32], 0.5 * 2.0 * -15.0);
    assert_eq!(row[64], 0.5 * 4.0 * -14.0);
    assert_eq!(row[96], 0.5 * 6.0 * -14.0);
    assert_eq!(row[128], 0.5 * 8.0 * -15.0);
    assert_eq!(row[255], 0.5 * 15.0 * -14.0);
}
