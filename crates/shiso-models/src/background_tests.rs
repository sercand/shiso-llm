//! The background policy's transparency threshold.

use super::background::is_transparent;

/// `n` RGBA pixels, every one opaque except `low` of them at `alpha`.
fn pixels(n: usize, low: usize, alpha: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 * n);
    for i in 0..n {
        out.extend_from_slice(&[10, 20, 30, if i < low { alpha } else { 255 }]);
    }
    out
}

#[test]
fn threshold_is_one_percent_below_half() {
    assert!(is_transparent(&pixels(100, 1, 127)));
    assert!(!is_transparent(&pixels(100, 1, 128)));
    assert!(!is_transparent(&pixels(100, 0, 0)));
    // Under one percent is noise, not a cut-out.
    assert!(!is_transparent(&pixels(1000, 9, 0)));
    assert!(is_transparent(&pixels(1000, 10, 0)));
}

#[test]
fn near_opaque_noise_is_opaque() {
    let mut noisy = Vec::new();
    for i in 0..1000u32 {
        noisy.extend_from_slice(&[1, 2, 3, 245 + u8::try_from(i % 11).expect("small")]);
    }
    assert!(!is_transparent(&noisy));
}
