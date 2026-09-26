//! Resampling keeps a tone's frequency and level and produces the expected
//! length, for the rates clients actually send.

use super::resample;
use crate::num::f64_of;

fn tone(freq: f64, rate: u32, secs: f64) -> Vec<f32> {
    let n = crate::num::usize_floor(secs * f64::from(rate));
    (0..n)
        .map(|i| {
            crate::num::f32_of(
                (2.0 * std::f64::consts::PI * freq * f64_of(i) / f64::from(rate)).sin() * 0.5,
            )
        })
        .collect()
}

/// Correlation of `x` against a unit tone at `freq`: about `amp/2`·len when the
/// tone is there.
fn level_at(x: &[f32], freq: f64, rate: u32) -> f64 {
    let (mut s, mut c) = (0.0, 0.0);
    for (i, v) in x.iter().enumerate() {
        let w = 2.0 * std::f64::consts::PI * freq * f64_of(i) / f64::from(rate);
        s += f64::from(*v) * w.sin();
        c += f64::from(*v) * w.cos();
    }
    2.0 * (s * s + c * c).sqrt() / f64_of(x.len())
}

#[test]
fn identity_is_exact() {
    let x = tone(440.0, 16_000, 0.1);
    assert_eq!(resample(&x, 16_000, 16_000).expect("ok"), x);
}

#[test]
fn common_rates_keep_the_tone_and_the_length() {
    for from in [8_000u32, 22_050, 44_100, 48_000] {
        let x = tone(1_000.0, from, 1.0);
        let y = resample(&x, from, 16_000).expect("ok");
        let expected = (x.len() * 16_000).div_ceil(usize::try_from(from).expect("fits"));
        assert_eq!(y.len(), expected, "from {from}");
        // Skip the edges, where the zero padding tapers the tone.
        let mid = &y[800..y.len() - 800];
        let level = level_at(mid, 1_000.0, 16_000);
        assert!((level - 0.5).abs() < 0.01, "from {from}: level {level}");
        assert!(
            level_at(mid, 3_000.0, 16_000) < 0.005,
            "from {from}: leakage"
        );
    }
}

#[test]
fn content_above_the_new_nyquist_is_removed() {
    let x = tone(11_000.0, 48_000, 1.0);
    let y = resample(&x, 48_000, 16_000).expect("ok");
    let rms = (y[800..y.len() - 800]
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        / f64_of(y.len() - 1600))
    .sqrt();
    assert!(rms < 0.01, "aliased energy {rms}");
}

#[test]
fn zero_rate_is_refused() {
    assert!(resample(&[0.0], 0, 16_000).is_err());
}

#[test]
fn implausible_rates_are_refused_before_any_table_is_built() {
    // A tiny file declaring a huge rate would otherwise size the phase table
    // by the rate: gigabytes from a few hundred bytes of input.
    let x = [0.0f32; 100];
    for rate in [1_000_000_007u32, 10_000_019, 384_001] {
        assert!(
            matches!(resample(&x, rate, 16_000), Err(crate::error::AudioError::BadRate { rate: r }) if r == rate),
            "rate {rate}"
        );
    }
    assert!(resample(&x, 384_000, 16_000).is_ok());
}
