//! Rate conversion to what the models consume.
//!
//! Polyphase windowed sinc: every output sample falls on one of `to/gcd`
//! fractional input positions, so the kernel is tabulated once per phase and
//! each output is a dot product. The cutoff sits just under the lower
//! Nyquist, so downsampling removes what would alias.

use crate::error::AudioError;
use crate::num::{f32_of, f64_of};

/// The highest rate accepted on either side. Professional audio stops at
/// 384 kHz; the bound exists because the phase table grows with the rate, so an
/// unchecked header declaring a gigahertz rate would cost gigabytes of table
/// for a file of a few hundred bytes.
pub(crate) const MAX_RATE: u32 = 384_000;

/// Zero crossings of the sinc kept on each side, at the lower rate.
const ZERO_CROSSINGS: f64 = 16.0;
/// The cutoff as a fraction of the lower Nyquist, leaving room for the
/// window's transition band.
const ROLLOFF: f64 = 0.95;

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Resamples `input` from `from` Hz to `to` Hz.
///
/// # Errors
///
/// [`AudioError::BadRate`] when either rate is zero or above [`MAX_RATE`].
pub(crate) fn resample(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>, AudioError> {
    for rate in [from, to] {
        if rate == 0 || rate > MAX_RATE {
            return Err(AudioError::BadRate { rate });
        }
    }
    if from == to {
        return Ok(input.to_vec());
    }
    let (from64, to64) = (u64::from(from), u64::from(to));
    let g = gcd(from64, to64);
    let phases = usize::try_from(to64 / g).map_err(|_| AudioError::BadRate { rate: to })?;
    // Cutoff in cycles per input sample.
    let fc = ROLLOFF * 0.5 * f64::from(from.min(to)) / f64::from(from);
    let half = crate::num::usize_floor((ZERO_CROSSINGS / (2.0 * fc)).ceil());
    let taps = 2 * half + 1;
    let mut table = vec![0.0f32; phases * taps];
    for p in 0..phases {
        let frac = f64_of(p) / f64_of(phases);
        for k in 0..taps {
            // Distance from the output position to input tap `k`.
            let d = f64_of(k) - f64_of(half) - frac;
            let x = 2.0 * fc * d;
            let sinc = if x.abs() < 1e-12 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let span = f64_of(half) + 1.0;
            let w = if d.abs() >= span {
                0.0
            } else {
                let r = (d + span) / (2.0 * span);
                0.42 - 0.5 * (2.0 * std::f64::consts::PI * r).cos()
                    + 0.08 * (4.0 * std::f64::consts::PI * r).cos()
            };
            table[p * taps + k] = f32_of(2.0 * fc * sinc * w);
        }
    }
    let out_len =
        usize::try_from((u64::try_from(input.len()).unwrap_or(u64::MAX) * to64).div_ceil(from64))
            .unwrap_or(usize::MAX);
    let mut out = Vec::with_capacity(out_len);
    let step = from64 / g; // input advance per output, in units of 1/phases
    let phases64 = to64 / g;
    for n in 0..out_len {
        let pos = u64::try_from(n).unwrap_or(u64::MAX) * step;
        let base = usize::try_from(pos / phases64).unwrap_or(usize::MAX);
        let phase = usize::try_from(pos % phases64).unwrap_or(0);
        let row = &table[phase * taps..(phase + 1) * taps];
        let mut acc = 0.0f32;
        for (k, w) in row.iter().enumerate() {
            let idx = (base + k).checked_sub(half);
            if let Some(v) = idx.and_then(|i| input.get(i)) {
                acc += v * w;
            }
        }
        out.push(acc);
    }
    Ok(out)
}

#[cfg(test)]
#[path = "resample_tests.rs"]
mod tests;
