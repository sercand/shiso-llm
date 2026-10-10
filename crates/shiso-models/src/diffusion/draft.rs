//! The half-resolution draft stage: the leading denoising steps of a job run
//! on a latent grid half as tall and half as wide, and the job finishes at
//! its own size.
//!
//! A step's cost is set by its token count - the linears are linear in it,
//! the attention quadratic - so a half-size step costs about a quarter of a
//! full one (4.85 s against 22.4 s for a 1024x1024 Qwen-Image-2.1 step on
//! an M4). The early, high-noise steps decide the layout, which a quarter
//! of the tokens already resolves; the late steps add the detail, which
//! needs them all.
//!
//! The hand-over keeps the draft's own noisy state ([`upsample2x_noisy`]):
//! it is upsampled by nearest neighbour and topped up with exactly the
//! fresh noise that makes it a state of the full grid's flow path, at a
//! sigma the hand-over itself fixes. Two alternatives were measured worse
//! on a 6-step distilled checkpoint at 1024x1024 (4 draft steps, then 2):
//! re-noising the draft's clean estimate `x - sigma v` after a bilinear
//! upsample let the full-size steps re-invent the detail (a distorted face,
//! extra lettering), and with one full-size step left it kept a visible
//! grid of the fresh noise.

/// Nearest 2x upsampling of a noisy draft `(rows, cols, channels)` state
/// `x` at `sigma_e`, corrected to a state of the full grid at the returned
/// `sigma_s`, in place of `noise` (`(2 rows, 2 cols, channels)` standard
/// normals).
///
/// With `x = (1 - sigma_e) x0 + sigma_e e`, `a * nearest(x)` carries noise
/// of covariance `a^2 sigma_e^2 J` (`J` all-ones over each 2x2 block, whose
/// one nonzero eigenvalue is 4). Fresh noise `z = w - alpha *
/// nearest(blockmean(w))` has covariance `I - keep * J` when `alpha = 1 -
/// sqrt(1 - 4 keep)`, so `a * nearest(x) + sigma_s z` has noise covariance
/// `sigma_s^2 I` - the flow path at `sigma_s`, cell by cell - when `a^2
/// sigma_e^2 = keep * sigma_s^2`, and carries `1 - sigma_s` of the signal
/// when `a (1 - sigma_e) = 1 - sigma_s`. Together: `sigma_s = sigma_e /
/// (sigma_e + sqrt(keep) (1 - sigma_e))`, `a = sqrt(keep) sigma_s /
/// sigma_e`. `keep` lies in (0, 1/4]; at 1/4, `sigma_s = 2 sigma_e / (1 +
/// sigma_e)` and the fresh noise is exactly the part orthogonal to the
/// block means.
pub(crate) fn upsample2x_noisy(
    state: &[f32],
    (rows, cols, channels): (usize, usize, usize),
    sigma_e: f32,
    keep: f32,
    noise: &mut [f32],
) -> f32 {
    let root = keep.sqrt();
    let sigma_s = sigma_e / (sigma_e + root * (1.0 - sigma_e));
    let gain = root * sigma_s / sigma_e;
    let alpha = 1.0 - (1.0 - 4.0 * keep).max(0.0).sqrt();
    let wide = 2 * cols;
    for row in 0..rows {
        for col in 0..cols {
            let cells = [
                (2 * row) * wide + 2 * col,
                (2 * row) * wide + 2 * col + 1,
                (2 * row + 1) * wide + 2 * col,
                (2 * row + 1) * wide + 2 * col + 1,
            ];
            let src = &state[(row * cols + col) * channels..][..channels];
            for ch in 0..channels {
                let mean = cells.iter().map(|&c| noise[c * channels + ch]).sum::<f32>() / 4.0;
                for &c in &cells {
                    let value = &mut noise[c * channels + ch];
                    *value = gain * src[ch] + sigma_s * (*value - alpha * mean);
                }
            }
        }
    }
    sigma_s
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    reason = "test fixtures: small grid sizes and sample counts as floats, the formula's own letters"
)]
mod tests {
    use super::upsample2x_noisy;

    /// With no fresh noise, every full-grid cell is `a` times its draft
    /// cell, and at `keep = 1/4` the hand-over sigma is `2 s / (1 + s)`.
    #[test]
    fn the_signal_is_scaled_nearest_neighbour() {
        let (h, w, c) = (2usize, 3usize, 2usize);
        let x: Vec<f32> = (0..h * w * c).map(|i| i as f32 - 3.0).collect();
        let sigma_e = 0.6;
        let mut out = vec![0.0f32; 4 * h * w * c];
        let sigma_s = upsample2x_noisy(&x, (h, w, c), sigma_e, 0.25, &mut out);
        assert!((sigma_s - 2.0 * sigma_e / (1.0 + sigma_e)).abs() < 1e-6);
        let a = 0.5 * sigma_s / sigma_e;
        for y in 0..2 * h {
            for xx in 0..2 * w {
                for ch in 0..c {
                    let want = a * x[((y / 2) * w + xx / 2) * c + ch];
                    let got = out[(y * 2 * w + xx) * c + ch];
                    assert!(
                        (got - want).abs() < 1e-5,
                        "({y}, {xx}, {ch}): {got} vs {want}"
                    );
                }
            }
        }
    }

    /// A draft state that is pure noise at `sigma_e` hands over as noise of
    /// variance `sigma_s^2`, uncorrelated between the four cells of a block,
    /// for every `keep`.
    #[test]
    fn pure_noise_hands_over_as_white_noise() {
        // Box-Muller over a fixed LCG: deterministic standard normals.
        let mut state = 12_345u64;
        let mut normal = move || {
            let mut u = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            let (u1, u2) = (u(), u());
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        };
        let (h, w, c) = (32usize, 32usize, 16usize);
        for keep in [0.25f32, 0.125] {
            let sigma_e = 0.7f32;
            let x: Vec<f32> = (0..h * w * c).map(|_| sigma_e * normal()).collect();
            let mut out: Vec<f32> = (0..4 * h * w * c).map(|_| normal()).collect();
            let sigma_s = upsample2x_noisy(&x, (h, w, c), sigma_e, keep, &mut out);
            let n = out.len() as f64;
            let var = out.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / n;
            let want = f64::from(sigma_s).powi(2);
            assert!(
                (var / want - 1.0).abs() < 0.03,
                "keep {keep}: variance {var} vs {want}"
            );
            // Horizontal neighbours inside a block.
            let mut cov = 0.0f64;
            let mut pairs = 0usize;
            for y in 0..2 * h {
                for bx in 0..w {
                    for ch in 0..c {
                        let l = out[(y * 2 * w + 2 * bx) * c + ch];
                        let r = out[(y * 2 * w + 2 * bx + 1) * c + ch];
                        cov += f64::from(l) * f64::from(r);
                        pairs += 1;
                    }
                }
            }
            let corr = cov / pairs as f64 / want;
            assert!(
                corr.abs() < 0.03,
                "keep {keep}: in-block correlation {corr}"
            );
        }
    }
}
