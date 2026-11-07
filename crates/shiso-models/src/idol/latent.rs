//! The `VAE` latent's layout and normalisation, on the host.
//!
//! The encoder answers a `(channels, height, width)` latent; the denoiser
//! reads a token sequence. The step between them is pure layout plus a
//! per-channel affine, which is why it lives here rather than on the device:
//! a wrong patch order is a tensor of the right shape with the wrong content,
//! and the model then runs to completion and produces a subtly wrong object.

/// Zero rows prepended to the `VAE` token stream.
///
/// The denoiser reads one conditioning sequence, and the two encoders disagree
/// on length by exactly these five rows: the tower emits 1 `CLS` + 4 registers +
/// 4096 patches, the `VAE` emits 4096 tokens. The prefix is what makes the two
/// line up; dropping it shifts every token by five and the model still runs.
pub const VAE_PREFIX_ROWS: usize = 5;

/// The per-feature affine applied to the `VAE` latent once it is packed.
///
/// The checkpoint carries `bn.running_mean` and `bn.running_var` over the
/// **packed** features - `channels * 4`, one per channel and sub-pixel - not
/// over the latent's channels: the reference normalises after the 2x2
/// patchify. The epsilon is the config's `batch_norm_eps` (1e-4), not the
/// group-norm epsilon.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentNorm {
    mean: Vec<f32>,
    inv_std: Vec<f32>,
}

impl LatentNorm {
    /// Builds the affine from the checkpoint's batch-norm statistics.
    #[must_use]
    pub fn new(mean: &[f32], var: &[f32], eps: f32) -> Self {
        Self {
            mean: mean.to_vec(),
            inv_std: var.iter().map(|v| 1.0 / (v + eps).sqrt()).collect(),
        }
    }

    /// How many packed features the affine covers.
    #[must_use]
    pub fn features(&self) -> usize {
        self.mean.len()
    }
}

/// Turns a `(channels, height, width)` latent into `(height/2 * width/2,
/// channels * 4)` tokens, normalised per packed feature.
///
/// The column order is the reference's sub-pixel interleave: channel-major,
/// then the 2x2 block in `(dy, dx)` order. A plain row-major reshape produces
/// the same shape with the four sub-pixels of a channel scattered across four
/// different channels.
///
/// # Panics
///
/// Panics when `latents` is not `channels * height * width` long, or when the
/// affine does not cover `channels * 4` features. Both are model-loading bugs, caught at
/// the first generation rather than silently truncating a tensor.
#[must_use]
pub fn tokenize(
    latents: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    norm: &LatentNorm,
) -> Vec<f32> {
    assert_eq!(
        latents.len(),
        channels * height * width,
        "a {channels}x{height}x{width} latent is {} values",
        channels * height * width
    );
    assert_eq!(
        norm.features(),
        channels * 4,
        "the batch-norm affine covers {} features, the packed latent has {}",
        norm.features(),
        channels * 4
    );
    let (half_h, half_w) = (height / 2, width / 2);
    let columns = channels * 4;
    let mut out = vec![0.0f32; half_h * half_w * columns];
    for y2 in 0..half_h {
        for x2 in 0..half_w {
            let token = y2 * half_w + x2;
            for channel in 0..channels {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let y = y2 * 2 + dy;
                        let x = x2 * 2 + dx;
                        let feature = channel * 4 + dy * 2 + dx;
                        let value = latents[(channel * height + y) * width + x];
                        out[token * columns + feature] =
                            (value - norm.mean[feature]) * norm.inv_std[feature];
                    }
                }
            }
        }
    }
    out
}

/// Prepends `prefix` zero rows to a token stream.
#[must_use]
pub fn with_prefix(tokens: &[f32], rows: usize, width: usize, prefix: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; (rows + prefix) * width];
    out[prefix * width..].copy_from_slice(tokens);
    out
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    reason = "the fixture is the sixteen small integers 0..16"
)]
mod tests {
    use super::*;

    #[test]
    fn the_sub_pixel_interleave_is_channel_major() {
        // A 4x4 latent with one channel: four tokens of four columns each.
        // Token (0,0) takes the 2x2 block at the top left, token (0,1) the
        // block beside it, and so on. A row-major reshape would give
        // 0,1,2,3,4,.. instead.
        let latents: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let norm = LatentNorm::new(&[0.0; 4], &[1.0; 4], 0.0);
        let out = tokenize(&latents, 1, 4, 4, &norm);
        assert_eq!(
            out,
            vec![
                0.0, 1.0, 4.0, 5.0, // token (0,0)
                2.0, 3.0, 6.0, 7.0, // token (0,1)
                8.0, 9.0, 12.0, 13.0, // token (1,0)
                10.0, 11.0, 14.0, 15.0, // token (1,1)
            ]
        );
    }

    #[test]
    fn normalisation_uses_the_batch_norm_epsilon() {
        let norm = LatentNorm::new(&[1.0; 4], &[4.0; 4], 1e-4);
        let out = tokenize(&[1.0; 4], 1, 2, 2, &norm);
        assert!(out[0].abs() < 1e-6, "the mean maps to zero, got {}", out[0]);
        let out = tokenize(&[3.0; 4], 1, 2, 2, &norm);
        assert!(
            (out[0] - 2.0 / (4.0f32 + 1e-4).sqrt()).abs() < 1e-6,
            "got {}",
            out[0]
        );
    }

    #[test]
    fn the_affine_belongs_to_the_packed_feature_not_the_channel() {
        // The checkpoint's batch norm runs after the 2x2 patchify, over the
        // 128 packed features: feature `c * 4 + p` has its own statistics, so
        // the four sub-pixels of one channel are normalised differently.
        let latents = [1.0, 2.0, 3.0, 4.0];
        let norm = LatentNorm::new(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], 0.0);
        let out = tokenize(&latents, 1, 2, 2, &norm);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn each_channel_keeps_its_own_affine() {
        // Two channels, one 2x2 block: the first four columns are channel 0's
        // sub-pixels, the next four channel 1's.
        let latents = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0];
        let norm = LatentNorm::new(&[0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0], &[1.0; 8], 0.0);
        let out = tokenize(&latents, 2, 2, 2, &norm);
        assert_eq!(out, vec![0.0; 8]);
    }

    #[test]
    fn the_prefix_is_zero_rows() {
        let out = with_prefix(&[1.0, 2.0], 1, 2, 5);
        assert_eq!(out.len(), 6 * 2);
        assert!(out[..10].iter().all(|value| *value == 0.0));
        assert_eq!(&out[10..], &[1.0, 2.0]);
    }
}
