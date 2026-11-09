//! `DINOv3` `ViT`-H+ configuration and the pixel normalisation in front of it.
//!
//! Every number here is read from the checkpoint's own `config.json`; the
//! tower does not guess a shape from the weights the way the `VAE` does, because
//! the gated MLP's three projections and the register tokens have no shape
//! that says what they are for.

/// The `ImageNet` mean the tower's inputs are normalised with.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// The `ImageNet` standard deviation the tower's inputs are normalised with.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// What one `DINOv3` checkpoint is built to run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DinoConfig {
    /// Hidden width.
    pub hidden: usize,
    /// Transformer blocks.
    pub layers: usize,
    /// Attention heads.
    pub heads: usize,
    /// Register tokens between the `CLS` token and the patches.
    pub registers: usize,
    /// Patch side, in pixels.
    pub patch: usize,
    /// Gated-MLP intermediate width.
    pub intermediate: usize,
    /// Layer-norm epsilon.
    pub layer_norm_eps: f64,
    /// `RoPE` base.
    pub rope_theta: f64,
    /// The checkpoint's training-time coordinate rescale.
    ///
    /// Kept for the record and **not applied at inference**: the reference
    /// only augments the patch coordinates while training, so a tower that
    /// scaled them here would be silently wrong.
    pub pos_embed_rescale: f64,
}

impl DinoConfig {
    /// The `ViT`-H+ checkpoint the conditioning reads.
    #[must_use]
    pub const fn vit_h_plus() -> Self {
        Self {
            hidden: 1280,
            layers: 32,
            heads: 20,
            registers: 4,
            patch: 16,
            intermediate: 5120,
            layer_norm_eps: 1e-5,
            rope_theta: 100.0,
            pos_embed_rescale: 2.0,
        }
    }

    /// The width of one attention head.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}

/// Applies the `ImageNet` affine to a planar `(3, H, W)` picture in `[0, 1]`.
///
/// The reference normalises in `[0, 1]` space, before the tower, not in byte
/// space: a caller that divided by 255 twice would darken every input.
#[must_use]
pub fn normalize_pixels(rgb: &[f32]) -> Vec<f32> {
    let plane = rgb.len() / 3;
    let mut out = rgb.to_vec();
    for channel in 0..3 {
        let mean = IMAGENET_MEAN[channel];
        let std = IMAGENET_STD[channel];
        for value in &mut out[channel * plane..(channel + 1) * plane] {
            *value = (*value - mean) / std;
        }
    }
    out
}
