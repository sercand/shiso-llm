//! The conditioning seam: a preprocessed picture to the two streams the
//! denoiser reads.
//!
//! The tower answers the semantic stream; the `VAE` answers a posterior draw,
//! which is packed 2x2, normalised with the checkpoint's batch-norm
//! statistics and given five zero rows so its length matches the tower's.
//! The tower runs first and draws nothing; the posterior's noise is the first
//! draw of the generation, which is the reference's order.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use shiso_vision::generator::ImageGenError;

use super::conditioning::{Conditioning, assemble};
use super::dinov3::encoder::DinoV3;
use super::draws::Draws;
use super::latent::{LatentNorm, VAE_PREFIX_ROWS, tokenize, with_prefix};
use crate::diffusion::ops::Ops;
use crate::flux2::vae_encoder::VaeEncoder;

/// Turns a preprocessed picture into the denoiser's conditioning.
pub trait Conditioner: Send + Sync + fmt::Debug {
    /// The conditioning for a planar `(3, side, side)` picture in `[0, 1]`,
    /// taking the `VAE` posterior's noise from `draws`.
    ///
    /// # Errors
    ///
    /// A device failure, or a stream of a shape the denoiser does not read.
    fn condition(
        &self,
        rgb: &[f32],
        side: usize,
        draws: &mut dyn Draws,
    ) -> Result<Conditioning, String>;
}

/// The two encoders on one device.
#[derive(Debug)]
pub struct IdolConditioner {
    ops: Arc<Ops>,
    tower: DinoV3,
    vae: VaeEncoder,
    norm: LatentNorm,
}

impl IdolConditioner {
    /// Opens the tower at `tower_dir` and the `VAE` component at `vae_dir`.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] for a checkpoint of another shape.
    pub(crate) fn load(
        ops: Arc<Ops>,
        tower_dir: &Path,
        vae_dir: &Path,
    ) -> Result<Self, ImageGenError> {
        let tower = DinoV3::load(tower_dir, &ops)?;
        let vae = VaeEncoder::load(vae_dir, &ops)?;
        let (mean, var, eps) = vae.batch_norm();
        if mean.len() != vae.latent_channels() * 4 {
            return Err(ImageGenError::Checkpoint(format!(
                "the VAE's batch norm covers {} features, its packed latent has {}",
                mean.len(),
                vae.latent_channels() * 4
            )));
        }
        let norm = LatentNorm::new(mean, var, eps);
        Ok(Self {
            ops,
            tower,
            vae,
            norm,
        })
    }
}

impl Conditioner for IdolConditioner {
    fn condition(
        &self,
        rgb: &[f32],
        side: usize,
        draws: &mut dyn Draws,
    ) -> Result<Conditioning, String> {
        let dinov3 = self
            .tower
            .tokens(&self.ops, rgb, side, side)
            .map_err(|e| e.to_string())?;
        let moments = self
            .vae
            .moments(&self.ops, rgb, side, side)
            .map_err(|e| e.to_string())?;
        let channels = self.vae.latent_channels();
        let noise = draws.normal(moments.len() / 2);
        let latent = VaeEncoder::sample(&moments, &noise).map_err(|e| e.to_string())?;
        let tokens = tokenize(&latent, channels, side / 8, side / 8, &self.norm);
        let rows = tokens.len() / (channels * 4);
        let vae = with_prefix(&tokens, rows, channels * 4, VAE_PREFIX_ROWS);
        self.ops.trim_pool();
        assemble(dinov3, vae).map_err(|e| e.to_string())
    }
}
