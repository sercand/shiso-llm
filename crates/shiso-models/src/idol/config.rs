//! The checkpoint's `config.json`: the three networks' shapes and the
//! representation they decode into.
//!
//! The graphs are written for one architecture, so the document is read to
//! **refuse** a checkpoint that describes another, not to reshape the graphs:
//! a block count or width read silently from a different export would run a
//! wrong model without an error.

use std::path::Path;

use serde::Deserialize;
use shiso_vision::generator::ImageGenError;

use super::gaussian::RepresentationConfig;

/// The denoiser's arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct DenoiserArgs {
    /// The latent's token count.
    pub q_token_length: usize,
    /// The latent's width.
    pub in_channels: usize,
    /// The camera's width.
    pub cam_channels: usize,
    /// The trunk's width.
    pub model_channels: usize,
    /// The tower stream's width.
    pub cond_channels: usize,
    /// The `VAE` stream's width.
    pub cond2_channels: usize,
    /// Joint blocks.
    pub num_blocks: usize,
    /// Refiner blocks per stream.
    pub num_refiner_blocks: usize,
    /// Attention heads.
    pub num_heads: usize,
    /// The MLP's width as a multiple of the trunk's.
    pub mlp_ratio: usize,
}

/// The octree decoder's arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct OctreeArgs {
    /// The trunk's width.
    pub model_channels: usize,
    /// The latent's width (the cross attention's context).
    pub cond_channels: usize,
    /// Blocks.
    pub num_blocks: usize,
    /// Attention heads.
    pub num_heads: usize,
    /// The MLP's width as a multiple of the trunk's.
    pub mlp_ratio: usize,
}

/// The Gaussian decoder's arguments.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GaussianArgs {
    /// The anchors' width (three coordinates).
    pub in_channels: usize,
    /// The trunk's width.
    pub model_channels: usize,
    /// The latent's width.
    pub cond_channels: usize,
    /// Blocks.
    pub num_blocks: usize,
    /// Attention heads.
    pub num_heads: usize,
    /// The MLP's width as a multiple of the trunk's.
    pub mlp_ratio: usize,
    /// How the outputs become Gaussians.
    pub representation_config: RepresentationDocument,
}

/// The representation block of the document, verbatim.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RepresentationDocument {
    /// Gaussians per anchor.
    pub num_gaussians: usize,
    /// Per-field output multipliers.
    pub lr: LrDocument,
    /// The offset's reach, in cells.
    pub perturbe_size: f32,
    /// The scale floor.
    #[serde(rename = "3d_filter_kernel_size")]
    pub kernel_size: f32,
    /// The scale bias.
    pub scaling_bias: f32,
    /// The opacity bias.
    pub opacity_bias: f32,
}

/// The per-field output multipliers.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct LrDocument {
    /// The offset's.
    #[serde(rename = "_xyz")]
    pub xyz: f32,
    /// The colour's.
    #[serde(rename = "_features_dc")]
    pub features_dc: f32,
    /// The opacity's.
    #[serde(rename = "_opacity")]
    pub opacity: f32,
    /// The scale's.
    #[serde(rename = "_scaling")]
    pub scaling: f32,
    /// The rotation's.
    #[serde(rename = "_rotation")]
    pub rotation: f32,
}

#[derive(Debug, Deserialize)]
struct Section<T> {
    args: T,
}

#[derive(Debug, Deserialize)]
struct Document {
    denoiser: Section<DenoiserArgs>,
    decoder: Section<OctreeArgs>,
    decoder_gs: Section<GaussianArgs>,
    image_size: usize,
    max_voxel_level: u32,
}

/// The whole checkpoint document.
#[derive(Debug, Clone, PartialEq)]
pub struct IdolConfig {
    /// The denoiser.
    pub denoiser: DenoiserArgs,
    /// The octree decoder.
    pub octree: OctreeArgs,
    /// The Gaussian decoder.
    pub gaussian: GaussianArgs,
    /// The encoders' input side.
    pub image_size: usize,
    /// The octree's deepest level.
    pub max_voxel_level: u32,
}

impl IdolConfig {
    /// The shipped checkpoint's document, which the graphs are written for.
    #[must_use]
    pub fn idol() -> Self {
        let rep = RepresentationConfig::idol();
        Self {
            denoiser: DenoiserArgs {
                q_token_length: 8192,
                in_channels: 16,
                cam_channels: 5,
                model_channels: 1024,
                cond_channels: 1280,
                cond2_channels: 128,
                num_blocks: 24,
                num_refiner_blocks: 2,
                num_heads: 16,
                mlp_ratio: 4,
            },
            octree: OctreeArgs {
                model_channels: 1024,
                cond_channels: 16,
                num_blocks: 4,
                num_heads: 16,
                mlp_ratio: 4,
            },
            gaussian: GaussianArgs {
                in_channels: 3,
                model_channels: 1024,
                cond_channels: 16,
                num_blocks: 16,
                num_heads: 16,
                mlp_ratio: 4,
                representation_config: RepresentationDocument {
                    num_gaussians: rep.num_gaussians,
                    lr: LrDocument {
                        xyz: rep.lr_xyz,
                        features_dc: rep.lr_features,
                        opacity: rep.lr_opacity,
                        scaling: rep.lr_scaling,
                        rotation: rep.lr_rotation,
                    },
                    perturbe_size: rep.perturbe_size,
                    kernel_size: rep.kernel_size,
                    scaling_bias: rep.scaling_bias,
                    opacity_bias: rep.opacity_bias,
                },
            },
            image_size: 1024,
            max_voxel_level: 8,
        }
    }

    /// Parses a document.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] for a document that does not parse.
    pub fn parse(text: &str) -> Result<Self, ImageGenError> {
        let doc: Document = serde_json::from_str(text)
            .map_err(|e| ImageGenError::Checkpoint(format!("the splat config.json: {e}")))?;
        Ok(Self {
            denoiser: doc.denoiser.args,
            octree: doc.decoder.args,
            gaussian: doc.decoder_gs.args,
            image_size: doc.image_size,
            max_voxel_level: doc.max_voxel_level,
        })
    }

    /// Reads `dir/config.json` and refuses one that is not [`IdolConfig::idol`].
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] for a missing, unparsable or different
    /// document, naming the first field that differs.
    pub fn read(dir: &Path) -> Result<Self, ImageGenError> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| ImageGenError::Checkpoint(format!("{}: {e}", path.display())))?;
        let config = Self::parse(&text)?;
        config.check()?;
        Ok(config)
    }

    /// Refuses a document the graphs were not written for.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] naming the section that differs.
    pub fn check(&self) -> Result<(), ImageGenError> {
        let want = Self::idol();
        let differs = |section: &str, got: &dyn std::fmt::Debug, wanted: &dyn std::fmt::Debug| {
            ImageGenError::Checkpoint(format!(
                "the splat checkpoint's {section} is {got:?}; this build runs {wanted:?}"
            ))
        };
        if self.denoiser != want.denoiser {
            return Err(differs("denoiser", &self.denoiser, &want.denoiser));
        }
        if self.octree != want.octree {
            return Err(differs("decoder", &self.octree, &want.octree));
        }
        if self.gaussian != want.gaussian {
            return Err(differs("decoder_gs", &self.gaussian, &want.gaussian));
        }
        if (self.image_size, self.max_voxel_level) != (want.image_size, want.max_voxel_level) {
            return Err(differs(
                "(image_size, max_voxel_level)",
                &(self.image_size, self.max_voxel_level),
                &(want.image_size, want.max_voxel_level),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint's own `config.json`, verbatim (2731c22).
    const SHIPPED: &str = r#"{
  "denoiser": {"name": "LatentSeqMMFlowModel", "args": {"q_token_length": 8192, "in_channels": 16,
    "cam_channels": 5, "model_channels": 1024, "cond_channels": 1280, "cond2_channels": 128,
    "num_blocks": 24, "num_refiner_blocks": 2, "num_heads": 16, "mlp_ratio": 4}},
  "decoder": {"name": "OctreeProbabilityFixedlenDecoder", "args": {"model_channels": 1024,
    "cond_channels": 16, "num_blocks": 4, "num_heads": 16, "mlp_ratio": 4}},
  "decoder_gs": {"name": "GaussianFixedlenDecoder", "args": {"in_channels": 3, "model_channels": 1024,
    "cond_channels": 16, "num_blocks": 16, "num_heads": 16, "mlp_ratio": 4,
    "representation_config": {"num_gaussians": 32, "lr": {"_xyz": 1.0, "_features_dc": 1.0,
      "_opacity": 1.0, "_scaling": 1.0, "_rotation": 0.1}, "perturbe_size": 1.5,
      "3d_filter_kernel_size": 0.0009, "scaling_bias": 0.004, "opacity_bias": 0.1}}},
  "image_size": 1024,
  "max_voxel_level": 8,
  "image_encoders": {"dinov3": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "flux2_vae": "black-forest-labs/FLUX.2-dev"}
}"#;

    #[test]
    fn the_shipped_document_is_the_one_the_graphs_run() {
        let config = IdolConfig::parse(SHIPPED).expect("the shipped document parses");
        assert_eq!(config, IdolConfig::idol());
        config.check().expect("and is accepted");
    }

    #[test]
    fn a_different_block_count_is_refused_by_name() {
        let text = SHIPPED.replace("\"num_blocks\": 24", "\"num_blocks\": 12");
        let err = IdolConfig::parse(&text)
            .expect("parses")
            .check()
            .expect_err("a 12-block denoiser is not this graph");
        assert!(err.to_string().contains("denoiser"), "{err}");
    }
}
