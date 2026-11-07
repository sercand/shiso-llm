//! The loaded splat model: the four device stages behind one
//! [`SplatGenerator`].
//!
//! A generation is preprocess, condition, integrate, decode, write: the
//! picture becomes the 1024 square, the encoders answer the conditioning
//! (the `VAE` posterior's noise is the job's first draw), the flow integrates
//! the latent and the camera, the octree places the anchors and the Gaussian
//! decoder expands them, and the cloud is written as a `.ply`. Every draw
//! comes from one [`SeededDraws`] over the job's seed, in the reference's
//! order.

use std::path::Path;
use std::sync::Arc;

use shiso_core::Backend;
use shiso_vision::generator::ImageGenError;
use shiso_vision::splat::{
    GeneratedSplat, SplatGenError, SplatGenerator, SplatJob, SplatLimits, SplatStepEvent,
};

use super::conditioner::{Conditioner, IdolConditioner};
use super::conditioning::Conditioning;
use super::config::IdolConfig;
use super::decoders::{GaussianNet, OctreeNet};
use super::denoiser::IdolDenoiser;
use super::draws::{Draws, SeededDraws};
use super::gaussian::{RawOutput, RepresentationConfig, activate};
use super::generator::{
    Denoiser, FlowState, GaussianDecoder, OctreeDecoder, Pipeline, PipelineConfig, PipelineError,
    run,
};
use super::octree::CHILDREN;
use super::ply::{GaussianCloud, write_ply};
use super::preprocess::{IMAGE_SIZE, Matting, PreprocessError, preprocess};
use crate::diffusion::ops::Ops;
use crate::splat_registry::SplatModelDesc;

/// The denoiser's joint sequence: 8192 latent tokens, 4101 conditioning rows
/// and the camera. One step's `completion_tokens`, counted once per step even
/// though guidance runs two passes.
pub const SEQUENCE_TOKENS: u32 = 12_294;

/// The `.ply` media type.
const MEDIA_TYPE: &str = "model/ply";

impl Denoiser for IdolDenoiser {
    fn velocity(
        &self,
        sample: &FlowState,
        timestep: f32,
        cond: &Conditioning,
    ) -> Result<FlowState, String> {
        let v = self
            .forward(&sample.latent, &sample.camera, timestep, cond)
            .map_err(|e| e.to_string())?;
        Ok(FlowState {
            latent: v.latent,
            camera: v.camera,
        })
    }
}

/// The softmax of each row of eight logits, in f32 as the reference takes it.
fn softmax_rows(logits: &[f32]) -> Vec<[f32; CHILDREN]> {
    logits
        .chunks_exact(CHILDREN)
        .map(|row| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut out = [0.0f32; CHILDREN];
            let mut sum = 0.0f32;
            for (slot, value) in out.iter_mut().zip(row) {
                *slot = (value - max).exp();
                sum += *slot;
            }
            for slot in &mut out {
                *slot /= sum;
            }
            out
        })
        .collect()
}

impl OctreeDecoder for OctreeNet {
    #[expect(
        clippy::cast_precision_loss,
        reason = "cell coordinates and resolutions are below 2^8, exact in f32"
    )]
    fn child_probs(
        &self,
        coords: &[[u32; 3]],
        level: u32,
        latent: &[f32],
    ) -> Result<Vec<[f32; CHILDREN]>, String> {
        if level == 0 {
            return Err("the octree's first split is level 1".to_owned());
        }
        // The parents' centres at their own resolution, and the children's
        // resolution: the reference's `(coords + 0.5) / 2^(level - 1)` and
        // `2^level`.
        let parent_side = (1u32 << (level - 1)) as f32;
        let centres: Vec<f32> = coords
            .iter()
            .flat_map(|c| c.map(|v| (v as f32 + 0.5) / parent_side))
            .collect();
        let logits = self
            .logits(&centres, (1u32 << level) as f32, latent)
            .map_err(|e| e.to_string())?;
        Ok(softmax_rows(&logits))
    }
}

impl GaussianDecoder for GaussianNet {
    fn decode(&self, anchors: &[[f32; 3]], latent: &[f32]) -> Result<GaussianCloud, String> {
        let flat: Vec<f32> = anchors.iter().flatten().copied().collect();
        let features = self.features(&flat, latent).map_err(|e| e.to_string())?;
        activate(
            &RawOutput {
                features: &features,
                anchors,
                perturbation: &self.perturbation,
                base_offset_scale: self.base_offset_scale,
            },
            &RepresentationConfig::idol(),
        )
        .map_err(|e| e.to_string())
    }
}

/// The loaded model.
#[derive(Debug)]
pub struct IdolGenerator {
    id: String,
    ops: Arc<Ops>,
    conditioner: Box<dyn Conditioner>,
    denoiser: Box<dyn Denoiser>,
    octree: Box<dyn OctreeDecoder>,
    gaussian: Box<dyn GaussianDecoder>,
    matte: Option<Box<dyn Matting>>,
    limits: SplatLimits,
}

impl IdolGenerator {
    /// Opens every part on `backend`.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] for a part of another shape, and
    /// [`ImageGenError::Backend`] for a device without the kernel set.
    pub fn open(desc: &SplatModelDesc, backend: &Arc<dyn Backend>) -> Result<Self, ImageGenError> {
        let config = IdolConfig::read(&desc.model_dir)?;
        if config.image_size != IMAGE_SIZE as usize {
            return Err(ImageGenError::Checkpoint(format!(
                "the splat checkpoint reads {} pixel pictures; the preprocessing stage makes {}",
                config.image_size, IMAGE_SIZE
            )));
        }
        let ops = Arc::new(Ops::new(backend)?);
        let matte = match &desc.matte {
            Some(path) => Some(open_matte(path, &ops)?),
            None => None,
        };
        Ok(Self {
            id: desc.id.clone(),
            conditioner: Box::new(IdolConditioner::load(
                Arc::clone(&ops),
                &desc.encoder_dir,
                &desc.vae_dir,
            )?),
            denoiser: Box::new(IdolDenoiser::load(&desc.model_dir, Arc::clone(&ops))?),
            octree: Box::new(OctreeNet::load(&desc.model_dir, Arc::clone(&ops))?),
            gaussian: Box::new(GaussianNet::load(&desc.model_dir, Arc::clone(&ops))?),
            matte,
            limits: SplatLimits::default(),
            ops,
        })
    }

    /// The generation with its draws handed in: the seam the end-to-end case
    /// replays the reference's own draws through.
    pub(crate) fn generate_with(
        &self,
        job: &SplatJob,
        draws: &mut dyn Draws,
        progress: Option<&dyn Fn(SplatStepEvent)>,
    ) -> Result<GeneratedSplat, SplatGenError> {
        let matte = self.matte.as_deref();
        let prepared = preprocess(&job.image, job.crop_margin, matte).map_err(refuse)?;
        let side = IMAGE_SIZE as usize;
        let rgb = planar(&prepared.rgb);
        let cond = self
            .conditioner
            .condition(&rgb, side, draws)
            .map_err(SplatGenError::Backend)?;
        let generation = run(
            &Pipeline {
                denoiser: &*self.denoiser,
                octree: &*self.octree,
                gaussian: &*self.gaussian,
                config: PipelineConfig::idol(),
                steps: job.steps,
                rescale_t: job.rescale_t,
                cfg_strength: job.cfg_strength,
                num_gaussians: job.num_gaussians,
            },
            &cond,
            draws,
            progress,
        );
        self.ops.trim_pool();
        let generation = generation.map_err(|e| match e {
            PipelineError::AnchorBudget { count, .. } => SplatGenError::InvalidGaussians { count },
            PipelineError::Stage { .. } => SplatGenError::Backend(e.to_string()),
        })?;
        let bytes =
            write_ply(&generation.cloud).map_err(|e| SplatGenError::Encoding(e.to_string()))?;
        Ok(GeneratedSplat {
            bytes,
            media_type: MEDIA_TYPE,
            num_gaussians: job.num_gaussians,
            completion_tokens: SEQUENCE_TOKENS.saturating_mul(job.steps),
        })
    }
}

impl SplatGenerator for IdolGenerator {
    fn model_id(&self) -> &str {
        &self.id
    }

    fn limits(&self) -> SplatLimits {
        self.limits
    }

    fn generate(
        &self,
        job: &SplatJob,
        progress: Option<&dyn Fn(SplatStepEvent)>,
    ) -> Result<GeneratedSplat, SplatGenError> {
        job.validate()?;
        let mut draws = SeededDraws::new(job.seed);
        self.generate_with(job, &mut draws, progress)
    }
}

/// The preprocessing stage's refusals in the contract's terms.
fn refuse(error: PreprocessError) -> SplatGenError {
    match error {
        PreprocessError::Decode(reason) => SplatGenError::UndecodableImage(reason),
        PreprocessError::TooLarge { .. } => SplatGenError::UndecodableImage(error.to_string()),
        PreprocessError::EmptyAlpha => SplatGenError::EmptyAlpha,
        PreprocessError::NoMatte => SplatGenError::NoMatte,
        PreprocessError::Matting(reason) => SplatGenError::Matting(reason),
    }
}

/// An RGB picture as the planar `(3, H, W)` field in `[0, 1]` the encoders
/// read.
fn planar(image: &image::RgbImage) -> Vec<f32> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let mut out = vec![0.0f32; 3 * width * height];
    for (x, y, pixel) in image.enumerate_pixels() {
        for channel in 0..3 {
            out[(channel * height + y as usize) * width + x as usize] =
                f32::from(pixel[channel]) / 255.0;
        }
    }
    out
}

/// Opens the background remover.
fn open_matte(path: &Path, ops: &Arc<Ops>) -> Result<Box<dyn Matting>, ImageGenError> {
    Ok(Box::new(super::birefnet::BiRefNet::load(
        path,
        Arc::clone(ops),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_softmax_is_a_distribution_per_row() {
        let rows = softmax_rows(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].iter().all(|p| (p - 0.125).abs() < 1e-6));
        assert!(rows[1][0] > 0.99);
        assert!((rows[1].iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn preprocessing_refusals_keep_their_meaning() {
        assert!(matches!(
            refuse(PreprocessError::NoMatte),
            SplatGenError::NoMatte
        ));
        assert!(matches!(
            refuse(PreprocessError::EmptyAlpha),
            SplatGenError::EmptyAlpha
        ));
        assert!(matches!(
            refuse(PreprocessError::Decode("x".to_owned())),
            SplatGenError::UndecodableImage(_)
        ));
        assert!(matches!(
            refuse(PreprocessError::Matting("x".to_owned())),
            SplatGenError::Matting(_)
        ));
    }

    #[test]
    fn a_picture_becomes_planar_unit_values() {
        let mut image = image::RgbImage::new(2, 1);
        image.put_pixel(1, 0, image::Rgb([255, 0, 51]));
        let out = planar(&image);
        assert_eq!(out, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.2]);
    }
}
