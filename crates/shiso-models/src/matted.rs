//! The background post-processing step every image generator runs behind.
//!
//! [`MattedGenerator`] wraps whichever family loaded and owns the job's
//! [`Background`]: it asks the family for a lossless PNG, decodes it, decides
//! the background, and encodes the format the job asked for. A family never
//! learns that a background was asked for, so a new family gets transparent,
//! opaque and automatic backgrounds without a line of its own.
//!
//! # The rules
//!
//! The model's own alpha wins whenever it drew a cut-out ([`is_transparent`]):
//! its matte is cleaner than anything inferred afterwards, and the colour it
//! leaves under its transparent pixels is arbitrary (a flat purple on
//! Qwen-Image-2.1), so the remover must never see or keep it.
//!
//! | asked | model drew a cut-out | answer |
//! |---|---|---|
//! | auto | yes | its alpha (JPEG: over white) |
//! | auto, opaque | no / any | over white, RGB |
//! | transparent | yes | its alpha, PNG; the remover does not run |
//! | transparent | no | the remover's matte of the picture over white, PNG |
//!
//! Every path that loses alpha composites over white first and never simply
//! drops the channel. Previews pass through unmatted: matting each one would
//! multiply the remover's cost by the step count.

use std::sync::Arc;

use shiso_vision::generator::{
    Background, GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageLimits,
    ImageOutputFormat, ImageStepEvent, InputReferenceImage,
};
use shiso_vision::matte::BackgroundRemover;

use crate::background::is_transparent;
use crate::diffusion::encode::{Channels, composite_over_white, encode_image};

/// An image generator with the background post-processing step in front.
#[derive(Debug)]
pub struct MattedGenerator {
    inner: Arc<dyn ImageGenerator>,
    remover: Option<Arc<dyn BackgroundRemover>>,
}

impl MattedGenerator {
    /// Wraps `inner`. Without a `remover`, a transparent background is still
    /// answered whenever the model draws a cut-out itself, and refused
    /// otherwise.
    #[must_use]
    pub fn new(
        inner: Arc<dyn ImageGenerator>,
        remover: Option<Arc<dyn BackgroundRemover>>,
    ) -> Self {
        Self { inner, remover }
    }

    /// Whether a background remover is loaded.
    #[must_use]
    pub fn has_remover(&self) -> bool {
        self.remover.is_some()
    }

    /// The remover's matte of `rgb` (`width` x `height`, already over white),
    /// as interleaved RGBA8.
    fn matte(&self, rgb: Vec<u8>, width: u32, height: u32) -> Result<Vec<u8>, ImageGenError> {
        let remover = self
            .remover
            .as_ref()
            .ok_or(ImageGenError::NoBackgroundRemover)?;
        let picture = image::RgbImage::from_raw(width, height, rgb).ok_or_else(|| {
            ImageGenError::BackgroundRemoval(format!(
                "a {width}x{height} picture does not match its pixel buffer"
            ))
        })?;
        let alpha = remover
            .alpha(&picture)
            .map_err(ImageGenError::BackgroundRemoval)?;
        if alpha.len() != picture.pixels().len() {
            return Err(ImageGenError::BackgroundRemoval(format!(
                "the remover answered {} alpha values for {width}x{height} pixels",
                alpha.len()
            )));
        }
        Ok(picture
            .pixels()
            .zip(alpha)
            .flat_map(|(p, a)| [p[0], p[1], p[2], a])
            .collect())
    }
}

impl ImageGenerator for MattedGenerator {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn limits(&self) -> ImageLimits {
        self.inner.limits()
    }

    fn max_references(&self) -> usize {
        self.inner.max_references()
    }

    fn removes_backgrounds(&self) -> bool {
        self.has_remover()
    }

    fn default_size_for_references(
        &self,
        references: &[InputReferenceImage],
    ) -> Result<(u32, u32), ImageGenError> {
        self.inner.default_size_for_references(references)
    }

    fn generate(
        &self,
        job: &ImageGenerationJob,
        progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        let lossless = ImageGenerationJob {
            output_format: ImageOutputFormat::Png,
            output_compression: None,
            ..job.clone()
        };
        let generated = self.inner.generate(&lossless, progress)?;
        let picture = image::load_from_memory(&generated.bytes)
            .map_err(|e| ImageGenError::Encoding(format!("reading the generated picture: {e}")))?
            .to_rgba8();
        let (width, height) = picture.dimensions();
        let rgba = picture.into_raw();
        let format = job.output_format;
        let compression = job.output_compression;
        let cut_out = is_transparent(&rgba);
        let (bytes, transparent) = match (job.background, cut_out) {
            (Background::Transparent, _) if format != ImageOutputFormat::Png => {
                return Err(ImageGenError::Encoding(
                    "a transparent background needs png output".to_owned(),
                ));
            }
            (Background::Auto | Background::Transparent, true) => {
                let bytes =
                    encode_image(&rgba, width, height, Channels::Rgba, format, compression)?;
                (bytes, format == ImageOutputFormat::Png)
            }
            (Background::Transparent, false) => {
                let matted = self.matte(composite_over_white(&rgba), width, height)?;
                let transparent = is_transparent(&matted);
                let bytes = encode_image(&matted, width, height, Channels::Rgba, format, None)?;
                (bytes, transparent)
            }
            (Background::Auto | Background::Opaque, _) => {
                let rgb = composite_over_white(&rgba);
                let bytes = encode_image(&rgb, width, height, Channels::Rgb, format, compression)?;
                (bytes, false)
            }
        };
        Ok(GeneratedImage {
            bytes,
            media_type: format.mime_type(),
            transparent,
            ..generated
        })
    }
}
