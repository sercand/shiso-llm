//! The image generation contract: generator trait, job description, and output types.
//!
//! This module is the narrow waist between the HTTP wire handlers, which turn
//! a request body into an [`ImageGenerationJob`], and the family implementations
//! in `shiso-models`, which answer [`ImageGenerator`]. Nothing here names a
//! model family: the job is a validated wish, the result is encoded bytes, and
//! the trait is the only door between them.

use std::fmt;
use std::str::FromStr;

use shiso_core::ids::Seed;

/// Supported image output encoding formats: PNG and JPEG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageOutputFormat {
    /// Portable Network Graphics (lossless).
    #[default]
    Png,
    /// JPEG (lossy raster).
    Jpeg,
}

impl ImageOutputFormat {
    /// MIME type string for HTTP headers and wire format responses.
    #[must_use]
    pub const fn mime_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
        }
    }
}

impl FromStr for ImageOutputFormat {
    type Err = ImageGenError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "png" => Ok(Self::Png),
            "jpeg" | "jpg" => Ok(Self::Jpeg),
            other => Err(ImageGenError::UnsupportedFormat(other.to_owned())),
        }
    }
}

/// Errors occurring during image generation job preparation or execution.
#[derive(Debug, thiserror::Error)]
pub enum ImageGenError {
    /// Prompt text is empty.
    #[error("prompt must not be empty")]
    EmptyPrompt,
    /// Requested width or height is outside allowed bounds or not divisible by 16.
    #[error(
        "invalid image dimensions {width}x{height}: must be between 256 and 2048 and divisible by 16"
    )]
    InvalidDimensions {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// Requested denoising step count is zero: a job with no steps has no
    /// answer to give, so it is the request's fault, not a device failure.
    #[error("invalid step count {steps}: at least one denoising step is required")]
    InvalidSteps {
        /// The step count as requested.
        steps: u32,
    },
    /// Format is not supported.
    #[error("unsupported image format {0:?}: only png and jpeg are supported")]
    UnsupportedFormat(String),
    /// Reference image decode or resolution error.
    #[error("reference image error: {0}")]
    ReferenceImage(String),
    /// Backend compute failure.
    #[error("compute backend failure: {0}")]
    Backend(String),
    /// Image encoding failure.
    #[error("image encoding failed: {0}")]
    Encoding(String),
    /// Model checkpoint loading failure.
    #[error("model loading failure: {0}")]
    Checkpoint(String),
}

/// Reference image for image-to-image or multi-reference editing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputReferenceImage {
    /// Raw image bytes (PNG, JPEG).
    pub bytes: Vec<u8>,
}

/// One image generation job to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageGenerationJob {
    /// Text description of the desired image.
    pub prompt: String,
    /// Output width in pixels (divisible by 16).
    pub width: u32,
    /// Output height in pixels (divisible by 16).
    pub height: u32,
    /// Number of denoising inference steps.
    pub steps: u32,
    /// Deterministic PRNG seed.
    pub seed: Seed,
    /// Requested encoding format (Png or Jpeg).
    pub output_format: ImageOutputFormat,
    /// Optional compression quality (0-100).
    pub output_compression: Option<u32>,
    /// Optional reference images for image-to-image or editing.
    pub input_references: Vec<InputReferenceImage>,
}

impl ImageGenerationJob {
    /// Validates prompt content, dimension constraints and the step count.
    ///
    /// # Errors
    /// Returns [`ImageGenError::EmptyPrompt`], [`ImageGenError::InvalidDimensions`]
    /// or [`ImageGenError::InvalidSteps`].
    pub fn validate(&self) -> Result<(), ImageGenError> {
        if self.prompt.trim().is_empty() {
            return Err(ImageGenError::EmptyPrompt);
        }
        if self.width < 256
            || self.width > 2048
            || self.height < 256
            || self.height > 2048
            || !self.width.is_multiple_of(16)
            || !self.height.is_multiple_of(16)
        {
            return Err(ImageGenError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }
        if self.steps == 0 {
            return Err(ImageGenError::InvalidSteps { steps: self.steps });
        }
        Ok(())
    }
}

/// Result of a completed image generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedImage {
    /// Encoded image file bytes.
    pub bytes: Vec<u8>,
    /// MIME type string ("image/png" or "image/jpeg").
    pub media_type: &'static str,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Estimated prompt tokens consumed.
    pub prompt_tokens: u32,
    /// Completion tokens accounted (e.g. latent patches * steps).
    pub completion_tokens: u32,
}

/// Progress event emitted during generation steps.
#[derive(Debug, Clone)]
pub struct ImageStepEvent {
    /// Step index (0-based).
    pub step: u32,
    /// Total inference steps.
    pub total_steps: u32,
    /// Optional partial image bytes if intermediate decode was requested.
    pub partial_image: Option<Vec<u8>>,
}

/// The image generator contract: models implement this trait.
pub trait ImageGenerator: Send + Sync + fmt::Debug {
    /// Stable model identifier reported to clients.
    fn model_id(&self) -> &str;

    /// Generates an image, optionally emitting step progress.
    ///
    /// The per-step partial decode is a full extra decode per step, so a
    /// generator pays that cost only when a progress consumer exists: a
    /// blocking caller has no consumer, and the worker hands it `None`.
    ///
    /// # Errors
    /// [`ImageGenError`] upon device error, invalid dimensions, or encoding failure.
    fn generate(
        &self,
        job: &ImageGenerationJob,
        progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError>;
}
