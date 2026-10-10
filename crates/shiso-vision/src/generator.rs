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

/// What one image model accepts and defaults to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageLimits {
    /// Width and height must be multiples of this.
    pub size_multiple: u32,
    /// Smallest accepted side, in pixels.
    pub min_side: u32,
    /// Largest accepted side, in pixels.
    pub max_side: u32,
    /// Size used when the request names none.
    pub default_size: (u32, u32),
    /// Steps used when the request names none.
    pub default_steps: u32,
    /// `Some(n)` when the checkpoint's schedule is fixed: any other step count is refused.
    pub fixed_steps: Option<u32>,
}

/// Errors occurring during image generation job preparation or execution.
#[derive(Debug, thiserror::Error)]
pub enum ImageGenError {
    /// Prompt text is empty.
    #[error("prompt must not be empty")]
    EmptyPrompt,
    /// The prompt encodes to more tokens than the model's text encoder
    /// reads. Refused rather than truncated: a cut prompt is a different
    /// request than the one the caller sent.
    #[error("prompt is {tokens} tokens, over this model's {max}-token limit")]
    PromptTooLong {
        /// Prompt tokens the request encodes to.
        tokens: u32,
        /// The most prompt tokens the model reads.
        max: u32,
    },
    /// Requested width or height is outside the model's bounds or off its grid.
    #[error(
        "width and height must be multiples of {multiple} between {min} and {max}, got {width}x{height}"
    )]
    InvalidDimensions {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// The grid the model's sides must sit on.
        multiple: u32,
        /// The model's smallest side.
        min: u32,
        /// The model's largest side.
        max: u32,
    },
    /// Requested denoising step count is zero: a job with no steps has no
    /// answer to give, so it is the request's fault, not a device failure.
    #[error("invalid step count {steps}: at least one denoising step is required")]
    InvalidSteps {
        /// The step count as requested.
        steps: u32,
    },
    /// The checkpoint's schedule is fixed and the request named another count:
    /// a distilled schedule run at a different length is a different, worse
    /// model, so it is refused rather than silently rewritten.
    #[error("this model's schedule is fixed at {fixed} steps, got {steps}")]
    StepsFixed {
        /// The step count as requested.
        steps: u32,
        /// The only count the model runs.
        fixed: u32,
    },
    /// The requested size's estimated decode working set is over the memory
    /// budget the generator runs under: the image is too large for this
    /// host, which is the request's to change, not a device failure.
    #[error(
        "a {width}x{height} image needs about {needed_mib} MiB of decoder working memory, \
         over this host's {budget_mib} MiB budget"
    )]
    DecodeTooLarge {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// The decoder's estimated working set, in MiB (rounded up).
        needed_mib: u64,
        /// The budget it was refused against, in MiB (rounded down).
        budget_mib: u64,
    },
    /// The job's peak memory - the weights it needs loaded at once plus its
    /// activations at this size - is over the host's working set under every
    /// residency the generator may use. The request's to change: a smaller
    /// image fits.
    #[error(
        "a {width}x{height} image needs about {needed_mib} MiB at its peak, over this \
         host's {budget_mib} MiB budget; the largest square image that fits is \
         {largest_side}x{largest_side}"
    )]
    TooLargeForMemory {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// The job's estimated peak, in MiB (rounded up).
        needed_mib: u64,
        /// The budget it was refused against, in MiB (rounded down).
        budget_mib: u64,
        /// The largest square side that fits (0 when none does).
        largest_side: u32,
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
    /// Output width in pixels (a multiple of the model's grid).
    pub width: u32,
    /// Output height in pixels (a multiple of the model's grid).
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
    /// Validates prompt content, the dimensions against the model's grid and
    /// bounds, and the step count against its schedule.
    ///
    /// # Errors
    /// Returns [`ImageGenError::EmptyPrompt`], [`ImageGenError::InvalidDimensions`],
    /// [`ImageGenError::InvalidSteps`] or [`ImageGenError::StepsFixed`].
    pub fn validate(&self, limits: &ImageLimits) -> Result<(), ImageGenError> {
        if self.prompt.trim().is_empty() {
            return Err(ImageGenError::EmptyPrompt);
        }
        let side_ok = |side: u32| {
            (limits.min_side..=limits.max_side).contains(&side)
                && side.is_multiple_of(limits.size_multiple)
        };
        if !side_ok(self.width) || !side_ok(self.height) {
            return Err(ImageGenError::InvalidDimensions {
                width: self.width,
                height: self.height,
                multiple: limits.size_multiple,
                min: limits.min_side,
                max: limits.max_side,
            });
        }
        if self.steps == 0 {
            return Err(ImageGenError::InvalidSteps { steps: self.steps });
        }
        if let Some(fixed) = limits.fixed_steps
            && self.steps != fixed
        {
            return Err(ImageGenError::StepsFixed {
                steps: self.steps,
                fixed,
            });
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

    /// What this model accepts and defaults to.
    fn limits(&self) -> ImageLimits;

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
