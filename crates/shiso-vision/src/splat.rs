//! The 3D splat generation contract: generator trait, job description, and output types.
//!
//! This is the same narrow waist [`crate::generator`] draws for images: the
//! wire handlers turn a request body into a [`SplatJob`], the family
//! implementations in `shiso-models` answer [`SplatGenerator`], and nothing
//! here names a model family; a file format is one of [`SplatFormat`]'s two
//! containers and nothing more.

use std::fmt;

use shiso_core::ids::Seed;

/// What the input picture is, which only chooses the default crop margin.
///
/// Deliberately not the wire's own mode enum: this crate must not depend on
/// the IR, so the request loses its wire vocabulary at the boundary, exactly
/// as an image request becomes an [`crate::generator::ImageGenerationJob`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplatMode {
    /// A render or a clean cut-out.
    #[default]
    Synthetic,
    /// A photograph with background.
    Real,
}

impl SplatMode {
    /// The crop margin this kind of picture takes when the request names none.
    #[must_use]
    pub const fn default_crop_margin(self) -> f32 {
        match self {
            Self::Synthetic => CROP_MARGIN_SYNTHETIC,
            Self::Real => CROP_MARGIN_REAL,
        }
    }
}

/// The file a finished splat is written as.
///
/// Both carry the same Gaussians. They differ in container and in what a
/// reader is expected to do with the values, which is the family's writer's
/// business; here a format is a name and a media type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplatFormat {
    /// The 3D Gaussian-splatting `.ply` every splat viewer reads.
    #[default]
    Ply,
    /// A binary glTF (`.glb`) carrying the Gaussians through the
    /// `KHR_gaussian_splatting` extension.
    Glb,
}

impl SplatFormat {
    /// The format's MIME type.
    #[must_use]
    pub const fn media_type(self) -> &'static str {
        match self {
            Self::Ply => "model/ply",
            Self::Glb => "model/gltf-binary",
        }
    }

    /// The format's name on the wire, which is also its file extension.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ply => "ply",
            Self::Glb => "glb",
        }
    }
}

/// Smallest accepted Gaussian count: 1024 anchors of 32.
pub const MIN_GAUSSIANS: u32 = 32_768;
/// Largest accepted Gaussian count.
pub const MAX_GAUSSIANS: u32 = 1_048_576;
/// Every anchor carries this many Gaussians, so a count is a multiple of it.
pub const GAUSSIANS_PER_ANCHOR: u32 = 32;
/// The count used when the request names none.
pub const DEFAULT_GAUSSIANS: u32 = 262_144;
/// The most Euler steps one request may ask for.
pub const MAX_STEPS: u32 = 100;
/// The step count used when the request names none.
pub const DEFAULT_STEPS: u32 = 50;
/// The guidance strength used when the request names none.
pub const DEFAULT_CFG_STRENGTH: f32 = 3.0;
/// The time-grid shift used when the request names none.
pub const DEFAULT_RESCALE_T: f32 = 3.0;
/// The refresh interval used when neither the request nor the operator names
/// one: every step runs in full, the reference's sampler.
pub const DEFAULT_REFRESH_EVERY: u32 = 1;
/// The longest refresh interval a request may ask for. Past it a held step
/// reads a guidance direction and conditioning keys several steps stale; four
/// is the longest interval that was measured.
pub const MAX_REFRESH_EVERY: u32 = 4;
/// Crop margin for a render or a clean cut-out.
pub const CROP_MARGIN_SYNTHETIC: f32 = 1.2;
/// Crop margin for a photograph with background.
pub const CROP_MARGIN_REAL: f32 = 1.7;

/// What one splat model accepts and defaults to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplatLimits {
    /// Steps used when the request names none.
    pub default_steps: u32,
    /// Gaussian count used when the request names none.
    pub default_num_gaussians: u32,
}

impl Default for SplatLimits {
    fn default() -> Self {
        Self {
            default_steps: DEFAULT_STEPS,
            default_num_gaussians: DEFAULT_GAUSSIANS,
        }
    }
}

/// One splat generation job to execute.
///
/// Every field is resolved: a request's omitted knob has already become the
/// operator's default, so a family implementation never has to guess.
#[derive(Debug, Clone, PartialEq)]
pub struct SplatJob {
    /// The input picture's encoded bytes (PNG or JPEG).
    pub image: Vec<u8>,
    /// What the picture is.
    pub mode: SplatMode,
    /// Deterministic PRNG seed.
    pub seed: Seed,
    /// Denoising steps.
    pub steps: u32,
    /// Classifier-free guidance strength.
    pub cfg_strength: f32,
    /// Time-grid shift towards high noise.
    pub rescale_t: f32,
    /// How many Gaussians to place.
    pub num_gaussians: u32,
    /// How much room the square crop leaves around the object.
    pub crop_margin: f32,
    /// The file the finished splat is written as.
    pub format: SplatFormat,
    /// Every `refresh_every`-th step runs in full, and so do the first two;
    /// the steps between hold the last full step's guidance direction and
    /// conditioning rows. One is the reference's sampler; more is faster and
    /// an approximation of it.
    pub refresh_every: u32,
}

impl SplatJob {
    /// Checks every knob against what any splat model can run.
    ///
    /// The bounds are the model family's, not a deployment's: a deployment
    /// chooses defaults, never a different accepted range.
    ///
    /// # Errors
    /// The first field that is out of range, in field order, as
    /// [`SplatGenError::InvalidSteps`], [`SplatGenError::InvalidGaussians`],
    /// [`SplatGenError::InvalidCfg`], [`SplatGenError::InvalidRescale`],
    /// [`SplatGenError::InvalidCropMargin`] or
    /// [`SplatGenError::InvalidRefresh`].
    pub fn validate(&self) -> Result<(), SplatGenError> {
        if self.steps == 0 || self.steps > MAX_STEPS {
            return Err(SplatGenError::InvalidSteps { steps: self.steps });
        }
        if self.num_gaussians < MIN_GAUSSIANS
            || self.num_gaussians > MAX_GAUSSIANS
            || !self.num_gaussians.is_multiple_of(GAUSSIANS_PER_ANCHOR)
        {
            return Err(SplatGenError::InvalidGaussians {
                count: self.num_gaussians,
            });
        }
        if !(self.cfg_strength.is_finite() && self.cfg_strength >= 0.0) {
            return Err(SplatGenError::InvalidCfg {
                value: self.cfg_strength,
            });
        }
        if !(self.rescale_t.is_finite() && self.rescale_t > 0.0) {
            return Err(SplatGenError::InvalidRescale {
                value: self.rescale_t,
            });
        }
        if !(self.crop_margin.is_finite() && self.crop_margin > 0.0) {
            return Err(SplatGenError::InvalidCropMargin {
                value: self.crop_margin,
            });
        }
        if self.refresh_every == 0 || self.refresh_every > MAX_REFRESH_EVERY {
            return Err(SplatGenError::InvalidRefresh {
                value: self.refresh_every,
            });
        }
        Ok(())
    }
}

/// Errors occurring during splat job preparation or execution.
#[derive(Debug, thiserror::Error)]
pub enum SplatGenError {
    /// The picture's bytes are not a decodable PNG or JPEG, or there are
    /// none at all.
    #[error("image could not be decoded: {0}")]
    UndecodableImage(String),
    /// The matte (or the erosion after it) left no opaque pixel to crop to.
    #[error("the image has no opaque pixels left after matting")]
    EmptyAlpha,
    /// The picture has no transparency to crop to and this server has no
    /// background remover loaded: the caller can send a cut-out instead.
    #[error(
        "the image has no transparent pixels and this server has no background remover \
         loaded; send an RGBA PNG with the background removed"
    )]
    NoMatte,
    /// The step count is zero or over the family's ceiling.
    #[error("steps must be between 1 and {MAX_STEPS}, got {steps}")]
    InvalidSteps {
        /// The step count as requested.
        steps: u32,
    },
    /// The Gaussian count is outside the family's range or off its grid.
    #[error(
        "num_gaussians must be a multiple of {GAUSSIANS_PER_ANCHOR} between \
         {MIN_GAUSSIANS} and {MAX_GAUSSIANS}, got {count}"
    )]
    InvalidGaussians {
        /// The count as requested.
        count: u32,
    },
    /// The guidance strength is negative or not a number.
    #[error("cfg_strength must be finite and at least 0, got {value}")]
    InvalidCfg {
        /// The value as requested.
        value: f32,
    },
    /// The time-grid shift is not positive or not a number.
    #[error("rescale_t must be finite and greater than 0, got {value}")]
    InvalidRescale {
        /// The value as requested.
        value: f32,
    },
    /// The crop margin is not positive or not a number.
    #[error("crop_margin must be finite and greater than 0, got {value}")]
    InvalidCropMargin {
        /// The value as requested.
        value: f32,
    },
    /// The refresh interval is zero or over the family's ceiling.
    #[error("refresh_every must be between 1 and {MAX_REFRESH_EVERY}, got {value}")]
    InvalidRefresh {
        /// The interval as requested.
        value: u32,
    },
    /// Background removal failed on the device.
    #[error("background removal failed: {0}")]
    Matting(String),
    /// Backend compute failure.
    #[error("compute backend failure: {0}")]
    Backend(String),
    /// Model checkpoint loading failure.
    #[error("model loading failure: {0}")]
    Checkpoint(String),
    /// Encoding the finished splat as its file failed.
    #[error("splat encoding failed: {0}")]
    Encoding(String),
}

/// Result of a completed splat generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedSplat {
    /// The encoded file's bytes, in the job's [`SplatFormat`].
    pub bytes: Vec<u8>,
    /// The file's MIME type: the format's [`SplatFormat::media_type`].
    pub media_type: &'static str,
    /// How many Gaussians the file holds.
    pub num_gaussians: u32,
    /// Completion tokens accounted: denoiser sequence length times steps.
    pub completion_tokens: u32,
}

/// Progress event emitted during generation steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplatStepEvent {
    /// Step index (0-based).
    pub step: u32,
    /// Total inference steps.
    pub total_steps: u32,
}

/// The splat generator contract: models implement this trait.
pub trait SplatGenerator: Send + Sync + fmt::Debug {
    /// Stable model identifier reported to clients.
    fn model_id(&self) -> &str;

    /// What this model accepts and defaults to.
    fn limits(&self) -> SplatLimits;

    /// Generates a splat, optionally emitting step progress.
    ///
    /// The job has already been validated; a generator still refuses a job
    /// its own checkpoint cannot serve rather than substituting a default.
    ///
    /// # Errors
    /// [`SplatGenError`] upon an undecodable picture, a device failure, or an
    /// encoding failure.
    fn generate(
        &self,
        job: &SplatJob,
        progress: Option<&dyn Fn(SplatStepEvent)>,
    ) -> Result<GeneratedSplat, SplatGenError>;
}
