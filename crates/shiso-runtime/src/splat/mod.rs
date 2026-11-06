//! 3D splat generation: defaults, request translation, and the queueing
//! service.
//!
//! The shape is [`crate::image`]'s: a wire handler turns a request into a
//! validated [`SplatJob`], the service queues it on a bounded channel, and one
//! worker thread runs it on the device. What differs is the vocabulary - steps
//! and Gaussians instead of a size - and that a splat's cost is not bounded by
//! a per-request token count.

mod core;

pub use core::build_job;

use shiso_vision::splat::{
    GAUSSIANS_PER_ANCHOR, MAX_GAUSSIANS, MAX_STEPS, MIN_GAUSSIANS, SplatGenError, SplatLimits,
};

use crate::error::AppError;

/// What a splat request that names no step count or Gaussian count gets.
///
/// The server-side counterpart of the wire's optional `steps` and
/// `num_gaussians`: the loaded model's [`SplatLimits`] say what it was built
/// to run at, and the operator may move either value at boot through
/// [`SplatOverrides`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplatDefaults {
    /// Denoising steps.
    pub steps: u32,
    /// How many Gaussians to place.
    pub num_gaussians: u32,
}

impl SplatDefaults {
    /// What the model itself defaults to.
    #[must_use]
    pub const fn of(limits: &SplatLimits) -> Self {
        Self {
            steps: limits.default_steps,
            num_gaussians: limits.default_num_gaussians,
        }
    }
}

/// The operator's explicit choices, each `None` when the flag was not given.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SplatOverrides {
    /// `--splat-default-steps`.
    pub steps: Option<u32>,
    /// `--splat-default-num-gaussians`.
    pub num_gaussians: Option<u32>,
}

/// An operator override the loaded model cannot honour.
///
/// Refused at boot rather than clamped: a default that silently moved is not
/// the default that was configured, and one the model refuses would fail
/// every request that relied on it.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum OverrideError {
    /// The step count is outside what any splat model runs.
    #[error(
        "--splat-default-steps {steps} is not servable: steps must be between 1 and {MAX_STEPS}"
    )]
    Steps {
        /// Steps as configured.
        steps: u32,
    },
    /// The Gaussian count is outside the family's range or off its grid.
    #[error(
        "--splat-default-num-gaussians {count} is not servable: it must be a multiple of \
         {GAUSSIANS_PER_ANCHOR} between {MIN_GAUSSIANS} and {MAX_GAUSSIANS}"
    )]
    Gaussians {
        /// The count as configured.
        count: u32,
    },
}

impl SplatOverrides {
    /// The model's defaults with these overrides applied.
    ///
    /// # Errors
    ///
    /// [`OverrideError`] when an override is one the model would refuse.
    pub fn resolve(self, limits: SplatLimits) -> Result<SplatDefaults, OverrideError> {
        let mut defaults = SplatDefaults::of(&limits);
        if let Some(steps) = self.steps {
            if steps == 0 || steps > MAX_STEPS {
                return Err(OverrideError::Steps { steps });
            }
            defaults.steps = steps;
        }
        if let Some(count) = self.num_gaussians {
            if !(MIN_GAUSSIANS..=MAX_GAUSSIANS).contains(&count)
                || !count.is_multiple_of(GAUSSIANS_PER_ANCHOR)
            {
                return Err(OverrideError::Gaussians { count });
            }
            defaults.num_gaussians = count;
        }
        Ok(defaults)
    }
}

/// The HTTP failure a [`SplatGenError`] answers with.
///
/// The caller-side refusals (an undecodable picture, an alpha that matting or
/// erosion emptied, a knob outside its range) are the request's fault and
/// answer 400; the device-side failures answer 500 with the cause kept as a
/// real source for the log.
#[must_use]
pub fn refuse(error: SplatGenError) -> AppError {
    let message = error.to_string();
    match &error {
        SplatGenError::UndecodableImage(_) | SplatGenError::EmptyAlpha => {
            AppError::bad_request(message, Some("image"), "invalid_request")
        }
        SplatGenError::InvalidSteps { .. } => {
            AppError::bad_request(message, Some("steps"), "invalid_request")
        }
        SplatGenError::InvalidGaussians { .. } => {
            AppError::bad_request(message, Some("num_gaussians"), "invalid_request")
        }
        SplatGenError::InvalidCfg { .. } => {
            AppError::bad_request(message, Some("cfg_strength"), "invalid_request")
        }
        SplatGenError::InvalidRescale { .. } => {
            AppError::bad_request(message, Some("rescale_t"), "invalid_request")
        }
        SplatGenError::InvalidCropMargin { .. } => {
            AppError::bad_request(message, Some("crop_margin"), "invalid_request")
        }
        SplatGenError::Matting(_)
        | SplatGenError::Backend(_)
        | SplatGenError::Checkpoint(_)
        | SplatGenError::Encoding(_) => AppError::internal("splat generation failed", error),
    }
}

#[cfg(test)]
mod tests {
    use shiso_core::ids::Seed;
    use shiso_ir::splat::{Generate, Mode};
    use shiso_vision::splat::SplatLimits;

    use super::{OverrideError, SplatDefaults, SplatOverrides, core};
    use crate::error::AppError;

    fn request() -> Generate {
        Generate {
            image: vec![1],
            mode: Mode::Synthetic,
            seed: None,
            steps: None,
            cfg_strength: None,
            rescale_t: None,
            num_gaussians: None,
            crop_margin: None,
        }
    }

    #[test]
    fn an_omitted_field_takes_the_model_default() {
        let defaults = SplatDefaults::of(&SplatLimits::default());
        let job = core::build_job(
            Generate {
                mode: Mode::Real,
                seed: Some(9),
                ..request()
            },
            defaults,
            Seed::new(9),
        )
        .expect("defaults are valid");
        assert_eq!((job.steps, job.num_gaussians), (50, 262_144));
        assert!(
            (job.crop_margin - 1.7).abs() < f32::EPSILON,
            "real mode picks the loose crop"
        );
    }

    #[test]
    fn an_explicit_margin_beats_the_mode() {
        let job = core::build_job(
            Generate {
                mode: Mode::Real,
                seed: Some(1),
                crop_margin: Some(1.4),
                ..request()
            },
            SplatDefaults::of(&SplatLimits::default()),
            Seed::new(1),
        )
        .expect("valid");
        assert!((job.crop_margin - 1.4).abs() < f32::EPSILON);
    }

    #[test]
    fn an_override_the_model_would_refuse_fails_at_boot() {
        let over = SplatOverrides {
            steps: Some(101),
            num_gaussians: None,
        };
        assert!(matches!(
            over.resolve(SplatLimits::default()),
            Err(OverrideError::Steps { .. })
        ));
        let over = SplatOverrides {
            steps: None,
            num_gaussians: Some(32_769),
        };
        assert!(matches!(
            over.resolve(SplatLimits::default()),
            Err(OverrideError::Gaussians { .. })
        ));
    }

    #[test]
    fn an_empty_image_is_a_client_error_on_the_image_field() {
        let err = core::build_job(
            Generate {
                image: Vec::new(),
                ..request()
            },
            SplatDefaults::of(&SplatLimits::default()),
            Seed::new(1),
        )
        .expect_err("empty image is refused");
        match err {
            AppError::BadRequest {
                param: Some(field), ..
            } => assert_eq!(field, "image"),
            other => panic!("expected a field-scoped bad request, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_alpha_is_a_client_error_on_the_image_field() {
        let err = super::refuse(shiso_vision::splat::SplatGenError::EmptyAlpha);
        assert!(matches!(
            err,
            AppError::BadRequest {
                param: Some(ref field),
                ..
            } if field == "image"
        ));
    }
}
