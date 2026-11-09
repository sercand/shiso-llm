//! Turning an IR splat request into a validated job.

use shiso_core::ids::Seed;
use shiso_ir::splat::{Generate, Mode};
use shiso_vision::splat::{
    DEFAULT_CFG_STRENGTH, DEFAULT_RESCALE_T, SplatGenError, SplatJob, SplatMode,
};

use super::{SplatDefaults, refuse};
use crate::error::AppError;

/// Fills a request's omitted knobs from the model's defaults and the mode's
/// crop margin, then validates the job.
///
/// The seed is resolved by the caller rather than here: `request.seed` is the
/// request's named seed or `None`, and drawing entropy is the caller's job, so
/// this function stays deterministic for a test and a replay.
///
/// # Errors
/// [`AppError::BadRequest`] naming the first field that is out of range, or
/// `image` when the request carries no bytes.
pub fn build_job(
    request: Generate,
    defaults: SplatDefaults,
    seed: Seed,
) -> Result<SplatJob, AppError> {
    if request.image.is_empty() {
        return Err(refuse(SplatGenError::UndecodableImage(
            "the request carries no image bytes".to_owned(),
        )));
    }
    let mode = match request.mode {
        Mode::Synthetic => SplatMode::Synthetic,
        Mode::Real => SplatMode::Real,
    };
    let job = SplatJob {
        image: request.image,
        mode,
        seed,
        steps: request.steps.unwrap_or(defaults.steps),
        cfg_strength: request.cfg_strength.unwrap_or(DEFAULT_CFG_STRENGTH),
        rescale_t: request.rescale_t.unwrap_or(DEFAULT_RESCALE_T),
        num_gaussians: request.num_gaussians.unwrap_or(defaults.num_gaussians),
        crop_margin: request
            .crop_margin
            .unwrap_or_else(|| mode.default_crop_margin()),
    };
    job.validate().map_err(refuse)?;
    Ok(job)
}
