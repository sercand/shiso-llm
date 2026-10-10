//! The Qwen-Image-2.1 sigma schedules: flow-matching Euler over a ladder of
//! sigmas from 1 down to a terminal 0, the latent stepped
//! `x += (sigma[i+1] - sigma[i]) * v` between rungs.
//!
//! The ladder is built from nodes `t` in (0, 1] ([`ScheduleSource`]): a
//! distilled checkpoint's own (`model_index.json` `sample_sigmas`, whose
//! count is the step count), else `linspace(1, 1/steps, steps)`. Either way
//! every node is time-shifted by `scheduler_config.json`, exactly as the
//! reference scheduler treats the sigmas a pipeline hands it, so a
//! fine-tune is described entirely by the two documents it ships:
//!
//! - With dynamic shifting the shift depends on the image's latent token
//!   count `seq`: `mu = base_shift + (max_shift - base_shift) / (max_seq -
//!   base_seq) * (seq - base_seq)` and `sigma = e^mu / (e^mu + (1/t - 1))`
//!   (the exponential time shift; the linear one is `mu / (mu + (1/t -
//!   1))`). Without it, the static `shift * t / (1 + (shift - 1) * t)`;
//!   a shift of 1 is the identity, so the official Turbo's sigmas run
//!   verbatim.
//! - A non-null `shift_terminal` then stretches the ladder so its last rung
//!   lands on that value:
//!   `sigma' = 1 - (1 - sigma) * (1 - shift_terminal) / (1 - sigma_last)`.
//!
//! The arithmetic runs in `f64` and each sigma is rounded to `f32` once;
//! the scheduler this mirrors computes in `f32`, which is at most a few
//! `f32` ulps away.

use serde::Deserialize;
use shiso_vision::generator::ImageGenError;

/// `scheduler_config.json`, the fields the ladder depends on.
#[derive(Debug, Clone, Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the document's own boolean switches, mirrored field for field so each one can be checked"
)]
pub(crate) struct ShiftConfig {
    /// Token count at which the dynamic shift is `base_shift`.
    pub base_image_seq_len: f64,
    /// Token count at which the dynamic shift is `max_shift`.
    pub max_image_seq_len: f64,
    /// `mu` at `base_image_seq_len`.
    pub base_shift: f64,
    /// `mu` at `max_image_seq_len`.
    pub max_shift: f64,
    /// The static shift, used when dynamic shifting is off.
    pub shift: f64,
    /// Where the stretched ladder's last rung lands, if anywhere.
    pub shift_terminal: Option<f64>,
    /// Whether `mu` follows the token count.
    pub use_dynamic_shifting: bool,
    /// The dynamic time-shift form, `exponential` or `linear`.
    pub time_shift_type: String,
    #[serde(default)]
    use_karras_sigmas: bool,
    #[serde(default)]
    use_exponential_sigmas: bool,
    #[serde(default)]
    use_beta_sigmas: bool,
    #[serde(default)]
    invert_sigmas: bool,
    #[serde(default)]
    stochastic_sampling: bool,
}

impl ShiftConfig {
    /// Refuses a scheduler option this ladder does not implement: each
    /// would change every sigma, so running without it is a different model.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] naming the option.
    pub(crate) fn validate(&self) -> Result<(), ImageGenError> {
        let unsupported = [
            ("use_karras_sigmas", self.use_karras_sigmas),
            ("use_exponential_sigmas", self.use_exponential_sigmas),
            ("use_beta_sigmas", self.use_beta_sigmas),
            ("invert_sigmas", self.invert_sigmas),
            ("stochastic_sampling", self.stochastic_sampling),
        ];
        if let Some((name, _)) = unsupported.iter().find(|(_, on)| *on) {
            return Err(ImageGenError::Checkpoint(format!(
                "scheduler option {name} is set; this scheduler does not implement it"
            )));
        }
        if self.use_dynamic_shifting
            && !matches!(self.time_shift_type.as_str(), "exponential" | "linear")
        {
            return Err(ImageGenError::Checkpoint(format!(
                "time_shift_type '{}' is not implemented (only 'exponential' and 'linear')",
                self.time_shift_type
            )));
        }
        if self.use_dynamic_shifting && self.max_image_seq_len <= self.base_image_seq_len {
            return Err(ImageGenError::Checkpoint(format!(
                "max_image_seq_len {} must exceed base_image_seq_len {}",
                self.max_image_seq_len, self.base_image_seq_len
            )));
        }
        if let Some(t) = self.shift_terminal
            && !(t > 0.0 && t < 1.0)
        {
            return Err(ImageGenError::Checkpoint(format!(
                "shift_terminal {t} must lie in (0, 1)"
            )));
        }
        if !(self.shift.is_finite() && self.shift > 0.0) {
            return Err(ImageGenError::Checkpoint(format!(
                "shift {} must be positive",
                self.shift
            )));
        }
        Ok(())
    }

    /// The dynamic shift's `mu` for `seq` latent tokens (not clamped: a
    /// larger image extrapolates the line, as the scheduler does).
    #[expect(
        clippy::cast_precision_loss,
        reason = "latent token counts are far below 2^52"
    )]
    pub(crate) fn mu(&self, seq: usize) -> f64 {
        let m =
            (self.max_shift - self.base_shift) / (self.max_image_seq_len - self.base_image_seq_len);
        let b = self.base_shift - m * self.base_image_seq_len;
        seq as f64 * m + b
    }
}

/// Where a job's nodes come from (see the module docs); both are shifted.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ScheduleSource<'a> {
    /// A checkpoint's own nodes (`sample_sigmas`); their count is the step
    /// count.
    Nodes(&'a [f32]),
    /// `linspace(1, 1/steps, steps)`.
    Linear,
}

/// Checks a checkpoint's nodes: non-empty, starting at 1 (pure noise, which
/// every shift keeps at 1), falling strictly within (0, 1]. A zero or a
/// repeat would be a step that does nothing, a rise one that adds noise.
///
/// # Errors
/// [`ImageGenError::Checkpoint`] naming the nodes.
pub(crate) fn check_nodes(nodes: &[f32]) -> Result<(), ImageGenError> {
    let in_range = |v: f32| v.is_finite() && v > 0.0 && v <= 1.0;
    if nodes
        .first()
        .is_none_or(|v| v.to_bits() != 1.0f32.to_bits())
        || nodes.iter().any(|&v| !in_range(v))
        || nodes.windows(2).any(|w| w[1] >= w[0])
    {
        return Err(ImageGenError::Checkpoint(format!(
            "sample_sigmas {nodes:?} must start at 1.0 and fall strictly within (0, 1]"
        )));
    }
    Ok(())
}

/// The sigma ladder for `steps` steps over an image of `seq` latent
/// tokens: `steps + 1` values from the first rung down to a terminal 0.
///
/// # Errors
/// [`ImageGenError::InvalidSteps`] for zero steps;
/// [`ImageGenError::StepsFixed`] when a node list has another length than
/// `steps`; [`ImageGenError::Checkpoint`] for nodes that are not a strictly
/// falling run from 1, or a linear shift whose `mu` is not positive.
#[expect(
    clippy::cast_possible_truncation,
    reason = "each sigma is rounded to the f32 the sampler steps in, once, on purpose"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "step counts are small, exact in f64"
)]
pub(crate) fn resolve(
    source: ScheduleSource<'_>,
    shift: &ShiftConfig,
    steps: usize,
    seq: usize,
) -> Result<Vec<f32>, ImageGenError> {
    if steps == 0 {
        return Err(ImageGenError::InvalidSteps { steps: 0 });
    }
    let nodes: Vec<f64> = match source {
        ScheduleSource::Nodes(nodes) => {
            if nodes.len() != steps {
                return Err(ImageGenError::StepsFixed {
                    steps: u32::try_from(steps).unwrap_or(u32::MAX),
                    fixed: u32::try_from(nodes.len()).unwrap_or(u32::MAX),
                });
            }
            check_nodes(nodes)?;
            nodes.iter().map(|&t| f64::from(t)).collect()
        }
        // `start + i * step` in f64, as a linspace forms it, then held as
        // the f32 the scheduler casts its nodes to.
        ScheduleSource::Linear => {
            let step = (1.0 / steps as f64 - 1.0) / (steps - 1).max(1) as f64;
            (0..steps)
                .map(|i| f64::from((1.0 + i as f64 * step) as f32))
                .collect()
        }
    };
    let mut sigmas: Vec<f64> = if shift.use_dynamic_shifting {
        let mu = shift.mu(seq);
        let scale = if shift.time_shift_type == "linear" {
            if mu.is_nan() || mu <= 0.0 {
                return Err(ImageGenError::Checkpoint(format!(
                    "a linear time shift needs a positive mu; {seq} latent tokens give {mu}"
                )));
            }
            mu
        } else {
            mu.exp()
        };
        nodes
            .iter()
            .map(|&t| scale / (scale + (1.0 / t - 1.0)))
            .collect()
    } else {
        // A shift of exactly 1 is `t / 1`, so shipped sigmas pass through
        // bit for bit.
        let s = shift.shift;
        nodes
            .iter()
            .map(|&t| s * t / (1.0 + (s - 1.0) * t))
            .collect()
    };
    if let Some(terminal) = shift.shift_terminal {
        let last = 1.0 - sigmas.last().copied().unwrap_or(0.0);
        let factor = last / (1.0 - terminal);
        for s in &mut sigmas {
            *s = 1.0 - (1.0 - *s) / factor;
        }
    }
    let mut out: Vec<f32> = sigmas.iter().map(|&s| s as f32).collect();
    out.push(0.0);
    Ok(out)
}

/// The Euler step's `dt` from rung `i` to rung `i + 1` (negative: sigma
/// falls), in the `f32` the latent is stepped in.
///
/// # Errors
/// [`ImageGenError::Backend`] when `i` is the terminal rung or past it.
pub(crate) fn euler_dt(sigmas: &[f32], i: usize) -> Result<f32, ImageGenError> {
    match (sigmas.get(i), sigmas.get(i + 1)) {
        (Some(&now), Some(&next)) => Ok(next - now),
        _ => Err(ImageGenError::Backend(format!(
            "no Euler step from rung {i} of a {}-rung ladder",
            sigmas.len()
        ))),
    }
}
