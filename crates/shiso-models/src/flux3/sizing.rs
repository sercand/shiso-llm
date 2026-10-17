//! What a FLUX.3 job needs from the device, and whether it fits.
//!
//! The numbers are estimates from shapes, not measurements: they feed
//! [`plan`](crate::diffusion::residency::plan), which decides whether the
//! text encoder and the transformer stay loaded together or take turns, and
//! refuses a job whose largest phase fits no residency. They err high, so a
//! job is refused with a named largest side rather than failing an
//! allocation mid-run.

use shiso_vision::generator::ImageGenError;

use super::vae::{PIXELS_PER_TOKEN, Vae};
use crate::diffusion::residency::{Footprint, JobPeaks, PlanDesc, Residency, ResidencyMode, plan};

/// The smallest side the refusal's "largest square that fits" search tries.
const MIN_SIDE: usize = 64;
/// The largest side the search tries.
const MAX_SIDE: usize = 4096;

/// A job's geometry as the sizing sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JobShape {
    /// Output width in pixels.
    pub width: usize,
    /// Output height in pixels.
    pub height: usize,
    /// Padded text rows.
    pub text_rows: usize,
    /// Reference rows (latent tokens of every reference).
    pub cond_rows: usize,
}

/// Everything fixed at load that the residency decision reads.
#[derive(Debug, Clone)]
pub(crate) struct JobSizer {
    /// The operator's choice.
    pub mode: ResidencyMode,
    /// The device's working set.
    pub working_set: u64,
    /// Bytes left to the rest of the host.
    pub reserve: u64,
    /// Component sizes.
    pub footprint: Footprint,
    /// The text encoder's residual width.
    pub encoder_hidden: usize,
    /// The text encoder's MLP width.
    pub encoder_inter: usize,
    /// The transformer's hidden width.
    pub hidden: usize,
    /// The transformer's MLP hidden width.
    pub mlp_hidden: usize,
    /// The text conditioning's width.
    pub txt_in: usize,
    /// The autoencoder's widest full-resolution field, in channels.
    pub vae_channels: usize,
}

impl JobSizer {
    /// The job's activation peaks per phase.
    pub(crate) fn peaks(&self, job: &JobShape) -> JobPeaks {
        let as_u64 = |v: usize| u64::try_from(v).unwrap_or(u64::MAX);
        let image_rows = (job.width / PIXELS_PER_TOKEN) * (job.height / PIXELS_PER_TOKEN);
        let rows = image_rows + job.cond_rows + job.text_rows;
        // Per row of the joint sequence: a handful of hidden-wide streams
        // (x, q, k, v, attention out, modulated copies) and the MLP's
        // doubled input and output.
        let per_row = 10 * self.hidden + 4 * self.mlp_hidden;
        let denoise = as_u64(rows * per_row * 4)
            // The text conditioning as uploaded, before its projection.
            + as_u64(job.text_rows * self.txt_in * 4);
        let encode = as_u64(job.text_rows * (8 * self.encoder_hidden + 3 * self.encoder_inter) * 4);
        let decode = Vae::working_set_bytes(self.vae_channels, job.width * job.height)
            // A reference is encoded before the denoiser loads; its pixels are
            // bounded by its tokens (256 pixels each).
            .max(Vae::working_set_bytes(
                self.vae_channels,
                job.cond_rows * PIXELS_PER_TOKEN * PIXELS_PER_TOKEN,
            ));
        JobPeaks {
            encode,
            denoise,
            decode,
        }
    }

    fn desc(&self, job: &JobShape) -> PlanDesc {
        PlanDesc {
            mode: self.mode,
            working_set: self.working_set,
            reserve: self.reserve,
            footprint: self.footprint,
            peaks: self.peaks(job),
        }
    }

    /// The job's residency, or [`ImageGenError::TooLargeForMemory`] naming
    /// the largest square side that does fit.
    ///
    /// # Errors
    /// As above.
    pub(crate) fn plan(&self, job: &JobShape) -> Result<Residency, ImageGenError> {
        plan(&self.desc(job)).map_err(|short| {
            let largest = (MIN_SIDE..=MAX_SIDE)
                .rev()
                .step_by(PIXELS_PER_TOKEN)
                .find(|&side| {
                    plan(&self.desc(&JobShape {
                        width: side,
                        height: side,
                        ..*job
                    }))
                    .is_ok()
                })
                .unwrap_or(0);
            ImageGenError::TooLargeForMemory {
                width: u32::try_from(job.width).unwrap_or(u32::MAX),
                height: u32::try_from(job.height).unwrap_or(u32::MAX),
                needed_mib: short.needed.div_ceil(1 << 20),
                budget_mib: short.budget >> 20,
                largest_side: u32::try_from(largest).unwrap_or(0),
            }
        })
    }
}
