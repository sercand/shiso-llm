//! Flow-matching Euler discrete stepping, independent of any family's shift.
//!
//! A family decides its time-shift parameter `mu` (FLUX.2 fits it to the image
//! token count and step count; another family may use a constant); this turns
//! that `mu` into the sigma ladder the sampling loop walks.

/// The flow-matching Euler discrete scheduler.
#[derive(Debug, Clone)]
pub(crate) struct FlowMatchEulerDiscreteScheduler {
    /// Timesteps in descending order (e.g. sigma * 1000.0).
    pub timesteps: Vec<f32>,
    /// Shifted sigmas.
    pub sigmas: Vec<f32>,
}

impl FlowMatchEulerDiscreteScheduler {
    /// Creates a scheduler for `num_inference_steps` with the exponential time
    /// shift `mu` applied to the linear sigma ladder.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "step indices are small inference-step counts, exact in f32"
    )]
    pub(crate) fn with_mu(mu: f32, num_inference_steps: usize) -> Self {
        let mut sigmas = Vec::with_capacity(num_inference_steps + 1);
        for i in 0..=num_inference_steps {
            let s = 1.0 - (i as f32) / (num_inference_steps as f32);
            let shifted = if s <= 0.0 {
                0.0
            } else if s >= 1.0 {
                1.0
            } else {
                let exp_mu = mu.exp();
                (exp_mu * s) / (exp_mu * s + (1.0 - s))
            };
            sigmas.push(shifted);
        }

        let mut timesteps = Vec::with_capacity(num_inference_steps);
        for &s in sigmas.iter().take(num_inference_steps) {
            timesteps.push(s * 1000.0);
        }

        Self { timesteps, sigmas }
    }
}
