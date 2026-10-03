//! Flow-matching Euler discrete scheduler with empirical mu time shifting.

/// Computes empirical mu shift factor based on image sequence length and steps.
///
/// The two linear segments and the interpolation between them are the
/// distilled checkpoint's own time-shift fit: the shift is a function of how
/// many latent tokens the image occupies and how few steps remain to spend.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "sequence lengths and step counts are small scheduling inputs, exact in f32 well beyond any model's range"
)]
pub(crate) fn compute_empirical_mu(image_seq_len: usize, num_steps: usize) -> f32 {
    let a1 = 8.738_095e-5_f32;
    let b1 = 1.898_333_3_f32;
    let a2 = 0.000_169_27_f32;
    let b2 = 0.456_666_66_f32;

    let len_f = image_seq_len as f32;
    if image_seq_len > 4300 {
        return a2 * len_f + b2;
    }
    let m_200 = a2 * len_f + b2;
    let m_10 = a1 * len_f + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * (num_steps as f32) + b
}

/// The flow-matching Euler discrete scheduler.
#[derive(Debug, Clone)]
pub(crate) struct FlowMatchEulerDiscreteScheduler {
    /// Timesteps in descending order (e.g. sigma * 1000.0).
    pub timesteps: Vec<f32>,
    /// Shifted sigmas.
    pub sigmas: Vec<f32>,
}

impl FlowMatchEulerDiscreteScheduler {
    /// Creates a scheduler for `image_seq_len` tokens across `num_inference_steps`.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "step indices are small inference-step counts, exact in f32"
    )]
    pub(crate) fn new(image_seq_len: usize, num_inference_steps: usize) -> Self {
        let mu = compute_empirical_mu(image_seq_len, num_inference_steps);
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
