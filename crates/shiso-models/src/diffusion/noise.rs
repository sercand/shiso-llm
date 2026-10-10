//! The seeded Gaussian noise every family starts its latents from.
//!
//! Standard-normal values by Box-Muller, drawn from the job's seed: the same
//! seed gives the same image on any machine, because the stream and the
//! transform are both fixed here, not by a platform RNG.

use shiso_core::ids::Seed;

/// `count` standard-normal values from `seed`'s stream, in pairs (cosine
/// then sine of one draw), the last pair cut to fit.
pub(crate) fn gaussian(seed: Seed, count: usize) -> Vec<f32> {
    let mut rng = seed;
    let mut noise = Vec::with_capacity(count + 2);
    while noise.len() < count {
        let u1 = rng.next_f32().max(f32::MIN_POSITIVE);
        let u2 = rng.next_f32();
        let radius = (-2.0f32 * u1.ln()).sqrt();
        let angle = std::f32::consts::TAU * u2;
        noise.push(radius * angle.cos());
        noise.push(radius * angle.sin());
    }
    noise.truncate(count);
    noise
}
