//! Where the pipeline's random numbers come from.
//!
//! A generation draws, in this order: the `VAE` posterior's noise, the
//! latent's and the camera's noise, one offset per octree cell per level, and
//! three jitters per anchor. The order is load-bearing - the same seed must
//! give the same object - so every stage takes its draws from one [`Draws`]
//! handed down the pipeline rather than from a generator of its own.
//!
//! Production draws from the job's [`Seed`]. A test replays the reference's
//! own recorded draws instead, which is how the end-to-end case compares
//! against a pipeline whose random generator this port does not reproduce.

use std::fmt;

use shiso_core::ids::Seed;

/// A source of the pipeline's random numbers.
pub trait Draws: Send + fmt::Debug {
    /// `count` standard-normal values.
    fn normal(&mut self, count: usize) -> Vec<f32>;
    /// One uniform value in `[0, 1)`.
    fn uniform(&mut self) -> f32;
}

/// Draws from a job's seed: uniforms from the seed's stream, normals by
/// Box-Muller over pairs of them.
#[derive(Debug, Clone)]
pub struct SeededDraws {
    rng: Seed,
}

impl SeededDraws {
    /// Draws from `seed`'s stream.
    #[must_use]
    pub const fn new(seed: Seed) -> Self {
        Self { rng: seed }
    }
}

impl Draws for SeededDraws {
    fn normal(&mut self, count: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(count + 1);
        while out.len() < count {
            // `max` keeps the logarithm finite on the one draw of exactly 0.
            let u1 = self.rng.next_f32().max(f32::MIN_POSITIVE);
            let u2 = self.rng.next_f32();
            let radius = (-2.0f32 * u1.ln()).sqrt();
            let angle = std::f32::consts::TAU * u2;
            out.push(radius * angle.cos());
            out.push(radius * angle.sin());
        }
        out.truncate(count);
        out
    }

    fn uniform(&mut self) -> f32 {
        self.rng.next_f32()
    }
}

/// Replays recorded draws, in the order they were recorded.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct Replay {
    normals: std::collections::VecDeque<f32>,
    uniforms: std::collections::VecDeque<f32>,
}

#[cfg(test)]
impl Replay {
    /// Queues normals and uniforms to hand out.
    pub(crate) fn new(normals: Vec<f32>, uniforms: Vec<f32>) -> Self {
        Self {
            normals: normals.into(),
            uniforms: uniforms.into(),
        }
    }

    /// What is left of each queue, so a case can prove every draw was used.
    pub(crate) fn remaining(&self) -> (usize, usize) {
        (self.normals.len(), self.uniforms.len())
    }
}

#[cfg(test)]
impl Draws for Replay {
    fn normal(&mut self, count: usize) -> Vec<f32> {
        assert!(
            self.normals.len() >= count,
            "{count} normals asked of a replay holding {}",
            self.normals.len()
        );
        self.normals.drain(..count).collect()
    }

    fn uniform(&mut self) -> f32 {
        self.uniforms
            .pop_front()
            .expect("a uniform asked of an exhausted replay")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a sample count of a few thousand is exact in f64"
    )]
    fn seeded_normals_have_unit_variance_and_zero_mean() {
        let values = SeededDraws::new(Seed::new(42)).normal(20_000);
        let n = values.len() as f64;
        let mean = values.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
        let var = values
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
        assert!(
            values.iter().any(|v| *v < -1.0),
            "a normal draw goes negative"
        );
    }

    #[test]
    fn the_same_seed_draws_the_same_stream() {
        let mut a = SeededDraws::new(Seed::new(9));
        let mut b = SeededDraws::new(Seed::new(9));
        assert_eq!(a.normal(5), b.normal(5));
        assert_eq!(a.uniform().to_bits(), b.uniform().to_bits());
    }

    #[test]
    fn an_odd_count_is_cut_to_size() {
        assert_eq!(SeededDraws::new(Seed::new(1)).normal(3).len(), 3);
    }

    #[test]
    fn a_replay_hands_back_its_queues_in_order() {
        let mut replay = Replay::new(vec![1.0, 2.0, 3.0], vec![0.25]);
        assert_eq!(replay.normal(2), vec![1.0, 2.0]);
        assert!((replay.uniform() - 0.25).abs() < f32::EPSILON);
        assert_eq!(replay.remaining(), (1, 0));
    }
}
