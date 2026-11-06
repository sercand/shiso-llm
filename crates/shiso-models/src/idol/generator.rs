//! The generation pipeline: noise in, a cloud out.
//!
//! The three device graphs (denoiser, octree, Gaussian decoder) sit behind
//! traits so the orchestration - the RNG order, the time grid, the guided
//! velocity, the anchor budget - can be tested without a device. That
//! orchestration is where a subtle error is most expensive: a seed consumed in
//! the wrong order still produces a plausible object, and the only way to
//! notice is a reference that agrees on every stage.

use std::fmt;

use shiso_core::ids::Seed;
use shiso_vision::splat::SplatStepEvent;

use super::conditioning::Conditioning;
use super::octree::{CHILDREN, place_anchors};
use super::ply::GaussianCloud;
use super::sampler::{TIMESTEP_SCALE, euler_step, guided_velocity, time_grid};

/// The denoiser's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineConfig {
    /// The latent's token count.
    pub q_token_length: usize,
    /// The latent's width.
    pub in_channels: usize,
    /// The camera's width. Drawn and carried, unused by this milestone's graph.
    pub cam_channels: usize,
    /// The octree's deepest level.
    pub max_level: u32,
    /// Gaussians per anchor.
    pub gaussians_per_anchor: u32,
}

impl PipelineConfig {
    /// The shipped checkpoint's shape.
    #[must_use]
    pub const fn idol() -> Self {
        Self {
            q_token_length: 8192,
            in_channels: 16,
            cam_channels: 5,
            max_level: 8,
            gaussians_per_anchor: 32,
        }
    }

    /// The latent's length in values.
    #[must_use]
    pub const fn latent_len(&self) -> usize {
        self.q_token_length * self.in_channels
    }
}

/// The denoiser: one velocity evaluation.
pub trait Denoiser: Send + Sync + fmt::Debug {
    /// The velocity at `latent` for `timestep` under `cond`.
    ///
    /// # Errors
    ///
    /// A device or checkpoint failure.
    fn velocity(
        &self,
        latent: &[f32],
        timestep: f32,
        cond: &Conditioning,
    ) -> Result<Vec<f32>, String>;
}

/// The octree decoder: child probabilities for one level's cells.
pub trait OctreeDecoder: Send + Sync + fmt::Debug {
    /// The child probabilities for every cell, in the order given.
    ///
    /// # Errors
    ///
    /// A device or checkpoint failure.
    fn child_probs(
        &self,
        coords: &[[u32; 3]],
        level: u32,
        latent: &[f32],
    ) -> Result<Vec<[f32; CHILDREN]>, String>;
}

/// The Gaussian decoder: anchors to a cloud.
pub trait GaussianDecoder: Send + Sync + fmt::Debug {
    /// The cloud for `anchors` under `latent`.
    ///
    /// # Errors
    ///
    /// A device or checkpoint failure.
    fn decode(&self, anchors: &[[f32; 3]], latent: &[f32]) -> Result<GaussianCloud, String>;
}

/// Everything one generation needs.
#[derive(Debug)]
pub struct Pipeline<'a> {
    /// The denoiser.
    pub denoiser: &'a dyn Denoiser,
    /// The octree decoder.
    pub octree: &'a dyn OctreeDecoder,
    /// The Gaussian decoder.
    pub gaussian: &'a dyn GaussianDecoder,
    /// The request's shape.
    pub config: PipelineConfig,
    /// Euler steps.
    pub steps: u32,
    /// The time grid's shift.
    pub rescale_t: f32,
    /// The guidance strength.
    pub cfg_strength: f32,
    /// How many Gaussians to place.
    pub num_gaussians: u32,
}

/// A pipeline refusal.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// The Gaussian count is not a whole number of anchors.
    #[error("{count} Gaussians is not a multiple of {per_anchor} per anchor")]
    AnchorBudget {
        /// The count as requested.
        count: u32,
        /// Gaussians per anchor.
        per_anchor: u32,
    },
    /// A device stage failed.
    #[error("{stage} failed: {reason}")]
    Stage {
        /// Which stage.
        stage: &'static str,
        /// The device's reason.
        reason: String,
    },
}

/// Runs the pipeline and answers the cloud.
///
/// The seed's draws happen in the reference's order: the latent noise, then the
/// camera, then the octree's offsets, then the leaf jitter. A caller that
/// reorders them gets a different object from the same seed, which is why the
/// order is stated here rather than left to each stage.
///
/// `progress` is called once per finished Euler step.
///
/// # Errors
///
/// [`PipelineError::AnchorBudget`] when the count is not a whole number of
/// anchors, and [`PipelineError::Stage`] when a device stage fails.
pub fn run(
    pipeline: &Pipeline<'_>,
    cond: &Conditioning,
    seed: Seed,
    progress: Option<&dyn Fn(SplatStepEvent)>,
) -> Result<GaussianCloud, PipelineError> {
    let per_anchor = pipeline.config.gaussians_per_anchor;
    if per_anchor == 0 || !pipeline.num_gaussians.is_multiple_of(per_anchor) {
        return Err(PipelineError::AnchorBudget {
            count: pipeline.num_gaussians,
            per_anchor,
        });
    }
    let mut rng = seed;
    let mut latent: Vec<f32> = (0..pipeline.config.latent_len())
        .map(|_| rng.next_f32())
        .collect();
    // The camera is drawn here so the stream matches the reference even though
    // this milestone's denoiser does not read it yet.
    let _camera: Vec<f32> = (0..pipeline.config.cam_channels)
        .map(|_| rng.next_f32())
        .collect();

    let grid = time_grid(pipeline.steps, pipeline.rescale_t);
    let negative = Conditioning::zero(cond.rows);
    let mut velocity = vec![0.0f32; latent.len()];
    let mut conditional = vec![0.0f32; latent.len()];
    let mut unconditional = vec![0.0f32; latent.len()];
    for (index, pair) in grid.windows(2).enumerate() {
        let (t, t_prev) = (pair[0], pair[1]);
        conditional = pipeline
            .denoiser
            .velocity(&latent, t * TIMESTEP_SCALE, cond)
            .map_err(|reason| PipelineError::Stage {
                stage: "the denoiser",
                reason,
            })?;
        if pipeline.cfg_strength > 0.0 {
            unconditional = pipeline
                .denoiser
                .velocity(&latent, t * TIMESTEP_SCALE, &negative)
                .map_err(|reason| PipelineError::Stage {
                    stage: "the denoiser",
                    reason,
                })?;
        } else {
            unconditional.clear();
            unconditional.resize(latent.len(), 0.0);
        }
        guided_velocity(
            &conditional,
            &unconditional,
            pipeline.cfg_strength,
            &mut velocity,
        );
        euler_step(&mut latent, &velocity, t, t_prev);
        if let Some(progress) = progress {
            // The step index is bounded by the request's 1..=100 step ceiling.
            let step = u32::try_from(index).unwrap_or(u32::MAX);
            progress(SplatStepEvent {
                step,
                total_steps: pipeline.steps,
            });
        }
    }

    let anchors_wanted = pipeline.num_gaussians / per_anchor;
    let mut probs_at = |coords: &[[u32; 3]], level: u32| {
        pipeline
            .octree
            .child_probs(coords, level, &latent)
            .unwrap_or_else(|_reason| vec![[0.125f32; CHILDREN]; coords.len()])
    };
    let anchors = place_anchors(
        anchors_wanted,
        pipeline.config.max_level,
        &mut rng,
        &mut probs_at,
    );
    let mut cloud = pipeline
        .gaussian
        .decode(&anchors, &latent)
        .map_err(|reason| PipelineError::Stage {
            stage: "the Gaussian decoder",
            reason,
        })?;
    // The request counts Gaussians; the decoder answers one group per anchor.
    cloud.truncate(pipeline.num_gaussians as usize);
    Ok(cloud)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idol::ply::{GaussianCloud, write_ply};

    /// A denoiser that answers a constant velocity and records the timesteps it
    /// was handed, so a test can prove the `t * 1000` bridge.
    #[derive(Debug, Default)]
    struct Constant {
        timesteps: std::sync::Mutex<Vec<f32>>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl Denoiser for Constant {
        fn velocity(
            &self,
            latent: &[f32],
            timestep: f32,
            _cond: &Conditioning,
        ) -> Result<Vec<f32>, String> {
            self.timesteps.lock().expect("lock").push(timestep);
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![1.0f32; latent.len()])
        }
    }

    /// An octree that puts all the mass in child 0, so every anchor lands in
    /// the low corner of the cube.
    #[derive(Debug)]
    struct LowCorner;

    impl OctreeDecoder for LowCorner {
        fn child_probs(
            &self,
            coords: &[[u32; 3]],
            _level: u32,
            _latent: &[f32],
        ) -> Result<Vec<[f32; CHILDREN]>, String> {
            let mut probs = [0.0f32; CHILDREN];
            probs[0] = 1.0;
            Ok(vec![probs; coords.len()])
        }
    }

    /// A Gaussian decoder that answers the model's group of Gaussians per
    /// anchor, each at its anchor.
    #[derive(Debug)]
    struct Echo;

    impl GaussianDecoder for Echo {
        fn decode(&self, anchors: &[[f32; 3]], _latent: &[f32]) -> Result<GaussianCloud, String> {
            let per_anchor = PipelineConfig::idol().gaussians_per_anchor as usize;
            let mut cloud = GaussianCloud {
                xyz: Vec::with_capacity(anchors.len() * per_anchor),
                features_dc: Vec::new(),
                scaling: Vec::new(),
                rotation: Vec::new(),
                opacity: Vec::new(),
            };
            for anchor in anchors {
                for _ in 0..per_anchor {
                    cloud.xyz.push(*anchor);
                    cloud.features_dc.push([0.0; 3]);
                    cloud.scaling.push([0.01; 3]);
                    cloud.rotation.push([1.0, 0.0, 0.0, 0.0]);
                    cloud.opacity.push(0.5);
                }
            }
            Ok(cloud)
        }
    }

    fn conditioning(rows: usize) -> Conditioning {
        Conditioning {
            dinov3: vec![0.0; rows * 1280],
            vae: vec![0.0; rows * 128],
            rows,
        }
    }

    fn pipeline<'a>(
        denoiser: &'a Constant,
        octree: &'a LowCorner,
        gaussian: &'a Echo,
    ) -> Pipeline<'a> {
        Pipeline {
            denoiser,
            octree,
            gaussian,
            config: PipelineConfig::idol(),
            steps: 3,
            rescale_t: 3.0,
            cfg_strength: 0.0,
            num_gaussians: 64,
        }
    }

    #[test]
    fn the_pipeline_answers_the_requested_gaussian_count() {
        let denoiser = Constant::default();
        let cloud = run(
            &pipeline(&denoiser, &LowCorner, &Echo),
            &conditioning(4),
            Seed::new(1),
            None,
        )
        .expect("runs");
        assert_eq!(cloud.len(), 64);
    }

    #[test]
    fn the_denoiser_is_handed_the_scaled_timestep() {
        let denoiser = Constant::default();
        run(
            &pipeline(&denoiser, &LowCorner, &Echo),
            &conditioning(4),
            Seed::new(1),
            None,
        )
        .expect("runs");
        let timesteps = denoiser.timesteps.lock().expect("lock").clone();
        // Three steps, no guidance, so three evaluations.
        assert_eq!(timesteps.len(), 3);
        assert!((timesteps[0] - 1000.0).abs() < 1e-3, "{}", timesteps[0]);
        for timestep in &timesteps {
            assert!((0.0..=1000.0).contains(timestep), "{timestep}");
        }
    }

    #[test]
    fn guidance_runs_the_denoiser_twice_per_step() {
        let denoiser = Constant::default();
        let mut pipeline = pipeline(&denoiser, &LowCorner, &Echo);
        pipeline.cfg_strength = 3.0;
        run(&pipeline, &conditioning(4), Seed::new(1), None).expect("runs");
        assert_eq!(
            denoiser.calls.load(std::sync::atomic::Ordering::SeqCst),
            6,
            "three steps, a conditional and an unconditional pass each"
        );
    }

    #[test]
    fn a_constant_velocity_moves_the_latent_by_its_own_value() {
        // The steps sum to -1, so `sample -= 1 * -1` adds one to every entry.
        // The anchors then sit where the octree put them, which is the low
        // corner of the cube.
        let denoiser = Constant::default();
        let cloud = run(
            &pipeline(&denoiser, &LowCorner, &Echo),
            &conditioning(4),
            Seed::new(1),
            None,
        )
        .expect("runs");
        for position in &cloud.xyz {
            for axis in position {
                assert!(*axis < 0.5, "anchor {axis} is not in the low corner");
            }
        }
    }

    #[test]
    fn progress_is_reported_once_per_step() {
        let denoiser = Constant::default();
        let seen = std::sync::Mutex::new(Vec::new());
        run(
            &pipeline(&denoiser, &LowCorner, &Echo),
            &conditioning(4),
            Seed::new(1),
            Some(&|event| seen.lock().expect("lock").push(event)),
        )
        .expect("runs");
        let seen = seen.into_inner().expect("lock");
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0].step, 0);
        assert_eq!(seen[2].total_steps, 3);
    }

    #[test]
    fn a_count_that_is_not_a_whole_number_of_anchors_is_refused() {
        let denoiser = Constant::default();
        let mut pipeline = pipeline(&denoiser, &LowCorner, &Echo);
        pipeline.num_gaussians = 65;
        assert!(matches!(
            run(&pipeline, &conditioning(4), Seed::new(1), None),
            Err(PipelineError::AnchorBudget { count: 65, .. })
        ));
    }

    #[test]
    fn the_same_seed_answers_the_same_cloud() {
        let first = run(
            &pipeline(&Constant::default(), &LowCorner, &Echo),
            &conditioning(4),
            Seed::new(42),
            None,
        )
        .expect("runs");
        let second = run(
            &pipeline(&Constant::default(), &LowCorner, &Echo),
            &conditioning(4),
            Seed::new(42),
            None,
        )
        .expect("runs");
        assert_eq!(first, second);
        let other = run(
            &pipeline(&Constant::default(), &LowCorner, &Echo),
            &conditioning(4),
            Seed::new(43),
            None,
        )
        .expect("runs");
        assert_ne!(first, other, "a different seed is a different object");
    }

    #[test]
    fn the_cloud_writes_as_a_ply() {
        let cloud = run(
            &pipeline(&Constant::default(), &LowCorner, &Echo),
            &conditioning(4),
            Seed::new(1),
            None,
        )
        .expect("runs");
        let bytes = write_ply(&cloud).expect("writes");
        assert!(bytes.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
        assert!(
            std::str::from_utf8(&bytes[..200])
                .expect("header")
                .contains("element vertex 64\n")
        );
    }
}
