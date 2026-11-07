//! The generation pipeline: noise in, a cloud out.
//!
//! The three device graphs (denoiser, octree, Gaussian decoder) sit behind
//! traits so the orchestration - the draw order, the time grid, the guided
//! velocity, the anchor budget - can be tested without a device. That
//! orchestration is where a subtle error is most expensive: a draw consumed in
//! the wrong order still produces a plausible object, and the only way to
//! notice is a reference that agrees on every stage.
//!
//! The flow integrates two things side by side: the object latent and the
//! camera token. The denoiser reads both and answers a velocity for both, and
//! the Euler step moves both; the camera is not decoded into the file, but a
//! camera left at its noise would feed the denoiser a different joint
//! sequence at every step than the one it was trained on.

use std::fmt;

use shiso_vision::splat::SplatStepEvent;

use super::conditioning::Conditioning;
use super::draws::Draws;
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
    /// The camera's width.
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

/// What the flow integrates: the object latent and the camera token. The
/// denoiser's velocity has the same shape.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowState {
    /// `(q_token_length, in_channels)` row-major.
    pub latent: Vec<f32>,
    /// `cam_channels` values.
    pub camera: Vec<f32>,
}

/// The denoiser: one velocity evaluation.
pub trait Denoiser: Send + Sync + fmt::Debug {
    /// The velocity at `sample` for the model timestep `timestep` (the
    /// sampler's `t` times [`TIMESTEP_SCALE`]) under `cond`.
    ///
    /// # Errors
    ///
    /// A device or checkpoint failure.
    fn velocity(
        &self,
        sample: &FlowState,
        timestep: f32,
        cond: &Conditioning,
    ) -> Result<FlowState, String>;
}

/// The octree decoder: child probabilities for one level's cells.
pub trait OctreeDecoder: Send + Sync + fmt::Debug {
    /// The child probabilities for every cell, in the order given. `coords`
    /// are the cells' integer coordinates at level `level - 1`; the answer is
    /// how each cell's points split among its children at `level`.
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

/// What a generation answers: the cloud, the flow's final state (the latent
/// the decoders read and the camera beside it) and the anchors.
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    /// The decoded Gaussians.
    pub cloud: GaussianCloud,
    /// The integrated latent and camera.
    pub sample: FlowState,
    /// The anchors the octree placed, in `[0, 1]^3`.
    pub anchors: Vec<[f32; 3]>,
}

fn stage(stage: &'static str) -> impl Fn(String) -> PipelineError {
    move |reason| PipelineError::Stage { stage, reason }
}

/// Runs the pipeline and answers the cloud.
///
/// The draws happen in the reference's order: the latent's noise, then the
/// camera's, then one offset per octree cell per level, then three jitters
/// per anchor. (The conditioning's posterior draw comes before all of these;
/// the caller makes it with the same source.)
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
    draws: &mut dyn Draws,
    progress: Option<&dyn Fn(SplatStepEvent)>,
) -> Result<Generation, PipelineError> {
    let per_anchor = pipeline.config.gaussians_per_anchor;
    if per_anchor == 0 || !pipeline.num_gaussians.is_multiple_of(per_anchor) {
        return Err(PipelineError::AnchorBudget {
            count: pipeline.num_gaussians,
            per_anchor,
        });
    }
    let mut sample = FlowState {
        latent: draws.normal(pipeline.config.latent_len()),
        camera: draws.normal(pipeline.config.cam_channels),
    };

    let grid = time_grid(pipeline.steps, pipeline.rescale_t);
    let negative = Conditioning::zero(cond.rows);
    let mut velocity = sample.clone();
    for (index, pair) in grid.windows(2).enumerate() {
        let (t, t_prev) = (pair[0], pair[1]);
        let timestep = t * TIMESTEP_SCALE;
        let conditional = pipeline
            .denoiser
            .velocity(&sample, timestep, cond)
            .map_err(stage("the denoiser"))?;
        let unconditional = if pipeline.cfg_strength > 0.0 {
            pipeline
                .denoiser
                .velocity(&sample, timestep, &negative)
                .map_err(stage("the denoiser"))?
        } else {
            FlowState {
                latent: vec![0.0; sample.latent.len()],
                camera: vec![0.0; sample.camera.len()],
            }
        };
        if conditional.latent.len() != sample.latent.len()
            || conditional.camera.len() != sample.camera.len()
            || unconditional.latent.len() != sample.latent.len()
            || unconditional.camera.len() != sample.camera.len()
        {
            return Err(PipelineError::Stage {
                stage: "the denoiser",
                reason: "a velocity of another shape than the sample".to_owned(),
            });
        }
        guided_velocity(
            &conditional.latent,
            &unconditional.latent,
            pipeline.cfg_strength,
            &mut velocity.latent,
        );
        guided_velocity(
            &conditional.camera,
            &unconditional.camera,
            pipeline.cfg_strength,
            &mut velocity.camera,
        );
        euler_step(&mut sample.latent, &velocity.latent, t, t_prev);
        euler_step(&mut sample.camera, &velocity.camera, t, t_prev);
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
    let latent = &sample.latent;
    let mut probs_at =
        |coords: &[[u32; 3]], level: u32| pipeline.octree.child_probs(coords, level, latent);
    let anchors = place_anchors(
        anchors_wanted,
        pipeline.config.max_level,
        draws,
        &mut probs_at,
    )
    .map_err(stage("the octree decoder"))?;
    let cloud = pipeline
        .gaussian
        .decode(&anchors, &sample.latent)
        .map_err(stage("the Gaussian decoder"))?;
    if cloud.len() != pipeline.num_gaussians as usize {
        return Err(PipelineError::Stage {
            stage: "the Gaussian decoder",
            reason: format!(
                "{} Gaussians for {} anchors of {per_anchor}",
                cloud.len(),
                anchors.len()
            ),
        });
    }
    Ok(Generation {
        cloud,
        sample,
        anchors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idol::draws::{Replay, SeededDraws};
    use crate::idol::ply::{GaussianCloud, write_ply};
    use shiso_core::ids::Seed;

    /// A denoiser that answers a constant velocity for both the latent and the
    /// camera, and records what it was handed.
    #[derive(Debug, Default)]
    struct Constant {
        timesteps: std::sync::Mutex<Vec<f32>>,
        samples: std::sync::Mutex<Vec<FlowState>>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl Denoiser for Constant {
        fn velocity(
            &self,
            sample: &FlowState,
            timestep: f32,
            _cond: &Conditioning,
        ) -> Result<FlowState, String> {
            self.timesteps.lock().expect("lock").push(timestep);
            self.samples.lock().expect("lock").push(sample.clone());
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(FlowState {
                latent: vec![1.0f32; sample.latent.len()],
                camera: vec![1.0f32; sample.camera.len()],
            })
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

    /// An octree whose device fails.
    #[derive(Debug)]
    struct Broken;

    impl OctreeDecoder for Broken {
        fn child_probs(
            &self,
            _coords: &[[u32; 3]],
            _level: u32,
            _latent: &[f32],
        ) -> Result<Vec<[f32; CHILDREN]>, String> {
            Err("the device fell over".to_owned())
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
        Conditioning::zero(rows)
    }

    fn pipeline<'a>(
        denoiser: &'a Constant,
        octree: &'a dyn OctreeDecoder,
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

    fn draws(seed: u64) -> SeededDraws {
        SeededDraws::new(Seed::new(seed))
    }

    #[test]
    fn the_pipeline_answers_the_requested_gaussian_count() {
        let denoiser = Constant::default();
        let generation = run(
            &pipeline(&denoiser, &LowCorner, &Echo),
            &conditioning(4),
            &mut draws(1),
            None,
        )
        .expect("runs");
        assert_eq!(generation.cloud.len(), 64);
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "the latent's 131072 values are exact in f64"
    )]
    fn the_latent_starts_from_standard_normal_noise() {
        // The reference draws standard normals: zero mean, unit variance, both
        // signs. A uniform draw in [0, 1) has none of the three.
        let denoiser = Constant::default();
        run(
            &pipeline(&denoiser, &LowCorner, &Echo),
            &conditioning(4),
            &mut draws(1),
            None,
        )
        .expect("runs");
        let first = denoiser.samples.lock().expect("lock")[0].latent.clone();
        let n = first.len() as f64;
        let mean = first.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
        let var = first
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
    }

    #[test]
    fn the_camera_is_integrated_beside_the_latent() {
        // A unit camera velocity over the whole grid moves the camera by minus
        // one (the steps `t - t_prev` sum to one), and the denoiser sees the moved camera on its second evaluation.
        let denoiser = Constant::default();
        let generation = run(
            &pipeline(&denoiser, &LowCorner, &Echo),
            &conditioning(4),
            &mut draws(1),
            None,
        )
        .expect("runs");
        let samples = denoiser.samples.lock().expect("lock").clone();
        let start = samples[0].camera.clone();
        assert_ne!(samples[1].camera, start, "the camera moved after one step");
        for (end, begin) in generation.sample.camera.iter().zip(&start) {
            assert!((end - begin + 1.0).abs() < 1e-5, "{end} - {begin}");
        }
    }

    #[test]
    fn an_octree_failure_is_refused_not_replaced_by_a_uniform_split() {
        let denoiser = Constant::default();
        let err = run(
            &pipeline(&denoiser, &Broken, &Echo),
            &conditioning(4),
            &mut draws(1),
            None,
        )
        .expect_err("a failed octree decoder fails the generation");
        assert!(
            matches!(
                err,
                PipelineError::Stage {
                    stage: "the octree decoder",
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn the_draws_are_taken_in_the_reference_order() {
        // Latent noise, camera noise, one offset per cell per level (all the
        // mass in one child: one cell per level), three jitters per anchor.
        // A replay holding exactly that many values is drained to the last.
        let config = PipelineConfig::idol();
        let anchors = 64 / config.gaussians_per_anchor as usize;
        let normals = vec![0.0f32; config.latent_len() + config.cam_channels];
        let uniforms = vec![0.5f32; config.max_level as usize + 3 * anchors];
        let mut replay = Replay::new(normals, uniforms);
        let denoiser = Constant::default();
        run(
            &pipeline(&denoiser, &LowCorner, &Echo),
            &conditioning(4),
            &mut replay,
            None,
        )
        .expect("runs");
        assert_eq!(replay.remaining(), (0, 0));
    }

    #[test]
    fn the_denoiser_is_handed_the_scaled_timestep() {
        let denoiser = Constant::default();
        run(
            &pipeline(&denoiser, &LowCorner, &Echo),
            &conditioning(4),
            &mut draws(1),
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
        run(&pipeline, &conditioning(4), &mut draws(1), None).expect("runs");
        assert_eq!(
            denoiser.calls.load(std::sync::atomic::Ordering::SeqCst),
            6,
            "three steps, a conditional and an unconditional pass each"
        );
    }

    #[test]
    fn a_constant_velocity_moves_the_latent_by_its_own_value() {
        // The steps `t - t_prev` sum to one, so `sample -= 1 * dt` takes one
        // from every entry.
        let denoiser = Constant::default();
        let generation = run(
            &pipeline(&denoiser, &LowCorner, &Echo),
            &conditioning(4),
            &mut draws(1),
            None,
        )
        .expect("runs");
        let start = denoiser.samples.lock().expect("lock")[0].latent.clone();
        for (end, begin) in generation.sample.latent.iter().zip(&start).take(100) {
            assert!((end - begin + 1.0).abs() < 1e-5, "{end} - {begin}");
        }
        for position in &generation.cloud.xyz {
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
            &mut draws(1),
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
            run(&pipeline, &conditioning(4), &mut draws(1), None),
            Err(PipelineError::AnchorBudget { count: 65, .. })
        ));
    }

    #[test]
    fn the_same_seed_answers_the_same_cloud() {
        let once = |seed| {
            run(
                &pipeline(&Constant::default(), &LowCorner, &Echo),
                &conditioning(4),
                &mut draws(seed),
                None,
            )
            .expect("runs")
        };
        assert_eq!(once(42), once(42));
        assert_ne!(once(42), once(43), "a different seed is a different object");
    }

    #[test]
    fn the_cloud_writes_as_a_ply() {
        let generation = run(
            &pipeline(&Constant::default(), &LowCorner, &Echo),
            &conditioning(4),
            &mut draws(1),
            None,
        )
        .expect("runs");
        let bytes = write_ply(&generation.cloud).expect("writes");
        assert!(bytes.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
        assert!(
            std::str::from_utf8(&bytes[..200])
                .expect("header")
                .contains("element vertex 64\n")
        );
    }
}
