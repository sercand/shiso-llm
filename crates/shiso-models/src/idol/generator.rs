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

/// A conditioning after the denoiser's own refinement of it: what every pass
/// of one request reads in its place.
///
/// The refinement sees neither the latent nor the timestep, so it is the same
/// on every step; computing it once per request, rather than once per pass,
/// saves its blocks on every pass but the first.
#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    /// Refined rows held in `stream`.
    pub rows: usize,
    /// The refined rows, `(rows, model_channels)` row-major, on the host.
    pub stream: Vec<f32>,
    /// How many rows of the sequence each held row stands for: 1, or the
    /// conditioning's whole length when its rows were all the same row (the
    /// zero conditioning of the guidance pass), which then refines to one
    /// row repeated and is held, and attended, once.
    pub copies: usize,
}

/// What one velocity pass does with the conditioning rows.
///
/// The conditioning is a third of the joint sequence and it is the same
/// picture on every step: what its rows contribute to the latent's attention
/// (their keys and values, block by block) moves only as fast as the timestep
/// and the latent do. A request may therefore run some steps with those rows
/// held from the last step that ran them, which leaves the latent and the
/// camera as the only rows a pass computes: two thirds of the attention and
/// of every linear. See [`Pipeline::refresh_every`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// Every row runs and nothing is kept: the reference's pass.
    Full,
    /// Every row runs, and the conditioning rows' keys and values are kept
    /// for the [`Pass::Held`] passes that follow. The answer is
    /// [`Pass::Full`]'s, bit for bit.
    Keep,
    /// Only the latent and the camera run, attending to their own keys and
    /// to the conditioning's as the last [`Pass::Keep`] left them.
    Held,
}

/// The denoiser: one velocity evaluation.
pub trait Denoiser: Send + Sync + fmt::Debug {
    /// The refined form of `cond` that [`Denoiser::velocity`] reads.
    ///
    /// # Errors
    ///
    /// A device or checkpoint failure.
    fn prepare(&self, cond: &Conditioning) -> Result<Prepared, String>;

    /// The velocity at `sample` for the model timestep `timestep` (the
    /// sampler's `t` times [`TIMESTEP_SCALE`]) under `prepared`, as `pass`
    /// says to run it.
    ///
    /// # Errors
    ///
    /// A device or checkpoint failure, or a [`Pass::Held`] with no
    /// [`Pass::Keep`] of the same conditioning before it.
    fn velocity(
        &self,
        sample: &FlowState,
        timestep: f32,
        prepared: &Prepared,
        pass: Pass,
    ) -> Result<FlowState, String>;

    /// Drops whatever the passes kept between them. Called once the flow is
    /// integrated, so nothing a request held outlives it.
    fn release(&self) {}
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
    /// Every `refresh_every`-th step runs in full, and so do the first two;
    /// the steps between hold what a full step computed that changes slowly.
    ///
    /// One is the reference's sampler: every step runs both passes over every
    /// row. Above one, a step between two full ones runs the conditional pass
    /// alone, as [`Pass::Held`], and adds the guidance direction (the
    /// conditional velocity less the unconditional one) the last full step
    /// measured. The unconditional pass and the conditioning's rows are what
    /// such a step saves: 6.1 s against a full step's 13.2 on an M4.
    ///
    /// It is an approximation. Measured on that M4 at 25 steps and 262144
    /// Gaussians, one seed each of two pictures, on renders of the result: a
    /// request falls from 362 s to 278 s at an interval of two. Against a
    /// 50-step request's renders the held sampler scores what the reference
    /// sampler scores at 25 steps (18.5 against 18.4 dB on one picture, 28.0
    /// against 28.6 on the other) at the same sharpness, and it stays 27.7
    /// and 31.9 dB from the reference sampler's own 25-step renders. An
    /// interval of three is 250 s and slightly softer. Holding the guidance
    /// direction alone, with the conditioning rows still run, saved less
    /// (300 s) and softened the object more.
    pub refresh_every: u32,
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
    /// The refresh interval is zero: no step would run in full.
    #[error("a refresh interval of zero runs no step in full")]
    Refresh,
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

/// A velocity of another shape than the sample is a graph bug; refused here
/// rather than at the first slice that panics on it.
fn same_shape(velocity: &FlowState, sample: &FlowState) -> Result<(), PipelineError> {
    if velocity.latent.len() == sample.latent.len() && velocity.camera.len() == sample.camera.len()
    {
        Ok(())
    } else {
        Err(PipelineError::Stage {
            stage: "the denoiser",
            reason: "a velocity of another shape than the sample".to_owned(),
        })
    }
}

/// Releases a denoiser's kept state when dropped.
struct Release<'a>(&'a dyn Denoiser);

impl Drop for Release<'_> {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// The two conditionings a request integrates under, refined.
#[derive(Debug, Clone, Copy)]
struct Guidance<'a> {
    /// The picture's.
    cond: &'a Prepared,
    /// The zero conditioning's, when guidance is on.
    negative: Option<&'a Prepared>,
}

/// A full step's guided velocity at `sample` into `velocity`: the conditional
/// pass (as `pass`, [`Pass::Full`] or [`Pass::Keep`]) and, when guidance is
/// on, the unconditional one. Answers the guidance direction, the conditional
/// velocity less the unconditional one, for the held steps that follow.
fn full_step(
    pipeline: &Pipeline<'_>,
    sample: &FlowState,
    timestep: f32,
    (guidance, pass): (Guidance<'_>, Pass),
    velocity: &mut FlowState,
) -> Result<FlowState, PipelineError> {
    let conditional = pipeline
        .denoiser
        .velocity(sample, timestep, guidance.cond, pass)
        .map_err(stage("the denoiser"))?;
    let unconditional = if let Some(negative) = guidance.negative {
        pipeline
            .denoiser
            .velocity(sample, timestep, negative, Pass::Full)
            .map_err(stage("the denoiser"))?
    } else {
        FlowState {
            latent: vec![0.0; sample.latent.len()],
            camera: vec![0.0; sample.camera.len()],
        }
    };
    same_shape(&conditional, sample)?;
    same_shape(&unconditional, sample)?;
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
    let less = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a - b).collect();
    Ok(FlowState {
        latent: less(&conditional.latent, &unconditional.latent),
        camera: less(&conditional.camera, &unconditional.camera),
    })
}

/// A held step's velocity at `sample` into `velocity`: the conditional pass
/// over the latent and the camera alone, plus `cfg` times the guidance
/// direction `direction` the last full step measured.
///
/// `(1 + cfg) c - cfg u` is `c + cfg (c - u)`: with the direction held, the
/// conditional velocity is still this step's own.
fn held_step(
    pipeline: &Pipeline<'_>,
    sample: &FlowState,
    timestep: f32,
    (cond, direction): (&Prepared, &FlowState),
    velocity: &mut FlowState,
) -> Result<(), PipelineError> {
    let conditional = pipeline
        .denoiser
        .velocity(sample, timestep, cond, Pass::Held)
        .map_err(stage("the denoiser"))?;
    same_shape(&conditional, sample)?;
    let cfg = pipeline.cfg_strength;
    let add = |out: &mut [f32], c: &[f32], d: &[f32]| {
        for ((out, c), d) in out.iter_mut().zip(c).zip(d) {
            *out = c + cfg * d;
        }
    };
    add(&mut velocity.latent, &conditional.latent, &direction.latent);
    add(&mut velocity.camera, &conditional.camera, &direction.camera);
    Ok(())
}

/// The steps a flow always runs in full before any is held.
///
/// The first steps decide which object the noise becomes, and a step held
/// there sends the flow somewhere else: measured on two pictures at 25 steps
/// and a refresh interval of two, renders of the result against the reference
/// sampler's score 31.9 and 27.7 dB with the first two steps full, and 24.1
/// and 22.6 dB with the second one held. The second full step costs 7 s.
const LEAD_STEPS: usize = 2;

/// Integrates the flow over the pipeline's time grid, moving `sample` from
/// noise to the object's latent, one progress event per step.
///
/// A step runs in full when it is one of the first [`LEAD_STEPS`] or its
/// index is a multiple of the refresh interval; any other is held (see
/// [`Pipeline::refresh_every`]).
fn integrate(
    pipeline: &Pipeline<'_>,
    guidance: Guidance<'_>,
    sample: &mut FlowState,
    progress: Option<&dyn Fn(SplatStepEvent)>,
) -> Result<(), PipelineError> {
    // Whatever the passes keep between them goes when the flow ends, on a
    // failed step as on the last one.
    let _kept = Release(pipeline.denoiser);
    let every = pipeline.refresh_every as usize;
    let grid = time_grid(pipeline.steps, pipeline.rescale_t);
    let mut velocity = sample.clone();
    // The guidance direction of the last full step, for the held ones.
    let mut direction: Option<FlowState> = None;
    for (index, pair) in grid.windows(2).enumerate() {
        let started = std::time::Instant::now();
        let (t, t_prev) = (pair[0], pair[1]);
        let timestep = t * TIMESTEP_SCALE;
        let held = index >= LEAD_STEPS && !index.is_multiple_of(every);
        if let Some(direction) = direction.as_ref().filter(|_| held) {
            held_step(
                pipeline,
                sample,
                timestep,
                (guidance.cond, direction),
                &mut velocity,
            )?;
        } else {
            // A sampler that never holds keeps nothing: the reference's
            // pass, and no memory for keys nobody will read.
            let pass = if every > 1 { Pass::Keep } else { Pass::Full };
            direction = Some(full_step(
                pipeline,
                sample,
                timestep,
                (guidance, pass),
                &mut velocity,
            )?);
        }
        euler_step(&mut sample.latent, &velocity.latent, t, t_prev);
        euler_step(&mut sample.camera, &velocity.camera, t, t_prev);
        tracing::debug!(
            target: "shiso::idol",
            step = index,
            step_ms = started.elapsed().as_millis(),
            "flow step"
        );
        if let Some(progress) = progress {
            // The step index is bounded by the request's 1..=100 step ceiling.
            let step = u32::try_from(index).unwrap_or(u32::MAX);
            progress(SplatStepEvent {
                step,
                total_steps: pipeline.steps,
            });
        }
    }
    Ok(())
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
    if pipeline.refresh_every == 0 {
        return Err(PipelineError::Refresh);
    }
    let mut sample = FlowState {
        latent: draws.normal(pipeline.config.latent_len()),
        camera: draws.normal(pipeline.config.cam_channels),
    };

    let positive = pipeline
        .denoiser
        .prepare(cond)
        .map_err(stage("the denoiser"))?;
    let negative = if pipeline.cfg_strength > 0.0 {
        Some(
            pipeline
                .denoiser
                .prepare(&Conditioning::zero(cond.rows))
                .map_err(stage("the denoiser"))?,
        )
    } else {
        None
    };
    integrate(
        pipeline,
        Guidance {
            cond: &positive,
            negative: negative.as_ref(),
        },
        &mut sample,
        progress,
    )?;

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
        prepares: std::sync::atomic::AtomicUsize,
        /// Each pass as `(which conditioning, how)`: the conditionings are
        /// numbered from one in the order they were prepared.
        passes: std::sync::Mutex<Vec<(usize, Pass)>>,
        releases: std::sync::atomic::AtomicUsize,
        /// What a held pass answers; the other passes answer one.
        held_answer: Option<f32>,
        /// What the second conditioning's passes answer.
        negative_answer: Option<f32>,
    }

    impl Denoiser for Constant {
        fn prepare(&self, cond: &Conditioning) -> Result<Prepared, String> {
            let before = self
                .prepares
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // `copies` carries which conditioning this is, for `passes`.
            Ok(Prepared {
                rows: cond.rows,
                stream: Vec::new(),
                copies: before + 1,
            })
        }

        fn velocity(
            &self,
            sample: &FlowState,
            timestep: f32,
            prepared: &Prepared,
            pass: Pass,
        ) -> Result<FlowState, String> {
            self.timesteps.lock().expect("lock").push(timestep);
            self.samples.lock().expect("lock").push(sample.clone());
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.passes
                .lock()
                .expect("lock")
                .push((prepared.copies, pass));
            let value = match (pass, prepared.copies) {
                (Pass::Held, _) => self.held_answer.unwrap_or(1.0),
                (_, 2) => self.negative_answer.unwrap_or(1.0),
                _ => 1.0,
            };
            Ok(FlowState {
                latent: vec![value; sample.latent.len()],
                camera: vec![value; sample.camera.len()],
            })
        }

        fn release(&self) {
            self.releases
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
            refresh_every: 1,
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
    fn the_reference_sampler_runs_every_pass_in_full_and_keeps_nothing() {
        let denoiser = Constant::default();
        let mut guided = pipeline(&denoiser, &LowCorner, &Echo);
        guided.cfg_strength = 3.0;
        run(&guided, &conditioning(4), &mut draws(1), None).expect("runs");
        let passes = denoiser.passes.lock().expect("lock").clone();
        assert_eq!(
            passes,
            [1, 2, 1, 2, 1, 2].map(|which| (which, Pass::Full)).to_vec()
        );
    }

    #[test]
    fn a_refresh_interval_holds_the_steps_between_full_ones() {
        use std::sync::atomic::Ordering::SeqCst;
        let denoiser = Constant::default();
        let mut held = pipeline(&denoiser, &LowCorner, &Echo);
        held.cfg_strength = 3.0;
        held.steps = 5;
        held.refresh_every = 2;
        run(&held, &conditioning(4), &mut draws(1), None).expect("runs");
        // Steps 0 and 1 lead and steps 2 and 4 are the interval's: all four
        // run both passes, the conditional one keeping its conditioning
        // rows. Step 3 runs the conditional pass alone.
        let full = [(1, Pass::Keep), (2, Pass::Full)];
        let between = [(1, Pass::Held)];
        let want: Vec<(usize, Pass)> = [&full[..], &full, &full, &between, &full].concat();
        assert_eq!(*denoiser.passes.lock().expect("lock"), want);
        assert_eq!(
            denoiser.releases.load(SeqCst),
            1,
            "released once, at the end"
        );
    }

    #[test]
    fn a_held_step_adds_the_last_full_steps_guidance_direction() {
        // Four steps of one quarter each (a unit rescale); the first two lead,
        // the third is held, the fourth is not (3 is a multiple of 3). A full
        // step's passes answer 1 and 0.5: guided 4 * 1 - 3 * 0.5 = 2.5,
        // direction 0.5. The held step's pass answers 2: 2 + 3 * 0.5 = 3.5.
        // The sample moves by -(3 * 2.5 + 3.5) / 4 = -2.75.
        let denoiser = Constant {
            held_answer: Some(2.0),
            negative_answer: Some(0.5),
            ..Constant::default()
        };
        let mut held = pipeline(&denoiser, &LowCorner, &Echo);
        held.cfg_strength = 3.0;
        held.rescale_t = 1.0;
        held.steps = 4;
        held.refresh_every = 3;
        let generation = run(&held, &conditioning(4), &mut draws(1), None).expect("runs");
        let start = denoiser.samples.lock().expect("lock")[0].clone();
        for (end, begin) in generation.sample.latent.iter().zip(&start.latent).take(100) {
            assert!((end - begin + 2.75).abs() < 1e-5, "{end} - {begin}");
        }
        for (end, begin) in generation.sample.camera.iter().zip(&start.camera) {
            assert!((end - begin + 2.75).abs() < 1e-5, "{end} - {begin}");
        }
    }

    #[test]
    fn the_first_two_steps_are_never_held() {
        // An interval longer than the request: only the lead runs in full.
        let denoiser = Constant::default();
        let mut held = pipeline(&denoiser, &LowCorner, &Echo);
        held.steps = 4;
        held.refresh_every = 4;
        run(&held, &conditioning(4), &mut draws(1), None).expect("runs");
        assert_eq!(
            *denoiser.passes.lock().expect("lock"),
            vec![
                (1, Pass::Keep),
                (1, Pass::Keep),
                (1, Pass::Held),
                (1, Pass::Held)
            ]
        );
    }

    #[test]
    fn holding_without_guidance_still_holds_the_conditioning() {
        let denoiser = Constant::default();
        let mut held = pipeline(&denoiser, &LowCorner, &Echo);
        held.refresh_every = 3;
        run(&held, &conditioning(4), &mut draws(1), None).expect("runs");
        assert_eq!(
            *denoiser.passes.lock().expect("lock"),
            vec![(1, Pass::Keep), (1, Pass::Keep), (1, Pass::Held)]
        );
    }

    #[test]
    fn a_refresh_interval_of_zero_is_refused() {
        let denoiser = Constant::default();
        let mut never = pipeline(&denoiser, &LowCorner, &Echo);
        never.refresh_every = 0;
        assert!(matches!(
            run(&never, &conditioning(4), &mut draws(1), None),
            Err(PipelineError::Refresh)
        ));
    }

    #[test]
    fn what_the_passes_kept_is_released_when_a_stage_fails() {
        use std::sync::atomic::Ordering::SeqCst;
        let denoiser = Constant::default();
        run(
            &pipeline(&denoiser, &Broken, &Echo),
            &conditioning(4),
            &mut draws(1),
            None,
        )
        .expect_err("the octree fails");
        assert_eq!(denoiser.releases.load(SeqCst), 1);
    }

    #[test]
    fn the_conditioning_is_prepared_once_per_stream() {
        use std::sync::atomic::Ordering::SeqCst;
        let denoiser = Constant::default();
        let mut guided = pipeline(&denoiser, &LowCorner, &Echo);
        guided.cfg_strength = 3.0;
        run(&guided, &conditioning(4), &mut draws(1), None).expect("runs");
        assert_eq!(
            denoiser.prepares.load(SeqCst),
            2,
            "the picture and the zero conditioning"
        );
        assert_eq!(denoiser.calls.load(SeqCst), 6);

        let denoiser = Constant::default();
        run(
            &pipeline(&denoiser, &LowCorner, &Echo),
            &conditioning(4),
            &mut draws(1),
            None,
        )
        .expect("runs");
        assert_eq!(
            denoiser.prepares.load(SeqCst),
            1,
            "no guidance, no zero conditioning"
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
