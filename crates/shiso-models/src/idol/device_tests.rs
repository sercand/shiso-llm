//! The family's device stages against independent references.
//!
//! `scripts/reference/idol_reference.py` runs the reference stages in float32
//! on the CPU and writes their inputs and outputs under
//! `test-vectors/references/idol/`. Each case here loads the stage's real
//! checkpoint from a `SHISO_TEST_MODEL_*` variable, runs it on the Metal
//! device and compares. Every case is `#[ignore]`, and one whose checkpoint,
//! fixture or device is missing says so on stderr and returns: a skipped case
//! is not a passing one.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the comparison's rms is an f64 sum narrowed to the f32 the device computes in, \
              and the e2e manifest's small counts are JSON numbers"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::Backend;

use crate::diffusion::ops::Ops;
use crate::flux2::vae_encoder::VaeEncoder;
use crate::idol::generator::Pass;
use crate::idol::latent::{LatentNorm, VAE_PREFIX_ROWS, tokenize, with_prefix};

pub(super) fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/idol")
}

#[expect(
    clippy::print_stderr,
    reason = "a skipped reference case must say why on stderr"
)]
pub(super) fn skip(case: &str, reason: &str) {
    eprintln!("SKIP idol {case}: {reason}");
}

/// The checkpoint directory `var` names, or the reason the case cannot run.
pub(super) fn checkpoint(var: &str) -> Result<PathBuf, String> {
    let path = std::env::var_os(var).ok_or_else(|| format!("{var} is not set"))?;
    let path = PathBuf::from(path);
    if path.is_dir() {
        Ok(path)
    } else {
        Err(format!("{var}={} is not a directory", path.display()))
    }
}

/// The fixture `name`, or the reason the case cannot run.
pub(super) fn fixture(name: &str) -> Result<PathBuf, String> {
    let path = fixture_dir().join(name);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "{} is missing; run scripts/reference/idol_reference.py",
            path.display()
        ))
    }
}

pub(super) fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The Metal device, or the reason the case cannot run.
pub(super) fn device() -> Result<Arc<dyn Backend>, String> {
    if !shiso_metal::metal_available() {
        return Err("no Metal device on this machine".to_owned());
    }
    let backend: Arc<dyn Backend> =
        shiso_metal::MetalBackend::new().map_err(|e| format!("opening Metal: {e}"))?;
    Ok(backend)
}

/// The fixture picture as a planar `(3, H, W)` field in `[0, 1]`.
pub(super) fn picture(path: &Path) -> (Vec<f32>, usize, usize) {
    let image = image::open(path)
        .unwrap_or_else(|e| panic!("decoding {}: {e}", path.display()))
        .to_rgb8();
    let (width, height) = (image.width() as usize, image.height() as usize);
    let mut planar = vec![0.0f32; 3 * width * height];
    for (x, y, pixel) in image.enumerate_pixels() {
        for channel in 0..3 {
            planar[(channel * height + y as usize) * width + x as usize] =
                f32::from(pixel[channel]) / 255.0;
        }
    }
    (planar, width, height)
}

/// Asserts `|got - want| <= tol * max(|want|, rms(want))` element by element,
/// and reports the worst ratio and the mean error.
pub(super) fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    close_mean(got, want, tol, f64::from(tol), what);
}

/// As [`close`], and also bounds the mean error at `mean_tol` of the rms: for
/// a stage whose worst element is one outlier among millions, the mean is the
/// sharper statement.
pub(super) fn close_mean(got: &[f32], want: &[f32], tol: f32, mean_tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let rms =
        (want.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / want.len() as f64).sqrt() as f32;
    let mut worst = 0f32;
    let mut at = 0usize;
    let mut total = 0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        total += f64::from((g - w).abs());
        let ratio = (g - w).abs() / w.abs().max(rms);
        if ratio.is_nan() || ratio > worst {
            worst = if ratio.is_nan() { f32::INFINITY } else { ratio };
            at = i;
        }
    }
    #[expect(
        clippy::print_stderr,
        reason = "the measured agreement is the case's report"
    )]
    {
        let mean = total / got.len() as f64 / f64::from(rms);
        eprintln!(
            "{what}: worst error {worst:.3e} of max(|w|, rms {rms:.3e}) at {at}, mean {mean:.3e} of rms"
        );
    }
    assert!(
        worst <= tol,
        "{what}[{at}]: device {} vs reference {} (worst {worst:.3e} > {tol:.1e})",
        got[at],
        want[at]
    );
    let mean = total / got.len() as f64 / f64::from(rms);
    assert!(
        mean <= mean_tol,
        "{what}: mean error {mean:.3e} > {mean_tol:.1e}"
    );
}

/// Every fixture and checkpoint a case needs, or the first reason it cannot run.
macro_rules! need {
    ($case:expr, $value:expr) => {
        match $value {
            Ok(value) => value,
            Err(reason) => {
                skip($case, &reason);
                return;
            }
        }
    };
}

#[test]
#[ignore = "needs the FLUX.2 VAE (SHISO_TEST_MODEL_FLUX2_VAE), the idol fixtures and a Metal device"]
fn the_vae_encoder_matches_the_reference_posterior() {
    let case = "vae encode";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_FLUX2_VAE"));
    let png = need!(case, fixture("preprocessed.png"));
    let moments_ref = need!(case, fixture("vae_moments.f32"));
    let noise = need!(case, fixture("vae_noise.f32"));
    let sample_ref = need!(case, fixture("vae_sample.f32"));
    let tokens_ref = need!(case, fixture("vae_tokens.f32"));
    let backend = need!(case, device());
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let encoder = VaeEncoder::load(&dir, &ops).expect("the VAE encoder loads");
    let (rgb, width, height) = picture(&png);
    let moments = encoder
        .moments(&ops, &rgb, width, height)
        .expect("the encoder runs");
    close(&moments, &read_f32(&moments_ref), 1e-3, "vae moments");
    let sample = VaeEncoder::sample(&moments, &read_f32(&noise)).expect("noise fits");
    close(&sample, &read_f32(&sample_ref), 1e-3, "vae sample");
    let (mean, var, eps) = encoder.batch_norm();
    let norm = LatentNorm::new(mean, var, eps);
    let channels = encoder.latent_channels();
    let tokens = tokenize(
        &read_f32(&sample_ref),
        channels,
        height / 8,
        width / 8,
        &norm,
    );
    let rows = tokens.len() / (channels * 4);
    let tokens = with_prefix(&tokens, rows, channels * 4, VAE_PREFIX_ROWS);
    close(&tokens, &read_f32(&tokens_ref), 1e-5, "vae tokens");
}

#[test]
#[ignore = "needs DINOv3 (SHISO_TEST_MODEL_DINOV3), the idol fixtures and a Metal device"]
fn the_image_tower_matches_the_reference_tokens() {
    let case = "dinov3 tower";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_DINOV3"));
    let png = need!(case, fixture("preprocessed.png"));
    let tokens_ref = need!(case, fixture("dino_tokens.f32"));
    let backend = need!(case, device());
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let tower = crate::idol::dinov3::encoder::DinoV3::load(&dir, &ops).expect("the tower loads");
    let (rgb, width, height) = picture(&png);
    let tokens = tower
        .tokens(&ops, &rgb, width, height)
        .expect("the tower runs");
    close(&tokens, &read_f32(&tokens_ref), 5e-4, "dinov3 tokens");
}

#[test]
#[ignore = "needs DINOv3 and the FLUX.2 VAE (SHISO_TEST_MODEL_DINOV3, SHISO_TEST_MODEL_FLUX2_VAE), the idol fixtures and a Metal device"]
fn the_conditioner_matches_both_reference_streams() {
    use crate::idol::conditioner::{Conditioner, IdolConditioner};
    use crate::idol::draws::Replay;

    let case = "conditioner";
    let tower = need!(case, checkpoint("SHISO_TEST_MODEL_DINOV3"));
    let vae = need!(case, checkpoint("SHISO_TEST_MODEL_FLUX2_VAE"));
    let png = need!(case, fixture("preprocessed.png"));
    let noise = need!(case, fixture("vae_noise.f32"));
    let dino_ref = need!(case, fixture("dino_tokens.f32"));
    let vae_ref = need!(case, fixture("vae_tokens.f32"));
    let backend = need!(case, device());
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let conditioner = IdolConditioner::load(ops, &tower, &vae).expect("both encoders load");
    let (rgb, width, height) = picture(&png);
    assert_eq!(width, height, "the fixture picture is square");
    let mut draws = Replay::new(read_f32(&noise), Vec::new());
    let cond = conditioner
        .condition(&rgb, width, &mut draws)
        .expect("the conditioner runs");
    assert_eq!(
        draws.remaining(),
        (0, 0),
        "the posterior takes every noise value"
    );
    assert_eq!(cond.rows, crate::idol::conditioning::COND_ROWS);
    close(
        &cond.dinov3,
        &read_f32(&dino_ref),
        5e-4,
        "conditioning tower stream",
    );
    close(
        &cond.vae,
        &read_f32(&vae_ref),
        5e-4,
        "conditioning vae stream",
    );
}

/// The fixture conditioning (the reference encoders' streams).
fn fixture_conditioning() -> Result<crate::idol::conditioning::Conditioning, String> {
    let dino = read_f32(&fixture("dino_tokens.f32")?);
    let vae = read_f32(&fixture("vae_tokens.f32")?);
    crate::idol::conditioning::assemble(dino, vae).map_err(|e| e.to_string())
}

#[test]
#[ignore = "needs the splat checkpoint (SHISO_TEST_MODEL_IDOL), the idol fixtures and a Metal device"]
fn the_denoiser_matches_the_reference_velocity() {
    use crate::idol::denoiser::IdolDenoiser;

    let case = "denoiser";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_IDOL"));
    let cond = need!(case, fixture_conditioning());
    let latent = read_f32(&need!(case, fixture("den_in_latent.f32")));
    let camera = read_f32(&need!(case, fixture("den_in_camera.f32")));
    let out_latent = read_f32(&need!(case, fixture("den_out_latent.f32")));
    let out_camera = read_f32(&need!(case, fixture("den_out_camera.f32")));
    let neg_latent = read_f32(&need!(case, fixture("den_neg_latent.f32")));
    let neg_camera = read_f32(&need!(case, fixture("den_neg_camera.f32")));
    let backend = need!(case, device());
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let denoiser = IdolDenoiser::load(&dir, ops).expect("the denoiser loads");
    let prepared = denoiser.context(&cond).expect("the conditioning refines");
    let v = denoiser
        .forward(&latent, &camera, 700.0, &prepared, Pass::Full)
        .expect("the denoiser runs");
    // Measured: worst 1.3e-4, mean 4.4e-6.
    close_mean(
        &v.latent,
        &out_latent,
        1e-3,
        5e-5,
        "denoiser latent velocity",
    );
    close(&v.camera, &out_camera, 1e-4, "denoiser camera velocity");
    let negative = crate::idol::conditioning::Conditioning::zero(cond.rows);
    let negative = denoiser
        .context(&negative)
        .expect("the zero conditioning refines");
    let v = denoiser
        .forward(&latent, &camera, 700.0, &negative, Pass::Full)
        .expect("the negative pass runs");
    // Zero conditioning gives 4101 identical context rows, which the
    // denoiser holds as one weighted row, and the 24 joint blocks amplify f32
    // rounding more there: the velocity agrees to a mean of 1.5e-4 (worst
    // 3.0e-3). Holding the rows out in full measured 2e-4 and 4.1e-3.
    close_mean(
        &v.latent,
        &neg_latent,
        1e-2,
        5e-4,
        "denoiser negative latent velocity",
    );
    close(
        &v.camera,
        &neg_camera,
        1e-3,
        "denoiser negative camera velocity",
    );
}

#[test]
#[ignore = "needs the splat checkpoint (SHISO_TEST_MODEL_IDOL), the idol fixtures and a Metal device"]
fn the_octree_decoder_matches_the_reference_logits() {
    use crate::idol::decoders::OctreeNet;

    let case = "octree decoder";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_IDOL"));
    let latent = read_f32(&need!(case, fixture("dec_latent.f32")));
    let coords = read_f32(&need!(case, fixture("oct_coords.f32")));
    let logits_ref = read_f32(&need!(case, fixture("oct_logits.f32")));
    let backend = need!(case, device());
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let net = OctreeNet::load(&dir, ops).expect("the octree decoder loads");
    // Level 3's parents sit on the 4x4x4 grid; their children at resolution 8.
    let centres: Vec<f32> = coords.iter().map(|c| (c + 0.5) / 4.0).collect();
    let logits = net
        .logits(&centres, 8.0, &latent)
        .expect("the decoder runs");
    close(&logits, &logits_ref, 1e-3, "octree logits");
}

#[test]
#[ignore = "needs the splat checkpoint (SHISO_TEST_MODEL_IDOL), the idol fixtures and a Metal device"]
fn the_gaussian_decoder_matches_the_reference_cloud() {
    use crate::idol::decoders::GaussianNet;
    use crate::idol::gaussian::{RawOutput, RepresentationConfig, activate};

    let case = "gaussian decoder";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_IDOL"));
    let latent = read_f32(&need!(case, fixture("dec_latent.f32")));
    let anchors = read_f32(&need!(case, fixture("gs_anchors.f32")));
    let features_ref = read_f32(&need!(case, fixture("gs_features.f32")));
    let backend = need!(case, device());
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let net = GaussianNet::load(&dir, ops).expect("the Gaussian decoder loads");
    let features = net.features(&anchors, &latent).expect("the decoder runs");
    // Measured: mean 1.8e-6, one element of 245760 at 2e-3.
    close_mean(&features, &features_ref, 5e-3, 1e-5, "gaussian features");

    // The activations, on the reference's own features.
    let points: Vec<[f32; 3]> = anchors
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let cloud = activate(
        &RawOutput {
            features: &features_ref,
            anchors: &points,
            perturbation: &net.perturbation,
            base_offset_scale: net.base_offset_scale,
        },
        &RepresentationConfig::idol(),
    )
    .expect("the reference features activate");
    let flat = |rows: &[[f32; 3]]| rows.iter().flatten().copied().collect::<Vec<f32>>();
    close(
        &flat(&cloud.xyz),
        &read_f32(&need!(case, fixture("gs_xyz.f32"))),
        1e-5,
        "gaussian xyz",
    );
    close(
        &flat(&cloud.features_dc),
        &read_f32(&need!(case, fixture("gs_features_dc.f32"))),
        1e-5,
        "gaussian colour",
    );
    close(
        &flat(&cloud.scaling),
        &read_f32(&need!(case, fixture("gs_scaling.f32"))),
        1e-5,
        "gaussian scaling",
    );
    let rotation: Vec<f32> = cloud.rotation.iter().flatten().copied().collect();
    close(
        &rotation,
        &read_f32(&need!(case, fixture("gs_rotation.f32"))),
        1e-5,
        "gaussian rotation",
    );
    close(
        &cloud.opacity,
        &read_f32(&need!(case, fixture("gs_opacity.f32"))),
        1e-5,
        "gaussian opacity",
    );
    let ply = crate::idol::ply::write_ply(&cloud).expect("the cloud writes");
    let ply_ref = std::fs::read(need!(case, fixture("gs.ply"))).expect("the reference ply");
    assert_eq!(
        ply.len(),
        ply_ref.len(),
        "the two files are the same length"
    );
    let header_end = ply_ref
        .windows(11)
        .position(|w| w == b"end_header\n")
        .expect("a header")
        + 11;
    assert_eq!(
        ply[..header_end],
        ply_ref[..header_end],
        "the headers are identical"
    );
    let body = |bytes: &[u8]| -> Vec<f32> {
        bytes[header_end..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    close(&body(&ply), &body(&ply_ref), 1e-4, "gaussian ply body");
}

/// The end-to-end case's recorded draws and settings.
struct E2e {
    noise: Vec<f32>,
    uniforms: Vec<f32>,
    steps: u32,
    rescale_t: f32,
    cfg_strength: f32,
    num_gaussians: u32,
}

fn e2e_fixture() -> Result<E2e, String> {
    let manifest = std::fs::read_to_string(fixture("e2e.json")?).map_err(|e| e.to_string())?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest).map_err(|e| e.to_string())?;
    let field = |name: &str| {
        manifest[name]
            .as_f64()
            .ok_or(format!("e2e.json has no {name}"))
    };
    let mut noise = read_f32(&fixture("e2e_noise_latent.f32")?);
    noise.extend(read_f32(&fixture("e2e_noise_camera.f32")?));
    let mut uniforms = Vec::new();
    for level in 1..=field("levels")? as usize {
        uniforms.extend(read_f32(&fixture(&format!(
            "e2e_octree_offsets_{level}.f32"
        ))?));
    }
    uniforms.extend(read_f32(&fixture("e2e_jitter.f32")?));
    Ok(E2e {
        noise,
        uniforms,
        steps: field("steps")? as u32,
        rescale_t: field("rescale_t")? as f32,
        cfg_strength: field("cfg_strength")? as f32,
        num_gaussians: field("num_gaussians")? as u32,
    })
}

/// A Gaussian decoder for the flow case, which checks the latent and never
/// looks at the cloud.
#[derive(Debug)]
struct NoCloud;

impl crate::idol::generator::GaussianDecoder for NoCloud {
    fn decode(
        &self,
        anchors: &[[f32; 3]],
        _latent: &[f32],
    ) -> Result<crate::idol::ply::GaussianCloud, String> {
        let n = anchors.len() * 32;
        Ok(crate::idol::ply::GaussianCloud {
            xyz: vec![[0.0; 3]; n],
            features_dc: vec![[0.0; 3]; n],
            scaling: vec![[0.01; 3]; n],
            rotation: vec![[1.0, 0.0, 0.0, 0.0]; n],
            opacity: vec![0.5; n],
        })
    }
}

/// An octree for the flow case: one child per cell, so the descent takes one
/// draw per level and the uniforms it is handed are never short.
#[derive(Debug)]
struct OneChild;

impl crate::idol::generator::OctreeDecoder for OneChild {
    fn child_probs(
        &self,
        coords: &[[u32; 3]],
        _level: u32,
        _latent: &[f32],
    ) -> Result<Vec<[f32; 8]>, String> {
        let mut probs = [0.0f32; 8];
        probs[0] = 1.0;
        Ok(vec![probs; coords.len()])
    }
}

#[test]
#[ignore = "needs the splat checkpoint (SHISO_TEST_MODEL_IDOL), the idol fixtures and a Metal device"]
fn the_flow_integrates_to_the_reference_latent() {
    use crate::idol::denoiser::IdolDenoiser;
    use crate::idol::draws::Replay;
    use crate::idol::generator::{Pipeline, PipelineConfig, run};

    let case = "end to end flow";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_IDOL"));
    let cond = need!(case, fixture_conditioning());
    let e2e = need!(case, e2e_fixture());
    let latent_ref = read_f32(&need!(case, fixture("e2e_latent.f32")));
    let camera_ref = read_f32(&need!(case, fixture("e2e_camera.f32")));
    let backend = need!(case, device());
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let denoiser = IdolDenoiser::load(&dir, ops).expect("the denoiser loads");
    let anchors = e2e.num_gaussians as usize / 32;
    let mut replay = Replay::new(e2e.noise, vec![0.5; 8 + 3 * anchors]);
    let generation = run(
        &Pipeline {
            denoiser: &denoiser,
            octree: &OneChild,
            gaussian: &NoCloud,
            config: PipelineConfig::idol(),
            steps: e2e.steps,
            rescale_t: e2e.rescale_t,
            cfg_strength: e2e.cfg_strength,
            num_gaussians: e2e.num_gaussians,
            refresh_every: 1,
        },
        &cond,
        &mut replay,
        None,
    )
    .expect("the flow runs");
    assert_eq!(
        replay.remaining().0,
        0,
        "the latent and camera noise are both drawn"
    );
    // Four guided steps carry the negative pass's error (mean 2e-4 of the
    // velocity) times the guidance weight: measured mean 1.4e-4, worst element
    // 2.9e-2 of the scale.
    close_mean(
        &generation.sample.latent,
        &latent_ref,
        5e-2,
        5e-4,
        "e2e latent",
    );
    close(&generation.sample.camera, &camera_ref, 1e-2, "e2e camera");
}

/// A denoiser that answers zero velocity, so the flow leaves the sample it
/// was handed where it is: the decode case's way to start from the
/// reference's own final latent.
#[derive(Debug)]
struct Still;

impl crate::idol::generator::Denoiser for Still {
    fn prepare(
        &self,
        cond: &crate::idol::conditioning::Conditioning,
    ) -> Result<crate::idol::generator::Prepared, String> {
        Ok(crate::idol::generator::Prepared {
            rows: cond.rows,
            stream: Vec::new(),
            copies: 1,
        })
    }

    fn velocity(
        &self,
        sample: &crate::idol::generator::FlowState,
        _timestep: f32,
        _prepared: &crate::idol::generator::Prepared,
        _pass: Pass,
    ) -> Result<crate::idol::generator::FlowState, String> {
        Ok(crate::idol::generator::FlowState {
            latent: vec![0.0; sample.latent.len()],
            camera: vec![0.0; sample.camera.len()],
        })
    }
}

#[test]
#[ignore = "needs the splat checkpoint (SHISO_TEST_MODEL_IDOL), the idol fixtures and a Metal device"]
fn the_decode_from_the_reference_latent_replays_the_reference_cloud() {
    use crate::idol::decoders::{GaussianNet, OctreeNet};
    use crate::idol::draws::Replay;
    use crate::idol::generator::{Pipeline, PipelineConfig, run};

    let case = "end to end decode";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_IDOL"));
    let e2e = need!(case, e2e_fixture());
    let mut start = read_f32(&need!(case, fixture("e2e_latent.f32")));
    start.extend(read_f32(&need!(case, fixture("e2e_camera.f32"))));
    let anchors_ref = read_f32(&need!(case, fixture("e2e_anchors.f32")));
    let ply_ref = std::fs::read(need!(case, fixture("e2e.ply"))).expect("the reference ply");
    let backend = need!(case, device());
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let octree = OctreeNet::load(&dir, Arc::clone(&ops)).expect("the octree decoder loads");
    let gaussian = GaussianNet::load(&dir, ops).expect("the Gaussian decoder loads");
    let mut replay = Replay::new(start, e2e.uniforms);
    let generation = run(
        &Pipeline {
            denoiser: &Still,
            octree: &octree,
            gaussian: &gaussian,
            config: PipelineConfig::idol(),
            steps: 1,
            rescale_t: e2e.rescale_t,
            cfg_strength: 0.0,
            num_gaussians: e2e.num_gaussians,
            refresh_every: 1,
        },
        &crate::idol::conditioning::Conditioning::zero(1),
        &mut replay,
        None,
    )
    .expect("the decode runs");
    assert_eq!(
        replay.remaining(),
        (0, 0),
        "every recorded draw is used, in order"
    );
    // Every octree split agreed, so the replayed offsets and jitters land the
    // anchors on the reference's points.
    let anchors: Vec<f32> = generation.anchors.iter().flatten().copied().collect();
    close(&anchors, &anchors_ref, 1e-6, "e2e anchors");
    let ply = crate::idol::ply::write_ply(&generation.cloud).expect("the cloud writes");
    assert_eq!(ply.len(), ply_ref.len(), "the same vertex count");
    let header_end = ply_ref
        .windows(11)
        .position(|w| w == b"end_header\n")
        .expect("a header")
        + 11;
    assert_eq!(
        ply[..header_end],
        ply_ref[..header_end],
        "identical headers"
    );
    let body = |bytes: &[u8]| -> Vec<f32> {
        bytes[header_end..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    close_mean(&body(&ply), &body(&ply_ref), 5e-3, 1e-5, "e2e ply body");
}

#[test]
#[ignore = "needs the splat checkpoint (SHISO_TEST_MODEL_IDOL), the idol fixtures and a Metal device"]
fn the_background_remover_matches_the_reference_matte() {
    use crate::idol::birefnet::BiRefNet;
    use crate::idol::preprocess::Matting;

    let case = "background remover";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_IDOL"));
    let png = need!(case, fixture("matte_input.png"));
    let alpha_ref =
        std::fs::read(need!(case, fixture("matte_alpha.u8"))).expect("the reference matte");
    let backend = need!(case, device());
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let net = BiRefNet::load(&dir.join("birefnet.safetensors"), ops).expect("the remover loads");
    let rgb = image::open(&png).expect("the fixture picture").to_rgb8();
    let alpha = net.alpha(&rgb).expect("the remover runs");
    assert_eq!(
        alpha.len(),
        alpha_ref.len(),
        "the matte is the picture's size"
    );
    // The matte is bytes: compare the counts that move, not a float bound.
    let mut off_by_more_than_one = 0usize;
    let mut worst = 0u8;
    for (got, want) in alpha.iter().zip(&alpha_ref) {
        let diff = got.abs_diff(*want);
        worst = worst.max(diff);
        if diff > 1 {
            off_by_more_than_one += 1;
        }
    }
    let share = off_by_more_than_one as f64 / alpha.len() as f64;
    #[expect(
        clippy::print_stderr,
        reason = "the measured agreement is the case's report"
    )]
    {
        eprintln!("matte: worst {worst} levels, {share:.3e} of pixels off by more than one");
    }
    assert!(
        share < 1e-3,
        "{share} of the matte is off by more than one level"
    );
    assert!(worst <= 16, "a pixel is {worst} levels off");
}

/// The fused kernel against the GEMM path it replaces, on ragged, joint and
/// decoder-like shapes. Both are f32 end to end, so they agree to rounding.
#[test]
#[ignore = "needs a Metal device"]
fn the_fused_attention_matches_the_gemm_path() {
    use crate::idol::nn::{AttnShape, Qkv, attention_gemm};
    use crate::idol::splat_ops::fused_attention;
    let case = "fused attention";
    let backend = need!(case, device());
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let signal = |len: usize, seed: u32| -> Vec<f32> {
        let mut state = seed | 0x9E37_79B9;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    };
    for (queries, keys, heads) in [(45, 77, 3), (4101, 4101, 16), (300, 8192, 16)] {
        let width = heads * 64;
        let q = ops
            .upload_scratch("test q", &signal(queries * width, 1))
            .expect("upload");
        let k = ops
            .upload_scratch("test k", &signal(keys * width, 2))
            .expect("upload");
        let v = ops
            .upload_scratch("test v", &signal(keys * width, 3))
            .expect("upload");
        let qkv = || Qkv {
            q: q.view(),
            k: k.view(),
            v: v.view(),
        };
        let shape = AttnShape {
            queries,
            keys,
            heads,
            head_dim: 64,
            heavy: None,
        };
        let fused = fused_attention(&ops, "test fused", qkv(), shape).expect("fused");
        let fused = ops
            .download("test", fused.view(), queries * width)
            .expect("read");
        let gemm = attention_gemm(&ops, "test gemm", qkv(), shape).expect("gemm");
        let gemm = ops
            .download("test", gemm.view(), queries * width)
            .expect("read");
        close(&fused, &gemm, 1e-5, "fused attention");
    }
}

/// One refined conditioning serves every pass: the joint blocks write over
/// the stream rows on the device, so a stream that was aliased rather than
/// copied in would hand the second pass the first pass's output.
#[test]
#[ignore = "needs the splat checkpoint (SHISO_TEST_MODEL_IDOL), the idol fixtures and a Metal device"]
fn a_prepared_stream_is_reusable() {
    use crate::idol::denoiser::IdolDenoiser;

    let case = "prepared stream";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_IDOL"));
    let cond = need!(case, fixture_conditioning());
    let latent = read_f32(&need!(case, fixture("den_in_latent.f32")));
    let camera = read_f32(&need!(case, fixture("den_in_camera.f32")));
    let backend = need!(case, device());
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let denoiser = IdolDenoiser::load(&dir, ops).expect("the denoiser loads");
    let prepared = denoiser.context(&cond).expect("the conditioning refines");
    let first = denoiser
        .forward(&latent, &camera, 700.0, &prepared, Pass::Full)
        .expect("the first pass runs");
    let second = denoiser
        .forward(&latent, &camera, 700.0, &prepared, Pass::Full)
        .expect("the second pass runs");
    assert_eq!(first.latent, second.latent, "latent velocity");
    assert_eq!(first.camera, second.camera, "camera velocity");
}

/// A latent, a camera and a conditioning drawn from a seed: the cases that
/// compare one pass against another need inputs, not a reference's.
fn drawn_inputs(
    config: &crate::idol::config::DenoiserArgs,
    rows: usize,
) -> (Vec<f32>, Vec<f32>, crate::idol::conditioning::Conditioning) {
    use crate::idol::draws::{Draws as _, SeededDraws};
    let mut draws = SeededDraws::new(shiso_core::ids::Seed::new(7));
    let latent = draws.normal(config.q_token_length * config.in_channels);
    let camera = draws.normal(config.cam_channels);
    let cond = crate::idol::conditioning::Conditioning {
        dinov3: draws.normal(rows * config.cond_channels),
        vae: draws.normal(rows * config.cond2_channels),
        rows,
    };
    (latent, camera, cond)
}

/// A pass that keeps the conditioning rows' keys answers what a full pass
/// answers, and a held pass straight after it - the same latent, the same
/// timestep - answers it again: the kept keys are then exactly the ones a
/// full pass would compute, attended in another order.
#[test]
#[ignore = "needs the splat checkpoint (SHISO_TEST_MODEL_IDOL) and a Metal device"]
fn a_held_pass_at_the_kept_state_repeats_the_full_pass() {
    use crate::idol::denoiser::IdolDenoiser;

    let case = "held pass";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_IDOL"));
    let backend = need!(case, device());
    let args = crate::idol::config::IdolConfig::read(&dir)
        .expect("the config reads")
        .denoiser;
    // Fewer conditioning rows than a picture's, so the case runs in seconds;
    // the kept buffers and the key layout do not depend on the count.
    let (latent, camera, cond) = drawn_inputs(&args, 517);
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let denoiser = IdolDenoiser::load(&dir, ops).expect("the denoiser loads");
    let prepared = denoiser.context(&cond).expect("the conditioning refines");

    // Nothing kept yet: a held pass has nothing to attend to.
    let err = denoiser
        .forward(&latent, &camera, 700.0, &prepared, Pass::Held)
        .expect_err("a held pass with nothing kept");
    assert!(err.to_string().contains("kept"), "{err}");

    let full = denoiser
        .forward(&latent, &camera, 700.0, &prepared, Pass::Full)
        .expect("the full pass runs");
    let keep = denoiser
        .forward(&latent, &camera, 700.0, &prepared, Pass::Keep)
        .expect("the keeping pass runs");
    assert_eq!(full.latent, keep.latent, "keeping changes no latent value");
    assert_eq!(full.camera, keep.camera, "keeping changes no camera value");

    let held = denoiser
        .forward(&latent, &camera, 700.0, &prepared, Pass::Held)
        .expect("the held pass runs");
    // Measured on an M4: worst 7.8e-5, mean 2.9e-6 of the rms.
    close_mean(
        &held.latent,
        &full.latent,
        1e-3,
        5e-5,
        "held latent velocity",
    );
    close(&held.camera, &full.camera, 1e-4, "held camera velocity");

    // At another timestep the held keys are stale, and the pass still runs:
    // that is the approximation, and it must not be the previous answer.
    let moved = denoiser
        .forward(&latent, &camera, 650.0, &prepared, Pass::Held)
        .expect("the held pass runs at another timestep");
    assert_ne!(moved.latent, held.latent);

    denoiser.release();
    let err = denoiser
        .forward(&latent, &camera, 700.0, &prepared, Pass::Held)
        .expect_err("a held pass after the release");
    assert!(err.to_string().contains("kept"), "{err}");
}

/// The refined latent is carried from a step's first pass to its second, and
/// from nowhere else: another latent or another timestep refines again.
#[test]
#[ignore = "needs the splat checkpoint (SHISO_TEST_MODEL_IDOL) and a Metal device"]
fn the_refined_latent_is_reused_only_for_the_same_latent_and_timestep() {
    use crate::idol::denoiser::IdolDenoiser;

    let case = "refined latent";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_IDOL"));
    let backend = need!(case, device());
    let args = crate::idol::config::IdolConfig::read(&dir)
        .expect("the config reads")
        .denoiser;
    let (latent, camera, cond) = drawn_inputs(&args, 517);
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let denoiser = IdolDenoiser::load(&dir, ops).expect("the denoiser loads");
    let prepared = denoiser.context(&cond).expect("the conditioning refines");
    let pass = |latent: &[f32], timestep: f32| {
        denoiser
            .forward(latent, &camera, timestep, &prepared, Pass::Full)
            .expect("the pass runs")
    };
    // The first pass refines; the second copies the first's rows.
    let fresh = pass(&latent, 700.0);
    let copied = pass(&latent, 700.0);
    assert_eq!(fresh.latent, copied.latent, "the copy is the refinement");
    // A pass in between at another timestep and another latent replaces the
    // carried rows, so the next one at the first state refines afresh and
    // must answer what it answered before.
    let mut other = latent.clone();
    other[0] += 1.0;
    assert_ne!(pass(&latent, 500.0).latent, fresh.latent);
    assert_ne!(pass(&other, 700.0).latent, fresh.latent);
    denoiser.release();
    assert_eq!(pass(&latent, 700.0).latent, fresh.latent);
}

/// A conditioning of one repeated row attends as one weighted key: the
/// collapsed pass against the same rows held out in full.
#[test]
#[ignore = "needs the splat checkpoint (SHISO_TEST_MODEL_IDOL), the idol fixtures and a Metal device"]
fn a_repeated_conditioning_row_attends_as_one_weighted_key() {
    use crate::idol::denoiser::IdolDenoiser;

    let case = "weighted key";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_IDOL"));
    let cond = need!(case, fixture_conditioning());
    let latent = read_f32(&need!(case, fixture("den_in_latent.f32")));
    let camera = read_f32(&need!(case, fixture("den_in_camera.f32")));
    let backend = need!(case, device());
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let denoiser = IdolDenoiser::load(&dir, ops).expect("the denoiser loads");
    let collapsed = denoiser
        .context(&crate::idol::conditioning::Conditioning::zero(cond.rows))
        .expect("the zero conditioning refines");
    assert_eq!((collapsed.rows, collapsed.copies), (1, cond.rows));
    let full = crate::idol::generator::Prepared {
        rows: cond.rows,
        stream: collapsed.stream.repeat(cond.rows),
        copies: 1,
    };
    let a = denoiser
        .forward(&latent, &camera, 700.0, &collapsed, Pass::Full)
        .expect("the collapsed pass runs");
    let b = denoiser
        .forward(&latent, &camera, 700.0, &full, Pass::Full)
        .expect("the full pass runs");
    // The same sum in another order, through 24 blocks that amplify rounding
    // on this conditioning (see the reference case's bound).
    close_mean(
        &a.latent,
        &b.latent,
        1e-2,
        5e-4,
        "collapsed latent velocity",
    );
    close(&a.camera, &b.camera, 1e-3, "collapsed camera velocity");
}

/// Where one denoiser pass spends its time: the batched pass (lower median
/// of four, after a warm-up) and the per-kernel profile of one more.
#[test]
#[ignore = "needs the splat checkpoint (SHISO_TEST_MODEL_IDOL), the idol fixtures and a Metal device"]
#[expect(
    clippy::print_stderr,
    reason = "the profile table is the case's report"
)]
fn the_denoiser_pass_profile() {
    use crate::idol::denoiser::IdolDenoiser;

    let case = "denoiser profile";
    let dir = need!(case, checkpoint("SHISO_TEST_MODEL_IDOL"));
    let cond = need!(case, fixture_conditioning());
    let latent = read_f32(&need!(case, fixture("den_in_latent.f32")));
    let camera = read_f32(&need!(case, fixture("den_in_camera.f32")));
    let backend = need!(case, device());
    let ops = Arc::new(Ops::new(&backend).expect("the diffusion kernel set"));
    let denoiser = IdolDenoiser::load(&dir, ops).expect("the denoiser loads");
    let prepared = if std::env::var_os("SHISO_IDOL_PROFILE_NEGATIVE").is_some() {
        denoiser.context(&crate::idol::conditioning::Conditioning::zero(cond.rows))
    } else {
        denoiser.context(&cond)
    }
    .expect("the conditioning refines");
    let pass = || {
        denoiser
            .forward(&latent, &camera, 700.0, &prepared, Pass::Full)
            .expect("the pass runs");
    };
    pass();
    let mut times: Vec<f64> = (0..4)
        .map(|_| {
            let began = std::time::Instant::now();
            pass();
            began.elapsed().as_secs_f64()
        })
        .collect();
    eprintln!("batched passes: {times:.3?}");
    times.sort_by(f64::total_cmp);
    let batched = times[1];
    let profiler = backend.as_profiler().expect("the Metal backend profiles");
    let region = shiso_core::ProfileRegion::open(profiler).expect("opens");
    pass();
    let report = region.finish().expect("finishes");
    eprintln!(
        "batched pass (lower median of 4) {batched:.3}s\n{}",
        report.table(40)
    );
}
