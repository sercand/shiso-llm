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
    reason = "the comparison's rms is an f64 sum narrowed to the f32 the device computes in"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::Backend;

use crate::diffusion::ops::Ops;
use crate::flux2::vae_encoder::VaeEncoder;
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
    let v = denoiser
        .forward(&latent, &camera, 700.0, &cond)
        .expect("the denoiser runs");
    // Measured: worst 3.2e-4, mean 9.3e-6.
    close_mean(
        &v.latent,
        &out_latent,
        1e-3,
        5e-5,
        "denoiser latent velocity",
    );
    close(&v.camera, &out_camera, 1e-4, "denoiser camera velocity");
    let negative = crate::idol::conditioning::Conditioning::zero(cond.rows);
    let v = denoiser
        .forward(&latent, &camera, 700.0, &negative)
        .expect("the negative pass runs");
    // Zero conditioning gives 4101 near-identical context rows, and the 24
    // joint blocks amplify f32 rounding more there: the streams agree to
    // 2.2e-6 after the first joint block and the velocity to a mean of 2e-4
    // (worst 4.1e-3).
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
