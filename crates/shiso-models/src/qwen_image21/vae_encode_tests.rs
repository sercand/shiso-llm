//! The Qwen-Image-2.1 VAE encoder on the host and on the device.
//!
//! The helpers (the pixel field, the normalization fold, the shortcut's
//! group size, the working-set estimate) are checked on any machine. The
//! real-weight cases read the export named by `SHISO_TEST_MODEL_QWEN_IMAGE21`
//! and grade it against `test-vectors/references/qwen_image21/vae_encode_*`,
//! captured by `qwen_image21_reference.py vae-encode` from the reference
//! encoder in f32 over the same BF16 weights. Without the variable (or
//! without a Metal device) those cases say so on stderr and return, which
//! looks like a pass - check stderr before reporting one.
#![expect(
    clippy::cast_precision_loss,
    reason = "fixture indices and sizes converted to floats are small, exact in f32"
)]
#![expect(
    clippy::print_stderr,
    reason = "the measured agreement is the case's report, and a skip must say why"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::Backend;
use shiso_vision::generator::ImageGenError;

use super::vae::{LatentGrid, QwenVae, QwenVaeDesc};
use super::vae_encode::{
    EncodePlan, QwenVaeEncoder, VaeEncodeTap, fold_latent_norm, pad_in_channels, rgba_field,
};
use crate::diffusion::ops::{Ops, avg_down_group};
use crate::diffusion::weights::StStore;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/qwen_image21")
}

fn read_f32_file(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// The capture's picture, interleaved RGBA8: `rgb(x, y, c) = (7x + 13y +
/// 29c) mod 256`, `a(x, y) = (3x + 5y) mod 256`.
fn capture_picture(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            let byte = |v: u32| u8::try_from(v % 256).expect("a byte");
            [
                byte(7 * x + 13 * y),
                byte(7 * x + 13 * y + 29),
                byte(7 * x + 13 * y + 58),
                byte(3 * x + 5 * y),
            ]
        })
        .collect()
}

/// `||got - want|| / ||want||`.
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length");
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (f64::from(g) - f64::from(w)).powi(2);
        den += f64::from(w).powi(2);
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

// --- host helpers --------------------------------------------------------------

#[test]
fn rgba_bytes_become_the_reference_input_field() {
    let field = rgba_field(&[0, 255, 51, 102, 255, 0, 204, 153], 2).expect("two pixels");
    // Channel-first planes of `2 * v / 255 - 1`.
    let want = [-1.0f32, 1.0, 1.0, -1.0, -0.6, 0.6, -0.2, 0.2];
    for (got, want) in field.iter().zip(want) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
    assert!(rgba_field(&[0; 7], 2).is_err());

    // The whole capture picture against the reference processor's output.
    let want = read_f32_file(&fixture_dir().join("vae_encode_input.f32"));
    let got = rgba_field(&capture_picture(192, 128), 192 * 128).expect("the picture");
    assert_eq!(got.len(), want.len());
    let worst = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(worst <= 1e-6, "largest difference {worst}");
}

/// The first convolution is widened to sixteen input channels so its
/// reduction is one the tiled kernel stages: each output's own taps keep
/// their values and their `[in][tap]` order, and every added channel's are
/// zero, so the padded convolution over a zero-padded field is the stored
/// one.
#[test]
fn the_first_convolution_is_padded_with_zero_channels() {
    // Two outputs, three inputs, two taps.
    let stored: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let padded = pad_in_channels(&stored, 2, 3, 5).expect("the shape fits");
    assert_eq!(
        padded,
        [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, //
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0,
        ]
    );
    assert_eq!(
        pad_in_channels(&stored, 2, 3, 3).expect("no padding"),
        stored
    );
    assert!(pad_in_channels(&stored, 2, 3, 2).is_err(), "narrowing");
    assert!(
        pad_in_channels(&stored, 5, 3, 4).is_err(),
        "a ragged weight"
    );
}

#[test]
fn the_folded_quant_conv_emits_the_normalised_mode() {
    // A 2-channel latent: the quant conv is 4 -> 4 and the mode is its
    // first two outputs.
    let z = 2usize;
    let w: Vec<f32> = (0..16).map(|i| 0.25 * i as f32 - 1.5).collect();
    let b = vec![0.5f32, -0.25, 9.0, 9.0];
    let (mean, std) = (vec![0.5126f32, -2.1025], vec![3.2001f32, 4.0379]);
    let x = [0.3f32, -1.2, 0.7, 2.0];
    let (wf, bf) = fold_latent_norm(&w, &b, &mean, &std).expect("shapes fit");
    assert_eq!((wf.len(), bf.len()), (z * 4, z));
    for o in 0..z {
        let raw: f64 = (0..4)
            .map(|i| f64::from(w[o * 4 + i]) * f64::from(x[i]))
            .sum::<f64>()
            + f64::from(b[o]);
        let want = (raw - f64::from(mean[o])) / f64::from(std[o]);
        let got: f64 = (0..4)
            .map(|i| f64::from(wf[o * 4 + i]) * f64::from(x[i]))
            .sum::<f64>()
            + f64::from(bf[o]);
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1.0),
            "{o}: {got} vs {want}"
        );
    }
    assert!(matches!(
        fold_latent_norm(&w, &b, &mean, &[1.0]),
        Err(ImageGenError::Checkpoint(_))
    ));
    assert!(matches!(
        fold_latent_norm(&w, &b[..3], &mean, &std),
        Err(ImageGenError::Checkpoint(_))
    ));
}

#[test]
fn the_down_shortcut_averages_four_folded_channels_in_every_shipped_block() {
    // Spatial only, then three blocks that also fold a (zero) front frame.
    assert_eq!(avg_down_group(96, 96, 2, 1).expect("fits"), 4);
    assert_eq!(avg_down_group(96, 192, 2, 2).expect("fits"), 4);
    assert_eq!(avg_down_group(384, 768, 2, 2).expect("fits"), 4);
    // The last block folds nothing: the shortcut is the identity.
    assert_eq!(avg_down_group(768, 768, 1, 1).expect("fits"), 1);
    assert!(matches!(
        avg_down_group(96, 100, 2, 1),
        Err(ImageGenError::Checkpoint(_))
    ));
}

// --- real weights ---------------------------------------------------------------

fn setup() -> Option<(PathBuf, PathBuf, Arc<dyn Backend>)> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21").map(PathBuf::from) else {
        eprintln!("note: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset, skipping the VAE encoder case");
        return None;
    };
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the VAE encoder case");
        return None;
    }
    let metal = shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let backend: Arc<dyn Backend> = metal;
    Some((
        root.join("image/vae"),
        root.join("image_assets/vae/config.json"),
        backend,
    ))
}

#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn encode_matches_reference() {
    let Some((weights, config, backend)) = setup() else {
        return;
    };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let encoder = QwenVaeEncoder::load(
        &QwenVaeDesc {
            weights: &weights,
            config: &config,
        },
        &ops,
    )
    .expect("the VAE encoder loads");
    let dir = fixture_dir();
    let grid = LatentGrid { h: 8, w: 12 };
    // The estimate grows with the picture and is dominated by the
    // full-resolution blocks: a 1024x1024 encode is a few gigabytes.
    // The plan read from the file's shapes is the loaded encoder's.
    let plan = EncodePlan::read(&StStore::open(&weights).expect("the VAE file opens"), 64)
        .expect("the shapes describe an encoder");
    assert_eq!(plan, encoder.plan());
    let at = |side: usize| plan.working_set_bytes(LatentGrid { h: side, w: side });
    assert!(at(16) < at(32) && at(32) < at(64));
    eprintln!("working set at 1024x1024: {} MiB", at(64) >> 20);
    assert!((1u64 << 30..8 << 30).contains(&at(64)), "{}", at(64));
    let picture = capture_picture(192, 128);
    let mut taps: Vec<(VaeEncodeTap, Vec<f32>)> = Vec::new();
    let mut tap = |stage: VaeEncodeTap, view: shiso_core::DeviceView<'_>, count: usize| {
        taps.push((stage, ops.download("tap", view, count)?));
        Ok(())
    };
    let began = std::time::Instant::now();
    let latents = encoder
        .encode_with(&ops, &picture, grid, Some(&mut tap))
        .expect("the encode runs");
    let got = ops
        .download("latents", latents.view(), grid.cells() * 64)
        .expect("readback");
    eprintln!("encode 192x128: {:.3} s", began.elapsed().as_secs_f64());
    for (stage, file) in [
        (VaeEncodeTap::Down(0), "vae_encode_down0.f32"),
        (VaeEncodeTap::Down(1), "vae_encode_down1.f32"),
        (VaeEncodeTap::Mid, "vae_encode_mid.f32"),
    ] {
        let want = read_f32_file(&dir.join(file));
        let stage_got = &taps
            .iter()
            .find(|(s, _)| *s == stage)
            .unwrap_or_else(|| panic!("{stage:?} was tapped"))
            .1;
        let err = rel_l2(stage_got, &want);
        eprintln!("{stage:?}: rel L2 {err:.3e}");
        assert!(err <= 1e-3, "{stage:?} rel L2 {err:e}");
    }
    let want = read_f32_file(&dir.join("vae_encode_latents.f32"));
    let err = rel_l2(&got, &want);
    eprintln!("latents: rel L2 {err:.3e}");
    assert!(err <= 1e-3, "latents rel L2 {err:e}");
}

/// A smooth picture with a hard-edged opaque disc on a half-transparent
/// ground: something a 16x autoencoder should bring back closely.
fn round_trip_picture(side: u32) -> Vec<u8> {
    let centre = i64::from(side / 2);
    let radius = i64::from(side / 4);
    let ramp = |v: u32| u8::try_from(v * 255 / side).expect("a byte");
    (0..side)
        .flat_map(|y| (0..side).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            let (dx, dy) = (i64::from(x) - centre, i64::from(y) - centre);
            if dx * dx + dy * dy < radius * radius {
                [220, ramp(x), 40, 255]
            } else {
                [ramp(x), ramp(y), 128, 128]
            }
        })
        .collect()
}

#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn encode_then_decode_round_trips() {
    let Some((weights, config, backend)) = setup() else {
        return;
    };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let desc = QwenVaeDesc {
        weights: &weights,
        config: &config,
    };
    let encoder = QwenVaeEncoder::load(&desc, &ops).expect("the VAE encoder loads");
    let decoder = QwenVae::load(&desc, &ops).expect("the VAE decoder loads");
    let side = 256u32;
    let grid = LatentGrid { h: 16, w: 16 };
    let picture = round_trip_picture(side);
    let latents = encoder
        .encode(&ops, &picture, grid)
        .expect("the encode runs");
    let image = decoder
        .decode_field(&ops, latents.view(), grid, u64::MAX, None)
        .expect("the decode runs");
    let plane = (side * side) as usize;
    let want = rgba_field(&picture, plane).expect("the picture");
    for (c, name) in ["R", "G", "B", "A"].iter().enumerate() {
        let mse = image[c * plane..(c + 1) * plane]
            .iter()
            .zip(&want[c * plane..(c + 1) * plane])
            .map(|(&g, &w)| (f64::from(g) - f64::from(w)).powi(2))
            .sum::<f64>()
            / plane as f64;
        let psnr = 10.0 * (4.0 / mse.max(1e-30)).log10();
        eprintln!("round trip {name}: PSNR {psnr:.2} dB");
        assert!(psnr >= 30.0, "channel {name} PSNR {psnr:.2} dB");
    }
}
