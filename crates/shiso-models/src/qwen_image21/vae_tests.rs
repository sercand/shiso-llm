//! The Qwen-Image-2.1 VAE decoder on the host and on the device.
//!
//! The helpers (denormalization fold, pixel conversion, working-set
//! estimate, shortcut channel map) are checked on any machine. The
//! real-weight cases read the export named by `SHISO_TEST_MODEL_QWEN_IMAGE21`
//! (the `turbo-int8` directory) and grade it against
//! `test-vectors/references/qwen_image21/vae_*`, captured from the reference
//! decoder in f32 over the same BF16 weights. Without the variable (or
//! without a Metal device) those cases say so on stderr and return, which
//! looks like a pass - check stderr before reporting one.
#![expect(
    clippy::cast_precision_loss,
    reason = "fixture indices and sizes converted to floats are small, exact in f32"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the f64 oracle sums are narrowed to the f32 the fixtures store"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_core::Backend;
use shiso_vision::generator::ImageGenError;

use super::vae::{
    DecodePlan, LatentGrid, QwenVae, QwenVaeDesc, VaeTap, fold_denorm, rgba8_from_field,
};
use crate::diffusion::ops::{Ops, dup_up_map};
use crate::diffusion::weights::StStore;

// --- host helpers --------------------------------------------------------------

/// The shipped decoder's widths: `conv_in` 1152, up blocks 1152 -> 1152 ->
/// 1152 -> 576 -> 288 -> 144, RGBA out.
fn shipped_plan() -> DecodePlan {
    DecodePlan {
        z_dim: 64,
        mid_c: 1152,
        blocks: vec![
            (1152, 1152, true),
            (1152, 1152, true),
            (1152, 576, true),
            (576, 288, true),
            (288, 144, false),
        ],
        out_c: 4,
    }
}

#[test]
fn the_folded_post_quant_conv_is_the_conv_of_the_denormalized_latent() {
    let (out_c, in_c) = (3usize, 4usize);
    let w: Vec<f32> = (0..out_c * in_c).map(|i| 0.1 * i as f32 - 0.4).collect();
    let b = vec![0.5f32, -0.25, 1.0];
    let mean = vec![0.5126f32, -2.1025, 3.8363, -0.0342];
    let std = vec![3.2001f32, 4.0379, 5.0756, 3.9037];
    let x = [0.3f32, -1.2, 0.7, 2.0];
    let (mut wf, mut bf) = (w.clone(), b.clone());
    fold_denorm(&mut wf, &mut bf, &mean, &std).expect("shapes fit");
    for o in 0..out_c {
        let mut want = f64::from(b[o]);
        let mut got = f64::from(bf[o]);
        for i in 0..in_c {
            let z = f64::from(x[i]) * f64::from(std[i]) + f64::from(mean[i]);
            want += f64::from(w[o * in_c + i]) * z;
            got += f64::from(wf[o * in_c + i]) * f64::from(x[i]);
        }
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1.0),
            "{o}: {got} vs {want}"
        );
    }
    let mut short = vec![0f32; 2];
    assert!(matches!(
        fold_denorm(&mut wf, &mut short, &mean, &std),
        Err(ImageGenError::Checkpoint(_))
    ));
}

#[test]
fn the_rgba_conversion_matches_the_reference_processor() {
    // Two pixels, channel-first: R, G, B, A planes.
    let field = [-1.0f32, 1.0, 0.0, 0.5, 2.0, -3.0, -0.25, 0.999];
    let got = rgba8_from_field(&field, 2).expect("a finite field");
    // round_half_even(clamp(x / 2 + 0.5, 0, 1) * 255): 0 -> 127.5 -> 128,
    // 0.5 -> 191.25 -> 191, -0.25 -> 95.625 -> 96, 0.999 -> 254.87 -> 255.
    assert_eq!(got, vec![0, 128, 255, 96, 255, 191, 0, 255]);
    assert!(matches!(
        rgba8_from_field(&[0.0, f32::NAN, 0.0, 0.0], 1),
        Err(ImageGenError::Backend(_))
    ));
    assert!(rgba8_from_field(&[0.0; 7], 2).is_err());
}

#[test]
fn the_dup_up_map_keeps_the_last_frame_of_each_channel_group() {
    // Temporal-and-spatial blocks: 8 slots per output channel, the single
    // frame keeps slots 4..8.
    assert_eq!(dup_up_map(1152, 1152, 2).expect("fits"), (8, 4, 8));
    assert_eq!(dup_up_map(1152, 576, 2).expect("fits"), (8, 4, 4));
    // Spatial only: 4 slots, all kept.
    assert_eq!(dup_up_map(576, 288, 1).expect("fits"), (4, 0, 2));
    assert!(matches!(
        dup_up_map(576, 100, 1),
        Err(ImageGenError::Checkpoint(_))
    ));
}

#[test]
fn the_working_set_grows_with_the_image_and_refuses_over_budget() {
    let plan = shipped_plan();
    let at = |side: usize| {
        plan.working_set_bytes(LatentGrid {
            h: side / 16,
            w: side / 16,
        })
    };
    let (s256, s512, s1024) = (at(256), at(512), at(1024));
    assert!(s256 < s512 && s512 < s1024);
    // The full-resolution 288- and 144-channel fields dominate: at 512x512
    // one 144-channel plane set is 151 MB, and the estimate is a handful.
    assert!((500 << 20..2 << 30).contains(&s512), "512x512: {s512}");
    let grid = LatentGrid { h: 64, w: 64 };
    plan.check_budget(grid, s1024)
        .expect("exactly the estimate fits");
    match plan.check_budget(grid, s1024 - 1) {
        Err(ImageGenError::DecodeTooLarge {
            width: 1024,
            height: 1024,
            needed_mib,
            ..
        }) => assert_eq!(needed_mib, s1024.div_ceil(1 << 20)),
        other => panic!("expected a DecodeTooLarge refusal, got {other:?}"),
    }
}

// --- the checkpoint --------------------------------------------------------------

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

/// `(weights, config)` of the export, or `None` (said on stderr).
#[expect(
    clippy::print_stderr,
    reason = "a skipped real-weight case must say why on stderr, or it reads as a pass"
)]
fn checkpoint() -> Option<(PathBuf, PathBuf)> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21") else {
        eprintln!(
            "note: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset, skipping the VAE real-weight case"
        );
        return None;
    };
    let root = PathBuf::from(root);
    Some((
        root.join("image/vae"),
        root.join("image_assets/vae/config.json"),
    ))
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

/// PSNR of one channel plane in [-1, 1] (peak-to-peak 2).
fn psnr(got: &[f32], want: &[f32]) -> f64 {
    let mse = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| (f64::from(g) - f64::from(w)).powi(2))
        .sum::<f64>()
        / want.len() as f64;
    10.0 * (4.0 / mse.max(1e-30)).log10()
}

#[derive(serde::Deserialize)]
struct Stats {
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

/// The unpack and the folded denormalization on the host, against the
/// reference's `post_quant_conv` output: tokens `(h*w, 64)` read as the
/// channel-first `(64, h, w)` field, through the folded 1x1 convolution.
#[test]
fn the_host_unpack_and_folded_denorm_reproduce_post_quant_conv() {
    let Some((weights, config)) = checkpoint() else {
        return;
    };
    let store = StStore::open(&weights).expect("the VAE opens");
    let stats: Stats =
        serde_json::from_str(&std::fs::read_to_string(&config).expect("config")).expect("json");
    let mut w = store.f32_vec("conv2.weight").expect("post_quant weight");
    let mut b = store.f32_vec("conv2.bias").expect("post_quant bias");
    fold_denorm(&mut w, &mut b, &stats.latents_mean, &stats.latents_std).expect("fits");
    let dir = fixture_dir();
    let tokens = read_f32_file(&dir.join("vae_latents_packed.f32"));
    let want = read_f32_file(&dir.join("vae_post_quant_conv.f32"));
    let (cells, z) = (256usize, 64usize);
    // Unpack: field[c][t] = tokens[t][c].
    let mut field = vec![0f32; z * cells];
    for t in 0..cells {
        for c in 0..z {
            field[c * cells + t] = tokens[t * z + c];
        }
    }
    let denorm_want = read_f32_file(&dir.join("vae_latents_post_denorm.f32"));
    let denorm: Vec<f32> = field
        .iter()
        .enumerate()
        .map(|(i, &v)| v * stats.latents_std[i / cells] + stats.latents_mean[i / cells])
        .collect();
    assert!(rel_l2(&denorm, &denorm_want) < 1e-6, "the unpack layout");
    let mut got = vec![0f32; z * cells];
    for o in 0..z {
        for t in 0..cells {
            let mut acc = f64::from(b[o]);
            for i in 0..z {
                acc += f64::from(w[o * z + i]) * f64::from(field[i * cells + t]);
            }
            got[o * cells + t] = acc as f32;
        }
    }
    let err = rel_l2(&got, &want);
    assert!(err <= 1e-5, "host post_quant rel L2 {err:e}");
}

#[expect(
    clippy::print_stderr,
    reason = "the device cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn device() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the VAE device case");
        return None;
    }
    let metal = shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let backend: Arc<dyn Backend> = metal;
    Some(backend)
}

fn load(ops: &Ops, weights: &Path, config: &Path) -> QwenVae {
    QwenVae::load(&QwenVaeDesc { weights, config }, ops).expect("the VAE decoder loads")
}

#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
#[expect(
    clippy::print_stderr,
    reason = "the measured agreement is the case's report"
)]
fn the_decoder_matches_the_reference_on_the_device() {
    let Some((weights, config)) = checkpoint() else {
        return;
    };
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let vae = load(&ops, &weights, &config);
    let dir = fixture_dir();
    let tokens = read_f32_file(&dir.join("vae_latents_packed.f32"));
    let d_tokens = ops.upload_f32("tokens", &tokens).expect("upload");
    let grid = LatentGrid { h: 16, w: 16 };
    let mut taps: Vec<(VaeTap, Vec<f32>)> = Vec::new();
    let mut tap = |stage: VaeTap, view: shiso_core::DeviceView<'_>, count: usize| {
        taps.push((stage, ops.download("tap", view, count)?));
        Ok(())
    };
    let image = vae
        .decode_field(&ops, d_tokens.view(), grid, u64::MAX, Some(&mut tap))
        .expect("the decode runs");
    for (stage, file) in [
        (VaeTap::PostQuant, "vae_post_quant_conv.f32"),
        (VaeTap::ConvIn, "vae_decoder_conv_in.f32"),
        (VaeTap::MidBlock, "vae_mid_block.f32"),
    ] {
        let want = read_f32_file(&dir.join(file));
        let got = &taps
            .iter()
            .find(|(s, _)| *s == stage)
            .unwrap_or_else(|| panic!("{stage:?} was tapped"))
            .1;
        let err = rel_l2(got, &want);
        eprintln!("{stage:?}: rel L2 {err:.3e}");
        assert!(err <= 1e-3, "{stage:?} rel L2 {err:e}");
    }
    let want = read_f32_file(&dir.join("vae_image.f32"));
    let plane = 256 * 256;
    assert_eq!(image.len(), 4 * plane);
    let total = rel_l2(&image, &want);
    for (c, name) in ["R", "G", "B", "A"].iter().enumerate() {
        let p = psnr(
            &image[c * plane..(c + 1) * plane],
            &want[c * plane..(c + 1) * plane],
        );
        eprintln!("image {name}: PSNR {p:.2} dB");
        assert!(p >= 40.0, "channel {name} PSNR {p:.2} dB");
    }
    eprintln!("image: rel L2 {total:.3e}");
    let rgba = rgba8_from_field(&image, plane).expect("finite");
    assert_eq!(rgba.len(), 4 * plane);
}

/// A deterministic latent with roughly unit spread (sum of four uniforms,
/// recentred), standing in for the transformer's output.
fn gaussian_like(len: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9u32;
    let mut uniform = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    (0..len)
        .map(|_| (uniform() + uniform() + uniform() + uniform() - 2.0) * 1.7)
        .collect()
}

#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
#[expect(
    clippy::print_stderr,
    reason = "the measured times are the case's report"
)]
fn decode_time_at_256_and_512() {
    let Some((weights, config)) = checkpoint() else {
        return;
    };
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let vae = load(&ops, &weights, &config);
    for side in [256usize, 512] {
        let grid = LatentGrid {
            h: side / 16,
            w: side / 16,
        };
        let tokens = gaussian_like(grid.cells() * 64);
        let d_tokens = ops.upload_f32("tokens", &tokens).expect("upload");
        let mut times = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            let rgba = vae
                .decode(&ops, d_tokens.view(), grid, u64::MAX)
                .expect("the decode runs");
            times.push(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(rgba.len(), side * side * 4);
        }
        eprintln!(
            "decode {side}x{side}: {:.0} ms cold, {:.0} / {:.0} ms warm (estimate {} MiB)",
            times[0],
            times[1],
            times[2],
            vae.plan().working_set_bytes(grid) >> 20
        );
        ops.trim_pool();
    }
}

#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn an_image_over_budget_is_refused_before_the_device_runs() {
    let Some((weights, config)) = checkpoint() else {
        return;
    };
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let vae = load(&ops, &weights, &config);
    assert_eq!(vae.plan(), &shipped_plan(), "the shipped widths");
    assert_eq!(vae.out_channels(), 4);
    let grid = LatentGrid { h: 64, w: 64 };
    // A token buffer too small for the grid: the refusal must come first.
    let d_tokens = ops.upload_f32("tokens", &[0.0; 64]).expect("upload");
    let budget = vae.plan().working_set_bytes(grid) - 1;
    assert!(matches!(
        vae.decode(&ops, d_tokens.view(), grid, budget),
        Err(ImageGenError::DecodeTooLarge { .. })
    ));
}
