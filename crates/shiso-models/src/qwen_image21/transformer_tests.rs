//! The transformer on the device against the independent reference.
//!
//! Every case reads the export named by `SHISO_TEST_MODEL_QWEN_IMAGE21` (an
//! install directory holding `image/diffusion_models/` and `image_assets/`,
//! e.g. the `turbo-int8` staging directory) and grades it against
//! `test-vectors/references/qwen_image21/{embed,block}*`, captured from the
//! reference transformer in f32 over the official BF16 weights. The
//! non-block weights are BF16 in both, so the embedders are held tight;
//! the blocks here are INT8 with a rotated input against the reference's
//! BF16, so they are graded by per-row cosine. Without the variable, or
//! without a Metal device, a case says so on stderr and returns, which
//! looks like a pass - check stderr before reporting one.
#![expect(
    clippy::cast_precision_loss,
    reason = "fixture sizes converted to floats are small, exact in f64"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture sigmas narrowed to the f32 the pipeline passes"
)]
#![expect(
    clippy::print_stderr,
    reason = "the measured agreement and timings are the device cases' report, and a skip must say why"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;
use shiso_core::Backend;

use super::config::QwenImageConfig;
use super::transformer::{DitDesc, QwenDit};
use super::vae::LatentGrid;
use crate::diffusion::ops::Ops;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/qwen_image21")
}

fn read_f32(name: &str) -> Vec<f32> {
    let path = fixture_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

fn read_json(name: &str) -> Value {
    let path = fixture_dir().join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("the fixture parses")
}

/// The install root, or `None` (said on stderr).
fn install() -> Option<PathBuf> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21") else {
        eprintln!("note: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset, skipping the transformer case");
        return None;
    };
    Some(PathBuf::from(root))
}

fn device() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the transformer case");
        return None;
    }
    let metal = shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let backend: Arc<dyn Backend> = metal;
    Some(backend)
}

/// Opens the device and the install, and loads checkpoint blocks `which`.
fn open(which: Option<&[usize]>) -> Option<(Ops, QwenDit)> {
    let root = install()?;
    let backend = device()?;
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let config =
        QwenImageConfig::read(&root.join("image_assets")).expect("the install's documents");
    let weights = root.join("image/diffusion_models");
    let desc = DitDesc {
        weights: &weights,
        config: &config.dit,
    };
    let dit = match which {
        Some(blocks) => QwenDit::load_blocks(&desc, &ops, blocks),
        None => QwenDit::load(&desc, &ops),
    }
    .expect("the transformer loads");
    Some((ops, dit))
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

/// `(min, mean)` cosine over `width`-wide rows.
fn row_cosines(got: &[f32], want: &[f32], width: usize) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "length");
    let cos: Vec<f64> = got
        .chunks_exact(width)
        .zip(want.chunks_exact(width))
        .map(|(g, w)| {
            let (mut dot, mut gg, mut ww) = (0f64, 0f64, 0f64);
            for (&a, &b) in g.iter().zip(w) {
                dot += f64::from(a) * f64::from(b);
                gg += f64::from(a).powi(2);
                ww += f64::from(b).powi(2);
            }
            dot / (gg.sqrt() * ww.sqrt()).max(f64::MIN_POSITIVE)
        })
        .collect();
    let min = cos.iter().copied().fold(f64::INFINITY, f64::min);
    (min, cos.iter().sum::<f64>() / cos.len() as f64)
}

/// `temb`, the raw shared modulation and `txt_in` against the reference at
/// four sigmas (the non-block weights are BF16 on both sides).
#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn the_embedders_match_the_reference() {
    let Some((ops, dit)) = open(Some(&[0])) else {
        return;
    };
    let embed = read_json("embed.json");
    let hidden = 4096;
    for (name, row) in embed["rows"].as_object().expect("rows") {
        let sigma = row["sigma"].as_f64().expect("sigma") as f32;
        let cond = dit
            .embedders()
            .time_cond(&ops, sigma)
            .expect("conditioning");
        let temb = ops
            .download("temb", cond.temb.view(), hidden)
            .expect("read");
        let modulation = ops
            .download("mod", cond.modulation.view(), 4 * hidden)
            .expect("read");
        let (et, em) = (
            rel_l2(&temb, &read_f32(row["temb"].as_str().expect("file"))),
            rel_l2(
                &modulation,
                &read_f32(row["modulation"].as_str().expect("file")),
            ),
        );
        eprintln!("{name}: temb rel L2 {et:.3e}, modulation rel L2 {em:.3e}");
        assert!(et <= 1e-3 && em <= 1e-3, "{name}");
    }
    let x = read_f32("embed_txt_in_x.f32");
    let rows = x.len() / hidden;
    let feats = ops.upload_f32("txt", &x).expect("upload");
    let out = dit
        .embedders()
        .txt_in(&ops, feats.view(), rows)
        .expect("txt_in");
    let got = ops
        .download("txt_in", out.view(), rows * hidden)
        .expect("read");
    let err = rel_l2(&got, &read_f32("embed_txt_in_out.f32"));
    eprintln!("txt_in: rel L2 {err:.3e}");
    assert!(err <= 1e-3, "txt_in rel L2 {err:e}");
}

/// A one-layer model over checkpoint block 0, then block 31 (INT8), against
/// the reference's one-layer BF16 model: 20 text tokens through the prefix,
/// a 4 x 4 target at sigma 0.704534 through the cached path.
#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn one_block_matches_the_reference_on_the_cached_path() {
    let block = read_json("block.json");
    let sigma = block["sigma"].as_f64().expect("sigma") as f32;
    let (hidden, head_dim, p, channels) = (4096usize, 128usize, 20usize, 64usize);
    let grid = LatentGrid { h: 4, w: 4 };
    let t = grid.cells();
    let encoder = read_f32("block_in_encoder.f32");
    let latents = read_f32("block_in_latents.f32");
    for (case, index) in [("block00", 0usize), ("block31", 31)] {
        let Some((ops, dit)) = open(Some(&[index])) else {
            return;
        };
        let feats = ops.upload_f32("encoder", &encoder).expect("upload");
        let lat = ops.upload_f32("latents", &latents).expect("upload");

        let txt = dit
            .embedders()
            .txt_in(&ops, feats.view(), p)
            .expect("txt_in");
        let got = ops
            .download("txt_in", txt.view(), p * hidden)
            .expect("read");
        let err = rel_l2(&got, &read_f32(&format!("{case}_txt_in.f32")));
        eprintln!("{case}: txt_in rel L2 {err:.3e}");
        assert!(err <= 1e-3, "{case} txt_in");

        // The text rows of the full block-causal pass's block output are
        // what the prefix's own (causal, t = 0) block computes; production
        // skips the last layer's text stream, so this reads it whole.
        let (_, text_stream) = dit
            .prefix_with_stream(&ops, feats.view(), p)
            .expect("the prefix with its stream");
        let text_out = ops
            .download("text stream", text_stream.view(), p * hidden)
            .expect("read");
        drop(text_stream);
        let full_out = read_f32(&format!("{case}_block_out.f32"));
        let (tmin, tmean) = row_cosines(&text_out, &full_out[..p * hidden], hidden);
        eprintln!("{case}: prefix text block out cosine min {tmin:.6} mean {tmean:.6}");
        assert!(tmin >= 0.999, "{case} prefix text stream");

        let mut kv = dit.prefix(&ops, feats.view(), p).expect("the prefix");
        assert_eq!(kv.tokens(), p);
        let keys = ops
            .download("k", kv.keys(0).expect("layer 0"), p * hidden)
            .expect("read");
        let k_head0: Vec<f32> = keys
            .chunks_exact(hidden)
            .flat_map(|row| row[..head_dim].iter().copied())
            .collect();
        let want_k = read_f32(&format!("{case}_k_head0.f32"));
        let (kmin, kmean) = row_cosines(&k_head0, &want_k[..p * head_dim], head_dim);
        eprintln!("{case}: prefix k head 0 cosine min {kmin:.6} mean {kmean:.6}");
        assert!(kmin >= 0.999, "{case} prefix keys");

        dit.prepare_steps(&ops, &mut kv, &[sigma, 1.0], grid)
            .expect("the step tables upload");
        let (x, _) = dit
            .target_stream(&ops, &kv, lat.view(), grid, sigma)
            .expect("the target stream");
        let stream = ops.download("stream", x.view(), t * hidden).expect("read");
        drop(x);
        let (bmin, bmean) = row_cosines(
            &stream,
            &read_f32(&format!("{case}_cached_block_out.f32")),
            hidden,
        );
        eprintln!("{case}: cached block out cosine min {bmin:.6} mean {bmean:.6}");
        assert!(bmin >= 0.999, "{case} block out");

        let out = ops.alloc_f32("velocity", t * channels).expect("alloc");
        dit.velocity(&ops, &kv, lat.view(), grid, sigma, out.view())
            .expect("the velocity");
        let velocity = ops
            .download("velocity", out.view(), t * channels)
            .expect("read");
        let want = read_f32(&format!("{case}_cached_out.f32"));
        let (vmin, vmean) = row_cosines(&velocity, &want, channels);
        eprintln!(
            "{case}: model out cosine min {vmin:.6} mean {vmean:.6}, rel L2 {:.3e}",
            rel_l2(&velocity, &want)
        );
        assert!(vmin >= 0.998, "{case} model out");
        // The same prefix at another sigma still answers (the table and the
        // cache are reused, not consumed).
        dit.velocity(&ops, &kv, lat.view(), grid, 1.0, out.view())
            .expect("a second step");
        let again = ops
            .download("velocity", out.view(), t * channels)
            .expect("read");
        let (amin, _) = row_cosines(
            &again,
            &read_f32(&format!("{case}_prefill_target_out.f32")),
            channels,
        );
        eprintln!("{case}: sigma 1.0 target out cosine min {amin:.6}");
        assert!(amin >= 0.998, "{case} sigma 1 out");
    }
}

/// A deterministic spread of values with roughly unit variance.
fn gaussian_like(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 0x9E37_79B9;
    let mut uniform = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32
    };
    (0..len)
        .map(|_| (uniform() + uniform() + uniform() + uniform() - 2.0) * 1.7)
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The whole 32-layer model at 512x512 (a 32 x 32 latent grid) after a
/// 30-token prompt: the prefix once, then one velocity call, each timed
/// warm (one discarded run, then the median of three).
#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn full_model_time_at_512() {
    let started = Instant::now();
    let Some((ops, dit)) = open(None) else {
        return;
    };
    eprintln!("load: {:.0} ms", started.elapsed().as_secs_f64() * 1e3);
    let (p, grid) = (30usize, LatentGrid { h: 32, w: 32 });
    let feats = ops
        .upload_f32("feats", &gaussian_like(p * 4096, 7))
        .expect("upload");
    let lat = ops
        .upload_f32("latents", &gaussian_like(grid.cells() * 64, 8))
        .expect("upload");
    let out = ops.alloc_f32("velocity", grid.cells() * 64).expect("alloc");
    let sync = || ops.backend().sync().expect("the queue drains");
    let mut prefix_ms = Vec::new();
    let mut kv = None;
    for _ in 0..4 {
        let t0 = Instant::now();
        kv = Some(dit.prefix(&ops, feats.view(), p).expect("prefix"));
        sync();
        prefix_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mut kv = kv.expect("a prefix ran");
    dit.prepare_steps(&ops, &mut kv, &[0.9], grid)
        .expect("the step tables upload");
    let mut velocity_ms = Vec::new();
    for _ in 0..4 {
        let t0 = Instant::now();
        dit.velocity(&ops, &kv, lat.view(), grid, 0.9, out.view())
            .expect("velocity");
        sync();
        velocity_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let v = ops
        .download("velocity", out.view(), grid.cells() * 64)
        .expect("read");
    assert!(v.iter().all(|x| x.is_finite()), "a finite velocity");
    eprintln!(
        "prefix (P={p}): runs {prefix_ms:.0?} ms, warm median {:.0} ms",
        median(prefix_ms[1..].to_vec())
    );
    eprintln!(
        "velocity (512x512, T={}): runs {velocity_ms:.0?} ms, warm median {:.0} ms",
        grid.cells(),
        median(velocity_ms[1..].to_vec())
    );
}
