//! The transformer's interleaved prefix: positions and segments on the
//! host, and one block over text, two condition images and a target on the
//! device.
//!
//! `test-vectors/references/qwen_image21/edit_block*` were captured by
//! `scripts/reference/qwen_image21_reference.py edit-block`: the reference
//! one-layer model in f32 over this install's own INT8 export, dequantized
//! and un-rotated, so the device case that runs the same weights is graded
//! on its kernels alone. The device cases read the export named by
//! `SHISO_TEST_MODEL_QWEN_IMAGE21`; without it, or without a Metal device,
//! a case says so on stderr and returns, which looks like a pass - check
//! stderr before reporting one.
#![expect(
    clippy::print_stderr,
    reason = "the measured agreement is the device cases' report, and a skip must say why"
)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use shiso_core::Backend;

use super::config::QwenImageConfig;
use super::rope::{Segment, angle_table, prefix_positions, segment_rows, target_positions_at};
use super::transformer::{DitDesc, PrefixInput, QwenDit};
use super::vae::LatentGrid;
use crate::diffusion::ops::{Ops, SliceExt};

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

/// The capture's layout: text 20, a 4x6 condition image, text 3, a 2x2
/// condition image; the target is 4x4.
fn capture_layout() -> Vec<Segment> {
    vec![
        Segment::Text(20),
        Segment::Image(LatentGrid { h: 4, w: 6 }),
        Segment::Text(3),
        Segment::Image(LatentGrid { h: 2, w: 2 }),
    ]
}

// --- host ------------------------------------------------------------------------

/// The interleaved positions against the reference's own rotary table:
/// text advances all three axes together, a picture holds the frame axis
/// where the text reached, centres its grid, and moves the position on by
/// its longer side; the target follows the same rule.
#[test]
fn prefix_positions_match_reference() {
    let layout = capture_layout();
    let (mut positions, frame) = prefix_positions(&layout);
    assert_eq!(positions.len(), 20 + 24 + 3 + 4);
    // 20 text, then the 4x6 picture at frame 20 moving on by 6, three text
    // at 26..29, the 2x2 picture at frame 29 moving on by 2.
    assert_eq!(positions[19], [19, 19, 19]);
    assert_eq!(positions[20], [20, -2, -3]);
    assert_eq!(positions[43], [20, 1, 2]);
    assert_eq!(positions[44], [26, 26, 26]);
    assert_eq!(positions[47], [29, -1, -1]);
    assert_eq!(frame, 31);
    positions.extend(target_positions_at(frame, LatentGrid { h: 4, w: 4 }));
    let (cos, sin) = angle_table(&positions, [16, 56, 56]);
    for (name, got) in [
        ("edit_block_rope_cos.f32", cos),
        ("edit_block_rope_sin.f32", sin),
    ] {
        let want = read_f32(name);
        assert_eq!(got.len(), want.len(), "{name}");
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        assert!(worst <= 2e-4, "{name}: largest difference {worst}");
    }
}

/// Two pictures with no text between them are two blocks: each has its own
/// frame position and its own segment, so neither attends the other in
/// both directions.
#[test]
fn adjacent_images_are_separate_blocks() {
    let grid = LatentGrid { h: 2, w: 2 };
    let layout = [Segment::Text(2), Segment::Image(grid), Segment::Image(grid)];
    let (positions, frame) = prefix_positions(&layout);
    assert_eq!(positions[2][0], 2, "the first picture's frame");
    assert_eq!(positions[6][0], 4, "the second picture's frame");
    assert_eq!(frame, 6, "the target's frame");
    assert_eq!(
        segment_rows(&layout),
        vec![(0..2, false), (2..6, true), (6..10, true)]
    );
    // A text-only prefix is one causal segment, with the target at its end.
    let text = [Segment::Text(5)];
    assert_eq!(segment_rows(&text), vec![(0..5, false)]);
    assert_eq!(prefix_positions(&text).1, 5);
}

/// Opens the device and the install, and loads checkpoint block `index`
/// as a one-layer model.
fn open(index: usize) -> Option<(Ops, QwenDit)> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21").map(PathBuf::from) else {
        eprintln!("note: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset, skipping the edit block case");
        return None;
    };
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the edit block case");
        return None;
    }
    let metal = shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let backend: Arc<dyn Backend> = metal;
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let config =
        QwenImageConfig::read(&root.join("image_assets")).expect("the install's documents");
    let weights = root.join("image/diffusion_models");
    let dit = QwenDit::load_blocks(
        &DitDesc {
            weights: &weights,
            config: &config.dit,
        },
        &ops,
        &[index],
    )
    .expect("the transformer loads");
    Some((ops, dit))
}

/// A deterministic spread of values in [-1, 1).
fn signal(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from(u16::try_from(state >> 16).expect("16 bits")) / 32_768.0 - 1.0
        })
        .collect()
}

fn digest(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for v in values {
        hasher.update(v.to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .take(8)
        .fold(String::new(), |mut hex, b| {
            let _ = write!(hex, "{b:02x}");
            hex
        })
}

/// A text-only prefix and one step over it, as digests of the bits. It
/// asserts nothing: it prints the number to compare by hand across a change
/// that must leave text-to-image alone (the digest belongs to the
/// checkpoint it was run on).
#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn text_only_prefix_digest() {
    let Some((ops, dit)) = open(0) else { return };
    let (p, hidden, channels) = (20usize, 4096usize, 64usize);
    let grid = LatentGrid { h: 4, w: 4 };
    let feats = ops
        .upload_f32("encoder", &signal(p * hidden, 7))
        .expect("upload");
    let latents = ops
        .upload_f32("latents", &signal(grid.cells() * channels, 8))
        .expect("upload");
    let mut kv = dit.prefix(&ops, feats.view(), p).expect("the prefix");
    let keys = kv.keys(&ops, 0).expect("layer 0");
    dit.prepare_steps(&ops, &mut kv, &[0.5], grid)
        .expect("the step tables");
    let out = ops
        .alloc_f32("velocity", grid.cells() * channels)
        .expect("alloc");
    dit.velocity(&ops, &kv, latents.view(), grid, 0.5, out.view())
        .expect("the velocity");
    let velocity = ops
        .download("velocity", out.view(), grid.cells() * channels)
        .expect("read");
    eprintln!(
        "text-only digest: keys {} velocity {}",
        digest(&keys),
        digest(&velocity)
    );
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

/// One block over text, a 4x6 condition image, text, a 2x2 condition image
/// and a 4x4 target: the cached prefix keys and values, and the target's
/// output through the prefix-cache pass, against the reference (which
/// measured its own full block-causal pass and cached pass 0.0 apart).
#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn edit_block_matches_reference() {
    let doc: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("edit_block.json")).expect("the fixture"),
    )
    .expect("parses");
    let sigma = 0.704_534_f32;
    assert!((doc["sigma"].as_f64().expect("sigma") - f64::from(sigma)).abs() < 1e-6);
    let (hidden, channels) = (4096usize, 64usize);
    let layout = capture_layout();
    let grid = LatentGrid { h: 4, w: 4 };
    let encoder = read_f32("edit_block_in_encoder.f32");
    let latents = read_f32("edit_block_in_latents.f32");
    // The encoder's rows: 20 text, 6 image slots, 3 text, 1 image slot.
    // The transformer reads the text rows alone.
    let text_rows: Vec<f32> = [0..20, 26..29]
        .into_iter()
        .flat_map(|rows| encoder[rows.start * hidden..rows.end * hidden].to_vec())
        .collect();
    let prefix_tokens = 20 + 24 + 3 + 4;
    for (case, index) in [("edit_block00", 0usize), ("edit_block31", 31)] {
        let Some((ops, dit)) = open(index) else {
            return;
        };
        let feats = ops.upload_f32("encoder", &text_rows).expect("upload");
        let lat = ops.upload_f32("latents", &latents).expect("upload");
        let cut = |start: usize, cells: usize| {
            lat.view()
                .cut((start * channels * 4) as u64, (cells * channels * 4) as u64)
                .expect("a latent run")
        };
        let mut kv = dit
            .prefix_layout(
                &ops,
                &PrefixInput {
                    layout: &layout,
                    text_feats: feats.view(),
                    latents: &[cut(0, 24), cut(24, 4)],
                },
            )
            .expect("the interleaved prefix");
        assert_eq!(kv.tokens(), prefix_tokens);
        assert_eq!(kv.target_frame(), 31);
        // The cache is read back as stored: packed halves on this device,
        // so each value carries half's rounding (2^-11 relative at most).
        for (name, got) in [
            ("prefix_k", kv.keys(&ops, 0).expect("keys")),
            ("prefix_v", kv.values(&ops, 0).expect("values")),
        ] {
            assert_eq!(got.len(), prefix_tokens * hidden);
            let err = rel_l2(&got, &read_f32(&format!("{case}_{name}.f32")));
            eprintln!("{case}: {name} rel L2 {err:.3e}");
            assert!(err <= 5e-4, "{case} {name} rel L2 {err:e}");
        }
        dit.prepare_steps(&ops, &mut kv, &[sigma], grid)
            .expect("the step tables");
        let out = ops
            .alloc_f32("velocity", grid.cells() * channels)
            .expect("alloc");
        dit.velocity(&ops, &kv, cut(28, 16), grid, sigma, out.view())
            .expect("the velocity");
        let velocity = ops
            .download("velocity", out.view(), grid.cells() * channels)
            .expect("read");
        for name in ["cached_out", "target_out"] {
            let err = rel_l2(&velocity, &read_f32(&format!("{case}_{name}.f32")));
            eprintln!("{case}: target vs {name} rel L2 {err:.3e}");
            assert!(err <= 2e-3, "{case} {name} rel L2 {err:e}");
        }
    }
}
