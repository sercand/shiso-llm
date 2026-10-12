//! The LFM2 tower: its index mappings on the host, and the shipped mmproj
//! against the independent reference on the device.
//!
//! The device cases are `#[ignore]`d and read the mmproj from
//! `SHISO_TEST_MMPROJ_D1` and the reference capture from
//! `test-vectors/vision/references/lfm2/`; each says on stderr why it skipped,
//! because a skipped case and a passing one print the same summary line.

#![expect(clippy::print_stderr, reason = "skips and measurements print a notice")]
#![expect(
    clippy::many_single_char_names,
    reason = "the index arithmetic uses the reference's own axis names"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "fixture sizes are a few million values at most"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer};
use shiso_vision::{ImageLimits, VisionItem, VisionTower, preprocess_image};

use super::Lfm2Tower;

/// The reference's `pixel_unshuffle`, step by step: four reshapes and
/// permutes over a `[height][width][channels]` block, each spelled as the
/// index arithmetic a contiguous reshape and a `permute(0, 2, 1, 3)` are.
fn reference_unshuffle(x: &[u32], height: usize, width: usize, c: usize, f: usize) -> Vec<u32> {
    // reshape (H, W, C) -> (H, W/f, C*f): a no-op on memory.
    let a = x.to_vec();
    // permute -> (W/f, H, C*f).
    let (d0, d1, d2) = (height, width / f, c * f);
    let mut b = vec![0u32; a.len()];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                b[(j * d0 + i) * d2 + k] = a[(i * d1 + j) * d2 + k];
            }
        }
    }
    // reshape -> (W/f, H/f, C*f*f): a no-op on memory. permute -> (H/f, W/f, C*f*f).
    let (e0, e1, e2) = (width / f, height / f, c * f * f);
    let mut out = vec![0u32; b.len()];
    for i in 0..e0 {
        for j in 0..e1 {
            for k in 0..e2 {
                out[(j * e0 + i) * e2 + k] = b[(i * e1 + j) * e2 + k];
            }
        }
    }
    out
}

/// The vision set's spatial merge, transcribed from
/// `kernel_vit_spatial_merge`'s own index arithmetic for one frame.
fn merge_kernel(x: &[u32], height: usize, width: usize, c: usize, m: usize) -> Vec<u32> {
    let (out_h, out_w) = (height / m, width / m);
    let row = m * m * c;
    let mut out = vec![0u32; out_h * out_w * row];
    for o in 0..out_h * out_w {
        let (oh, ow) = (o / out_w, o % out_w);
        for i in 0..row {
            let (slot, d) = (i / c, i % c);
            let (ph, pw) = (oh * m + slot / m, ow * m + slot % m);
            out[o * row + i] = x[(ph * width + pw) * c + d];
        }
    }
    out
}

#[test]
fn the_merge_kernel_is_the_reference_unshuffle() {
    // Every element is its own index, so equal outputs are equal mappings.
    for (h, w, c) in [(2, 2, 1), (4, 6, 3), (8, 32, 5), (32, 8, 2), (6, 10, 4)] {
        let x: Vec<u32> = (0..u32::try_from(h * w * c).expect("small")).collect();
        assert_eq!(
            merge_kernel(&x, h, w, c, 2),
            reference_unshuffle(&x, h, w, c, 2),
            "{h}x{w}x{c}"
        );
    }
}

#[test]
fn a_merged_row_is_two_by_two_patches_row_major() {
    // The closed form both of the above implement: block position
    // `dy * 2 + dx` holds patch `(2i + dy, 2j + dx)`.
    let (h, w, c) = (4usize, 6usize, 1usize);
    let x: Vec<u32> = (0..u32::try_from(h * w).expect("small")).collect();
    let out = reference_unshuffle(&x, h, w, c, 2);
    let (i, j) = (1usize, 2usize);
    let row = &out[(i * (w / 2) + j) * 4..][..4];
    let at = |y: usize, xx: usize| u32::try_from(y * w + xx).expect("small");
    assert_eq!(
        row,
        [at(2, 4), at(2, 5), at(3, 4), at(3, 5)].as_slice(),
        "row (1, 2)"
    );
}

// -- the shipped mmproj against the reference --------------------------------

/// Where the reference capture lives, relative to the workspace root.
fn reference_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/vision/references/lfm2")
}

fn metal() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the device case");
        return None;
    }
    Some(shiso_metal::MetalBackend::new().expect("the Metal library compiles"))
}

/// The tower, or the reason the case skipped.
fn tower(backend: &Arc<dyn Backend>) -> Option<Lfm2Tower> {
    let Ok(path) = std::env::var("SHISO_TEST_MMPROJ_D1") else {
        eprintln!("note: SHISO_TEST_MMPROJ_D1 is not set, skipping");
        return None;
    };
    if !reference_dir().join("manifest.json").is_file() {
        eprintln!(
            "note: no reference capture at {}, skipping",
            reference_dir().display()
        );
        return None;
    }
    Some(Lfm2Tower::open(backend, Path::new(&path)).expect("the mmproj opens"))
}

/// Little-endian f32 values from a capture file.
fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

/// A device buffer's f32 values.
fn download(backend: &Arc<dyn Backend>, buffer: &DeviceBuffer, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 4];
    backend
        .copy_d2h(
            buffer.slice(0, (count * 4) as u64).expect("view"),
            &mut bytes,
        )
        .expect("read back");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

/// Cosine similarity and the largest absolute error, over every value.
fn agreement(got: &[f32], want: &[f32]) -> (f64, f32) {
    assert_eq!(got.len(), want.len(), "length");
    let (mut dot, mut gg, mut ww, mut worst) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for (&g, &w) in got.iter().zip(want) {
        dot += f64::from(g) * f64::from(w);
        gg += f64::from(g) * f64::from(g);
        ww += f64::from(w) * f64::from(w);
        worst = worst.max((g - w).abs());
    }
    (dot / (gg.sqrt() * ww.sqrt()), worst)
}

/// Per-row cosine, the worst over all rows: one soft token gone wrong is a
/// failure even when the whole block's cosine hides it.
fn worst_row_cosine(got: &[f32], want: &[f32], width: usize) -> f64 {
    got.chunks(width)
        .zip(want.chunks(width))
        .map(|(g, w)| agreement(g, w).0)
        .fold(1.0f64, f64::min)
}

/// One case of the capture's manifest.
struct Case {
    name: String,
    grid_h: usize,
    grid_w: usize,
    tokens: usize,
}

fn cases() -> Vec<Case> {
    let text = std::fs::read_to_string(reference_dir().join("manifest.json")).expect("manifest");
    let manifest: serde_json::Value = serde_json::from_str(&text).expect("manifest json");
    let field = |case: &serde_json::Value, key: &str| {
        case[key]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or_else(|| panic!("manifest case lacks {key}"))
    };
    manifest["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| Case {
            name: case["name"].as_str().expect("name").to_owned(),
            grid_h: field(case, "grid_h"),
            grid_w: field(case, "grid_w"),
            tokens: field(case, "tokens"),
        })
        .collect()
}

/// The tower over the reference's own pixel rows: the encoder alone, with
/// the preprocessor out of the comparison.
///
/// The reference ran the checkpoint's BF16 weights in f32. Pointed at the
/// BF16 mmproj this tower agrees with it to cosine 1.000000 and a largest
/// error of 1e-4 on every case, so the graph is the reference's. Pointed at
/// the shipped `Q8_0` mmproj it measured cosine 0.99978-0.99986, worst row
/// 0.9974, largest error 0.025 (M1 Max): that is what the quantization costs,
/// and the budget below sits under it rather than at a round number chosen
/// in advance. The measured values are printed on every run.
#[test]
#[ignore = "requires a Metal device and the d1 mmproj"]
fn the_tower_matches_the_reference_features() {
    let Some(backend) = metal() else { return };
    let Some(tower) = tower(&backend) else { return };
    let width = tower.out_hidden() as usize;
    for case in cases() {
        let dir = reference_dir();
        let pixels = read_f32(&dir.join(format!("{}-pixels.f32", case.name)));
        let item = VisionItem::new(pixels.into_boxed_slice(), 768, 1, case.grid_h, case.grid_w)
            .expect("the reference's rows form an item");
        assert_eq!(item.pad_count(2), case.tokens, "{}: pad count", case.name);
        let stages = tower.encode_stages(&backend, &item).expect("encode");
        let got = download(&backend, &stages.projected, case.tokens * width);
        let want = read_f32(&dir.join(format!("{}-features.f32", case.name)));
        let (cosine, worst) = agreement(&got, &want);
        let row = worst_row_cosine(&got, &want, width);
        eprintln!(
            "note: {}: {}x{} patches, {} tokens: cosine {cosine:.6}, worst row cosine {row:.6}, \
             max abs error {worst:.4}",
            case.name, case.grid_h, case.grid_w, case.tokens
        );
        // Warm, through the path the server takes, the best of three;
        // printed, not asserted.
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let encoded = tower
                .encode(&backend, std::slice::from_ref(&item))
                .expect("encode");
            assert_eq!(encoded.pad_total() as usize, case.tokens);
            best = best.min(start.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!("note: {}: encode {best:.1} ms", case.name);
        assert!(cosine > 0.999, "{}: cosine {cosine}", case.name);
        assert!(row > 0.99, "{}: worst row cosine {row}", case.name);
    }
}

/// The preprocessor over the capture's source pictures: the same grid as the
/// reference's processor, and pixel rows within what two cubic resamplers
/// disagree by.
#[test]
#[ignore = "requires a Metal device and the d1 mmproj"]
fn the_preprocessor_matches_the_reference_processor() {
    let Some(backend) = metal() else { return };
    let Some(tower) = tower(&backend) else { return };
    for case in cases() {
        let dir = reference_dir();
        let png = std::fs::read(dir.join(format!("{}.png", case.name))).expect("picture");
        let item =
            preprocess_image(&png, tower.preprocess(), ImageLimits::default()).expect("preprocess");
        assert_eq!(
            item.grid(),
            (1, case.grid_h, case.grid_w),
            "{}: grid",
            case.name
        );
        let want = read_f32(&dir.join(format!("{}-pixels.f32", case.name)));
        let (cosine, worst) = agreement(item.as_flat(), &want);
        let mean = item
            .as_flat()
            .iter()
            .zip(&want)
            .map(|(g, w)| f64::from((g - w).abs()))
            .sum::<f64>()
            / want.len() as f64;
        eprintln!(
            "note: {}: pixels cosine {cosine:.6}, mean abs error {mean:.5}, max {worst:.4}",
            case.name
        );
        // Values are in [-1, 1]: a mean error of 0.01 is about one and a
        // quarter grey levels, which is what two cubic filters with
        // different rounding disagree by on a resized picture.
        assert!(mean < 0.01, "{}: mean pixel error {mean}", case.name);
    }
}
