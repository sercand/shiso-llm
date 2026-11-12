//! The Qwen3-VL vision tower of the text-encoder file against the
//! reference tower.
//!
//! `test-vectors/references/qwen_image21/vision*` were captured by
//! `scripts/reference/qwen_image21_reference.py vision`: the reference
//! tower in f32 on the CPU over the same BF16 `model.visual.*` tensors, on
//! one 384x256 picture that already sits on the processor's grid, so
//! neither side resamples it. The case packs the same picture, runs the
//! tower on the device and grades the merged rows and the three deepstack
//! row sets per row.
//!
//! Requires a Metal device and `SHISO_TEST_MODEL_QWEN_IMAGE21` (a directory
//! with `image/text_encoders/` and `image_assets/text_encoder/`); without
//! either it says so on stderr and returns.
#![expect(
    clippy::print_stderr,
    reason = "the measured agreement is the case's report, and a skip must say why"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer};
use shiso_vision::{VisionTower, pack_rgb};

use crate::diffusion::weights::safetensors_files;
use crate::qwen35::vision::Qwen35Tower;

/// Lowest per-row cosine against the f32 reference.
const MIN_COSINE: f64 = 0.9999;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/qwen_image21")
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

fn download(backend: &Arc<dyn Backend>, buffer: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0u8; usize::try_from(buffer.len()).expect("fits")];
    backend
        .copy_d2h(buffer.view(), &mut bytes)
        .expect("the rows read back");
    bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// `(lowest row cosine, largest absolute difference)`.
fn agreement(got: &[f32], want: &[f32], width: usize) -> (f64, f32) {
    assert_eq!(got.len(), want.len(), "row counts differ");
    let mut min_cos = f64::INFINITY;
    let mut max_abs = 0f32;
    for (g, w) in got.chunks_exact(width).zip(want.chunks_exact(width)) {
        let (mut gw, mut gg, mut ww) = (0f64, 0f64, 0f64);
        for (&a, &b) in g.iter().zip(w) {
            gw += f64::from(a) * f64::from(b);
            gg += f64::from(a) * f64::from(a);
            ww += f64::from(b) * f64::from(b);
            max_abs = max_abs.max((a - b).abs());
        }
        min_cos = min_cos.min(gw / (gg.sqrt() * ww.sqrt()));
    }
    (min_cos, max_abs)
}

#[test]
#[ignore = "requires a Metal device and the Qwen-Image checkpoint"]
fn deepstack_rows_match_reference() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21").map(PathBuf::from) else {
        eprintln!("note: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset, skipping the vision tower case");
        return;
    };
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the vision tower case");
        return;
    }
    let metal = shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let backend: Arc<dyn Backend> = metal;

    let weights = safetensors_files(&root.join("image/text_encoders")).expect("a text encoder");
    let config = root.join("image_assets/text_encoder/config.json");
    let tower = Qwen35Tower::open_checkpoint_file(&backend, &weights[0], &config)
        .unwrap_or_else(|e| panic!("the tower loads: {e}"));
    assert_eq!(tower.deepstack_taps(), 3);
    assert_eq!(tower.geometry(), (27, 1152, 4096));

    let doc: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("vision.json")).expect("the vision fixture"),
    )
    .expect("parses");
    let side =
        |key: &str| u32::try_from(doc["picture"][key].as_u64().expect("a side")).expect("fits");
    let (width, height) = (side("width"), side("height"));
    // The capture's picture: v(x, y, c) = (7x + 13y + 29c) mod 256.
    let rgb: Vec<u8> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            [0u32, 1, 2].map(|c| u8::try_from((7 * x + 13 * y + 29 * c) % 256).expect("a byte"))
        })
        .collect();
    let item = pack_rgb(&rgb, width, height, tower.preprocess()).expect("the picture packs");

    let (rows, deep) = tower
        .encode_deep(&backend, &item)
        .expect("the tower encodes");
    assert_eq!(deep.rows.len(), 3);
    let out_hidden = 4096;
    let mut cases = vec![("rows", download(&backend, &rows))];
    for (i, buffer) in deep.rows.iter().enumerate() {
        cases.push((
            ["deepstack0", "deepstack1", "deepstack2"][i],
            download(&backend, buffer),
        ));
    }
    for (name, got) in cases {
        let file = doc["files"][name].as_str().expect("a fixture file");
        let want = read_f32(&fixture_dir().join(file));
        let (min_cos, max_abs) = agreement(&got, &want, out_hidden);
        eprintln!(
            "vision {name}: {} rows, min cos {min_cos:.7}, max abs diff {max_abs:.5}",
            got.len() / out_hidden
        );
        assert!(
            min_cos >= MIN_COSINE,
            "{name}: min row cosine {min_cos} is under {MIN_COSINE}"
        );
    }
}
