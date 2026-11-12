//! The INT8 `ConvRot` Qwen3-VL text encoder against the BF16 reference.
//!
//! `test-vectors/references/qwen_image21/te_*` were captured by
//! `scripts/reference/qwen_image21_reference.py text-encoder-streamed`: the
//! official BF16 weights in f32, one reference decoder layer at a time,
//! cross-checked against the pipeline's own prompt-embedding code on a
//! two-layer cut. This case runs the per-row INT8 export (the file the
//! family ships) on the device and grades its last-layer features per token;
//! the bar is a cosine, because INT8 weights are not expected to match BF16
//! to a tolerance, only to point the same way.
//!
//! The device case fails by design at the 0.995 bar: the checkpoint itself
//! misses it. The worst token is the user turn's `<|im_start|>` (min 0.98657
//! at the last layer, mean 0.9978 on `cat`; every earlier tap stays above
//! 0.9992). `scripts/reference/qwen_image21_reference.py
//! text-encoder-int8-emulation` separates the export's rounding from kernel
//! error: it runs the f32 reference layers over the INT8 weights dequantized
//! and un-rotated and gets min 0.986558 on that same token, and the device
//! against that emulation is cos 1.0000000, so the kernels add nothing and
//! the shortfall is INT8 quantization of the encoder. The bar is kept at
//! 0.995 so the miss stays visible rather than being tuned away.
//!
//! Requires a Metal device and `SHISO_TEST_MODEL_QWEN_IMAGE21` (the
//! `turbo-int8` directory, with `image/text_encoders/` and
//! `image_assets/`); without either it says so on stderr and returns.
#![expect(
    clippy::print_stderr,
    reason = "the measured agreement and timings are the case's report"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_core::Backend;

use super::prompt::QwenImagePrompt;
use crate::diffusion::ops::Ops;
use crate::diffusion::qwen3_text::{
    ImageRows, TapSet, TextEncoder, TextEncoderDesc, VisionPrefill,
};
use crate::diffusion::weights::safetensors_files;
use crate::qwen35::vision::Qwen35Tower;
use shiso_vision::{VisionTower, pack_rgb};

/// Lowest per-token cosine the INT8 features may have against BF16.
const MIN_COSINE: f64 = 0.995;

/// Lowest per-token cosine of the vision-language prefill against the
/// reference layers run over the same INT8 weights: kernels only, so the bar
/// is far above the BF16 one.
const MIN_EDIT_COSINE: f64 = 0.9995;

/// The prompts the reference captured, in the order they run here: the
/// 60-word one first, so its first prefill is the cold one.
const CASES: [&str; 2] = ["scene60", "cat"];

/// The debugging taps the reference recorded (every sixth layer before the
/// last).
const DEBUG_LAYERS: [usize; 5] = [5, 11, 17, 23, 29];

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

fn row_cosines(got: &[f32], want: &[f32], width: usize) -> Vec<f64> {
    assert_eq!(got.len(), want.len(), "row counts differ");
    got.chunks_exact(width)
        .zip(want.chunks_exact(width))
        .map(|(g, w)| {
            let (mut gw, mut gg, mut ww) = (0f64, 0f64, 0f64);
            for (&a, &b) in g.iter().zip(w) {
                gw += f64::from(a) * f64::from(b);
                gg += f64::from(a) * f64::from(a);
                ww += f64::from(b) * f64::from(b);
            }
            gw / (gg.sqrt() * ww.sqrt())
        })
        .collect()
}

#[expect(
    clippy::cast_precision_loss,
    reason = "a token count, far below f32's exact-integer range"
)]
fn min_mean(cos: &[f64]) -> (f64, f64) {
    let min = cos.iter().copied().fold(f64::INFINITY, f64::min);
    (min, cos.iter().sum::<f64>() / cos.len() as f64)
}

fn setup() -> Option<(PathBuf, Arc<dyn Backend>)> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21") else {
        eprintln!(
            "note: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset, skipping the INT8 text encoder case"
        );
        return None;
    };
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the INT8 text encoder case");
        return None;
    }
    let metal = shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let backend: Arc<dyn Backend> = metal;
    Some((PathBuf::from(root), backend))
}

fn load(root: &Path, ops: &Ops, taps: TapSet) -> TextEncoder {
    let weights = safetensors_files(&root.join("image/text_encoders")).expect("weights listed");
    TextEncoder::load(
        &TextEncoderDesc {
            weights: &weights,
            config: &root.join("image_assets/text_encoder/config.json"),
            prefix: "model.",
            taps,
        },
        ops,
    )
    .expect("the INT8 text encoder loads")
}

fn reference_ids(case: &str) -> Vec<u32> {
    let path = fixture_dir().join("te_streamed.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let doc: serde_json::Value = serde_json::from_str(&text).expect("te_streamed.json parses");
    doc["cases"][case]["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("an id")).expect("fits"))
        .collect()
}

#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn the_int8_text_encoder_points_where_the_bf16_one_does() {
    let Some((root, backend)) = setup() else {
        return;
    };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let prompt =
        QwenImagePrompt::load(&root.join("image_assets/processor")).expect("the prompt loads");
    let fixtures = fixture_dir();

    let began = Instant::now();
    let encoder = load(&root, &ops, TapSet::LastPreNorm);
    let width = encoder.context_width();
    eprintln!(
        "INT8 text encoder load: {:.2} s",
        began.elapsed().as_secs_f64()
    );

    let mut failures = Vec::new();
    for case in CASES {
        let ids = prompt
            .tokenize(&prompt_text(case))
            .unwrap_or_else(|e| panic!("{case}: {e}"));
        assert_eq!(
            ids.ids,
            reference_ids(case),
            "{case}: our ids are the reference's"
        );
        let mut runs = Vec::new();
        let mut features = Vec::new();
        for _ in 0..3 {
            let began = Instant::now();
            features = encoder.prefill(&ops, &ids.ids).expect("the prefill runs");
            runs.push(began.elapsed().as_secs_f64());
        }
        eprintln!(
            "{case}: {} tokens; encode first {:.3} s, then {:.3} s, {:.3} s",
            ids.ids.len(),
            runs[0],
            runs[1],
            runs[2]
        );
        let kept = &features[ids.drop * width..];
        let want = read_f32(&fixtures.join(format!("te_{case}_hidden.f32")));
        let cos = row_cosines(kept, &want, width);
        let (min, mean) = min_mean(&cos);
        eprintln!("{case}: last layer per-token cosine min {min:.6} mean {mean:.6}");
        eprintln!("{case}: per-token {cos:.5?}");
        if min < MIN_COSINE {
            failures.push(format!("{case}: min cosine {min:.6} < {MIN_COSINE}"));
        }
    }
    drop(encoder);
    ops.trim_pool();

    // Where a drift starts: the residual stream every sixth layer.
    let debug = load(&root, &ops, TapSet::Layers(DEBUG_LAYERS.to_vec()));
    let hidden = debug.context_width() / DEBUG_LAYERS.len();
    for case in CASES {
        let ids = prompt.tokenize(&prompt_text(case)).expect("encodes");
        let features = debug.prefill(&ops, &ids.ids).expect("the prefill runs");
        let tokens = ids.ids.len();
        for (slot, layer) in DEBUG_LAYERS.iter().enumerate() {
            let mut got = Vec::with_capacity((tokens - ids.drop) * hidden);
            for row in ids.drop..tokens {
                let start = row * debug.context_width() + slot * hidden;
                got.extend_from_slice(&features[start..start + hidden]);
            }
            let want = read_f32(&fixtures.join(format!("te_{case}_layer{layer:02}.f32")));
            let (min, mean) = min_mean(&row_cosines(&got, &want, hidden));
            eprintln!("{case}: after layer {layer:2}: cosine min {min:.6} mean {mean:.6}");
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

fn prompt_text(case: &str) -> String {
    let path = fixture_dir().join("te_streamed.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let doc: serde_json::Value = serde_json::from_str(&text).expect("te_streamed.json parses");
    doc["cases"][case]["prompt"]
        .as_str()
        .expect("a prompt")
        .to_string()
}

/// The capture's picture: `v(x, y, c) = (7x + 13y + 29c) mod 256`, RGB.
fn capture_picture(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            [0u32, 1, 2].map(|c| u8::try_from((7 * x + 13 * y + 29 * c) % 256).expect("a byte"))
        })
        .collect()
}

/// A list of small non-negative integers out of a fixture.
fn numbers(value: &serde_json::Value) -> Vec<u32> {
    value
        .as_array()
        .expect("a list")
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("a number")).expect("fits"))
        .collect()
}

/// The vision-language prefill of an edit prompt against
/// `qwen_image21_reference.py edit-text-encoder-int8`: the reference
/// decoder layers in f32 over this same INT8 file dequantized, the
/// reference tower's rows at the image slots and its deepstack sets after
/// layers 0 to 2, three-axis positions. Both sides read the same weights,
/// so what is graded is the device path: the tower, the row splice, the
/// table rotation and the injections.
#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn edit_features_match_reference() {
    edit_features_against_the_reference(false);
}

/// The same prompt with its attention on the fused grouped-query tile, which
/// a prompt this short would not reach on its own: the features meet the
/// same bar, so the tile is the per-row attention and not a cheaper one.
#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_QWEN_IMAGE21"]
fn edit_features_match_reference_on_the_fused_attention() {
    edit_features_against_the_reference(true);
}

/// Encodes the reference's picture prompt and grades the kept features
/// against the reference's, with the prefill's attention on the fused tile
/// when `fused`.
#[expect(
    clippy::too_many_lines,
    reason = "one case: tokenize, encode the picture, prefill, grade, straight-line"
)]
fn edit_features_against_the_reference(fused: bool) {
    let Some((root, backend)) = setup() else {
        return;
    };
    let fixtures = fixture_dir();
    let doc: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures.join("edit_te.json")).expect("the edit fixture"),
    )
    .expect("parses");
    let side =
        |key: &str| u32::try_from(doc["picture"][key].as_u64().expect("a side")).expect("fits");
    let (width, height) = (side("width"), side("height"));
    let slots = (height as usize / 32, width as usize / 32);

    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let ops = if fused {
        ops.fusing_short_prompts()
    } else {
        ops
    };
    let prompt =
        QwenImagePrompt::load(&root.join("image_assets/processor")).expect("the prompt loads");
    let ids = prompt
        .tokenize_edit(doc["prompt"].as_str().expect("a prompt"), &[slots])
        .expect("the edit prompt encodes");
    assert_eq!(
        ops.fuses_gqa(ids.ids.len(), 128),
        fused,
        "the prompt's attention runs on the kernel this case is about"
    );
    assert_eq!(ids.ids, numbers(&doc["ids"]), "our ids are the reference's");
    let positions = ids.positions();
    for (axis, want) in (0..3).map(|a| (a, numbers(&doc["position_ids"][a]))) {
        let got: Vec<u32> = positions.iter().map(|p| p[axis]).collect();
        assert_eq!(got, want, "axis {axis} positions are the reference's");
    }

    let weights = safetensors_files(&root.join("image/text_encoders")).expect("weights listed");
    let config = root.join("image_assets/text_encoder/config.json");
    let began = Instant::now();
    let tower =
        Qwen35Tower::open_checkpoint_file(&backend, &weights[0], &config).expect("the tower loads");
    let rgb = capture_picture(width, height);
    let item = pack_rgb(&rgb, width, height, tower.preprocess()).expect("the picture packs");
    let (rows, deep) = tower
        .encode_deep(&backend, &item)
        .expect("the tower encodes");
    eprintln!(
        "tower load + encode: {:.2} s",
        began.elapsed().as_secs_f64()
    );

    let began = Instant::now();
    let encoder = load(&root, &ops, TapSet::LastPreNorm);
    let hidden = encoder.context_width();
    eprintln!(
        "INT8 text encoder load: {:.2} s",
        began.elapsed().as_secs_f64()
    );
    let (run, _) = &ids.images[0];
    let images = [ImageRows {
        at: run.clone(),
        main: rows.view(),
        deep: deep
            .rows
            .iter()
            .map(shiso_core::DeviceBuffer::view)
            .collect(),
    }];
    let began = Instant::now();
    let features = encoder
        .prefill_vision(
            &ops,
            &VisionPrefill {
                ids: &ids.ids,
                positions: &positions,
                images: &images,
            },
        )
        .expect("the vision prefill runs");
    eprintln!(
        "vision prefill of {} tokens: {:.3} s",
        ids.ids.len(),
        began.elapsed().as_secs_f64()
    );

    let kept = &features[ids.drop * hidden..];
    let want = read_f32(&fixtures.join("edit_te_hidden.f32"));
    let cos = row_cosines(kept, &want, hidden);
    let (min, mean) = min_mean(&cos);
    eprintln!("edit: last layer per-token cosine min {min:.7} mean {mean:.7}");
    assert!(
        min >= MIN_EDIT_COSINE,
        "min cosine {min:.7} < {MIN_EDIT_COSINE}; per-token {cos:.5?}"
    );

    // Without the deepstack sets the features must differ: the injection
    // is not a no-op the bar above would also pass.
    let bare = [ImageRows {
        deep: Vec::new(),
        ..images[0].clone()
    }];
    assert!(
        encoder
            .prefill_vision(
                &ops,
                &VisionPrefill {
                    ids: &ids.ids,
                    positions: &positions,
                    images: &bare,
                },
            )
            .is_err(),
        "a picture without its deepstack sets is refused"
    );
}
