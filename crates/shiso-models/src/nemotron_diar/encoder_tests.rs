//! The stacked projection, the encoder and the head, against the reference's
//! own intermediates over its own mel.
//!
//! Needs `SHISO_TEST_MODEL_DIAR`, the committed diarizer fixtures and a GPU
//! device. The reference framework's f32 export is the one compared: its
//! values are the reference's weights exactly, so what is measured is the
//! computation alone.

use std::path::{Path, PathBuf};

use shiso_audio::MelFrames;

use super::Encoder;
use crate::audio_ops::Ops;
use crate::audio_ops::load::download_f32;
use crate::nemotron_diar::config::{DiarConfig, locate};
use crate::nemotron_diar::weights::DiarWeights;

#[expect(clippy::print_stderr, reason = "the measured error is the report")]
fn say(line: &str) {
    eprintln!("{line}");
}

pub(crate) fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/audio")
}

pub(crate) fn f32s(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The reference extractor's valid mel rows of `clip`.
pub(crate) fn mel(clip: &str) -> MelFrames {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join(format!("mel-{clip}-centred.json")))
            .expect("the mel fixture exists"),
    )
    .expect("json");
    let valid = usize::try_from(meta["frames_valid"].as_u64().expect("valid")).expect("fits");
    let mut rows = f32s(&fixtures().join(format!("mel-{clip}-centred.f32")));
    rows.truncate(valid * 128);
    MelFrames::from_rows(rows, 128).expect("whole rows")
}

/// Worst difference relative to each reference row's largest magnitude.
pub(crate) fn worst(got: &[f32], want: &[f32], width: usize) -> f32 {
    assert_eq!(got.len(), want.len(), "lengths");
    got.chunks(width)
        .zip(want.chunks(width))
        .map(|(g, w)| {
            let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            g.iter()
                .zip(w)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
                / scale
        })
        .fold(0.0f32, f32::max)
}

/// The loaded export at `SHISO_TEST_MODEL_DIAR/<sub>`, or `None` having said why.
pub(crate) fn loaded(sub: &str) -> Option<(Ops, DiarConfig, DiarWeights)> {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_DIAR").map(PathBuf::from) else {
        say("SKIP: SHISO_TEST_MODEL_DIAR is unset");
        return None;
    };
    let path = root.join(sub);
    if !path.exists() {
        say(&format!("SKIP: {} is absent", path.display()));
        return None;
    }
    let ops = Ops::new(crate::audio_ops::testing::device()?);
    let located = locate(&path).expect("reads").expect("this family");
    let store = located.store().expect("opens");
    let config = DiarConfig::read(&located, &store).expect("reads");
    let weights = DiarWeights::load(&ops, &located, &config, &store).expect("loads");
    Some((ops, config, weights))
}

/// Measured 2026-09-25 on M1 Max against the reference in f32: embeddings
/// below 1e-7 (row-relative), encoder 6.2e-6, first-step probabilities 2.4e-6
/// (absolute). Budgets are about twice the measured worst.
const EMBED_BUDGET: f32 = 1e-6;
const ENCODER_BUDGET: f32 = 1.5e-5;
const PROBS_BUDGET: f32 = 5e-6;

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_DIAR"]
fn the_first_step_matches_the_reference() {
    let Some((ops, config, weights)) = loaded("nvidia") else {
        return;
    };
    let encoder = Encoder::new(&ops, &config, &weights);
    for clip in ["speech-en", "two-speakers"] {
        let embeds = encoder.embed(&mel(clip)).expect("projects");
        let want = f32s(&fixtures().join(format!("diar-{clip}-embeds.f32")));
        let e = worst(&embeds, &want, 512);
        let step0 = f32s(&fixtures().join(format!("diar-{clip}-step0.f32")));
        let n = step0.len() / 512;
        let hidden = encoder.encode(&embeds[..n * 512]).expect("encodes");
        let h = worst(
            &download_f32(&ops, hidden.view(), n * 512).expect("reads"),
            &step0,
            512,
        );
        let logits = encoder.head(&hidden, n).expect("scores");
        let emitted = 340.min(n) * 8;
        let got: Vec<f32> = logits[..emitted * 8]
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        let probs = f32s(&fixtures().join(format!("diar-{clip}-probs.f32")));
        let p = got
            .iter()
            .zip(&probs[..emitted * 8])
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        say(&format!(
            "{clip}: embeds {e:.2e}, encoder {h:.2e}, first-step probs {p:.2e}"
        ));
        assert!(e <= EMBED_BUDGET, "{clip}: embeds {e}");
        assert!(h <= ENCODER_BUDGET, "{clip}: encoder {h}");
        assert!(p <= PROBS_BUDGET, "{clip}: probs {p}");
    }
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_DIAR"]
fn a_window_of_embeddings_is_the_same_slice_of_the_whole() {
    let Some((ops, config, weights)) = loaded("nvidia") else {
        return;
    };
    let encoder = Encoder::new(&ops, &config, &weights);
    // speech-en-odd's last group is partial, so the window meets the padding.
    let frames = mel("speech-en-odd");
    let whole = encoder.embed(&frames).expect("projects");
    let total = whole.len() / 512;
    for (first, count) in [(0, 3), (5, 40), (total - 4, 4)] {
        let window = encoder
            .embed_window(&frames, first, count)
            .expect("projects");
        // Not bit-for-bit: the row count picks the multiply's kernel and how
        // it splits k, which changes the summation order and nothing else.
        let gap = worst(&window, &whole[first * 512..(first + count) * 512], 512);
        assert!(gap <= 5e-6, "{first}+{count}: {gap}");
    }
}
