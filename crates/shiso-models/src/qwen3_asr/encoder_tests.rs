//! The tower against the reference's own windowed encoder output over its
//! own mel, for every container the tower ships in.
//!
//! Needs `SHISO_TEST_MODEL_QWEN3ASR`, `SHISO_TEST_MODEL_R2T2`, the reference
//! fixtures under `test-vectors/references/audio/` and a GPU device. The
//! reference ran in f32 from the HF bf16 weights, so the HF export measures
//! the computation alone and the quantised exports add their rounding on
//! top. The reference mel is fed in, not ours, so the tower is all that is
//! compared. Run in release: the HF exports widen their small parameters on
//! the host.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use shiso_audio::MelFrames;

use super::{Tower, audio_tokens};
use crate::audio_ops::Ops;
use crate::audio_ops::load::download_f32;
use crate::qwen3_asr::config::tests::{Export, exports, say};
use crate::qwen3_asr::config::{Qwen3AsrConfig, locate};
use crate::qwen3_asr::weights::QwenAsrWeights;

/// Output width of the tower.
const OUT: usize = 2048;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/audio")
}

fn f32s(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn meta(clip: &str) -> serde_json::Value {
    let path = fixtures().join(format!("qwen3asr-{clip}-mel.json"));
    serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .expect("json")
}

/// The reference's mel of `clip`, row-major `[frames][128]`.
fn mel(clip: &str) -> MelFrames {
    let frames = usize::try_from(meta(clip)["frames"].as_u64().expect("frames")).expect("fits");
    let rows = f32s(&fixtures().join(format!("qwen3asr-{clip}-mel.f32")));
    assert_eq!(rows.len(), frames * 128, "{clip}: mel length");
    MelFrames::from_rows(rows, 128).expect("whole rows")
}

/// The reference's audio token count for `clip`.
fn n_audio(clip: &str) -> usize {
    usize::try_from(meta(clip)["n_audio_tokens"].as_u64().expect("count")).expect("fits")
}

/// The reference's windowed tower output for `clip`, `[n_audio][2048]`.
fn reference(clip: &str) -> Vec<f32> {
    f32s(&fixtures().join(format!("qwen3asr-{clip}-encoder.f32")))
}

/// Worst difference relative to each reference row's largest magnitude.
fn worst(got: &[f32], want: &[f32], width: usize) -> f32 {
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

fn ops() -> Option<Ops> {
    crate::audio_ops::testing::device().map(Ops::new)
}

/// The export loaded, or `None` having said why.
fn load(ops: &Ops, export: &Export) -> Option<(Qwen3AsrConfig, QwenAsrWeights)> {
    if !export.path.exists() {
        say(&format!(
            "SKIP {}: {} is absent",
            export.name,
            export.path.display()
        ));
        return None;
    }
    let located = locate(&export.path, None)
        .expect("readable")
        .expect("this family");
    let stores = located.stores().expect("opens");
    let config = Qwen3AsrConfig::read(&located, &stores).expect("reads");
    let weights = QwenAsrWeights::load(ops, &located, &config, &stores).expect("loads");
    Some((config, weights))
}

/// The tower's rows for `clip`, read back, and the wall time of the encode
/// including the read-back.
fn run(ops: &Ops, tower: &Tower<'_>, clip: &str) -> (Vec<f32>, usize, f64) {
    let frames = mel(clip);
    let start = Instant::now();
    let encoded = tower.encode(&frames).expect("encodes");
    let rows = download_f32(ops, encoded.buffer.view(), encoded.tokens * OUT).expect("reads");
    (rows, encoded.tokens, start.elapsed().as_secs_f64())
}

/// Row-relative budgets, measured 2026-09-27 on M1 Max against the reference
/// in f32 over the four clips (worst clip in brackets):
///
/// - HF bf16: 4.0e-5 (two-speakers);
/// - MLX: 4.0e-5 (two-speakers) - its tower is the HF bf16 tower;
/// - GGUF bf16 pair: 3.2e-5 (two-speakers);
/// - GGUF `Q8_0` pair: 8.9e-2 (weather-es);
/// - the fine-tune's GGUF `Q8_0` pair: 7.8e-2 (weather-es).
///
/// Each budget is about twice the measured worst. The `Q8_0` pairs carry the
/// rounding of 8-bit linears through 24 layers, in line with the diarizer's
/// `Q8_0` exports.
fn budget(export: &str) -> Option<f32> {
    match export {
        "qwen hf" | "qwen mlx" => Some(8e-5),
        "qwen gguf bf16" => Some(7e-5),
        "qwen gguf q8_0" => Some(0.18),
        "r2t2 gguf q8_0" => Some(0.16),
        // The fine-tune retrained the decoder only; its tower is the base
        // model's, so the HF copy would measure the HF path twice. Its GGUF
        // pair stays in: that mmproj's conv_out is Q8_0.
        _ => None,
    }
}

const CLIPS: [&str; 4] = ["speech-en", "speech-en-odd", "weather-es", "two-speakers"];

#[test]
fn audio_tokens_follow_the_reference_lengths() {
    // The lengths the reference encoder's output had for these frame counts.
    for (frames, tokens) in [
        (3000, 390),
        (1234, 161),
        (1104, 144),
        (4104, 534),
        (100, 13),
        (99, 13),
        (50, 7),
        (8, 1),
        (1, 1),
        (0, 0),
    ] {
        assert_eq!(audio_tokens(frames), tokens, "{frames} frames");
    }
}

#[test]
#[ignore = "requires a GPU device and the real checkpoints"]
fn the_tower_matches_the_reference() {
    let Some(ops) = ops() else {
        return;
    };
    let mut measured = 0;
    for export in exports() {
        let Some(allowed) = budget(export.name) else {
            continue;
        };
        let Some((config, weights)) = load(&ops, &export) else {
            continue;
        };
        let tower = Tower::new(&ops, &weights.tower, &config.audio);
        // One untimed pass, so the timings below leave out pipeline compiles.
        drop(run(&ops, &tower, "speech-en-odd"));
        let mut line = format!("{}:", export.name);
        let mut export_worst = 0.0f32;
        for clip in CLIPS {
            let (got, tokens, secs) = run(&ops, &tower, clip);
            assert_eq!(tokens, n_audio(clip), "{}: {clip} tokens", export.name);
            let e = worst(&got, &reference(clip), OUT);
            write!(line, " {clip} {e:.2e} ({secs:.3}s)").expect("a String takes it");
            export_worst = export_worst.max(e);
        }
        say(&line);
        assert!(
            export_worst <= allowed,
            "{}: worst {export_worst:.3e} over the budget {allowed:.1e}",
            export.name
        );
        measured += 1;
    }
    say(&format!("measured {measured} exports"));
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_QWEN3ASR"]
fn a_tail_chunk_is_padded() {
    let Some(ops) = ops() else {
        return;
    };
    let Some(export) = exports().into_iter().find(|e| e.name == "qwen hf") else {
        return say("SKIP: SHISO_TEST_MODEL_QWEN3ASR is unset");
    };
    let Some((config, weights)) = load(&ops, &export) else {
        return;
    };
    let tower = Tower::new(&ops, &weights.tower, &config.audio);
    // 1104 frames: eleven whole chunks and a tail of four, which yields one
    // token. Its conv stem must run over the tail zero-filled to 100 frames
    // as the reference fills it; an unpadded tail moves that token.
    let clip = "speech-en-odd";
    let (got, tokens, _) = run(&ops, &tower, clip);
    assert_eq!(tokens, n_audio(clip));
    assert_eq!(tokens, 11 * 13 + 1);
    let want = reference(clip);
    let tail = worst(&got[(tokens - 1) * OUT..], &want[(tokens - 1) * OUT..], OUT);
    let all = worst(&got, &want, OUT);
    say(&format!(
        "{clip}: tail token {tail:.2e}, every token {all:.2e}"
    ));
    let allowed = budget(export.name).expect("the HF export has a budget");
    assert!(tail <= allowed, "tail token {tail}");
    assert!(all <= allowed, "every token {all}");
}

/// `mel`'s frames `[from, to)`.
fn frames_of(mel: &MelFrames, from: usize, to: usize) -> MelFrames {
    MelFrames::from_rows(mel.as_slice()[from * 128..to * 128].to_vec(), 128).expect("rows")
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_QWEN3ASR"]
fn clips_sharing_a_pass_are_each_encoded_as_alone() {
    let Some(ops) = ops() else {
        return;
    };
    let mut ran = 0;
    for export in exports()
        .into_iter()
        .filter(|e| ["qwen hf", "qwen gguf q8_0", "qwen mlx"].contains(&e.name))
    {
        let Some((config, weights)) = load(&ops, &export) else {
            continue;
        };
        let tower = Tower::new(&ops, &weights.tower, &config.audio);
        let long = mel("two-speakers");
        // The four clips, and pieces that must still run alone: 1.8 s is 23
        // tokens (the 32-row tile's own sums), 0.7 s is one short chunk; and
        // a 25-chunk piece whose last stem pass has one chunk.
        let mels: Vec<MelFrames> = CLIPS
            .iter()
            .map(|c| mel(c))
            .chain([
                frames_of(&long, 0, 180),
                frames_of(&long, 200, 270),
                frames_of(&long, 300, 2_750),
            ])
            .collect();
        let refs: Vec<&MelFrames> = mels.iter().collect();
        let mut ws = crate::audio_ops::Workspace::new();
        let many = tower.encode_many(&refs, &mut ws).expect("encodes");
        assert_eq!(many.len(), mels.len());
        for (i, (mel, got)) in mels.iter().zip(&many).enumerate() {
            let alone = tower.encode(mel).expect("encodes");
            assert_eq!(got.tokens, alone.tokens, "{} clip {i}", export.name);
            let n = got.tokens * OUT;
            let (a, b) = (
                download_f32(&ops, got.buffer.view(), n).expect("reads"),
                download_f32(&ops, alone.buffer.view(), n).expect("reads"),
            );
            let same = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(
                same,
                "{} clip {i}: a shared pass moved a value",
                export.name
            );
        }
        say(&format!(
            "{}: {} clips, each as alone",
            export.name,
            mels.len()
        ));
        ran += 1;
    }
    assert!(ran > 0, "no export ran");
}
