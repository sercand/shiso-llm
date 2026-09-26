//! The encoder against the reference's own encoder output, and against itself
//! fed in pieces.
//!
//! Needs `SHISO_TEST_MODEL_ASR`, the committed audio fixtures and a GPU
//! device. Parity runs on the f32 export, where the only difference from the
//! reference is accumulation order.

use std::path::{Path, PathBuf};

use shiso_core::DeviceBuffer;

use super::{Encoder, EncoderState};
use crate::audio_ops::Ops;
use crate::audio_ops::load::upload_f32;
use crate::nemotron_asr::config::{AsrConfig, locate};
use crate::nemotron_asr::weights::AsrWeights;

#[expect(
    clippy::print_stdout,
    reason = "a skipped case says why, and the measured error is the report"
)]
fn say(line: &str) {
    println!("{line}");
}

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

/// The valid mel rows of a clip, as the reference extractor produced them.
fn mel(clip: &str) -> Vec<f32> {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join(format!("mel-{clip}-centred.json")))
            .expect("the mel fixture exists"),
    )
    .expect("json");
    let valid = usize::try_from(meta["frames_valid"].as_u64().expect("valid")).expect("fits");
    let mut rows = f32s(&fixtures().join(format!("mel-{clip}-centred.f32")));
    rows.truncate(valid * 128);
    rows
}

struct Rig {
    ops: Ops,
    config: AsrConfig,
    weights: AsrWeights,
}

fn rig() -> Option<Rig> {
    let root = PathBuf::from(std::env::var_os("SHISO_TEST_MODEL_ASR")?);
    let ops = Ops::new(crate::audio_ops::testing::device()?);
    let located = locate(&root.join("nvidia"))
        .expect("readable")
        .expect("this family");
    let store = located.store().expect("opens");
    let config = AsrConfig::read(&located, &store).expect("reads");
    let weights = AsrWeights::load(&ops, &located, &config, &store).expect("loads");
    Some(Rig {
        ops,
        config,
        weights,
    })
}

fn read_all(ops: &Ops, buf: &DeviceBuffer, floats: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; floats * 4];
    let view = buf
        .slice(0, u64::try_from(floats * 4).expect("fits"))
        .expect("in range");
    ops.backend()
        .copy_d2h(view, &mut bytes)
        .expect("reads back");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Runs the encoder over `mel` pushed in `pieces` (cycled), then finishes.
fn encode(rig: &Rig, mel: &[f32], lookahead: usize, pieces: &[usize]) -> Vec<f32> {
    let encoder = Encoder::new(&rig.ops, &rig.weights, &rig.config);
    let mut state = EncoderState::new(&rig.ops, &rig.config, lookahead).expect("a state");
    let rows = mel.len() / 128;
    let mut out = Vec::new();
    let (mut at, mut i) = (0usize, 0usize);
    while at < rows {
        let take = pieces[i % pieces.len()].min(rows - at);
        let dev = upload_f32(&rig.ops, &mel[at * 128..(at + take) * 128]).expect("uploads");
        let chunk = encoder
            .push(&mut state, dev.view(), take, false)
            .expect("pushes");
        out.extend(read_all(&rig.ops, &chunk.buffer, chunk.frames * 1024));
        at += take;
        i += 1;
    }
    let tail = encoder.finish(&mut state).expect("finishes");
    out.extend(read_all(&rig.ops, &tail.buffer, tail.frames * 1024));
    out
}

/// Worst difference relative to each reference row's largest magnitude.
fn worst_row_rel(got: &[f32], want: &[f32]) -> f32 {
    got.chunks(1024)
        .zip(want.chunks(1024))
        .map(|(g, w)| {
            let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            g.iter()
                .zip(w)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
                / scale
        })
        .fold(0.0f32, f32::max)
}

/// Measured 2026-09-25 on M1 Max against the reference in f32: worst 5.7e-6
/// (speech-en, lookahead 3), 3.4e-6 (lookahead 13), 1.4e-6 (the odd-length
/// clip). The budget is twice the worst: accumulation order and nothing else.
const ENCODER_BUDGET: f32 = 1.2e-5;

/// How far two push schedules may drift apart: accumulation order only. The
/// push's row count picks the multiply's kernel and split, and where its
/// first query falls sets which keys the attention's softmax folds together.
/// Measured 2026-09-26 on M1 Max: 5.1e-6 (lookahead 0), 5.0e-7 (lookahead 13).
const PIECES_BUDGET: f32 = 1e-5;

#[test]
#[ignore = "requires a GPU device"]
fn the_encoder_matches_the_reference_offline() {
    let Some(rig) = rig() else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    for (clip, lookahead) in [
        ("speech-en", 13usize),
        ("speech-en", 3),
        ("speech-en-odd", 13),
    ] {
        let want = f32s(&fixtures().join(format!("asr-{clip}-r{lookahead}-encoder.f32")));
        let got = encode(&rig, &mel(clip), lookahead, &[usize::MAX]);
        assert_eq!(
            got.len(),
            want.len(),
            "{clip} r{lookahead}: {} frames, want {}",
            got.len() / 1024,
            want.len() / 1024
        );
        let worst = worst_row_rel(&got, &want);
        say(&format!(
            "encoder {clip} r{lookahead}: worst row-relative error {worst:.2e}"
        ));
        assert!(worst <= ENCODER_BUDGET, "{clip} r{lookahead}: {worst}");
    }
}

#[test]
#[ignore = "requires a GPU device"]
fn pushing_in_pieces_changes_nothing() {
    let Some(rig) = rig() else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    let mel = mel("weather-es");
    for lookahead in [13usize, 0] {
        let whole = encode(&rig, &mel, lookahead, &[usize::MAX]);
        let pieces = encode(&rig, &mel, lookahead, &[1, 7, 333]);
        assert_eq!(whole.len(), pieces.len(), "r{lookahead}: frame count");
        // Not bit-for-bit: a push's row count picks the multiply's kernel and
        // how it splits k, which changes the summation order. What must not
        // change is anything past accumulation order.
        let worst = worst_row_rel(&pieces, &whole);
        say(&format!(
            "pieces r{lookahead}: worst row-relative difference {worst:.2e}"
        ));
        assert!(
            worst <= PIECES_BUDGET,
            "r{lookahead}: pushing in pieces changed the output by {worst}"
        );
    }
}

#[test]
#[ignore = "requires a GPU device"]
fn the_frame_count_follows_the_subsampling_rule() {
    let Some(rig) = rig() else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    let mel = mel("speech-en");
    let f = |l: usize| l / 2 + 1;
    for n in [1usize, 24, 25, 105, 106, 3000] {
        let got = encode(&rig, &mel[..n * 128], 13, &[usize::MAX]);
        assert_eq!(got.len() / 1024, f(f(f(n))), "{n} mel frames");
    }
}
