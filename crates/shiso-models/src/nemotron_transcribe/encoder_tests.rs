//! The encoder against the reference's captures, stage by stage.
//!
//! Needs `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER` naming the encoder
//! `.gguf` and the captures under
//! `test-vectors/references/nemotron_transcribe/`; each case says on stderr
//! why it skipped.

use std::path::{Path, PathBuf};

use shiso_audio::{MelFrontend, Pcm};

use super::config::PeeConfig;
use super::frontend::mel_spec;
use super::weights::PeeWeights;
use super::{Encoded, encode};
use crate::audio_ops::Ops;
use crate::audio_ops::store::TensorStore;

/// The encoder export's config and filterbank, or `None` (said) when this
/// box has none.
#[expect(clippy::print_stderr, reason = "a skipped case says why")]
pub(crate) fn reference() -> Option<(PeeConfig, Vec<f32>)> {
    let Some(path) = std::env::var_os("SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER") else {
        eprintln!("SKIP: SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER is unset");
        return None;
    };
    let store = TensorStore::open_gguf(Path::new(&path)).expect("the encoder export opens");
    let c = PeeConfig::from_gguf(store.gguf().expect("a GGUF store")).expect("its config reads");
    let fb = crate::audio_ops::store::dequant(&store.require("preprocessor.fb").expect("fb"))
        .expect("fb reads");
    Some((c, fb))
}

/// The capture directory `rel` under the reference root, or `None` (said).
#[expect(clippy::print_stderr, reason = "a skipped case says why")]
pub(crate) fn refs(rel: &str) -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/nemotron_transcribe")
        .join(rel);
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!("SKIP: no reference capture at {}", dir.display());
        None
    }
}

/// Little-endian f32s.
pub(crate) fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .expect("the capture reads")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The clip `name`, 16 kHz mono, from `~/models/clips` or
/// `SHISO_TEST_CLIPS`.
pub(crate) fn clip_pcm(name: &str) -> Pcm {
    let dir = std::env::var_os("SHISO_TEST_CLIPS").map_or_else(
        || PathBuf::from(std::env::var_os("HOME").expect("HOME")).join("models/clips"),
        PathBuf::from,
    );
    let bytes = std::fs::read(dir.join(format!("{name}.wav"))).expect("the clip reads");
    // A 16-bit mono WAV: walk the chunks to `data`.
    let mut at = 12usize;
    loop {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes([bytes[at + 4], bytes[at + 5], bytes[at + 6], bytes[at + 7]])
            as usize;
        if id == b"data" {
            let samples = bytes[at + 8..at + 8 + len]
                .chunks_exact(2)
                .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                .collect();
            return Pcm::new(samples, 16_000).expect("16 kHz");
        }
        at += 8 + len + (len & 1);
    }
}

/// The clip `name` encoded on the device, with the ops it ran on.
fn encoded(name: &str) -> Option<(PeeConfig, Encoded, Ops)> {
    let (c, fb) = reference()?;
    let backend = crate::audio_ops::testing::device()?;
    let ops = Ops::new(backend);
    let path = std::env::var_os("SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER")?;
    let store = TensorStore::open_gguf(Path::new(&path)).expect("opens");
    let w = PeeWeights::load(&ops, &store, &c).expect("the encoder loads");
    let mel = MelFrontend::new(mel_spec(&c, &fb))
        .expect("spec")
        .compute(clip_pcm(name).samples());
    let e = encode(&ops, &w, &c, &mel).expect("encodes");
    Some((c, e, ops))
}

/// Worst per-row cosine and worst absolute difference of `got` and `want`,
/// rows of `width`.
fn agreement(got: &[f32], want: &[f32], width: usize) -> (f64, f32) {
    let mut worst_cos = 1.0f64;
    let mut worst_abs = 0.0f32;
    for (g, w) in got.chunks_exact(width).zip(want.chunks_exact(width)) {
        let (mut dot, mut ng, mut nw) = (0.0f64, 0.0f64, 0.0f64);
        for (a, b) in g.iter().zip(w) {
            dot += f64::from(*a) * f64::from(*b);
            ng += f64::from(*a) * f64::from(*a);
            nw += f64::from(*b) * f64::from(*b);
            worst_abs = worst_abs.max((a - b).abs());
        }
        if ng > 0.0 && nw > 0.0 {
            worst_cos = worst_cos.min(dot / (ng.sqrt() * nw.sqrt()));
        }
    }
    (worst_cos, worst_abs)
}

#[expect(
    clippy::print_stderr,
    reason = "the measured agreement is the test's report"
)]
fn stages_match(clip: &str) {
    let Some(dir) = refs(&format!("{clip}/verbatim")) else {
        return;
    };
    let Some((c, e, ops)) = encoded(clip) else {
        return;
    };
    let speech = floats(&dir.join("asr_out.f32"));
    let probs = floats(&dir.join("diar_probs.f32"));
    let fused = floats(&dir.join("fused.f32"));
    assert_eq!(e.frames * c.asr.d_model, speech.len(), "frame count");
    let (sc, sa) = agreement(&e.speech, &speech, c.asr.d_model);
    let worst_p = e
        .activity
        .iter()
        .zip(&probs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rows = crate::audio_ops::load::download_f32(&ops, e.rows.view(), e.frames * c.fusion_dim)
        .expect("rows read");
    let (fc, fa) = agreement(&rows, &fused, c.fusion_dim);
    let low_rows = |got: &[f32], want: &[f32], width: usize| -> Vec<usize> {
        got.chunks_exact(width)
            .zip(want.chunks_exact(width))
            .enumerate()
            .filter(|(_, (g, w))| agreement(g, w, width).0 < 0.9999)
            .map(|(i, _)| i)
            .collect()
    };
    let speech_low = low_rows(&e.speech, &speech, c.asr.d_model);
    let fused_low = low_rows(&rows, &fused, c.fusion_dim);
    let flips = e
        .activity
        .iter()
        .zip(&probs)
        .filter(|(a, b)| (**a > c.threshold) != (**b > c.threshold))
        .count();
    eprintln!(
        "{clip}: speech cos {sc:.6} |d| {sa:.4} ({} rows < 0.9999: {:?}); activity |d| {worst_p:.4}, \
         {flips} of {} bins flip; fused cos {fc:.6} |d| {fa:.4} ({} rows < 0.9999: {:?})",
        speech_low.len(),
        &speech_low[..speech_low.len().min(12)],
        probs.len(),
        fused_low.len(),
        &fused_low[..fused_low.len().min(12)],
    );
    // The reference runs on the CPU in f32 with its own reduction orders: a
    // few near-silent rows and a few activities at the threshold land on the
    // other side, so the grade is how many rows agree, not the worst one.
    let frames = e.frames;
    assert!(
        speech_low.len() * 1000 <= frames,
        "{} of {frames} speech rows below 0.9999",
        speech_low.len()
    );
    assert!(flips * 1000 <= probs.len(), "{flips} speaker bins flip");
    assert!(
        fused_low.len() * 200 <= frames,
        "{} of {frames} fused rows below 0.9999",
        fused_low.len()
    );
}

#[test]
#[ignore = "needs a Metal device, the speech encoder export and the reference captures"]
fn short_en_stages_match_reference() {
    stages_match("short_en");
}

#[test]
#[ignore = "needs a Metal device, the speech encoder export and the reference captures"]
fn meeting_60s_stages_match_reference() {
    stages_match("meeting_60s");
}

#[test]
#[ignore = "needs a Metal device, the speech encoder export and the reference captures"]
fn meeting_10m_stages_match_reference() {
    stages_match("meeting_10m");
}
