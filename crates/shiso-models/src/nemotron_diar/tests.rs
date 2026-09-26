//! Every export, through the registry and the `Diarizer` contract, against
//! the reference's offline probabilities and its thresholded activity.
//!
//! Needs `SHISO_TEST_MODEL_DIAR`, the committed diarizer fixtures and a GPU
//! device. The exports hold the same weights at different precisions, so each
//! gets its own measured budget. The exact ones must reproduce the reference's
//! segments; the quantized ones must hear the same speakers and agree on who
//! is active in all but a measured share of 10 ms frames, because a quantized
//! weight moves a probability near 0.5 across it and merges or splits a pause.

use std::path::PathBuf;

use super::cache::segments;
use super::encoder::tests::{f32s, fixtures, mel};

#[expect(clippy::print_stderr, reason = "the measured error is the report")]
fn say(line: &str) {
    eprintln!("{line}");
}

/// Per export: its path under the model root, the probability budget, and
/// the share of frames whose thresholded speaker set may differ from the
/// reference's (`None`: the segments must be identical).
///
/// Measured 2026-09-25 on M1 Max over `two-speakers`, `speech-en` and the
/// three-step `long-dialogue`: the f32 export and both bf16 exports 8.0e-6
/// with identical segments (the f32 export's values are bf16 values, so the
/// three read the same weights); the audio-framework `q8_0` GGUF 8.6e-2 and
/// the `sortformer` `q8_0` GGUF 4.2e-2, with the same speakers and at most
/// 0.14% and 0.056% of frames disagreeing (the first merges one short pause
/// on `long-dialogue`). Budgets are about twice the measured worst.
const EXPORTS: [(&str, f32, Option<f64>); 5] = [
    ("nvidia", 1.6e-5, None),
    ("mlx", 1.6e-5, None),
    ("gguf/nemotron-3-diarization-bf16.gguf", 1.6e-5, None),
    ("gguf/nemotron-3-diarization-q8_0.gguf", 0.18, Some(0.003)),
    (
        "nvidia/Nemotron-3-Diarization.q8_0.gguf",
        0.09,
        Some(0.0012),
    ),
];

type Segment = (usize, usize, usize);

fn reference_segments(clip: &str) -> Vec<Segment> {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join(format!("diar-{clip}.json"))).expect("meta"),
    )
    .expect("json");
    meta["segments_at_0_5"]
        .as_array()
        .expect("segments")
        .iter()
        .map(|s| {
            let at = |i: usize| usize::try_from(s[i].as_u64().expect("count")).expect("fits");
            (at(0), at(1), at(2))
        })
        .collect()
}

/// The share of frames whose set of speakers above half differs between two
/// probability blocks `[frames][8]`.
fn frame_disagreement(got: &[f32], want: &[f32]) -> f64 {
    let frames = want.len() / 8;
    let differ = got
        .chunks(8)
        .zip(want.chunks(8))
        .filter(|(g, w)| g.iter().zip(*w).any(|(a, b)| (*a > 0.5) != (*b > 0.5)))
        .count();
    crate::audio_ops::f64_of(differ) / crate::audio_ops::f64_of(frames.max(1))
}

fn speakers_of(segments: &[Segment]) -> Vec<usize> {
    let mut out: Vec<usize> = segments.iter().map(|s| s.0).collect();
    out.sort_unstable();
    out.dedup();
    out
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_DIAR"]
fn every_export_diarizes_like_the_reference() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_DIAR").map(PathBuf::from) else {
        return say("SKIP: SHISO_TEST_MODEL_DIAR is unset");
    };
    let Some(backend) = crate::audio_ops::testing::device() else {
        return;
    };
    for (sub, budget, share) in EXPORTS {
        let path = root.join(sub);
        if !path.exists() {
            say(&format!("SKIP {sub}: absent"));
            continue;
        }
        let diarizer = crate::audio_registry::open_diarizer(&path, &backend)
            .unwrap_or_else(|e| panic!("{sub}: {e}"));
        assert_eq!(diarizer.speakers(), 8);
        for clip in ["two-speakers", "speech-en", "long-dialogue"] {
            let frames = mel(clip);
            let activity = diarizer.diarize(&frames).expect("diarizes");
            assert_eq!(activity.frames(), frames.valid(), "{sub} {clip}");
            let probs: Vec<f32> = (0..activity.frames())
                .flat_map(|f| (0..8).map(move |s| (f, s)))
                .map(|(f, s)| activity.prob(f, s))
                .collect();
            let want = f32s(&fixtures().join(format!("diar-{clip}-probs.f32")));
            let worst = probs
                .iter()
                .zip(&want)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            let got_segments = segments(&probs, 8, 0.5);
            let want_segments = reference_segments(clip);
            let differ = frame_disagreement(&probs, &want);
            say(&format!(
                "{sub} {clip}: probs {worst:.2e}, {} segments, frames disagreeing {:.3}%",
                got_segments.len(),
                differ * 100.0
            ));
            assert!(worst <= budget, "{sub} {clip}: probs {worst}");
            match share {
                None => assert_eq!(got_segments, want_segments, "{sub} {clip}: segments"),
                Some(share) => {
                    assert_eq!(
                        speakers_of(&got_segments),
                        speakers_of(&want_segments),
                        "{sub} {clip}: speakers"
                    );
                    assert!(differ <= share, "{sub} {clip}: {differ} of frames differ");
                }
            }
        }
    }
}
