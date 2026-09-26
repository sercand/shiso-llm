//! The speaker cache keeps what the reference keeps, and the offline pass
//! answers what the reference answers.

use super::{SpeakerCache, run_offline, segments};
use crate::nemotron_diar::config::OFFLINE;
use crate::nemotron_diar::encoder::Encoder;
use crate::nemotron_diar::encoder::tests::{f32s, fixtures, loaded, mel};

#[expect(clippy::print_stderr, reason = "the measured error is the report")]
fn say(line: &str) {
    eprintln!("{line}");
}

/// 300 frames, eight speakers: speaker 0 talks over the first half, speaker 1
/// over the second, nobody else at all.
fn two_halves() -> Vec<f32> {
    let mut p = vec![0.05f32; 300 * 8];
    for f in 0..300 {
        p[f * 8 + usize::from(f >= 150)] = 0.9;
    }
    p
}

#[test]
fn compression_keeps_each_speakers_best_frames_grouped_and_time_ordered() {
    let probs = two_halves();
    let cache = SpeakerCache::new(OFFLINE, 8, 4);
    let kept = cache.select(&probs, 300);
    assert_eq!(kept.len(), OFFLINE.cache);
    // Speaker-major: the frames of speaker 0, its silence frame, then speaker
    // 1's, its silence frame, then the silence frames of the six others, then
    // silence to fill.
    let silence = 300;
    let first_silence = kept
        .iter()
        .position(|f| *f == silence)
        .expect("a silence frame");
    let speaker0 = &kept[..first_silence];
    assert!(!speaker0.is_empty());
    assert!(speaker0.windows(2).all(|w| w[0] < w[1]), "time-ordered");
    assert!(speaker0.iter().all(|f| *f < 150), "only speaker 0's frames");
    let rest = &kept[first_silence + 1..];
    let second_silence = rest
        .iter()
        .position(|f| *f == silence)
        .expect("a second silence");
    let speaker1 = &rest[..second_silence];
    assert!(speaker1.windows(2).all(|w| w[0] < w[1]), "time-ordered");
    assert!(
        speaker1.iter().all(|f| (150..300).contains(f)),
        "only speaker 1's frames"
    );
    assert!(
        rest[second_silence..].iter().all(|f| *f == silence),
        "the others keep only silence"
    );
}

#[test]
fn a_frame_below_half_is_never_kept_for_that_speaker() {
    let mut probs = two_halves();
    // Speaker 0 is at 0.5 exactly on its even frames: not above half.
    for f in (0..150).step_by(2) {
        probs[f * 8] = 0.5;
    }
    let cache = SpeakerCache::new(OFFLINE, 8, 4);
    let kept = cache.select(&probs, 300);
    let first_silence = kept
        .iter()
        .position(|f| *f == 300)
        .expect("a silence frame");
    assert!(kept[..first_silence].iter().all(|f| f % 2 == 1), "{kept:?}");
}

#[test]
fn segments_are_runs_above_half_ordered_by_start() {
    // Two speakers, 6 frames.
    let probs = [
        0.9, 0.1, //
        0.9, 0.1, //
        0.2, 0.8, //
        0.2, 0.8, //
        0.6, 0.7, //
        0.1, 0.1,
    ];
    assert_eq!(
        segments(&probs, 2, 0.5),
        vec![(0, 0, 2), (1, 2, 5), (0, 4, 5)]
    );
}

/// Measured 2026-09-25 on M1 Max against the reference in f32 (absolute
/// probability, compression included): speech-en 2.7e-6, two-speakers 3.1e-6,
/// speech-en-odd 8.3e-7, long-dialogue (three steps) 8.0e-6. The budget is
/// about twice the worst.
const OFFLINE_BUDGET: f32 = 1.6e-5;

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_DIAR"]
fn the_offline_pass_matches_the_reference() {
    let Some((ops, config, weights)) = loaded("nvidia") else {
        return;
    };
    let encoder = Encoder::new(&ops, &config, &weights);
    for clip in [
        "speech-en",
        "two-speakers",
        "speech-en-odd",
        "long-dialogue",
    ] {
        let frames = mel(clip);
        let probs = run_offline(&encoder, &weights.silence, &frames).expect("diarizes");
        let want = f32s(&fixtures().join(format!("diar-{clip}-probs.f32")));
        assert_eq!(
            probs.len(),
            frames.valid() * 8,
            "{clip}: one row per mel frame"
        );
        assert_eq!(probs.len(), want.len(), "{clip}");
        let worst = probs
            .iter()
            .zip(&want)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(fixtures().join(format!("diar-{clip}.json"))).expect("meta"),
        )
        .expect("json");
        let want_segments: Vec<(usize, usize, usize)> = meta["segments_at_0_5"]
            .as_array()
            .expect("segments")
            .iter()
            .map(|s| {
                let at = |i: usize| usize::try_from(s[i].as_u64().expect("count")).expect("fits");
                (at(0), at(1), at(2))
            })
            .collect();
        let got_segments = segments(&probs, 8, 0.5);
        say(&format!(
            "{clip}: offline probs {worst:.2e}, {} segments",
            got_segments.len()
        ));
        assert!(worst <= OFFLINE_BUDGET, "{clip}: probs {worst}");
        assert_eq!(got_segments, want_segments, "{clip}: segments");
    }
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_DIAR"]
fn no_rows_is_empty_and_a_few_rows_still_diarize() {
    let Some((ops, config, weights)) = loaded("nvidia") else {
        return;
    };
    let encoder = Encoder::new(&ops, &config, &weights);
    let none = shiso_audio::MelFrames::from_rows(Vec::new(), 128).expect("no rows");
    assert!(matches!(
        run_offline(&encoder, &weights.silence, &none),
        Err(shiso_audio::AudioError::Empty)
    ));
    let few = shiso_audio::MelFrames::from_rows(vec![-5.0; 5 * 128], 128).expect("rows");
    let probs = run_offline(&encoder, &weights.silence, &few).expect("diarizes");
    assert_eq!(probs.len(), 5 * 8);
}
