//! Chunk planning: silence cuts near every mark, turn cuts from speaker
//! activity, and the fallback between them.

use super::*;

const N_MELS: usize = 4;

/// `rows` loud frames (every bin 0.0) with a quiet 100 ms window (every bin
/// -5.0) starting at each of `quiet`.
fn mel(rows: usize, quiet: &[usize]) -> MelFrames {
    let mut data = vec![0.0f32; rows * N_MELS];
    for &q in quiet {
        for v in &mut data[q * N_MELS..(q + WINDOW) * N_MELS] {
            *v = -5.0;
        }
    }
    MelFrames::from_rows(data, N_MELS).expect("whole rows")
}

/// Speaker activity at 0.08 s a frame: each `(speaker, frames)` run in turn,
/// the speaker at 0.9 and everyone else at 0.0; `None` is silence.
fn activity(speakers: usize, runs: &[(Option<usize>, usize)]) -> SpeakerActivity {
    let mut probs = Vec::new();
    for &(who, frames) in runs {
        for _ in 0..frames {
            for s in 0..speakers {
                probs.push(if who == Some(s) { 0.9 } else { 0.0 });
            }
        }
    }
    SpeakerActivity::new(0.08, speakers, probs).expect("geometry")
}

const MAX: usize = 3_000;
const SEARCH: usize = 500;
const WINDOW: usize = 10;
const MEL_SECS: f64 = 0.01;

#[test]
fn a_short_clip_has_no_cut() {
    assert!(silence_cuts(&mel(2_000, &[500]), MAX, SEARCH, WINDOW).is_empty());
    // A clip of exactly one chunk needs no cut either.
    assert!(silence_cuts(&mel(3_000, &[]), MAX, SEARCH, WINDOW).is_empty());
}

#[test]
fn a_long_clip_is_cut_at_its_quietest_window_near_each_mark() {
    // Two equally quiet windows inside 3000 ± 500: the earlier wins. The
    // second mark (5800) has no quiet window near it, so every window ties
    // and the first one searched, 5300, is taken.
    let cuts = silence_cuts(&mel(7_000, &[2_800, 3_400]), MAX, SEARCH, WINDOW);
    assert_eq!(cuts, vec![2_800, 5_300]);
    // A quieter window later in the range beats an earlier, louder one.
    let mut m = mel(4_000, &[2_800]).as_slice().to_vec();
    for v in &mut m[3_400 * N_MELS..3_410 * N_MELS] {
        *v = -9.0;
    }
    let m = MelFrames::from_rows(m, N_MELS).expect("whole rows");
    assert_eq!(silence_cuts(&m, MAX, SEARCH, WINDOW), vec![3_400]);
}

#[test]
fn a_turn_change_is_a_cut() {
    // Speaker 0 to 4.0 s, silence to 5.6 s, speaker 1 to 9.6 s: the silence
    // between different speakers splits at its midpoint, 4.8 s.
    let a = activity(2, &[(Some(0), 50), (None, 20), (Some(1), 50)]);
    assert_eq!(turn_cuts(&a, &mel(960, &[]), MEL_SECS), vec![480]);
}

#[test]
fn a_long_turn_is_cut_inside_at_its_quiet_window() {
    // Speaker 0 for 10 s, then speaker 1 for 40 s: the turn change at 10 s,
    // and the second turn's 30 s mark (40 s) finds the quiet window at 42 s.
    let a = activity(2, &[(Some(0), 125), (Some(1), 500)]);
    let m = mel(5_000, &[4_200]);
    assert_eq!(turn_cuts(&a, &m, MEL_SECS), vec![1_000, 4_200]);
}

#[test]
fn a_flicker_under_half_a_second_does_not_cut() {
    // 0.4 s of speaker 1 inside speaker 0's turn is a cough, not a turn.
    let a = activity(2, &[(Some(0), 40), (Some(1), 5), (Some(0), 45)]);
    assert!(turn_cuts(&a, &mel(720, &[]), MEL_SECS).is_empty());
    // Six frames last 0.48 s: still short. Seven (0.56 s) are a turn.
    let a = activity(2, &[(Some(0), 40), (Some(1), 6), (Some(0), 44)]);
    assert!(turn_cuts(&a, &mel(720, &[]), MEL_SECS).is_empty());
    let a = activity(2, &[(Some(0), 40), (Some(1), 7), (Some(0), 43)]);
    assert_eq!(turn_cuts(&a, &mel(720, &[]), MEL_SECS), vec![320, 376]);
    // At a real turn change a third voice's flicker joins the longer
    // neighbour, so the one cut stays where speaker 0 stops.
    let a = activity(3, &[(Some(0), 40), (Some(2), 5), (Some(1), 45)]);
    assert_eq!(turn_cuts(&a, &mel(720, &[]), MEL_SECS), vec![320]);
}

#[test]
fn a_silent_or_single_speaker_activity_falls_back_to_silence_cuts() {
    let m = mel(7_000, &[2_800, 6_100]);
    let expected = silence_cuts(&m, MAX, SEARCH, WINDOW);
    assert_eq!(expected, vec![2_800, 6_100]);
    // Nobody reaches 0.5 anywhere.
    let silent = activity(2, &[(None, 875)]);
    assert_eq!(turn_cuts(&silent, &m, MEL_SECS), expected);
    // One speaker throughout, with a pause: silence between two runs of the
    // same speaker joins them.
    let single = activity(2, &[(Some(1), 300), (None, 100), (Some(1), 475)]);
    assert_eq!(turn_cuts(&single, &m, MEL_SECS), expected);
}
