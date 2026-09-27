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

/// The default offline rule, skipping silence as `skip` says.
fn rule(skip: bool) -> OfflineRule {
    OfflineRule::for_10ms_mel(skip)
}

/// Spans that tile `0..valid` with no gap or overlap, every transcribed one
/// at most `max` frames.
fn assert_tiles(spans: &[Span], valid: usize, max: usize) {
    assert_eq!(spans.first().map(|s| s.start), Some(0), "{spans:?}");
    assert_eq!(spans.last().map(|s| s.end), Some(valid), "{spans:?}");
    for pair in spans.windows(2) {
        assert_eq!(pair[0].end, pair[1].start, "{spans:?}");
    }
    for s in spans {
        assert!(s.start < s.end, "{spans:?}");
        assert!(!s.transcribe || s.end - s.start <= max, "{spans:?}");
    }
}

/// The spans `cuts` make of `0..valid`, every one transcribed.
fn spans_at(cuts: &[usize], valid: usize) -> Vec<Span> {
    let mut edges = vec![0];
    edges.extend_from_slice(cuts);
    edges.push(valid);
    edges
        .windows(2)
        .map(|w| Span {
            start: w[0],
            end: w[1],
            transcribe: true,
        })
        .collect()
}

fn span(start: usize, end: usize, transcribe: bool) -> Span {
    Span {
        start,
        end,
        transcribe,
    }
}

#[test]
fn offline_spans_cover_the_clip_exactly() {
    let m = mel(12_000, &[2_900, 6_100, 9_000]);
    let cases = [
        // Speech, a long silence, a long turn, another silence at the end.
        activity(
            2,
            &[
                (Some(0), 200),
                (None, 100),
                (Some(1), 800),
                (None, 50),
                (Some(0), 300),
            ],
        ),
        // Activity that stops short of the clip: the uncovered tail is
        // transcribed, not guessed to be silent.
        activity(1, &[(Some(0), 100), (None, 900)]),
        // Activity that runs past the clip.
        activity(2, &[(Some(1), 700), (None, 1_000)]),
    ];
    for a in &cases {
        for skip in [true, false] {
            let spans = plan_offline(&m, Some(a), MEL_SECS, &rule(skip)).expect("a plan");
            assert_tiles(&spans, 12_000, 3_500);
        }
    }
    let spans = plan_offline(&m, None, MEL_SECS, &rule(true)).expect("a plan");
    assert_tiles(&spans, 12_000, 3_500);
    // An empty clip has nothing to cover.
    let empty = MelFrames::from_rows(Vec::new(), N_MELS).expect("no rows");
    assert!(
        plan_offline(&empty, None, MEL_SECS, &rule(true))
            .expect("a plan")
            .is_empty()
    );
}

#[test]
fn a_silence_under_three_seconds_is_transcribed() {
    // 37 diarizer frames of silence last 2.96 s: under the threshold, so the
    // plan is the turn cuts, every span transcribed.
    let a = activity(2, &[(Some(0), 50), (None, 37), (Some(1), 50)]);
    let m = mel(1_096, &[]);
    let spans = plan_offline(&m, Some(&a), MEL_SECS, &rule(true)).expect("a plan");
    assert_eq!(spans, spans_at(&turn_cuts(&a, &m, MEL_SECS), 1_096));
    assert_eq!(spans.len(), 2);
    // Activity between the floor and the dominance threshold is speech for
    // the skip rule: 4 s at 0.4 is not skipped, though no speaker owns it.
    let mut probs = vec![0.9f32; 50];
    probs.extend(std::iter::repeat_n(0.4f32, 50));
    probs.extend(std::iter::repeat_n(0.9f32, 50));
    let a = SpeakerActivity::new(0.08, 1, probs).expect("geometry");
    let spans = plan_offline(&mel(1_200, &[]), Some(&a), MEL_SECS, &rule(true)).expect("a plan");
    assert_eq!(spans, vec![span(0, 1_200, true)]);
}

#[test]
fn a_long_silence_is_skipped_but_its_margins_are_kept() {
    // Speaker 0 to 4 s, silence to 8 s, speaker 1 to 12 s. The silence keeps
    // 0.2 s of margin on each side, and the turn cut at its midpoint (6 s)
    // falls inside the skipped span, so it goes.
    let a = activity(2, &[(Some(0), 50), (None, 50), (Some(1), 50)]);
    let m = mel(1_200, &[]);
    assert_eq!(turn_cuts(&a, &m, MEL_SECS), vec![600]);
    let spans = plan_offline(&m, Some(&a), MEL_SECS, &rule(true)).expect("a plan");
    assert_eq!(
        spans,
        vec![
            span(0, 420, true),
            span(420, 780, false),
            span(780, 1_200, true)
        ]
    );
    // 38 frames last 3.04 s: long enough.
    let a = activity(1, &[(Some(0), 50), (None, 38), (Some(0), 50)]);
    let spans = plan_offline(&mel(1_104, &[]), Some(&a), MEL_SECS, &rule(true)).expect("a plan");
    assert_eq!(
        spans,
        vec![
            span(0, 420, true),
            span(420, 684, false),
            span(684, 1_104, true)
        ]
    );
    // At the clip's edges there is no speech to keep a margin for.
    let a = activity(1, &[(None, 50), (Some(0), 50), (None, 50)]);
    let spans = plan_offline(&mel(1_200, &[]), Some(&a), MEL_SECS, &rule(true)).expect("a plan");
    assert_eq!(
        spans,
        vec![
            span(0, 380, false),
            span(380, 820, true),
            span(820, 1_200, false)
        ]
    );
    // A turn cut outside the silence stays: speaker 0, 1, a long pause, 1.
    let a = activity(
        2,
        &[(Some(0), 50), (Some(1), 50), (None, 50), (Some(1), 50)],
    );
    let m = mel(1_600, &[]);
    assert_eq!(turn_cuts(&a, &m, MEL_SECS), vec![400]);
    let spans = plan_offline(&m, Some(&a), MEL_SECS, &rule(true)).expect("a plan");
    assert_eq!(
        spans,
        vec![
            span(0, 400, true),
            span(400, 820, true),
            span(820, 1_180, false),
            span(1_180, 1_600, true)
        ]
    );
}

#[test]
fn a_long_turn_is_split_at_thirty_five_seconds() {
    // One speaker for 50 s, quiet at 34.9 s: the cut lands there, the last
    // frame the 35 s bound allows a quiet window to start at.
    let a = activity(1, &[(Some(0), 625)]);
    let m = mel(5_000, &[3_490]);
    let spans = plan_offline(&m, Some(&a), MEL_SECS, &rule(true)).expect("a plan");
    assert_eq!(spans, vec![span(0, 3_490, true), span(3_490, 5_000, true)]);
    // With no quiet window at all, 80 s of one voice still comes back in
    // pieces no longer than 35 s.
    let a = activity(1, &[(Some(0), 1_000)]);
    let spans = plan_offline(&mel(8_000, &[]), Some(&a), MEL_SECS, &rule(true)).expect("a plan");
    assert_tiles(&spans, 8_000, 3_500);
    assert!(spans.iter().all(|s| s.transcribe));
    // A tighter bound than the turn rule's own chunks splits them further,
    // by the silence rule at the tighter mark.
    let tight = OfflineRule {
        max_frames: 2_000,
        ..rule(true)
    };
    let m = mel(5_000, &[1_400, 3_300]);
    let spans = plan_offline(&m, Some(&a), MEL_SECS, &tight).expect("a plan");
    assert_tiles(&spans, 5_000, 2_000);
    assert_eq!(spans.first(), Some(&span(0, 1_400, true)));
}

#[test]
fn without_activity_the_plan_is_the_silence_rule() {
    let m = mel(7_000, &[2_800, 6_100]);
    let cuts = silence_cuts(&m, 3_000, 500, 10);
    assert_eq!(cuts, vec![2_800, 6_100]);
    for skip in [true, false] {
        let spans = plan_offline(&m, None, MEL_SECS, &rule(skip)).expect("a plan");
        assert_eq!(spans, spans_at(&cuts, 7_000));
    }
    assert_eq!(
        plan_offline(&mel(900, &[]), None, MEL_SECS, &rule(true)).expect("a plan"),
        vec![span(0, 900, true)]
    );
}

#[test]
fn an_all_silent_clip_has_no_transcribed_span() {
    let a = activity(2, &[(None, 125)]);
    let spans = plan_offline(&mel(1_000, &[]), Some(&a), MEL_SECS, &rule(true)).expect("a plan");
    assert_eq!(spans, vec![span(0, 1_000, false)]);
    // A clip whose last few mel frames fall past the last diarizer frame:
    // that frame covers them.
    let a = activity(2, &[(None, 124)]);
    let spans = plan_offline(&mel(997, &[]), Some(&a), MEL_SECS, &rule(true)).expect("a plan");
    assert_eq!(spans, vec![span(0, 997, false)]);
    // Long enough to be cut by the silence rule: the cuts inside the silence
    // go, and one span covers it.
    let a = activity(1, &[(None, 875)]);
    let spans =
        plan_offline(&mel(7_000, &[2_800]), Some(&a), MEL_SECS, &rule(true)).expect("a plan");
    assert_eq!(spans, vec![span(0, 7_000, false)]);
}

#[test]
fn skip_silence_off_transcribes_everything() {
    let a = activity(2, &[(Some(0), 50), (None, 50), (Some(1), 50)]);
    let m = mel(1_200, &[]);
    let spans = plan_offline(&m, Some(&a), MEL_SECS, &rule(false)).expect("a plan");
    assert_eq!(spans, spans_at(&turn_cuts(&a, &m, MEL_SECS), 1_200));
    let a = activity(2, &[(None, 125)]);
    let spans = plan_offline(&mel(1_000, &[]), Some(&a), MEL_SECS, &rule(false)).expect("a plan");
    assert_eq!(spans, vec![span(0, 1_000, true)]);
}

#[test]
fn a_rule_or_plan_that_cannot_hold_is_refused() {
    let m = mel(1_000, &[]);
    let refused = |r: &OfflineRule, secs: f64| {
        matches!(
            plan_offline(&m, None, secs, r),
            Err(AudioError::BadSpec { .. })
        )
    };
    // A bound no wider than the search leaves no mark to search around.
    assert!(refused(
        &OfflineRule {
            max_frames: 500,
            ..rule(true)
        },
        MEL_SECS
    ));
    assert!(refused(
        &OfflineRule {
            window_frames: 0,
            ..rule(true)
        },
        MEL_SECS
    ));
    assert!(refused(
        &OfflineRule {
            skip_silence_secs: 0.0,
            ..rule(true)
        },
        MEL_SECS
    ));
    assert!(refused(
        &OfflineRule {
            pad_secs: f64::NAN,
            ..rule(true)
        },
        MEL_SECS
    ));
    assert!(refused(&rule(true), 0.0));
    // A plan with a gap, an overlap, an empty span or a short end is not a
    // plan.
    assert!(check_tiling(&[span(0, 10, true), span(10, 20, false)], 20).is_ok());
    for bad in [
        vec![span(0, 10, true), span(11, 20, true)],
        vec![span(0, 10, true), span(9, 20, true)],
        vec![span(0, 10, true), span(10, 10, true), span(10, 20, true)],
        vec![span(0, 19, true)],
        vec![span(1, 20, true)],
        Vec::new(),
    ] {
        assert!(
            matches!(check_tiling(&bad, 20), Err(AudioError::BadSpec { .. })),
            "{bad:?}"
        );
    }
    assert!(check_tiling(&[], 0).is_ok());
}
