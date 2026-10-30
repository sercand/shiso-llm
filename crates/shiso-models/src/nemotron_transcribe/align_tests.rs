//! The alignment against the reference's own word times.
//!
//! The reference-fed cases need `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER`
//! (for the tokenizer) and the captures' alignment dumps; each says on stderr
//! why it skipped. Fed the reference's log-probabilities and speaker
//! activity, the port must give its words and times to the millisecond.

use std::path::Path;

use super::{Activity, AlignDesc, TranscriptWord, align, head_windows, targets, transcript_words};
use crate::nemotron_transcribe::encoder_tests::floats;
use crate::nemotron_transcribe::spm::PieceId;
use crate::nemotron_transcribe::spm::tests::{aligned_refs, tokenizer};

/// The reference's words: text, start, end.
pub(crate) fn reference_words(dir: &Path) -> Vec<(String, f64, f64)> {
    let doc: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("transcript.json")).expect("the transcript reads"),
    )
    .expect("the transcript parses");
    doc["words"]
        .as_array()
        .expect("words")
        .iter()
        .map(|w| {
            (
                w["word"].as_str().expect("a word").to_owned(),
                w["start"].as_f64().expect("a start"),
                w["end"].as_f64().expect("an end"),
            )
        })
        .collect()
}

/// The clip's length the reference aligned against.
pub(crate) fn duration(dir: &Path) -> f64 {
    let b = std::fs::read(dir.join("ctc_duration.f64")).expect("the duration reads");
    f64::from_le_bytes(b[..8].try_into().expect("eight bytes"))
}

#[expect(clippy::print_stderr, reason = "the agreement is the test's report")]
fn words_match(rel: &str) {
    let Some(dir) = aligned_refs(rel) else {
        return;
    };
    let Some(tok) = tokenizer() else {
        return;
    };
    let transcript =
        std::fs::read_to_string(dir.join("ctc_transcript.txt")).expect("the transcript reads");
    let log_probs = floats(&dir.join("ctc_logprobs.f32"));
    let rows80 = floats(&dir.join("diar_probs.f32"));
    let rows10 = floats(&dir.join("spk_probs10.f32"));
    let frames = 2 * rows80.len() / 8;
    let t = targets(&tok, &transcript, frames).expect("the transcript has targets");
    let want_columns: Vec<u32> = std::fs::read(dir.join("ctc_columns.i32"))
        .expect("the columns read")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let got_columns: Vec<u32> = t.columns.iter().map(|&PieceId(id)| id).collect();
    assert_eq!(got_columns, want_columns, "{rel}: the classes aligned");
    assert_eq!(log_probs.len(), frames * t.columns.len(), "{rel}: frames");
    let got = align(
        &AlignDesc {
            targets: &t,
            log_probs: &log_probs,
            frames,
            activity: Some(Activity {
                rows80: &rows80,
                rows10: &rows10,
                speakers: 8,
            }),
            duration: duration(&dir),
        },
        2,
    )
    .expect("aligns");
    let want = reference_words(&dir);
    let differ: Vec<usize> = got
        .iter()
        .zip(&want)
        .enumerate()
        .filter(|(_, (g, w))| g.text != w.0 || g.start != w.1 || g.end != w.2)
        .map(|(i, _)| i)
        .collect();
    eprintln!(
        "{rel}: {} words, {} reference words, {} differ",
        got.len(),
        want.len(),
        differ.len()
    );
    for &i in differ.iter().take(8) {
        eprintln!("  #{i}: ours {:?}, reference {:?}", got[i], want[i]);
    }
    assert_eq!(got.len(), want.len(), "{rel}: word count");
    assert!(differ.is_empty(), "{rel}: {} words differ", differ.len());
}

#[test]
fn short_en_verbatim_words_match_reference() {
    words_match("short_en/verbatim");
}

#[test]
fn short_en_speakers_words_match_reference() {
    words_match("short_en/speakers");
}

#[test]
fn short_en_translate_words_match_reference() {
    words_match("short_en/translate");
}

#[test]
fn meeting_60s_verbatim_words_match_reference() {
    words_match("meeting_60s/verbatim");
}

#[test]
fn meeting_60s_speakers_words_match_reference() {
    words_match("meeting_60s/speakers");
}

#[test]
fn meeting_10m_verbatim_words_match_reference() {
    words_match("meeting_10m/verbatim");
}

#[test]
fn speaker_tags_start_turns() {
    let words = transcript_words("hi <spk:1> yo\u{2014}there <SPK:0>ok");
    let want = [
        ("hi", None),
        ("yo", Some(1)),
        ("there", Some(1)),
        ("ok", Some(0)),
    ];
    assert_eq!(
        words,
        want.iter()
            .map(|(w, tag)| TranscriptWord {
                word: (*w).to_owned(),
                tag: *tag,
            })
            .collect::<Vec<_>>()
    );
    let untagged = transcript_words("  a  b ");
    assert!(untagged.iter().all(|w| w.tag == Some(0)));
    assert_eq!(untagged.len(), 2);
    // A malformed tag is a word.
    assert_eq!(transcript_words("<spk:> x").len(), 2);
}

#[test]
fn head_windows_cover_the_clip_once() {
    assert_eq!(head_windows(138, 500, 50, 50).len(), 1);
    let w = head_windows(1201, 500, 50, 50);
    assert_eq!(
        w.iter()
            .map(|w| (w.start, w.end, w.context_start, w.context_end))
            .collect::<Vec<_>>(),
        [
            (0, 500, 0, 550),
            (500, 1000, 450, 1050),
            (1000, 1201, 950, 1201)
        ]
    );
}

#[test]
fn the_reference_log_is_close_to_the_true_one() {
    for x in [1e-6f32, 0.01, 0.3, 0.5, 0.707_106_77, 1.0, 1.5, 7.0] {
        assert!((super::ref_log(x) - x.ln()).abs() < 1e-6, "{x}");
    }
    assert_eq!(super::ref_log(0.0), f32::NEG_INFINITY);
}

#[test]
fn rounding_is_decimal_with_ties_to_even() {
    assert_eq!(super::round_decimals(0.637_681_159_420_289_8, 3), 0.638);
    assert_eq!(super::round_decimals(0.0625, 3), 0.062);
    assert_eq!(super::round_decimals(2.280_000_000_000_000_2, 3), 2.28);
}

// ---- the helpers, by hand ----

use super::{
    AlignError, Moves, SpeakerStream, Targets, band_starts, blocked_dot, character_script,
    digit_value, is_space, minimum_frames, pairwise_sum, split_words,
};

#[test]
fn speaker_tags_fold_case_as_the_reference_does() {
    // The long s folds to `s` and the Kelvin sign to `k`; any script's
    // decimal digits number the speaker.
    let words = transcript_words("<\u{17f}p\u{212a}:2>a <SPK:\u{ff11}\u{ff12}>b <spk:\u{663}>c");
    let tags: Vec<_> = words.iter().map(|w| (w.word.as_str(), w.tag)).collect();
    assert_eq!(tags, [("a", Some(2)), ("b", Some(12)), ("c", Some(3))]);
    // A letter that only looks like one is not folded.
    assert_eq!(transcript_words("<\u{3c3}pk:1> x").len(), 2);
}

#[test]
fn digit_values_cover_every_script_and_stop_at_nine() {
    assert_eq!(digit_value('7'), Some(7));
    assert_eq!(digit_value('\u{669}'), Some(9), "Arabic-Indic nine");
    assert_eq!(digit_value('\u{ff10}'), Some(0), "fullwidth zero");
    // The mathematical runs abut: each is ten digits, then the next zero.
    assert_eq!(digit_value('\u{1d7d7}'), Some(9));
    assert_eq!(digit_value('\u{1d7d8}'), Some(0));
    assert_eq!(digit_value('\u{1fbf9}'), Some(9), "the last run");
    assert_eq!(digit_value(':'), None, "one past '9'");
    assert_eq!(digit_value('/'), None, "one before '0'");
    assert_eq!(digit_value('\u{65f}'), None, "between runs");
    assert_eq!(digit_value('a'), None);
}

#[test]
fn spaces_are_the_reference_isspace() {
    for c in [
        '\t', '\u{1c}', ' ', '\u{85}', '\u{a0}', '\u{2009}', '\u{202f}', '\u{3000}',
    ] {
        assert!(is_space(c), "{c:?} is a space");
    }
    for c in ['x', '\u{200b}', '\u{feff}', '\u{180e}'] {
        assert!(!is_space(c), "{c:?} is not");
    }
    assert_eq!(
        split_words("a\u{3000}b\u{a0}\u{a0}c\u{200b}d"),
        ["a", "b", "c\u{200b}d"]
    );
}

#[test]
fn character_scripts_are_refused_but_korean_is_kept() {
    let script = |text: &str| character_script(&transcript_words(text));
    assert_eq!(script("hello there"), None);
    assert_eq!(script("こんにちは"), Some("Japanese"));
    assert_eq!(script("漢字 かな"), Some("Japanese"), "kana settles Han");
    assert_eq!(script("你好"), Some("Chinese"));
    assert_eq!(script("สวัสดี"), Some("Thai"));
    // Hangul anywhere keeps the words, Han beside it included.
    assert_eq!(script("안녕 你好"), None);
    // An event tag's text is not the transcript's script.
    assert_eq!(script("ok <ev:你好>"), None);
}

#[test]
fn minimum_frames_count_one_per_piece_and_one_between_repeats() {
    assert_eq!(minimum_frames(&[0]), 0);
    assert_eq!(minimum_frames(&[0, 4, 0]), 1);
    assert_eq!(minimum_frames(&[0, 4, 0, 5, 0]), 2);
    assert_eq!(minimum_frames(&[0, 4, 0, 4, 0]), 3);
    assert_eq!(minimum_frames(&[0, 4, 0, 4, 0, 4, 0, 5, 0]), 6);
}

/// Values whose sum depends on the order they are added in.
fn uneven(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let i = i as f64;
            (i * 0.37).sin() * 10f64.powi((i as i32 % 7) - 3) + 1.0 / (i + 3.0)
        })
        .collect()
}

#[test]
fn pairwise_sum_reduces_in_the_reference_order() {
    // Under 8: left to right.
    let a = uneven(7);
    assert_eq!(pairwise_sum(&a), a.iter().fold(0.0, |s, v| s + v));
    // 9: eight lanes summed as a tree, then the tail.
    let a = uneven(9);
    let tree = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
    assert_eq!(pairwise_sum(&a), tree + a[8]);
    // 16: lane k holds a[k] + a[k + 8].
    let a = uneven(16);
    let r: Vec<f64> = (0..8).map(|k| a[k] + a[k + 8]).collect();
    let tree = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    assert_eq!(pairwise_sum(&a), tree);
    // Past 128: halves rounded down to a multiple of 8, recursively.
    let a = uneven(129);
    assert_eq!(
        pairwise_sum(&a),
        pairwise_sum(&a[..64]) + pairwise_sum(&a[64..])
    );
    let a = uneven(257);
    assert_eq!(
        pairwise_sum(&a),
        pairwise_sum(&a[..128]) + pairwise_sum(&a[128..])
    );
    for n in [8, 127, 128, 129, 255, 256, 257, 511, 512, 513, 4099] {
        let a = uneven(n);
        let plain: f64 = a.iter().sum();
        assert!(
            (pairwise_sum(&a) - plain).abs() <= 1e-12 * plain.abs().max(1.0),
            "{n}"
        );
    }
}

#[test]
fn blocked_dot_chains_blocks_of_256() {
    let chain = |a: &[f64], b: &[f64], r: std::ops::Range<usize>| {
        r.fold(0.0f64, |acc, k| a[k].mul_add(b[k], acc))
    };
    let dot = |n: usize| (uneven(n), uneven(n + 5)[5..].to_vec());
    // Up to 256: one fused chain.
    for n in [1, 8, 255, 256] {
        let (a, b) = dot(n);
        assert_eq!(blocked_dot(&a, &b), chain(&a, &b, 0..n), "{n}");
    }
    // 257 to 511: halves rounded up to 4.
    let (a, b) = dot(300);
    assert_eq!(
        blocked_dot(&a, &b),
        chain(&a, &b, 0..152) + chain(&a, &b, 152..300)
    );
    let (a, b) = dot(257);
    assert_eq!(
        blocked_dot(&a, &b),
        chain(&a, &b, 0..132) + chain(&a, &b, 132..257)
    );
    // 512 and past: whole blocks of 256 first, summed in order.
    let (a, b) = dot(512);
    assert_eq!(
        blocked_dot(&a, &b),
        chain(&a, &b, 0..256) + chain(&a, &b, 256..512)
    );
    let (a, b) = dot(1000);
    let want = ((chain(&a, &b, 0..256) + chain(&a, &b, 256..512)) + chain(&a, &b, 512..756))
        + chain(&a, &b, 756..1000);
    assert_eq!(blocked_dot(&a, &b), want);
    for n in [511, 513, 2048] {
        let (a, b) = dot(n);
        let plain: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!(
            (blocked_dot(&a, &b) - plain).abs() <= 1e-12 * plain.abs().max(1.0),
            "{n}"
        );
    }
}

#[test]
fn band_starts_follow_the_anchors() {
    // No anchor: the line from the first state to the last, a band of three
    // states around it, kept reachable at two states a frame.
    assert_eq!(
        band_starts(&[0; 5], &[1, 2, 3, 4], 9, 1, 3),
        [0, 1, 3, 5, 6]
    );
    // The greedy onsets 5 6 7 at frames 1-3 are the target's first trigram,
    // so the line bends through state 3 (piece 6) at frame 2.
    let greedy = [0, 5, 6, 7, 0, 0, 0, 0, 0, 0];
    assert_eq!(
        band_starts(&greedy, &[5, 6, 7, 8, 9], 11, 1, 3),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 8]
    );
}

#[test]
fn moves_pack_four_to_a_byte() {
    let mut m = Moves::new(9);
    for (cell, step) in [(0, 2), (1, 1), (3, 2), (4, 1), (8, 2)] {
        m.set(cell, step);
    }
    let got: Vec<u8> = (0..9).map(|c| m.get(c)).collect();
    assert_eq!(got, [2, 1, 0, 2, 1, 0, 0, 0, 2]);
}

/// One word, one piece (column 1), over `frames` frames.
fn one_word() -> Targets {
    Targets {
        words: transcript_words("a"),
        streams: vec![SpeakerStream {
            tag: Some(0),
            words: vec![0],
            labels: vec![0, 1, 0],
            state_word: vec![None, Some(0), None],
        }],
        columns: vec![PieceId(0), PieceId(1)],
    }
}

#[test]
fn activity_short_of_whole_rows_is_refused() {
    let t = one_word();
    let log_probs = [-0.1f32, -2.0, -2.0, -0.1, -0.1, -2.0, -0.1, -2.0];
    let run = |rows80: &[f32], rows10: &[f32]| {
        align(
            &AlignDesc {
                targets: &t,
                log_probs: &log_probs,
                frames: 4,
                activity: Some(Activity {
                    rows80,
                    rows10,
                    speakers: 8,
                }),
                duration: 0.32,
            },
            2,
        )
    };
    let whole = [1.0f32; 16];
    // Fewer values than one row of speakers, a ragged buffer, none at all.
    for (rows80, rows10) in [
        (&whole[..4], &whole[..]),
        (&whole[..12], &whole[..]),
        (&whole[..], &whole[..9]),
        (&whole[..0], &whole[..]),
    ] {
        let err = run(rows80, rows10).expect_err("not whole rows");
        assert!(matches!(err, AlignError::Inputs { .. }), "{err:?}");
    }
    let words = run(&whole, &whole).expect("whole rows align");
    assert_eq!(words.len(), 1);
}
