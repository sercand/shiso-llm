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
