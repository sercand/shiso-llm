//! Transcript assembly: words, segments, speakers, subtitles.

use super::*;

fn tp(text: &str, frame: u32) -> TimedPiece {
    TimedPiece {
        text: text.to_owned(),
        frame,
    }
}

#[test]
fn pieces_group_into_timed_words() {
    let w = words(
        &[
            tp("Hel", 0),
            tp("lo", 1),
            tp(",", 2),
            tp(" world", 5),
            tp(".", 6),
        ],
        0.08,
    );
    assert_eq!(w.len(), 2);
    assert_eq!(w[0].text, "Hello,");
    assert!((w[0].start - 0.0).abs() < 1e-9 && (w[0].end - 0.24).abs() < 1e-9);
    assert_eq!(w[1].text, "world.");
    assert!((w[1].end - 0.56).abs() < 1e-9);
}

#[test]
fn segments_split_on_sentences_and_gaps() {
    let w = words(
        &[
            tp("Hi.", 0),
            tp(" How", 10),
            tp(" are", 12),
            tp(" you", 14),
            tp(" there", 40),
        ],
        0.08,
    );
    let s = segments(&w);
    let texts: Vec<&str> = s.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["Hi.", "How are you", "there"]);
    assert_eq!(s.iter().map(|s| s.id).collect::<Vec<_>>(), [0, 1, 2]);
}

#[test]
fn turns_merge_short_gaps_and_drop_blips() {
    // 10 ms frames, 2 speakers: A 0-1.0 s with a 50 ms dropout, B 1.2-2.0 s, a 30 ms A blip at 1.5 s.
    let frames = 200;
    let mut probs = vec![0.0f32; frames * 2];
    for t in 0..100 {
        if !(40..45).contains(&t) {
            probs[t * 2] = 0.9;
        }
    }
    for t in 120..200 {
        probs[t * 2 + 1] = 0.8;
    }
    for t in 150..153 {
        probs[t * 2] = 0.7;
    }
    let act = SpeakerActivity::new(0.01, 2, probs).expect("shape");
    let rule = TurnRule {
        threshold: 0.5,
        min_secs: 0.1,
        merge_gap_secs: 0.1,
    };
    let t = turns(&act, &rule);
    assert_eq!(t.len(), 2, "{t:?}");
    assert_eq!((t[0].speaker, t[1].speaker), (0, 1));
    assert!((t[0].end - 1.0).abs() < 1e-9 && (t[1].start - 1.2).abs() < 1e-9);
}

#[test]
fn words_follow_the_dominant_speaker() {
    let mut probs = vec![0.0f32; 100 * 2];
    for t in 0..50 {
        probs[t * 2] = 0.9;
    }
    for t in 50..100 {
        probs[t * 2 + 1] = 0.9;
    }
    let act = SpeakerActivity::new(0.01, 2, probs).expect("shape");
    let w = words(
        &[tp("one", 0), tp(" two", 3), tp(" three", 6), tp(" four", 9)],
        0.08,
    );
    let who = assign_speakers(&w, &act);
    assert_eq!(who, [0, 0, 1, 1]);
    let segs = diarized_segments(&w, &who);
    assert_eq!(segs.len(), 2);
    assert_eq!(
        (segs[0].speaker, segs[0].text.as_str()),
        (Some(0), "one two")
    );
    assert_eq!(
        (segs[1].speaker, segs[1].text.as_str()),
        (Some(1), "three four")
    );
}

#[test]
fn silence_inherits_the_previous_speaker() {
    let mut probs = vec![0.0f32; 100 * 2];
    for t in 0..20 {
        probs[t * 2 + 1] = 0.9;
    }
    let act = SpeakerActivity::new(0.01, 2, probs).expect("shape");
    let w = words(&[tp("a", 0), tp(" b", 10)], 0.08); // b at 0.8 s is silent
    assert_eq!(assign_speakers(&w, &act), [1, 1]);
}

#[test]
fn labels_and_subtitles() {
    assert_eq!(speaker_label(0), "A");
    assert_eq!(speaker_label(25), "Z");
    assert_eq!(speaker_label(26), "AA");
    let segs = [
        Segment {
            id: 0,
            start: 0.0,
            end: 1.5,
            text: "Hello.".into(),
            speaker: None,
        },
        Segment {
            id: 1,
            start: 3_723.456_7,
            end: 3_725.0,
            text: "Bye.".into(),
            speaker: Some(1),
        },
    ];
    assert_eq!(
        srt(&segs),
        "1\n00:00:00,000 --> 00:00:01,500\nHello.\n\n2\n01:02:03,457 --> 01:02:05,000\nB: Bye.\n\n"
    );
    assert_eq!(
        vtt(&segs),
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello.\n\n01:02:03.457 --> 01:02:05.000\n<v B>Bye.\n\n"
    );
}

#[test]
fn activity_shape_is_checked() {
    assert!(SpeakerActivity::new(0.01, 2, vec![0.0; 5]).is_err());
    assert!(SpeakerActivity::new(0.0, 2, vec![0.0; 4]).is_err());
}

#[test]
fn vtt_cue_text_is_escaped() {
    // `<` opens a cue tag and `&` an entity in WebVTT; a transcript that says
    // "AT&T" or "a < b" must survive a player. SubRip carries text verbatim.
    let segs = [Segment {
        id: 0,
        start: 0.0,
        end: 1.0,
        text: "AT&T said a < b > c".into(),
        speaker: None,
    }];
    assert!(
        vtt(&segs).contains("\nAT&amp;T said a &lt; b &gt; c\n"),
        "{}",
        vtt(&segs)
    );
    assert!(srt(&segs).contains("\nAT&T said a < b > c\n"));
}

fn word(text: &str, start: f64, end: f64) -> Word {
    Word {
        text: text.to_owned(),
        start,
        end,
    }
}

#[test]
fn a_chunked_transcript_changes_speaker_only_at_a_chunk_cut() {
    // Speaker 0 until 0.9 s, silence to 1.1 s, speaker 1 after; the chunks
    // are cut at 1.0 s, the middle of the silence.
    let probs = (0..20)
        .flat_map(|t| match t {
            0..9 => [0.9, 0.0],
            9..11 => [0.1, 0.1],
            _ => [0.0, 0.9],
        })
        .collect();
    let activity = SpeakerActivity::new(0.1, 2, probs).expect("valid");
    // The second chunk's first word starts at its first frame, inside the
    // silence: by itself it would inherit the previous speaker.
    let words = [
        word("one", 0.0, 0.4),
        word("two", 0.5, 0.95),
        word("three", 1.0, 1.08),
        word("four", 1.5, 1.9),
    ];
    assert_eq!(assign_speakers(&words, &activity), [0, 0, 0, 1]);
    assert_eq!(
        assign_speakers_by_chunk(&words, &activity, &[1.0]),
        [0, 0, 1, 1]
    );
}

#[test]
fn a_chunk_nobody_holds_keeps_each_words_own_speaker() {
    // Speaker 1 speaks, then a chunk of silence, then speaker 0.
    let probs = (0..30)
        .flat_map(|t| match t {
            0..10 => [0.0, 0.9],
            10..20 => [0.1, 0.1],
            _ => [0.9, 0.0],
        })
        .collect();
    let activity = SpeakerActivity::new(0.1, 2, probs).expect("valid");
    let words = [
        word("a", 0.2, 0.4),
        word("b", 1.2, 1.4),
        word("c", 2.2, 2.4),
    ];
    assert_eq!(
        assign_speakers_by_chunk(&words, &activity, &[1.0, 2.0]),
        assign_speakers(&words, &activity)
    );
    assert_eq!(
        assign_speakers_by_chunk(&words, &activity, &[1.0, 2.0]),
        [1, 1, 0]
    );
}

#[test]
fn a_chunked_words_end_at_the_next_word_or_its_chunks_end() {
    // Two chunks, frames [0, 100) and [100, 250), at 0.01 s a frame: each
    // piece has only the frame it starts at.
    let pieces = [
        tp("Hel", 0),
        tp("lo", 20),
        tp(" there.", 60),
        tp(" Next", 100),
        tp(" one", 180),
    ];
    let w = words_in_chunks(&pieces, 0.01, &[100, 250]);
    let times: Vec<(&str, f64, f64)> = w
        .iter()
        .map(|w| (w.text.as_str(), w.start, w.end))
        .collect();
    let want = [
        ("Hello", 0.0, 0.6),
        ("there.", 0.6, 1.0),
        ("Next", 1.0, 1.8),
        ("one", 1.8, 2.5),
    ];
    assert_eq!(times.len(), want.len(), "{times:?}");
    for (got, want) in times.iter().zip(want) {
        assert_eq!(got.0, want.0);
        assert!(
            (got.1 - want.1).abs() < 1e-9 && (got.2 - want.2).abs() < 1e-9,
            "{got:?} against {want:?}"
        );
    }
    // A one-word chunk lasts the whole chunk, not one frame.
    let lone = words_in_chunks(&[tp("Yes.", 300)], 0.01, &[300, 420]);
    assert!((lone[0].end - 4.2).abs() < 1e-9, "{lone:?}");
    // Without chunk ends a word ends one frame after its last piece, as
    // `words` has it.
    assert_eq!(words_in_chunks(&pieces, 0.08, &[]), words(&pieces, 0.08));
}
