//! Each response format's exact body, and the stream's exact frames, for one
//! fixed transcript.

use serde_json::{Value, json};
use shiso_audio::transcript::{Segment, Word};

use super::{TranscriptionFrames, body};
use crate::audio::{JobFailure, Transcript, TranscriptEvent};
use crate::openai::transcription_wire::{Granularity, ResponseFormat};

fn transcript(diarized: bool) -> Transcript {
    let words = vec![
        Word {
            text: "Hello".into(),
            start: 0.0,
            end: 0.4,
        },
        Word {
            text: "there.".into(),
            start: 0.4,
            end: 0.8,
        },
        Word {
            text: "Hi.".into(),
            start: 1.2,
            end: 1.5,
        },
    ];
    let segments = vec![
        Segment {
            id: 0,
            start: 0.0,
            end: 0.8,
            text: "Hello there.".into(),
            speaker: diarized.then_some(0),
        },
        Segment {
            id: 1,
            start: 1.2,
            end: 1.5,
            text: "Hi.".into(),
            speaker: diarized.then_some(1),
        },
    ];
    Transcript {
        text: "Hello there. Hi.".into(),
        language: Some("en-US".into()),
        duration: 1.52,
        words,
        segments,
        diarized,
    }
}

fn json_of(format: ResponseFormat, granularities: &[Granularity], diarized: bool) -> Value {
    let (content_type, text) = body(format, &transcript(diarized), granularities);
    assert!(
        content_type.starts_with("application/json"),
        "{content_type}"
    );
    serde_json::from_str(&text).expect("json")
}

#[test]
fn json_is_the_text_and_the_duration_used() {
    assert_eq!(
        json_of(ResponseFormat::Json, &[], false),
        json!({"text": "Hello there. Hi.", "usage": {"type": "duration", "seconds": 2}})
    );
}

#[test]
fn text_is_the_text_alone() {
    let (content_type, text) = body(ResponseFormat::Text, &transcript(false), &[]);
    assert!(content_type.starts_with("text/plain"), "{content_type}");
    assert_eq!(text, "Hello there. Hi.\n");
}

#[test]
fn verbose_json_carries_segments_by_default_and_words_when_asked() {
    let plain = json_of(ResponseFormat::VerboseJson, &[], false);
    assert_eq!(plain["task"], "transcribe");
    assert_eq!(plain["language"], "en-US");
    assert_eq!(plain["duration"], 1.52);
    assert_eq!(plain["segments"].as_array().map(Vec::len), Some(2));
    assert_eq!(plain["segments"][1]["id"], 1);
    assert_eq!(plain["segments"][1]["text"], "Hi.");
    assert!(plain.get("words").is_none());
    let words = json_of(ResponseFormat::VerboseJson, &[Granularity::Word], false);
    assert_eq!(
        words["words"][1],
        json!({"word": "there.", "start": 0.4, "end": 0.8})
    );
    assert!(words.get("segments").is_none(), "only what was asked");
    let both = json_of(
        ResponseFormat::VerboseJson,
        &[Granularity::Word, Granularity::Segment],
        false,
    );
    assert!(both.get("words").is_some() && both.get("segments").is_some());
}

#[test]
fn diarized_json_labels_each_segment_with_its_speaker() {
    let value = json_of(ResponseFormat::DiarizedJson, &[], true);
    assert_eq!(value["task"], "transcribe");
    assert_eq!(value["text"], "Hello there. Hi.");
    assert_eq!(
        value["segments"][1],
        json!({"type": "transcript.text.segment", "id": "seg_1", "start": 1.2, "end": 1.5,
               "text": "Hi.", "speaker": "B"})
    );
    assert_eq!(value["usage"]["seconds"], 2);
}

#[test]
fn subtitles_are_the_segments_timed() {
    let (_, srt) = body(ResponseFormat::Srt, &transcript(false), &[]);
    assert!(
        srt.starts_with("1\n00:00:00,000 --> 00:00:00,800\nHello there.\n"),
        "{srt}"
    );
    let (_, vtt) = body(ResponseFormat::Vtt, &transcript(false), &[]);
    assert!(
        vtt.starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:00.800\n"),
        "{vtt}"
    );
}

fn frames(diarized: bool, events: &[TranscriptEvent]) -> Vec<Value> {
    let mut encoder = TranscriptionFrames::new(diarized);
    events
        .iter()
        .flat_map(|e| encoder.on_event(e))
        .map(|f| serde_json::from_str(&f.data).expect("json frame"))
        .collect()
}

#[test]
fn a_plain_stream_is_deltas_then_the_whole_text() {
    let t = transcript(false);
    let got = frames(
        false,
        &[
            TranscriptEvent::Delta("Hello".into()),
            TranscriptEvent::Delta(" there. Hi.".into()),
            TranscriptEvent::Done(t),
        ],
    );
    assert_eq!(
        got,
        vec![
            json!({"type": "transcript.text.delta", "delta": "Hello"}),
            json!({"type": "transcript.text.delta", "delta": " there. Hi."}),
            json!({"type": "transcript.text.done", "text": "Hello there. Hi.",
                   "usage": {"type": "duration", "seconds": 2}}),
        ]
    );
}

#[test]
fn a_diarized_stream_is_segments_then_the_whole_text() {
    let t = transcript(true);
    let first = t.segments[0].clone();
    let got = frames(
        true,
        &[
            TranscriptEvent::Delta("Hello".into()),
            TranscriptEvent::Segment(first),
            TranscriptEvent::Done(t),
        ],
    );
    assert_eq!(got.len(), 3, "{got:?}");
    assert_eq!(got[0]["type"], "transcript.text.segment");
    assert_eq!(got[0]["speaker"], "A");
    assert_eq!(
        got[1]["id"], "seg_1",
        "the last segment arrives with the end"
    );
    assert_eq!(got[1]["speaker"], "B");
    assert_eq!(got[2]["type"], "transcript.text.done");
}

#[test]
fn a_failure_mid_stream_is_an_error_frame() {
    let got = frames(
        false,
        &[TranscriptEvent::Failed(JobFailure {
            message: "the device fell over".into(),
            caller: false,
        })],
    );
    assert_eq!(
        got,
        vec![json!({"type": "error", "error": {"message": "the device fell over"}})]
    );
}
