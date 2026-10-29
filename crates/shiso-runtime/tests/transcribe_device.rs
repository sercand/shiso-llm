//! Transcription through the speech decoder on the Metal device, booted the
//! way `shiso serve` boots it, against the reference runtime's transcripts.
//!
//! Needs `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE` (the decoder `.gguf`, its
//! encoder, tokenizer and template beside it), the reference captures and the
//! clips under `~/models/clips` (or `SHISO_TEST_CLIPS`):
//! `cargo test --release -p shiso-runtime --features metal --test transcribe_device -- --ignored --nocapture`.
#![cfg(all(target_os = "macos", feature = "metal"))]

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use shiso_audio::{AudioDecodePolicy, DecodeLimits, Pcm};
use shiso_ir::{Block, MediaSource, Message, Request, Role, ThinkingDirective};
use shiso_runtime::api::transcribe::{Preset, TranscribeOutput, TranscribeRequest};
use shiso_runtime::options::ServeOptions;
use shiso_runtime::runtime::Runtime;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

fn pcm(path: &Path) -> Pcm {
    let bytes = std::fs::read(path).expect("the clip reads");
    let limits = DecodeLimits::new(NonZeroU32::new(3600).expect("non-zero"));
    shiso_audio::decode_audio(&bytes, &limits, &AudioDecodePolicy::default()).expect("decodes")
}

/// The reference runtime's transcript text for `clip` under `preset`.
fn reference_text(clip: &str, preset: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../test-vectors/references/nemotron_transcribe/{clip}/{preset}/transcript.json"
    ));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("the transcript reads"))
            .expect("parses");
    json["text"].as_str().expect("a text field").to_owned()
}

/// The reference runtime's words for `clip` under `preset`: text, start, end.
fn reference_words(clip: &str, preset: &str) -> Vec<(String, f64, f64)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../test-vectors/references/nemotron_transcribe/{clip}/{preset}/transcript.json"
    ));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("the transcript reads"))
            .expect("parses");
    json["words"]
        .as_array()
        .expect("a words field")
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

fn transcribe(rt: &Runtime, pcm: Pcm, preset: Preset, words: bool) -> TranscribeOutput {
    futures::executor::block_on(rt.transcribe(TranscribeRequest { pcm, preset, words }))
        .expect("transcribes")
}

/// Word times on `short_en`: the decoder's own words, timed by the tower's
/// head. The comma the decoder leaves out is part of the word it follows, so
/// words are compared without commas; one 40 ms frame is the budget.
#[expect(clippy::print_stderr, reason = "the run reports how close it came")]
fn short_en_words_match(solo: &TranscribeOutput) {
    let words = solo.words.as_ref().expect("words were asked for");
    let want = reference_words("short_en", "verbatim");
    assert_eq!(words.len(), want.len(), "word count");
    let mut exact = 0;
    let mut worst = 0.0f64;
    for (got, want) in words.iter().zip(&want) {
        assert_eq!(got.text.replace(',', ""), want.0.replace(',', ""));
        let off = (got.start - want.1).abs().max((got.end - want.2).abs());
        if off == 0.0 {
            exact += 1;
        } else {
            eprintln!("  {got:?} against the reference's {want:?}");
        }
        worst = worst.max(off);
    }
    eprintln!(
        "short_en words: {exact} of {} timed exactly, worst {worst:.3} s off",
        want.len()
    );
    assert!(worst <= 0.0405, "a word is {worst} s off");
}

#[test]
#[ignore = "requires a Metal device, both speech exports, the reference captures and the clips"]
#[expect(
    clippy::print_stderr,
    reason = "a skipped device test says why, and the run reports"
)]
fn transcripts_match_the_reference() {
    let Some(decoder) = env_path("SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE") else {
        eprintln!("SKIP: SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE is unset");
        return;
    };
    let clips = env_path("SHISO_TEST_CLIPS")
        .unwrap_or_else(|| env_path("HOME").expect("HOME").join("models/clips"));
    let refs = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/nemotron_transcribe");
    if !refs.is_dir() || !clips.join("short_en.wav").is_file() {
        eprintln!(
            "SKIP: no captures at {} or no clips at {}",
            refs.display(),
            clips.display()
        );
        return;
    }
    let serve = ServeOptions {
        model: Some(decoder.clone()),
        ctx: 8192,
        spec_k: 3,
        expert_slots: 1600,
        ..ServeOptions::default()
    };
    let (state, join) =
        shiso_runtime::boot::boot_language_model(&serve, &decoder).expect("the decoder boots");
    let rt = Runtime::from_state(state, Some(join)).expect("a runtime");

    let short = pcm(&clips.join("short_en.wav"));
    let solo = transcribe(&rt, short.clone(), Preset::Verbatim, true);
    eprintln!("short_en verbatim: {:?}", solo.text);
    // The reference's CPU path quantizes activations before every quantized
    // multiply; on this clip that flips one near-tie (0.13 nats) between a
    // comma and the next word after "so". Every other character is graded
    // exactly, and at most one comma may differ.
    let reference = reference_text("short_en", "verbatim");
    let commas = |s: &str| s.matches(',').count();
    assert_eq!(solo.text.replace(',', ""), reference.replace(',', ""));
    assert!(commas(&solo.text).abs_diff(commas(&reference)) <= 1);
    assert!(solo.turns.is_empty());

    short_en_words_match(&solo);

    // Review Focus 1: two requests at once share the scheduler and the
    // expert stream, and each answers what it answers alone.
    let meeting = pcm(&clips.join("meeting_60s.wav"));
    let speakers = transcribe(&rt, meeting.clone(), Preset::Speakers, false);
    eprintln!("meeting_60s speakers: {:?}", speakers.text);
    assert!(
        !speakers.turns.is_empty(),
        "the speakers preset yields speaker turns"
    );
    let (a, b) = futures::executor::block_on(futures::future::join(
        rt.transcribe(TranscribeRequest {
            pcm: short,
            preset: Preset::Verbatim,
            words: true,
        }),
        rt.transcribe(TranscribeRequest {
            pcm: meeting,
            preset: Preset::Speakers,
            words: false,
        }),
    ));
    assert_eq!(a.expect("transcribes"), solo);
    assert_eq!(b.expect("transcribes"), speakers);

    // Chat audio (Task 5.4): a question about the recording, thinking on,
    // the audio part first - the template moves it to the end of the turn.
    let wav = std::fs::read(clips.join("meeting_60s.wav")).expect("the clip reads");
    let mut question = Request::new(
        "speech",
        vec![Message::new(
            Role::User,
            vec![
                Block::Audio(MediaSource::Inline(
                    base64::engine::general_purpose::STANDARD.encode(&wav),
                )),
                Block::Text(
                    "How many people speak in this recording? Answer with one number.".into(),
                ),
            ],
        )],
    );
    // Thinking is this family's default.
    assert_eq!(question.thinking, ThinkingDirective::Unspecified);
    question.sampling.temperature = Some(0.0);
    question.max_tokens = 2048;
    let answer = futures::executor::block_on(rt.generate(question)).expect("answers");
    // Asked with its thinking block open, this decoder answers a short
    // question at once and ends the turn without closing the block, so the
    // answer arrives as the trace; either place holds it.
    let mut text = answer.message.text();
    if text.trim().is_empty() {
        text = answer
            .message
            .content
            .iter()
            .filter_map(|block| match block {
                Block::Thinking(trace) => Some(trace.text.as_str()),
                _ => None,
            })
            .collect();
    }
    eprintln!("chat answer: {text:?} ({:?})", answer.stop);
    let count: u32 = text
        .trim()
        .trim_end_matches('.')
        .parse()
        .expect("the answer is one number");
    assert!(count >= 1);

    // Silence is answered empty, without encoding.
    let silence = Pcm::new(vec![0.0; 32_000], 16_000).expect("pcm");
    let quiet = transcribe(&rt, silence, Preset::Verbatim, true);
    assert_eq!(quiet.text, "");
    assert_eq!(quiet.words, Some(Vec::new()), "silence has no words");
}
