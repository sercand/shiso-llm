//! The chunk plan and the events a chunk's ids become, on the host; and the
//! whole stream against one transcription of the whole clip, on the device.

use std::sync::Arc;

use shiso_audio::chunks::silence_cuts;
use shiso_audio::{
    DecodeLimits, Detokenizer, Language, MelFrames, MelFrontend, PieceVocab, SpeechRecognizer,
    StreamOptions, TokenEvent, decode_audio,
};

use super::{Planner, SPACE, chunk_events};
use crate::qwen3_asr::config::{ASR_TEXT, is_special};

const LANGUAGE: u32 = 11528;
const ENGLISH: u32 = 6364;
const NONE: u32 = 2240;
const THIS: u32 = 1986;
const WEEK: u32 = 2003;
const DOT: u32 = 13;
const A: u32 = 64;
const B: u32 = 65;

/// A byte-level vocabulary of the checkpoint's size holding the handful of
/// pieces these tests spell, and the added tokens where the checkpoint has
/// them.
fn vocab() -> Arc<PieceVocab> {
    let mut tokens = vec!["!".to_owned(); 151_643];
    for (id, piece) in [
        (LANGUAGE, "language"),
        (ENGLISH, "\u{120}English"),
        (NONE, "\u{120}None"),
        (THIS, "This"),
        (WEEK, "\u{120}week"),
        (DOT, "."),
        (SPACE, "\u{120}"),
        (A, "a"),
        (B, "b"),
    ] {
        tokens[usize::try_from(id).expect("fits")] = piece.to_owned();
    }
    let specials: Vec<(u32, String)> = (151_643..=ASR_TEXT)
        .map(|id| {
            let text = if id == ASR_TEXT {
                "<asr_text>".to_owned()
            } else {
                format!("<|added{id}|>")
            };
            (id, text)
        })
        .collect();
    Arc::new(PieceVocab::from_byte_level(tokens, &specials).expect("a vocabulary"))
}

fn at(events: &[TokenEvent]) -> Vec<(u32, u32)> {
    events.iter().map(|e| (e.token.get(), e.frame)).collect()
}

#[test]
fn frames_interpolate_by_characters_inside_a_chunk() {
    let v = vocab();
    let ids = [LANGUAGE, ENGLISH, ASR_TEXT, THIS, WEEK, DOT];
    let chunk = chunk_events(&v, &ids, None, 100..200, false);
    // "This week." is 10 characters: "This" starts at 0, " week" at 4 and
    // "." at 9, over frames 100..200. The prefix is not emitted.
    assert_eq!(at(&chunk.events), [(THIS, 100), (WEEK, 140), (DOT, 190)]);
    assert_eq!(chunk.text, "This week.");
    assert_eq!(chunk.language.as_deref(), Some("en"));

    // After a chunk that wrote text, a space opens this one at its first
    // frame; the others keep their frames.
    let next = chunk_events(&v, &ids, None, 100..200, true);
    assert_eq!(
        at(&next.events),
        [(SPACE, 100), (THIS, 100), (WEEK, 140), (DOT, 190)]
    );
    assert_eq!(next.text, " This week.");

    // Forced: no prefix to drop, the language is the forced one.
    let forced = chunk_events(&v, &[THIS, WEEK], Some("English"), 0..90, false);
    assert_eq!(at(&forced.events), [(THIS, 0), (WEEK, 40)]);
    assert_eq!(forced.language.as_deref(), Some("en"));
}

#[test]
fn a_chunk_emits_only_the_ids_of_its_parsed_text() {
    let v = vocab();
    // Whitespace the parse strips is not emitted, at either end.
    let padded = chunk_events(
        &v,
        &[LANGUAGE, ENGLISH, ASR_TEXT, SPACE, THIS, WEEK, SPACE],
        None,
        0..100,
        false,
    );
    assert_eq!(at(&padded.events), [(THIS, 0), (WEEK, 44)]);
    // Silence: the model says so and writes nothing.
    let silent = chunk_events(&v, &[LANGUAGE, NONE, ASR_TEXT], None, 0..100, true);
    assert!(silent.events.is_empty(), "{:?}", silent.events);
    assert_eq!(silent.language, None);
    // A degenerate repeat, collapsed by the parse to one copy: the ids stop
    // where the decoded text stops agreeing with the collapsed text.
    let mut looping = vec![LANGUAGE, ENGLISH, ASR_TEXT, THIS];
    for _ in 0..30 {
        looping.extend([A, B]);
    }
    let collapsed = chunk_events(&v, &looping, None, 0..100, false);
    assert_eq!(collapsed.text, "Thisab");
    assert_eq!(
        collapsed
            .events
            .iter()
            .map(|e| e.token.get())
            .collect::<Vec<_>>(),
        [THIS, A, B]
    );
    // No tag at all: the whole output is text, with no language.
    let untagged = chunk_events(&v, &[THIS, WEEK], None, 0..100, false);
    assert_eq!(untagged.text, "This week");
    assert_eq!(untagged.language, None);
}

#[test]
fn the_specials_are_the_added_tokens_marked_special() {
    for id in [
        151_643, 151_644, 151_645, 151_669, 151_670, 151_676, 151_703,
    ] {
        assert!(is_special(id), "{id}");
    }
    // `<asr_text>`, `<think>`, `<non_speech>` and the byte alphabet are text.
    for id in [ASR_TEXT, 151_667, 151_675, 0, THIS, 151_642] {
        assert!(!is_special(id), "{id}");
    }
    // Past every table: the GGUF pads to 151936, and nothing past it is text.
    for id in [151_705, 151_935, 151_936, u32::MAX] {
        assert!(is_special(id), "{id}");
    }
    // Against the checkpoint's own flags, where it is on disk.
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN3ASR") else {
        crate::qwen3_asr::config::tests::say(
            "SKIP the tokenizer_config.json half: SHISO_TEST_MODEL_QWEN3ASR is unset",
        );
        return;
    };
    let path = std::path::Path::new(&root).join("qwen/tokenizer_config.json");
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("readable")).expect("json");
    let added = config["added_tokens_decoder"]
        .as_object()
        .expect("an object");
    for (id, entry) in added {
        let id: u32 = id.parse().expect("an id");
        assert_eq!(
            is_special(id),
            entry["special"].as_bool().expect("a flag"),
            "{id} {}",
            entry["content"]
        );
    }
}

/// Rows of `n` 4-bin frames whose energy is `level(t)`.
fn rows(n: usize, level: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).flat_map(|t| [level(t); 4]).collect()
}

/// Every end `planner` gives when `data` arrives `step` rows at a time and
/// then finishes.
fn plan(planner: &mut Planner, data: &[f32], step: usize) -> Vec<usize> {
    let mut pending: Vec<f32> = Vec::new();
    let mut out = Vec::new();
    let mut take = |pending: &mut Vec<f32>, ends: Vec<usize>, start: usize| {
        if let Some(&last) = ends.last() {
            pending.drain(..(last - start) * 4);
        }
        out.extend(ends);
    };
    for piece in data.chunks(step * 4) {
        pending.extend_from_slice(piece);
        let start = planner.start();
        let ends = planner.ends(&pending, 4, false);
        take(&mut pending, ends, start);
    }
    let start = planner.start();
    let ends = planner.ends(&pending, 4, true);
    take(&mut pending, ends, start);
    out
}

#[test]
fn a_cut_list_is_honoured() {
    let data = rows(1_000, |_| 1.0);
    let mut planner = Planner::new(Some(&[300, 800])).expect("a plan");
    assert_eq!(plan(&mut planner, &data, 250), [300, 800, 1_000]);
    // A cut is a chunk end the moment its row is in, not a push later.
    let mut planner = Planner::new(Some(&[300, 800])).expect("a plan");
    assert_eq!(
        planner.ends(&data[..299 * 4], 4, false),
        Vec::<usize>::new()
    );
    assert_eq!(planner.ends(&data[..300 * 4], 4, false), [300]);
    // A cut at or past the end of the clip is not a chunk of its own.
    let mut planner = Planner::new(Some(&[300, 1_000, 5_000])).expect("a plan");
    assert_eq!(plan(&mut planner, &data, 1_000), [300, 1_000]);
    // Given cuts are the plan even where the silence rule would cut more.
    let long = rows(9_000, |t| if t % 1_000 < 10 { -5.0 } else { 1.0 });
    let mut planner = Planner::new(Some(&[4_000])).expect("a plan");
    assert_eq!(plan(&mut planner, &long, 1_600), [4_000, 9_000]);
    // A plan of no frames, or out of order, is refused.
    assert!(Planner::new(Some(&[0, 10])).is_err());
    assert!(Planner::new(Some(&[10, 10])).is_err());
}

#[test]
fn without_cuts_the_plan_is_the_silence_rule_over_the_whole_clip() {
    // Quiet windows at uneven places, so each cut is found by the search and
    // not at the mark.
    let level = |t: usize| {
        let quiet = [2_730, 5_910, 9_400, 12_020];
        if quiet.iter().any(|&q| t >= q && t < q + 12) {
            -6.0
        } else {
            #[expect(clippy::cast_precision_loss, reason = "a small test pattern")]
            let wobble = (t % 37) as f32 / 37.0;
            wobble
        }
    };
    let data = rows(13_000, level);
    let whole = MelFrames::from_rows(data.clone(), 4).expect("rows");
    let mut want = silence_cuts(&whole, 3_000, 500, 10);
    assert!(want.len() >= 3, "{want:?}");
    want.push(13_000);
    for step in [1, 160, 1_600, 3_509, 13_000] {
        let mut planner = Planner::new(None).expect("a plan");
        assert_eq!(
            plan(&mut planner, &data, step),
            want,
            "pushed {step} at a time"
        );
    }
    // A clip no longer than one chunk is one chunk.
    let mut planner = Planner::new(None).expect("a plan");
    assert_eq!(plan(&mut planner, &rows(3_000, |_| 1.0), 700), [3_000]);
}

fn detokenize(vocab: &Arc<PieceVocab>, events: &[TokenEvent]) -> String {
    let mut detok = Detokenizer::new(Arc::clone(vocab));
    for e in events {
        detok.push(e.token.get());
    }
    detok.flush();
    detok.text().to_owned()
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_QWEN3ASR"]
fn the_whole_stream_matches_one_shot_transcription() {
    use crate::qwen3_asr::config::tests::say;
    use crate::qwen3_asr::decoder::{Decoder, KvCache};
    use crate::qwen3_asr::encoder::Tower;
    use crate::qwen3_asr::{config, prompt};

    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN3ASR") else {
        say("SKIP: SHISO_TEST_MODEL_QWEN3ASR is unset");
        return;
    };
    let Some(backend) = crate::audio_ops::testing::device() else {
        return;
    };
    let path = std::path::Path::new(&root).join("gguf/Qwen3-ASR-1.7B-Q8_0.gguf");
    let asr = crate::qwen3_asr::open(&path, None, &backend)
        .expect("loads")
        .expect("this family");
    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/audio/clips/speech-en.wav");
    let bytes = std::fs::read(&wav).expect("the clip");
    let limits = DecodeLimits::new(std::num::NonZeroU32::new(600).expect("non-zero"));
    let pcm = decode_audio(&bytes, &limits, None).expect("decodes");

    // One shot: the whole clip framed and floored at once, one prompt.
    let m = &asr.model;
    let whole = MelFrontend::new(config::mel_spec())
        .expect("a frontend")
        .compute(pcm.samples());
    let audio = Tower::new(&m.ops, &m.weights.tower, &m.config.audio)
        .encode(&whole)
        .expect("encodes");
    let mut cache = KvCache::new(&m.config.text);
    let ids = Decoder::new(&m.ops, &m.weights.text, &m.config.text)
        .transcribe(&mut cache, &audio, None)
        .expect("decodes");
    let mut raw = Detokenizer::new(Arc::clone(&m.vocab));
    for &id in &ids {
        raw.push(id);
    }
    raw.flush();
    let one_shot = prompt::parse(raw.text(), None);

    // The stream: the recognizer's own frames, pushed as the server pushes.
    let mel = MelFrontend::new(asr.mel_spec().clone())
        .expect("a frontend")
        .compute(pcm.samples());
    let n = mel.n_mels();
    let valid = mel.valid();
    let mut stream = asr
        .open_stream(&StreamOptions {
            language: Language::Auto,
            lookahead: 0,
            cuts: None,
        })
        .expect("opens");
    let mut events = Vec::new();
    for rows in mel.as_slice()[..valid * n].chunks(1_600 * n) {
        let piece = MelFrames::from_rows(rows.to_vec(), n).expect("rows");
        events.extend(stream.push(&piece).expect("pushes"));
    }
    events.extend(stream.finish().expect("finishes"));
    let text = detokenize(&m.vocab, &events);
    say(&format!(
        "{} events over {valid} frames, language {:?}: {text:?}",
        events.len(),
        stream.language()
    ));
    assert!(events.iter().all(|e| !asr.is_special(e.token.get())));
    assert_eq!(text, one_shot.text);
    assert_eq!(stream.language(), Some("en"));
    assert_eq!(stream.frames(), u32::try_from(valid).expect("fits"));
    let frames: Vec<u32> = events.iter().map(|e| e.frame).collect();
    assert!(frames.windows(2).all(|w| w[0] <= w[1]), "{frames:?}");
    assert!(
        frames
            .iter()
            .all(|&f| usize::try_from(f).expect("fits") < valid),
        "{frames:?}"
    );
    assert_eq!(frames.first(), Some(&0));
}
