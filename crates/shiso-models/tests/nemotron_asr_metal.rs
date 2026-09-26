//! The whole recognizer path through the public API, for every export:
//! WAV bytes -> this tree's decoder and mel frontend -> the registry's
//! recognizer -> tokens, compared with the reference's.
//!
//! Needs `SHISO_TEST_MODEL_ASR` and a Metal device. The f32 export must match
//! the reference token for token even through this tree's own frontend; the
//! quantised exports are held to a measured edit distance.
#![cfg(target_os = "macos")]

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_audio::{DecodeLimits, Language, MelFrontend, MelStream, StreamOptions, decode_audio};
use shiso_core::Backend;
use shiso_models::audio_registry::open_speech_recognizer;

#[expect(
    clippy::print_stdout,
    reason = "a skipped case says why; the distances and speeds are the report"
)]
fn say(line: &str) {
    println!("{line}");
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/audio")
}

fn reference_tokens(clip: &str) -> Vec<u32> {
    let text = std::fs::read_to_string(fixtures().join(format!("asr-{clip}-r13-auto.json")))
        .expect("the fixture exists");
    let doc: serde_json::Value = serde_json::from_str(&text).expect("json");
    doc["tokens"]
        .as_array()
        .expect("tokens")
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("id")).expect("fits"))
        .collect()
}

/// Levenshtein distance over token ids.
fn edit_distance(a: &[u32], b: &[u32]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(x != y))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Token edit distance to the reference, per export. Measured 2026-09-25 on
/// M1 Max through this tree's own decoder and frontend, streaming 100 ms
/// pieces: 0 for the f32, 8-bit and `Q8_0` exports on both clips; 3 of 210 for
/// the 4-bit `Q4_K_M` export on speech-en, 0 on weather-es. The f32 export is held
/// to exactness; the quantised ones to the measured distance plus two.
const BUDGETS: [(&str, usize); 4] = [("hf", 0), ("mlx", 2), ("asr-q8", 2), ("parakeet-q4k", 5)];

#[test]
#[ignore = "requires a Metal device"]
fn every_export_transcribes_through_the_public_path() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_ASR").map(PathBuf::from) else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    let exports = [
        ("hf", root.join("nvidia")),
        ("mlx", root.join("mlx")),
        (
            "asr-q8",
            root.join("nvidia/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"),
        ),
        (
            "parakeet-q4k",
            root.join("gguf/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"),
        ),
    ];
    let metal = shiso_metal::MetalBackend::new().expect("the Metal library compiles");
    let backend: Arc<dyn Backend> = metal;
    let limits = DecodeLimits::new(NonZeroU32::new(600).expect("non-zero"));
    for (name, path) in exports {
        if !path.exists() {
            say(&format!("SKIP {name}: absent"));
            continue;
        }
        let recognizer = open_speech_recognizer(&path, &backend).expect("the registry opens it");
        let budget = BUDGETS
            .iter()
            .find(|(n, _)| *n == name)
            .map_or(0, |(_, b)| *b);
        for clip in ["speech-en", "weather-es"] {
            let bytes = std::fs::read(fixtures().join(format!("clips/{clip}.wav")))
                .expect("the clip exists");
            let pcm = decode_audio(&bytes, &limits, None).expect("decodes");
            let frontend = Arc::new(
                MelFrontend::new(recognizer.mel_spec().clone()).expect("the spec is valid"),
            );
            let start = Instant::now();
            let mut stream = recognizer
                .open_stream(&StreamOptions {
                    language: Language::Auto,
                    lookahead: 13,
                })
                .expect("a stream opens");
            let mut mel = MelStream::new(Arc::clone(&frontend)).expect("streams");
            let mut tokens = Vec::new();
            // Feed the audio in 100 ms pieces, as a live client would.
            for piece in pcm.samples().chunks(1600) {
                let frames = mel.push(piece);
                tokens.extend(
                    stream
                        .push(&frames)
                        .expect("pushes")
                        .into_iter()
                        .map(|e| e.token.get()),
                );
            }
            let frames = mel.finish();
            tokens.extend(
                stream
                    .push(&frames)
                    .expect("pushes the tail")
                    .into_iter()
                    .map(|e| e.token.get()),
            );
            tokens.extend(
                stream
                    .finish()
                    .expect("finishes")
                    .into_iter()
                    .map(|e| e.token.get()),
            );
            let wall = start.elapsed().as_secs_f64();
            let distance = edit_distance(&tokens, &reference_tokens(clip));
            say(&format!(
                "{name} {clip}: {} tokens, edit distance {distance}, {:.1}x real time",
                tokens.len(),
                pcm.duration_secs() / wall
            ));
            assert!(
                distance <= budget,
                "{name} {clip}: edit distance {distance} over {budget}"
            );
        }
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn the_first_chunk_leaves_as_soon_as_its_mel_rows_arrive() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_ASR").map(PathBuf::from) else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    let metal = shiso_metal::MetalBackend::new().expect("the Metal library compiles");
    let backend: Arc<dyn Backend> = metal;
    let recognizer = open_speech_recognizer(&root.join("nvidia"), &backend).expect("opens");
    let frontend = MelFrontend::new(recognizer.mel_spec().clone()).expect("valid");
    for lookahead in [13usize, 3] {
        // The first chunk needs 1 + 8R mel rows; it must not wait for more.
        let rows = 1 + 8 * lookahead;
        let samples: Vec<f32> = (0..rows * 160)
            .map(|i| (f32::from(u8::try_from(i % 97).expect("small")) / 97.0 - 0.5) * 0.1)
            .collect();
        let mel = frontend.compute(&samples);
        assert_eq!(mel.valid(), rows);
        let mut stream = recognizer
            .open_stream(&StreamOptions {
                language: Language::Auto,
                lookahead,
            })
            .expect("opens");
        stream.push(&mel).expect("pushes");
        assert_eq!(
            stream.frames(),
            u32::try_from(lookahead + 1).expect("fits"),
            "r{lookahead}"
        );
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn a_broken_export_of_this_family_reports_its_own_defect() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_ASR").map(PathBuf::from) else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    let hf = root.join("nvidia");
    let dir = tempfile::tempdir().expect("a temp dir");
    for name in ["config.json", "tokenizer.json", "model.safetensors"] {
        std::os::unix::fs::symlink(hf.join(name), dir.path().join(name)).expect("links");
    }
    std::fs::write(
        dir.path().join("processor_config.json"),
        r#"{"prompt_dictionary": {"en-US": "zero"}}"#,
    )
    .expect("writes");
    let metal = shiso_metal::MetalBackend::new().expect("the Metal library compiles");
    let backend: Arc<dyn Backend> = metal;
    let err = open_speech_recognizer(dir.path(), &backend).expect_err("refused");
    let text = err.to_string();
    assert!(!text.contains("no speech recognizer family"), "{text}");
    assert!(text.contains("prompt"), "{text}");
}
