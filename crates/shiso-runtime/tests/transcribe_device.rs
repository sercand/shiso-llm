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

use shiso_audio::{AudioDecodePolicy, DecodeLimits, Pcm};
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

fn transcribe(rt: &Runtime, pcm: Pcm, preset: Preset) -> TranscribeOutput {
    futures::executor::block_on(rt.transcribe(TranscribeRequest { pcm, preset }))
        .expect("transcribes")
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
    let solo = transcribe(&rt, short.clone(), Preset::Verbatim);
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

    // Review Focus 1: two requests at once share the scheduler and the
    // expert stream, and each answers what it answers alone.
    let meeting = pcm(&clips.join("meeting_60s.wav"));
    let speakers = transcribe(&rt, meeting.clone(), Preset::Speakers);
    eprintln!("meeting_60s speakers: {:?}", speakers.text);
    assert!(
        !speakers.turns.is_empty(),
        "the speakers preset yields speaker turns"
    );
    let (a, b) = futures::executor::block_on(futures::future::join(
        rt.transcribe(TranscribeRequest {
            pcm: short,
            preset: Preset::Verbatim,
        }),
        rt.transcribe(TranscribeRequest {
            pcm: meeting,
            preset: Preset::Speakers,
        }),
    ));
    assert_eq!(a.expect("transcribes"), solo);
    assert_eq!(b.expect("transcribes"), speakers);

    // Silence is answered empty, without encoding.
    let silence = Pcm::new(vec![0.0; 32_000], 16_000).expect("pcm");
    assert_eq!(transcribe(&rt, silence, Preset::Verbatim).text, "");
}
