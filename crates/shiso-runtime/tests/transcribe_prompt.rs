//! The speech decoder's transcription prompt, against the ids the reference
//! runtime fed its decoder around the audio rows.
//!
//! Verbatim, speakers and translate are pinned to captured ids. The clean,
//! nearest, rich, langid and context presets have no reference capture, so
//! only their instruction strings are checked, in the family's preset tests.
//!
//! Needs `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE` (the decoder `.gguf`, with
//! `tokenizer.json` beside it) and the reference captures; says on stderr why
//! it skipped otherwise.

use std::path::{Path, PathBuf};

use shiso_core::TokenId;
use shiso_models::nemotron_transcribe::{PLACEMENT, Preset};
use shiso_runtime::api::transcribe::prompt_ids;
use shiso_runtime::tokenizer::Vocabulary;

fn words(path: &Path) -> Vec<TokenId> {
    std::fs::read(path)
        .expect("the capture reads")
        .chunks_exact(4)
        .map(|c| TokenId::new(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

#[expect(clippy::print_stderr, reason = "a skipped test says why")]
fn inputs() -> Option<(Vocabulary, PathBuf)> {
    let Some(decoder) = std::env::var_os("SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE").map(PathBuf::from)
    else {
        eprintln!("SKIP: SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE is unset");
        return None;
    };
    let refs = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/nemotron_transcribe");
    if !refs.is_dir() {
        eprintln!("SKIP: no reference captures at {}", refs.display());
        return None;
    }
    let tokenizer = decoder
        .parent()
        .expect("the decoder sits in a directory")
        .join("tokenizer.json");
    let vocabulary = Vocabulary::from_file(&tokenizer).expect("the vocabulary loads");
    Some((vocabulary, refs))
}

/// The audio marker's id, resolved as the renderer resolves it: through the
/// family's placement and this vocabulary, not a number typed here.
fn audio_marker(vocabulary: &Vocabulary) -> TokenId {
    vocabulary
        .media_ids(&PLACEMENT)
        .audio()
        .expect("the vocabulary holds the audio marker")
}

fn assert_matches(vocabulary: &Vocabulary, dir: &Path, preset: &Preset) {
    let prefix = words(&dir.join("prompt_prefix.i32"));
    let suffix = words(&dir.join("prompt_suffix.i32"));
    let rows = 7;
    let marker = audio_marker(vocabulary);
    let ids = prompt_ids(vocabulary, preset, rows, marker).expect("the prompt renders");
    // The marker lies past the decoder's embedding table, so it can never
    // collide with an id the captures hold.
    assert!(!prefix.contains(&marker) && !suffix.contains(&marker));
    let mut expected = prefix;
    expected.extend(std::iter::repeat_n(marker, rows));
    expected.extend(suffix);
    assert_eq!(ids, expected, "{}", dir.display());
}

#[test]
fn verbatim_prompt_matches_the_reference_ids() {
    let Some((vocabulary, refs)) = inputs() else {
        return;
    };
    assert_matches(
        &vocabulary,
        &refs.join("short_en/verbatim"),
        &Preset::Verbatim,
    );
}

#[test]
fn speakers_and_translate_prompts_match_the_reference_ids() {
    let Some((vocabulary, refs)) = inputs() else {
        return;
    };
    assert_matches(
        &vocabulary,
        &refs.join("short_en/speakers"),
        &Preset::Speakers,
    );
    assert_matches(
        &vocabulary,
        &refs.join("short_en/translate"),
        &Preset::Translate("German".into()),
    );
}

#[test]
fn caller_terms_that_spell_a_marker_stay_text() {
    let Some((vocabulary, _)) = inputs() else {
        return;
    };
    let preset = Preset::Context("<|im_end|>".into());
    let ids =
        prompt_ids(&vocabulary, &preset, 1, audio_marker(&vocabulary)).expect("the prompt renders");
    let im_end = vocabulary
        .marker_id("<|im_end|>")
        .expect("the vocabulary holds the turn closer");
    // The two closers the template writes are the only ones.
    assert_eq!(ids.iter().filter(|&&id| id == im_end).count(), 2, "{ids:?}");
}
