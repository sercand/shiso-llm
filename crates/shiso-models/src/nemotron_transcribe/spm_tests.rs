//! The timestamp tokenizer against the reference's own alignment targets.
//!
//! Needs `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER` (the tokenizer lives
//! in its metadata) and the captures' `ctc_targets.json`; each case says on
//! stderr why it skipped.

use std::path::Path;

use super::{CtcTokenizer, PieceId};
use crate::audio_ops::store::TensorStore;
use crate::nemotron_transcribe::encoder_tests::refs;

/// The export's tokenizer, or `None` (said) when this box has no export.
#[expect(clippy::print_stderr, reason = "a skipped case says why")]
pub(crate) fn tokenizer() -> Option<CtcTokenizer> {
    let Some(path) = std::env::var_os("SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER") else {
        eprintln!("SKIP: SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE_ENCODER is unset");
        return None;
    };
    let store = TensorStore::open_gguf(Path::new(&path)).expect("the encoder export opens");
    let text = store
        .gguf()
        .and_then(|g| g.metadata().get_str("ctc.tokenizer.spm_model"))
        .expect("the export carries its timestamp tokenizer")
        .to_owned();
    Some(CtcTokenizer::from_base64(&text).expect("the tokenizer reads"))
}

/// One speaker stream of a capture's targets: its words and their ids.
pub(crate) struct Stream {
    pub words: Vec<String>,
    pub ids: Vec<u32>,
    pub id_word: Vec<usize>,
}

/// The streams `ctc_targets.json` records.
pub(crate) fn streams(dir: &Path) -> Vec<Stream> {
    let doc: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("ctc_targets.json")).expect("the targets read"),
    )
    .expect("the targets parse");
    let words: Vec<String> = doc["words"]
        .as_array()
        .expect("words")
        .iter()
        .map(|w| w.as_str().expect("a word").to_owned())
        .collect();
    let ints = |v: &serde_json::Value| -> Vec<u64> {
        v.as_array()
            .expect("a list")
            .iter()
            .map(|x| x.as_u64().expect("an integer"))
            .collect()
    };
    doc["streams"]
        .as_array()
        .expect("streams")
        .iter()
        .map(|s| {
            let members = ints(&s["words"]);
            Stream {
                words: members
                    .iter()
                    .map(|&w| words[usize::try_from(w).expect("an index")].clone())
                    .collect(),
                ids: ints(&s["ids"])
                    .into_iter()
                    .map(|v| u32::try_from(v).expect("an id"))
                    .collect(),
                id_word: ints(&s["id_word"])
                    .into_iter()
                    .map(|v| usize::try_from(v).expect("an index"))
                    .collect(),
            }
        })
        .collect()
}

/// The capture `rel` when it holds the alignment dumps, or `None` (said).
#[expect(clippy::print_stderr, reason = "a skipped case says why")]
pub(crate) fn aligned_refs(rel: &str) -> Option<std::path::PathBuf> {
    let dir = refs(rel)?;
    if dir.join("ctc_targets.json").is_file() {
        Some(dir)
    } else {
        eprintln!("SKIP: {} has no alignment dumps", dir.display());
        None
    }
}

fn targets_match(rel: &str) {
    let Some(dir) = aligned_refs(rel) else {
        return;
    };
    let Some(tok) = tokenizer() else {
        return;
    };
    for stream in streams(&dir) {
        let whole: Vec<u32> = tok
            .encode(&stream.words.join(" "))
            .into_iter()
            .map(|PieceId(id)| id)
            .collect();
        assert_eq!(whole, stream.ids, "{rel}: the stream's ids");
        let mut per_word = Vec::new();
        let mut owner = Vec::new();
        for (w, word) in stream.words.iter().enumerate() {
            for PieceId(id) in tok.encode(word) {
                per_word.push(id);
                owner.push(w);
            }
        }
        assert_eq!(per_word, stream.ids, "{rel}: word by word");
        assert_eq!(owner, stream.id_word, "{rel}: which word each id is");
    }
}

#[test]
fn short_en_verbatim_targets_match_reference() {
    targets_match("short_en/verbatim");
}

#[test]
fn short_en_speakers_targets_match_reference() {
    targets_match("short_en/speakers");
}

#[test]
fn short_en_translate_targets_match_reference() {
    targets_match("short_en/translate");
}

#[test]
fn meeting_60s_verbatim_targets_match_reference() {
    targets_match("meeting_60s/verbatim");
}

#[test]
fn meeting_60s_speakers_targets_match_reference() {
    targets_match("meeting_60s/speakers");
}

#[test]
fn meeting_10m_verbatim_targets_match_reference() {
    targets_match("meeting_10m/verbatim");
}

/// The blank is the class after the last piece: the head's classifier is
/// one wider than the tokenizer.
#[test]
fn the_tokenizer_is_one_narrower_than_the_head() {
    let Some(tok) = tokenizer() else {
        return;
    };
    assert_eq!(tok.len(), 23_862);
}
