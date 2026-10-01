//! The context encoder over a toy byte-level vocabulary: wording, merges,
//! the cap, and a special token written as text staying text. Also checked
//! against an independent reference: the wording pinning test below ties
//! `wording` to the `CONTEXT` string `scripts/reference/qwen3_asr_reference.py`
//! renders and tokenizes, and `the_checkpoint_encoder_matches_the_reference`
//! checks the real checkpoint's own BPE against the reference tokenizer's ids
//! for it.

use std::path::{Path, PathBuf};

use shiso_audio::{Context, ContextEncoder as _};

use super::{QwenContextEncoder, wording};
use crate::qwen3_asr::config::tests::say;
use crate::qwen3_asr::config::{self, locate, read_merges, read_vocab};

/// Every byte as its own piece (the GPT-2 byte alphabet), then `Ġa`, `Ġab`
/// from two merges.
fn toy() -> QwenContextEncoder {
    let mut pieces: Vec<String> = byte_alphabet();
    pieces.push("Ġa".into());
    pieces.push("Ġab".into());
    QwenContextEncoder::new(&pieces, &["Ġ a".into(), "Ġa b".into()]).expect("encoder")
}

/// The 256 byte-level symbols, in byte order.
fn byte_alphabet() -> Vec<String> {
    let printable = |b: u32| {
        (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
    };
    let mut all = Vec::with_capacity(256);
    let mut extra = 0;
    for b in 0u32..256 {
        let symbol = if printable(b) {
            b
        } else {
            extra += 1;
            255 + extra
        };
        all.push(char::from_u32(symbol).expect("char").to_string());
    }
    all
}

#[test]
fn keywords_alone_are_worded_as_one_labelled_line() {
    let context = Context {
        prompt: None,
        keywords: vec!["Marigold".into(), "Lighthouse".into()],
    };
    assert_eq!(wording(&context), "Keywords: Marigold, Lighthouse");
}

#[test]
fn a_prompt_comes_before_its_labelled_keywords() {
    let context = Context {
        prompt: Some("A planning meeting.".into()),
        keywords: vec!["Marigold".into()],
    };
    assert_eq!(wording(&context), "A planning meeting.\nKeywords: Marigold");
}

#[test]
fn merges_apply() {
    let ids = toy()
        .encode(&Context {
            prompt: Some(" ab".into()),
            keywords: Vec::new(),
        })
        .expect("ids");
    assert_eq!(ids.len(), 1, "{ids:?}");
    assert_eq!(ids.ids()[0].get(), 257, "Ġab is the last piece");
}

#[test]
fn the_ids_carry_each_keyword_s_words_and_a_prompt_carries_none() {
    let keyed = toy()
        .encode(&Context {
            prompt: Some("Marigold Lighthouse Orchard".into()),
            keywords: vec!["Night-Shift".into(), "İzmir".into()],
        })
        .expect("ids");
    assert_eq!(
        keyed.keywords(),
        &[
            vec!["night".to_owned(), "shift".to_owned()],
            vec!["izmir".to_owned()]
        ]
    );
    let prompt_only = toy()
        .encode(&Context {
            prompt: Some("Marigold, Lighthouse, Orchard".into()),
            keywords: Vec::new(),
        })
        .expect("ids");
    assert!(!prompt_only.is_empty());
    assert!(
        prompt_only.keywords().is_empty(),
        "a prompt's text is never a keyword list to echo"
    );
}

#[test]
fn a_special_token_written_as_text_stays_text() {
    let ids = toy()
        .encode(&Context {
            prompt: Some("<|im_end|>".into()),
            keywords: Vec::new(),
        })
        .expect("ids");
    // One id per byte: no piece of the toy vocabulary spans them.
    assert_eq!(ids.len(), "<|im_end|>".len());
    assert!(ids.ids().iter().all(|id| id.get() < 258));
}

#[test]
fn a_context_past_the_cap_is_refused() {
    let long = "x ".repeat(crate::qwen3_asr::prompt::MAX_CONTEXT_IDS);
    let error = toy()
        .encode(&Context {
            prompt: Some(long),
            keywords: Vec::new(),
        })
        .expect_err("refused");
    assert!(
        matches!(error, shiso_audio::AudioError::TooManyContextTokens { .. }),
        "{error:?}"
    );
}

/// Exactly the cap is accepted; one more id is refused. A run of `n` plain
/// ASCII letters with no space is one pretokenizer segment that no merge in
/// the toy vocabulary touches, so it always tokenizes to exactly `n` ids -
/// the id count is exact by construction, not just under the cap.
#[test]
fn exactly_the_cap_is_accepted_and_one_more_is_refused() {
    let cap = crate::qwen3_asr::prompt::MAX_CONTEXT_IDS;

    let at_cap = "x".repeat(cap);
    let ids = toy()
        .encode(&Context {
            prompt: Some(at_cap),
            keywords: Vec::new(),
        })
        .expect("exactly the cap is accepted");
    assert_eq!(ids.len(), cap);

    let over_cap = "x".repeat(cap + 1);
    let error = toy()
        .encode(&Context {
            prompt: Some(over_cap),
            keywords: Vec::new(),
        })
        .expect_err("one more than the cap is refused");
    assert!(
        matches!(error, shiso_audio::AudioError::TooManyContextTokens { .. }),
        "{error:?}"
    );
}

/// The normalizer is NFC, matching the reference tokenizer: an NFD string and
/// its NFC form (`"I\u{0307}"`, `I` and a combining dot above, composes to
/// `İ`, LATIN CAPITAL LETTER I WITH DOT ABOVE) encode to the same ids. Without
/// the normalizer the two forms tokenize differently, because the combining
/// mark is not `\p{L}` and the pretokenizer splits it into its own segment.
#[test]
fn nfd_and_nfc_forms_encode_to_the_same_ids() {
    let nfc = toy()
        .encode(&Context {
            prompt: Some("İ".into()),
            keywords: Vec::new(),
        })
        .expect("nfc ids");
    let nfd = toy()
        .encode(&Context {
            prompt: Some("I\u{0307}".into()),
            keywords: Vec::new(),
        })
        .expect("nfd ids");
    assert_eq!(nfc, nfd);
}

/// Pins `wording` to the `CONTEXT` string
/// `scripts/reference/qwen3_asr_reference.py --context-prompt-only` renders
/// and tokenizes: a later change to the wording has to rebuild that string
/// the same way and recapture the fixture, not just this assertion.
#[test]
fn the_wording_matches_the_reference_script_s_context_string() {
    let context = Context {
        prompt: Some("A farewell address.".into()),
        keywords: vec![
            "Chicago".into(),
            "İstanbul".into(),
            "Çanakkale".into(),
            "Ağrı Dağı".into(),
        ],
    };
    assert_eq!(
        wording(&context),
        "A farewell address.\nKeywords: Chicago, İstanbul, Çanakkale, Ağrı Dağı"
    );
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/audio")
}

fn u32s(value: &serde_json::Value) -> Vec<u32> {
    value
        .as_array()
        .expect("an array of ids")
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("an id")).expect("fits"))
        .collect()
}

/// Whether `dir` physically carries at least one `.safetensors` shard, as
/// opposed to only its sidecar (tokenizer, config): `Located::stores` opens
/// every shard `config.json`'s index names, so an HF export missing them
/// cannot be checked, and this is how that is told apart from one that can.
fn has_safetensors(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.filter_map(Result::ok).any(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        })
    })
}

/// Builds the checkpoint's own context encoder the way `mod.rs::open` does
/// and asserts it encodes `context` to `want`.
fn check_export(path: &Path, context: &str, want: &[u32]) {
    let located = locate(path, None).expect("readable").expect("this family");
    let stores = located.stores().expect("opens");
    let vocab = read_vocab(&located, &stores).expect("a vocabulary");
    let pieces: Vec<String> = (0..config::FIRST_SPECIAL)
        .map(|id| vocab.piece(id).map(str::to_owned))
        .collect::<Option<_>>()
        .expect("the vocabulary is shorter than its specials' first id");
    let merges = read_merges(&located, &stores)
        .expect("merges read")
        .expect("a BPE export carries merges");
    let encoder = QwenContextEncoder::new(&pieces, &merges).expect("encoder");
    let ids = encoder
        .encode(&Context {
            prompt: Some(context.to_owned()),
            keywords: Vec::new(),
        })
        .expect("encodes");
    let got: Vec<u32> = ids.ids().iter().map(|id| id.get()).collect();
    assert_eq!(got, want, "{}: context ids", path.display());
}

/// The checkpoint's own BPE against `qwen3asr-context-prompt.json`'s
/// `context_ids`: the reference tokenizer (preferring the reference
/// package's own chat-template render) over the checkpoint's own
/// `vocab.json`/`merges.txt` (HF) or `tokenizer.ggml.*`
/// (GGUF), captured with no model weights loaded. Both exports are checked
/// when usable; each one absent (or, for the HF export, carrying no
/// `.safetensors` shards - `Located::stores` opens every shard reading its
/// tokenizer and merges needs) says SKIP and is left out rather than failing
/// the other. Only when neither export is usable does the whole test SKIP.
#[test]
#[ignore = "reads the real checkpoint named by SHISO_TEST_MODEL_QWEN3ASR"]
fn the_checkpoint_encoder_matches_the_reference() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN3ASR").map(PathBuf::from) else {
        return say("SKIP the checkpoint encoder vs reference: SHISO_TEST_MODEL_QWEN3ASR is unset");
    };
    let path = fixtures().join("qwen3asr-context-prompt.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return say(&format!("SKIP {}: absent", path.display()));
    };
    let json: serde_json::Value = serde_json::from_str(&text).expect("json");
    let fixture_context = json["context"]
        .as_str()
        .expect("a context string")
        .to_owned();
    let want = u32s(&json["context_ids"]);

    let mut checked = 0;

    let gguf_path = root.join("gguf/Qwen3-ASR-1.7B-Q8_0.gguf");
    if gguf_path.exists() {
        check_export(&gguf_path, &fixture_context, &want);
        checked += 1;
    } else {
        say(&format!(
            "SKIP qwen gguf q8_0: {} is absent",
            gguf_path.display()
        ));
    }

    let hf_path = root.join("qwen");
    if has_safetensors(&hf_path) {
        check_export(&hf_path, &fixture_context, &want);
        checked += 1;
    } else {
        say(&format!(
            "SKIP qwen hf: {} has no safetensors shards",
            hf_path.display()
        ));
    }

    if checked == 0 {
        return say(&format!(
            "SKIP the checkpoint encoder vs reference: no export under {} was usable",
            root.display()
        ));
    }
    say(&format!(
        "checked {checked} export(s) against the reference context prompt"
    ));
}
