//! The context encoder over a toy byte-level vocabulary: wording, merges,
//! the cap, and a special token written as text staying text.

use shiso_audio::{Context, ContextEncoder as _};

use super::{QwenContextEncoder, wording};

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
fn keywords_alone_are_worded_as_one_line() {
    let context = Context {
        prompt: None,
        keywords: vec!["Unity".into(), "SQLite".into()],
    };
    assert_eq!(wording(&context), "Unity, SQLite");
}

#[test]
fn a_prompt_comes_before_its_keywords() {
    let context = Context {
        prompt: Some("A planning meeting.".into()),
        keywords: vec!["Unity".into()],
    };
    assert_eq!(wording(&context), "A planning meeting.\nUnity");
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
        matches!(error, shiso_audio::AudioError::BadContext { .. }),
        "{error:?}"
    );
}
