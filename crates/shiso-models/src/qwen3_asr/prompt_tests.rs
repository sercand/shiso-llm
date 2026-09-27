//! The prompt against the reference processor's own ids, and the output
//! parser and repetition collapse against the reference's rules (notes §3).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_audio::Detokenizer;

use super::{Parsed, collapse_repeats, forced_suffix, ids, parse};
use crate::qwen3_asr::config::tests::{exports, say};
use crate::qwen3_asr::config::{AUDIO_PAD, languages, locate, read_vocab};

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

fn parsed(language: Option<&str>, text: &str) -> Parsed {
    Parsed {
        language: language.map(str::to_owned),
        text: text.to_owned(),
    }
}

#[test]
fn the_prompt_ids_match_the_reference() {
    let mut seen = 0;
    for model in ["qwen3asr", "r2t2"] {
        for clip in ["speech-en", "speech-en-odd", "weather-es", "two-speakers"] {
            let path = fixtures().join(format!("{model}-{clip}-prompt.json"));
            let Ok(text) = std::fs::read_to_string(&path) else {
                say(&format!("SKIP {}: absent", path.display()));
                continue;
            };
            let json: serde_json::Value = serde_json::from_str(&text).expect("json");
            let n_audio = usize::try_from(json["auto"]["n_audio_pad"].as_u64().expect("count"))
                .expect("fits");
            assert_eq!(
                ids(n_audio, None).expect("the auto prompt"),
                u32s(&json["auto"]["ids"]),
                "{model} {clip}: auto"
            );
            let language = json["forced"]["language"].as_str().expect("a name");
            assert_eq!(
                ids(n_audio, Some(language)).expect("a supported language"),
                u32s(&json["forced"]["ids"]),
                "{model} {clip}: forced {language}"
            );
            seen += 1;
        }
    }
    say(&format!("compared {seen} prompt fixtures"));
}

#[test]
fn the_prompt_is_fifteen_ids_around_the_audio() {
    // Notes §3: 15 ids plus one placeholder per audio row, the placeholders
    // one unbroken run.
    let p = ids(390, None).expect("builds");
    assert_eq!(p.len(), 405);
    let first = p
        .iter()
        .position(|&t| t == AUDIO_PAD)
        .expect("a placeholder");
    assert_eq!(first, 9);
    assert!(p[first..first + 390].iter().all(|&t| t == AUDIO_PAD));
    assert_eq!(p.iter().filter(|&&t| t == AUDIO_PAD).count(), 390);
    assert_eq!(ids(390, Some("english")).expect("any case").len(), 408);
}

#[test]
fn an_unsupported_forced_language_is_refused() {
    assert!(ids(13, Some("Klingon")).is_err());
    assert!(ids(13, Some("")).is_err());
}

#[test]
fn every_supported_language_can_be_forced() {
    for (_, name) in languages() {
        let suffix = forced_suffix(name).expect("every supported language has a suffix");
        assert!(suffix.len() >= 3, "{name}");
    }
}

#[test]
#[ignore = "reads the vocabulary of the checkpoints named by SHISO_TEST_MODEL_QWEN3ASR and SHISO_TEST_MODEL_R2T2"]
fn every_forced_suffix_spells_its_language() {
    let mut checked = 0;
    for export in exports() {
        if !export.path.exists() {
            say(&format!("SKIP {}: absent", export.name));
            continue;
        }
        let located = locate(&export.path, None)
            .expect("readable")
            .expect("this family");
        let stores = located.stores().expect("opens");
        let vocab = Arc::new(read_vocab(&located, &stores).expect("a vocabulary"));
        for (_, name) in languages() {
            let mut detok = Detokenizer::new(vocab.clone());
            for &id in &forced_suffix(name).expect("supported") {
                detok.push(id);
            }
            assert_eq!(
                detok.text(),
                format!("language {name}<asr_text>"),
                "{}: {name}",
                export.name
            );
        }
        checked += 1;
    }
    say(&format!("checked {checked} vocabularies"));
}

#[test]
fn parsing_splits_language_and_text() {
    assert_eq!(
        parse("language English<asr_text>Hello there.", None),
        parsed(Some("English"), "Hello there.")
    );
    // Stripped on both sides, and the language normalised to its canonical
    // capitalisation.
    assert_eq!(
        parse("  language gERMAN<asr_text> Guten Tag. \n", None),
        parsed(Some("German"), "Guten Tag.")
    );
    // Only the first tag splits.
    assert_eq!(
        parse("language English<asr_text>a<asr_text>b", None),
        parsed(Some("English"), "a<asr_text>b")
    );
    // The language is the first meta line that names one.
    assert_eq!(
        parse("noise\nlanguage French\n<asr_text>Bonjour", None),
        parsed(Some("French"), "Bonjour")
    );
    // No tag: the whole string is the text.
    assert_eq!(parse("just words", None), parsed(None, "just words"));
    // A forced language: the whole output is the text, even with a tag in it.
    assert_eq!(
        parse(" Hola<asr_text>x ", Some("Spanish")),
        parsed(Some("Spanish"), "Hola<asr_text>x")
    );
    assert_eq!(parse("   ", None), parsed(None, ""));
}

#[test]
fn language_none_is_silence() {
    assert_eq!(parse("language None<asr_text>", None), parsed(None, ""));
    assert_eq!(parse("language none<asr_text>  ", None), parsed(None, ""));
    // The reference keeps text the model wrote anyway, with no language.
    assert_eq!(
        parse("language None<asr_text> hm", None),
        parsed(None, "hm")
    );
}

#[test]
fn repeats_collapse_at_twenty() {
    // A run of one character longer than the threshold is one character; at
    // the threshold it stays.
    assert_eq!(
        collapse_repeats(&format!("a{}b", "x".repeat(21)), 20),
        "axb"
    );
    let twenty = format!("a{}b", "x".repeat(20));
    assert_eq!(collapse_repeats(&twenty, 20), twenty);
    // A pattern of up to twenty characters repeated twenty times or more is
    // one copy, and what follows it is collapsed in turn.
    let text = format!("start {}end{}", "ha ".repeat(25), "yo".repeat(30));
    assert_eq!(collapse_repeats(&text, 20), "start ha endyo");
    // Nineteen repeats stay.
    let nineteen = format!("{}!", "ab".repeat(19));
    assert_eq!(collapse_repeats(&nineteen, 20), nineteen);
    // Characters, not bytes: a multi-byte run collapses the same way.
    assert_eq!(collapse_repeats(&"ğ".repeat(25), 20), "ğ");
    assert_eq!(collapse_repeats(&"好的".repeat(20), 20), "好的");
    // Short strings are left alone.
    assert_eq!(collapse_repeats("abc", 20), "abc");
    // The collapse runs inside the parser, after the strip and before the
    // split: the stripped run is 39 copies of "no " and a bare "no", which
    // the reference leaves as one copy and the remainder.
    assert_eq!(
        parse(
            &format!("language English<asr_text>{}", "no ".repeat(40)),
            None
        ),
        parsed(Some("English"), "no no")
    );
}
