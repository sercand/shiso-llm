//! The presets against the checkpoint's own card strings.

use super::{Preset, PresetError, parse_preset};

/// The `default_transcription_prompt` of the checkpoint's
/// `processor_config.json`, copied from the file.
const CARD_DEFAULT: &str = "Audio transcription. Language: as spoken. Wording: verbatim, \
                            including fillers and repetitions. Formatting: punctuation and \
                            capitalization. Output: transcript only.";

#[test]
fn verbatim_instruction_is_the_card_default() {
    assert_eq!(Preset::Verbatim.instruction(), CARD_DEFAULT);
    let parsed = parse_preset("verbatim", None, None).expect("the default preset parses");
    assert_eq!(parsed, Preset::Verbatim);
}

#[test]
fn speakers_instruction_is_the_one_the_reference_ran() {
    assert_eq!(
        Preset::Speakers.instruction(),
        "Transcribe this audio with punctuation and capitalization. Include all spoken words \
         and verbal fillers. Write numbers in their standard written form. Put speaker tags \
         <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
    );
}

#[test]
fn translate_names_its_target() {
    let parsed = parse_preset("translate", Some("German"), None).expect("parses");
    assert_eq!(
        parsed.instruction(),
        "Please translate the given speech to German."
    );
}

#[test]
fn a_prompt_becomes_context_terms() {
    let parsed = parse_preset("verbatim", None, Some("Kubernetes, etcd")).expect("parses");
    assert_eq!(
        parsed.instruction(),
        "Transcribe with punctuation and capitalization. Context terms: Kubernetes, etcd."
    );
}

#[test]
fn context_prompt_with_speakers_preset_is_refused() {
    for name in [
        "speakers",
        "nearest",
        "rich",
        "langid",
        "translate",
        "clean",
    ] {
        let refused = parse_preset(name, Some("German"), Some("Kubernetes"));
        assert!(
            matches!(refused, Err(PresetError::PromptNotAccepted { .. })),
            "{name}: {refused:?}"
        );
    }
}

#[test]
fn translate_without_language_is_refused() {
    assert_eq!(
        parse_preset("translate", None, None),
        Err(PresetError::LanguageRequired {
            preset: "translate"
        })
    );
}

#[test]
fn a_language_on_a_preset_that_takes_none_is_refused() {
    assert_eq!(
        parse_preset("verbatim", Some("en"), None),
        Err(PresetError::LanguageNotAccepted { preset: "verbatim" })
    );
}

#[test]
fn langid_reads_its_labels_from_the_language_field() {
    let parsed = parse_preset("langid", Some("en, de ,fr"), None).expect("parses");
    assert_eq!(
        parsed,
        Preset::LangId(vec!["en".into(), "de".into(), "fr".into()])
    );
    assert_eq!(
        parsed.instruction(),
        "Identify the spoken language. Respond with exactly one label from en, de, fr and \
         nothing else."
    );
}

#[test]
fn unknown_preset_is_refused() {
    assert!(matches!(
        parse_preset("karaoke", None, None),
        Err(PresetError::Unknown { .. })
    ));
}

#[test]
fn speakers_keeps_its_tags() {
    assert!(Preset::Speakers.keeps_speakers());
    assert!(!Preset::Verbatim.keeps_speakers());
}
