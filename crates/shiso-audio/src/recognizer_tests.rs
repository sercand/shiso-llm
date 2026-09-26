//! The recognizer contract is usable as trait objects, and its options carry
//! what a caller put in them.

use super::{Language, RecognizerStream, SpeechRecognizer, StreamOptions, TokenEvent};
use crate::error::AudioError;

#[test]
fn both_traits_are_object_safe() {
    fn takes(_: Option<&dyn SpeechRecognizer>, _: Option<&dyn RecognizerStream>) {}
    takes(None, None);
}

#[test]
fn options_carry_the_language_and_lookahead() {
    let auto = StreamOptions {
        language: Language::Auto,
        lookahead: 13,
        cuts: None,
    };
    assert_eq!(auto.language, Language::Auto);
    let tagged = StreamOptions {
        language: Language::Tag("en-US".into()),
        lookahead: 3,
        cuts: Some(vec![480, 3_200]),
    };
    assert_eq!(tagged.language, Language::Tag("en-US".into()));
    assert_eq!(tagged.cuts.as_deref(), Some(&[480, 3_200][..]));
    assert_eq!(
        TokenEvent {
            token: shiso_core::TokenId::new(7),
            frame: 2
        },
        TokenEvent {
            token: shiso_core::TokenId::new(7),
            frame: 2
        }
    );
}

#[test]
fn language_and_lookahead_refusals_name_what_was_asked() {
    let e = AudioError::UnknownLanguage {
        tag: "xx-YY".into(),
    };
    assert!(e.to_string().contains("xx-YY"), "{e}");
    let e = AudioError::BadLookahead {
        got: 5,
        allowed: vec![0, 3, 6, 13],
    };
    assert!(
        e.to_string().contains('5') && e.to_string().contains("13"),
        "{e}"
    );
}
