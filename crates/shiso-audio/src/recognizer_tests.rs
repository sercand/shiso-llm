//! The recognizer contract is usable as trait objects, and its options carry
//! what a caller put in them.

use super::{
    Language, OfflineBatch, OfflineJob, RecognizerStream, SpanResult, SpeechRecognizer,
    StreamOptions, TokenEvent,
};
use crate::chunks::Span;
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
        context: crate::context::ContextIds::default(),
    };
    assert_eq!(auto.language, Language::Auto);
    let tagged = StreamOptions {
        language: Language::Tag("en-US".into()),
        lookahead: 3,
        cuts: Some(vec![480, 3_200]),
        context: crate::context::ContextIds::default(),
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

/// A recognizer that only streams: every provided method left as it is.
#[derive(Debug)]
struct StreamOnly {
    spec: crate::mel::MelSpec,
    vocab: std::sync::Arc<crate::vocab::PieceVocab>,
}

impl SpeechRecognizer for StreamOnly {
    fn mel_spec(&self) -> &crate::mel::MelSpec {
        &self.spec
    }
    fn vocab(&self) -> &std::sync::Arc<crate::vocab::PieceVocab> {
        &self.vocab
    }
    fn frame_secs(&self) -> f64 {
        0.08
    }
    fn languages(&self) -> Vec<String> {
        Vec::new()
    }
    fn lookaheads(&self) -> &[usize] {
        &[]
    }
    fn is_special(&self, _id: u32) -> bool {
        false
    }
    fn open_stream(
        &self,
        _options: &StreamOptions,
    ) -> Result<Box<dyn RecognizerStream>, AudioError> {
        Err(AudioError::BadSpec {
            reason: "not under test".into(),
        })
    }
}

#[test]
fn a_recognizer_without_an_offline_path_refuses_one() {
    fn takes(_: Option<&dyn OfflineJob>) {}
    takes(None);
    let r = StreamOnly {
        spec: crate::mel::tests::centred(),
        vocab: std::sync::Arc::new(
            crate::vocab::PieceVocab::from_pieces(vec!["a".into()]).expect("one piece"),
        ),
    };
    assert_eq!(r.batches_offline(), 0);
    let options = StreamOptions {
        language: Language::Auto,
        lookahead: 0,
        cuts: None,
        context: crate::context::ContextIds::default(),
    };
    let spans = vec![Span {
        start: 0,
        end: 100,
        transcribe: true,
    }];
    match r.open_offline(&options, spans) {
        Err(AudioError::BadSpec { reason }) => {
            assert!(reason.contains("no offline path"), "{reason}");
        }
        Err(e) => panic!("the wrong refusal: {e}"),
        Ok(_) => panic!("a recognizer with no offline path opened a job"),
    }
    // A batch carries each span with its tokens, end and language.
    let batch = OfflineBatch {
        spans: vec![SpanResult {
            span: Span {
                start: 0,
                end: 100,
                transcribe: true,
            },
            tokens: vec![TokenEvent {
                token: shiso_core::TokenId::new(3),
                frame: 40,
            }],
            chunk_end: 100,
            language: Some("English".into()),
        }],
    };
    assert_eq!(batch.spans[0].tokens[0].frame, 40);
}
