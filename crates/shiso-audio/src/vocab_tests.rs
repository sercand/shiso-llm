//! Vocabularies from every source, and detokenization.

use std::sync::Arc;

use super::{Detokenizer, PieceVocab};

fn vocab() -> Arc<PieceVocab> {
    let pieces = [
        "<unk>", "<en-US>", "▁", "▁Hello", ",", "▁world", ".", "<", "3", "<de-DE>",
    ];
    Arc::new(
        PieceVocab::from_pieces(pieces.iter().map(|p| (*p).to_owned()).collect()).expect("dense"),
    )
}

#[test]
fn deltas_spaces_and_language_tags() {
    let mut d = Detokenizer::new(vocab());
    let deltas: Vec<Option<String>> = [3u32, 4, 5, 6, 1, 99]
        .iter()
        .map(|id| d.push(*id))
        .collect();
    assert_eq!(
        deltas,
        [
            Some("Hello".into()),
            Some(",".into()),
            Some(" world".into()),
            Some(".".into()),
            None,
            None
        ]
    );
    assert_eq!(d.text(), "Hello, world.");
    assert_eq!(d.language(), Some("en-US"));
}

#[test]
fn a_bare_angle_bracket_is_text() {
    let v = vocab();
    assert_eq!(v.language_of(7), None);
    let mut d = Detokenizer::new(v);
    d.push(7);
    d.push(8);
    assert_eq!(d.text(), "<3");
}

#[test]
fn tokenizer_json_with_added_tokens() {
    let json = r#"{"model":{"type":"BPE","vocab":{"<unk>":0,"▁a":1,"b":2}},
        "added_tokens":[{"id":3,"content":"<pad>"}]}"#;
    let v = PieceVocab::from_tokenizer_json(json).expect("dense");
    assert_eq!(v.len(), 4);
    assert_eq!(v.piece(1), Some("▁a"));
    assert_eq!(v.piece(3), Some("<pad>"));
    let gap = r#"{"model":{"vocab":{"a":0,"b":2}},"added_tokens":[]}"#;
    assert!(PieceVocab::from_tokenizer_json(gap).is_err());
}

#[test]
fn sentencepiece_model_bytes() {
    // ModelProto { pieces: [{piece:"<unk>",score:0,type:2}, {piece:"▁hi",score:-1.5}] , trainer_spec: {...} }
    fn piece(p: &str, with_type: bool) -> Vec<u8> {
        let mut inner = vec![0x0A, u8::try_from(p.len()).expect("short")];
        inner.extend_from_slice(p.as_bytes());
        inner.push(0x15); // field 2, fixed32 score
        inner.extend_from_slice(&(-1.5f32).to_le_bytes());
        if with_type {
            inner.extend_from_slice(&[0x18, 0x02]); // field 3 varint
        }
        let mut out = vec![0x0A, u8::try_from(inner.len()).expect("short")];
        out.extend(inner);
        out
    }
    let mut bytes = piece("<unk>", true);
    bytes.extend(piece("▁hi", false));
    bytes.extend_from_slice(&[0x12, 0x02, 0x08, 0x01]); // field 2 (trainer spec), skipped
    let v = PieceVocab::from_sentencepiece(&bytes).expect("parses");
    assert_eq!((v.len(), v.piece(1)), (2, Some("▁hi")));
    assert!(PieceVocab::from_sentencepiece(&[0x0A, 0x7F]).is_err());
}

#[test]
fn special_pieces_are_not_languages() {
    // `<unk>` and `<pad>` sit in the real checkpoint's vocabulary; if either
    // counted as a tag, a later `<pad>` would overwrite the detected language.
    let pieces = ["<unk>", "<en-US>", "<pad>", "<blank>", "<es>", "▁ok"];
    let v = Arc::new(
        PieceVocab::from_pieces(pieces.iter().map(|p| (*p).to_owned()).collect()).expect("dense"),
    );
    assert_eq!(v.language_of(0), None);
    assert_eq!(v.language_of(2), None);
    assert_eq!(v.language_of(3), None);
    assert_eq!(v.language_of(4), None);
    let mut d = Detokenizer::new(v);
    d.push(1);
    d.push(2);
    assert_eq!(d.language(), Some("en-US"));
}

#[test]
fn a_tag_between_sentences_leaves_one_space() {
    // The model tags every sentence: "me.▁<en-US>▁It". Stripping the tag must
    // not leave the two boundary spaces around it.
    let pieces = ["<unk>", "▁me.", "▁", "<en-US>", "▁It", "▁was."];
    let v = Arc::new(
        PieceVocab::from_pieces(pieces.iter().map(|p| (*p).to_owned()).collect()).expect("dense"),
    );
    let mut d = Detokenizer::new(v);
    for id in [1u32, 2, 3, 4, 5] {
        d.push(id);
    }
    assert_eq!(d.text(), "me. It was.");
    assert_eq!(d.language(), Some("en-US"));
}
