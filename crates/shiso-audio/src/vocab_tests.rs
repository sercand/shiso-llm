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

#[test]
fn byte_level_symbols_decode_to_their_bytes() {
    // "Ġhello" is the GPT-2 byte-level spelling of " hello" (0x20 maps to
    // U+0120), and "Ċ" is a lone newline (0x0A maps to U+010A).
    let tokens = ["Ġhello".to_owned(), "Ċ".to_owned()];
    let v = Arc::new(PieceVocab::from_byte_level(tokens.to_vec(), &[]).expect("valid symbols"));
    assert!(v.is_byte_level());
    let mut d = Detokenizer::new(v);
    assert_eq!(d.push(0), Some(" hello".to_owned()));
    assert_eq!(d.push(1), Some("\n".to_owned()));
    assert_eq!(d.text(), " hello\n");
}

#[test]
fn a_character_split_across_tokens_is_emitted_once_whole() {
    // "ğ" is the two UTF-8 bytes 0xC4 0x9F; the GPT-2 byte alphabet spells
    // them as one symbol character each ("Ä", "Ł"). "中" is the three bytes
    // 0xE4 0xB8 0xAD; split 1+2 the symbols are "ä" and "¸Ń" (one token
    // spelling two bytes). Neither character is completed until its last
    // byte-level token arrives.
    let tokens = [
        "Ä".to_owned(),
        "Ł".to_owned(),
        "ä".to_owned(),
        "¸Ń".to_owned(),
    ];
    let v = Arc::new(PieceVocab::from_byte_level(tokens.to_vec(), &[]).expect("valid symbols"));
    let mut d = Detokenizer::new(v);
    let first = d.push(0);
    assert!(first.is_none() || first == Some(String::new()));
    assert_eq!(d.push(1), Some("ğ".to_owned()));
    let third = d.push(2);
    assert!(third.is_none() || third == Some(String::new()));
    assert_eq!(d.push(3), Some("中".to_owned()));
    assert_eq!(d.text(), "ğ中");
    assert!(!d.text().contains('\u{FFFD}'));
}

#[test]
fn specials_decode_as_their_text() {
    // `<asr_text>` is an added token, not a byte-level symbol: its content is
    // the literal text, never run through the byte table.
    let tokens = ["Ġhi".to_owned()];
    let specials = [(1u32, "<asr_text>".to_owned())];
    let v =
        Arc::new(PieceVocab::from_byte_level(tokens.to_vec(), &specials).expect("valid specials"));
    let mut d = Detokenizer::new(v);
    assert_eq!(d.push(0), Some(" hi".to_owned()));
    assert_eq!(d.push(1), Some("<asr_text>".to_owned()));
    assert_eq!(d.text(), " hi<asr_text>");
}

#[expect(clippy::print_stderr, reason = "a skipped case says why on stderr")]
fn say(line: &str) {
    eprintln!("{line}");
}

#[test]
#[ignore = "needs SHISO_TEST_MODEL_QWEN3ASR and the qwen3asr tokens reference fixture"]
fn the_real_vocab_round_trips_a_sentence() {
    let Ok(root) = std::env::var("SHISO_TEST_MODEL_QWEN3ASR") else {
        say("SKIP the_real_vocab_round_trips_a_sentence: SHISO_TEST_MODEL_QWEN3ASR is not set");
        return;
    };
    let root = std::path::Path::new(&root).join("qwen");
    let vocab_path = root.join("vocab.json");
    let config_path = root.join("tokenizer_config.json");
    let Ok(vocab_text) = std::fs::read_to_string(&vocab_path) else {
        say(&format!(
            "SKIP the_real_vocab_round_trips_a_sentence: {} is absent",
            vocab_path.display()
        ));
        return;
    };
    let config_text = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
        panic!(
            "{} reads once vocab.json exists: {e}",
            config_path.display()
        )
    });

    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/audio/qwen3asr-speech-en-tokens.json");
    let Ok(fixture_text) = std::fs::read_to_string(&fixture_path) else {
        say(&format!(
            "SKIP the_real_vocab_round_trips_a_sentence: {} is absent; capture it first",
            fixture_path.display()
        ));
        return;
    };

    let vocab_json: serde_json::Value =
        serde_json::from_str(&vocab_text).expect("vocab.json parses");
    let vocab_obj = vocab_json.as_object().expect("a token->id object");
    let mut tokens: Vec<Option<String>> = vec![None; vocab_obj.len()];
    for (piece, id) in vocab_obj {
        let id = usize::try_from(id.as_u64().expect("an integer id")).expect("fits");
        tokens[id] = Some(piece.clone());
    }
    let tokens: Vec<String> = tokens
        .into_iter()
        .enumerate()
        .map(|(id, p)| p.unwrap_or_else(|| panic!("vocab.json id {id} has no piece")))
        .collect();

    let config_json: serde_json::Value =
        serde_json::from_str(&config_text).expect("tokenizer_config.json parses");
    let added = config_json
        .get("added_tokens_decoder")
        .and_then(serde_json::Value::as_object)
        .expect("added_tokens_decoder");
    let mut specials: Vec<(u32, String)> = Vec::new();
    let mut skip_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for (id, entry) in added {
        let id: u32 = id.parse().expect("a numeric added-token id");
        let content = entry
            .get("content")
            .and_then(serde_json::Value::as_str)
            .expect("an added token's content");
        specials.push((id, content.to_owned()));
        // A recognizer stream never pushes a stop/control id to the
        // detokenizer, the same way it never pushes a blank id
        // (`SpeechRecognizer::is_special`): `skip_special_tokens=True` on the
        // reference decode dropped exactly these.
        if entry.get("special").and_then(serde_json::Value::as_bool) == Some(true) {
            skip_ids.insert(id);
        }
    }

    let vocab = Arc::new(
        PieceVocab::from_byte_level(tokens, &specials).expect("the real vocab is well-formed"),
    );
    let mut d = Detokenizer::new(vocab);

    let fixture: serde_json::Value =
        serde_json::from_str(&fixture_text).expect("the fixture is JSON");
    let ids = fixture["auto"]["generated_ids"]
        .as_array()
        .expect("generated_ids array");
    let expected = fixture["auto"]["decoded_text"]
        .as_str()
        .expect("decoded_text string");

    for id in ids {
        let id = u32::try_from(id.as_u64().expect("an integer id")).expect("fits");
        if skip_ids.contains(&id) {
            continue;
        }
        d.push(id);
    }
    assert_eq!(d.text(), expected);
    say("the_real_vocab_round_trips_a_sentence: ran against the real checkpoint (not skipped)");
}
