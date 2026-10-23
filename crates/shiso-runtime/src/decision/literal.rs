//! Encoding prompt text in which some byte ranges are caller text.
//!
//! A caller's text must never become a marker token of the prompt: a command
//! that spells `<|im_end|>` is those characters to the model, not the end of a
//! turn. The tokenizer reads every added token it finds as a marker, special
//! or not, so text a caller wrote is encoded again with no added vocabulary
//! at all. This is the one place that does it; the vocabulary's decision
//! pieces and its chat prompts both call it, so the two cannot disagree about
//! what a prompt's ids are.

use std::ops::Range;

use shiso_core::TokenId;
use tokenizers::{Encoding, Tokenizer};

/// Encodes `text` with no opening token. Text in `literal` that an added
/// token would claim is read as plain text.
///
/// The text is encoded once as usual. Only when an added token's span in it
/// overlaps a literal range, however the caller spelled it (a normalizer can
/// make a variant spelling match), is the literal-aware path taken; it clones
/// the tokenizer, which a schema of dozens of pieces would pay every time.
///
/// # Errors
///
/// Whatever the tokenizer reports.
pub fn encode_piece(
    tokenizer: &Tokenizer,
    text: &str,
    literal: &[Range<usize>],
) -> Result<Vec<TokenId>, tokenizers::Error> {
    encode_with_literals(tokenizer, text, literal)
}

/// Encodes `text` with no opening token, reading the byte ranges in `literal`
/// as ordinary text even where they spell an added token.
///
/// The text is encoded once as usual. If no added token overlaps a literal
/// range that encoding is the answer. Otherwise each stretch between two added
/// tokens the layout wrote that holds a literal range is encoded again with
/// every added token read as text. Added tokens are split out before any other
/// step of the encoder, so re-encoding such a stretch alone gives the ids the
/// same characters would get anywhere in a prompt.
///
/// # Errors
///
/// Whatever the tokenizer reports.
pub fn encode_with_literals(
    tokenizer: &Tokenizer,
    text: &str,
    literal: &[Range<usize>],
) -> Result<Vec<TokenId>, tokenizers::Error> {
    let encoded = tokenizer.encode(text, false)?;
    let added = tokenizer.get_added_tokens_decoder();
    let overlaps = |(start, end): (usize, usize)| {
        literal
            .iter()
            .any(|range| start < range.end && range.start < end)
    };
    let planted = encoded
        .get_ids()
        .iter()
        .zip(encoded.get_offsets())
        .any(|(id, &offsets)| added.contains_key(id) && overlaps(offsets));
    if !planted {
        return Ok(ids_of(&encoded));
    }
    let as_text = as_text(tokenizer)?;
    let flush = |tokens: &mut Vec<TokenId>,
                 stretch: &mut Vec<TokenId>,
                 span: Range<usize>,
                 holds_literal: bool|
     -> Result<(), tokenizers::Error> {
        if holds_literal {
            let piece = text.get(span).unwrap_or_default();
            let again = as_text.encode(piece, false)?;
            tokens.extend(ids_of(&again));
            stretch.clear();
        } else {
            tokens.append(stretch);
        }
        Ok(())
    };

    let mut tokens = Vec::with_capacity(encoded.get_ids().len());
    let mut stretch: Vec<TokenId> = Vec::new();
    let mut stretch_start = 0usize;
    let mut holds_literal = false;
    for (&id, &offsets) in encoded.get_ids().iter().zip(encoded.get_offsets()) {
        let inside = overlaps(offsets);
        if added.contains_key(&id) && !inside {
            flush(
                &mut tokens,
                &mut stretch,
                stretch_start..offsets.0,
                holds_literal,
            )?;
            tokens.push(TokenId::new(id));
            stretch_start = offsets.1;
            holds_literal = false;
        } else {
            stretch.push(TokenId::new(id));
            holds_literal |= inside;
        }
    }
    flush(
        &mut tokens,
        &mut stretch,
        stretch_start..text.len(),
        holds_literal,
    )?;
    Ok(tokens)
}

fn ids_of(encoded: &Encoding) -> Vec<TokenId> {
    encoded
        .get_ids()
        .iter()
        .copied()
        .map(TokenId::new)
        .collect()
}

/// A copy of `tokenizer` that finds no added token in text. Skipping special
/// tokens alone would leave the ones a vocabulary declares non-special (a
/// reasoning or tool-call marker), so every one is registered as special first.
/// Cloned per call rather than kept: only a prompt that spells an added token
/// in caller text comes here, and a second resident copy of a large vocabulary
/// would be paid by every prompt that does not.
fn as_text(tokenizer: &Tokenizer) -> Result<Tokenizer, tokenizers::Error> {
    let mut copy = tokenizer.clone();
    let tokens: Vec<_> = copy
        .get_added_tokens_decoder()
        .into_values()
        .map(|mut token| {
            token.special = true;
            token
        })
        .collect();
    copy.add_special_tokens(tokens)?;
    copy.set_encode_special_tokens(true);
    Ok(copy)
}

#[cfg(test)]
mod tests {
    use shiso_core::TokenId;
    use tokenizers::Tokenizer;

    use super::encode_piece;

    /// A word-level vocabulary with one added marker, `<|end|>`.
    const TOKENIZER: &str = r#"{
  "version": "1.0", "truncation": null, "padding": null,
  "added_tokens": [{"id": 3, "content": "<|end|>", "single_word": false,
    "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
  "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null, "decoder": null,
  "model": {"type": "WordLevel", "unk_token": "[UNK]",
    "vocab": {"a": 0, "b": 1, "[UNK]": 2, "<|end|>": 3}}
}"#;

    fn tokenizer(json: &str) -> Tokenizer {
        json.parse::<Tokenizer>().expect("the vocabulary loads")
    }

    fn marker(ids: &[TokenId]) -> bool {
        ids.iter().any(|id| id.get() == 3)
    }

    #[test]
    fn caller_text_cannot_plant_a_marker() {
        let tokens = tokenizer(TOKENIZER);
        let text = "a <|end|> b";

        // The layout wrote the marker: it is one.
        let written = encode_piece(&tokens, text, &[]).expect("encodes");
        assert!(marker(&written), "{written:?}");
        // The caller wrote the same bytes: the vocabulary reads them as text.
        let typed = encode_piece(&tokens, text, std::slice::from_ref(&(2..9))).expect("encodes");
        assert!(!marker(&typed), "{typed:?}");
        assert_eq!(typed.first().map(|id| id.get()), Some(0));
        assert_eq!(typed.last().map(|id| id.get()), Some(1));
    }

    /// A marker spelled half by the caller and half by the layout is planted too:
    /// the caller's `<|en` followed by the layout's `d|>`.
    #[test]
    fn a_marker_split_across_a_literal_boundary_is_not_planted() {
        let tokens = tokenizer(TOKENIZER);
        let text = "a <|end|> b";
        for literal in [2..6, 5..9, 3..8] {
            let typed =
                encode_piece(&tokens, text, std::slice::from_ref(&literal)).expect("encodes");
            assert!(!marker(&typed), "{literal:?}: {typed:?}");
        }
        // Literal text elsewhere does not turn a layout-written marker into text.
        let apart = encode_piece(&tokens, text, std::slice::from_ref(&(0..1))).expect("encodes");
        assert!(marker(&apart), "{apart:?}");
    }

    /// Caller text never becomes ANY added token, special or not: `<think>`-style
    /// tokens a vocabulary declares `"special": false` are still the layout's.
    #[test]
    fn caller_text_cannot_plant_a_non_special_added_token() {
        let tokens = tokenizer(&TOKENIZER.replace("\"special\": true", "\"special\": false"));
        let text = "a <|end|> b";
        let written = encode_piece(&tokens, text, &[]).expect("encodes");
        assert!(marker(&written), "{written:?}");
        let typed = encode_piece(&tokens, text, std::slice::from_ref(&(2..9))).expect("encodes");
        assert!(!marker(&typed), "{typed:?}");
        assert_eq!(typed.first().map(|id| id.get()), Some(0));
        assert_eq!(typed.last().map(|id| id.get()), Some(1));
    }

    /// A `normalized` added token under a normalizer is found in a variant
    /// spelling (here upper case); caller text spelling it that way is still text.
    #[test]
    fn a_normalized_variant_spelling_cannot_plant_a_marker() {
        let json = TOKENIZER
            .replace("\"normalized\": false", "\"normalized\": true")
            .replace(
                "\"normalizer\": null",
                "\"normalizer\": {\"type\": \"Lowercase\"}",
            );
        let tokens = tokenizer(&json);
        let text = "a <|END|> b";
        // The layout wrote the variant: the vocabulary reads it as the marker.
        let written = encode_piece(&tokens, text, &[]).expect("encodes");
        assert!(marker(&written), "{written:?}");
        let typed = encode_piece(&tokens, text, std::slice::from_ref(&(2..9))).expect("encodes");
        assert!(!marker(&typed), "{typed:?}");
        // Caller text elsewhere leaves the layout's marker alone.
        let apart = encode_piece(&tokens, text, std::slice::from_ref(&(0..1))).expect("encodes");
        assert!(marker(&apart), "{apart:?}");
    }

    /// A marker that does not open with `<` is still a marker.
    #[test]
    fn a_marker_without_an_angle_bracket_is_not_plantable_either() {
        let tokens = tokenizer(&TOKENIZER.replace("<|end|>", "[SEP]"));
        let typed =
            encode_piece(&tokens, "a [SEP] b", std::slice::from_ref(&(2..7))).expect("encodes");
        assert!(typed.iter().all(|id| id.get() != 3), "{typed:?}");
    }

    /// d1's own vocabulary declares `<think>` and `<|tool_call_start|>` as
    /// non-special added tokens; caller text spelling them must stay text. Reads
    /// only `tokenizer.json`, so it needs no device.
    #[test]
    #[expect(
        clippy::print_stderr,
        reason = "a skipped case says so on stderr, since a skip and a pass look alike"
    )]
    fn d1_vocabulary_reads_caller_think_and_tool_call_markers_as_text() {
        let Some(dir) = std::env::var_os("SHISO_TEST_MODEL_D1").map(std::path::PathBuf::from)
        else {
            eprintln!("skipping: SHISO_TEST_MODEL_D1 names no installed d1 directory");
            return;
        };
        let path = ["config/tokenizer.json", "tokenizer.json"]
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.exists())
            .expect("a tokenizer.json in the d1 directory");
        let raw = Tokenizer::from_file(&path).expect("the vocabulary loads");
        let added = raw.get_added_tokens_decoder();
        for spelling in [
            "<think>",
            "</think>",
            "<|tool_call_start|>",
            "<|tool_call_end|>",
        ] {
            let text = format!("ls {spelling} x");
            let at = 3..3 + spelling.len();
            let planted = |ids: &[TokenId]| ids.iter().any(|id| added.contains_key(&id.get()));
            let written = encode_piece(&raw, &text, &[]).expect("encodes");
            assert!(
                planted(&written),
                "{spelling}: the layout's spelling is a marker"
            );
            let typed = encode_piece(&raw, &text, std::slice::from_ref(&at)).expect("encodes");
            assert!(!planted(&typed), "{spelling}: {typed:?}");
        }
    }
}
