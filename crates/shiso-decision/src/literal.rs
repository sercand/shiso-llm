//! Encoding prompt text in which some byte ranges are caller text.
//!
//! A caller's text must never become a marker token of the prompt: a command
//! that spells `<|im_end|>` is those characters to the model, not the end of a
//! turn. The tokenizer reads every added token it finds as a marker, special
//! or not, so text a caller wrote is encoded again with no added vocabulary
//! at all. This is the one place that does it; the decider and the server's
//! routes both call it, so the two cannot disagree about what a prompt's ids
//! are.
//!
//! A seam for the server and the decider, not part of the embedding API:
//! embedders use [`crate::Decider`].

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
