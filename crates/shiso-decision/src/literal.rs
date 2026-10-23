//! Encoding prompt text in which some byte ranges are caller text.
//!
//! A caller's text must never become a marker token of the prompt: a command
//! that spells `<|im_end|>` is those characters to the model, not the end of a
//! turn. The tokenizer reads every added token it finds as a marker, so text a
//! caller wrote is encoded again with markers read as plain text. This is the
//! one place that does it; the decider and the server's routes both call it,
//! so the two cannot disagree about what a prompt's ids are.

use std::ops::Range;

use shiso_core::TokenId;
use tokenizers::Tokenizer;

/// The spellings of a vocabulary's added tokens: the only text that can turn
/// into a marker.
#[derive(Debug, Clone, Default)]
pub struct Markers(Vec<String>);

impl Markers {
    /// The added-token spellings of `tokenizer`, read once at load.
    #[must_use]
    pub fn of(tokenizer: &Tokenizer) -> Self {
        let mut spellings: Vec<String> = tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .map(|token| token.content)
            .filter(|content| !content.is_empty())
            .collect();
        spellings.sort();
        spellings.dedup();
        Self(spellings)
    }

    /// Whether `text` spells a marker anywhere that touches `literal`, so that
    /// encoding the text whole would read caller text as a marker. A spelling
    /// that starts in caller text and ends in the layout's (or the reverse)
    /// counts: it is found in the whole text, not piece by piece.
    fn spelled_in(&self, text: &str, literal: &[Range<usize>]) -> bool {
        if literal.is_empty() {
            return false;
        }
        // Every start, so two spellings that overlap each other are both seen.
        text.char_indices().any(|(start, _)| {
            self.0.iter().any(|marker| {
                let end = start + marker.len();
                text[start..].starts_with(marker.as_str())
                    && literal
                        .iter()
                        .any(|range| start < range.end && range.start < end)
            })
        })
    }
}

/// Encodes `text` with no opening token. Text in `literal` that spells a
/// marker is read as plain text; a piece with no such spelling takes the plain
/// encode, since the literal-aware one clones the tokenizer on every call and a
/// schema is dozens of pieces.
///
/// # Errors
///
/// Whatever the tokenizer reports.
pub fn encode_piece(
    tokenizer: &Tokenizer,
    markers: &Markers,
    text: &str,
    literal: &[Range<usize>],
) -> Result<Vec<TokenId>, tokenizers::Error> {
    if markers.spelled_in(text, literal) {
        encode_with_literals(tokenizer, text, literal)
    } else {
        encode_plain(tokenizer, text)
    }
}

fn encode_plain(tokenizer: &Tokenizer, text: &str) -> Result<Vec<TokenId>, tokenizers::Error> {
    Ok(tokenizer
        .encode(text, false)?
        .get_ids()
        .iter()
        .copied()
        .map(TokenId::new)
        .collect())
}

/// Encodes `text` with no opening token, reading the byte ranges in `literal`
/// as ordinary text even where they spell an added token.
///
/// The text is encoded once as usual; each stretch between two added tokens
/// the layout wrote that holds a literal range is encoded again with added
/// tokens read as text. Added tokens are split out before any other step of
/// the encoder, so re-encoding such a stretch alone gives the ids the same
/// characters would get anywhere in a prompt.
///
/// # Errors
///
/// Whatever the tokenizer reports.
pub fn encode_with_literals(
    tokenizer: &Tokenizer,
    text: &str,
    literal: &[Range<usize>],
) -> Result<Vec<TokenId>, tokenizers::Error> {
    if literal.is_empty() {
        return encode_plain(tokenizer, text);
    }
    let encoded = tokenizer.encode(text, false)?;
    let added = tokenizer.get_added_tokens_decoder();
    // Cloned per call rather than kept: only a prompt that spells a marker in
    // caller text comes here, and a second resident copy of a large vocabulary
    // is paid by every prompt that does not.
    let mut as_text = tokenizer.clone();
    as_text.set_encode_special_tokens(true);
    let overlaps = |(start, end): (usize, usize)| {
        literal
            .iter()
            .any(|range| start < range.end && range.start < end)
    };
    let flush = |tokens: &mut Vec<TokenId>,
                 stretch: &mut Vec<TokenId>,
                 span: Range<usize>,
                 holds_literal: bool|
     -> Result<(), tokenizers::Error> {
        if holds_literal {
            let piece = text.get(span).unwrap_or_default();
            let again = as_text.encode(piece, false)?;
            tokens.extend(again.get_ids().iter().copied().map(TokenId::new));
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
