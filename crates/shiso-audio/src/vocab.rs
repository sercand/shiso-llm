//! Subword vocabularies and incremental detokenization.
//!
//! Speech models emit ids into a piece vocabulary where `▁` marks a word
//! boundary. Three containers carry the pieces: a GGUF string array (the
//! caller hands the strings over), a `tokenizer.json`, and a serialised
//! `SentencePiece` model. The latter is read with a minimal protobuf walk,
//! because only one field of one message matters and a dependency would buy
//! nothing else.
//!
//! Multilingual models in automatic-language mode emit a language tag piece
//! such as `<en-US>` after the text. The tag is not text: [`Detokenizer`]
//! strips it and reports it as the language.

use std::sync::Arc;

use crate::error::AudioError;

const BOUNDARY: char = '\u{2581}';

/// A dense id → piece table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PieceVocab {
    pieces: Vec<String>,
}

fn vocab_err(reason: impl Into<String>) -> AudioError {
    AudioError::Vocab {
        reason: reason.into(),
    }
}

/// `<ll-CC>`: two or three lowercase letters, a hyphen, two uppercase ones.
///
/// The region is required. A bare `<ll>` shape is also what special pieces
/// look like - `<unk>`, `<pad>` - and taking those for tags would report
/// "pad" as the language and overwrite the real one.
fn tag_language(piece: &str) -> Option<&str> {
    let inner = piece.strip_prefix('<')?.strip_suffix('>')?;
    let (lang, region) = inner.split_once('-')?;
    let lang_ok = (2..=3).contains(&lang.len()) && lang.bytes().all(|b| b.is_ascii_lowercase());
    let region_ok = region.len() == 2 && region.bytes().all(|b| b.is_ascii_uppercase());
    (lang_ok && region_ok).then_some(inner)
}

fn varint(b: &[u8], at: &mut usize) -> Result<u64, AudioError> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *b.get(*at).ok_or_else(|| vocab_err("a truncated varint"))?;
        *at += 1;
        v |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err(vocab_err("an overlong varint"))
}

/// Walks one protobuf message, calling `on_len` for each length-delimited
/// field and skipping every other wire type.
fn walk(
    b: &[u8],
    mut on_len: impl FnMut(u64, &[u8]) -> Result<(), AudioError>,
) -> Result<(), AudioError> {
    let mut at = 0;
    while at < b.len() {
        let key = varint(b, &mut at)?;
        let (field, wire) = (key >> 3, key & 7);
        match wire {
            0 => {
                varint(b, &mut at)?;
            }
            1 => at += 8,
            5 => at += 4,
            2 => {
                let len =
                    usize::try_from(varint(b, &mut at)?).map_err(|_| vocab_err("a huge field"))?;
                let body = b
                    .get(at..at + len)
                    .ok_or_else(|| vocab_err("a truncated field"))?;
                on_len(field, body)?;
                at += len;
            }
            _ => return Err(vocab_err(format!("wire type {wire} is not protobuf"))),
        }
        if at > b.len() {
            return Err(vocab_err("a truncated fixed field"));
        }
    }
    Ok(())
}

impl PieceVocab {
    /// A vocabulary whose ids are the positions of `pieces`.
    ///
    /// # Errors
    ///
    /// [`AudioError::Vocab`] for an empty list.
    pub fn from_pieces(pieces: Vec<String>) -> Result<Self, AudioError> {
        if pieces.is_empty() {
            return Err(vocab_err("no pieces"));
        }
        Ok(Self { pieces })
    }

    /// Reads `model.vocab` and `added_tokens` from a `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// [`AudioError::Vocab`] for a malformed document or ids with gaps.
    pub fn from_tokenizer_json(text: &str) -> Result<Self, AudioError> {
        let doc: serde_json::Value =
            serde_json::from_str(text).map_err(|e| vocab_err(e.to_string()))?;
        let mut by_id: Vec<Option<String>> = Vec::new();
        let mut put = |id: u64, piece: &str| -> Result<(), AudioError> {
            let id = usize::try_from(id).map_err(|_| vocab_err("an id past usize"))?;
            if by_id.len() <= id {
                by_id.resize(id + 1, None);
            }
            by_id[id] = Some(piece.to_owned());
            Ok(())
        };
        let vocab = doc
            .pointer("/model/vocab")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| vocab_err("no model.vocab object"))?;
        for (piece, id) in vocab {
            put(
                id.as_u64().ok_or_else(|| vocab_err("a non-integer id"))?,
                piece,
            )?;
        }
        for added in doc
            .get("added_tokens")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
        {
            let id = added
                .get("id")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| vocab_err("an added token without id"))?;
            let content = added
                .get("content")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| vocab_err("an added token without content"))?;
            put(id, content)?;
        }
        let pieces = by_id
            .into_iter()
            .enumerate()
            .map(|(id, p)| p.ok_or_else(|| vocab_err(format!("id {id} has no piece"))))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_pieces(pieces)
    }

    /// Reads the pieces of a serialised `SentencePiece` model.
    ///
    /// # Errors
    ///
    /// [`AudioError::Vocab`] for bytes that are not such a model.
    pub fn from_sentencepiece(bytes: &[u8]) -> Result<Self, AudioError> {
        let mut pieces = Vec::new();
        walk(bytes, |field, body| {
            if field == 1 {
                let mut piece = None;
                walk(body, |f, b| {
                    if f == 1 {
                        piece = Some(
                            String::from_utf8(b.to_vec())
                                .map_err(|_| vocab_err("a piece that is not UTF-8"))?,
                        );
                    }
                    Ok(())
                })?;
                pieces.push(piece.ok_or_else(|| vocab_err("a piece entry without its string"))?);
            }
            Ok(())
        })?;
        Self::from_pieces(pieces)
    }

    /// Pieces in the table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Never true; a vocabulary is refused empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// The piece for `id`, or `None` past the table (blank and pad ids).
    #[must_use]
    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces
            .get(usize::try_from(id).ok()?)
            .map(String::as_str)
    }

    /// The language a tag piece names.
    #[must_use]
    pub fn language_of(&self, id: u32) -> Option<&str> {
        tag_language(self.piece(id)?)
    }
}

/// Ids to text, one delta at a time.
#[derive(Debug)]
pub struct Detokenizer {
    vocab: Arc<PieceVocab>,
    text: String,
    language: Option<String>,
}

impl Detokenizer {
    /// Starts an empty transcript.
    #[must_use]
    pub fn new(vocab: Arc<PieceVocab>) -> Self {
        Self {
            vocab,
            text: String::new(),
            language: None,
        }
    }

    /// Adds one id; returns the text it appended.
    pub fn push(&mut self, id: u32) -> Option<String> {
        if let Some(lang) = self.vocab.language_of(id) {
            self.language = Some(lang.to_owned());
            return None;
        }
        let piece = self.vocab.piece(id)?;
        let mut delta = piece.replace(BOUNDARY, " ");
        // No leading space at the start, and none stacked on a space already
        // written: a stripped tag sits between two word boundaries, and the
        // text should read as if it had never been there.
        if self.text.is_empty() || self.text.ends_with(' ') {
            delta = delta.trim_start().to_owned();
        }
        if delta.is_empty() {
            return None;
        }
        self.text.push_str(&delta);
        Some(delta)
    }

    /// The transcript so far.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The language a tag reported, if any.
    #[must_use]
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }
}

#[cfg(test)]
#[path = "vocab_tests.rs"]
mod tests;
