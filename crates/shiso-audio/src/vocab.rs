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
//!
//! A second family of models tokenizes with GPT-2 byte-level BPE instead:
//! there is no `▁`, and a piece is a run of symbols from a 256-character byte
//! alphabet rather than text itself. [`PieceVocab::from_byte_level`] builds
//! this mode, and [`Detokenizer::push`] maps each piece's symbols back to raw
//! bytes and buffers them until they complete a UTF-8 character - a
//! multi-byte character can end up split across two token ids, and must never
//! be reported as text (or as U+FFFD) until it is whole.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use crate::error::AudioError;

const BOUNDARY: char = '\u{2581}';

/// Whether a byte is one of GPT-2's 188 "printable" bytes, which the
/// byte-to-unicode bijection maps to themselves (their own code point).
/// Every other byte maps to U+0100 onward, in ascending byte order.
fn is_gpt2_printable_byte(b: u8) -> bool {
    matches!(b, 33..=126 | 161..=172 | 174..=255)
}

/// The inverse of GPT-2's byte-level bijection: symbol character → raw byte.
///
/// Built once and shared, because every byte-level push looks a handful of
/// characters up in it.
static BYTE_LEVEL_TABLE: LazyLock<HashMap<char, u8>> = LazyLock::new(|| {
    let mut table = HashMap::with_capacity(256);
    for b in 0u8..=255 {
        if is_gpt2_printable_byte(b) {
            let ch = char::from_u32(u32::from(b)).expect("every printable byte is a scalar value");
            table.insert(ch, b);
        }
    }
    let mut next = 0u32;
    for b in 0u8..=255 {
        if !is_gpt2_printable_byte(b) {
            let ch = char::from_u32(256 + next).expect("U+0100.. is always a scalar value");
            table.insert(ch, b);
            next += 1;
        }
    }
    table
});

/// How a vocabulary's pieces turn into text.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    /// `SentencePiece`: `▁` marks a word boundary; pieces are already text.
    Piece,
    /// GPT-2 byte-level BPE: pieces are symbols in the byte alphabet, decoded
    /// through [`BYTE_LEVEL_TABLE`], except the ids in this set, whose piece
    /// is used as literal text (added tokens such as `<asr_text>`).
    ByteLevel { literal: HashSet<u32> },
}

/// A dense id → piece table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PieceVocab {
    pieces: Vec<String>,
    mode: Mode,
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
        Ok(Self {
            pieces,
            mode: Mode::Piece,
        })
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

    /// A GPT-2 byte-level BPE vocabulary.
    ///
    /// `tokens` are the ids `0..tokens.len()`, in order: the byte-alphabet
    /// symbols from `vocab.json` sorted by id, or from GGUF's
    /// `tokenizer.ggml.tokens` when `tokenizer.ggml.model = gpt2`. `specials`
    /// are the added tokens that follow (ids `tokens.len()` and up, from
    /// `tokenizer_config.json`'s `added_tokens_decoder`); their text is used
    /// literally by [`Detokenizer::push`] rather than decoded through the
    /// byte table, so `<asr_text>` survives even though it was never part of
    /// the byte alphabet. A special whose own id sits inside `0..tokens.len()`
    /// is allowed too, replacing that slot.
    ///
    /// # Errors
    ///
    /// [`AudioError::Vocab`] for an empty token list, a gap between the ids
    /// `specials` covers and the ids `tokens` covers, or a non-special piece
    /// holding a character the byte table does not map to a byte.
    pub fn from_byte_level(
        tokens: Vec<String>,
        specials: &[(u32, String)],
    ) -> Result<Self, AudioError> {
        if tokens.is_empty() {
            return Err(vocab_err("no tokens"));
        }
        for token in &tokens {
            if let Some(bad) = token.chars().find(|c| !BYTE_LEVEL_TABLE.contains_key(c)) {
                return Err(vocab_err(format!(
                    "byte-level piece {token:?} has a character {bad:?} the byte table does not map"
                )));
            }
        }
        let mut by_id: Vec<Option<String>> = tokens.into_iter().map(Some).collect();
        let mut literal = HashSet::with_capacity(specials.len());
        for (id, text) in specials {
            let idx = usize::try_from(*id).map_err(|_| vocab_err("an id past usize"))?;
            if by_id.len() <= idx {
                by_id.resize(idx + 1, None);
            }
            by_id[idx] = Some(text.clone());
            literal.insert(*id);
        }
        let pieces = by_id
            .into_iter()
            .enumerate()
            .map(|(id, p)| p.ok_or_else(|| vocab_err(format!("id {id} has no piece"))))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            pieces,
            mode: Mode::ByteLevel { literal },
        })
    }

    /// Whether this vocabulary decodes through the GPT-2 byte table instead
    /// of `SentencePiece`'s `▁` word marker.
    #[must_use]
    pub fn is_byte_level(&self) -> bool {
        matches!(self.mode, Mode::ByteLevel { .. })
    }

    /// Whether `id`'s piece is literal text in byte-level mode (an added
    /// token). Always `false` outside byte-level mode.
    fn is_literal(&self, id: u32) -> bool {
        match &self.mode {
            Mode::ByteLevel { literal } => literal.contains(&id),
            Mode::Piece => false,
        }
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
    /// Byte-level mode only: raw bytes decoded from pushed symbols that do
    /// not yet form a complete UTF-8 prefix. Empty in `SentencePiece` mode.
    pending: Vec<u8>,
}

impl Detokenizer {
    /// Starts an empty transcript.
    #[must_use]
    pub fn new(vocab: Arc<PieceVocab>) -> Self {
        Self {
            vocab,
            text: String::new(),
            language: None,
            pending: Vec::new(),
        }
    }

    /// Adds one id; returns the text it appended.
    pub fn push(&mut self, id: u32) -> Option<String> {
        if let Some(lang) = self.vocab.language_of(id) {
            self.language = Some(lang.to_owned());
            return None;
        }
        let piece = self.vocab.piece(id)?.to_owned();
        if self.vocab.is_byte_level() {
            return self.push_byte_level(id, &piece);
        }
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

    /// The byte-level branch of [`Self::push`]: an added token's piece is
    /// appended as literal text; anything else is bytes through the GPT-2
    /// table, buffered until they complete a character.
    fn push_byte_level(&mut self, id: u32, piece: &str) -> Option<String> {
        if self.vocab.is_literal(id) {
            let mut delta = self.take_valid_prefix();
            delta.push_str(piece);
            return self.commit(delta);
        }
        self.pending.reserve(piece.chars().count());
        for ch in piece.chars() {
            // `PieceVocab::from_byte_level` refused any non-literal piece
            // holding a character the table does not map, so every piece
            // reaching here maps every char to a byte.
            let byte = *BYTE_LEVEL_TABLE
                .get(&ch)
                .expect("from_byte_level validated every byte-level piece");
            self.pending.push(byte);
        }
        let delta = self.take_valid_prefix();
        self.commit(delta)
    }

    /// Drains the longest valid UTF-8 prefix of `pending` into a `String`,
    /// leaving an incomplete trailing sequence (if any) buffered.
    fn take_valid_prefix(&mut self) -> String {
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_owned();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                let out = String::from_utf8(self.pending[..valid_up_to].to_vec())
                    .expect("valid_up_to bounds a valid UTF-8 prefix");
                self.pending.drain(..valid_up_to);
                out
            }
        }
    }

    /// Appends `delta` to the transcript, or returns `None` for an empty one
    /// (nothing new to report, the same convention as the `SentencePiece` path).
    fn commit(&mut self, delta: String) -> Option<String> {
        if delta.is_empty() {
            return None;
        }
        self.text.push_str(&delta);
        Some(delta)
    }

    /// Ends a byte-level stream: an incomplete character still in the pending
    /// buffer is reported as U+FFFD instead of being silently dropped. Call at
    /// most once, after the last [`Self::push`].
    ///
    /// `SentencePiece` mode has no pending buffer, so this is always `None`
    /// there.
    pub fn flush(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }
        self.pending.clear();
        self.commit("\u{FFFD}".to_owned())
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
