//! The timestamp head's tokenizer: the serialised subword model the export
//! carries base64-encoded in `ctc.tokenizer.spm_model`, encoding text to the
//! head's class ids.
//!
//! Only what this checkpoint's model uses is built, and anything else is
//! refused when the model is read: merge-ranked pieces (the highest-scoring
//! adjacent pair merges first, the leftmost on a tie), user-defined pieces
//! matched whole and never merged, a precompiled character map for
//! normalisation, a dummy `▁` in front, spaces written as `▁`, and runs of
//! unknown pieces reported as one unknown id. The alignment's per-word and
//! whole-stream encodings must agree id for id, so "close" is not good
//! enough: a different split moves every word after it.

use base64::Engine as _;
use shiso_audio::AudioError;

/// The word-boundary marker spaces become.
const BOUNDARY: &str = "\u{2581}";

/// Piece types of the serialised model.
const NORMAL: u64 = 1;
const UNKNOWN: u64 = 2;
const CONTROL: u64 = 3;
const USER_DEFINED: u64 = 4;

/// The merge-ranked model type.
const MERGE_RANKED: u64 = 2;

fn refuse(reason: impl Into<String>) -> AudioError {
    AudioError::Vocab {
        reason: reason.into(),
    }
}

/// One field's payload in a serialised message.
#[derive(Debug, Clone, Copy)]
enum Payload<'a> {
    Varint(u64),
    Fixed32(u32),
    Bytes(&'a [u8]),
    Fixed64,
}

impl<'a> Payload<'a> {
    fn bytes(self) -> &'a [u8] {
        match self {
            Self::Bytes(b) => b,
            _ => &[],
        }
    }

    fn varint(self) -> Option<u64> {
        match self {
            Self::Varint(v) => Some(v),
            _ => None,
        }
    }

    fn fixed32(self) -> Option<u32> {
        match self {
            Self::Fixed32(v) => Some(v),
            _ => None,
        }
    }
}

/// The fields of one serialised message, in order. Only one model and three
/// of its messages are read, so a walk this small beats a dependency.
struct Fields<'a> {
    bytes: &'a [u8],
    at: usize,
    failed: bool,
}

impl<'a> Fields<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            at: 0,
            failed: false,
        }
    }

    fn varint(&mut self) -> Result<u64, AudioError> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self
                .bytes
                .get(self.at)
                .ok_or_else(|| refuse("a truncated varint"))?;
            self.at += 1;
            v |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(refuse("an overlong varint"))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], AudioError> {
        let b = self
            .bytes
            .get(self.at..self.at.saturating_add(n))
            .ok_or_else(|| refuse("a truncated field"))?;
        self.at += n;
        Ok(b)
    }

    fn field(&mut self) -> Result<(u64, Payload<'a>), AudioError> {
        let key = self.varint()?;
        let payload = match key & 7 {
            0 => Payload::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Payload::Fixed64
            }
            2 => {
                let n = usize::try_from(self.varint()?).map_err(|_| refuse("a huge field"))?;
                Payload::Bytes(self.take(n)?)
            }
            5 => {
                let b = self.take(4)?;
                Payload::Fixed32(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            wire => return Err(refuse(format!("wire type {wire} is not a model"))),
        };
        Ok((key >> 3, payload))
    }
}

impl<'a> Iterator for Fields<'a> {
    type Item = (u64, Result<Payload<'a>, AudioError>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.at >= self.bytes.len() {
            return None;
        }
        match self.field() {
            Ok((field, payload)) => Some((field, Ok(payload))),
            Err(e) => {
                self.failed = true;
                Some((0, Err(e)))
            }
        }
    }
}

/// The last varint `field` of `message` holds.
// `None`: the message does not set the field, so its default applies.
fn field_varint(message: &[u8], field: u64) -> Result<Option<u64>, AudioError> {
    let mut found = None;
    for (f, payload) in Fields::new(message) {
        let payload = payload?;
        if f == field {
            found = payload.varint().or(found);
        }
    }
    Ok(found)
}

/// [`field_varint`] read as a flag.
// `None`: the message does not set the flag, so its default applies.
fn field_bool(message: &[u8], field: u64) -> Result<Option<bool>, AudioError> {
    Ok(field_varint(message, field)?.map(|v| v != 0))
}

/// A piece's id: its position in the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PieceId(pub u32);

/// The precompiled normalisation map: a double-array trie over input bytes
/// whose values index NUL-terminated replacements.
#[derive(Debug, Clone)]
struct CharMap {
    units: Vec<u32>,
    replacements: Vec<u8>,
}

impl CharMap {
    fn parse(blob: &[u8]) -> Result<Self, AudioError> {
        let size = blob
            .get(..4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
            .ok_or_else(|| refuse("a character map without its trie size"))?;
        let trie = blob
            .get(4..4 + size)
            .filter(|t| t.len().is_multiple_of(4) && !t.is_empty())
            .ok_or_else(|| refuse("a character map whose trie overruns it"))?;
        Ok(Self {
            units: trie
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            replacements: blob[4 + size..].to_vec(),
        })
    }

    /// The longest prefix of `input` the map rewrites: its byte length and
    /// its replacement.
    fn longest(&self, input: &[u8]) -> Option<(usize, &[u8])> {
        let unit = |pos: usize| self.units.get(pos).copied();
        let offset = |u: u32| ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize;
        let mut best = None;
        let mut pos = offset(unit(0)?);
        for (i, &byte) in input.iter().enumerate() {
            pos ^= usize::from(byte);
            let u = unit(pos)?;
            if u & ((1 << 31) | 0xFF) != u32::from(byte) {
                break;
            }
            pos ^= offset(u);
            if (u >> 8) & 1 == 1 {
                let value = (unit(pos)? & ((1 << 31) - 1)) as usize;
                let tail = self.replacements.get(value..)?;
                let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
                best = Some((i + 1, &tail[..end]));
            }
        }
        best
    }
}

/// A symbol of the text being merged: a byte range, linked both ways.
struct Symbol {
    start: usize,
    len: usize,
    frozen: bool,
    prev: Option<usize>,
    next: Option<usize>,
}

/// A candidate merge: the highest score pops first, the leftmost on a tie.
struct Pair {
    score: f32,
    left: usize,
    right: usize,
    len: usize,
}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Pair {}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .total_cmp(&other.score)
            .then(other.left.cmp(&self.left))
    }
}

/// The timestamp head's tokenizer.
#[derive(Debug, Clone)]
pub struct CtcTokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    /// Mergeable and user-defined pieces by text.
    ids: std::collections::HashMap<String, PieceId>,
    /// User-defined pieces, longest first, matched whole.
    user: Vec<String>,
    unknown: PieceId,
    map: Option<CharMap>,
}

/// The byte length of the UTF-8 character `bytes` starts with, by its lead
/// byte alone, as the model's own matcher reads it.
fn lead_len(bytes: &[u8]) -> usize {
    const LEN: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    bytes
        .first()
        .map_or(0, |b| LEN[usize::from(b >> 4)].min(bytes.len()))
}

impl CtcTokenizer {
    /// Reads the export's `ctc.tokenizer.spm_model` value.
    ///
    /// # Errors
    ///
    /// [`AudioError::Vocab`] for text that is not base64 or a model this
    /// tokenizer does not encode exactly.
    pub fn from_base64(text: &str) -> Result<Self, AudioError> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(text.trim())
            .map_err(|e| refuse(format!("the timestamp tokenizer is not base64: {e}")))?;
        Self::from_model(&bytes)
    }

    /// Reads a serialised model.
    ///
    /// # Errors
    ///
    /// [`AudioError::Vocab`] for bytes that are not a model, or a model with
    /// a feature this tokenizer leaves out.
    pub fn from_model(bytes: &[u8]) -> Result<Self, AudioError> {
        let mut pieces = Vec::new();
        let mut scores = Vec::new();
        let mut types = Vec::new();
        let mut trainer = None;
        let mut normalizer = None;
        for (field, body) in Fields::new(bytes) {
            let body = body?;
            match field {
                1 => {
                    let (mut piece, mut score, mut kind) = (None, 0.0f32, NORMAL);
                    for (f, b) in Fields::new(body.bytes()) {
                        let b = b?;
                        match f {
                            1 => {
                                piece = Some(
                                    String::from_utf8(b.bytes().to_vec())
                                        .map_err(|_| refuse("a piece that is not UTF-8"))?,
                                );
                            }
                            2 => score = b.fixed32().map_or(score, f32::from_bits),
                            3 => kind = b.varint().unwrap_or(kind),
                            _ => {}
                        }
                    }
                    pieces.push(piece.ok_or_else(|| refuse("a piece without its text"))?);
                    scores.push(score);
                    types.push(kind);
                }
                2 => trainer = Some(body.bytes().to_vec()),
                3 => normalizer = Some(body.bytes().to_vec()),
                _ => {}
            }
        }
        let trainer = trainer.ok_or_else(|| refuse("a model without its trainer spec"))?;
        let model_type = field_varint(&trainer, 3)?.unwrap_or(1);
        if model_type != MERGE_RANKED {
            return Err(refuse(format!(
                "model type {model_type}; the timestamp tokenizer encodes merge-ranked models only"
            )));
        }
        if field_bool(&trainer, 35)?.unwrap_or(false) {
            return Err(refuse("byte fallback is not built"));
        }
        if field_bool(&trainer, 24)?.unwrap_or(false) {
            return Err(refuse("whitespace as a suffix is not built"));
        }
        let normalizer = normalizer.unwrap_or_default();
        if !field_bool(&normalizer, 3)?.unwrap_or(true) {
            return Err(refuse("a model without the dummy prefix is not built"));
        }
        if field_bool(&normalizer, 4)?.unwrap_or(true) {
            return Err(refuse("whitespace collapsing is not built"));
        }
        if !field_bool(&normalizer, 5)?.unwrap_or(true) {
            return Err(refuse("unescaped whitespace is not built"));
        }
        let mut map = None;
        for (f, b) in Fields::new(&normalizer) {
            let b = b?;
            if f == 2 && !b.bytes().is_empty() {
                map = Some(CharMap::parse(b.bytes())?);
            }
            if f == 6 && !b.bytes().is_empty() {
                return Err(refuse("a normalisation rule table is not built"));
            }
        }
        let mut ids = std::collections::HashMap::with_capacity(pieces.len());
        let mut user = Vec::new();
        let mut unknown = None;
        for (i, (piece, kind)) in pieces.iter().zip(&types).enumerate() {
            let id = PieceId(u32::try_from(i).map_err(|_| refuse("too many pieces"))?);
            match *kind {
                NORMAL => {
                    ids.insert(piece.clone(), id);
                }
                USER_DEFINED => {
                    ids.insert(piece.clone(), id);
                    user.push(piece.clone());
                }
                UNKNOWN => unknown = Some(id),
                CONTROL => {}
                other => {
                    return Err(refuse(format!("piece type {other} is not built")));
                }
            }
        }
        user.sort_by_key(|p| std::cmp::Reverse(p.len()));
        Ok(Self {
            pieces,
            scores,
            ids,
            user,
            unknown: unknown.ok_or_else(|| refuse("a model without an unknown piece"))?,
            map,
        })
    }

    /// Pieces in the model; the head's blank class is the one after them.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Never true: a model without its unknown piece is refused.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// The text of piece `id`.
    #[must_use]
    pub fn piece(&self, id: PieceId) -> Option<&str> {
        self.pieces.get(id.0 as usize).map(String::as_str)
    }

    /// The user-defined piece `text` starts with, longest first.
    fn user_prefix(&self, text: &[u8]) -> Option<usize> {
        self.user
            .iter()
            .find(|p| text.starts_with(p.as_bytes()))
            .map(String::len)
    }

    /// `text` normalised: the dummy prefix, the character map (user-defined
    /// pieces copied as they are, a byte that starts no character replaced
    /// by U+FFFD), then spaces as `▁`.
    fn normalize(&self, text: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(text.len() + 3);
        out.extend_from_slice(BOUNDARY.as_bytes());
        let input = text.as_bytes();
        let mut at = 0;
        let push = |out: &mut Vec<u8>, piece: &[u8]| {
            for &b in piece {
                if b == b' ' {
                    out.extend_from_slice(BOUNDARY.as_bytes());
                } else {
                    out.push(b);
                }
            }
        };
        while at < input.len() {
            let rest = &input[at..];
            if let Some(n) = self.user_prefix(rest) {
                push(&mut out, &rest[..n]);
                at += n;
            } else if let Some((n, replacement)) = self.map.as_ref().and_then(|m| m.longest(rest)) {
                push(&mut out, replacement);
                at += n;
            } else {
                // `text` is a `str`, so every character is whole.
                let n = text[at..].chars().next().map_or(1, char::len_utf8);
                push(&mut out, &rest[..n]);
                at += n;
            }
        }
        out
    }

    /// `text` as piece ids.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<PieceId> {
        if text.is_empty() {
            return Vec::new();
        }
        let normalized = self.normalize(text);
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut at = 0;
        while at < normalized.len() {
            let rest = &normalized[at..];
            let (len, frozen) = self
                .user_prefix(rest)
                .map_or_else(|| (lead_len(rest), false), |n| (n, true));
            let index = symbols.len();
            symbols.push(Symbol {
                start: at,
                len,
                frozen,
                prev: index.checked_sub(1),
                next: None,
            });
            at += len;
            if at < normalized.len() {
                symbols[index].next = Some(index + 1);
            }
        }
        let text_of = |s: &Symbol| &normalized[s.start..s.start + s.len];
        let mut agenda = std::collections::BinaryHeap::new();
        let candidate = |symbols: &[Symbol], left: Option<usize>, right: Option<usize>| {
            let (l, r) = (left?, right?);
            let (a, b) = (&symbols[l], &symbols[r]);
            if a.frozen || b.frozen {
                return None;
            }
            let merged = std::str::from_utf8(&normalized[a.start..b.start + b.len]).ok()?;
            let id = self.ids.get(merged)?;
            Some(Pair {
                score: self.scores[id.0 as usize],
                left: l,
                right: r,
                len: a.len + b.len,
            })
        };
        for i in 1..symbols.len() {
            agenda.extend(candidate(&symbols, Some(i - 1), Some(i)));
        }
        while let Some(top) = agenda.pop() {
            let (l, r) = (top.left, top.right);
            if symbols[l].len == 0
                || symbols[r].len == 0
                || symbols[l].len + symbols[r].len != top.len
            {
                continue;
            }
            symbols[l].len += symbols[r].len;
            symbols[l].next = symbols[r].next;
            if let Some(n) = symbols[r].next {
                symbols[n].prev = Some(l);
            }
            symbols[r].len = 0;
            agenda.extend(candidate(&symbols, symbols[l].prev, Some(l)));
            agenda.extend(candidate(&symbols, Some(l), symbols[l].next));
        }
        let mut out: Vec<PieceId> = Vec::new();
        let mut cursor = (!symbols.is_empty()).then_some(0);
        while let Some(i) = cursor {
            let id = std::str::from_utf8(text_of(&symbols[i]))
                .ok()
                .and_then(|t| self.ids.get(t).copied())
                .unwrap_or(self.unknown);
            // A run of unknown pieces is one unknown id.
            if !(id == self.unknown && out.last() == Some(&self.unknown)) {
                out.push(id);
            }
            cursor = symbols[i].next;
        }
        out
    }
}

#[cfg(test)]
#[path = "spm_tests.rs"]
pub(crate) mod tests;
