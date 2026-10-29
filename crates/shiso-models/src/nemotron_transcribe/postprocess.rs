//! What is done to the decoder's text before a caller sees it: the loop guard
//! and the tag handling.
//!
//! A greedy speech decoder that loses its place repeats the last phrase until
//! its token budget runs out. The guard keeps one copy of such a run; the
//! thresholds (a character 21 times, a longer block 20 times) are far past
//! anything speech repeats, so a stutter or a chanted phrase survives.
//!
//! The transcript can carry three kinds of tag the vocabulary holds as single
//! tokens: `<spk:N>` speaker turns, `<ev:*>` events and `<sty:*>` styles.

/// The spellings every transcript tag starts with: speaker turns, events and
/// styles. The vocabulary holds each tag as one special token, and a server
/// that kept the model off its special tokens would keep it off these too.
pub const TAG_PREFIXES: [&str; 3] = ["<spk:", "<ev:", "<sty:"];

/// Times one character must repeat back to back before it is a loop.
const CHAR_LOOP: usize = 21;
/// Times a block of two or more characters must repeat before it is a loop.
const BLOCK_LOOP: usize = 20;
/// The longest block, in characters, the guard looks for.
const MAX_BLOCK: usize = 4096;

/// `text` with every loop collapsed to one copy of what repeated.
///
/// At each position the shortest repeating block wins, so `"aaaa…"` collapses
/// as one character rather than as pairs of it.
#[must_use]
pub fn collapse_loops(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut at = 0usize;
    while at < chars.len() {
        if let Some((period, copies)) = loop_at(&chars, at) {
            out.extend(&chars[at..at + period]);
            at += period * copies;
        } else {
            out.push(chars[at]);
            at += 1;
        }
    }
    out
}

/// The shortest block starting at `at` that repeats often enough to be a
/// loop, as its length and how many copies run back to back.
fn loop_at(chars: &[char], at: usize) -> Option<(usize, usize)> {
    let rest = chars.len() - at;
    for period in 1..=MAX_BLOCK {
        let need = if period == 1 { CHAR_LOOP } else { BLOCK_LOOP };
        if period * need > rest {
            return None;
        }
        // Cheap reject first: the copy after the block must start the same.
        if chars[at] != chars[at + period] {
            continue;
        }
        let block = &chars[at..at + period];
        let copies = 1 + chars[at + period..]
            .chunks_exact(period)
            .take_while(|chunk| *chunk == block)
            .count();
        if copies >= need {
            return Some((period, copies));
        }
    }
    None
}

/// The tag a transcript's text holds at its start, and its length in bytes.
fn tag_at(text: &str) -> Option<(Tag, usize)> {
    let inner = &text[..text.find('>')?];
    let close = inner.len();
    let inner = inner.strip_prefix('<')?;
    if inner.contains('<') || inner.chars().any(char::is_whitespace) {
        return None;
    }
    let tag = if let Some(n) = inner.strip_prefix("spk:") {
        Tag::Speaker(n.parse().ok()?)
    } else if inner.starts_with("ev:") || inner.starts_with("sty:") {
        Tag::Other
    } else {
        return None;
    };
    Some((tag, close + 1))
}

/// A tag the transcript can carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tag {
    /// `<spk:N>`.
    Speaker(u8),
    /// `<ev:*>` or `<sty:*>`.
    Other,
}

/// The text with its tags split out, in order.
fn pieces(text: &str) -> Vec<(Option<Tag>, &str)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut at = 0usize;
    while at < text.len() {
        if text.as_bytes()[at] == b'<'
            && let Some((tag, len)) = tag_at(&text[at..])
        {
            out.push((None, &text[start..at]));
            out.push((Some(tag), &text[at..at + len]));
            at += len;
            start = at;
            continue;
        }
        at += text[at..].chars().next().map_or(1, char::len_utf8);
    }
    out.push((None, &text[start..]));
    out
}

/// Joins words with single spaces, so a removed tag leaves no gap.
fn join_words<'a>(parts: impl Iterator<Item = &'a str>) -> String {
    parts
        .flat_map(str::split_whitespace)
        .collect::<Vec<_>>()
        .join(" ")
}

/// `text` without its `<ev:*>` and `<sty:*>` tags, and without its
/// `<spk:N>` tags unless `keep_speakers`; whitespace is normalized to single
/// spaces.
#[must_use]
pub fn strip_tags(text: &str, keep_speakers: bool) -> String {
    join_words(
        pieces(text)
            .into_iter()
            .filter(|(tag, _)| match tag {
                None => true,
                Some(Tag::Speaker(_)) => keep_speakers,
                Some(Tag::Other) => false,
            })
            .map(|(_, s)| s),
    )
}

/// The transcript as speaker turns: the speaker each `<spk:N>` names and the
/// text up to the next one, its other tags stripped. Text before the first
/// tag is a turn with no speaker; empty turns are dropped.
#[must_use]
pub fn speaker_turns(text: &str) -> Vec<(Option<u8>, String)> {
    let mut turns: Vec<(Option<u8>, String)> = Vec::new();
    let mut speaker = None;
    let mut words: Vec<&str> = Vec::new();
    let mut flush = |speaker: Option<u8>, words: &mut Vec<&str>| {
        if !words.is_empty() {
            turns.push((speaker, join_words(words.drain(..))));
        }
    };
    for (tag, piece) in pieces(text) {
        match tag {
            None => words.push(piece),
            Some(Tag::Speaker(n)) => {
                flush(speaker, &mut words);
                speaker = Some(n);
            }
            Some(Tag::Other) => {}
        }
    }
    flush(speaker, &mut words);
    turns
        .into_iter()
        .filter(|(_, text)| !text.is_empty())
        .collect()
}

#[cfg(test)]
#[path = "postprocess_tests.rs"]
mod tests;
