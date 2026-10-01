//! Keeping marker text a caller wrote from becoming a marker.
//!
//! # Why the rendered string cannot tell the two apart
//!
//! A chat template writes a media marker as text - `<|image_pad|>` - and the
//! vocabulary turns that text into the marker's id. A caller who types the same
//! characters into a message produces the same string, and once the template
//! has joined everything into one prompt nothing in it says which occurrence
//! came from where. Encoded as is, the caller's copy becomes a placeholder with
//! no picture behind it: refused as a marker count mismatch, or, let through,
//! fed to the model as a row that is not the text the caller asked about.
//!
//! So the caller's copies are swapped for an escape before the template runs,
//! and swapped back after it, with the byte range each one lands on recorded.
//! The encoder then reads those ranges as ordinary text. The escape is built
//! from private-use characters, so no template writes one of its own, and a
//! caller who sends one is escaped like a marker: what comes back out of the
//! template is exactly the text that went in.

use std::ops::Range;

use serde_json::Value;

/// Opens an escape. Private use, so no template writes it.
const OPEN: char = '\u{E000}';
/// Closes an escape.
const CLOSE: char = '\u{E001}';

/// The caller text the escapes stand for, by escape index.
#[derive(Debug, Default)]
pub struct Escaped {
    originals: Vec<String>,
}

impl Escaped {
    /// Whether nothing was escaped, which is every prompt that never spells a
    /// marker.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.originals.is_empty()
    }
}

/// The escape could not be undone: the template altered one beyond
/// recognition. Escapes are opaque to every shipped template, so this is a
/// template doing character surgery on message text rather than a request
/// defect.
#[derive(Debug, thiserror::Error)]
#[error("the chat template altered an escaped span of message text")]
pub struct Unescape;

/// Escapes every occurrence of `markers` in the message text of `messages`.
///
/// Only text a caller wrote is touched: text parts, a turn's string content and
/// its reasoning trace. The structural keys the template branches on - roles,
/// part types, call ids - are left alone, and tool-call arguments are dumped
/// as JSON by the template, which escapes the angle brackets a marker is
/// spelled with anyway.
pub fn escape(messages: &mut [Value], markers: &[&str]) -> Escaped {
    let mut escaped = Escaped::default();
    if markers.is_empty() {
        return escaped;
    }
    for message in messages {
        let Some(turn) = message.as_object_mut() else {
            continue;
        };
        if let Some(Value::String(trace)) = turn.get_mut("reasoning_content") {
            escape_string(trace, markers, &mut escaped);
        }
        match turn.get_mut("content") {
            Some(Value::String(text)) => escape_string(text, markers, &mut escaped),
            Some(Value::Array(parts)) => {
                for part in parts {
                    if let Some(Value::String(text)) = part.get_mut("text") {
                        escape_string(text, markers, &mut escaped);
                    }
                }
            }
            _ => {}
        }
    }
    escaped
}

/// Rewrites `text` with each marker, and each escape character, replaced by an
/// escape naming its index in `escaped`.
fn escape_string(text: &mut String, markers: &[&str], escaped: &mut Escaped) {
    if !text.contains(OPEN) && !text.contains(CLOSE) && !markers.iter().any(|m| text.contains(m)) {
        return;
    }
    let mut out = String::with_capacity(text.len() + 8);
    let mut rest = text.as_str();
    while let Some((at, len)) = next_protected(rest, markers) {
        out.push_str(&rest[..at]);
        let index = escaped.originals.len();
        escaped.originals.push(rest[at..at + len].to_owned());
        out.push(OPEN);
        out.push_str(&index.to_string());
        out.push(CLOSE);
        rest = &rest[at + len..];
    }
    out.push_str(rest);
    *text = out;
}

/// The earliest protected span in `text`, as its start and length: a marker,
/// or a lone escape character. At one position the longest marker wins, so a
/// marker that is a prefix of another does not split it.
fn next_protected(text: &str, markers: &[&str]) -> Option<(usize, usize)> {
    let escape_char = text
        .char_indices()
        .find(|(_, c)| *c == OPEN || *c == CLOSE)
        .map(|(at, c)| (at, c.len_utf8()));
    markers
        .iter()
        .filter_map(|marker| text.find(marker).map(|at| (at, marker.len())))
        .chain(escape_char)
        .min_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))
}

/// Undoes [`escape`] on the rendered prompt, returning the prompt as the
/// template would have written it from the caller's own text, and the byte
/// range each escaped span occupies in it.
///
/// A template may drop a turn - older reasoning traces are routinely left out -
/// or write one twice, so escapes are matched by index rather than by order.
///
/// # Errors
///
/// [`Unescape`] when an escape no longer parses, or names an index that was
/// never handed out.
pub fn unescape(
    rendered: String,
    escaped: &Escaped,
) -> Result<(String, Vec<Range<usize>>), Unescape> {
    if escaped.is_empty() {
        return Ok((rendered, Vec::new()));
    }
    let mut out = String::with_capacity(rendered.len());
    let mut ranges = Vec::new();
    let mut rest = rendered.as_str();
    while let Some(open) = rest.find(OPEN) {
        out.push_str(&rest[..open]);
        let after = &rest[open + OPEN.len_utf8()..];
        let close = after.find(CLOSE).ok_or(Unescape)?;
        let index: usize = after[..close].parse().map_err(|_bad| Unescape)?;
        let original = escaped.originals.get(index).ok_or(Unescape)?;
        let start = out.len();
        out.push_str(original);
        ranges.push(start..out.len());
        rest = &after[close + CLOSE.len_utf8()..];
    }
    if rest.contains(CLOSE) {
        return Err(Unescape);
    }
    out.push_str(rest);
    Ok((out, ranges))
}

#[cfg(test)]
mod tests {
    use super::{escape, unescape};
    use serde_json::json;

    const MARKERS: &[&str] = &["<|image_pad|>", "<|video_pad|>"];

    #[test]
    fn a_round_trip_returns_the_callers_text_and_marks_each_marker() {
        let mut messages = vec![json!({
            "role": "user",
            "content": [{"type": "text", "text": "a <|image_pad|> b <|video_pad|>"}],
        })];
        let escaped = escape(&mut messages, MARKERS);
        let text = messages[0]["content"][0]["text"]
            .as_str()
            .expect("text")
            .to_owned();
        assert!(!text.contains("<|image_pad|>"), "{text}");
        let rendered = format!("<s>{text}<e><|image_pad|>");
        let (restored, ranges) = unescape(rendered, &escaped).expect("unescapes");
        assert_eq!(
            restored,
            "<s>a <|image_pad|> b <|video_pad|><e><|image_pad|>"
        );
        let spans: Vec<&str> = ranges.iter().map(|r| &restored[r.clone()]).collect();
        assert_eq!(spans, vec!["<|image_pad|>", "<|video_pad|>"]);
    }

    #[test]
    fn a_dropped_or_repeated_turn_still_restores_by_index() {
        let mut messages = vec![
            json!({"role": "assistant", "content": [], "reasoning_content": "x<|image_pad|>"}),
            json!({"role": "user", "content": "y<|video_pad|>"}),
        ];
        let escaped = escape(&mut messages, MARKERS);
        let user = messages[1]["content"].as_str().expect("text").to_owned();
        let (restored, ranges) = unescape(format!("{user}{user}"), &escaped).expect("unescapes");
        assert_eq!(restored, "y<|video_pad|>y<|video_pad|>");
        assert_eq!(ranges.len(), 2);
    }

    #[test]
    fn an_escape_character_the_caller_sent_comes_back_unchanged() {
        let mut messages = vec![json!({"role": "user", "content": "a\u{E000}0\u{E001}b"})];
        let escaped = escape(&mut messages, MARKERS);
        let text = messages[0]["content"].as_str().expect("text").to_owned();
        let (restored, _ranges) = unescape(text, &escaped).expect("unescapes");
        assert_eq!(restored, "a\u{E000}0\u{E001}b");
    }

    #[test]
    fn text_without_markers_is_left_alone() {
        let mut messages = vec![json!({"role": "user", "content": "plain"})];
        let escaped = escape(&mut messages, MARKERS);
        assert!(escaped.is_empty());
        assert_eq!(messages[0]["content"], "plain");
    }
}
