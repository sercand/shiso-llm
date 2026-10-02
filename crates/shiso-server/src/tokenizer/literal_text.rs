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

/// Escapes every occurrence of `markers` in every string of `messages`.
///
/// Every string, not only text parts: a tool call's arguments are caller text
/// too, and a template writes a string argument into the prompt raw - Qwen's
/// writes each one between `<parameter=...>` tags - so an agent editing a file
/// that spells a marker would otherwise put live markers in the prompt. The
/// structural values a template branches on - roles, part types, call ids -
/// never spell a marker, so escaping them is a no-op. Object keys are left
/// alone: a template matches on them, and none of them is caller prose.
pub fn escape(messages: &mut [Value], markers: &[&str]) -> Escaped {
    let mut escaped = Escaped::default();
    escape_more(messages, markers, &mut escaped);
    escaped
}

/// [`escape`] over more values - the tool list - numbering on from `escaped`,
/// so one [`unescape`] undoes both.
pub fn escape_more(values: &mut [Value], markers: &[&str], escaped: &mut Escaped) {
    if markers.is_empty() {
        return;
    }
    for value in values {
        escape_value(value, markers, escaped);
    }
}

/// Escapes every string inside `value`.
fn escape_value(value: &mut Value, markers: &[&str], escaped: &mut Escaped) {
    match value {
        Value::String(text) => escape_string(text, markers, escaped),
        Value::Array(items) => {
            for item in items {
                escape_value(item, markers, escaped);
            }
        }
        Value::Object(fields) => {
            for field in fields.values_mut() {
                escape_value(field, markers, escaped);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
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

    #[test]
    fn tool_call_arguments_and_tool_documents_are_escaped() {
        let mut messages = vec![json!({
            "role": "assistant",
            "content": [],
            "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "write",
                "arguments": {"content": "x <|image_pad|> y", "n": 1},
            }}],
        })];
        let mut escaped = escape(&mut messages, MARKERS);
        let mut tools = vec![json!({"type": "function", "function": {
            "name": "read", "description": "returns <|video_pad|> verbatim",
        }})];
        super::escape_more(&mut tools, MARKERS, &mut escaped);
        let argument = messages[0]["tool_calls"][0]["function"]["arguments"]["content"]
            .as_str()
            .expect("text")
            .to_owned();
        let description = tools[0]["function"]["description"]
            .as_str()
            .expect("text")
            .to_owned();
        assert!(!argument.contains("<|image_pad|>"), "{argument}");
        assert!(!description.contains("<|video_pad|>"), "{description}");
        let (restored, ranges) =
            unescape(format!("{argument}|{description}"), &escaped).expect("unescapes");
        assert_eq!(restored, "x <|image_pad|> y|returns <|video_pad|> verbatim");
        assert_eq!(ranges.len(), 2);
    }
}
