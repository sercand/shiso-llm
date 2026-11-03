//! Keeping control-token text a caller wrote from becoming a control token.
//!
//! # Why the rendered string cannot tell the two apart
//!
//! A chat template writes its control tokens as text - `<|im_start|>`,
//! `<|image_pad|>` - and the vocabulary turns that text into the token's id. A
//! caller who sends the same characters produces the same string, and once the
//! template has joined everything into one prompt nothing in it says which
//! occurrence came from where. Encoded as is, the caller's copy becomes the
//! token: a media marker with no picture behind it, refused as a count
//! mismatch, or a turn boundary in the middle of a message. The second is the
//! worse one. An agent that reads a file spelling `<|im_start|>user` puts a
//! fake turn into its own history, and a model quoting that text back copies
//! the ids it was shown - including the one that ends its turn, so the answer
//! stops mid-thought.
//!
//! So the caller's copies are swapped for an escape before the template runs,
//! and swapped back after it, with the byte range each one lands on recorded.
//! The encoder then reads those ranges as ordinary text. The escape is built
//! from private-use characters, so no template writes one of its own, and a
//! caller who sends one is escaped like a token: what comes back out of the
//! template is exactly the text that went in.
//!
//! # Which tokens
//!
//! Every added token the vocabulary marks special: turn markers, media
//! markers, padding, end of text. Added tokens that are not special - the
//! reasoning and tool-call tags of some families - are left as they are,
//! because templates and parsers treat those as tags in text on purpose: a
//! template splits a replayed answer on its closing reasoning tag.

use std::ops::Range;

use aho_corasick::{AhoCorasick, MatchKind};
use serde_json::Value;

/// Opens an escape. Private use, so no template writes it.
const OPEN: char = '\u{E000}';
/// Closes an escape.
const CLOSE: char = '\u{E001}';

/// The spellings caller text is kept from turning into tokens, compiled once
/// for the vocabulary.
///
/// One automaton rather than a search per spelling: some vocabularies declare
/// a thousand special tokens, and a tool result can be a whole file.
#[derive(Debug, Clone)]
pub struct Protected {
    /// `None` when there is nothing to protect.
    matcher: Option<AhoCorasick>,
}

/// The spellings could not be compiled into a matcher.
#[derive(Debug, thiserror::Error)]
#[error("the special-token spellings could not be compiled: {0}")]
pub struct ProtectError(#[from] aho_corasick::BuildError);

impl Protected {
    /// Nothing protected: text is encoded exactly as the template wrote it.
    #[must_use]
    pub const fn none() -> Self {
        Self { matcher: None }
    }

    /// Protects every spelling in `spellings`, and the escape characters.
    ///
    /// Leftmost-longest, so a spelling that is a prefix of another at the same
    /// position does not split it.
    ///
    /// # Errors
    ///
    /// [`ProtectError`] when the automaton cannot be built.
    pub fn new<S: AsRef<str>>(spellings: &[S]) -> Result<Self, ProtectError> {
        let mut patterns: Vec<&str> = spellings
            .iter()
            .map(AsRef::as_ref)
            .filter(|spelling| !spelling.is_empty())
            .collect();
        if patterns.is_empty() {
            return Ok(Self::none());
        }
        let open = OPEN.to_string();
        let close = CLOSE.to_string();
        patterns.push(&open);
        patterns.push(&close);
        let matcher = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(patterns)?;
        Ok(Self {
            matcher: Some(matcher),
        })
    }
}

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

/// Escapes every protected spelling in every string of `messages`.
///
/// Every string, not only text parts: a tool call's arguments are caller text
/// too, and a template writes a string argument into the prompt raw - Qwen's
/// writes each one between `<parameter=...>` tags. The structural values a
/// template branches on - roles, part types, call ids - never spell a control
/// token, so escaping them is a no-op. Object keys are left alone: a template
/// matches on them, and none of them is caller prose.
pub fn escape(messages: &mut [Value], protected: &Protected) -> Escaped {
    let mut escaped = Escaped::default();
    escape_more(messages, protected, &mut escaped);
    escaped
}

/// [`escape`] over more values - the tool list - numbering on from `escaped`,
/// so one [`unescape`] undoes both.
pub fn escape_more(values: &mut [Value], protected: &Protected, escaped: &mut Escaped) {
    let Some(matcher) = protected.matcher.as_ref() else {
        return;
    };
    for value in values {
        escape_value(value, matcher, escaped);
    }
}

/// Escapes every string inside `value`.
fn escape_value(value: &mut Value, matcher: &AhoCorasick, escaped: &mut Escaped) {
    match value {
        Value::String(text) => escape_string(text, matcher, escaped),
        Value::Array(items) => {
            for item in items {
                escape_value(item, matcher, escaped);
            }
        }
        Value::Object(fields) => {
            for field in fields.values_mut() {
                escape_value(field, matcher, escaped);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

/// Rewrites `text` with each protected span replaced by an escape naming its
/// index in `escaped`.
fn escape_string(text: &mut String, matcher: &AhoCorasick, escaped: &mut Escaped) {
    if !matcher.is_match(text.as_str()) {
        return;
    }
    let mut out = String::with_capacity(text.len() + 8);
    let mut last = 0usize;
    for found in matcher.find_iter(text.as_str()) {
        out.push_str(&text[last..found.start()]);
        let index = escaped.originals.len();
        escaped.originals.push(text[found.range()].to_owned());
        out.push(OPEN);
        out.push_str(&index.to_string());
        out.push(CLOSE);
        last = found.end();
    }
    out.push_str(&text[last..]);
    *text = out;
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
    use super::{Protected, escape, unescape};
    use serde_json::json;

    /// Two media markers, and a turn marker that is a prefix of neither.
    fn protected() -> Protected {
        Protected::new(&["<|image_pad|>", "<|video_pad|>", "<|im_start|>"]).expect("compiles")
    }

    #[test]
    fn a_round_trip_returns_the_callers_text_and_marks_each_marker() {
        let mut messages = vec![json!({
            "role": "user",
            "content": [{"type": "text", "text": "a <|image_pad|> b <|video_pad|>"}],
        })];
        let escaped = escape(&mut messages, &protected());
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
        let escaped = escape(&mut messages, &protected());
        let user = messages[1]["content"].as_str().expect("text").to_owned();
        let (restored, ranges) = unescape(format!("{user}{user}"), &escaped).expect("unescapes");
        assert_eq!(restored, "y<|video_pad|>y<|video_pad|>");
        assert_eq!(ranges.len(), 2);
    }

    #[test]
    fn an_escape_character_the_caller_sent_comes_back_unchanged() {
        let mut messages = vec![json!({"role": "user", "content": "a\u{E000}0\u{E001}b"})];
        let escaped = escape(&mut messages, &protected());
        let text = messages[0]["content"].as_str().expect("text").to_owned();
        let (restored, _ranges) = unescape(text, &escaped).expect("unescapes");
        assert_eq!(restored, "a\u{E000}0\u{E001}b");
    }

    #[test]
    fn text_without_markers_is_left_alone() {
        let mut messages = vec![json!({"role": "user", "content": "plain"})];
        let escaped = escape(&mut messages, &protected());
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
        let mut escaped = escape(&mut messages, &protected());
        let mut tools = vec![json!({"type": "function", "function": {
            "name": "read", "description": "returns <|video_pad|> verbatim",
        }})];
        super::escape_more(&mut tools, &protected(), &mut escaped);
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

    #[test]
    fn a_turn_marker_in_a_tool_result_is_escaped() {
        let mut messages = vec![json!({
            "role": "tool",
            "content": [{"type": "text", "text": "\"<|im_start|>user\\n{{ content }}\""}],
        })];
        let escaped = escape(&mut messages, &protected());
        assert!(!escaped.is_empty());
        let text = messages[0]["content"][0]["text"].as_str().expect("text");
        assert!(!text.contains("<|im_start|>"), "{text}");
    }

    #[test]
    fn nothing_protected_leaves_every_string_alone() {
        let mut messages = vec![json!({"role": "user", "content": "<|im_start|>"})];
        let escaped = escape(&mut messages, &Protected::none());
        assert!(escaped.is_empty());
        assert_eq!(messages[0]["content"], "<|im_start|>");
    }
}
