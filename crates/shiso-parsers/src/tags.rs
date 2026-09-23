//! Generic tag-driven stream parser with suffix hold-back.
//!
//! # Why hold-back exists
//!
//! Language models stream output token by token. Token boundaries do not align with
//! semantic delimiters: a tag like `<|channel>thought\n` or `<tool_call>` can easily
//! be split across multiple tokens (for instance `<|`, `channel`, `>thought\n`).
//! If each token's text were immediately emitted as prose, the opening bytes of the
//! tag would leak to the client as text before the parser had a chance to recognise
//! the structure.
//!
//! [`TagParser`] prevents leaks by holding back any trailing suffix that forms a
//! proper prefix of an expected tag. The withheld suffix is only released once
//! incoming text either completes the tag (triggering a state transition) or
//! proves that the suffix was regular prose all along.

use std::sync::Arc;

use crate::gemma_dict::gemma_dict_to_json;
use crate::schemes::StreamParser;
use crate::segment::Segment;
use crate::string_params::StringParams;

/// Format of the payload between tool call opening and closing tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolBody {
    /// No tool calls expected.
    None,
    /// Standard JSON object containing `name` and `arguments` keys.
    Json,
    /// Gemma 4 `call:NAME{...}` grammar.
    GemmaCall,
    /// Qwen 3.5 XML grammar: `<function=NAME>` wrapping one
    /// `<parameter=KEY>` … `</parameter>` block per argument.
    QwenXml,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    Text,
    Thinking,
    ToolCall(String),
}

/// Everything a [`TagParser`] is built from.
#[derive(Debug)]
pub(crate) struct TagParserDesc {
    /// The reasoning opener, empty for a scheme with none.
    pub(crate) think_open: &'static str,
    /// The reasoning closer, empty for a scheme with none.
    pub(crate) think_close: &'static str,
    /// The tool-call opener, empty for a scheme with none.
    pub(crate) tool_open: &'static str,
    /// The tool-call closer, empty for a scheme with none.
    pub(crate) tool_close: &'static str,
    /// How the text between the tool tags is read.
    pub(crate) tool_body: ToolBody,
    /// Whether the turn starts inside the reasoning trace.
    pub(crate) in_thinking: bool,
    /// The request's string-typed tool parameters.
    pub(crate) strings: Arc<StringParams>,
}

/// A streaming tag parser that extracts reasoning and tool-call blocks.
#[derive(Debug)]
pub struct TagParser {
    think_open: &'static str,
    think_close: &'static str,
    tool_open: &'static str,
    tool_close: &'static str,
    tool_body: ToolBody,
    state: State,
    buffer: String,
    strings: Arc<StringParams>,
}

impl TagParser {
    /// Creates a new [`TagParser`] configured with the given tag strings and body format.
    ///
    /// `in_thinking` is the state the turn *starts* in, which is not always
    /// prose: a template whose generation prompt ends with the reasoning opener
    /// has already spent it, so the model emits the trace and then a bare
    /// closer. A parser that started in prose would hand the whole trace to the
    /// client as text and leak the closing tag behind it.
    #[must_use]
    pub(crate) fn new(desc: TagParserDesc) -> Self {
        Self {
            think_open: desc.think_open,
            think_close: desc.think_close,
            tool_open: desc.tool_open,
            tool_close: desc.tool_close,
            tool_body: desc.tool_body,
            state: if desc.in_thinking {
                State::Thinking
            } else {
                State::Text
            },
            buffer: String::new(),
            strings: desc.strings,
        }
    }

    fn decode_gemma_call(body: &str) -> Option<Segment> {
        let trimmed = body.trim_start();
        let after_call = trimmed.strip_prefix("call:")?.trim_start();
        let brace_idx = after_call.find('{')?;
        let name = after_call[..brace_idx].trim();
        let dict_str = &after_call[brace_idx..];

        let valid_name = !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-');
        if !valid_name {
            return None;
        }

        let arguments = gemma_dict_to_json(dict_str).ok()?;
        Some(Segment::ToolCall {
            name: name.to_string(),
            arguments,
        })
    }

    fn decode_json_call(body: &str) -> Option<Segment> {
        let val = serde_json::from_str::<serde_json::Value>(body).ok()?;
        let name = val.get("name").and_then(serde_json::Value::as_str)?;
        let arguments = match val.get("arguments") {
            Some(serde_json::Value::String(s)) => {
                if serde_json::from_str::<serde_json::Value>(s).is_ok() {
                    s.clone()
                } else {
                    serde_json::to_string(s).unwrap_or_else(|_| "{}".to_string())
                }
            }
            Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
            None => "{}".to_string(),
        };
        Some(Segment::ToolCall {
            name: name.to_string(),
            arguments,
        })
    }

    /// Reads Qwen 3.5's XML call body:
    ///
    /// ```text
    /// <function=get_weather>
    /// <parameter=city>
    /// Paris
    /// </parameter>
    /// </function>
    /// ```
    ///
    /// The template writes a value with `tojson` when it is a mapping or a
    /// sequence and with `| string` otherwise, so a value's type is not in the
    /// wire format and has to be recovered. What comes back here is the JSON
    /// value the text parses as - an object, an array, a number, a boolean or
    /// null - and the text itself when it parses as none of those or as a JSON
    /// string, since a bare `Paris` is not JSON and a quoted `"Paris"` was
    /// quoted by the model rather than by the template.
    ///
    /// The guess is made only where the request did not answer the question: a
    /// parameter its schema declares as a string is the text, whatever the text
    /// looks like.
    fn decode_qwen_xml_call(body: &str, strings: &StringParams) -> Option<Segment> {
        const FUNCTION_OPEN: &str = "<function=";
        const PARAMETER_OPEN: &str = "<parameter=";

        let after_open = body.find(FUNCTION_OPEN)? + FUNCTION_OPEN.len();
        let rest = &body[after_open..];
        let name_end = rest.find('>')?;
        let name = rest[..name_end].trim();
        let valid_name = !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-');
        if !valid_name {
            return None;
        }

        let mut arguments = serde_json::Map::new();
        let mut rest = &rest[name_end + 1..];
        // A `</function>` is where the parameters stop; a body that never wrote
        // one is still a call whose parameters are all here, because the caller
        // only reaches this after the closing tool tag was matched. The *last*
        // one, because a value may contain the tag - HTML, code, or text about
        // this very format - and the first would cut that value short.
        if let Some(end) = rest.rfind("</function>") {
            rest = &rest[..end];
        }
        while let Some(start) = rest.find(PARAMETER_OPEN) {
            let after = &rest[start + PARAMETER_OPEN.len()..];
            let Some(key_end) = after.find('>') else {
                break;
            };
            let key = after[..key_end].trim().to_string();
            let value_region = &after[key_end + 1..];
            // The close that ends this value is the first *structural* one, not
            // the first occurrence: a value that contains `</parameter>` is
            // still one value.
            let (raw, consumed) = match structural_parameter_close(value_region) {
                Some(close) => (&value_region[..close], close + PARAMETER_CLOSE_TAG.len()),
                // An unterminated parameter runs to the end of the call: the
                // model wrote a value and lost the closing tag, and the value is
                // still what it meant to pass.
                None => (value_region, value_region.len()),
            };
            let value = if strings.is_string(name, &key) {
                serde_json::Value::String(strip_one_newline(raw).to_owned())
            } else {
                qwen_xml_value(raw)
            };
            arguments.insert(key, value);
            rest = &value_region[consumed..];
        }

        Some(Segment::ToolCall {
            name: name.to_string(),
            arguments: serde_json::Value::Object(arguments).to_string(),
        })
    }

    fn decode_tool_call(&self, body: &str) -> Segment {
        let parsed = match self.tool_body {
            ToolBody::GemmaCall => Self::decode_gemma_call(body),
            ToolBody::Json => Self::decode_json_call(body),
            ToolBody::QwenXml => Self::decode_qwen_xml_call(body, &self.strings),
            ToolBody::None => None,
        };
        parsed.unwrap_or_else(|| {
            Segment::Text(format!("{}{body}{}", self.tool_open, self.tool_close))
        })
    }

    fn step_text(&mut self, segments: &mut Vec<Segment>) -> bool {
        let mut first_match = None;

        if !self.think_open.is_empty()
            && let Some(idx) = self.buffer.find(self.think_open)
        {
            first_match = Some((idx, true));
        }

        if !self.tool_open.is_empty()
            && let Some(idx) = self.buffer.find(self.tool_open)
        {
            match first_match {
                Some((existing, _)) if idx < existing => {
                    first_match = Some((idx, false));
                }
                None => {
                    first_match = Some((idx, false));
                }
                _ => {}
            }
        }

        if let Some((start, is_think)) = first_match {
            if start > 0 {
                segments.push(Segment::Text(self.buffer[..start].to_string()));
            }
            let tag_len = if is_think {
                self.think_open.len()
            } else {
                self.tool_open.len()
            };
            self.buffer = self.buffer[start + tag_len..].to_string();

            if is_think {
                segments.push(Segment::ThinkingStart);
                self.state = State::Thinking;
            } else {
                self.state = State::ToolCall(String::new());
            }
            true
        } else {
            let hold = hold_back_len(&self.buffer, &[self.think_open, self.tool_open]);
            if hold > 0 {
                let split_pos = self.buffer.len() - hold;
                if split_pos > 0 {
                    segments.push(Segment::Text(self.buffer[..split_pos].to_string()));
                    self.buffer = self.buffer[split_pos..].to_string();
                }
                false
            } else {
                if !self.buffer.is_empty() {
                    segments.push(Segment::Text(std::mem::take(&mut self.buffer)));
                }
                false
            }
        }
    }

    fn step_thinking(&mut self, segments: &mut Vec<Segment>) -> bool {
        if self.think_close.is_empty() {
            if !self.buffer.is_empty() {
                segments.push(Segment::Thinking(std::mem::take(&mut self.buffer)));
            }
            return false;
        }

        if let Some(start) = self.buffer.find(self.think_close) {
            if start > 0 {
                segments.push(Segment::Thinking(self.buffer[..start].to_string()));
            }
            self.buffer = self.buffer[start + self.think_close.len()..].to_string();
            segments.push(Segment::ThinkingEnd);
            self.state = State::Text;
            true
        } else {
            let hold = hold_back_len(&self.buffer, &[self.think_close]);
            if hold > 0 {
                let split_pos = self.buffer.len() - hold;
                if split_pos > 0 {
                    segments.push(Segment::Thinking(self.buffer[..split_pos].to_string()));
                    self.buffer = self.buffer[split_pos..].to_string();
                }
            } else if !self.buffer.is_empty() {
                segments.push(Segment::Thinking(std::mem::take(&mut self.buffer)));
            }
            false
        }
    }

    /// Where the call's own closing tag starts in the buffer, once it has
    /// arrived.
    ///
    /// For the Qwen XML body that is the first occurrence the call's structure
    /// has reached ([`structural_tool_close`]); an occurrence inside a value is
    /// left in the buffer, and the hold-back below moves it into the body,
    /// because it holds only a trailing *proper* prefix of the tag. Other
    /// bodies keep the first occurrence: their payloads have their own quoting.
    fn tool_close_at(&self) -> Option<usize> {
        if self.tool_body != ToolBody::QwenXml {
            return self.buffer.find(self.tool_close);
        }
        let State::ToolCall(body) = &self.state else {
            return None;
        };
        let mut from = 0;
        while let Some(rel) = self.buffer[from..].find(self.tool_close) {
            let at = from + rel;
            // Body and buffer together: a tag the check reads can straddle the
            // two, since the hold-back only keeps prefixes of the tool closer.
            let mut before = body.clone();
            before.push_str(&self.buffer[..at]);
            // A body with no `<function=` has no value to protect, so its first
            // closer ends it: a malformed call is handed back as text at once
            // rather than swallowing the rest of the turn until it ends.
            if !before.contains("<function=") || structural_tool_close(&before) {
                return Some(at);
            }
            from = at + self.tool_close.len();
        }
        None
    }

    fn step_tool_call(&mut self, segments: &mut Vec<Segment>) -> bool {
        if self.tool_close.is_empty() {
            self.state = State::Text;
            return false;
        }

        let close_at = self.tool_close_at();
        let State::ToolCall(body) = &mut self.state else {
            return false;
        };

        if let Some(start) = close_at {
            body.push_str(&self.buffer[..start]);
            self.buffer = self.buffer[start + self.tool_close.len()..].to_string();
            let completed_body = std::mem::take(body);
            self.state = State::Text;
            let seg = self.decode_tool_call(&completed_body);
            segments.push(seg);
            true
        } else {
            let hold = hold_back_len(&self.buffer, &[self.tool_close]);
            if hold > 0 {
                let split_pos = self.buffer.len() - hold;
                body.push_str(&self.buffer[..split_pos]);
                self.buffer = self.buffer[split_pos..].to_string();
            } else {
                body.push_str(&std::mem::take(&mut self.buffer));
            }
            false
        }
    }
}

impl StreamParser for TagParser {
    fn push(&mut self, text: &str) -> Vec<Segment> {
        self.buffer.push_str(text);
        let mut segments = Vec::new();

        loop {
            let keep_going = match self.state {
                State::Text => self.step_text(&mut segments),
                State::Thinking => self.step_thinking(&mut segments),
                State::ToolCall(_) => self.step_tool_call(&mut segments),
            };
            if !keep_going {
                break;
            }
        }

        segments
    }

    /// Greedy only for the Qwen XML body, whose tags are the ones a sampled
    /// token has been seen to corrupt: a held-back `<tool_call>` prefix of two
    /// or more bytes (a lone `<` is too common in prose and code), anywhere in a
    /// call between tags, and inside a value only while its tail can still
    /// become `</parameter>`. Reasoning never is.
    fn wants_greedy(&self) -> bool {
        if self.tool_body != ToolBody::QwenXml {
            return false;
        }
        match &self.state {
            State::Thinking => false,
            State::Text => longest_matching_prefix_suffix(&self.buffer, self.tool_open) >= 2,
            State::ToolCall(body) => {
                let mut seen = body.clone();
                seen.push_str(&self.buffer);
                match open_value(&seen) {
                    None => true,
                    Some(value) => longest_matching_prefix_suffix(value, PARAMETER_CLOSE_TAG) >= 2,
                }
            }
        }
    }

    fn finish(&mut self) -> Vec<Segment> {
        let mut segments = Vec::new();
        match std::mem::replace(&mut self.state, State::Text) {
            State::Text => {
                if !self.buffer.is_empty() {
                    segments.push(Segment::Text(std::mem::take(&mut self.buffer)));
                }
            }
            State::Thinking => {
                if !self.buffer.is_empty() {
                    segments.push(Segment::Thinking(std::mem::take(&mut self.buffer)));
                }
                segments.push(Segment::ThinkingEnd);
            }
            State::ToolCall(body) => {
                let raw = format!(
                    "{}{body}{}",
                    self.tool_open,
                    std::mem::take(&mut self.buffer)
                );
                if !raw.is_empty() {
                    segments.push(Segment::Text(raw));
                }
            }
        }
        segments
    }
}

/// One `<parameter=…>` body with the template's framing newlines removed.
///
/// Exactly one newline is stripped from each end, because exactly one is what
/// the template put there: a value's own leading or trailing blank line is part
/// of the value, and trimming freely would eat it. Interior newlines are never
/// touched - a multi-line parameter is the case the XML grammar exists for.
fn strip_one_newline(raw: &str) -> &str {
    let text = raw
        .strip_prefix("\r\n")
        .or_else(|| raw.strip_prefix('\n'))
        .unwrap_or(raw);
    text.strip_suffix("\r\n")
        .or_else(|| text.strip_suffix('\n'))
        .unwrap_or(text)
}

/// One `<parameter=…>` body, as the value it stands for, when the request did
/// not declare its type: the JSON it parses as, or the text.
fn qwen_xml_value(raw: &str) -> serde_json::Value {
    let text = strip_one_newline(raw);
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(serde_json::Value::String(_)) | Err(_) => serde_json::Value::String(text.to_string()),
        Ok(value) => value,
    }
}

/// The text of the parameter value still open at the end of `call`, or `None`
/// when the call is between tags. A value is open once its `<parameter=KEY>`
/// has seen `>` and no structural `</parameter>` has closed it.
///
/// The last `<parameter=` is taken as the current one, so a value that itself
/// contains `<parameter=` reads as a new parameter. The format has no escape for
/// that; the cost is some greedy steps inside such a value.
fn open_value(call: &str) -> Option<&str> {
    const OPEN: &str = "<parameter=";
    let start = call.rfind(OPEN)?;
    let name_end = call[start + OPEN.len()..].find('>')?;
    let value = &call[start + OPEN.len() + name_end + 1..];
    match structural_parameter_close_strict(value) {
        Some(_) => None,
        None => Some(value),
    }
}

fn longest_matching_prefix_suffix(s: &str, tag: &str) -> usize {
    if tag.is_empty() || s.is_empty() {
        return 0;
    }
    let max_check = s.len().min(tag.len().saturating_sub(1));
    for len in (1..=max_check).rev() {
        if s.is_char_boundary(s.len() - len) && tag.starts_with(&s[s.len() - len..]) {
            return len;
        }
    }
    0
}

fn hold_back_len(s: &str, tags: &[&str]) -> usize {
    let mut max_hold = 0;
    for &tag in tags {
        if !tag.is_empty() {
            max_hold = max_hold.max(longest_matching_prefix_suffix(s, tag));
        }
    }
    max_hold
}

const PARAMETER_CLOSE_TAG: &str = "</parameter>";

/// The offset of the first `</parameter>` in `region` that is structural:
/// followed, after whitespace, by another parameter, the end of the function, or
/// nothing yet. `None` when every close in `region` is followed by other text,
/// so the value is still open. The greedy-sampling query (Task 12) uses this
/// directly.
fn structural_parameter_close_strict(region: &str) -> Option<usize> {
    let mut from = 0;
    while let Some(found) = region[from..].find(PARAMETER_CLOSE_TAG) {
        let at = from + found;
        let after = region[at + PARAMETER_CLOSE_TAG.len()..].trim_start();
        if after.is_empty() || after.starts_with("<parameter=") || after.starts_with("</function>")
        {
            return Some(at);
        }
        from = at + PARAMETER_CLOSE_TAG.len();
    }
    None
}

/// The offset of the `</parameter>` that closes the value starting `region`:
/// the first structural one. With no structural close at all, the last
/// `</parameter>` is taken, which is the most text a truncated call can still
/// mean.
fn structural_parameter_close(region: &str) -> Option<usize> {
    structural_parameter_close_strict(region).or_else(|| region.rfind(PARAMETER_CLOSE_TAG))
}

/// Whether a `</tool_call>` found right after `before` is the call's own close:
/// the call's structure has ended there. Anything else is a value that happens to
/// contain the tag.
///
/// `</parameter>` counts as well as `</function>`, so a model that drops the
/// function close still ends its call. A value that itself ends in a valid
/// closing sequence cannot be told apart from a real close; the format has no
/// escape, and the round-trip tests list that case as one it cannot carry.
pub(crate) fn structural_tool_close(before: &str) -> bool {
    let trimmed = before.trim_end();
    trimmed.ends_with("</function>") || trimmed.ends_with("</parameter>")
}
