//! Reasoning and tool-calling schemes supported by shiso.
//!
//! Different model families structure their thinking traces and tool-call payloads
//! using different tag conventions. This module configures and constructs
//! [`StreamParser`] implementations matching each scheme.

use crate::segment::Segment;
use crate::tags::{TagParser, ToolBody};

/// Which reasoning tags a stream carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningScheme {
    /// No reasoning tags; all tokens are emitted as prose.
    None,
    /// Gemma 4 reasoning channel: `<|channel>thought\n` … `<channel|>`.
    Gemma4,
    /// Qwen 3 reasoning tags: `<think>` … `</think>`.
    Qwen3,
}

/// Which tool-call tags a stream carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallScheme {
    /// No tool call tags; all tokens are emitted as prose.
    None,
    /// Gemma 4 tool calling: `<|tool_call>call:NAME{...}<tool_call|>`.
    Gemma4,
    /// Qwen 3 tool calling: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`.
    Qwen3,
    /// Qwen 3.5 tool calling: the same brackets around an XML body,
    /// `<function=NAME>` wrapping one `<parameter=KEY>` block per argument.
    Qwen35,
}

/// A stateful parser over decoded text.
pub trait StreamParser: Send + std::fmt::Debug {
    /// Consumes `text` (any chunking) and returns the segments it completes.
    fn push(&mut self, text: &str) -> Vec<Segment>;

    /// The end of the turn: releases held-back text; an unterminated thinking or
    /// tool block is closed (thinking as `ThinkingEnd`, a tool call as `Text` of
    /// its raw bytes, since a half call is not a call).
    fn finish(&mut self) -> Vec<Segment>;

    /// Whether the next token should be picked greedily because a sampled one
    /// could only break the format here: inside a tool call's tags, or at a
    /// possible closing tag. Prose, reasoning and argument values sample as the
    /// request asked. Parsers with no such structure never ask.
    fn wants_greedy(&self) -> bool {
        false
    }
}

/// A pass-through parser that emits all text as [`Segment::Text`] without inspecting tags.
#[derive(Debug, Default)]
pub struct PassthroughParser;

impl StreamParser for PassthroughParser {
    fn push(&mut self, text: &str) -> Vec<Segment> {
        if text.is_empty() {
            Vec::new()
        } else {
            vec![Segment::Text(text.to_string())]
        }
    }

    fn finish(&mut self) -> Vec<Segment> {
        Vec::new()
    }
}

/// The reasoning tags a scheme brackets its trace with, `("", "")` for a scheme
/// that has none.
const fn reasoning_tags(reasoning: ReasoningScheme) -> (&'static str, &'static str) {
    match reasoning {
        ReasoningScheme::None => ("", ""),
        ReasoningScheme::Gemma4 => ("<|channel>thought\n", "<channel|>"),
        ReasoningScheme::Qwen3 => ("<think>", "</think>"),
    }
}

/// Whether `prompt` leaves a reasoning block open, so that the turn continuing
/// it starts inside the trace rather than in prose.
///
/// This is a question about the rendered prompt and not about the family,
/// because it is the chat template that decides it and the same template
/// decides it both ways. Qwen 3.5's closes the generation prompt with
/// `<think>\n` when thinking is on - the model then emits the trace and a bare
/// `</think>`, never an opener - and with `<think>\n\n</think>\n\n` when it is
/// off, which is a trace that is already over. Qwen 3's closes it with neither
/// and leaves the opener to the model. One rule reads all three: the trace is
/// open when the last opener in the prompt has no closer after it.
#[must_use]
pub fn resumes_thinking(reasoning: ReasoningScheme, prompt: &str) -> bool {
    let (open, close) = reasoning_tags(reasoning);
    if open.is_empty() {
        return false;
    }
    let Some(last_open) = prompt.rfind(open) else {
        return false;
    };
    match prompt.rfind(close) {
        Some(last_close) => last_close < last_open,
        None => true,
    }
}

/// Builds the parser for a scheme pair. `(None, None)` is a pass-through that emits `Text` only.
///
/// The turn is taken to start in prose. A caller that has the rendered prompt
/// should use [`parser_for_prompt`] instead, because a template can spend the
/// reasoning opener itself and leave the model nothing to emit but the trace.
#[must_use]
pub fn parser(reasoning: ReasoningScheme, tools: ToolCallScheme) -> Box<dyn StreamParser> {
    build(reasoning, tools, false)
}

/// Builds the parser for a scheme pair, for a turn that continues `prompt`.
///
/// An empty `prompt` is a turn with no rendered text behind it - the raw
/// completions surface, which was never run through a template - and starts in
/// prose, which is what [`parser`] does.
#[must_use]
pub fn parser_for_prompt(
    reasoning: ReasoningScheme,
    tools: ToolCallScheme,
    prompt: &str,
) -> Box<dyn StreamParser> {
    build(reasoning, tools, resumes_thinking(reasoning, prompt))
}

/// The half both entries share.
fn build(
    reasoning: ReasoningScheme,
    tools: ToolCallScheme,
    in_thinking: bool,
) -> Box<dyn StreamParser> {
    if reasoning == ReasoningScheme::None && tools == ToolCallScheme::None {
        return Box::new(PassthroughParser);
    }

    let (think_open, think_close) = reasoning_tags(reasoning);

    let (tool_open, tool_close, tool_body) = match tools {
        ToolCallScheme::None => ("", "", ToolBody::None),
        ToolCallScheme::Gemma4 => ("<|tool_call>", "<tool_call|>", ToolBody::GemmaCall),
        ToolCallScheme::Qwen3 => ("<tool_call>", "</tool_call>", ToolBody::Json),
        ToolCallScheme::Qwen35 => ("<tool_call>", "</tool_call>", ToolBody::QwenXml),
    };

    Box::new(TagParser::new(
        think_open,
        think_close,
        tool_open,
        tool_close,
        tool_body,
        in_thinking,
    ))
}

/// Builds a parser configured for Gemma 4 reasoning and tool calls.
#[must_use]
pub fn gemma4() -> Box<dyn StreamParser> {
    parser(ReasoningScheme::Gemma4, ToolCallScheme::Gemma4)
}

/// Builds a parser configured for Qwen 3 reasoning and tool calls.
#[must_use]
pub fn qwen3() -> Box<dyn StreamParser> {
    parser(ReasoningScheme::Qwen3, ToolCallScheme::Qwen3)
}

/// Builds a parser configured for Qwen 3.5: the same reasoning tags as Qwen 3
/// around an XML tool-call body, for a turn that continues `prompt`.
#[must_use]
pub fn qwen35(prompt: &str) -> Box<dyn StreamParser> {
    parser_for_prompt(ReasoningScheme::Qwen3, ToolCallScheme::Qwen35, prompt)
}

/// Builds a pass-through parser that emits all tokens as prose.
#[must_use]
pub fn passthrough() -> Box<dyn StreamParser> {
    parser(ReasoningScheme::None, ToolCallScheme::None)
}
