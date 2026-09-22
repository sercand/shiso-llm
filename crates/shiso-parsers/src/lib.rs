//! Reasoning and tool-call parsers with stream hold-back.
//!
//! # Why this crate exists
//!
//! When language models generate text, structured output such as reasoning traces
//! and tool calls are encoded directly into the token stream using delimiter tags.
//! For example:
//!
//! - Gemma 4 emits reasoning inside `<|channel>thought\n` … `<channel|>`, and tool
//!   calls as `<|tool_call>call:NAME{...}<tool_call|>`.
//! - Qwen 3 emits reasoning inside `<think>` … `</think>`, and tool calls as
//!   `<tool_call>{"name": ..., "arguments": ...}</tool_call>`.
//!
//! Token generation yields fragments that do not align with tag boundaries. A tag
//! like `<|channel>thought\n` may arrive across several tokens. If tokens were decoded
//! and immediately streamed as prose, the beginning of a tag would leak to the client
//! before the complete marker arrived.
//!
//! This crate provides stateful stream parsers that hold back any trailing suffix of
//! the decoded text that could still become a tag opener or closer. When the tag
//! completes, it transitions the stream state and produces structured [`Segment`]s
//! in production order. If the suffix diverges from the expected tag, it is released
//! as ordinary prose.

#![deny(unsafe_code)]

pub mod gemma_dict;
pub mod schemes;
pub mod segment;
pub mod tags;

pub use gemma_dict::{GemmaDictError, gemma_dict_to_json};
pub use schemes::{
    PassthroughParser, ReasoningScheme, StreamParser, ToolCallScheme, gemma4, parser,
    parser_for_prompt, passthrough, qwen3, qwen35, resumes_thinking,
};
pub use segment::Segment;
pub use tags::TagParser;
