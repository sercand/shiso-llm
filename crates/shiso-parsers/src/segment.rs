//! One piece of a turn, in production order.
//!
//! As a model generates text, reasoning markers and tool-calling tags emerge in
//! the stream. Rather than waiting for a turn to conclude before extracting
//! structured payloads, [`Segment`] classifies the output piece by piece while
//! preserving chronological order.

/// One piece of a turn, in production order.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Segment {
    /// Ordinary prose intended for the client.
    Text(String),
    /// Opening marker of a model's reasoning trace.
    ThinkingStart,
    /// A fragment of reasoning content.
    Thinking(String),
    /// Closing marker of a model's reasoning trace.
    ThinkingEnd,
    /// A whole call: the name and its arguments as a JSON text (an object, or the
    /// raw text when it did not convert).
    ToolCall {
        /// Name of the tool being called.
        name: String,
        /// Tool arguments formatted as JSON.
        arguments: String,
    },
}
