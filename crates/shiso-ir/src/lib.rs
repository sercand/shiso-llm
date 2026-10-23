//! The neutral conversation IR: the narrow waist between the wire formats and
//! the engine.
//!
//! Each API surface parses its own body into a [`Request`] and encodes a
//! [`Response`] or a stream of [`StreamEvent`] back out into its own shape.
//! Nothing between the two knows which surface a value came from.
//!
//! # One message type, both directions
//!
//! The conversation is a list of [`Message`], and a [`Response`] carries one.
//! Continuing a conversation is therefore:
//!
//! ```
//! # use shiso_ir::{Message, Request, Response, Role, StopReason, Usage};
//! # fn demo(mut request: Request, response: Response) -> Request {
//! request.push(response.into_message());
//! request
//! # }
//! ```
//!
//! There is no response-only message shape and no list of candidate answers to
//! pick from. Both existed in the shape this replaces, and together they meant a
//! generated turn could not be put back into the conversation that produced it
//! without being taken apart and rebuilt - which is a step every caller has to
//! get right, and which silently drops whatever the rebuild forgets.
//!
//! # A turn is an ordered list of blocks
//!
//! Text, thinking, tool calls, tool results and media are all [`Block`]s of one
//! list, in the order they were produced. Nothing rides beside that list. See
//! [`message`] for why the alternative loses ordering the moment a turn
//! interleaves two kinds of content.
//!
//! # Shaped after Messages and Responses, not chat completions
//!
//! Where the two families disagree, this follows the block-structured one:
//! instructions are a field rather than a turn, a tool result is a block rather
//! than a role, a thinking trace is a block rather than a marker inside the text,
//! and a stream announces blocks rather than fragments of one flat string. The
//! chat-completions shape is a surface that adapts to this, not the other way
//! round.
//!
//! # What is not here
//!
//! HTTP, routers and error types for either: those are the server's. Engine
//! types: the engine sees resolved sampling values, never a caller's request.
//! And no conversion into an engine type, because turning a [`Request`] into one
//! needs the server's preset and its seed source as inputs - see [`request`].
#![forbid(unsafe_code)]

pub mod decision;
mod embed;
pub mod image;
mod message;
mod request;
mod response;
mod stream;
mod tools;

pub use embed::{EmbedEncoding, EmbedInput, EmbedRequest, EmbedResponse, EmbedUsage};
pub use message::{Block, MediaKind, MediaSource, Message, Role, Thinking, ToolResult, ToolUse};
pub use request::{Effort, Request, ResponseFormat, Sampling, ServiceClass, ThinkingDirective};
pub use response::{Alternative, Logprobs, Response, StopReason, TokenLogprob, Usage};
pub use stream::{BlockDelta, BlockStart, EventStream, ResponseBuilder, StreamError, StreamEvent};
pub use tools::{RepetitionDetection, ToolChoice, ToolDefinition};

#[cfg(test)]
mod tests;
