//! The same turn, delivered as it is produced.
//!
//! A stream is the content blocks of a [`Message`] announced, filled in and
//! closed one at a time, followed by the accounting. That is not an arbitrary
//! framing: it is the block list of [`crate::Response::message`], in order, with
//! the text of each block arriving in pieces.
//!
//! # The accumulator is the proof
//!
//! [`ResponseBuilder`] folds a stream back into exactly the [`crate::Response`] a
//! non-streaming run would have produced. That makes the equivalence a thing the
//! code states and the tests check, rather than a property two separately written
//! encoders are each trying to preserve. A surface that streams and a surface that
//! does not are then reading the same turn, and a field that only the blocking
//! path fills in is a compile error instead of a missing value in one client.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::message::{Block, Message, Role, Thinking, ToolUse};
use crate::response::{Response, StopReason, Usage};

/// One event of a generated turn.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StreamEvent {
    /// The turn is starting. Carries what a caller needs before any content.
    Start {
        /// Bare response identifier, as [`crate::Response::id`].
        id: String,
        /// The model that is answering.
        model: String,
        /// Seconds since the epoch when the turn began.
        created: u64,
    },
    /// A content block opens at `index`, which is its position in the finished
    /// message.
    BlockStart {
        /// Position of the block in the message being built.
        index: u32,
        /// What kind of block is opening, and the parts of it that are known at
        /// the moment it opens.
        block: BlockStart,
    },
    /// More of the block at `index`.
    BlockDelta {
        /// The block this extends.
        index: u32,
        /// The piece that arrived.
        delta: BlockDelta,
    },
    /// The block at `index` is complete.
    BlockStop {
        /// The block that closed.
        index: u32,
    },
    /// The turn is complete.
    Stop {
        /// Why generation ended.
        stop: StopReason,
        /// What the turn cost.
        usage: Usage,
        /// Sampled ids that reached no earlier event, so that the ids a caller
        /// saw add up to the tokens it was charged for.
        token_ids: Vec<u32>,
    },
    /// Generation failed part way. Terminal: nothing follows it.
    Failed {
        /// What went wrong, in the producer's own words.
        message: String,
    },
}

/// A block opening, with whatever is known at that moment.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BlockStart {
    /// Prose.
    Text,
    /// A reasoning trace.
    Thinking,
    /// A tool call. Both fields are known before any arguments arrive, which is
    /// what lets a caller show the call while it is still being written.
    ToolUse {
        /// The identifier the result will quote back.
        id: String,
        /// The tool being called.
        name: String,
    },
    /// A refusal.
    Refusal,
}

/// A piece of an open block.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BlockDelta {
    /// More prose.
    Text {
        /// The fragment; concatenate in arrival order.
        text: String,
        /// The ids that produced it.
        token_ids: Vec<u32>,
    },
    /// More of a reasoning trace.
    Thinking {
        /// The fragment; concatenate in arrival order.
        text: String,
        /// The ids that produced it.
        token_ids: Vec<u32>,
    },
    /// The signature for the open thinking block, which arrives once it is whole.
    ThinkingSignature(String),
    /// More of a tool call's arguments, as the JSON text the model is writing.
    ToolInput {
        /// The fragment; concatenate in arrival order.
        fragment: String,
        /// The ids that produced it.
        token_ids: Vec<u32>,
    },
    /// More of a refusal.
    Refusal {
        /// The fragment; concatenate in arrival order.
        text: String,
    },
}

/// A stream of events, which is what generation hands a surface encoder.
pub type EventStream = std::pin::Pin<Box<dyn futures::Stream<Item = StreamEvent> + Send>>;

/// Why a stream could not be folded into a turn.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StreamError {
    /// An event arrived before the stream said it was starting.
    #[error("a {what} event arrived before the stream started")]
    NotStarted {
        /// The event kind that arrived too early.
        what: &'static str,
    },
    /// A delta or a stop named a block that never opened.
    #[error("block {index} was never opened")]
    UnknownBlock {
        /// The index the event named.
        index: u32,
    },
    /// A delta did not match the kind of block it named.
    #[error("block {index} is {open}, and a {delta} fragment cannot extend it")]
    BlockKindMismatch {
        /// The block the delta named.
        index: u32,
        /// What that block actually is.
        open: &'static str,
        /// What the delta would have extended.
        delta: &'static str,
    },
    /// A tool call's arguments never became valid JSON.
    ///
    /// Reported rather than repaired: the arguments are what the tool is about to
    /// be run with, and a guess at what the model meant is a call the caller never
    /// authorized.
    #[error("the arguments of the call to {name} are not valid JSON: {source}")]
    ToolInput {
        /// The tool whose arguments did not parse.
        name: String,
        /// The parse failure.
        #[source]
        source: serde_json::Error,
    },
    /// The stream ended without saying the turn was complete.
    #[error("the stream ended with no terminal event")]
    Unterminated,
    /// The producer reported a failure.
    #[error("generation failed: {message}")]
    Failed {
        /// What the producer said.
        message: String,
    },
}

/// Folds a stream of events into the turn it describes.
///
/// Blocks are kept by index rather than pushed as they open, because the indices
/// are the message's own order and a producer is free to close one block after
/// opening the next. Collecting them in a map keyed by index and reading the map
/// in order at the end is what makes the finished message's order the producer's
/// order rather than the arrival order.
#[derive(Debug, Default)]
pub struct ResponseBuilder {
    head: Option<Head>,
    blocks: BTreeMap<u32, Partial>,
    token_ids: Vec<u32>,
    end: Option<(StopReason, Usage)>,
}

/// What the opening event carried.
#[derive(Debug, Clone)]
struct Head {
    id: String,
    model: String,
    created: u64,
}

/// A block being filled in.
#[derive(Debug)]
enum Partial {
    Text(String),
    Thinking {
        text: String,
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
    Refusal(String),
}

impl Partial {
    /// The word this kind of block is called in a mismatch report.
    const fn kind(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::Thinking { .. } => "thinking",
            Self::ToolUse { .. } => "a tool call",
            Self::Refusal(_) => "a refusal",
        }
    }

    /// Closes the block into the finished content.
    fn finish(self) -> Result<Block, StreamError> {
        Ok(match self {
            Self::Text(text) => Block::Text(text),
            Self::Thinking { text, signature } => Block::Thinking(Thinking { text, signature }),
            Self::ToolUse { id, name, input } => {
                // An empty argument run is an object with no fields, which is what
                // a no-argument call looks like on every wire format. Parsing ""
                // would fail and report a JSON error about a call that was fine.
                let parsed = if input.trim().is_empty() {
                    Value::Object(serde_json::Map::new())
                } else {
                    serde_json::from_str(&input).map_err(|source| StreamError::ToolInput {
                        name: name.clone(),
                        source,
                    })?
                };
                Block::ToolUse(ToolUse {
                    id,
                    name,
                    input: parsed,
                })
            }
            Self::Refusal(text) => Block::Refusal(text),
        })
    }
}

impl ResponseBuilder {
    /// A builder with nothing in it.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes one event.
    ///
    /// # Errors
    ///
    /// [`StreamError`] when the event cannot belong to the turn built so far: it
    /// arrived before the start, it named a block that never opened, or it would
    /// extend a block of another kind.
    pub fn accept(&mut self, event: StreamEvent) -> Result<(), StreamError> {
        match event {
            StreamEvent::Start { id, model, created } => {
                self.head = Some(Head { id, model, created });
            }
            StreamEvent::BlockStart { index, block } => {
                self.require_started("block start")?;
                self.blocks.insert(index, open(block));
            }
            StreamEvent::BlockDelta { index, delta } => {
                self.require_started("block delta")?;
                self.extend(index, delta)?;
            }
            // A close carries nothing: the block already holds everything it was
            // given, and the finished content is built once, at the end.
            StreamEvent::BlockStop { index } => {
                self.require_started("block stop")?;
                if !self.blocks.contains_key(&index) {
                    return Err(StreamError::UnknownBlock { index });
                }
            }
            StreamEvent::Stop {
                stop,
                usage,
                token_ids,
            } => {
                self.require_started("stop")?;
                self.token_ids.extend(token_ids);
                self.end = Some((stop, usage));
            }
            StreamEvent::Failed { message } => return Err(StreamError::Failed { message }),
        }
        Ok(())
    }

    /// Builds the turn the stream described.
    ///
    /// # Errors
    ///
    /// [`StreamError::Unterminated`] when the stream never said it was complete,
    /// and [`StreamError::ToolInput`] when a call's arguments are not valid JSON.
    pub fn finish(self) -> Result<Response, StreamError> {
        let head = self.head.ok_or(StreamError::Unterminated)?;
        let (stop, usage) = self.end.ok_or(StreamError::Unterminated)?;
        let mut content = Vec::with_capacity(self.blocks.len());
        for (_index, partial) in self.blocks {
            content.push(partial.finish()?);
        }
        Ok(Response {
            id: head.id,
            model: head.model,
            created: head.created,
            message: Message::new(Role::Assistant, content),
            stop,
            usage,
            logprobs: None,
            token_ids: self.token_ids,
        })
    }

    /// Refuses an event that arrived before the stream started.
    fn require_started(&self, what: &'static str) -> Result<(), StreamError> {
        if self.head.is_none() {
            return Err(StreamError::NotStarted { what });
        }
        Ok(())
    }

    /// Applies one fragment to the block it names.
    fn extend(&mut self, index: u32, delta: BlockDelta) -> Result<(), StreamError> {
        let open = self
            .blocks
            .get_mut(&index)
            .ok_or(StreamError::UnknownBlock { index })?;
        let mismatch = |open: &Partial, delta: &'static str| StreamError::BlockKindMismatch {
            index,
            open: open.kind(),
            delta,
        };
        match delta {
            BlockDelta::Text { text, token_ids } => match open {
                Partial::Text(body) => {
                    body.push_str(&text);
                    self.token_ids.extend(token_ids);
                }
                other => return Err(mismatch(other, "text")),
            },
            BlockDelta::Thinking { text, token_ids } => match open {
                Partial::Thinking { text: body, .. } => {
                    body.push_str(&text);
                    self.token_ids.extend(token_ids);
                }
                other => return Err(mismatch(other, "thinking")),
            },
            BlockDelta::ThinkingSignature(value) => match open {
                Partial::Thinking { signature, .. } => *signature = Some(value),
                other => return Err(mismatch(other, "thinking-signature")),
            },
            BlockDelta::ToolInput {
                fragment,
                token_ids,
            } => match open {
                Partial::ToolUse { input, .. } => {
                    input.push_str(&fragment);
                    self.token_ids.extend(token_ids);
                }
                other => return Err(mismatch(other, "tool-argument")),
            },
            BlockDelta::Refusal { text } => match open {
                Partial::Refusal(body) => body.push_str(&text),
                other => return Err(mismatch(other, "refusal")),
            },
        }
        Ok(())
    }
}

/// The empty block one opening event describes.
fn open(block: BlockStart) -> Partial {
    match block {
        BlockStart::Text => Partial::Text(String::new()),
        BlockStart::Thinking => Partial::Thinking {
            text: String::new(),
            signature: None,
        },
        BlockStart::ToolUse { id, name } => Partial::ToolUse {
            id,
            name,
            input: String::new(),
        },
        BlockStart::Refusal => Partial::Refusal(String::new()),
    }
}
