//! The conversation: turns, and the blocks a turn is made of.
//!
//! One rule shapes everything here: **a turn's whole content is an ordered list
//! of blocks**, and every kind of content is a block. Text is a block, a thinking
//! trace is a block, a tool call is a block, a tool result is a block, an image is
//! a block. There are no side fields carrying content that the block list does not
//! mention.
//!
//! # Why that rule is worth the churn
//!
//! The alternative - a message with `content` plus a `tool_calls` list plus a
//! `reasoning` field beside it - loses ordering the moment a turn interleaves two
//! kinds. A model that says something, calls a tool, says something else and calls
//! another has produced one ordered sequence, and a shape with parallel fields can
//! only record "some text and two calls". Replaying that turn into the next
//! request then reorders it.
//!
//! The rule also has a second consequence that matters more: because a generated
//! turn is a [`Message`], a [`crate::Response`] can carry one directly, and
//! continuing a conversation is `messages.push(response.message)` with nothing in
//! between. A response shape of its own - a list of "choices", each with its own
//! content fields - cannot be appended to anything, which is why appending used to
//! mean rebuilding.
//!
//! # What is not a role
//!
//! [`Role`] has two variants, because a conversation has two speakers. System
//! instructions are not a turn and live on [`crate::Request::system`]; a tool
//! result is not a speaker and is a block inside the user turn that carries it.
//! Both were roles in the shape this replaced, and both produced the same
//! question at every call site: what does it mean when this one has tool calls?

use serde_json::Value;

/// Who produced a turn.
///
/// Closed on purpose. An unknown role string reaching the renderer is a request
/// that should have been refused at the edge, and the adapter that parsed it is
/// the only place that still knows which field and which surface to name in the
/// refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// A turn from the caller. Tool results ride here, in block order.
    User,
    /// A generated turn, whatever it contains.
    Assistant,
}

impl Role {
    /// The spelling both wire formats and every chat template use.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One turn of a conversation.
///
/// The same type in both directions: a request's history is a list of these and a
/// response carries one, so a generated turn goes back into the next request
/// unchanged.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Message {
    /// Who produced the turn.
    pub role: Role,
    /// Everything the turn contains, in the order it was produced.
    pub content: Vec<Block>,
}

impl Message {
    /// A turn with the given blocks.
    #[must_use]
    pub const fn new(role: Role, content: Vec<Block>) -> Self {
        Self { role, content }
    }

    /// A caller turn of plain text.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, vec![Block::Text(text.into())])
    }

    /// A generated turn of plain text.
    #[must_use]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(Role::Assistant, vec![Block::Text(text.into())])
    }

    /// The turn's visible text, blocks joined in order.
    ///
    /// Visible is the operative word: a thinking block is not text the caller
    /// asked for and a tool call is not prose, so neither appears here. A consumer
    /// that wants those reads the blocks.
    #[must_use]
    pub fn text(&self) -> String {
        let mut out = String::new();
        for block in &self.content {
            if let Block::Text(part) = block {
                out.push_str(part);
            }
        }
        out
    }

    /// The tool calls this turn makes, in block order.
    pub fn tool_uses(&self) -> impl Iterator<Item = &ToolUse> {
        self.content.iter().filter_map(|block| match block {
            Block::ToolUse(use_) => Some(use_),
            _ => None,
        })
    }

    /// The media this turn carries, tagged and in block order.
    ///
    /// Order is the whole point of returning a sequence rather than a pair of
    /// counts. A turn that sends a video and then an image renders two different
    /// markers, and a pair of counts agrees with itself while putting them in the
    /// wrong order - which is a defect nothing downstream can detect, because
    /// every number still adds up.
    pub fn media(&self) -> impl Iterator<Item = (MediaKind, &MediaSource)> {
        self.content.iter().filter_map(|block| match block {
            Block::Image(source) => Some((MediaKind::Image, source)),
            Block::Video(source) => Some((MediaKind::Video, source)),
            Block::Audio(source) => Some((MediaKind::Audio, source)),
            _ => None,
        })
    }

    /// Whether the turn contains nothing at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

/// One piece of a turn.
///
/// Every variant is content, and content is the only thing a turn has. Adding a
/// modality means adding a variant here rather than a field beside the list,
/// which is what keeps the ordering guarantee true as the list grows.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Block {
    /// Prose, from either speaker.
    Text(String),
    /// A reasoning trace, kept as data rather than as markers wrapped around the
    /// text. A trace that travels as `<think>...</think>` inside a text block has
    /// to be re-parsed by everything that reads the turn, and every consumer that
    /// forgets shows the trace to the caller.
    Thinking(Thinking),
    /// A call the model asked for.
    ToolUse(ToolUse),
    /// The answer to one call, in the turn that carries it back.
    ToolResult(ToolResult),
    /// A still image.
    Image(MediaSource),
    /// A video. Distinct from an image because the marker and the pad-token count
    /// differ, not because the bytes arrive differently.
    Video(MediaSource),
    /// A sound clip. The bytes are a container the decoder sniffs; the model
    /// hears them through its own audio tower.
    Audio(MediaSource),
    /// Output withheld by a safety decision, with the text shown instead.
    Refusal(String),
}

/// A reasoning trace.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Thinking {
    /// The trace, with no wire markers around it.
    pub text: String,
    /// The provider signature that lets a trace be replayed in a later turn, when
    /// one was issued.
    ///
    /// Opaque here on purpose: only the surface that issued it can check it, and
    /// nothing between the two is entitled to look.
    pub signature: Option<String>,
}

impl Thinking {
    /// A trace with no signature.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            signature: None,
        }
    }
}

/// A call the model asked for.
#[derive(Debug, Clone, PartialEq)]
///
/// Not `#[non_exhaustive]`, and the reason is the same everywhere it is missing
/// in this workspace: that attribute exists so a library can add a field without
/// breaking callers it cannot recompile. Every caller here is in this repository
/// and recompiles with it, and this is a type other crates have to BUILD - so all
/// the attribute would buy is that they cannot. A new field should break every
/// site that now has to supply it.
pub struct ToolUse {
    /// Identifier the matching result quotes back.
    pub id: String,
    /// The tool the model wants run.
    pub name: String,
    /// Arguments, already parsed. A string here would push a second parse - and
    /// its failure mode - into every consumer.
    pub input: Value,
}

impl ToolUse {
    /// A call with parsed arguments.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
        }
    }
}

/// The answer to one call.
#[derive(Debug, Clone, PartialEq)]
///
/// Not `#[non_exhaustive]`, and the reason is the same everywhere it is missing
/// in this workspace: that attribute exists so a library can add a field without
/// breaking callers it cannot recompile. Every caller here is in this repository
/// and recompiles with it, and this is a type other crates have to BUILD - so all
/// the attribute would buy is that they cannot. A new field should break every
/// site that now has to supply it.
pub struct ToolResult {
    /// The call this answers, by [`ToolUse::id`].
    pub tool_use_id: String,
    /// What the tool produced. Blocks, so a tool may answer with an image.
    pub content: Vec<Block>,
    /// Whether the tool failed.
    ///
    /// A field rather than a text prefix on the content: a prefix is a convention
    /// every reader has to know, and a reader that does not know it shows the
    /// model a failure that reads like a result.
    pub is_error: bool,
}

/// Which modality a media block is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    /// A still image.
    Image,
    /// A video.
    Video,
    /// A sound clip.
    Audio,
}

/// Where a media item's bytes come from.
///
/// One type for both modalities: the inline-or-remote distinction, the rule that
/// classifies a string, and the reason a remote address is refused by default are
/// identical for an image and a video. What differs is everything after the bytes
/// are in hand, which is why the *block* is per modality and this is not.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MediaSource {
    /// Bytes carried in the request, as the base64 the decoder reads.
    Inline(String),
    /// An address the server would have to fetch.
    ///
    /// Kept apart from inline bytes so a remote address can never reach the
    /// base64 decoder and be reported as a decode failure, which says nothing
    /// about what the caller actually did wrong.
    Remote(String),
}

impl MediaSource {
    /// Classifies a wire string: an http or https address is remote, and anything
    /// else - a data URI, raw base64 - is inline.
    #[must_use]
    pub fn classify(value: impl Into<String>) -> Self {
        let value = value.into();
        if value.starts_with("http://") || value.starts_with("https://") {
            Self::Remote(value)
        } else {
            Self::Inline(value)
        }
    }
}

impl ToolResult {
    /// A successful result.
    #[must_use]
    pub fn new(tool_use_id: impl Into<String>, content: Vec<Block>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content,
            is_error: false,
        }
    }

    /// A failed one.
    ///
    /// A constructor rather than a flag the caller sets afterwards: a tool
    /// failure that reaches the model looking like a result is the failure mode
    /// this field exists to prevent, and the two ways to build one should be as
    /// far apart as the two meanings.
    #[must_use]
    pub fn error(tool_use_id: impl Into<String>, content: Vec<Block>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content,
            is_error: true,
        }
    }
}
