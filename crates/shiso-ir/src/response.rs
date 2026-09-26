//! What one request produced.
//!
//! A [`Response`] is a [`Message`] plus the accounting around it. That is the
//! whole shape, and it is the point: continuing a conversation is
//! `request.messages.push(response.message)`, with no conversion step and nothing
//! that can be dropped on the way.
//!
//! # No list of alternatives
//!
//! One request, one turn. A response that carried a list of candidate turns would
//! push the question "which one" into every consumer - the renderer, the
//! accounting, the code that appends to a history - and every one of them would
//! answer "the first". A surface whose wire format asks for several runs one
//! request per run and assembles the list itself, which is the only place that
//! knows how its own wire format wants them numbered.

use crate::message::Message;

/// A complete generated turn, with its accounting.
#[derive(Debug, Clone, PartialEq)]
///
/// Not `#[non_exhaustive]`, and the reason is the same everywhere it is missing
/// in this workspace: that attribute exists so a library can add a field without
/// breaking callers it cannot recompile. Every caller here is in this repository
/// and recompiles with it, and this is a type other crates have to BUILD - so all
/// the attribute would buy is that they cannot. A new field should break every
/// site that now has to supply it.
pub struct Response {
    /// Identifier for this response, without any surface's prefix.
    ///
    /// Bare because each wire format spells it differently, and a value that
    /// already carried one surface's prefix would have to be un-prefixed by the
    /// next.
    pub id: String,
    /// The model that produced it, as the request named it.
    pub model: String,
    /// Seconds since the epoch at the moment the response was completed.
    pub created: u64,
    /// The generated turn, ready to be appended to the conversation that asked
    /// for it.
    pub message: Message,
    /// Why generation ended.
    pub stop: StopReason,
    /// What the request cost.
    pub usage: Usage,
    /// Per-token probabilities, when the caller asked for them.
    pub logprobs: Option<Logprobs>,
    /// The sampled ids that produced the turn, when the caller asked for them.
    ///
    /// Empty otherwise, rather than optional: "the caller did not ask" and "the
    /// turn was empty" lead to the same encoding, and a second state for them
    /// would have to be kept in step with the message for no reader's benefit.
    /// The count matches [`Usage::completion_tokens`] whenever it is non-empty,
    /// which is what lets a streaming caller and a blocking one account for the
    /// same generation the same way.
    pub token_ids: Vec<u32>,
}

impl Response {
    /// The generated turn, for a caller that is appending it to a conversation.
    ///
    /// The method exists to make the round trip the obvious thing to reach for -
    /// the field is public and does the same - and to be the place the doc for it
    /// lives.
    #[must_use]
    pub fn into_message(self) -> Message {
        self.message
    }
}

/// Why generation ended.
///
/// Closed, and every variant means something different to a caller deciding what
/// to do next. The shape this replaced carried an open string plus a separate
/// field holding the stop sequence that matched, which made "which sequence
/// stopped it" a question you could ask about a response that did not stop on a
/// sequence at all.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StopReason {
    /// The model finished its turn.
    EndTurn,
    /// The token budget ran out. The turn is truncated and continuing it is the
    /// caller's decision.
    MaxTokens,
    /// One of the caller's stop sequences was produced. It is carried here rather
    /// than beside it, so it cannot be present on a response that stopped some
    /// other way.
    StopSequence(String),
    /// The model ended its turn to call a tool. The calls are blocks of
    /// [`Response::message`].
    ToolUse,
    /// A safety decision withheld the output; the text shown instead is a refusal
    /// block of the message.
    Refusal,
    /// The server's deadline cut generation short.
    ///
    /// Deliberately not folded into [`Self::MaxTokens`]: a caller that cannot
    /// tell a deadline from a budget silently loses output and has no way to
    /// learn that it did. Each surface decides how to spell it, which is where
    /// the compatibility cost of a non-standard value belongs.
    Timeout,
}

/// What a request cost.
///
/// Every counter is tokens except the two at the end, which are the timings a
/// caller needs to tell a slow model from a slow queue.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
///
/// Not `#[non_exhaustive]`. A `Default` alone does not make a type an adapter can
/// fill in: from another crate the only way to use one is to default it and then
/// assign, which is exactly the pattern the lints here reject. Every caller is in
/// this repository and recompiles, so the attribute would buy nothing but that
/// fight.
pub struct Usage {
    /// Tokens in the rendered prompt.
    pub prompt_tokens: u64,
    /// Tokens generated, thinking included.
    pub completion_tokens: u64,
    /// Prompt tokens answered from the cache rather than computed.
    pub cached_prompt_tokens: u64,
    /// Prompt tokens that are audio soft tokens: the pad runs of the clips.
    pub audio_prompt_tokens: u64,
    /// Generated tokens that were part of a thinking block.
    pub reasoning_tokens: u64,
    /// Speculated tokens the verify step accepted. Zero when speculation is off.
    pub accepted_prediction_tokens: u64,
    /// Milliseconds from accepting the request to its first generated token.
    pub time_to_first_token_ms: f64,
    /// Tokens a second after the first one.
    pub tokens_per_second: f64,
}

/// Per-token probabilities for one generated turn.
#[derive(Debug, Clone, PartialEq)]
///
/// Not `#[non_exhaustive]`, and the reason is the same everywhere it is missing
/// in this workspace: that attribute exists so a library can add a field without
/// breaking callers it cannot recompile. Every caller here is in this repository
/// and recompiles with it, and this is a type other crates have to BUILD - so all
/// the attribute would buy is that they cannot. A new field should break every
/// site that now has to supply it.
pub struct Logprobs {
    /// One entry per generated token, in order.
    pub tokens: Vec<TokenLogprob>,
}

/// The probability report for one generated token.
#[derive(Debug, Clone, PartialEq)]
///
/// Not `#[non_exhaustive]`, and the reason is the same everywhere it is missing
/// in this workspace: that attribute exists so a library can add a field without
/// breaking callers it cannot recompile. Every caller here is in this repository
/// and recompiles with it, and this is a type other crates have to BUILD - so all
/// the attribute would buy is that they cannot. A new field should break every
/// site that now has to supply it.
pub struct TokenLogprob {
    /// The token's text, as the vocabulary spells it.
    pub token: String,
    /// The log-probability the model gave the token it chose.
    pub logprob: f32,
    /// The alternatives it did not choose, most likely first.
    pub top: Vec<Alternative>,
}

/// One token the model considered and did not emit.
#[derive(Debug, Clone, PartialEq)]
///
/// Not `#[non_exhaustive]`, and the reason is the same everywhere it is missing
/// in this workspace: that attribute exists so a library can add a field without
/// breaking callers it cannot recompile. Every caller here is in this repository
/// and recompiles with it, and this is a type other crates have to BUILD - so all
/// the attribute would buy is that they cannot. A new field should break every
/// site that now has to supply it.
pub struct Alternative {
    /// The token's text.
    pub token: String,
    /// Its log-probability.
    pub logprob: f32,
}
