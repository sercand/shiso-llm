//! What the caller asked for.
//!
//! A [`Request`] is the whole of what generation reads. Each API surface parses
//! its own wire body into one, and nothing downstream knows which surface the
//! value came from.
//!
//! # What is deliberately absent
//!
//! Fields a surface only echoes back. A request carries what generation *reads*;
//! a value that travels in and out again without changing a token is the parsing
//! surface's business, and it keeps its parsed wire body to echo from. Putting
//! those here would mean every new surface widens a type that generation never
//! consults.
//!
//! # Why every sampling field is optional
//!
//! `None` means the caller said nothing, which is different from every value the
//! caller could have said - including zero. The server's resolved preset fills the
//! gaps, and it can only do that if "silent" survives as its own state. The same
//! rule is why there is no `From` impl from this type into the engine's sampling
//! record: that conversion needs the preset and the seed source as inputs, and a
//! `From` impl has nowhere to take them from, so it would have to invent
//! defaults. Both mistakes produce plausible output rather than a compile error.

use serde_json::Value;

use crate::message::{Block, Message};
use crate::tools::{RepetitionDetection, ToolChoice, ToolDefinition};

/// One generation request.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Request {
    /// The model the caller named.
    pub model: String,
    /// Standing instructions for the whole conversation.
    ///
    /// Not a turn. Both wire formats this serves carry instructions beside the
    /// conversation rather than inside it, every chat template renders them in a
    /// place of their own, and treating them as a turn meant every consumer had
    /// to ask whether the first message was really a message.
    pub system: Vec<Block>,
    /// The conversation so far, oldest first.
    pub messages: Vec<Message>,
    /// Tools the caller is offering. Empty means none.
    pub tools: Vec<ToolDefinition>,
    /// How the caller wants them used. `None` defers to the server.
    pub tool_choice: Option<ToolChoice>,
    /// Sampling knobs the caller set.
    pub sampling: Sampling,
    /// Cap on generated tokens, already resolved against the server's own limit.
    pub max_tokens: u32,
    /// Floor on generated tokens. Zero means no floor.
    pub min_tokens: u32,
    /// Sequences that end generation when produced.
    pub stop: Vec<String>,
    /// Stream the turn as it is produced instead of answering once.
    pub stream: bool,
    /// Constraint on the shape of the output. `None` is unconstrained prose.
    pub response_format: Option<ResponseFormat>,
    /// What the caller said about thinking.
    pub thinking: ThinkingDirective,
    /// Whether to keep historical thinking blocks when re-rendering the
    /// conversation. `None` leaves the decision to the model's own template.
    pub preserve_thinking: Option<bool>,
    /// Per-request override for the repeated-output detector.
    pub repetition_detection: Option<RepetitionDetection>,
    /// Additive bias per token id, already parsed.
    pub logit_bias: Vec<(u32, f32)>,
    /// How many alternatives to report per generated token. `None` reports none.
    pub top_logprobs: Option<u8>,
    /// Seed for reproducible sampling. `None` means fresh entropy, which is a
    /// decision the server makes rather than a zero this type could carry.
    pub seed: Option<u64>,
    /// Deadline in seconds. `None` takes the server's.
    pub timeout_secs: Option<f32>,
    /// Report the sampled token ids alongside the text.
    pub return_token_ids: bool,
    /// Which class of service the caller asked for.
    ///
    /// Here rather than left on the wire body because the scheduler reads it:
    /// it decides whether this request's rows run at all while other work is
    /// resident, which is as much a part of generating it as its budget.
    pub service_class: ServiceClass,
}

impl Request {
    /// A request for `model` over `messages`, with everything else unset.
    ///
    /// Every unset field means "the server decides", which is what makes this a
    /// usable starting point rather than a set of hidden defaults: the values the
    /// server fills in are the ones it would have filled in for a caller that
    /// named none of them.
    #[must_use]
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            system: Vec::new(),
            messages,
            tools: Vec::new(),
            tool_choice: None,
            sampling: Sampling::default(),
            max_tokens: 0,
            min_tokens: 0,
            stop: Vec::new(),
            stream: false,
            response_format: None,
            thinking: ThinkingDirective::Unspecified,
            preserve_thinking: None,
            repetition_detection: None,
            logit_bias: Vec::new(),
            top_logprobs: None,
            seed: None,
            timeout_secs: None,
            return_token_ids: false,
            service_class: ServiceClass::Standard,
        }
    }

    /// Appends a generated turn, so the next request continues this conversation.
    ///
    /// This is the round trip the whole message shape exists for: the turn that
    /// comes back from generation is the turn that goes into the next request,
    /// with no conversion and nothing dropped.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }
}

/// The sampling knobs a caller may set.
///
/// `None` is "the caller said nothing" everywhere, and it is never the same as a
/// value: a caller that sets `temperature: 0.0` is asking for greedy decoding, and
/// a caller that sets nothing is asking for whatever the server prefers.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
///
/// Not `#[non_exhaustive]`. A `Default` alone does not make a type an adapter can
/// fill in: from another crate the only way to use one is to default it and then
/// assign, which is exactly the pattern the lints here reject. Every caller is in
/// this repository and recompiles, so the attribute would buy nothing but that
/// fight.
pub struct Sampling {
    /// Softmax temperature.
    pub temperature: Option<f32>,
    /// Candidates kept by the top-k stage.
    pub top_k: Option<u32>,
    /// Cumulative-probability cutoff.
    pub top_p: Option<f32>,
    /// Cutoff expressed in standard deviations of the logit distribution rather
    /// than in probability mass.
    pub top_n_sigma: Option<f32>,
    /// Floor on a token's probability, as a fraction of the best token's.
    pub min_p: Option<f32>,
    /// Multiplicative penalty on tokens already in the context.
    pub repetition_penalty: Option<f32>,
    /// Additive penalty for a token having appeared at all.
    pub presence_penalty: Option<f32>,
    /// Additive penalty scaling with how often a token has appeared.
    pub frequency_penalty: Option<f32>,
}

/// A constraint on the shape of the generated text.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ResponseFormat {
    /// Valid JSON, any shape.
    JsonObject,
    /// JSON matching a schema.
    JsonSchema {
        /// The schema's name, for logs and errors.
        name: String,
        /// The schema, kept raw so the caller's own spelling reaches the decoder.
        schema: Value,
        /// Whether the decoder must refuse anything the schema does not allow.
        strict: bool,
    },
}

/// How the scheduler treats a request while other work is resident.
///
/// Two classes and no numeric priority. What a flex caller buys is the whole
/// rule - its rows run only when no standard request has work - and a priority
/// number would invite a ladder of classes whose interleaving nobody asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ServiceClass {
    /// Served as it arrives, sharing the device with every other standard
    /// request.
    #[default]
    Standard,
    /// Served only when no standard request has work: admitted when the
    /// standard traffic has drained, and paused, with its state kept, whenever
    /// a standard request arrives. It waits rather than being refused.
    Flex,
}

/// What the caller said about thinking.
///
/// Every wire format spells this differently and some spell it twice. The edge
/// resolves all of them into this one directive, and nothing downstream reads a
/// wire field. The directive never encodes a server or model default: folding
/// those in is a later step, and it can only happen correctly while
/// [`Self::Unspecified`] is still distinguishable from an explicit choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ThinkingDirective {
    /// The caller said nothing.
    #[default]
    Unspecified,
    /// The caller turned thinking off.
    Off,
    /// The caller turned thinking on, with an explicit token budget or without
    /// one. `None` defers to the model's own cap rather than to a fixed number: a
    /// cap that is too low injects an end-of-thinking marker mid-reasoning, and
    /// the turn that follows it is worse than no thinking at all.
    On {
        /// The budget in tokens, when the caller gave one.
        budget: Option<u32>,
    },
    /// The caller asked for a level of effort rather than a budget.
    ///
    /// Kept symbolic all the way to budget resolution, because the number a level
    /// is worth depends on the model's own cap. Resolving it at the edge is how a
    /// middle rung becomes a fixed token count that no server setting can reach.
    Effort(Effort),
}

impl ThinkingDirective {
    /// Whether the caller stated any intent at all, on or off.
    ///
    /// Server-side policy applies only when they did not, so "off" and "silent"
    /// have to stay distinguishable right up to the point policy is applied.
    #[must_use]
    pub const fn is_explicit(self) -> bool {
        !matches!(self, Self::Unspecified)
    }
}

/// How hard the caller asked the model to think.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effort {
    /// The rung below [`Self::Low`].
    Minimal,
    /// The least thinking above none.
    Low,
    /// The middle rung, and what most callers ask for.
    Medium,
    /// The second-highest rung. Distinct from [`Self::XHigh`] because the budget
    /// differs even where a template renders the two the same way.
    High,
    /// The top rung.
    XHigh,
}

impl Effort {
    /// The spelling a chat template accepts.
    ///
    /// The top rung renders as `xhigh` rather than `max`: templates validate this
    /// value against a fixed list and refuse a request that uses a word they do
    /// not know, so the spelling is part of the contract rather than a label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Minimal | Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }

    /// Parses the wire vocabulary, which every surface shares.
    ///
    /// `"none"` is not a level: it turns thinking off, which is a different
    /// answer, so it is [`ThinkingDirective::Off`] rather than a rung. `None` here
    /// means the spelling is unknown, and the caller must refuse the request
    /// rather than pick a rung - an unknown spelling that silently became the top
    /// rung is the most expensive possible guess.
    #[must_use]
    pub fn parse(value: &str) -> Option<ThinkingDirective> {
        Some(match value {
            "none" => ThinkingDirective::Off,
            "minimal" => ThinkingDirective::Effort(Self::Minimal),
            "low" => ThinkingDirective::Effort(Self::Low),
            "medium" => ThinkingDirective::Effort(Self::Medium),
            "high" => ThinkingDirective::Effort(Self::High),
            "xhigh" | "max" => ThinkingDirective::Effort(Self::XHigh),
            _ => return None,
        })
    }
}
