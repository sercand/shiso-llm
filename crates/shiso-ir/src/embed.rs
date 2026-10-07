//! An embedding request and its answer.
//!
//! Separate from [`crate::Request`] because nothing about it is a
//! conversation: there are no turns, no roles, no sampling and no generated
//! message. What it shares with one is the content vocabulary - an input is a
//! list of [`Block`]s in reading order, so a picture, a clip or a run of text
//! means the same thing here as in a chat turn and is decoded by the same code.

use crate::message::Block;

/// One request: several inputs, each embedded on its own.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedRequest {
    /// The model the request named.
    pub model: String,
    /// The inputs, in the order their vectors are answered.
    pub inputs: Vec<EmbedInput>,
    /// The width the vectors are cut to, when the caller asked for one.
    pub dimensions: Option<u32>,
    /// How the vectors are written back.
    pub encoding: EmbedEncoding,
    /// The task the caller named (a search query, a document, ...), which
    /// selects a prompt the model was trained with. `None` embeds the input
    /// as it is.
    pub task: Option<String>,
}

/// One input.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbedInput {
    /// Content blocks in reading order: text and media, interleaved.
    Content(Vec<Block>),
    /// Token ids the caller already produced, embedded as the text they
    /// spell.
    Tokens(Vec<u32>),
}

/// How a vector is written on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbedEncoding {
    /// A list of numbers.
    #[default]
    Float,
    /// The little-endian `f32` bytes, base64-encoded.
    Base64,
}

/// Positions a request's inputs occupied, by what filled them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmbedUsage {
    /// Text tokens, including the model's own markers.
    pub text: u32,
    /// Picture soft tokens.
    pub image: u32,
    /// Video-frame soft tokens.
    pub video: u32,
    /// Audio soft tokens.
    pub audio: u32,
}

impl EmbedUsage {
    /// Every position.
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.text
            .saturating_add(self.image)
            .saturating_add(self.video)
            .saturating_add(self.audio)
    }

    /// How many modalities contributed at least one position.
    #[must_use]
    pub fn modalities(&self) -> usize {
        [self.text, self.image, self.video, self.audio]
            .iter()
            .filter(|n| **n > 0)
            .count()
    }
}

/// The answer: one vector per input, in input order.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedResponse {
    /// The model that produced the vectors, as the request named it.
    pub model: String,
    /// The vectors, unit length, each at the width the request asked for.
    pub vectors: Vec<Vec<f32>>,
    /// The encoding the request asked for.
    pub encoding: EmbedEncoding,
    /// Positions across every input.
    pub usage: EmbedUsage,
}
