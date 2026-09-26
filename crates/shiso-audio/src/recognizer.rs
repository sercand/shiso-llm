//! The contract a speech recognizer meets and the server drives.
//!
//! A recognizer is a model, loaded on one device; a stream is one utterance
//! being recognised there. The split mirrors how the work is shared: the weights are loaded once
//! and read by every stream, while each stream owns its own caches and decoder
//! state and advances only when its caller pushes more audio. Offline
//! recognition is a stream fed everything at once and finished, so there is one
//! path to get right rather than two.
//!
//! Nothing here assumes a transducer, a CTC head or an attention decoder: a
//! stream turns mel frames into timed token ids, and how it does so is the
//! family's business.

use std::fmt;
use std::sync::Arc;

use shiso_core::TokenId;

use crate::error::AudioError;
use crate::mel::{MelFrames, MelSpec};
use crate::vocab::PieceVocab;

/// The language a stream is conditioned on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Language {
    /// Let the model decide, where it can.
    Auto,
    /// A locale tag the model's own dictionary resolves, such as `en-US` or `en`.
    Tag(String),
}

/// How one stream is to be run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamOptions {
    /// Language conditioning.
    pub language: Language,
    /// Encoder frames of future context each frame may see; one of
    /// [`SpeechRecognizer::lookaheads`].
    pub lookahead: usize,
}

/// One emitted token and the encoder frame it was emitted at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenEvent {
    /// The token.
    pub token: TokenId,
    /// Encoder frame index, from the start of the stream.
    pub frame: u32,
}

/// A loaded speech recognizer.
pub trait SpeechRecognizer: Send + Sync + fmt::Debug {
    /// The frontend its streams expect.
    fn mel_spec(&self) -> &MelSpec;

    /// The pieces its token ids index.
    fn vocab(&self) -> &Arc<PieceVocab>;

    /// Seconds per encoder frame.
    fn frame_secs(&self) -> f64;

    /// Locale tags its language conditioning accepts.
    fn languages(&self) -> Vec<String>;

    /// Lookaheads its encoder was trained for.
    fn lookaheads(&self) -> &[usize];

    /// Whether `id` is never text: blank, padding, unknown.
    fn is_special(&self, id: u32) -> bool;

    /// Starts a stream on the device the recognizer was loaded on. The weights
    /// live there, so a stream elsewhere is not something to ask for.
    ///
    /// # Errors
    ///
    /// [`AudioError::UnknownLanguage`], [`AudioError::BadLookahead`], or a
    /// device error while allocating the stream's state.
    fn open_stream(&self, options: &StreamOptions)
    -> Result<Box<dyn RecognizerStream>, AudioError>;
}

/// One utterance in progress.
pub trait RecognizerStream: Send {
    /// Adds mel frames (every valid row, in order) and returns the tokens that
    /// became final.
    ///
    /// # Errors
    ///
    /// A device error, or [`AudioError::BadSpec`] for frames of the wrong width.
    fn push(&mut self, mel: &MelFrames) -> Result<Vec<TokenEvent>, AudioError>;

    /// Flushes the tail exactly as a whole-utterance pass would, and returns the
    /// remaining tokens. The stream accepts nothing afterwards.
    ///
    /// # Errors
    ///
    /// A device error.
    fn finish(&mut self) -> Result<Vec<TokenEvent>, AudioError>;

    /// Encoder frames consumed so far.
    fn frames(&self) -> u32;
}

#[cfg(test)]
#[path = "recognizer_tests.rs"]
mod tests;
