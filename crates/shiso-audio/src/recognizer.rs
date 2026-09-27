//! The contract a speech recognizer meets and the server drives.
//!
//! A recognizer is a model, loaded on one device; a stream is one utterance
//! being recognised there. The split mirrors how the work is shared: the weights are loaded once
//! and read by every stream, while each stream owns its own caches and decoder
//! state and advances only when its caller pushes more audio. Offline
//! recognition is a stream fed everything at once and finished, so there is one
//! path to get right rather than two.
//!
//! The one exception is a recognizer that can decode several chunks at once
//! ([`SpeechRecognizer::batches_offline`]): a whole clip planned up front
//! ([`crate::chunks::plan_offline`]) lets it transcribe a batch of spans per
//! weight read instead of one. That path is an [`OfflineJob`], and a
//! recognizer that does not batch never sees it.
//!
//! Nothing here assumes a transducer, a CTC head or an attention decoder: a
//! stream turns mel frames into timed token ids, and how it does so is the
//! family's business.

use std::fmt;
use std::sync::Arc;

use shiso_core::TokenId;

use crate::chunks::Span;
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
    /// [`SpeechRecognizer::lookaheads`], or ignored by a recognizer that
    /// lists none.
    pub lookahead: usize,
    /// Mel frames at which a chunk must end, strictly increasing, from the
    /// start of the stream: the diarizer's speaker turns, say, as
    /// [`crate::chunks::turn_cuts`] plans them. `None` leaves the plan to the
    /// recognizer. A recognizer that does not chunk ignores it.
    pub cuts: Option<Vec<u32>>,
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

    /// Lookaheads its encoder was trained for. Empty means the recognizer has
    /// no lookahead: its streams ignore [`StreamOptions::lookahead`].
    fn lookaheads(&self) -> &[usize];

    /// Whether its streams transcribe in chunks and honour
    /// [`StreamOptions::cuts`]. When true, a diarized job is worth holding
    /// until its speaker turns are known, so that each chunk is one turn.
    fn chunks_at_cuts(&self) -> bool {
        false
    }

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

    /// How many spans one [`OfflineJob::step`] transcribes at most. Zero, the
    /// default, means the recognizer has no batched offline path: a caller
    /// streams instead and never calls [`Self::open_offline`].
    fn batches_offline(&self) -> usize {
        0
    }

    /// Starts transcribing a whole clip already planned into `spans`
    /// ([`crate::chunks::plan_offline`]), which tile its mel frames exactly.
    /// [`StreamOptions::cuts`] is ignored: the spans are the cuts. Only
    /// called when [`Self::batches_offline`] is above zero.
    ///
    /// # Errors
    ///
    /// By default, [`AudioError::BadSpec`]: the recognizer has no offline
    /// path. Otherwise, as [`Self::open_stream`].
    fn open_offline(
        &self,
        _options: &StreamOptions,
        _spans: Vec<Span>,
    ) -> Result<Box<dyn OfflineJob>, AudioError> {
        Err(AudioError::BadSpec {
            reason: "this recognizer has no offline path; stream the clip instead".into(),
        })
    }
}

/// One transcribed span of an offline job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanResult {
    /// The span, as planned.
    pub span: Span,
    /// Its tokens, in order. Their frames are the recognizer's frames
    /// ([`SpeechRecognizer::frame_secs`]) counted from the start of the
    /// clip, not of the span: for a recognizer whose frame is a mel frame,
    /// absolute mel frames.
    pub tokens: Vec<TokenEvent>,
    /// Where the span ends, in the same frames: what
    /// [`RecognizerStream::chunk_ends`] would have reported for it.
    pub chunk_end: u32,
    /// The language the recognizer found in its own output for this span,
    /// as [`RecognizerStream::language`].
    pub language: Option<String>,
}

/// The spans one [`OfflineJob::step`] hands back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflineBatch {
    /// Spans whose text is final, continuing the plan order from the last
    /// step's, each once over the job. Skipped spans (`transcribe: false`)
    /// never appear. A step may transcribe spans it does not hand back yet
    /// (a job may batch them out of order) and hand back none.
    pub spans: Vec<SpanResult>,
}

/// A whole clip being transcribed a batch of spans at a time.
pub trait OfflineJob: Send {
    /// Transcribes the next batch of up to
    /// [`SpeechRecognizer::batches_offline`] transcribed spans of `mel`, the
    /// whole clip's frames, the same on every call, and hands back the spans
    /// that are then final, in plan order.
    ///
    /// # Errors
    ///
    /// A device error, or [`AudioError::BadSpec`] for frames of the wrong
    /// width or fewer than the plan covers.
    // None: every transcribed span has been returned; the job is done.
    fn step(&mut self, mel: &MelFrames) -> Result<Option<OfflineBatch>, AudioError>;
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

    /// The language the stream found in the model's own output, for a model
    /// that names it in words rather than with a tag piece
    /// ([`PieceVocab::language_of`]). `None` leaves the transcript's language
    /// to the tags the detokenizer saw.
    fn language(&self) -> Option<&str> {
        None
    }

    /// The encoder frames at which the chunks transcribed so far end,
    /// increasing, for a recognizer that transcribes in chunks and places
    /// each token only somewhere inside its chunk: every token a chunk emits
    /// comes back from the same call that adds its end here. The transcript
    /// ends a chunk's last word at its end
    /// ([`crate::transcript::words_in_chunks`]). Empty for a recognizer
    /// whose token frames are its own: its words end one frame after their
    /// last piece.
    fn chunk_ends(&self) -> &[u32] {
        &[]
    }
}

#[cfg(test)]
#[path = "recognizer_tests.rs"]
mod tests;
