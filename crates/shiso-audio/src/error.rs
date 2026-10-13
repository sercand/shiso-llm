//! The one error type of the host audio pipeline.
//!
//! Every variant is something the server turns into a 400 naming what was
//! wrong, except [`AudioError::Io`], which is a local failure. None of them is
//! recovered from by substituting a default: a sample format this crate does
//! not read is refused, not guessed at.

use shiso_vision::VisionError;

/// Why audio could not be decoded, framed or detokenized.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AudioError {
    /// The bytes are a container no decoder in this build handled.
    ///
    /// WAV is always read; anything else needs the `decode-native` feature or,
    /// with `decode-ffmpeg`, an `--ffmpeg-path` the operator set.
    #[error(
        "no audio decoder in this build handled {container}{}; WAV is always accepted, \
         other containers need the `decode-native` feature or --ffmpeg-path",
        codec.as_deref().map(|c| format!(" ({c})")).unwrap_or_default()
    )]
    UnsupportedContainer {
        /// The sniffed container name, e.g. `mp3`.
        container: &'static str,
        /// The codec, when the container was read far enough to know it.
        codec: Option<String>,
    },
    /// A WAV file whose structure is broken.
    #[error("the WAV file is malformed: {reason}")]
    MalformedWav {
        /// What was wrong.
        reason: &'static str,
    },
    /// A WAV sample encoding this crate does not read.
    #[error("WAV format tag {tag:#06x} with {bits} bits per sample is not supported")]
    UnsupportedWavFormat {
        /// The format tag (after resolving `WAVE_FORMAT_EXTENSIBLE`).
        tag: u16,
        /// Bits per sample.
        bits: u16,
    },
    /// The audio holds no samples.
    #[error("the audio holds no samples")]
    Empty,
    /// The audio is longer than the operator allows.
    #[error("the audio is {seconds:.1} s long; this server accepts at most {max} s")]
    TooLong {
        /// The decoded duration, or a lower bound on it.
        seconds: f64,
        /// The configured cap.
        max: u32,
    },
    /// A sample rate of zero, or one the resampler cannot express.
    #[error("sample rate {rate} Hz is not usable")]
    BadRate {
        /// The offending rate.
        rate: u32,
    },
    /// The subprocess decoder failed; the inner error says how.
    #[error(transparent)]
    Decoder(#[from] VisionError),
    /// A frontend spec whose fields contradict each other.
    #[error("the mel frontend spec is inconsistent: {reason}")]
    BadSpec {
        /// Which constraint failed.
        reason: String,
    },
    /// A vocabulary that cannot be read or is not dense.
    #[error("the vocabulary is unusable: {reason}")]
    Vocab {
        /// What was wrong.
        reason: String,
    },
    /// A locale tag the model's language dictionary does not hold.
    #[error("language {tag:?} is not one this model recognises")]
    UnknownLanguage {
        /// The tag as the caller gave it.
        tag: String,
    },
    /// The request's context cannot be carried: too long, or not encodable.
    #[error("the context is unusable: {reason}")]
    BadContext {
        /// Why.
        reason: String,
    },
    /// A context tokenized to more ids than a recognizer's own prompt cap
    /// carries.
    ///
    /// A variant of its own rather than folded into [`Self::BadContext`]'s
    /// free-text `reason`: a caller in front of the encoder (the server) may
    /// enforce a lower cap of its own and needs the numbers to restate the
    /// refusal in its own terms instead of the recognizer's internal one.
    #[error("the context is {tokens} tokens; this model's prompt carries at most {max}")]
    TooManyContextTokens {
        /// How many ids the context tokenized to.
        tokens: usize,
        /// The recognizer's own cap.
        max: usize,
    },
    /// A lookahead the model was not trained for.
    #[error("lookahead {got} is not supported; this model offers {allowed:?}")]
    BadLookahead {
        /// What was asked for.
        got: usize,
        /// What the checkpoint lists.
        allowed: Vec<usize>,
    },
    /// A checkpoint that cannot be read as the model it claims to be.
    #[error("checkpoint: {what}")]
    Checkpoint {
        /// What was missing or malformed, naming the tensor or key.
        what: String,
    },
    /// The device failed.
    #[error(transparent)]
    Device(#[from] shiso_core::CoreError),
    /// A local read failed.
    #[error("{what}: {source}")]
    Io {
        /// What was being read.
        what: &'static str,
        /// The underlying error.
        source: std::io::Error,
    },
}
