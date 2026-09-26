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
    /// The bytes are a container this crate does not parse and no subprocess
    /// decoder was configured.
    #[error(
        "{container} audio needs the subprocess decoder; this server was started without \
         --audio-allow-ffmpeg"
    )]
    NeedsDecoder {
        /// The sniffed container name, e.g. `mp3`.
        container: &'static str,
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
