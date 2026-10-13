//! Request bytes to model-rate mono.
//!
//! WAV is parsed here, and everything else goes down the decoders this build
//! has, in order: the pure-Rust ones (`decode-native`), then the ffmpeg
//! subprocess when the operator named a binary. Either way the result is capped
//! in duration BEFORE resampling where the decoder allows it, so an hour-long
//! upload to a server that accepts ten minutes costs a parse, not a resample.

#[cfg(feature = "decode-ffmpeg")]
use std::num::NonZeroUsize;
use std::num::{NonZeroU32, NonZeroU64};

#[cfg(feature = "decode-ffmpeg")]
use shiso_vision::{FfmpegPolicy, VisionError, run_decoder};

use crate::error::AudioError;
use crate::num::f64_of;
use crate::pcm::{MODEL_RATE, Pcm};
use crate::resample::resample;
use crate::wav::{is_wav, parse_wav};

/// Operator caps on one decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    max_seconds: NonZeroU32,
}

impl DecodeLimits {
    /// Caps a decode at `max_seconds` of audio.
    #[must_use]
    pub const fn new(max_seconds: NonZeroU32) -> Self {
        Self { max_seconds }
    }

    /// The cap in seconds.
    #[must_use]
    pub const fn max_seconds(&self) -> u32 {
        self.max_seconds.get()
    }
}

/// Wall-clock budget for one decode when the operator gave none.
const DEFAULT_TIMEOUT: NonZeroU64 = NonZeroU64::MIN.saturating_add(19);

/// Operator policy for decoding audio that is not WAV.
///
/// Unlike video there is no opt-in flag: any container the build can decode
/// without a subprocess is accepted. The ffmpeg step is the one thing that
/// stays opt-in, because it exists only when the operator named a binary.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AudioDecodePolicy {
    timeout_secs: NonZeroU64,
    #[cfg(feature = "decode-ffmpeg")]
    ffmpeg: Option<FfmpegPolicy>,
}

impl AudioDecodePolicy {
    /// The default deadline and no ffmpeg step, as a constant so a caller can
    /// hold a `static` of it.
    pub const DEFAULT: Self = Self::new(DEFAULT_TIMEOUT);

    /// A policy with the given decode deadline and no ffmpeg step.
    #[must_use]
    pub const fn new(timeout_secs: NonZeroU64) -> Self {
        Self {
            timeout_secs,
            #[cfg(feature = "decode-ffmpeg")]
            ffmpeg: None,
        }
    }

    /// Adds the ffmpeg step, run under `policy`'s own caps.
    #[cfg(feature = "decode-ffmpeg")]
    #[must_use]
    pub fn with_ffmpeg(mut self, policy: FfmpegPolicy) -> Self {
        self.ffmpeg = Some(policy);
        self
    }

    /// Wall-clock budget for one decode, in seconds.
    #[must_use]
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs.get()
    }

    /// The ffmpeg step's policy, when the operator pointed the server at one.
    #[cfg(feature = "decode-ffmpeg")]
    #[must_use]
    pub fn ffmpeg(&self) -> Option<&FfmpegPolicy> {
        self.ffmpeg.as_ref()
    }
}

impl Default for AudioDecodePolicy {
    /// The default deadline and no ffmpeg step: WAV, plus whatever the build
    /// decodes natively.
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Names the container `bytes` start like. Used for the refusal message and
/// the log; the subprocess decoder probes the stream itself.
#[must_use]
pub fn sniff_container(bytes: &[u8]) -> &'static str {
    match bytes {
        _ if is_wav(bytes) => "wav",
        [b'f', b'L', b'a', b'C', ..] => "flac",
        [b'O', b'g', b'g', b'S', ..] => "ogg",
        [b'I', b'D', b'3', ..] => "mp3",
        [0xFF, second, ..] if second & 0xE0 == 0xE0 => "mp3",
        [_, _, _, _, b'f', b't', b'y', b'p', ..] => "mp4",
        [0x1A, 0x45, 0xDF, 0xA3, ..] => "webm",
        _ => "unknown",
    }
}

fn too_long(pcm: &Pcm, limits: DecodeLimits) -> Result<(), AudioError> {
    let seconds = pcm.duration_secs();
    if seconds > f64::from(limits.max_seconds()) {
        return Err(AudioError::TooLong {
            seconds,
            max: limits.max_seconds(),
        });
    }
    Ok(())
}

/// Decodes request bytes to [`MODEL_RATE`] mono.
///
/// # Errors
///
/// - [`AudioError::Empty`] for no bytes or no samples.
/// - [`AudioError::UnsupportedContainer`] for a non-WAV container no decoder in
///   this build handled.
/// - [`AudioError::TooLong`] past the cap.
/// - The WAV reader's refusals.
/// - [`AudioError::Decoder`] carrying the subprocess decoder's own failure.
pub fn decode_audio(
    bytes: &[u8],
    limits: &DecodeLimits,
    policy: &AudioDecodePolicy,
) -> Result<Pcm, AudioError> {
    if bytes.is_empty() {
        return Err(AudioError::Empty);
    }
    let container = sniff_container(bytes);
    let pcm = if container == "wav" {
        let native = parse_wav(bytes)?;
        too_long(&native, *limits)?;
        let rate = native.rate();
        Pcm::new(resample(native.samples(), rate, MODEL_RATE)?, MODEL_RATE)?
    } else {
        decode_with_steps(bytes, container, *limits, policy)?
    };
    if pcm.samples().is_empty() {
        return Err(AudioError::Empty);
    }
    too_long(&pcm, *limits)?;
    tracing::debug!(
        container,
        seconds = f64_of(pcm.samples().len()) / f64::from(MODEL_RATE),
        "decoded audio"
    );
    Ok(pcm)
}

/// Offers a non-WAV container to each decoder this build has, in order.
fn decode_with_steps(
    bytes: &[u8],
    container: &'static str,
    limits: DecodeLimits,
    policy: &AudioDecodePolicy,
) -> Result<Pcm, AudioError> {
    #[cfg(feature = "decode-ffmpeg")]
    if let Some(ffmpeg) = policy.ffmpeg() {
        return decode_ffmpeg(bytes, limits, ffmpeg);
    }
    let _ = (bytes, limits, policy);
    Err(AudioError::UnsupportedContainer {
        container,
        codec: None,
    })
}

/// The ffmpeg step: the container goes to the subprocess and raw `f32` samples
/// come back at the model rate.
#[cfg(feature = "decode-ffmpeg")]
fn decode_ffmpeg(
    bytes: &[u8],
    limits: DecodeLimits,
    policy: &FfmpegPolicy,
) -> Result<Pcm, AudioError> {
    // Bound the decoder's output by the duration cap plus one second, so an
    // overlong upload is stopped by the decoder rather than read in full.
    let max = limits.max_seconds();
    let cap_bytes =
        usize::try_from((u64::from(max) + 1) * u64::from(MODEL_RATE) * 4).unwrap_or(usize::MAX);
    let capped = policy.clone().with_max_output_bytes(
        NonZeroUsize::new(cap_bytes.min(policy.max_output_bytes())).unwrap_or(NonZeroUsize::MIN),
    );
    let limit = (max + 1).to_string();
    let raw = run_decoder(
        &capped,
        &[
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-t",
            &limit,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
        ],
        bytes,
    )
    .map_err(|error| match error {
        VisionError::DecoderOutputCap { .. } => AudioError::TooLong {
            seconds: f64::from(max) + 1.0,
            max,
        },
        other => AudioError::Decoder(other),
    })?;
    let samples: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Pcm::new(samples, MODEL_RATE)
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod tests;
