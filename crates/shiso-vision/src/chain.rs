//! The video decode chain: which decoders a request is offered to, in order.
//!
//! A build has some subset of three kinds of step - an OS decoder, a pure-Rust
//! decoder and an ffmpeg subprocess - and the chain offers the container to each
//! in turn. Only a typed [`Attempt::Unsupported`] moves the chain forward. An
//! error from a step (a malformed stream, a cap hit, a decoder that failed
//! mid-stream) ends the request, because retrying the same bytes on another
//! decoder would hide a corrupt upload behind whichever decoder happens to
//! tolerate it, and would spend the request's time budget twice.
//!
//! Every step shares one seam, [`VideoStep`], so the chain logic is tested with
//! fakes rather than with real decoders.

use std::num::{NonZeroU64, NonZeroUsize};

use crate::error::{Container, VisionError};
#[cfg(feature = "decode-ffmpeg")]
use crate::ffmpeg::FfmpegPolicy;
use crate::video::{DEFAULT_MAX_FRAMES, DecodedClip};

/// Wall-clock budget for one decode when the operator gave none.
const DEFAULT_TIMEOUT_SECS: u64 = 20;

/// Bytes of decoded RGB an in-process step may produce for one request: the
/// same bound the ffmpeg runner applies to its output pipe.
const DEFAULT_MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;

/// Operator policy for decoding video that is not a GIF.
///
/// There is deliberately no `enabled` flag: the disabled state is the ABSENCE
/// of a policy (`Option<&VideoDecodePolicy>` is [`None`]), so no value exists
/// whose other fields are meaningless and no path has to remember to check a
/// boolean before decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct VideoDecodePolicy {
    timeout_secs: NonZeroU64,
    max_frames: NonZeroUsize,
    max_output_bytes: NonZeroUsize,
    #[cfg(feature = "decode-ffmpeg")]
    ffmpeg: Option<FfmpegPolicy>,
}

impl VideoDecodePolicy {
    /// A policy with the given deadline and frame cap, and no ffmpeg step.
    #[must_use]
    pub fn new(timeout_secs: NonZeroU64, max_frames: NonZeroUsize) -> Self {
        Self {
            timeout_secs,
            max_frames,
            max_output_bytes: NonZeroUsize::new(DEFAULT_MAX_OUTPUT_BYTES)
                .unwrap_or(NonZeroUsize::MIN),
            #[cfg(feature = "decode-ffmpeg")]
            ffmpeg: None,
        }
    }

    /// Caps the decoded RGB bytes an in-process step may produce.
    #[must_use]
    pub fn with_max_output_bytes(mut self, bytes: NonZeroUsize) -> Self {
        self.max_output_bytes = bytes;
        self
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

    /// Frames one clip may contribute.
    #[must_use]
    pub fn max_frames(&self) -> usize {
        self.max_frames.get()
    }

    /// Decoded RGB bytes an in-process step may produce.
    #[must_use]
    pub fn max_output_bytes(&self) -> usize {
        self.max_output_bytes.get()
    }

    /// The ffmpeg step's policy, when the operator pointed the server at one.
    #[cfg(feature = "decode-ffmpeg")]
    #[must_use]
    pub fn ffmpeg(&self) -> Option<&FfmpegPolicy> {
        self.ffmpeg.as_ref()
    }
}

impl Default for VideoDecodePolicy {
    /// The default caps and no ffmpeg step.
    ///
    /// This is the shape of an ENABLED policy, not the default posture of the
    /// pipeline, which is to hold no policy at all.
    fn default() -> Self {
        Self::new(
            NonZeroU64::new(DEFAULT_TIMEOUT_SECS).unwrap_or(NonZeroU64::MIN),
            NonZeroUsize::new(DEFAULT_MAX_FRAMES).unwrap_or(NonZeroUsize::MIN),
        )
    }
}

/// What one step is asked to decode.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct StepInput<'a> {
    /// The request's container bytes, as received.
    pub bytes: &'a [u8],
    /// What the bytes' magic says they are.
    pub container: Container,
    /// The sampling rate the caller wants frames at.
    pub target_fps: f64,
    /// The operator's policy.
    pub policy: &'a VideoDecodePolicy,
}

impl<'a> StepInput<'a> {
    /// Describes one decode.
    #[must_use]
    pub fn new(
        bytes: &'a [u8],
        container: Container,
        target_fps: f64,
        policy: &'a VideoDecodePolicy,
    ) -> Self {
        Self {
            bytes,
            container,
            target_fps,
            policy,
        }
    }
}

/// What a step made of the input.
#[derive(Debug)]
pub enum Attempt {
    /// The step decoded it. The clip carries the rate its frames represent: the
    /// requested one for a step that samples while decoding at that rate, the
    /// rate actually kept for a step that picks frames by plan.
    Done(DecodedClip),
    /// The step cannot handle this input and says why. The chain moves on.
    Unsupported(String),
}

/// One decoder in the chain.
pub trait VideoStep {
    /// A short stable name for logs and the [`VisionError::NoVideoDecoder`]
    /// message, such as `ffmpeg`.
    fn name(&self) -> &'static str;

    /// Decodes `input` or declines it.
    ///
    /// # Errors
    ///
    /// Any [`VisionError`]: an error ends the chain, it is never retried on the
    /// next step. Only [`Attempt::Unsupported`] falls through.
    fn decode(&self, input: &StepInput<'_>) -> Result<Attempt, VisionError>;
}

/// Offers `input` to each step in order.
///
/// # Errors
///
/// The first step error, unchanged; [`VisionError::NoVideoDecoder`] listing
/// every step that declined (possibly none) when all of them did.
pub fn run_chain(
    steps: &[&dyn VideoStep],
    input: &StepInput<'_>,
) -> Result<DecodedClip, VisionError> {
    let mut tried = Vec::with_capacity(steps.len());
    for step in steps {
        match step.decode(input)? {
            Attempt::Done(clip) => return Ok(clip),
            Attempt::Unsupported(reason) => {
                tracing::debug!(step = step.name(), %reason, "decoder step declined");
                tried.push(step.name());
            }
        }
    }
    Err(VisionError::NoVideoDecoder {
        container: input.container,
        codec: None,
        tried,
    })
}

/// The ffmpeg subprocess, as a chain step.
#[cfg(feature = "decode-ffmpeg")]
#[derive(Debug)]
pub(crate) struct FfmpegStep<'a>(pub(crate) &'a FfmpegPolicy);

#[cfg(feature = "decode-ffmpeg")]
impl VideoStep for FfmpegStep<'_> {
    fn name(&self) -> &'static str {
        "ffmpeg"
    }

    fn decode(&self, input: &StepInput<'_>) -> Result<Attempt, VisionError> {
        let frames = crate::ffmpeg::decode_frames(input.bytes, input.target_fps, self.0)?;
        DecodedClip::new(frames, input.target_fps).map(Attempt::Done)
    }
}

#[cfg(test)]
#[path = "chain_tests.rs"]
mod tests;
