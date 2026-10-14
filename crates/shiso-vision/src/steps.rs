//! The in-process decode steps: the OS decoder and, with `decode-native`, the
//! AV1 decoder, both driven over a demuxed track.
//!
//! Both are the same loop with a different decoder behind it, so the loop is
//! written once over [`FrameSource`]. It decodes every sample (later frames may
//! reference earlier ones) but asks for RGB only for the frames the
//! [`SamplePlan`] keeps, checks the deadline and the output cap between
//! samples, and reports the rate the kept frames actually represent.
//!
//! A decoder's [`HwError::Unavailable`] and [`HwError::Unsupported`] decline the
//! step and let the chain move on. [`HwError::Failed`] ends the request: the
//! decoder was there and broke on this input.

use std::time::{Duration, Instant};

use shiso_media_sys::{HwCodec, HwDecoder, HwDecoderDesc, HwError, HwFrame, HwTimestamp};

use crate::chain::{Attempt, StepInput, VideoStep};
use crate::demux::{DemuxedTrack, SamplePlan, VideoCodec};
use crate::error::VisionError;
use crate::video::{DecodedClip, DecodedFrame};

/// What the shared loop needs from a decoder.
trait FrameSource {
    fn decode(
        &mut self,
        sample: &[u8],
        pts: HwTimestamp,
        keep: bool,
    ) -> Result<Vec<HwFrame>, HwError>;
    fn flush(&mut self) -> Result<Vec<HwFrame>, HwError>;
}

impl FrameSource for HwDecoder {
    fn decode(
        &mut self,
        sample: &[u8],
        pts: HwTimestamp,
        keep: bool,
    ) -> Result<Vec<HwFrame>, HwError> {
        HwDecoder::decode(self, sample, pts, keep)
    }

    fn flush(&mut self) -> Result<Vec<HwFrame>, HwError> {
        HwDecoder::flush(self)
    }
}

#[cfg(feature = "decode-native")]
impl FrameSource for shiso_media_sys::Av1Decoder {
    fn decode(
        &mut self,
        sample: &[u8],
        pts: HwTimestamp,
        keep: bool,
    ) -> Result<Vec<HwFrame>, HwError> {
        shiso_media_sys::Av1Decoder::decode(self, sample, pts, keep)
    }

    fn flush(&mut self) -> Result<Vec<HwFrame>, HwError> {
        shiso_media_sys::Av1Decoder::flush(self)
    }
}

/// The codec as the OS decoders name it.
fn hw_codec(codec: VideoCodec) -> HwCodec {
    match codec {
        VideoCodec::H264 => HwCodec::H264,
        VideoCodec::Hevc => HwCodec::Hevc,
        VideoCodec::Vp9 => HwCodec::Vp9,
        VideoCodec::Av1 => HwCodec::Av1,
    }
}

/// How a step ended: a clip, or a polite decline.
enum Drive {
    Frames(DecodedClip),
    Declined(String),
}

/// Maps a decoder error that ends the request.
fn failed(error: &HwError) -> VisionError {
    VisionError::DecoderFailed {
        detail: error.to_string(),
    }
}

/// Runs `source` over every sample of `track`.
fn drive(
    source: &mut dyn FrameSource,
    track: &DemuxedTrack<'_>,
    input: &StepInput<'_>,
) -> Result<Drive, VisionError> {
    let policy = input.policy;
    let plan = SamplePlan::new(track, input.target_fps, policy.max_frames());
    let deadline = Instant::now().checked_add(Duration::from_secs(policy.timeout_secs()));
    let mut frames: Vec<HwFrame> = Vec::with_capacity(plan.kept());
    let mut bytes = 0usize;
    let mut seen_first = false;

    for (index, sample) in track.samples().iter().enumerate() {
        if deadline.is_some_and(|at| Instant::now() >= at) {
            return Err(VisionError::DecoderTimeout {
                secs: policy.timeout_secs(),
            });
        }
        let data = track
            .sample_bytes(index)
            .ok_or_else(|| VisionError::DecoderFailed {
                detail: format!("sample {index} lies outside the container"),
            })?;
        let keep = plan.keep(index);
        match source.decode(data, HwTimestamp::new(sample.pts().ticks()), keep) {
            Ok(out) => {
                seen_first = true;
                bytes = take(&mut frames, out, bytes, policy.max_output_bytes())?;
            }
            // The decoder learns the stream's real format from the first sample,
            // so a refusal there is the same decline as one from `open`.
            Err(HwError::Unsupported { reason, .. } | HwError::Unavailable { reason })
                if !seen_first =>
            {
                return Ok(Drive::Declined(reason));
            }
            Err(error) => return Err(failed(&error)),
        }
    }
    let rest = source.flush().map_err(|error| failed(&error))?;
    take(&mut frames, rest, bytes, policy.max_output_bytes())?;

    frames.sort_by_key(HwFrame::pts);
    let kept = frames.len();
    let converted = frames
        .into_iter()
        .map(|frame| DecodedFrame::from_rgb8(frame.width(), frame.height(), frame.into_rgb()))
        .collect::<Result<Vec<_>, _>>()?;
    // The rate the kept frames represent over the clip's own duration, not the
    // requested one: the plan keeps at least a few frames of a short clip.
    let duration = widen(track.samples().len()) / plan.native_fps();
    let fps = if duration > 0.0 {
        widen(kept) / duration
    } else {
        input.target_fps
    };
    DecodedClip::new(converted, fps).map(Drive::Frames)
}

/// Adds `out` to `frames`, enforcing the output cap. Returns the new byte count.
fn take(
    frames: &mut Vec<HwFrame>,
    out: Vec<HwFrame>,
    mut bytes: usize,
    cap: usize,
) -> Result<usize, VisionError> {
    for frame in out {
        bytes = bytes.saturating_add(frame.rgb().len());
        if bytes > cap {
            return Err(VisionError::DecoderOutputCap { cap });
        }
        frames.push(frame);
    }
    Ok(bytes)
}

/// A count as a float for the rate arithmetic.
fn widen(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// The OS decoder (`VideoToolbox`, NVDEC) as a chain step.
#[derive(Debug)]
pub(crate) struct OsStep<'a, 'b>(pub(crate) &'a DemuxedTrack<'b>);

impl VideoStep for OsStep<'_, '_> {
    fn name(&self) -> &'static str {
        "os"
    }

    fn decode(&self, input: &StepInput<'_>) -> Result<Attempt, VisionError> {
        let track = self.0;
        let desc = HwDecoderDesc::new(
            hw_codec(track.codec()),
            track.config().record(),
            track.width(),
            track.height(),
        );
        let mut decoder = match HwDecoder::open(&desc) {
            Ok(decoder) => decoder,
            Err(HwError::Unavailable { reason } | HwError::Unsupported { reason, .. }) => {
                return Ok(Attempt::Unsupported(reason));
            }
            Err(error) => return Err(failed(&error)),
        };
        match drive(&mut decoder, track, input)? {
            Drive::Frames(clip) => Ok(Attempt::Done(clip)),
            Drive::Declined(reason) => Ok(Attempt::Unsupported(reason)),
        }
    }
}

/// The in-process AV1 decoder as a chain step.
#[cfg(feature = "decode-native")]
#[derive(Debug)]
pub(crate) struct Av1Step<'a, 'b>(pub(crate) &'a DemuxedTrack<'b>);

#[cfg(feature = "decode-native")]
impl VideoStep for Av1Step<'_, '_> {
    fn name(&self) -> &'static str {
        "rust"
    }

    fn decode(&self, input: &StepInput<'_>) -> Result<Attempt, VisionError> {
        let track = self.0;
        if track.codec() != VideoCodec::Av1 {
            return Ok(Attempt::Unsupported(format!(
                "the native step decodes AV1 only, not {}",
                track.codec()
            )));
        }
        let desc = HwDecoderDesc::new(
            HwCodec::Av1,
            track.config().record(),
            track.width(),
            track.height(),
        );
        let mut decoder = match shiso_media_sys::Av1Decoder::open(&desc) {
            Ok(decoder) => decoder,
            Err(HwError::Unavailable { reason } | HwError::Unsupported { reason, .. }) => {
                return Ok(Attempt::Unsupported(reason));
            }
            Err(error) => return Err(failed(&error)),
        };
        match drive(&mut decoder, track, input)? {
            Drive::Frames(clip) => Ok(Attempt::Done(clip)),
            Drive::Declined(reason) => Ok(Attempt::Unsupported(reason)),
        }
    }
}

/// A step for a container the demuxer declined: it names itself in the
/// refusal and moves the chain on to ffmpeg, if the build has it.
#[derive(Debug)]
pub(crate) struct DemuxDeclined(pub(crate) String);

impl VideoStep for DemuxDeclined {
    fn name(&self) -> &'static str {
        "demux"
    }

    fn decode(&self, _input: &StepInput<'_>) -> Result<Attempt, VisionError> {
        Ok(Attempt::Unsupported(self.0.clone()))
    }
}

#[cfg(test)]
#[path = "steps_tests.rs"]
mod tests;
