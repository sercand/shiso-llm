//! The centred frontend, fed incrementally.
//!
//! A streaming encoder consumes mel frames as audio arrives, and those frames
//! must be the ones an offline pass over the whole signal would have produced.
//! A frame at the stream's left edge sees the same zero padding, and every
//! other frame sees the same samples, because pre-emphasis carries its one
//! sample of memory across pushes. The arithmetic is the offline frontend's
//! own [`MelFrontend::frame_row`], so the equality is bitwise, not approximate.

use std::sync::Arc;

use crate::error::AudioError;
use crate::mel::{Framing, LogScale, MelFrames, MelFrontend, Preemphasis};

/// An incremental centred log-mel frontend.
#[derive(Debug)]
pub struct MelStream {
    frontend: Arc<MelFrontend>,
    /// Pre-emphasised signal from absolute padded index `base` onwards.
    buf: Vec<f64>,
    base: usize,
    prev: Option<f64>,
    seen: usize,
    next_frame: usize,
}

impl MelStream {
    /// Starts a stream.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadSpec`] unless the frontend uses centred framing.
    pub fn new(frontend: Arc<MelFrontend>) -> Result<Self, AudioError> {
        if frontend.spec().framing != Framing::Centred {
            return Err(AudioError::BadSpec {
                reason: "only centred framing streams".into(),
            });
        }
        if matches!(frontend.spec().log_scale, LogScale::WhisperClamp) {
            return Err(AudioError::BadSpec {
                reason: "this log scale needs every frame's value before it can floor and \
                         rescale any of them, so it cannot stream"
                    .into(),
            });
        }
        let left = frontend.spec().n_fft / 2;
        Ok(Self {
            frontend,
            buf: vec![0.0; left],
            base: 0,
            prev: None,
            seen: 0,
            next_frame: 0,
        })
    }

    /// Frames emitted so far.
    #[must_use]
    pub fn frames_emitted(&self) -> usize {
        self.next_frame
    }

    fn append(&mut self, samples: &[f32]) {
        for &x in samples {
            let x = f64::from(x);
            let y = match (self.frontend.spec().preemphasis, self.prev) {
                (Preemphasis::Signal(c), Some(p)) => x - f64::from(c) * p,
                _ => x,
            };
            self.prev = Some(x);
            self.buf.push(y);
        }
        self.seen += samples.len();
    }

    fn emit_until(&mut self, end_frame: usize) -> MelFrames {
        let spec = self.frontend.spec();
        let (hop, mels) = (spec.hop_length, spec.n_mels);
        let count = end_frame.saturating_sub(self.next_frame);
        let mut data = vec![0.0f32; count * mels];
        for (i, t) in (self.next_frame..end_frame).enumerate() {
            self.frontend.frame_row(
                &self.buf,
                t * hop - self.base,
                &mut data[i * mels..(i + 1) * mels],
            );
        }
        self.next_frame = end_frame;
        // Drop what no later frame reads.
        let keep_from = self.next_frame * hop;
        if keep_from > self.base {
            self.buf.drain(..keep_from - self.base);
            self.base = keep_from;
        }
        MelFrames::from_parts(data, mels, count, count)
    }

    /// Adds samples and returns the frames that became complete.
    pub fn push(&mut self, samples: &[f32]) -> MelFrames {
        self.append(samples);
        let spec = self.frontend.spec();
        let have = spec.n_fft / 2 + self.seen;
        let complete = if have < spec.n_fft {
            0
        } else {
            (have - spec.n_fft) / spec.hop_length + 1
        };
        let end = complete.min(self.seen / spec.hop_length);
        self.emit_until(end.max(self.next_frame))
    }

    /// Ends the stream and returns the remaining valid frames.
    #[must_use]
    pub fn finish(mut self) -> MelFrames {
        let spec = self.frontend.spec();
        let (right, valid) = (spec.n_fft / 2, self.seen / spec.hop_length);
        self.buf.extend(std::iter::repeat_n(0.0, right));
        self.emit_until(valid.max(self.next_frame))
    }
}

#[cfg(test)]
#[path = "mel_stream_tests.rs"]
mod tests;
