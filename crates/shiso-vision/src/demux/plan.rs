//! Which samples an in-process decoder converts to RGB.
//!
//! The ffmpeg step samples with its own `fps=` filter. An in-process decoder
//! has to decode every sample anyway (later frames reference earlier ones),
//! but colour conversion is the expensive part of a kept frame and is wasted on
//! a dropped one, so the decision is made up front, per sample, from the
//! timestamps the container already gave.
//!
//! # Decode order is not display order
//!
//! Samples arrive in decode order; with B-frames a later-shown frame is
//! decoded first (`I P B B` decodes the `P` before the two `B`s it sits
//! after). The sampler's indices are positions on the DISPLAY timeline, so
//! they are mapped back through the presentation timestamps: the samples are
//! sorted by timestamp, the sampler picks display positions, and each picked
//! position marks the decode-order sample that lands there. Asking
//! [`SamplePlan::keep`] with a decode index therefore answers "is the frame
//! this sample produces one of the frames to keep", which is what a decoder
//! needs to know at the moment it is handed the sample.

use std::num::NonZeroU32;

use super::{DemuxedTrack, Pts, Sample};
use crate::image::widen;
use crate::video::{DEFAULT_MIN_FRAMES, sample_indices};

/// The samples to keep, by decode index.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplePlan {
    keep: Vec<bool>,
    kept: usize,
    native_fps: f64,
}

impl SamplePlan {
    /// Plans which of `track`'s samples to keep so the clip plays at
    /// `target_fps`, keeping at most `max_frames`.
    ///
    /// Uses the same sampler as every other path ([`sample_indices`]) with a
    /// temporal patch size of one: whole temporal groups are formed later, by
    /// [`crate::DecodedClip::into_sampled`], exactly as for the ffmpeg step's
    /// output, so both paths hand the caller frames at the target rate and the
    /// group rounding happens in one place.
    #[must_use]
    pub fn new(track: &DemuxedTrack<'_>, target_fps: f64, max_frames: usize) -> Self {
        let pts: Vec<Pts> = track.samples().iter().map(Sample::pts).collect();
        Self::from_pts(&pts, track.timescale(), target_fps, max_frames)
    }

    /// The plan for samples with presentation times `pts`, in decode order.
    pub(crate) fn from_pts(
        pts: &[Pts],
        timescale: NonZeroU32,
        target_fps: f64,
        max_frames: usize,
    ) -> Self {
        // Decode indices in display order. Equal timestamps (which a broken
        // file can declare) keep their decode order, so the mapping is total
        // and deterministic either way.
        let mut display: Vec<usize> = (0..pts.len()).collect();
        display.sort_by_key(|&index| (pts.get(index).copied(), index));

        let native_fps = native_fps(pts, timescale);
        let mut keep = vec![false; pts.len()];
        for position in sample_indices(
            pts.len(),
            native_fps,
            target_fps,
            DEFAULT_MIN_FRAMES,
            max_frames,
            1,
        ) {
            if let Some(slot) = display.get(position).and_then(|&d| keep.get_mut(d)) {
                *slot = true;
            }
        }
        let kept = keep.iter().filter(|&&k| k).count();
        Self {
            keep,
            kept,
            native_fps,
        }
    }

    /// Whether the frame decoded from sample `decode_index` (its position in
    /// [`DemuxedTrack::samples`]) is kept. False past the last sample.
    #[must_use]
    pub fn keep(&self, decode_index: usize) -> bool {
        self.keep.get(decode_index).copied().unwrap_or(false)
    }

    /// How many samples are kept.
    #[must_use]
    pub fn kept(&self) -> usize {
        self.kept
    }

    /// The track's frame rate as the timestamps give it, or zero when they
    /// give none (fewer than two samples, or every timestamp equal), which the
    /// sampler replaces with [`crate::DEFAULT_FPS`].
    #[must_use]
    pub fn native_fps(&self) -> f64 {
        self.native_fps
    }
}

/// Frames per second from the spread of presentation times: `n - 1` intervals
/// over the span from the first shown frame to the last.
///
/// From the span rather than from a per-frame duration because containers
/// state durations unreliably (Matroska blocks often carry none) while every
/// sample has a timestamp. Zero when there is no span to measure.
fn native_fps(pts: &[Pts], timescale: NonZeroU32) -> f64 {
    let (Some(first), Some(last)) = (pts.iter().min(), pts.iter().max()) else {
        return 0.0;
    };
    let span = i128::from(last.ticks()) - i128::from(first.ticks());
    let (Ok(span), Ok(intervals)) = (
        u64::try_from(span),
        u64::try_from(pts.len().saturating_sub(1)),
    ) else {
        return 0.0;
    };
    if span == 0 || intervals == 0 {
        return 0.0;
    }
    widen(intervals) * f64::from(timescale.get()) / widen(span)
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
