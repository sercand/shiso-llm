//! Display-order release of decoded frames.
//!
//! A decoder hands frames back in the order it finishes them, which for a
//! stream with B-frames is not the order they are shown. [`Reorder`] holds the
//! last few decoded frames and releases the earliest one only once `depth`
//! later frames have been decoded, so a frame whose presentation time is still
//! to come can still slot in front of it. `flush` releases the rest.
//!
//! Frames the caller will not keep take part in the ordering (as a timestamp
//! with no pixels) because a kept frame behind them must still wait for them.

use crate::{HwError, HwFrame, HwTimestamp};

/// Frames held for display-order release.
#[derive(Debug)]
pub(super) struct Reorder {
    depth: usize,
    pending: Vec<(HwTimestamp, Option<HwFrame>)>,
    released: Option<HwTimestamp>,
}

impl Reorder {
    /// A buffer that releases a frame once `depth` later ones are decoded.
    pub(super) fn new(depth: usize) -> Self {
        Self {
            depth,
            pending: Vec::new(),
            released: None,
        }
    }

    /// Adds a decoded frame; `frame` is `None` for one the caller drops.
    pub(super) fn push(&mut self, pts: HwTimestamp, frame: Option<HwFrame>) {
        self.pending.push((pts, frame));
    }

    /// Releases every frame that can no longer be overtaken, in display order.
    ///
    /// # Errors
    ///
    /// [`HwError::Failed`] when a frame arrives behind one already released:
    /// the stream reorders deeper than `depth`, and handing the frame back out
    /// of order would break the display-order promise silently.
    pub(super) fn ready(&mut self) -> Result<Vec<HwFrame>, HwError> {
        let mut out = Vec::new();
        while self.pending.len() > self.depth {
            self.release_earliest(&mut out)?;
        }
        Ok(out)
    }

    /// Releases everything, in display order.
    ///
    /// # Errors
    ///
    /// As [`Reorder::ready`].
    pub(super) fn drain(&mut self) -> Result<Vec<HwFrame>, HwError> {
        let mut out = Vec::new();
        while !self.pending.is_empty() {
            self.release_earliest(&mut out)?;
        }
        Ok(out)
    }

    fn release_earliest(&mut self, out: &mut Vec<HwFrame>) -> Result<(), HwError> {
        let Some(at) = self
            .pending
            .iter()
            .enumerate()
            .min_by_key(|(_, (pts, _))| *pts)
            .map(|(at, _)| at)
        else {
            return Ok(());
        };
        let (pts, frame) = self.pending.remove(at);
        if let Some(last) = self.released
            && pts <= last
        {
            return Err(HwError::Failed {
                detail: format!(
                    "frame at pts {} decoded after pts {} was already returned: the stream \
                     reorders deeper than {} frames, or repeats a timestamp",
                    pts.ticks(),
                    last.ticks(),
                    self.depth
                ),
            });
        }
        self.released = Some(pts);
        out.extend(frame);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(pts: i64) -> HwFrame {
        HwFrame::new(1, 1, vec![0; 3], HwTimestamp::new(pts)).expect("1x1")
    }

    fn ticks(frames: &[HwFrame]) -> Vec<i64> {
        frames.iter().map(|f| f.pts().ticks()).collect()
    }

    #[test]
    fn releases_in_display_order_once_depth_is_exceeded() {
        let mut r = Reorder::new(2);
        // Decode order I0 P3 B1 B2 P6 B4 B5.
        let mut seen = Vec::new();
        for pts in [0, 3, 1, 2, 6, 4, 5] {
            r.push(HwTimestamp::new(pts), Some(frame(pts)));
            seen.extend(ticks(&r.ready().expect("in depth")));
        }
        seen.extend(ticks(&r.drain().expect("drains")));
        assert_eq!(seen, [0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn dropped_frames_order_but_are_not_returned() {
        let mut r = Reorder::new(0);
        r.push(HwTimestamp::new(0), None);
        r.push(HwTimestamp::new(1), Some(frame(1)));
        assert_eq!(ticks(&r.ready().expect("ordered")), [1]);
    }

    #[test]
    fn a_frame_behind_a_released_one_fails() {
        let mut r = Reorder::new(0);
        r.push(HwTimestamp::new(5), Some(frame(5)));
        assert_eq!(ticks(&r.ready().expect("first")), [5]);
        r.push(HwTimestamp::new(3), Some(frame(3)));
        assert!(matches!(r.ready(), Err(HwError::Failed { .. })));
    }
}
