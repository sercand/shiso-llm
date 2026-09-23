//! A per-layer tap on the residual streams: one row, copied back after the
//! prologue and after every island, for bisecting a divergence layer by layer.
//!
//! The step dump (`SHISO_STEP_DUMP`) says what each launch wrote, as
//! statistics over a whole allocation. That finds a NaN or a kernel that wrote
//! nothing; it cannot find a layer whose output is finite, plausible and a
//! little wrong, because the only way to see that is to diff the vector against
//! an independent implementation element by element. This tap is the vector.
//!
//! # What is copied, and when
//!
//! One row of [`Buf::Residual`](crate::sparse::scratch::Buf::Residual) - every
//! stream of it, `[streams][hidden]` `f32`, raw rather than normalized - at
//! three kinds of point, which are exactly the boundaries a pass already has:
//!
//! - [`TapPoint::Embed`]: after the prologue, which ends with the broadcast of
//!   the embedding across the streams;
//! - [`TapPoint::Mixer`]: after a layer's token-mixer island, whose last launch
//!   is the combine that adds the mixer's contribution. On a layer that carries
//!   the n-gram block that block ran inside the same island, ahead of its mix,
//!   so its add is in this row too;
//! - [`TapPoint::Layer`]: after the layer's routed feed-forward island, which is
//!   the residual the next layer starts from.
//!
//! The row is the pass's **last** one, which for the pass that produces a
//! frontier's logits is the frontier itself.
//!
//! # A tapped pass is never replayed
//!
//! A replayed island issues nothing from this process, so there is no point
//! between two islands at which the host can read anything. A tapped pass is
//! therefore run with the graph tier withheld: every island is encoded
//! directly, and nothing is captured either. The plan itself is not re-made -
//! a one-row pass keeps the kernel choices it was planned with for capture -
//! so the launches issued are the ones a replay would have issued, and the
//! numbers are the numbers the untapped pass computes.
//!
//! Each tap point costs a device barrier and a forty-kilobyte copy. Nothing of
//! that is paid when no tap is armed: the executor checks an `Option` per
//! island and takes the ordinary path.

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::LayerIdx;

/// Where in a pass a residual row was copied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TapPoint {
    /// After the prologue: the embedding broadcast across every stream.
    Embed,
    /// After this layer's token-mixer island.
    Mixer(LayerIdx),
    /// After this layer's routed feed-forward island: the layer's output.
    Layer(LayerIdx),
}

/// One copied row.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TapRecord {
    /// Where it was copied.
    pub point: TapPoint,
    /// `streams * hidden` floats, stream-major.
    pub values: Vec<f32>,
}

/// An armed tap, and what it has copied so far.
///
/// Each tapped pass replaces what the previous one left: the tap describes the
/// last pass it saw, which is how a caller that arms it before a prompt's final
/// chunk gets that chunk's frontier and nothing else.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualTap {
    streams: u32,
    hidden: u32,
    records: Vec<TapRecord>,
}

impl ResidualTap {
    /// A tap over a residual of `streams` streams of `hidden` floats each.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when either is zero: a tap that copied
    /// nothing would read as a pass that wrote nothing.
    pub fn new(streams: u32, hidden: u32) -> CoreResult<Self> {
        if streams == 0 || hidden == 0 {
            return Err(CoreError::InvalidValue {
                key: "residual tap",
                value: format!("{streams} streams of {hidden}"),
            });
        }
        Ok(Self {
            streams,
            hidden,
            records: Vec::new(),
        })
    }

    /// Streams per row.
    #[must_use]
    pub const fn streams(&self) -> u32 {
        self.streams
    }

    /// Floats per stream.
    #[must_use]
    pub const fn hidden(&self) -> u32 {
        self.hidden
    }

    /// Floats in one row.
    #[must_use]
    pub const fn row_len(&self) -> usize {
        self.streams as usize * self.hidden as usize
    }

    /// What the last tapped pass copied, in the order it ran.
    #[must_use]
    pub fn records(&self) -> &[TapRecord] {
        &self.records
    }

    /// Forgets the previous pass, at the start of a new one.
    pub fn restart(&mut self) {
        self.records.clear();
    }

    /// Adds one row.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when `values` is not one row, and
    /// [`CoreError::InvalidValue`] when `point` was already copied in this
    /// pass - two rows under one name would make the later one silently win in
    /// any file layout keyed by the point.
    pub fn push(&mut self, point: TapPoint, values: Vec<f32>) -> CoreResult<()> {
        if values.len() != self.row_len() {
            return Err(CoreError::ShapeMismatch {
                field: "residual tap row",
                expected: self.row_len() as u64,
                got: values.len() as u64,
            });
        }
        if self.records.iter().any(|r| r.point == point) {
            return Err(CoreError::InvalidValue {
                key: "residual tap point",
                value: format!("{point:?} copied twice in one pass"),
            });
        }
        self.records.push(TapRecord { point, values });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_of_the_wrong_width_or_a_repeated_point_is_refused() {
        let mut tap = ResidualTap::new(2, 3).expect("a shape");
        assert!(tap.push(TapPoint::Embed, vec![0.0; 5]).is_err());
        tap.push(TapPoint::Embed, vec![0.0; 6]).expect("one row");
        assert!(tap.push(TapPoint::Embed, vec![0.0; 6]).is_err());
        tap.push(TapPoint::Mixer(LayerIdx::FIRST), vec![1.0; 6])
            .expect("another point");
        assert_eq!(tap.records().len(), 2);
        tap.restart();
        assert!(tap.records().is_empty());
        assert!(ResidualTap::new(0, 3).is_err());
    }
}
