//! The Qwen-Image-2.1 three-axis rotary positions.
//!
//! A head's 128 dimensions are split over three axes - frame, height,
//! width - at widths 16, 56 and 56 (the config's `axes_dims_rope`), so 8,
//! 28 and 28 rotation pairs in that order. Pairs are interleaved: pair `i`
//! is `(x[2i], x[2i+1])`, not the two halves of the head. Within an axis of
//! width `d`, pair `j` turns at frequency `theta^(-2j/d)` with theta 10000.
//!
//! Positions: text token `i` sits at `i` on all three axes. The target
//! image's `h x w` latent grid (row-major tokens) sits at frame `P`, the
//! text length, with its height and width positions centred on zero:
//! `range(-(n - n/2), n/2)`, so a 4-row grid spans -2..=1. Negative
//! positions are just negative angles; the angles are computed directly in
//! `f64` per (token, pair), so no table indexing by sign is involved.
//!
//! An edit's prefix interleaves text with condition images
//! ([`Segment`]). One position runs through it: a text token takes it on
//! all three axes and moves it on by one; an image block holds the frame
//! axis at it for every one of its tokens, lays them on a grid centred like
//! the target's, and moves it on by the longer of its two sides. The target
//! is the last such block, at the position the prefix reached - for a
//! text-only prefix, the text length, as above.
//!
//! The device applies the rotation from a per-token cosine/sine table
//! `(tokens, head_dim / 2)` built here (see `Ops::qk_norm_rope`): the
//! prefix's table once per job, the target's once per grid.

use crate::qwen_image21::vae::LatentGrid;

/// The rotary base frequency.
pub(crate) const THETA: f64 = 10_000.0;

/// One token's (frame, height, width) position.
pub(crate) type AxisPos = [i64; 3];

/// Text tokens `0..p`: position `i` on every axis.
#[cfg(test)]
pub(crate) fn text_positions(p: usize) -> Vec<AxisPos> {
    (0..p)
        .map(|i| {
            let i = i64::try_from(i).unwrap_or(i64::MAX);
            [i, i, i]
        })
        .collect()
}

/// `range(-(n - n/2), n/2)`: `n` positions centred on zero.
fn centred(n: usize) -> impl Iterator<Item = i64> {
    let n = i64::try_from(n).unwrap_or(i64::MAX);
    -(n - n / 2)..n / 2
}

/// The target grid's tokens in row-major order, at frame `p` (the text
/// length) with centred height and width.
#[cfg(test)]
pub(crate) fn target_positions(p: usize, grid: LatentGrid) -> Vec<AxisPos> {
    target_positions_at(i64::try_from(p).unwrap_or(i64::MAX), grid)
}

/// An image block's tokens in row-major order, all at `frame`, with
/// centred height and width: the target, or a condition image.
pub(crate) fn target_positions_at(frame: i64, grid: LatentGrid) -> Vec<AxisPos> {
    centred(grid.h)
        .flat_map(|y| centred(grid.w).map(move |x| [frame, y, x]))
        .collect()
}

/// One run of an interleaved prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Segment {
    /// This many text tokens.
    Text(usize),
    /// One condition image's latent grid, a token per cell.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the edit pipeline that builds this lands later on this branch"
        )
    )]
    Image(LatentGrid),
}

impl Segment {
    /// Tokens the segment holds.
    pub(crate) fn rows(self) -> usize {
        match self {
            Self::Text(n) => n,
            Self::Image(grid) => grid.cells(),
        }
    }
}

/// The positions of an interleaved prefix, and the frame position the
/// target block after it takes.
pub(crate) fn prefix_positions(layout: &[Segment]) -> (Vec<AxisPos>, i64) {
    let mut positions = Vec::with_capacity(layout.iter().map(|s| s.rows()).sum());
    let mut position = 0i64;
    for segment in layout {
        match *segment {
            Segment::Text(n) => {
                let n = i64::try_from(n).unwrap_or(i64::MAX);
                positions.extend((position..position + n).map(|i| [i, i, i]));
                position += n;
            }
            Segment::Image(grid) => {
                positions.extend(target_positions_at(position, grid));
                position += i64::try_from(grid.h.max(grid.w)).unwrap_or(i64::MAX);
            }
        }
    }
    (positions, position)
}

/// Each segment's rows in the prefix and whether it is an image block, in
/// order. A block is one segment however its neighbours are laid out: two
/// images side by side are two entries.
pub(crate) fn segment_rows(layout: &[Segment]) -> Vec<(std::ops::Range<usize>, bool)> {
    let mut start = 0;
    layout
        .iter()
        .map(|segment| {
            let rows = start..start + segment.rows();
            start = rows.end;
            (rows, matches!(segment, Segment::Image(_)))
        })
        .collect()
}

/// The `(cos, sin)` tables of `positions`, each `(tokens, sum(axes) / 2)`:
/// pair columns ordered frame, height, width.
#[expect(
    clippy::cast_precision_loss,
    reason = "positions and axis widths are far below 2^52, exact in f64"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "each cosine and sine is rounded once to the f32 the device rotates in"
)]
pub(crate) fn angle_table(positions: &[AxisPos], axes: [usize; 3]) -> (Vec<f32>, Vec<f32>) {
    let freqs: Vec<(usize, f64)> = axes
        .iter()
        .enumerate()
        .flat_map(|(a, &d)| (0..d / 2).map(move |j| (a, THETA.powf(-2.0 * j as f64 / d as f64))))
        .collect();
    let mut cos = Vec::with_capacity(positions.len() * freqs.len());
    let mut sin = Vec::with_capacity(positions.len() * freqs.len());
    for pos in positions {
        for &(axis, freq) in &freqs {
            let angle = pos[axis] as f64 * freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}
