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
//! The device applies the rotation from a per-token cosine/sine table
//! `(tokens, head_dim / 2)` built here (see `Ops::qk_norm_rope`): the
//! prefix's table once per job, the target's once per grid.

use crate::qwen_image21::vae::LatentGrid;

/// The rotary base frequency.
pub(crate) const THETA: f64 = 10_000.0;

/// One token's (frame, height, width) position.
pub(crate) type AxisPos = [i64; 3];

/// Text tokens `0..p`: position `i` on every axis.
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
pub(crate) fn target_positions(p: usize, grid: LatentGrid) -> Vec<AxisPos> {
    let frame = i64::try_from(p).unwrap_or(i64::MAX);
    centred(grid.h)
        .flat_map(|y| centred(grid.w).map(move |x| [frame, y, x]))
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
