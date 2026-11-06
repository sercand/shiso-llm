//! The tower's rotary position tables.
//!
//! `DINOv3` has no learned position embedding: the patch coordinates enter
//! through a 2D rotary embedding applied to the queries and keys of every
//! block, and only to the patch tokens - the `CLS` and register tokens carry no
//! position. The tables are built from the input's grid, not from the
//! checkpoint's training size, because the tower was trained with random
//! rescaling and runs at any resolution.

use super::DinoConfig;

/// The cosine and sine tables for a `grid_h` x `grid_w` patch grid.
///
/// Both are `(grid_h * grid_w, head_dim)` row-major: one row per patch, in the
/// same row-major order the patch embeddings arrive in. The angle vector is
/// `[y frequencies..., x frequencies...]` of `head_dim/2` entries, tiled to
/// `head_dim`, which is the shape the half-split rotation reads.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the table is built in f64 and stored as f32, which is the precision the tower \
              runs at; the indices are grid positions of a 64x64 patch grid"
)]
pub fn rope_tables(config: &DinoConfig, grid_h: usize, grid_w: usize) -> (Vec<f32>, Vec<f32>) {
    let head_dim = config.head_dim();
    let quarters = head_dim / 4;
    // inv_freq[i] = 1 / theta^(4i / head_dim), the reference's
    // `1 / base ** arange(0, 1, 4 / head_dim)`.
    let inv_freq: Vec<f64> = (0..quarters)
        .map(|i| 1.0 / config.rope_theta.powf(4.0 * i as f64 / head_dim as f64))
        .collect();
    let half = quarters * 2;
    let mut cos = vec![0.0f32; grid_h * grid_w * head_dim];
    let mut sin = vec![0.0f32; grid_h * grid_w * head_dim];
    for y in 0..grid_h {
        for x in 0..grid_w {
            let token = y * grid_w + x;
            // Patch centres in [-1, 1]: the reference's
            // `2 * arange(0.5, n) / n - 1`.
            let center_y = 2.0 * ((y as f64 + 0.5) / grid_h as f64) - 1.0;
            let center_x = 2.0 * ((x as f64 + 0.5) / grid_w as f64) - 1.0;
            let mut angles = vec![0.0f64; half];
            for (axis, center) in [center_y, center_x].into_iter().enumerate() {
                for (i, frequency) in inv_freq.iter().enumerate() {
                    angles[axis * quarters + i] = 2.0 * std::f64::consts::PI * center * frequency;
                }
            }
            for column in 0..head_dim {
                // `angles.tile(2)`: the second half repeats the first.
                let angle = angles[column % half];
                cos[token * head_dim + column] = angle.cos() as f32;
                sin[token * head_dim + column] = angle.sin() as f32;
            }
        }
    }
    (cos, sin)
}
