//! Host-side rotary frequency tables.
//!
//! Both `RoPE` flavours the family uses get their frequency table computed once
//! on the host and uploaded as a plain `f32` buffer, so the rotation kernels
//! carry no transcendental state and the two conventions (interleaved 4-axis
//! for the transformer, rotate-half 1-axis for the text encoder) differ only
//! in which table and which kernel reads it.

/// Interleaved per-axis frequency table for the transformer's 4-axis `RoPE`.
///
/// For each axis the table holds `axes_dim[axis] / 2` frequencies,
/// `theta^(-j / (axes_dim[axis] / 2))`, concatenated axis by axis. The kernel
/// rotates adjacent dimension pairs of an axis with that axis's position times
/// the pair's frequency.
#[expect(
    clippy::cast_precision_loss,
    reason = "axis halves are small head-dimension counts, far below f32's exact-integer range"
)]
pub(crate) fn rope_4d_freqs(axes_dim: &[usize; 4], theta: f32) -> Vec<f32> {
    let mut freqs = Vec::new();
    for &axis in axes_dim {
        let half = axis / 2;
        for j in 0..half {
            freqs.push(
                theta.powf(-f32::from(u16::try_from(j).unwrap_or(u16::MAX)) / half.max(1) as f32),
            );
        }
    }
    freqs
}

/// Rotate-half frequency table for the text encoder's 1-axis `RoPE`:
/// `theta^(-2i / head_dim)` for `i in 0..head_dim/2`.
#[expect(
    clippy::cast_precision_loss,
    reason = "head_dim is a small dimension count, far below f32's exact-integer range"
)]
pub(crate) fn rope_half_freqs(head_dim: usize, theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| {
            theta.powf(
                -2.0 * f32::from(u16::try_from(i).unwrap_or(u16::MAX)) / head_dim.max(1) as f32,
            )
        })
        .collect()
}
