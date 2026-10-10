//! Host-side rotary frequency tables.
//!
//! The transformer's interleaved 4-axis `RoPE` gets its frequency table
//! computed once on the host and uploaded as a plain `f32` buffer, so the
//! rotation kernel carries no transcendental state. The text encoder's
//! rotate-half table lives with the shared encoder
//! (`crate::diffusion::qwen3_text`).

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
