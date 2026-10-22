//! The classification head on the host: `relu(pre x + b)` then a second
//! linear layer. Two small products on one row cost less than a launch.

/// `w[out][inp] * x + b`, accumulated in f64.
fn dense(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f64> {
    w.chunks_exact(x.len().max(1))
        .zip(b)
        .map(|(row, bias)| {
            f64::from(*bias)
                + row
                    .iter()
                    .zip(x)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum::<f64>()
        })
        .collect()
}

/// The logits for one first-position vector `x` of width `dim`: `pre_w` is
/// `[dim][dim]`, `cls_w` is `[labels][dim]`, both row-major.
///
/// Weights shorter than the shapes imply give fewer outputs rather than a
/// panic; [`super::validate`] has already checked the real ones.
#[must_use]
pub(super) fn apply(
    pre_w: &[f32],
    pre_b: &[f32],
    cls_w: &[f32],
    cls_b: &[f32],
    x: &[f32],
    dim: usize,
    labels: usize,
) -> Vec<f32> {
    let x = &x[..dim.min(x.len())];
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing is the f32 the head returns"
    )]
    let hidden: Vec<f32> = dense(pre_w, pre_b, x)
        .into_iter()
        .map(|v| v.max(0.0) as f32)
        .collect();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing is the f32 the head returns"
    )]
    dense(cls_w, cls_b, &hidden)
        .into_iter()
        .take(labels)
        .map(|v| v as f32)
        .collect()
}
