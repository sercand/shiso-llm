//! The classification head on the host: `relu(pre x + b)` then a second
//! linear layer. Two small products on one row cost less than a launch.

use super::ClassifyError;

/// `w[out][inp] * x + b`, accumulated in f64; `w`, `b` and `x` must agree
/// with `out` and `inp`.
fn dense(w: &[f32], b: &[f32], x: &[f32], inp: usize) -> Result<Vec<f64>, ClassifyError> {
    if x.len() != inp || w.len() != b.len() * inp {
        return Err(ClassifyError::BadTensor {
            name: "head".to_owned(),
            why: format!(
                "weights of {} values and a bias of {} do not fit an input of {} (width {inp})",
                w.len(),
                b.len(),
                x.len()
            ),
        });
    }
    Ok(w.chunks_exact(inp.max(1))
        .zip(b)
        .map(|(row, bias)| {
            f64::from(*bias)
                + row
                    .iter()
                    .zip(x)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum::<f64>()
        })
        .collect())
}

/// The logits for one first-position vector `x` of width `dim`: `pre_w` is
/// `[dim][dim]`, `cls_w` is `[labels][dim]`, both row-major, each bias one
/// value per output.
///
/// # Errors
///
/// [`ClassifyError::BadTensor`] when a length disagrees with `dim` or
/// `labels`; the real weights were checked by [`super::validate`], so this
/// is a caller's mistake.
pub(super) fn apply(
    pre_w: &[f32],
    pre_b: &[f32],
    cls_w: &[f32],
    cls_b: &[f32],
    x: &[f32],
    (dim, labels): (usize, usize),
) -> Result<Vec<f32>, ClassifyError> {
    if pre_b.len() != dim || cls_b.len() != labels {
        return Err(ClassifyError::BadTensor {
            name: "head".to_owned(),
            why: format!(
                "biases of {} and {} are not {dim} and {labels}",
                pre_b.len(),
                cls_b.len()
            ),
        });
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing is the f32 the head returns"
    )]
    let hidden: Vec<f32> = dense(pre_w, pre_b, x, dim)?
        .into_iter()
        .map(|v| v.max(0.0) as f32)
        .collect();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing is the f32 the head returns"
    )]
    Ok(dense(cls_w, cls_b, &hidden, dim)?
        .into_iter()
        .map(|v| v as f32)
        .collect())
}
