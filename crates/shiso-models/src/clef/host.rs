//! The head's host arithmetic: everything that grows with the schema rather
//! than with the prompt.
#![expect(
    clippy::cast_precision_loss,
    reason = "row widths and span lengths are a few thousand at most"
)]

use shiso_core::{CoreError, CoreResult, DType};

/// A layer norm with its weight and bias on the host.
#[derive(Debug)]
pub(super) struct HostNorm {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) eps: f32,
}

impl HostNorm {
    /// `(x - mean) / sqrt(var + eps) * weight + bias`, the biased variance
    /// `torch.nn.LayerNorm` takes.
    pub(super) fn apply(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + self.eps).sqrt();
        x.iter()
            .zip(self.weight.iter().zip(&self.bias))
            .map(|(v, (w, b))| (v - mean) * inv * w + b)
            .collect()
    }
}

/// The mean of rows `span` of a `[rows][width]` matrix.
pub(super) fn mean_rows(rows: &[f32], width: usize, span: std::ops::Range<usize>) -> Vec<f32> {
    let count = span.len() as f32;
    let mut sum = vec![0.0f32; width];
    for row in rows[span.start * width..span.end * width].chunks_exact(width) {
        for (s, v) in sum.iter_mut().zip(row) {
            *s += v;
        }
    }
    sum.iter().map(|s| s / count).collect()
}

/// `x / max(|x|, eps)`.
pub(super) fn l2_normalized(x: &[f32], eps: f32) -> Vec<f32> {
    let norm = dot(x, x).sqrt().max(eps);
    x.iter().map(|v| v / norm).collect()
}

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(super) fn softmax(x: &[f32]) -> Vec<f32> {
    let top = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|v| (v - top).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

fn f16_at(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}

/// Bytes one `width`-wide row occupies in `dtype`, or `None` for an encoding
/// this reader does not cut rows from.
fn row_bytes(dtype: DType, width: usize) -> Option<usize> {
    match dtype {
        DType::F32 => Some(width * 4),
        DType::F16 | DType::BF16 => Some(width * 2),
        DType::Q8_0 if width.is_multiple_of(32) => Some(width / 32 * 34),
        DType::Q6_K if width.is_multiple_of(256) => Some(width / 256 * 210),
        _ => None,
    }
}

/// Row `index` of a `[rows][width]` matrix stored as `dtype`, as f32.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for an encoding this reader does not cut rows
/// from, and [`CoreError::OutOfBounds`] for a row past the payload.
pub(super) fn dequant_row(
    data: &[u8],
    dtype: DType,
    width: usize,
    index: usize,
) -> CoreResult<Vec<f32>> {
    let row = row_bytes(dtype, width).ok_or(CoreError::Unsupported {
        what: "an output embedding in this encoding",
        backend: "clef",
    })?;
    let bytes = data
        .get(index * row..(index + 1) * row)
        .ok_or(CoreError::OutOfBounds {
            offset: (index * row) as u64,
            len: row as u64,
            buffer_bytes: data.len() as u64,
        })?;
    let mut out = Vec::with_capacity(width);
    match dtype {
        DType::F32 => out.extend(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b)),
        ),
        DType::F16 => out.extend(bytes.chunks_exact(2).map(f16_at)),
        DType::BF16 => out.extend(
            bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)),
        ),
        DType::Q8_0 => {
            for block in bytes.chunks_exact(34) {
                let d = f16_at(&block[..2]);
                out.extend(block[2..].iter().map(|q| d * f32::from(q.cast_signed())));
            }
        }
        DType::Q6_K => {
            for block in bytes.chunks_exact(210) {
                dequant_q6_k(block, &mut out);
            }
        }
        _ => unreachable!("row_bytes admits only the encodings above"),
    }
    Ok(out)
}

/// One `Q6_K` super-block of 256 values: 128 bytes of low nibbles, 64 of high
/// bit pairs, 16 signed scales and an f16 multiplier, as ggml lays it out.
fn dequant_q6_k(block: &[u8], out: &mut Vec<f32>) {
    let (ql, rest) = block.split_at(128);
    let (qh, rest) = rest.split_at(64);
    let (scales, d) = rest.split_at(16);
    let d = f16_at(d);
    let mut values = [0.0f32; 256];
    for half in 0..2 {
        let ql = &ql[half * 64..];
        let qh = &qh[half * 32..];
        let sc = &scales[half * 8..];
        let y = &mut values[half * 128..];
        for l in 0..32 {
            let is = l / 16;
            let q = |low: u8, shift: u32| -> f32 {
                f32::from((i16::from(low) | (i16::from((qh[l] >> shift) & 3) << 4)) - 32)
            };
            let s = |i: usize| f32::from(sc[i].cast_signed());
            y[l] = d * s(is) * q(ql[l] & 0xF, 0);
            y[l + 32] = d * s(is + 2) * q(ql[l + 32] & 0xF, 2);
            y[l + 64] = d * s(is + 4) * q(ql[l] >> 4, 4);
            y[l + 96] = d * s(is + 6) * q(ql[l + 32] >> 4, 6);
        }
    }
    out.extend_from_slice(&values);
}
