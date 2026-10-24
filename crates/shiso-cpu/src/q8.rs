//! The `Q8_0` block format and its dot product.
//!
//! A block is 34 bytes: an f16 scale `d` (little-endian) and 32 signed quants.
//! Every `Q8_0` kernel, backbone and head, reduces a row through
//! [`q8_row_dot`], so two kernels that compute the same product agree bit for
//! bit instead of differing by summation order.

use half::f16;

/// Quants per block.
pub(crate) const QK8_0: usize = 32;
/// Bytes per block: the f16 scale plus [`QK8_0`] quants.
pub(crate) const BLOCK_Q8_0_BYTES: usize = 34;

/// The block's scale.
fn scale(block: &[u8]) -> f32 {
    f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32()
}

/// `row · x` for one `Q8_0` row.
///
/// Per block the quants are summed against `x` in f32 in index order, scaled
/// by `d`, and the blocks are added in block order. This order is a CPU-internal
/// contract, not one shared with the Metal or CUDA kernels (they reduce by
/// simd trees): it keeps `gemm`, `gemm64` and the matvecs bit-identical to each
/// other. Do not reassociate it.
pub(crate) fn q8_row_dot(row: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len(), x.len() / QK8_0 * BLOCK_Q8_0_BYTES);
    let mut acc = 0.0f32;
    for (block, xs) in row
        .chunks_exact(BLOCK_Q8_0_BYTES)
        .zip(x.chunks_exact(QK8_0))
    {
        let mut sum = 0.0f32;
        for (q, v) in block[2..].iter().zip(xs) {
            sum += f32::from(q.cast_signed()) * v;
        }
        acc += scale(block) * sum;
    }
    acc
}

/// Dequantizes one block into `out`.
pub(crate) fn q8_dequant_block(block: &[u8], out: &mut [f32; QK8_0]) {
    debug_assert_eq!(block.len(), BLOCK_Q8_0_BYTES);
    let d = scale(block);
    for (o, q) in out.iter_mut().zip(&block[2..]) {
        *o = d * f32::from(q.cast_signed());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE_SIXTY_FOURTH_F16: u16 = 0x2400;

    fn block(d: u16) -> Vec<u8> {
        let mut b = d.to_le_bytes().to_vec();
        b.extend((-16i8..16).map(i8::cast_unsigned));
        b
    }

    #[test]
    fn the_dot_is_the_dequantized_dot() {
        let row: Vec<u8> = [block(ONE_SIXTY_FOURTH_F16), block(ONE_SIXTY_FOURTH_F16)].concat();
        let x: Vec<f32> = (0..64u8).map(|i| 0.5 * f32::from(i)).collect();
        let mut want = 0.0f64;
        for (b, xs) in row.chunks(BLOCK_Q8_0_BYTES).zip(x.chunks(QK8_0)) {
            let mut deq = [0.0f32; QK8_0];
            q8_dequant_block(b, &mut deq);
            want += deq
                .iter()
                .zip(xs)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>();
        }
        let got = f64::from(q8_row_dot(&row, &x));
        assert!(want != 0.0);
        assert!(((got - want) / want).abs() < 1e-6, "{got} vs {want}");
    }

    #[test]
    fn a_zero_scale_block_contributes_zero() {
        let row: Vec<u8> = [block(0), block(ONE_SIXTY_FOURTH_F16)].concat();
        let x = vec![1.0f32; 64];
        let only_second = q8_row_dot(&row[BLOCK_Q8_0_BYTES..], &x[QK8_0..]);
        assert_eq!(q8_row_dot(&row, &x).to_bits(), only_second.to_bits());
        assert_eq!(q8_row_dot(&block(0), &x[..QK8_0]).to_bits(), 0);
    }

    #[test]
    fn dequant_matches_scale_times_quant() {
        let mut out = [0.0f32; QK8_0];
        q8_dequant_block(&block(ONE_SIXTY_FOURTH_F16), &mut out);
        assert!((out[0] + 0.25).abs() < f32::EPSILON);
        assert!((out[31] - 15.0 / 64.0).abs() < f32::EPSILON);
    }
}
