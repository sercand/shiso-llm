//! Host decoders for the ternary block formats and the 2-bit affine layout
//! that re-encodes them.
//!
//! These are the reference the device kernels are graded against, and nothing
//! on the load path calls them: a checkpoint's weights stay in the file's own
//! encoding on the device. They are here rather than in the kernel test crate
//! because a loader test that compares two containers of the same checkpoint
//! needs them too.
//!
//! All three formats decode a code `c` to `scale * (c - 1)`, with one scale per
//! 128 consecutive weights of a row:
//!
//! - [`DType::Pq2_0`]: `[f16 scale][32 bytes]`, element `j` in bits
//!   `2 * (j % 4)` of byte `j / 4`. Codes run 0..=3.
//! - [`DType::Ptq1_0`]: `[24 bytes][2 bytes][f16 scale]`. The 24 bytes hold five
//!   trits each and are read in two stages, 16 bytes then 8: trit `n` of byte
//!   `m` of a stage starting at element `e` and byte `b` is element
//!   `e + n * width + m`. The last 2 bytes hold four trits each, elements
//!   `120 + n * 2 + h`. Trit `n` of byte `q` is `((q * 3^n) mod 256) * 3 >> 8`,
//!   which reads the base-3 digits most significant first.
//! - the 2-bit affine layout: little-endian `u32` words with element `16 * w + l`
//!   at bits `2 * l` of word `w` - byte for byte the [`DType::Pq2_0`] code
//!   bytes - and `f16` scale and bias arrays, one each per group.

use half::f16;
use shiso_core::DType;

use crate::reader::{GgufError, GgufResult};

/// Weights one block or group covers.
pub const GROUP: usize = 128;

/// Bytes of one [`DType::Pq2_0`] block.
const PQ2_BYTES: usize = 34;
/// Bytes of one [`DType::Ptq1_0`] block.
const PTQ1_BYTES: usize = 28;

/// Powers of three a trit is shifted to the top of its byte by.
const POW3: [u8; 5] = [1, 3, 9, 27, 81];

/// The subject the errors in this module name.
const SUBJECT: &str = "ternary block";

/// A layout error naming what was required and what was given.
fn layout_error(want: &str, got: String) -> GgufError {
    GgufError::LayoutMismatch {
        tensor: Box::from(SUBJECT),
        want: Box::from(want),
        got: got.into_boxed_str(),
    }
}

/// Trit `n` of a packed byte.
const fn trit(byte: u8, n: usize) -> u8 {
    let shifted = byte.wrapping_mul(POW3[n]);
    // The product is at most 255 * 3, so the shift brings it back below 3.
    ((shifted as u16 * 3) >> 8) as u8
}

/// The `f16` stored little-endian at `bytes[at..at + 2]`, widened.
fn half_at(bytes: &[u8], at: usize) -> f32 {
    f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32()
}

/// Decodes one [`DType::Pq2_0`] block's codes, 0..=3, into `codes`.
pub fn pq2_0_codes(block: &[u8; PQ2_BYTES], codes: &mut [u8; GROUP]) {
    for (j, code) in codes.iter_mut().enumerate() {
        *code = (block[2 + j / 4] >> (2 * (j % 4))) & 3;
    }
}

/// Decodes one [`DType::Ptq1_0`] block's trits, 0..=2, into `codes`.
pub fn ptq1_0_codes(block: &[u8; PTQ1_BYTES], codes: &mut [u8; GROUP]) {
    // (first byte, bytes, trits a byte, first element)
    let stages = [(0, 16, 5, 0), (16, 8, 5, 80), (24, 2, 4, 120)];
    for (first, width, trits, start) in stages {
        for n in 0..trits {
            for m in 0..width {
                codes[start + n * width + m] = trit(block[first + m], n);
            }
        }
    }
}

/// Decodes a whole ternary matrix to `f32`, row-major `[rows, in_dim]`.
///
/// # Errors
///
/// [`GgufError::LayoutMismatch`] when `dtype` is not a ternary block format,
/// `in_dim` is not a whole number of blocks, or `bytes` is not exactly the
/// matrix.
pub fn decode_ternary(
    dtype: DType,
    bytes: &[u8],
    in_dim: usize,
    rows: usize,
) -> GgufResult<Vec<f32>> {
    let block_bytes = match dtype {
        DType::Pq2_0 => PQ2_BYTES,
        DType::Ptq1_0 => PTQ1_BYTES,
        other => return Err(layout_error("pq2_0 or ptq1_0", format!("{other:?}"))),
    };
    let blocks = whole_groups(in_dim)?
        .checked_mul(rows)
        .ok_or_else(|| layout_error("a matrix that fits in memory", format!("{rows} rows")))?;
    if bytes.len() != blocks * block_bytes {
        return Err(layout_error(
            "exactly the matrix's blocks",
            format!("{} bytes for {blocks} blocks", bytes.len()),
        ));
    }
    let mut out = Vec::with_capacity(blocks * GROUP);
    let mut codes = [0_u8; GROUP];
    for block in bytes.chunks_exact(block_bytes) {
        let scale = if dtype == DType::Pq2_0 {
            let mut fixed = [0_u8; PQ2_BYTES];
            fixed.copy_from_slice(block);
            pq2_0_codes(&fixed, &mut codes);
            half_at(block, 0)
        } else {
            let mut fixed = [0_u8; PTQ1_BYTES];
            fixed.copy_from_slice(block);
            ptq1_0_codes(&fixed, &mut codes);
            half_at(block, 26)
        };
        out.extend(codes.iter().map(|&c| scale * (f32::from(c) - 1.0)));
    }
    Ok(out)
}

/// Decodes a 2-bit affine matrix at group size 128 to `f32`, row-major
/// `[rows, in_dim]`: `scale * code + bias` per weight.
///
/// # Errors
///
/// [`GgufError::LayoutMismatch`] when `in_dim` is not a whole number of groups
/// or any array is not exactly the length the shape implies.
pub fn decode_affine2(
    words: &[u8],
    scales: &[u8],
    biases: &[u8],
    in_dim: usize,
    rows: usize,
) -> GgufResult<Vec<f32>> {
    let groups = whole_groups(in_dim)?
        .checked_mul(rows)
        .ok_or_else(|| layout_error("a matrix that fits in memory", format!("{rows} rows")))?;
    if words.len() != groups * GROUP / 4 || scales.len() != groups * 2 || biases.len() != groups * 2
    {
        return Err(layout_error(
            "words, scales and biases of the matrix's groups",
            format!(
                "{} / {} / {} bytes for {groups} groups",
                words.len(),
                scales.len(),
                biases.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(groups * GROUP);
    for (g, packed) in words.chunks_exact(GROUP / 4).enumerate() {
        let scale = half_at(scales, 2 * g);
        let bias = half_at(biases, 2 * g);
        out.extend((0..GROUP).map(|j| {
            let code = (packed[j / 4] >> (2 * (j % 4))) & 3;
            scale.mul_add(f32::from(code), bias)
        }));
    }
    Ok(out)
}

/// Groups in a row of `in_dim`, refusing a row that is not whole groups.
fn whole_groups(in_dim: usize) -> GgufResult<usize> {
    if in_dim == 0 || !in_dim.is_multiple_of(GROUP) {
        return Err(layout_error(
            "a row of whole 128-weight groups",
            format!("{in_dim} weights"),
        ));
    }
    Ok(in_dim / GROUP)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One `PQ2_0` block with scale `d` and codes `codes`.
    fn pq2_block(d: f32, codes: &[u8; GROUP]) -> Vec<u8> {
        let mut block = f16::from_f32(d).to_le_bytes().to_vec();
        for four in codes.chunks_exact(4) {
            block.push(four[0] | (four[1] << 2) | (four[2] << 4) | (four[3] << 6));
        }
        block
    }

    #[test]
    fn pq2_0_reads_codes_from_the_low_bits_up() {
        let mut codes = [1_u8; GROUP];
        codes[..4].copy_from_slice(&[3, 2, 1, 0]);
        let block = pq2_block(0.5, &codes);
        assert_eq!(block[2], 0x1B);
        let out = decode_ternary(DType::Pq2_0, &block, GROUP, 1).expect("one block");
        assert_eq!(&out[..5], &[1.0, 0.5, 0.0, -0.5, 0.0]);
    }

    #[test]
    fn ptq1_0_stages_put_each_trit_at_its_element() {
        // Byte 0 of the first stage holds trits (2, 0, 0, 0, 1) most
        // significant first: 2*81 + 1 = 163, which the format stores as the
        // ceiling of 163 * 256 / 243.
        let mut block = [0_u8; PTQ1_BYTES];
        block[0] = u8::try_from((163_u32 * 256).div_ceil(243)).expect("byte");
        let mut codes = [9_u8; GROUP];
        ptq1_0_codes(&block, &mut codes);
        assert_eq!(codes[0], 2);
        assert_eq!(codes[16], 0);
        assert_eq!(codes[64], 1);
        assert!(codes.iter().all(|&c| c <= 2));
    }

    #[test]
    fn zero_bytes_are_minus_one_everywhere_in_ptq1_0() {
        let mut block = [0_u8; PTQ1_BYTES];
        block[26..].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        let out = decode_ternary(DType::Ptq1_0, &block, GROUP, 1).expect("one block");
        assert!(out.iter().all(|&v| v.to_bits() == (-2.0_f32).to_bits()));
    }

    #[test]
    fn affine2_with_bias_minus_scale_is_pq2_0() {
        let codes: [u8; GROUP] = std::array::from_fn(|j| u8::try_from(j * 7 % 4).expect("code"));
        let block = pq2_block(0.25, &codes);
        let scale = f16::from_f32(0.25).to_le_bytes();
        let bias = f16::from_f32(-0.25).to_le_bytes();
        let affine = decode_affine2(&block[2..], &scale, &bias, GROUP, 1).expect("one group");
        let ternary = decode_ternary(DType::Pq2_0, &block, GROUP, 1).expect("one block");
        assert_eq!(affine, ternary);
    }

    #[test]
    fn shapes_that_are_not_whole_blocks_are_refused() {
        assert!(decode_ternary(DType::Pq2_0, &[0; 34], 64, 1).is_err());
        assert!(decode_ternary(DType::Pq2_0, &[0; 33], GROUP, 1).is_err());
        assert!(decode_ternary(DType::Q8_0, &[0; 34], GROUP, 1).is_err());
        assert!(decode_affine2(&[0; 32], &[0; 2], &[0; 4], GROUP, 1).is_err());
    }
}
