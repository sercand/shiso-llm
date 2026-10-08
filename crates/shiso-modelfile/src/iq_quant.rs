//! Host reference decoders for the mixed-width `i`-quant formats, plus the
//! 2-bit `K`-quant that shares their block geometry.
//!
//! These are the reference the device kernels are graded against, and nothing
//! on the load path calls them: a checkpoint's weights stay in the file's own
//! encoding on the device. They are here rather than in the kernel test crate
//! because a loader test that compares two containers of the same checkpoint
//! needs them too.
//!
//! Every format maps a 256-weight block onto a small set of fixed tables in
//! [`crate::quant_tables`]; the decoders below are the bit-exact statement of
//! that mapping, in the multiplication order the tables were fitted under:
//!
//! - [`DType::Q2_K`]: 2-bit weights with a scale *and* a minimum per 16-weight
//!   subblock, read plane by plane: subblock `2 * j + p` of a 128-weight plane
//!   takes scale byte `8 * plane + 2 * j + p` and reads bits `2 * j` of the
//!   weight bytes `32 * plane + 16 * p..+16`.
//! - [`DType::IQ2_XXS`]: 8-bit indices into a 256-entry grid of eight 6-bit
//!   weights, with 7-bit sign-pattern indices and one 4-bit scale per
//!   32-weight group packed beside them.
//! - [`DType::IQ2_XS`]: 9-bit indices into a 512-entry grid, seven sign bits
//!   folded into each index word, two 4-bit scales a byte.
//! - [`DType::IQ2_S`]: 10-bit grid indices split across two planes, one sign
//!   bit per weight (stored in the block's second 32 index bytes), two 4-bit
//!   scales a byte.
//! - [`DType::IQ3_XXS`]: 8-bit indices into a 256-entry grid of four 6-bit
//!   weights, with the same sign-and-scale word as [`DType::IQ2_XXS`].
//! - [`DType::IQ3_S`]: 9-bit grid indices split across two planes, per-weight
//!   signs, and one byte carrying two odd 4-bit scales (`1 + 2 * nibble`).
//! - [`DType::IQ1_S`]: 11-bit indices into a 2048-entry ternary grid, with a
//!   3-bit scale multiplier and a grid-shift sign sharing one `u16` per
//!   32-weight group; the shift is the constant `IQ1S_DELTA` (0.125).
//! - [`DType::IQ1_M`]: the same grid, but no `d` field - the block scale is
//!   assembled from the top nibbles of the scale bytes, and each 16-weight
//!   subblock carries its own 3-bit scale multiplier.
//! - [`DType::IQ4_XS`]: 4-bit indices into [`crate::quant_tables::IQ4NL_VALUES`]
//!   with 6-bit sub-block scales split across two planes, offset so that 32
//!   means "no scaling" (`d * (ls - 32)`).

use half::f16;
use shiso_core::DType;

use crate::quant::{
    BlockIq1M, BlockIq1S, BlockIq2S, BlockIq2Xs, BlockIq2Xxs, BlockIq3S, BlockIq3Xxs, BlockIq4Xs,
    BlockQ2K, QK_K,
};
use crate::quant_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, IQ4NL_VALUES,
    KMASK_IQ2XS, KSIGNS_IQ2XS,
};
use crate::reader::GgufError;

/// The subject the errors in this module name.
const SUBJECT: &str = "i-quant block";

/// The grid shift of the 1-bit formats: a weight is `grid + delta`, with the
/// sign of the shift stored per group rather than per weight.
const IQ1S_DELTA: f32 = 0.125;

/// A layout error naming what was required and what was given.
fn layout_error(want: &str, got: String) -> GgufError {
    GgufError::LayoutMismatch {
        tensor: Box::from(SUBJECT),
        want: Box::from(want),
        got: got.into_boxed_str(),
    }
}

/// The block bytes as a fixed array, the length the `dtype`'s struct demands.
fn fixed_bytes<const N: usize>(block: &[u8], dtype: DType) -> Result<[u8; N], GgufError> {
    block.try_into().map_err(|_| {
        layout_error(
            &format!("{dtype:?} block of {N} bytes"),
            format!("{} bytes", block.len()),
        )
    })
}

/// The `f16` scale stored as bits in a block header.
fn half(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

/// The packed byte `j` of a `u64` grid entry (lane order: byte 0 first).
const fn byte64(grid: u64, j: usize) -> u8 {
    ((grid >> (8 * j)) & 0xFF) as u8
}

/// The packed byte `j` of a `u32` grid entry.
const fn byte32(grid: u32, j: usize) -> u8 {
    ((grid >> (8 * j)) & 0xFF) as u8
}

/// The packed byte `j` of a `u64` grid entry, signed (the ternary grids).
const fn sbyte64(grid: u64, j: usize) -> i8 {
    byte64(grid, j).cast_signed()
}

/// The sign lane `j` selects: the 2-bit formats apply their grid byte times
/// a per-lane sign drawn from a 128-entry pattern table.
fn signed(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] == 0 {
        1.0
    } else {
        -1.0
    }
}

/// The 1-bit formats' grid shift, negative when the group's sign bit is set.
fn delta(bit_set: u8) -> f32 {
    if bit_set == 0 {
        IQ1S_DELTA
    } else {
        -IQ1S_DELTA
    }
}

/// Decodes one [`DType::Q2_K`] block into 256 weights.
pub fn decode_q2_k_block(block: &[u8; 84], out: &mut [f32; QK_K]) {
    let b = bytemuck::pod_read_unaligned::<BlockQ2K>(block);
    let d = half(b.d);
    let dmin = half(b.dmin);
    let mut next = 0_usize;
    let mut is = 0_usize;
    for plane in 0..2 {
        for j in 0..4 {
            let shift = 2 * j;
            for sub in 0..2 {
                let sc = b.scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0xF);
                let ml = dmin * f32::from(sc >> 4);
                let qbase = plane * 32 + sub * 16;
                for l in 0..16 {
                    out[next] = dl * f32::from((b.qs[qbase + l] >> shift) & 3) - ml;
                    next += 1;
                }
            }
        }
    }
}

/// Decodes one [`DType::IQ2_XXS`] block into 256 weights.
pub fn decode_iq2_xxs_block(block: &[u8; 66], out: &mut [f32; QK_K]) {
    let b = bytemuck::pod_read_unaligned::<BlockIq2Xxs>(block);
    let d = half(b.d);
    let mut next = 0_usize;
    for ib in 0..QK_K / 32 {
        // Two index words then two sign-and-scale words per 32-weight group.
        let aux0 = u32::from(b.qs[4 * ib]) | (u32::from(b.qs[4 * ib + 1]) << 16);
        let aux1 = u32::from(b.qs[4 * ib + 2]) | (u32::from(b.qs[4 * ib + 3]) << 16);
        let db = d * (0.5 + f32::from(((aux1 >> 28) & 0xF) as u8)) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[((aux0 >> (8 * l)) & 0xFF) as usize];
            let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 0x7F) as usize];
            for j in 0..8 {
                out[next + j] = db * f32::from(byte64(grid, j)) * signed(signs, j);
            }
            next += 8;
        }
    }
}

/// Decodes one [`DType::IQ2_XS`] block into 256 weights.
pub fn decode_iq2_xs_block(block: &[u8; 74], out: &mut [f32; QK_K]) {
    let b = bytemuck::pod_read_unaligned::<BlockIq2Xs>(block);
    let d = half(b.d);
    let mut next = 0_usize;
    for ib in 0..QK_K / 32 {
        let sc = b.scales[ib];
        let db = [
            d * (0.5 + f32::from(sc & 0xF)) * 0.25,
            d * (0.5 + f32::from(sc >> 4)) * 0.25,
        ];
        for l in 0..4 {
            let q = b.qs[4 * ib + l];
            let grid = IQ2XS_GRID[usize::from(q & 511)];
            let signs = KSIGNS_IQ2XS[usize::from(q >> 9)];
            for j in 0..8 {
                out[next + j] = db[l / 2] * f32::from(byte64(grid, j)) * signed(signs, j);
            }
            next += 8;
        }
    }
}

/// Decodes one [`DType::IQ2_S`] block into 256 weights.
pub fn decode_iq2_s_block(block: &[u8; 82], out: &mut [f32; QK_K]) {
    let b = bytemuck::pod_read_unaligned::<BlockIq2S>(block);
    let d = half(b.d);
    let mut next = 0_usize;
    for ib in 0..QK_K / 32 {
        let sc = b.scales[ib];
        let db = [
            d * (0.5 + f32::from(sc & 0xF)) * 0.25,
            d * (0.5 + f32::from(sc >> 4)) * 0.25,
        ];
        for l in 0..4 {
            let idx =
                usize::from(b.qs[4 * ib + l]) | ((usize::from(b.qh[ib]) << (8 - 2 * l)) & 0x300);
            let grid = IQ2S_GRID[idx];
            // The signs live in the second half of `qs`, one byte per quad.
            let signs = b.qs[QK_K / 8 + 4 * ib + l];
            for j in 0..8 {
                out[next + j] = db[l / 2] * f32::from(byte64(grid, j)) * signed(signs, j);
            }
            next += 8;
        }
    }
}

/// Decodes one [`DType::IQ3_XXS`] block into 256 weights.
pub fn decode_iq3_xxs_block(block: &[u8; 98], out: &mut [f32; QK_K]) {
    let b = bytemuck::pod_read_unaligned::<BlockIq3Xxs>(block);
    let d = half(b.d);
    let mut next = 0_usize;
    for ib in 0..QK_K / 32 {
        // The sign-and-scale word follows the 64 grid-index bytes.
        let s = QK_K / 4 + 4 * ib;
        let aux = u32::from_le_bytes([b.qs[s], b.qs[s + 1], b.qs[s + 2], b.qs[s + 3]]);
        let db = d * (0.5 + f32::from(((aux >> 28) & 0xF) as u8)) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 0x7F) as usize];
            let g1 = IQ3XXS_GRID[usize::from(b.qs[8 * ib + 2 * l])];
            let g2 = IQ3XXS_GRID[usize::from(b.qs[8 * ib + 2 * l + 1])];
            for j in 0..4 {
                out[next + j] = db * f32::from(byte32(g1, j)) * signed(signs, j);
                out[next + j + 4] = db * f32::from(byte32(g2, j)) * signed(signs, j + 4);
            }
            next += 8;
        }
    }
}

/// Decodes one [`DType::IQ3_S`] block into 256 weights.
pub fn decode_iq3_s_block(block: &[u8; 110], out: &mut [f32; QK_K]) {
    let b = bytemuck::pod_read_unaligned::<BlockIq3S>(block);
    let d = half(b.d);
    let mut next = 0_usize;
    // One scale byte covers two 32-weight groups, and one `qh` byte the
    // high bits of one group's eight indices.
    for g in 0..QK_K / 64 {
        let sc = b.scales[g];
        let db = [
            d * (1.0 + 2.0 * f32::from(sc & 0xF)),
            d * (1.0 + 2.0 * f32::from(sc >> 4)),
        ];
        // `h` indexes `db` but also the `qh` byte, the `qs` half and the
        // `signs` half of the pair, so the loop stays index-shaped.
        #[expect(
            clippy::needless_range_loop,
            reason = "h indexes four parallel arrays, not just db"
        )]
        for h in 0..2 {
            let qh = u32::from(b.qh[2 * g + h]);
            let qbase = 8 * (2 * g + h);
            let sbase = 4 * (2 * g + h);
            for l in 0..4 {
                let g1 = IQ3S_GRID
                    [usize::from(b.qs[qbase + 2 * l]) | (((qh << (8 - 2 * l)) & 256) as usize)];
                let g2 = IQ3S_GRID
                    [usize::from(b.qs[qbase + 2 * l + 1]) | (((qh << (7 - 2 * l)) & 256) as usize)];
                let signs = b.signs[sbase + l];
                for j in 0..4 {
                    out[next + j] = db[h] * f32::from(byte32(g1, j)) * signed(signs, j);
                    out[next + j + 4] = db[h] * f32::from(byte32(g2, j)) * signed(signs, j + 4);
                }
                next += 8;
            }
        }
    }
}

/// Decodes one [`DType::IQ1_S`] block into 256 weights.
pub fn decode_iq1_s_block(block: &[u8; 50], out: &mut [f32; QK_K]) {
    let b = bytemuck::pod_read_unaligned::<BlockIq1S>(block);
    let d = half(b.d);
    let mut next = 0_usize;
    for ib in 0..QK_K / 32 {
        let qh = b.qh[ib];
        let dl = d * f32::from(2 * ((qh >> 12) & 7) + 1);
        let delta = if qh & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        for l in 0..4 {
            let idx = usize::from(b.qs[4 * ib + l]) | (usize::from((qh >> (3 * l)) & 7) << 8);
            let grid = IQ1S_GRID[idx];
            for j in 0..8 {
                out[next + j] = dl * (f32::from(sbyte64(grid, j)) + delta);
            }
            next += 8;
        }
    }
}

/// Decodes one [`DType::IQ1_M`] block into 256 weights.
pub fn decode_iq1_m_block(block: &[u8; 56], out: &mut [f32; QK_K]) {
    let b = bytemuck::pod_read_unaligned::<BlockIq1M>(block);
    // The scale words pack a 3-bit multiplier per 16 weights in their low
    // six bits; the top nibble of each odd byte is free and carries the
    // block `f16` instead, four nibbles to the word pair.
    let word = |i: usize| u16::from_le_bytes([b.scales[2 * i], b.scales[2 * i + 1]]);
    let scale = (word(0) >> 12)
        | ((word(1) >> 8) & 0x00F0)
        | ((word(2) >> 4) & 0x0F00)
        | (word(3) & 0xF000);
    let d = half(scale);
    let mut next = 0_usize;
    for ib in 0..QK_K / 32 {
        let w = word(ib / 2);
        let shift = 6 * (ib % 2);
        let dl = [
            d * f32::from(2 * ((w >> shift) & 7) + 1),
            d * f32::from(2 * ((w >> (shift + 3)) & 7) + 1),
        ];
        let qh0 = b.qh[2 * ib];
        let qh1 = b.qh[2 * ib + 1];
        let idx = [
            usize::from(b.qs[4 * ib]) | ((usize::from(qh0) << 8) & 0x700),
            usize::from(b.qs[4 * ib + 1]) | ((usize::from(qh0) << 4) & 0x700),
            usize::from(b.qs[4 * ib + 2]) | ((usize::from(qh1) << 8) & 0x700),
            usize::from(b.qs[4 * ib + 3]) | ((usize::from(qh1) << 4) & 0x700),
        ];
        let delta = [
            delta(qh0 & 0x08),
            delta(qh0 & 0x80),
            delta(qh1 & 0x08),
            delta(qh1 & 0x80),
        ];
        for m in 0..4 {
            let grid = IQ1S_GRID[idx[m]];
            for j in 0..8 {
                out[next + j] = dl[m / 2] * (f32::from(sbyte64(grid, j)) + delta[m]);
            }
            next += 8;
        }
    }
}

/// Decodes one [`DType::IQ4_XS`] block into 256 weights.
pub fn decode_iq4_xs_block(block: &[u8; 136], out: &mut [f32; QK_K]) {
    let b = bytemuck::pod_read_unaligned::<BlockIq4Xs>(block);
    let d = half(b.d);
    let mut next = 0_usize;
    for ib in 0..QK_K / 32 {
        let ls = u16::from((b.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF)
            | ((b.scales_h >> (2 * ib)) & 3) << 4;
        let dl = d * (f32::from(ls) - 32.0);
        for j in 0..16 {
            let q = b.qs[16 * ib + j];
            out[next + j] = dl * f32::from(IQ4NL_VALUES[usize::from(q & 0xF)]);
            out[next + j + 16] = dl * f32::from(IQ4NL_VALUES[usize::from(q >> 4)]);
        }
        next += 32;
    }
}

/// Decodes one block of a mixed `i`-quant or [`DType::Q2_K`] tensor.
///
/// `block` must be exactly the block size the `dtype` names and `out` exactly
/// 256 weights. Formats this module does not decode - including the ones with
/// their own decoders elsewhere in the crate - are rejected rather than
/// guessed at.
///
/// # Errors
///
/// [`GgufError::LayoutMismatch`] when `dtype` is not one of the formats above,
/// `block` is not exactly the format's block, or `out` is not 256 weights.
pub fn decode_block(dtype: DType, block: &[u8], out: &mut [f32]) -> Result<(), GgufError> {
    let got = out.len();
    let out: &mut [f32; QK_K] = out
        .try_into()
        .map_err(|_| layout_error("256 output weights", format!("{got} weights")))?;
    match dtype {
        DType::Q2_K => decode_q2_k_block(&fixed_bytes::<84>(block, dtype)?, out),
        DType::IQ2_XXS => decode_iq2_xxs_block(&fixed_bytes::<66>(block, dtype)?, out),
        DType::IQ2_XS => decode_iq2_xs_block(&fixed_bytes::<74>(block, dtype)?, out),
        DType::IQ2_S => decode_iq2_s_block(&fixed_bytes::<82>(block, dtype)?, out),
        DType::IQ3_XXS => decode_iq3_xxs_block(&fixed_bytes::<98>(block, dtype)?, out),
        DType::IQ3_S => decode_iq3_s_block(&fixed_bytes::<110>(block, dtype)?, out),
        DType::IQ1_S => decode_iq1_s_block(&fixed_bytes::<50>(block, dtype)?, out),
        DType::IQ1_M => decode_iq1_m_block(&fixed_bytes::<56>(block, dtype)?, out),
        DType::IQ4_XS => decode_iq4_xs_block(&fixed_bytes::<136>(block, dtype)?, out),
        other => {
            return Err(layout_error(
                "q2_K or an i-quant block",
                format!("{other:?}"),
            ));
        }
    }
    Ok(())
}
