//! The pure half of the tier: row identifiers, and one row's weights.
//!
//! Nothing here opens a file, allocates, or locks. That is deliberate - this is
//! the part that has to reproduce the reference arithmetic bit for bit, and a
//! function that only reads its arguments is a function a test can pin against
//! an oracle.
//!
//! # Two row codecs, because the checkpoint decides
//!
//! A checkpoint ships its n-gram table in one of two formats and says which in
//! the tensor's own type, so both are implemented and the format is a parameter
//! rather than a build decision. One packs thirty-two weights into an
//! eighteen-byte block with a shared half-precision scale. The other is a
//! six-bit tail-biting trellis over the whole row - one half-precision row
//! scale, then a hundred and twenty bytes of ring - decoded through a
//! multiply-and-sum codebook and finished with a per-head bias.
//!
//! # Why the rounding is written out
//!
//! The trellis codec rounds through half precision twice: once on the codebook
//! value and once on the finished weight. Both roundings are load-bearing - the
//! reference decoder does them, the device kernel does them, and dropping
//! either moves the last bits of every weight in the table. So the two
//! conversions are transcribed here rather than delegated, including the
//! subnormal paths, which the rows actually reach.

use shiso_core::ids::TokenId;

use crate::handle::{PleParams, PleRowType};
use crate::{PleError, PleResult};

/// Gram widths a table may be partitioned into.
///
/// The hash reads at most this many trailing tokens, so it is also the width of
/// the multiplier table a checkpoint has to supply.
pub const MAX_NGRAM: u32 = 4;

/// Hash heads a table may have.
///
/// One row is read per head per position, so this bounds a decode step's gather
/// as well as the offset and vocabulary tables.
pub const MAX_HEADS: u32 = 32;

/// Weights one block of the block-quantized codec carries.
pub const BLOCK_ELEMS: u32 = 32;

/// Bytes one block of the block-quantized codec occupies: a half-precision
/// scale, then thirty-two four-bit weights.
pub const BLOCK_BYTES: u32 = 18;

/// Bits one weight of the trellis codec occupies.
pub const TRELLIS_BITS: u32 = 6;

/// The row width the trellis codec is reachable at.
///
/// The ring wraps at the row width, so the width is part of the code rather
/// than a loop bound: a row of a different width is a different code and this
/// decoder would silently produce the wrong weights for it.
pub const TRELLIS_HEAD_DIM: u32 = 160;

/// Half-precision words one trellis row occupies, scale included.
const TRELLIS_WORDS: usize = 1 + (TRELLIS_HEAD_DIM as usize) * (TRELLIS_BITS as usize) / 16;

/// The sixteen levels the block codec's four-bit weights index.
const NL_LEVELS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Bytes one row occupies in the file.
///
/// The row stride is what turns a row identifier into a file offset, so it is
/// derived from the format in exactly one place and every caller reads it from
/// here.
#[must_use]
pub const fn row_bytes(params: &PleParams) -> u32 {
    match params.row_type() {
        PleRowType::TrellisK6 => 2 + params.head_dim() * TRELLIS_BITS / 8,
        PleRowType::BlockNl4 => params.head_dim() / BLOCK_ELEMS * BLOCK_BYTES,
        // Codes at four bits, then one scale and one bias per group, both
        // brain floats.
        PleRowType::MlxAffine4 => params.head_dim() / 2 + params.head_dim() / BLOCK_ELEMS * 4,
        PleRowType::Fp8E4M3 { .. } => params.head_dim(),
    }
}

/// Writes the row identifiers for positions `pos0 .. pos0 + n`.
///
/// `tokens` holds the session's path by absolute position and must reach
/// `pos0 + n`, because a position's grams read the tokens behind it. The output
/// is token-major: `n` groups of one identifier per head.
///
/// # The window, and what cuts it
///
/// A sequence boundary anywhere inside a position's window truncates it, and
/// the truncation is transitive: once cut, every older slot of that window
/// reads as the boundary token too. A token that *is* the boundary does not cut
/// its own context, which is why the scan starts one token back. Positions
/// before the start of the path read as the boundary token as well, so the
/// first token of a session hashes the same grams whether or not anything
/// precedes it.
///
/// # Errors
///
/// [`PleError::BufferTooSmall`] when `out` cannot hold `n * n_heads`
/// identifiers or `tokens` does not reach `pos0 + n`. A short buffer is a
/// caller bug and truncating it would hand back a gather that looks complete.
pub fn row_ids(
    params: &PleParams,
    tokens: &[TokenId],
    pos0: u32,
    n: u32,
    out: &mut [u64],
) -> PleResult<()> {
    let heads = params.n_heads();
    let want = u64::from(n) * u64::from(heads);
    if (out.len() as u64) < want {
        return Err(PleError::BufferTooSmall {
            what: "row identifier output",
            want,
            got: out.len() as u64,
        });
    }
    let reach = u64::from(pos0) + u64::from(n);
    if (tokens.len() as u64) < reach {
        return Err(PleError::BufferTooSmall {
            what: "token history",
            want: reach,
            got: tokens.len() as u64,
        });
    }

    let eos = u64::from(params.eos_token().get());
    let ngram = params.ngram_size();
    let per_order = params.heads_per_ngram();

    for i in 0..n {
        let pos = pos0 + i;
        let mut ctx = [eos; MAX_NGRAM as usize];
        ctx[0] = u64::from(tokens[pos as usize].get());
        let mut cut = false;
        for s in 1..ngram {
            let have = !cut && pos >= s;
            let token = if have {
                u64::from(tokens[(pos - s) as usize].get())
            } else {
                eos
            };
            cut = cut || !have || token == eos;
            ctx[s as usize] = if cut { eos } else { token };
        }

        // One hash per gram width; the heads of a width share it and differ
        // only by their modulus and their offset into the table. Everything is
        // unsigned and wraps, which is what makes the remainder non-negative
        // without a fixup.
        let base = (i as usize) * (heads as usize);
        for order in 2..=ngram {
            let mut mixed = ctx[0].wrapping_mul(params.multiplier(0));
            for j in 1..order {
                mixed ^= ctx[j as usize].wrapping_mul(params.multiplier(j));
            }
            let group = (order - 2) * per_order;
            for g in 0..per_order {
                let h = (group + g) as usize;
                out[base + h] = mixed % params.head_vocab(h) + params.head_offset(h);
            }
        }
    }
    Ok(())
}

/// Decodes one row's quantized bytes into `head_dim` weights.
///
/// `head` is the hash head the row belongs to, which is the row's index modulo
/// the head count. The trellis codec adds that head's bias; the block codec has
/// none.
///
/// # Errors
///
/// [`PleError::BufferTooSmall`] when `src` is shorter than one row or `dst`
/// cannot hold one row's weights.
pub fn dequant_row(params: &PleParams, src: &[u8], head: u32, dst: &mut [f32]) -> PleResult<()> {
    let stride = row_bytes(params) as usize;
    if src.len() < stride {
        return Err(PleError::BufferTooSmall {
            what: "quantized row",
            want: stride as u64,
            got: src.len() as u64,
        });
    }
    let width = params.head_dim() as usize;
    if dst.len() < width {
        return Err(PleError::BufferTooSmall {
            what: "decoded row",
            want: width as u64,
            got: dst.len() as u64,
        });
    }
    match params.row_type() {
        PleRowType::TrellisK6 => decode_trellis(params, src, head, dst),
        PleRowType::BlockNl4 => decode_block(src, width, dst),
        PleRowType::MlxAffine4 => decode_affine(src, width, dst),
        PleRowType::Fp8E4M3 { scale_bits } => {
            let scale = f32::from_bits(scale_bits);
            for (out, &byte) in dst.iter_mut().zip(src).take(width) {
                *out = e4m3(byte) * scale;
            }
        }
    }
    Ok(())
}

/// One FP8 E4M3 byte: one sign, four exponent bits biased by seven, three
/// mantissa bits, no infinities and the all-ones pattern not a number.
#[must_use]
pub fn e4m3(byte: u8) -> f32 {
    let e = i32::from((byte >> 3) & 0xF);
    let bits = byte & 7;
    let m = f32::from(bits);
    let v = if e == 0 {
        m * 2f32.powi(-9)
    } else if e == 15 && bits == 7 {
        f32::NAN
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if byte & 0x80 == 0 { v } else { -v }
}

/// The affine codec: `code * scale + bias` per weight, one scale and bias per
/// thirty-two weights, codes low nibble first.
fn decode_affine(src: &[u8], width: usize, dst: &mut [f32]) {
    let groups = width / BLOCK_ELEMS as usize;
    let params = &src[width / 2..];
    let bf16 = |at: usize| {
        f32::from_bits(u32::from(u16::from_le_bytes([params[2 * at], params[2 * at + 1]])) << 16)
    };
    for (i, out) in dst.iter_mut().enumerate().take(width) {
        let byte = src[i / 2];
        let q = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let g = i / BLOCK_ELEMS as usize;
        *out = f32::from(q) * bf16(g) + bf16(groups + g);
    }
}

/// The block codec: a half-precision scale per thirty-two weights, then the
/// weights as four-bit indices into a fixed level table, low nibbles first and
/// high nibbles sixteen apart.
fn decode_block(src: &[u8], width: usize, dst: &mut [f32]) {
    let blocks = width / BLOCK_ELEMS as usize;
    for b in 0..blocks {
        let blk = &src[b * BLOCK_BYTES as usize..];
        let scale = half_to_float(u16::from_le_bytes([blk[0], blk[1]]));
        let out = &mut dst[b * BLOCK_ELEMS as usize..];
        for j in 0..BLOCK_ELEMS as usize / 2 {
            let q = blk[2 + j];
            out[j] = scale * f32::from(NL_LEVELS[(q & 0x0F) as usize]);
            out[j + 16] = scale * f32::from(NL_LEVELS[(q >> 4) as usize]);
        }
    }
}

/// The trellis codec.
///
/// Bit `m` of weight `i` sits at ring position `((i - m/6) mod width) * 6 +
/// m % 6`, so the sixteen bits of a weight's state are scattered backwards
/// around the row and the row's last weights wrap into its first bytes - which
/// is what "tail-biting" means and why the row cannot be decoded a block at a
/// time. The state goes through the codebook as a multiply whose four bytes are
/// summed, and the sum is read as a half-precision number offset by 1024. Then
/// one fused multiply-add against the row scale and the head bias, and one last
/// rounding to half precision, exactly as the device kernel contracts it.
fn decode_trellis(params: &PleParams, src: &[u8], head: u32, dst: &mut [f32]) {
    let width = params.head_dim() as usize;
    let mut words = [0u16; TRELLIS_WORDS];
    let used = 1 + width * TRELLIS_BITS as usize / 16;
    for (w, slot) in words.iter_mut().enumerate().take(used) {
        *slot = u16::from_le_bytes([src[w * 2], src[w * 2 + 1]]);
    }
    let scale = half_to_float(words[0]);
    let codebook_scale = half_to_float(0x1eee);
    let codebook_bias = half_to_float(0xc931);
    let bias = params.head_bias_row(head);

    for i in 0..width {
        let mut state = 0u32;
        for m in 0..16u32 {
            let back = (m / TRELLIS_BITS) as usize;
            let pos = if i >= back {
                i - back
            } else {
                i + width - back
            };
            let sb = pos * TRELLIS_BITS as usize + (m % TRELLIS_BITS) as usize;
            let bit = u32::from(words[1 + (sb >> 4)] >> (sb & 15)) & 1;
            state |= bit << m;
        }
        let prod = state.wrapping_mul(0x83DC_D12D);
        let sum = (prod & 0xFF) + ((prod >> 8) & 0xFF) + ((prod >> 16) & 0xFF) + (prod >> 24);
        let level = 1024.0_f32 + sum_as_f32(sum);
        let code = half_to_float(float_to_half(level * codebook_scale + codebook_bias));
        let offset = bias.map_or(0.0, |row| half_to_float(row[i]));
        dst[i] = half_to_float(float_to_half(code.mul_add(scale, offset)));
    }
}

/// The byte sum of a codebook product, as a float.
///
/// Bounded by four times 255, so the conversion is exact and the lint that
/// warns about it is answered once here rather than at the call site.
#[expect(
    clippy::cast_precision_loss,
    reason = "the sum of four bytes is at most 1020 and every such integer is exactly representable"
)]
const fn sum_as_f32(sum: u32) -> f32 {
    sum as f32
}

/// The low sixteen bits of a value already known to fit them.
const fn low16(bits: u32) -> u16 {
    (bits & 0xFFFF) as u16
}

/// Half precision to single, subnormals renormalized rather than flushed.
///
/// The rows reach values below the smallest normal half, and a decoder that
/// flushed them would zero real weights.
#[must_use]
pub const fn half_to_float(half: u16) -> f32 {
    let sign = ((half as u32) & 0x8000) << 16;
    let exp = ((half as u32) >> 10) & 0x1F;
    let mant = (half as u32) & 0x3FF;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut shifted = mant;
            let mut extra = 0u32;
            while shifted & 0x400 == 0 {
                shifted <<= 1;
                extra += 1;
            }
            sign | ((127 - 15 - extra + 1) << 23) | ((shifted & 0x3FF) << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Single precision to half, round to nearest with ties to even, subnormals
/// included.
///
/// The reference rounds this way and so does the device intrinsic the kernel
/// uses; a truncating conversion here would disagree with both in the last bit
/// of most weights.
#[must_use]
pub const fn float_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let magnitude = bits & 0x7FFF_FFFF;
    if magnitude >= 0x7F80_0000 {
        // Infinity keeps its sign; a NaN keeps a payload bit so it stays a NaN.
        let payload = if magnitude.trailing_zeros() >= 23 {
            0
        } else {
            0x200
        };
        return low16(sign | 0x7C00 | payload);
    }
    if magnitude >= 0x477F_F000 {
        return low16(sign | 0x7C00);
    }
    if magnitude < 0x3300_0000 {
        return low16(sign);
    }
    let exp = magnitude >> 23;
    let mant = magnitude & 0x7F_FFFF;
    if magnitude < 0x3880_0000 {
        // Subnormal: the significand with its hidden bit, scaled into units of
        // two to the minus twenty-four.
        let full = mant | 0x80_0000;
        let shift = 126 - exp;
        let mut low = full >> shift;
        let rest = full & ((1 << shift) - 1);
        let midpoint = 1 << (shift - 1);
        if rest > midpoint || (rest == midpoint && low & 1 == 1) {
            low += 1;
        }
        return low16(sign | low);
    }
    let mut half = sign | ((exp - 112) << 10) | (mant >> 13);
    let rest = mant & 0x1FFF;
    if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    low16(half)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_survives_a_round_trip_through_single() {
        for raw in 0u32..=0xFFFF {
            let half = low16(raw);
            let exp = (half >> 10) & 0x1F;
            if exp == 31 {
                continue; // infinities and NaNs are not round-tripped by value
            }
            let back = float_to_half(half_to_float(half));
            assert_eq!(back, half, "half {half:#06x} did not survive");
        }
    }

    #[test]
    fn the_smallest_subnormal_half_is_renormalized_rather_than_flushed() {
        // 2^-24, the smallest positive half, is what a flushing converter loses.
        assert_eq!(half_to_float(1).to_bits(), (2.0_f32).powi(-24).to_bits());
        assert_eq!(float_to_half((2.0_f32).powi(-24)), 1);
    }

    #[test]
    fn rounding_to_half_breaks_ties_towards_the_even_neighbour() {
        // Halfway between 1.0 and the next half up, which is 1 + 2^-10.
        let tie_down = 1.0_f32 + (2.0_f32).powi(-11);
        assert_eq!(float_to_half(tie_down), float_to_half(1.0), "ties to even");
        // Halfway between 1 + 2^-10 and 1 + 2^-9 rounds up, to the even one.
        let tie_up = 1.0_f32 + (2.0_f32).powi(-10) + (2.0_f32).powi(-11);
        assert_eq!(float_to_half(tie_up), float_to_half(1.0) + 2);
    }

    #[test]
    fn a_value_past_the_half_range_saturates_to_infinity_with_its_sign() {
        assert_eq!(float_to_half(1.0e9), 0x7C00);
        assert_eq!(float_to_half(-1.0e9), 0xFC00);
        assert_eq!(float_to_half(f32::INFINITY), 0x7C00);
    }
}
