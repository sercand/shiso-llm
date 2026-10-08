//! Quantized block geometry.
//!
//! Every format a tensor can be stored in gets one `#[repr(C)]` struct whose
//! fields appear in the same order and width as the format defines them, and
//! every one of those structs carries a compile-time size assert. The point of
//! the assert is narrow and important: a field transcribed in the wrong order,
//! or a padding rule misunderstood, becomes a build failure instead of a wrong
//! tensor read during inference. Byte-offset constants and hand-rolled
//! little-endian readers cannot fail that way, which is why they are not used
//! here.
//!
//! Nothing in this module dequantizes anything. Decoding is kernel-side work;
//! this module answers "how wide is one block, and how many bytes does this
//! tensor occupy".

use std::mem::size_of;

use shiso_core::DType;

use crate::reader::TensorShape;

/// Elements per block for the `K`-quant and `i`-quant families.
pub const QK_K: usize = 256;
/// Elements per block for the microscaling 4-bit format.
pub const QK_MXFP4: usize = 32;
/// Elements per block for the legacy 32-element formats.
pub const QK_LEGACY: usize = 32;

/// Implemented by every [`QuantBlock`] struct, so a borrowed payload can be
/// reinterpreted as a slice of blocks with no unsafe code at the call site.
///
/// `Pod` already guarantees no padding, `Copy`, and any-bit-pattern validity;
/// `DTYPE` and `ELEMS` tie the struct to the format id and element count it
/// describes, which is what lets [`crate::reader::TensorView::blocks`] check
/// the tensor's declared type before casting.
pub trait QuantBlock: bytemuck::Pod {
    /// The format this struct lays out.
    const DTYPE: DType;
    /// Elements covered by one block.
    const ELEMS: usize;
}

/// 8-bit block: one fp16 scale and 32 signed weights.
///
/// This format has no historical C struct to transcribe; its geometry is the
/// type table's `q8_0` row, 32 elements in 34 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ8_0 {
    /// fp16 scale, kept as bits so the struct stays plain-old-data.
    pub d: u16,
    /// The 32 signed weights.
    pub qs: [i8; 32],
}
const _: () = assert!(size_of::<BlockQ8_0>() == 34);
impl QuantBlock for BlockQ8_0 {
    const DTYPE: DType = DType::Q8_0;
    const ELEMS: usize = QK_LEGACY;
}

/// 2-bit `K`-quant: per-subblock scales and minima plus packed weights.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ2K {
    /// One scale nibble pair per 16-element subblock.
    pub scales: [u8; QK_K / 16],
    /// Packed 2-bit weights, four per byte.
    pub qs: [u8; QK_K / 4],
    /// fp16 block scale, high bits.
    pub d: u16,
    /// fp16 block minimum, high bits.
    pub dmin: u16,
}
const _: () = assert!(size_of::<BlockQ2K>() == 84);
impl QuantBlock for BlockQ2K {
    const DTYPE: DType = DType::Q2_K;
    const ELEMS: usize = QK_K;
}

/// Elements one ternary block covers.
pub const QK_TERNARY: usize = 128;

/// Ternary weights in 2-bit slots: an `f16` scale, then four codes a byte.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockPq2_0 {
    /// fp16 scale, high bits.
    pub d: u16,
    /// Packed 2-bit codes, four per byte from the low bits up.
    pub qs: [u8; QK_TERNARY / 4],
}
const _: () = assert!(size_of::<BlockPq2_0>() == 34);
impl QuantBlock for BlockPq2_0 {
    const DTYPE: DType = DType::Pq2_0;
    const ELEMS: usize = QK_TERNARY;
}

/// Ternary weights five trits to a byte, with the `f16` scale last.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockPtq1_0 {
    /// Five trits a byte: 120 weights.
    pub qs: [u8; 24],
    /// Four trits a byte: the last 8 weights.
    pub qh: [u8; 2],
    /// fp16 scale, high bits.
    pub d: u16,
}
const _: () = assert!(size_of::<BlockPtq1_0>() == 28);
impl QuantBlock for BlockPtq1_0 {
    const DTYPE: DType = DType::Ptq1_0;
    const ELEMS: usize = QK_TERNARY;
}

/// 4-bit `K`-quant: super-block scale and minimum, 12 scale bytes, packed
/// 4-bit weights.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ4K {
    /// fp16 super-block scale.
    pub d: u16,
    /// fp16 super-block minimum.
    pub dmin: u16,
    /// Six 6-bit scale/min pairs packed into 12 bytes.
    pub scales: [u8; 12],
    /// Packed 4-bit weights, two per byte.
    pub qs: [u8; QK_K / 2],
}
const _: () = assert!(size_of::<BlockQ4K>() == 144);
impl QuantBlock for BlockQ4K {
    const DTYPE: DType = DType::Q4_K;
    const ELEMS: usize = QK_K;
}

/// 5-bit `K`-quant: like [`BlockQ4K`] plus the fifth bit plane.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ5K {
    /// fp16 super-block scale.
    pub d: u16,
    /// fp16 super-block minimum.
    pub dmin: u16,
    /// Six 6-bit scale/min pairs packed into 12 bytes.
    pub scales: [u8; 12],
    /// High bit plane, one bit per weight.
    pub qh: [u8; QK_K / 8],
    /// Low four bits of each weight.
    pub qs: [u8; QK_K / 2],
}
const _: () = assert!(size_of::<BlockQ5K>() == 176);
impl QuantBlock for BlockQ5K {
    const DTYPE: DType = DType::Q5_K;
    const ELEMS: usize = QK_K;
}

/// 6-bit `K`-quant: two weight planes, 16 signed scales, one fp16 scale.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ6K {
    /// Low four bits of each weight.
    pub ql: [u8; QK_K / 2],
    /// High two bits of each weight, four per byte.
    pub qh: [u8; QK_K / 4],
    /// One signed scale per 16-element subblock.
    pub scales: [i8; QK_K / 16],
    /// fp16 block scale.
    pub d: u16,
}
const _: () = assert!(size_of::<BlockQ6K>() == 210);
impl QuantBlock for BlockQ6K {
    const DTYPE: DType = DType::Q6_K;
    const ELEMS: usize = QK_K;
}

/// 8-bit `K`-quant: a block-wide `f32` scale, 256 signed weights, and the
/// per-subblock sums a dot product accumulates.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ8K {
    /// Block scale as `f32`, unlike the fp16 scales of the other `K` formats.
    pub d: f32,
    /// The 256 signed weights.
    pub qs: [i8; QK_K],
    /// Precomputed sum of each 16-element subblock.
    pub bsums: [i16; QK_K / 16],
}
const _: () = assert!(size_of::<BlockQ8K>() == 292);
impl QuantBlock for BlockQ8K {
    const DTYPE: DType = DType::Q8_K;
    const ELEMS: usize = QK_K;
}

/// 2-bit `i`-quant, extra-small: an fp16 scale and 32 grid indices.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIq2Xxs {
    /// fp16 scale.
    pub d: u16,
    /// Eight 8-bit indices per `u16`, decoding through
    /// [`crate::quant_tables::IQ2XXS_GRID`] and
    /// [`crate::quant_tables::KSIGNS_IQ2XS`].
    pub qs: [u16; QK_K / 8],
}
const _: () = assert!(size_of::<BlockIq2Xxs>() == 66);
impl QuantBlock for BlockIq2Xxs {
    const DTYPE: DType = DType::IQ2_XXS;
    const ELEMS: usize = QK_K;
}

/// 2-bit `i`-quant, small: an fp16 scale, grid indices with the sign folded
/// into each index word, and 4-bit sub-block scales.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIq2Xs {
    /// fp16 scale.
    pub d: u16,
    /// Eight weights per `u16`: the low nine bits index
    /// [`crate::quant_tables::IQ2XS_GRID`] and the top seven select one of
    /// the 128 sign patterns ([`crate::quant_tables::KSIGNS64`] is their
    /// mask form).
    pub qs: [u16; QK_K / 8],
    /// Two 4-bit scales a byte, one per 16-weight half of a 32-weight group.
    pub scales: [u8; QK_K / 32],
}
const _: () = assert!(size_of::<BlockIq2Xs>() == 74);
impl QuantBlock for BlockIq2Xs {
    const DTYPE: DType = DType::IQ2_XS;
    const ELEMS: usize = QK_K;
}

/// 2-bit `i`-quant, small with explicit signs: a wider grid than
/// [`BlockIq2Xs`] and one sign bit per weight instead of per index word.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIq2S {
    /// fp16 scale.
    pub d: u16,
    /// Low eight bits of each index into [`crate::quant_tables::IQ2S_GRID`],
    /// then 32 bytes of signs, one bit per weight.
    pub qs: [u8; QK_K / 4],
    /// High two bits of each grid index, four indices a byte.
    pub qh: [u8; QK_K / 32],
    /// Two 4-bit scales a byte, one per 16-weight half of a 32-weight group.
    pub scales: [u8; QK_K / 32],
}
const _: () = assert!(size_of::<BlockIq2S>() == 82);
impl QuantBlock for BlockIq2S {
    const DTYPE: DType = DType::IQ2_S;
    const ELEMS: usize = QK_K;
}

/// 3-bit `i`-quant, extra-extra-small: an fp16 scale and one packed byte
/// stream of grid indices and sign-plus-scale words.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIq3Xxs {
    /// fp16 scale.
    pub d: u16,
    /// 64 indices into [`crate::quant_tables::IQ3XXS_GRID`], then 32 bytes
    /// each packing four 7-bit sign-pattern indices and a 4-bit group scale.
    pub qs: [u8; 3 * QK_K / 8],
}
const _: () = assert!(size_of::<BlockIq3Xxs>() == 98);
impl QuantBlock for BlockIq3Xxs {
    const DTYPE: DType = DType::IQ3_XXS;
    const ELEMS: usize = QK_K;
}

/// 3-bit `i`-quant, small: grid indices split across two planes, per-weight
/// signs, and 4-bit group scales.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIq3S {
    /// fp16 scale.
    pub d: u16,
    /// Low eight bits of each index into [`crate::quant_tables::IQ3S_GRID`].
    pub qs: [u8; QK_K / 4],
    /// High bit pair of each grid index, four indices a byte per group.
    pub qh: [u8; QK_K / 32],
    /// One sign bit per weight.
    pub signs: [u8; QK_K / 8],
    /// Two 4-bit scales a byte, one per 32-weight group.
    pub scales: [u8; QK_K / 64],
}
const _: () = assert!(size_of::<BlockIq3S>() == 110);
impl QuantBlock for BlockIq3S {
    const DTYPE: DType = DType::IQ3_S;
    const ELEMS: usize = QK_K;
}

/// 1-bit `i`-quant, small: grid indices whose high bits, scale multiplier,
/// and grid-shift sign share one `u16` per 32-weight group.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIq1S {
    /// fp16 scale.
    pub d: u16,
    /// Low eight bits of each index into [`crate::quant_tables::IQ1S_GRID`].
    pub qs: [u8; QK_K / 8],
    /// Per 32-weight group: the high three bits of four grid indices, a 3-bit
    /// scale multiplier, and the sign of the grid shift.
    pub qh: [u16; QK_K / 32],
}
const _: () = assert!(size_of::<BlockIq1S>() == 50);
impl QuantBlock for BlockIq1S {
    const DTYPE: DType = DType::IQ1_S;
    const ELEMS: usize = QK_K;
}

/// 1-bit `i`-quant, medium: like [`BlockIq1S`] but with no `d` field - the
/// fp16 block scale is assembled from nibbles of the scale bytes at decode.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIq1M {
    /// Low eight bits of each index into [`crate::quant_tables::IQ1S_GRID`].
    pub qs: [u8; QK_K / 8],
    /// High three bits of two grid indices a byte, plus each pair's
    /// grid-shift sign.
    pub qh: [u8; QK_K / 16],
    /// 3-bit scale per 16 weights; the top nibbles of these bytes also carry
    /// the fp16 block scale.
    pub scales: [u8; QK_K / 32],
}
const _: () = assert!(size_of::<BlockIq1M>() == 56);
impl QuantBlock for BlockIq1M {
    const DTYPE: DType = DType::IQ1_M;
    const ELEMS: usize = QK_K;
}

/// 4-bit `i`-quant, extra-small: an fp16 scale, 6-bit sub-block scales split
/// across two planes, and nibble indices into the 4-bit non-linear codebook.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIq4Xs {
    /// fp16 scale.
    pub d: u16,
    /// High two bits of each of the eight 6-bit sub-block scales.
    pub scales_h: u16,
    /// Low four bits of each sub-block scale, two a byte.
    pub scales_l: [u8; QK_K / 64],
    /// 4-bit indices into [`crate::quant_tables::IQ4NL_VALUES`], two a byte.
    pub qs: [u8; QK_K / 2],
}
const _: () = assert!(size_of::<BlockIq4Xs>() == 136);
impl QuantBlock for BlockIq4Xs {
    const DTYPE: DType = DType::IQ4_XS;
    const ELEMS: usize = QK_K;
}

/// Microscaling 4-bit: one shared exponent plus 16 packed nibbles.
///
/// Note the size, 17 bytes: the shared exponent is a single byte, not the fp16
/// pair the other 32-element formats lead with.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockMxfp4 {
    /// Shared exponent, `e8m0` encoding.
    pub e: u8,
    /// Two 4-bit weights per byte.
    pub qs: [u8; QK_MXFP4 / 2],
}
const _: () = assert!(size_of::<BlockMxfp4>() == 17);
impl QuantBlock for BlockMxfp4 {
    const DTYPE: DType = DType::MXFP4;
    const ELEMS: usize = QK_MXFP4;
}

/// Legacy 5-bit block: scale, minimum, high-bit plane, low nibbles.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ5_1 {
    /// fp16 scale.
    pub d: u16,
    /// fp16 minimum.
    pub m: u16,
    /// Fifth bit of each weight.
    pub qh: [u8; 4],
    /// Low four bits of each weight.
    pub qs: [u8; QK_LEGACY / 2],
}
const _: () = assert!(size_of::<BlockQ5_1>() == 24);
impl QuantBlock for BlockQ5_1 {
    const DTYPE: DType = DType::Q5_1;
    const ELEMS: usize = QK_LEGACY;
}

/// 4-bit non-linear block: an fp16 scale and indices into a fixed 16-entry
/// table.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockIq4Nl {
    /// fp16 scale.
    pub d: u16,
    /// 4-bit indices into [`crate::quant_tables::IQ4NL_VALUES`].
    pub qs: [u8; QK_LEGACY / 2],
}
const _: () = assert!(size_of::<BlockIq4Nl>() == 18);
impl QuantBlock for BlockIq4Nl {
    const DTYPE: DType = DType::IQ4_NL;
    const ELEMS: usize = QK_LEGACY;
}

/// Geometry of the trellis and n-gram codecs.
///
/// These formats are not `#[repr(C)]` blocks: a trellis block is a 16x16 tile
/// of `K`-bit weights, so its width depends on `K` and its decoding is a
/// Hadamard-domain walk. There is therefore no struct to mirror, and the
/// geometry lives here as constants plus the two derived sizes the reader
/// needs.
pub mod exl3 {
    use shiso_core::DType;

    use crate::reader::TensorShape;

    /// Weights in one trellis tile.
    pub const TILE_ELEMS: usize = 256;
    /// Weights in one n-gram row.
    pub const NGRAM6_ELEMS: usize = 160;
    /// Bytes in one n-gram row: an fp16 scale word plus a 160x6-bit ring.
    pub const NGRAM6_BYTES: usize = 122;

    /// Bits per weight for a trellis format, `None` for every other format.
    #[must_use]
    pub const fn k_bits(dtype: DType) -> Option<u32> {
        match dtype {
            DType::Exl3K4 => Some(4),
            DType::Exl3K5 => Some(5),
            DType::Exl3K6 => Some(6),
            _ => None,
        }
    }

    /// Bytes in one tile: 16 rows of `K` bits times 16 columns.
    ///
    /// The bit width is a small number by construction - the only producer is
    /// [`k_bits`] - so an unrepresentable width yields zero rather than a
    /// wrapped size.
    #[must_use]
    pub fn tile_bytes(k_bits: u32) -> usize {
        (TILE_ELEMS / 8) * usize::from(u8::try_from(k_bits).unwrap_or(0))
    }

    /// Bytes of the two fp16 scale vectors appended after a tensor's tiles.
    ///
    /// A tensor shaped `[k, n]` - or `[k, n, experts]` - carries, per expert,
    /// an input vector of `k` fp16 words and an output vector of `n` fp16
    /// words. They live inside the tensor's own payload so a tensor stays one
    /// contiguous `(offset, bytes)` span, which is what lets the weight cache
    /// and the device-side buffer slices treat it as a single range.
    ///
    /// The arithmetic saturates rather than wrapping: a shape large enough to
    /// overflow this product is rejected by the caller's bounds check, and a
    /// saturated value is what makes that rejection happen.
    #[must_use]
    pub fn scale_bytes(shape: &TensorShape) -> u64 {
        let experts = shape.dim(2).unwrap_or(1);
        let n = shape.dim(1).unwrap_or(1);
        let k = shape.dim(0).unwrap_or(0);
        experts
            .saturating_mul(2)
            .saturating_mul(k.saturating_add(n))
    }
}

/// Total payload bytes of `shape` in format `dtype`: whole blocks rounded up,
/// plus the scale-vector tail the trellis formats append.
///
/// `None` means the product overflowed `u64`, which the reader turns into a
/// typed error rather than a truncated tensor.
#[must_use]
pub fn block_bytes_for(dtype: DType, shape: &TensorShape) -> Option<u64> {
    let elements = shape.elements().ok()?;
    let elems_per_block = u64::from(dtype.block_elems().get());
    let bytes_per_block = u64::from(dtype.block_bytes().get());
    let blocks = elements.saturating_add(elems_per_block - 1) / elems_per_block;
    let tiles = blocks.checked_mul(bytes_per_block)?;
    if exl3::k_bits(dtype).is_some() {
        tiles.checked_add(exl3::scale_bytes(shape))
    } else {
        Some(tiles)
    }
}

/// One of the eight 6-bit `(scale, min)` pairs a [`BlockQ4K`] packs into 12
/// bytes.
const fn q4k_scale_min(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Decodes one [`BlockQ4K`] to its 256 values, on the host.
///
/// A reference decode for the few places a host reads a `q4_K` table row by
/// row - a block drafter's selector codebooks, one 256-wide super-block a
/// row - rather than a load-path conversion: a matrix the device reads stays
/// packed.
#[must_use]
pub fn dequant_q4_k(block: &BlockQ4K) -> [f32; QK_K] {
    let d = half::f16::from_bits(block.d).to_f32();
    let dmin = half::f16::from_bits(block.dmin).to_f32();
    let mut out = [0.0_f32; QK_K];
    for chunk in 0..QK_K / 64 {
        let (sc1, m1) = q4k_scale_min(2 * chunk, &block.scales);
        let (sc2, m2) = q4k_scale_min(2 * chunk + 1, &block.scales);
        let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
        let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
        let qs = &block.qs[32 * chunk..32 * chunk + 32];
        for (l, &q) in qs.iter().enumerate() {
            out[64 * chunk + l] = d1 * f32::from(q & 0xF) - min1;
            out[64 * chunk + 32 + l] = d2 * f32::from(q >> 4) - min2;
        }
    }
    out
}

#[cfg(test)]
mod q4k_tests {
    use super::*;

    #[test]
    fn q4_k_decodes_scale_times_code_minus_min_per_32() {
        let mut block = BlockQ4K {
            d: half::f16::from_f32(0.5).to_bits(),
            dmin: half::f16::from_f32(0.25).to_bits(),
            scales: [0; 12],
            qs: [0; QK_K / 2],
        };
        // Sub-block 0: scale 2, min 4. Sub-block 5 (high scale bits path):
        // scale 1 | (1 << 4) = 17, min 3.
        block.scales[0] = 2;
        block.scales[4] = 4;
        block.scales[9] = 1 | (3 << 4);
        block.scales[1] = 1 << 6;
        block.qs[0] = 0x73;
        block.qs[64] = 0x50;
        let out = dequant_q4_k(&block);
        assert_eq!(
            out[0].to_bits(),
            (0.5_f32 * 2.0 * 3.0 - 0.25 * 4.0).to_bits()
        );
        // Element 160 is the high nibble of qs[64]: sub-block 5.
        assert_eq!(
            out[160].to_bits(),
            (0.5_f32 * 17.0 * 5.0 - 0.25 * 3.0).to_bits()
        );
    }
}
