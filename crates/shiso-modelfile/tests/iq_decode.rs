//! Host reference decoder cases for [`shiso_modelfile::iq_quant`].
//!
//! Three classes per format, plus the dispatch entry point:
//!
//! - a zero-codes block decodes to the format's center value times its scale
//!   (the center is not zero for the `i`-quants: the grids are centered on a
//!   mid weight, and the 1-bit grids on `-1 + delta`);
//! - a hand-built block with one known grid index, sign pattern and scale
//!   nibble decodes to exactly `table value x scale` at the lanes that index
//!   owns, and to the baseline everywhere else - this is what pins the
//!   bit-extraction order, the part a wrong shift gets silently wrong;
//! - a block of fixed pseudo-random codes decodes to values that all live in
//!   the format's value set {grid byte x scale x sign (+/- delta)}, a cheap
//!   tripwire for systematic bit-order errors that a single hand-built block
//!   could miss.
//!
//! Expected values are written as exact `f32` arithmetic (every product here
//! is a dyadic rational, so `==` is honest) or recomputed from the pinned
//! codebook tables, never from the decoder under test.

#![expect(
    clippy::float_cmp,
    reason = "every expected value is a dyadic rational computed in the same multiplication order as the reference decode, so exact comparison is the point of the test"
)]

use std::collections::HashSet;

use half::f16;
use shiso_core::DType;
use shiso_modelfile::iq_quant::{
    decode_block, decode_iq1_m_block, decode_iq1_s_block, decode_iq2_s_block, decode_iq2_xs_block,
    decode_iq2_xxs_block, decode_iq3_s_block, decode_iq3_xxs_block, decode_iq4_xs_block,
    decode_q2_k_block,
};
use shiso_modelfile::quant_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, IQ4NL_VALUES,
};

/// Weights one block covers.
const QK: usize = 256;

/// Writes an `f16` into a block's bytes at a byte offset.
fn put_f16(block: &mut [u8], at: usize, value: f32) {
    block[at..at + 2].copy_from_slice(&f16::from_f32(value).to_le_bytes());
}

/// The packed byte `j` of a `u64` grid entry (lane order: byte 0 first).
fn byte64(grid: u64, j: usize) -> u8 {
    ((grid >> (8 * j)) & 0xFF) as u8
}

/// The packed byte `j` of a `u32` grid entry.
fn byte32(grid: u32, j: usize) -> u8 {
    ((grid >> (8 * j)) & 0xFF) as u8
}

/// The packed byte `j` of a `u64` grid entry, signed (the 1-bit grids).
fn sbyte64(grid: u64, j: usize) -> i8 {
    byte64(grid, j).cast_signed()
}

/// A reproducible byte source: a fixed-seed LCG, no RNG dependency.
struct Lcg(u64);

impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 56) as u8
    }

    fn fill(&mut self, bytes: &mut [u8]) {
        for b in bytes {
            *b = self.next_u8();
        }
    }
}

/// A block of pseudo-random bytes with the block scale pinned to `d` (the
/// smoke cases must not draw `d` from the LCG: an `f16` exponent of 0x1F
/// would decode to NaN and poison the membership check).
fn random_block(len: usize, d_at: Option<usize>) -> Vec<u8> {
    let mut block = vec![0_u8; len];
    Lcg(0x2545_F491_4F6C_DD1D).fill(&mut block);
    if let Some(at) = d_at {
        put_f16(&mut block, at, 1.0);
    }
    block
}

/// The set of `f32` bit patterns, for membership assertions.
fn bits(values: impl IntoIterator<Item = f32>) -> HashSet<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

/// Asserts every decoded weight is one of the format's representable values.
fn assert_all_in(out: &[f32; QK], allowed: &HashSet<u32>, ctx: &str) {
    for (i, &v) in out.iter().enumerate() {
        assert!(
            allowed.contains(&v.to_bits()),
            "{ctx}: slot {i} decoded to {v}, outside the format's value set"
        );
    }
}

/// The distinct grid bytes of a `u64` grid, as weights.
fn grid_bytes_u64(table: &[u64], lanes: usize) -> Vec<u8> {
    let mut set: Vec<u8> = table
        .iter()
        .flat_map(|&g| (0..lanes).map(move |j| byte64(g, j)))
        .collect();
    set.sort_unstable();
    set.dedup();
    set
}

/// The distinct grid bytes of a `u32` grid, as weights.
fn grid_bytes_u32(table: &[u32]) -> Vec<u8> {
    let mut set: Vec<u8> = table
        .iter()
        .flat_map(|&g| (0..4).map(move |j| byte32(g, j)))
        .collect();
    set.sort_unstable();
    set.dedup();
    set
}

// ================================ q2_K =====================================

#[test]
fn test_iq_q2_k_zero_block_decodes_to_zero() {
    let mut block = [0_u8; 84];
    put_f16(&mut block, 80, 2.0);
    put_f16(&mut block, 82, 1.0);
    let mut out = [-1.0_f32; QK];
    decode_q2_k_block(&block, &mut out);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "zero codes and zero minima must land on 0"
    );
}

#[test]
fn test_iq_q2_k_reads_nibbles_and_shifted_planes() {
    let mut block = [0_u8; 84];
    put_f16(&mut block, 80, 2.0);
    put_f16(&mut block, 82, 1.0);
    // Sub-block 0: scale nibble 1, min nibble 3 -> dl = 2, ml = 3.
    block[0] = 0x31;
    // Weight l reads bits 2 * j of byte qs[l], so the four codes inside one
    // byte are 32 weights apart, not four neighbours.
    block[16] = 0b01_10_00_11; // weight 0: code 3; weight 32: code 0
    block[17] = 0x02; // weight 1: code 2
    block[18] = 0x05; // weight 2: code 1; weight 34: code 1
    // Sub-block 1 (weights 16..32) reads qs[16..32] at the same shift:
    // scale nibble 2, min nibble 4 -> dl = 4, ml = 4.
    block[1] = 0x42;
    block[32] = 0x02;
    // Sub-block 2 (weights 32..48) reads qs[0..16] at shift 2:
    // scale nibble 1, min nibble 1 -> dl = 2, ml = 1.
    block[2] = 0x11;
    // Sub-block 8 (weights 128..144) reads the second 32-byte plane.
    block[8] = 0x01;
    block[48] = 0x02;

    let mut out_arr = [0.0_f32; QK];
    decode_q2_k_block(&block, &mut out_arr);
    // dl * code - ml with dl = 2, ml = 3.
    assert_eq!(out_arr[0], 2.0 * 3.0 - 3.0);
    assert_eq!(out_arr[1], 2.0 * 2.0 - 3.0);
    assert_eq!(out_arr[2], 2.0 * 1.0 - 3.0);
    assert_eq!(out_arr[3], -3.0);
    assert_eq!(out_arr[15], -3.0);
    // Sub-block 1: dl = 4, ml = 4.
    assert_eq!(out_arr[16], 4.0 * 2.0 - 4.0);
    assert_eq!(out_arr[17], -4.0);
    // Sub-block 2: shift 2 reads qs[0] bits 2..4 at weight 32 (code 0) and
    // qs[2] bits 2..4 at weight 34 (code 1).
    assert_eq!(out_arr[32], -1.0);
    assert_eq!(out_arr[34], 2.0 * 1.0 - 1.0);
    // Sub-block 8: dl = 2, ml = 0, second plane.
    assert_eq!(out_arr[128], 4.0);
    assert_eq!(out_arr[129], 0.0);
    // Untouched sub-blocks: dl = 0, ml = 0.
    assert_eq!(out_arr[64], 0.0);
    assert_eq!(out_arr[255], 0.0);
}

#[test]
fn test_iq_q2_k_random_block_stays_in_its_value_set() {
    let mut block = random_block(84, None);
    put_f16(&mut block, 80, 1.0);
    put_f16(&mut block, 82, 0.5);
    let mut out = [0.0_f32; QK];
    decode_q2_k_block(&block.try_into().expect("84 bytes"), &mut out);
    let allowed = bits((0..16_u8).flat_map(|a| {
        (0..16_u8).flat_map(move |m| {
            (0..4_u8).map(move |c| f32::from(a) * f32::from(c) - 0.5 * f32::from(m))
        })
    }));
    assert_all_in(&out, &allowed, "q2_K");
}

// ================================ iq2_xxs ==================================

#[test]
fn test_iq_iq2_xxs_zero_block_is_center_value_times_scale() {
    let mut block = [0_u8; 66];
    put_f16(&mut block, 0, 1.0);
    let mut out = [0.0_f32; QK];
    decode_iq2_xxs_block(&block, &mut out);
    // db = d * 0.5 * 0.25, grid[0] = all 0x08: every weight is d.
    assert!(out.iter().all(|&v| v == 1.0));
}

#[test]
fn test_iq_iq2_xxs_index_and_signs_land_at_their_lanes() {
    let mut block = [0_u8; 66];
    put_f16(&mut block, 0, 1.0);
    // qs is a u16 array at byte 2. aux32[0] = qs[0] | qs[1] << 16 selects the
    // four grid indices; aux32[1] = qs[2] | qs[3] << 16 carries the scale
    // nibble (top 4 bits) and four 7-bit sign indices.
    block[2..4].copy_from_slice(&1_u16.to_le_bytes()); // grid index 1 for l = 0
    block[6..8].copy_from_slice(&1_u16.to_le_bytes()); // signs index 1 for l = 0
    block[8..10].copy_from_slice(&0x8000_u16.to_le_bytes()); // scale nibble 8
    let mut out = [0.0_f32; QK];
    decode_iq2_xxs_block(&block, &mut out);
    let db = (0.5 + 8.0) * 0.25;
    // Grid entry 1 = 0x080808080808082b: lane 0 is 0x2b, lanes 1..8 are 8.
    // ksigns[1] = 129 = 0b1000_0001 negates lanes 0 and 7.
    assert_eq!(out[0], -(db * 43.0));
    assert_eq!(out[1], db * 8.0);
    assert_eq!(out[6], db * 8.0);
    assert_eq!(out[7], -(db * 8.0));
    // l = 1 reads grid index 0 (byte 1 of aux32[0]) and signs index 0.
    assert_eq!(out[8], db * 8.0);
    // Untouched groups: db = 0.125, grid 0 -> d.
    assert_eq!(out[32], 1.0);
}

#[test]
fn test_iq_iq2_xxs_random_block_stays_in_its_value_set() {
    let block = random_block(66, Some(0));
    let mut out = [0.0_f32; QK];
    decode_iq2_xxs_block(&block.try_into().expect("66 bytes"), &mut out);
    let allowed = bits((0..16_u8).flat_map(|s| {
        let db = (0.5 + f32::from(s)) * 0.25;
        grid_bytes_u64(&IQ2XXS_GRID, 8)
            .into_iter()
            .flat_map(move |g| [db * f32::from(g), -db * f32::from(g)])
    }));
    assert_all_in(&out, &allowed, "iq2_xxs");
}

// ================================ iq2_xs ===================================

#[test]
fn test_iq_iq2_xs_zero_block_is_center_value_times_scale() {
    let mut block = [0_u8; 74];
    put_f16(&mut block, 0, 1.0);
    let mut out = [0.0_f32; QK];
    decode_iq2_xs_block(&block, &mut out);
    assert!(out.iter().all(|&v| v == 1.0));
}

#[test]
fn test_iq_iq2_xs_index_sign_and_scale_nibbles_land_at_their_lanes() {
    let mut block = [0_u8; 74];
    put_f16(&mut block, 0, 1.0);
    // Two 4-bit scales per byte, one per half of the 32-weight group.
    block[66] = 0x20;
    // qs[0]: grid index 511 (all 0x2b), signs index 1 (lanes 0 and 7).
    block[2..4].copy_from_slice(&(511_u16 | (1 << 9)).to_le_bytes());
    // qs[2]: grid index 511 in the second half, no signs.
    block[6..8].copy_from_slice(&511_u16.to_le_bytes());
    let mut out = [0.0_f32; QK];
    decode_iq2_xs_block(&block, &mut out);
    let db0 = (0.5 + 0.0) * 0.25;
    let db1 = (0.5 + 2.0) * 0.25;
    assert_eq!(out[0], -(db0 * 43.0));
    assert_eq!(out[1], db0 * 43.0);
    assert_eq!(out[7], -(db0 * 43.0));
    // qs[1] = 0: grid 0, first half.
    assert_eq!(out[8], db0 * 8.0);
    // Second half uses the high scale nibble.
    assert_eq!(out[16], db1 * 43.0);
    assert_eq!(out[24], db1 * 8.0);
    // Untouched group.
    assert_eq!(out[32], 1.0);
}

#[test]
fn test_iq_iq2_xs_random_block_stays_in_its_value_set() {
    let block = random_block(74, Some(0));
    let mut out = [0.0_f32; QK];
    decode_iq2_xs_block(&block.try_into().expect("74 bytes"), &mut out);
    let allowed = bits((0..16_u8).flat_map(|s| {
        let db = (0.5 + f32::from(s)) * 0.25;
        grid_bytes_u64(&IQ2XS_GRID, 8)
            .into_iter()
            .flat_map(move |g| [db * f32::from(g), -db * f32::from(g)])
    }));
    assert_all_in(&out, &allowed, "iq2_xs");
}

// ================================ iq2_s ====================================

#[test]
fn test_iq_iq2_s_zero_block_is_center_value_times_scale() {
    let mut block = [0_u8; 82];
    put_f16(&mut block, 0, 1.0);
    let mut out = [0.0_f32; QK];
    decode_iq2_s_block(&block, &mut out);
    assert!(out.iter().all(|&v| v == 1.0));
}

#[test]
fn test_iq_iq2_s_high_bits_and_signs_land_at_their_lanes() {
    let mut block = [0_u8; 82];
    put_f16(&mut block, 0, 1.0);
    block[74] = 0x10; // scales[0]: low nibble 0, high nibble 1
    block[66] = 0x03; // qh[0]: bits 0..2 -> l = 0 gets high bits 0b11
    block[34] = 0xFF; // signs for lane group 0 (qs[32]) negate all eight
    let mut out = [0.0_f32; QK];
    decode_iq2_s_block(&block, &mut out);
    // l = 0 index: qs[0] | (qh[0] << 8) & 0x300 = 0x300.
    let grid = IQ2S_GRID[0x300];
    for (j, v) in out.iter().enumerate().take(8) {
        assert_eq!(*v, -(0.125 * f32::from(byte64(grid, j))));
    }
    // l = 1 takes high bits from qh bits 2..4 = 0 -> grid 0, first half.
    assert_eq!(out[8], 1.0);
    // l = 2, 3 fall in the second half (scale nibble 1).
    assert_eq!(out[16], 3.0);
    assert_eq!(out[24], 3.0);
    // Untouched group.
    assert_eq!(out[32], 1.0);
}

#[test]
fn test_iq_iq2_s_random_block_stays_in_its_value_set() {
    let block = random_block(82, Some(0));
    let mut out = [0.0_f32; QK];
    decode_iq2_s_block(&block.try_into().expect("82 bytes"), &mut out);
    let allowed = bits((0..16_u8).flat_map(|s| {
        let db = (0.5 + f32::from(s)) * 0.25;
        grid_bytes_u64(&IQ2S_GRID, 8)
            .into_iter()
            .flat_map(move |g| [db * f32::from(g), -db * f32::from(g)])
    }));
    assert_all_in(&out, &allowed, "iq2_s");
}

// ================================ iq3_xxs ==================================

#[test]
fn test_iq_iq3_xxs_zero_block_is_center_value_times_scale() {
    let mut block = [0_u8; 98];
    put_f16(&mut block, 0, 1.0);
    let mut out = [0.0_f32; QK];
    decode_iq3_xxs_block(&block, &mut out);
    // db = d * 0.5 * 0.5, grid[0] = all 0x04: every weight is d.
    assert!(out.iter().all(|&v| v == 1.0));
}

#[test]
fn test_iq_iq3_xxs_index_and_signs_land_at_their_lanes() {
    let mut block = [0_u8; 98];
    put_f16(&mut block, 0, 1.0);
    // The scale-and-signs word of group 0 lives at qs[64..68] = bytes 66..70.
    block[66..70].copy_from_slice(&0x5000_0001_u32.to_le_bytes());
    block[2] = 1; // grid index 1 for the first quad
    let mut out = [0.0_f32; QK];
    decode_iq3_xxs_block(&block, &mut out);
    let db = (0.5 + 5.0) * 0.5;
    // Grid entry 1 = 0x04040414: quad bytes 0x14, 4, 4, 4.
    // ksigns[1] = 129 negates lanes 0 and 7 of the eight-weight lane.
    assert_eq!(out[0], -(db * 20.0));
    assert_eq!(out[1], db * 4.0);
    assert_eq!(out[3], db * 4.0);
    // The second grid index of l = 0 is qs[1] = 0 (all 4s), and its signs
    // come from lanes 4..8 of the same pattern: only lane 7 (bit 7) is
    // negated there.
    assert_eq!(out[4], db * 4.0);
    assert_eq!(out[5], db * 4.0);
    assert_eq!(out[7], -(db * 4.0));
    // Untouched groups: db = 0.25, grid 0 -> d.
    assert_eq!(out[32], 1.0);
}

#[test]
fn test_iq_iq3_xxs_random_block_stays_in_its_value_set() {
    let block = random_block(98, Some(0));
    let mut out = [0.0_f32; QK];
    decode_iq3_xxs_block(&block.try_into().expect("98 bytes"), &mut out);
    let allowed = bits((0..16_u8).flat_map(|s| {
        let db = (0.5 + f32::from(s)) * 0.5;
        grid_bytes_u32(&IQ3XXS_GRID)
            .into_iter()
            .flat_map(move |g| [db * f32::from(g), -db * f32::from(g)])
    }));
    assert_all_in(&out, &allowed, "iq3_xxs");
}

// ================================ iq3_s ====================================

#[test]
fn test_iq_iq3_s_zero_block_is_center_value_times_scale() {
    let mut block = [0_u8; 110];
    put_f16(&mut block, 0, 1.0);
    let mut out = [0.0_f32; QK];
    decode_iq3_s_block(&block, &mut out);
    // db = d * (1 + 0), grid[0] = all 0x01: every weight is d.
    assert!(out.iter().all(|&v| v == 1.0));
}

#[test]
fn test_iq_iq3_s_high_bit_signs_and_scales_land_at_their_lanes() {
    let mut block = [0_u8; 110];
    put_f16(&mut block, 0, 1.0);
    block[106] = 0x21; // scales[0]: db1 = 1 + 2*1, db2 = 1 + 2*2
    block[66] = 0x01; // qh[0] bit 0 -> first grid index of the group gets bit 8
    block[74] = 0b0001_0000; // signs[0]: negate lane 4 of the first eight
    let mut out = [0.0_f32; QK];
    decode_iq3_s_block(&block, &mut out);
    // First quad: index 256 (qh bit 0 shifted to bit 8), scale db1 = 3.
    let grid = IQ3S_GRID[256];
    for (j, v) in out.iter().enumerate().take(4) {
        assert_eq!(*v, 3.0 * f32::from(byte32(grid, j)));
    }
    // Second quad: qh bit 1 is clear -> index 0, and lane 4 is negated.
    assert_eq!(out[4], -3.0);
    assert_eq!(out[5], 3.0);
    // l = 1 reads qh bits 2..3, also clear -> grid 0.
    assert_eq!(out[8], 3.0);
    // Second 32-weight group of the pair uses qh[1] = 0 and db2 = 5.
    assert_eq!(out[32], 5.0);
}

#[test]
fn test_iq_iq3_s_random_block_stays_in_its_value_set() {
    let block = random_block(110, Some(0));
    let mut out = [0.0_f32; QK];
    decode_iq3_s_block(&block.try_into().expect("110 bytes"), &mut out);
    let allowed = bits((0..16_u8).flat_map(|s| {
        let db = 1.0 + 2.0 * f32::from(s);
        grid_bytes_u32(&IQ3S_GRID)
            .into_iter()
            .flat_map(move |g| [db * f32::from(g), -db * f32::from(g)])
    }));
    assert_all_in(&out, &allowed, "iq3_s");
}

// ================================ iq1_s ====================================

#[test]
fn test_iq_iq1_s_zero_block_is_center_value_times_scale() {
    let mut block = [0_u8; 50];
    put_f16(&mut block, 0, 1.0);
    let mut out = [0.0_f32; QK];
    decode_iq1_s_block(&block, &mut out);
    // dl = d, delta = +0.125, grid[0] = all -1.
    assert!(out.iter().all(|&v| v == -0.875));
}

#[test]
fn test_iq_iq1_s_scale_delta_and_high_bits_land_at_their_lanes() {
    let mut block = [0_u8; 50];
    put_f16(&mut block, 0, 1.0);
    // qh[0] (u16 at byte 34): scale multiplier 5, top bit sets delta < 0,
    // low three bits are the high index bits of the first quad.
    block[34..36].copy_from_slice(&(0x8000_u16 | (5 << 12) | 3).to_le_bytes());
    let mut out = [0.0_f32; QK];
    decode_iq1_s_block(&block, &mut out);
    // First quad: index 3 << 8 = 768, dl = 11, delta = -0.125.
    let grid = IQ1S_GRID[768];
    for (j, v) in out.iter().enumerate().take(8) {
        assert_eq!(*v, 11.0 * (f32::from(sbyte64(grid, j)) - 0.125));
    }
    // Second quad: high bits (qh >> 3) & 7 = 0 -> grid 0 (all -1).
    assert_eq!(out[8], 11.0 * -1.125);
    // Group 1: qh[1] = 0 -> dl = 1, delta = +0.125, grid 0.
    assert_eq!(out[32], -0.875);
}

#[test]
fn test_iq_iq1_s_random_block_stays_in_its_value_set() {
    let block = random_block(50, Some(0));
    let mut out = [0.0_f32; QK];
    decode_iq1_s_block(&block.try_into().expect("50 bytes"), &mut out);
    let allowed = bits((0..8_u8).flat_map(|k| {
        let dl = 2.0 * f32::from(k) + 1.0;
        [-0.125_f32, 0.125]
            .into_iter()
            .flat_map(move |delta| [-1.0_f32, 0.0, 1.0].map(move |w| dl * (w + delta)))
    }));
    assert_all_in(&out, &allowed, "iq1_s");
}

// ================================ iq1_m ====================================

#[test]
fn test_iq_iq1_m_zero_block_decodes_to_zero() {
    // iq1_m has no d field: the block scale is assembled from nibbles of the
    // scale bytes, so an all-zero block has d = 0 and every weight is 0.
    let block = [0_u8; 56];
    let mut out = [-1.0_f32; QK];
    decode_iq1_m_block(&block, &mut out);
    assert!(out.iter().all(|&v| v == 0.0));
}

#[test]
fn test_iq_iq1_m_assembled_scale_and_per_group_delta_land_at_their_lanes() {
    let mut block = [0_u8; 56];
    // d nibbles: the high nibbles of scales[1], scales[3], scales[5],
    // scales[7] become bits 0..3, 4..7, 8..11, 12..15 of the f16. d = 1.0 is
    // 0x3C00, so only the last two nibbles are set: scales[5] (byte 53) gets
    // 0xC0 and scales[7] (byte 55) gets 0x30. Those are bits 12..15 of each
    // u16 scale word, which the 3-bit group scales never read, so every
    // group scale stays 0 and every dl = d = 1.0.
    block[53] = 0xC0;
    block[55] = 0x30;
    block[32] = 0x78; // qh[0]: idx[0] high bits 0, delta[0] negative (bit 3),
    //        idx[1] high bits 7 (bits 4..6), delta[1] positive
    block[4] = 5; // qs[4]: first grid index of group 1 (qs starts at byte 0)
    let mut out = [0.0_f32; QK];
    decode_iq1_m_block(&block, &mut out);
    // Weights 0..8: grid 0 (all -1) with delta -0.125.
    assert!(out[..8].iter().all(|&v| v == -1.125));
    // Weights 8..16: grid index 0x700 with delta +0.125.
    let grid = IQ1S_GRID[0x700];
    for j in 0..8 {
        assert_eq!(out[8 + j], f32::from(sbyte64(grid, j)) + 0.125);
    }
    // Group 1, first quad: index 5, delta +0.125.
    let grid = IQ1S_GRID[5];
    for j in 0..8 {
        assert_eq!(out[32 + j], f32::from(sbyte64(grid, j)) + 0.125);
    }
    // Group 1, second quad: index 0.
    assert!(out[40..48].iter().all(|&v| v == -0.875));
    // Untouched group 2.
    assert!(out[64..72].iter().all(|&v| v == -0.875));
}

#[test]
fn test_iq_iq1_m_random_block_stays_in_its_value_set() {
    let mut block = random_block(56, None);
    // The block scale is assembled from the high nibbles of the odd scale
    // bytes (bits 12..15 of each u16 word, which the 3-bit group scales
    // never read): pin d to 1.0 = 0x3C00 (nibbles 0, 0, 0xC, 3) without
    // constraining the groups.
    block[49] &= 0x0F;
    block[51] &= 0x0F;
    block[53] = (block[53] & 0x0F) | 0xC0;
    block[55] = (block[55] & 0x0F) | 0x30;
    let mut out = [0.0_f32; QK];
    decode_iq1_m_block(&block.try_into().expect("56 bytes"), &mut out);
    let allowed = bits((0..8_u8).flat_map(|k| {
        let dl = 2.0 * f32::from(k) + 1.0;
        [-0.125_f32, 0.125]
            .into_iter()
            .flat_map(move |delta| [-1.0_f32, 0.0, 1.0].map(move |w| dl * (w + delta)))
    }));
    assert_all_in(&out, &allowed, "iq1_m");
}

// ================================ iq4_xs ===================================

#[test]
fn test_iq_iq4_xs_all_code_seven_is_scale_times_table_value() {
    let mut block = [0_u8; 136];
    put_f16(&mut block, 0, 1.0);
    // ls = 33 for every 32-weight group: low nibble 1, high two bits 2.
    for byte in &mut block[4..8] {
        *byte = 0x11;
    }
    block[2..4].copy_from_slice(&0xAAAA_u16.to_le_bytes());
    block[8..].fill(0x77);
    let mut out = [0.0_f32; QK];
    decode_iq4_xs_block(&block, &mut out);
    assert!(out.iter().all(|&v| v == f32::from(IQ4NL_VALUES[7])));
}

#[test]
fn test_iq_iq4_xs_nibbles_and_scales_land_at_their_lanes() {
    let mut block = [0_u8; 136];
    put_f16(&mut block, 0, 2.0);
    // Group 0: ls = 33 -> dl = 2. Group 1: ls = 32 -> dl = 0.
    block[4] = 0x01;
    block[2..4].copy_from_slice(&0x000A_u16.to_le_bytes());
    block[8] = 0x30; // low nibble 0 at lane 0, high nibble 3 at lane 16
    // Group 7: ls = 63 -> dl = 31 * d (the extended-range top of the scale).
    block[7] = 0xF0;
    block[2..4].copy_from_slice(&(0x000A_u16 | (3 << 14)).to_le_bytes());
    block[8 + 16 * 7] = 0x0F;
    let mut out = [0.0_f32; QK];
    decode_iq4_xs_block(&block, &mut out);
    assert_eq!(out[0], 2.0 * f32::from(IQ4NL_VALUES[0]));
    assert_eq!(out[16], 2.0 * f32::from(IQ4NL_VALUES[3]));
    // Group 1 decodes to zero: ls - 32 = 0.
    assert!(out[32..64].iter().all(|&v| v == 0.0));
    // Group 7, lane 0: code 15.
    assert_eq!(out[224], 62.0 * f32::from(IQ4NL_VALUES[15]));
}

#[test]
fn test_iq_iq4_xs_random_block_stays_in_its_value_set() {
    let block = random_block(136, Some(0));
    let mut out = [0.0_f32; QK];
    decode_iq4_xs_block(&block.try_into().expect("136 bytes"), &mut out);
    let allowed = bits((0..64_u8).flat_map(|ls| {
        let dl = f32::from(ls) - 32.0;
        IQ4NL_VALUES.into_iter().map(move |v| dl * f32::from(v))
    }));
    assert_all_in(&out, &allowed, "iq4_xs");
}

// ================================ decode_block =============================

/// A zero block of `len` bytes with `d` written at `d_at` (if any).
fn zero_block(len: usize, d_at: Option<usize>) -> Vec<u8> {
    let mut block = vec![0_u8; len];
    if let Some(at) = d_at {
        put_f16(&mut block, at, 1.0);
    }
    block
}

#[test]
fn test_iq_decode_block_dispatches_every_owned_format() {
    let cases: [(DType, usize, Option<usize>); 9] = [
        (DType::Q2_K, 84, Some(80)),
        (DType::IQ2_XXS, 66, Some(0)),
        (DType::IQ2_XS, 74, Some(0)),
        (DType::IQ2_S, 82, Some(0)),
        (DType::IQ3_XXS, 98, Some(0)),
        (DType::IQ3_S, 110, Some(0)),
        (DType::IQ1_S, 50, Some(0)),
        (DType::IQ1_M, 56, None),
        (DType::IQ4_XS, 136, Some(0)),
    ];
    for (dtype, len, d_at) in cases {
        let block = zero_block(len, d_at);
        let mut out = [7.0_f32; QK];
        decode_block(dtype, &block, &mut out).expect("owned dtype with exact lengths");
        let mut direct = [0.0_f32; QK];
        match dtype {
            DType::Q2_K => decode_q2_k_block(&block.try_into().expect("84"), &mut direct),
            DType::IQ2_XXS => decode_iq2_xxs_block(&block.try_into().expect("66"), &mut direct),
            DType::IQ2_XS => decode_iq2_xs_block(&block.try_into().expect("74"), &mut direct),
            DType::IQ2_S => decode_iq2_s_block(&block.try_into().expect("82"), &mut direct),
            DType::IQ3_XXS => decode_iq3_xxs_block(&block.try_into().expect("98"), &mut direct),
            DType::IQ3_S => decode_iq3_s_block(&block.try_into().expect("110"), &mut direct),
            DType::IQ1_S => decode_iq1_s_block(&block.try_into().expect("50"), &mut direct),
            DType::IQ1_M => decode_iq1_m_block(&block.try_into().expect("56"), &mut direct),
            DType::IQ4_XS => decode_iq4_xs_block(&block.try_into().expect("136"), &mut direct),
            other => panic!("{other:?} is not owned by decode_block"),
        }
        assert_eq!(
            out.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
            direct.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
            "{dtype:?} dispatch disagrees with its direct decoder"
        );
    }
}

#[test]
fn test_iq_decode_block_rejects_wrong_lengths() {
    let mut out = [0.0_f32; QK];
    assert!(decode_block(DType::IQ4_XS, &[0_u8; 135], &mut out).is_err());
    assert!(decode_block(DType::IQ4_XS, &[0_u8; 137], &mut out).is_err());
    assert!(decode_block(DType::IQ2_XS, &[0_u8; 74], &mut [0.0_f32; 255]).is_err());
    assert!(decode_block(DType::IQ2_XS, &[0_u8; 74], &mut [0.0_f32; 257]).is_err());
    assert!(decode_block(DType::Q2_K, &[], &mut out).is_err());
}

#[test]
fn test_iq_decode_block_rejects_formats_it_does_not_own() {
    let mut out = [0.0_f32; QK];
    // Right length, wrong format: iq4_nl shares no block size but even a
    // plausible length must not dispatch.
    assert!(decode_block(DType::IQ4_NL, &[0_u8; 136], &mut out).is_err());
    assert!(decode_block(DType::F32, &[0_u8; 84], &mut out).is_err());
    assert!(decode_block(DType::Q8_0, &[0_u8; 84], &mut out).is_err());
    assert!(decode_block(DType::Pq2_0, &[0_u8; 82], &mut out).is_err());
}
