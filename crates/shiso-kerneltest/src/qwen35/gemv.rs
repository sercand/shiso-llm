//! `q35` quantized gemv / gather / prefill-GEMM tests: `q5_K`, `q6_K`, `q4_K`.
//!
//! Task 5 (`docs/superpowers/plans/tasks/task-05-qwen35-msl-kernels.md`). A
//! `Q4_K_M` checkpoint carries a `q6_K` `token_embd` and 48 `q5_K` tensors, and the
//! generic kernel set reads none of them, so these kernels are the only thing
//! between the engine and a wrong weight read on that layout.
//!
//! Each case builds a synthetic block whose dequantization is known element by
//! element and compares the device against a naive CPU dot product over the same
//! numbers, so a failure names the kernel or its geometry rather than a second
//! implementation.
//!
//! The `#[test]` wrappers live in each backend's own driver -
//! `crates/shiso-metal/tests/q35_gemv.rs`, `crates/shiso-cuda/tests/q35_gemv.rs`.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "kernel argument structs take i32/u32/u64 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]

use crate::oracle;

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use shiso_core::{Extent3, FnConst, FnConstValue, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::quant::{BlockQ4K, BlockQ5K, BlockQ6K};

/// `f16` bits for 1.0: the `q5_K` super-block scales and the `q6_K` block scale
/// are both 1, so a decoded weight is its scale pair times the raw value.
const ONE_F16: u16 = 0x3C00;
/// One quant block per fixture row, so a row is exactly one `q5_K` block.
const IN: usize = 256;
/// Fixture rows of the matvec fixture, one quant block per output row.
const OUT: usize = 32;
/// `(32 / nxpsg) * nsg = (32 / 16) * 2` output rows per matvec threadgroup,
/// which is what turns 32 output rows into a grid of 8.
const ROWS_PER_TG: usize = 4;

/// Scale/min bytes that exercise both halves of the 6-bit pair unpack: groups
/// 0..3 take the low six bits of one byte, groups 4..7 splice two bytes.
const K_SCALES: [u8; 12] = [
    0x21, 0x45, 0x09, 0x1E, 0x33, 0x2A, 0x0C, 0x17, 0x3F, 0x04, 0x28, 0x1B,
];
/// Signed per-16-element `q6_K` scales. Both signs appear, which is what makes
/// `q6_K` symmetric and what no `q4_K` fixture can reach.
const Q6_SCALES: [i8; 16] = [
    17, -33, 5, -1, 64, -64, 22, -7, 1, -1, 31, -31, 12, -12, 40, -40,
];

/// The matvec specialization the engine dispatches: two simdgroups, 16 lanes per
/// x-split, which is the 4-rows-per-threadgroup geometry above.
const EXT_SHAPE: [FnConst; 2] = [
    FnConst {
        index: 600,
        value: FnConstValue::I16(2),
    },
    FnConst {
        index: 601,
        value: FnConstValue::I16(16),
    },
];

/// The entry name and constant table for one codec's matvec, in whichever of
/// the two spellings this device wants.
///
/// The question is asked of the backend rather than answered by a `cfg`, for
/// the same reason `shiso_engine::launch::ext_name_for` asks it: which spelling
/// a kernel has is a property of the device. `_r1_1` is one token row and
/// `_nx16` is the sixteen-lane x-split [`EXT_SHAPE`] names, so the two
/// spellings address the same instantiation.
fn matvec_entry(
    gpu: &Gpu,
    by_name: &'static str,
    unspecialized: &'static str,
) -> (&'static str, &'static [FnConst]) {
    if gpu.device().specializes_by_name() {
        (by_name, &[])
    } else {
        (unspecialized, &EXT_SHAPE)
    }
}

/// The GEMM's constant table, or nothing on a device that spells its
/// specialization into the entry name.
///
/// [`GEMM_BROADCAST`] never varies with the launch, so a device that cannot
/// take a constant at launch compiles the one variant and is handed an empty
/// table, exactly as `shiso_engine::launch::mm_consts_for` does. That variant
/// guards its output tile unconditionally, which this fixture cannot tell apart
/// from the unguarded one: 256 output rows and 32 tokens are both whole numbers
/// of tiles.
fn gemm_consts(gpu: &Gpu) -> &'static [FnConst] {
    if gpu.device().specializes_by_name() {
        &[]
    } else {
        &GEMM_BROADCAST
    }
}

/// The prefill GEMM's two broadcast toggles, both off for a plain `C = A @ B`.
const GEMM_BROADCAST: [FnConst; 2] = [
    FnConst {
        index: 700,
        value: FnConstValue::Bool(false),
    },
    FnConst {
        index: 701,
        value: FnConstValue::Bool(false),
    },
];

/// The copied `kernel_mul_mm` tiles 64 rows by 32 columns per threadgroup, which
/// is what its 6144 bytes of threadgroup memory is made of (a 64x32 `f16` A tile
/// plus a 32x32 `f16` B tile, with both broadcast toggles off).
const TILE_ROWS: usize = 64;
const TILE_COLS: usize = 32;
const GEMM_SMEM: u32 = 6144;
const GEMM_OUT: usize = 256;
const GEMM_IN: usize = 256;
const GEMM_TOK: usize = 32;

/// Rows of the `q6_K` gather table. The ids below address rows 0..3, so four
/// rows is the smallest table that still reaches every row the gather can pick.
const VOCAB: usize = 4;
/// Tokens of the gather fixture, which is also the length of its id row.
const TOK: usize = 3;
/// Elements in one `q6_K` block. The gather launches one threadgroup per block,
/// so this is both the span a threadgroup unpacks and the grid-x divisor.
const QK_K: usize = 256;

/// The extended-matvec argument block (`metal/generic/mul_mv_ext.metal`), shared
/// by the `q4_K`, `q5_K` and `q6_K` instantiations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MulMvExtArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    /// The C layout leaves four bytes here; spelled out so `Pod` can hold.
    _pad0: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    /// The C layout leaves four bytes here; spelled out so `Pod` can hold.
    _pad1: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
    /// Trailing padding to the struct's eight-byte alignment.
    _pad2: [i8; 4],
}
const _: () = assert!(size_of::<MulMvExtArgs>() == 112);

/// The prefill GEMM argument block (`metal/qwen35/gemm_q4k.metal`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MulMmArgs {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    /// The C layout leaves four bytes here; spelled out so `Pod` can hold.
    _pad0: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
    /// Trailing padding to the struct's eight-byte alignment.
    _pad2: [i8; 4],
}
const _: () = assert!(size_of::<MulMmArgs>() == 88);

/// The quantized gather argument block, which the `q6_K` gather shares with the
/// `q4_K` one.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GetRowsQuantArgs {
    n_embd: i32,
    n_vocab: i32,
    n_tokens: i32,
    /// The C layout leaves four bytes here; spelled out so `Pod` can hold.
    _pad0: i32,
    src_row_bytes: u64,
    dst_row_bytes: u64,
    token_stride: u64,
}
const _: () = assert!(size_of::<GetRowsQuantArgs>() == 40);

/// Raw `q5_K` weights for one row: the whole 0..31 range, so both nibble halves
/// of `qs` and every bit position of the fifth-bit plane get written.
fn q5_values(row: usize) -> [u8; IN] {
    let mut values = [0u8; IN];
    for (e, slot) in values.iter_mut().enumerate() {
        *slot = ((e + row * 7) % 32) as u8;
    }
    values
}

/// Raw `q6_K` weights for one row: the whole 0..63 range, so all four quarters of
/// a block and both nibble halves of `ql` are covered.
fn q6_values(row: usize) -> [u8; IN] {
    let mut values = [0u8; IN];
    for (e, slot) in values.iter_mut().enumerate() {
        *slot = ((e * 3 + row * 5) % 64) as u8;
    }
    values
}

/// Packs `values` into a `q5_K` block, the inverse of [`dequant_q5_k`]. Element
/// `e` lives in byte `(group / 2) * 32 + l0 + i` of `qs`, where `group = e / 32`
/// also says which nibble of that byte is this element and which bit of
/// `qh[l0 + i]` carries the fifth bit.
fn pack_q5_k(values: &[u8; IN]) -> BlockQ5K {
    let mut block = BlockQ5K {
        d: ONE_F16,
        dmin: ONE_F16,
        scales: K_SCALES,
        qh: [0u8; 32],
        qs: [0u8; 128],
    };
    for (e, &v) in values.iter().enumerate() {
        let group = e / 32;
        let l0 = (e / 16 % 2) * 16;
        let i = e % 16;
        let slot = &mut block.qs[(group / 2) * 32 + l0 + i];
        if group % 2 == 0 {
            *slot |= v & 0x0F;
        } else {
            *slot |= (v & 0x0F) << 4;
        }
        if v & 0x10 != 0 {
            block.qh[l0 + i] |= 1 << group;
        }
    }
    block
}

/// The `qh` bit position carrying the top two bits of each 32-element quarter,
/// in the order the kernel walks them: quarter `q` reads `qh[l] >> 2q`, while the
/// same `q` picks the `ql` byte at `32 * (q % 2)` and the nibble at `4 * (q / 2)`.
/// `dequantize_dense_q6_K` and the vendor's `dequantize_q6_K` agree on this
/// mapping, which is why the three indexes below move together.
const QH_SHIFTS: [u32; 4] = [0, 2, 4, 6];

/// Packs 256 raw 6-bit values into a `q6_K` block: the inverse of
/// [`dequant_q6_k`]. Element `e` sits at `l = e % 32` inside its quarter, so the
/// quarter picks the `ql` byte (odd quarters live in the pair's second 32 bytes),
/// the nibble (low for quarters 0 and 1, high for the next two) and the 2-bit
/// field of `qh`.
fn pack_q6_k(values: &[u8; IN]) -> BlockQ6K {
    let mut block = BlockQ6K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: Q6_SCALES,
        d: ONE_F16,
    };
    for (e, &v) in values.iter().enumerate() {
        let half = e / 128;
        let quarter = (e % 128) / 32;
        let l = e % 32;
        block.ql[half * 64 + (quarter % 2) * 32 + l] |= (v & 0x0F) << ((quarter / 2) * 4);
        block.qh[half * 32 + l] |= (v >> 4) << QH_SHIFTS[quarter];
    }
    block
}

/// Element `e` of a `q5_K` block as `dequantize_dense_q5_K` reads it. The group
/// index picks the scale pair, the nibble half and the fifth-bit position, and
/// the value is built as `dl * q - ml` with that association.
fn dequant_q5_k(block: &BlockQ5K, e: usize) -> f32 {
    let group = e / 32;
    let l0 = (e / 16 % 2) * 16;
    let i = e % 16;
    let (sc, smin) = oracle::scale_min_k4(group, 0, &block.scales);
    let dl = oracle::f16_to_f32(block.d) * f32::from(sc);
    let ml = oracle::f16_to_f32(block.dmin) * f32::from(smin);
    let nibble = u32::from(block.qs[(group / 2) * 32 + l0 + i] >> ((group % 2) * 4) & 0x0F);
    let fifth = u32::from(block.qh[l0 + i] & (1 << group) != 0) << 4;
    dl * (nibble | fifth) as f32 - ml
}

/// Element `e` of a `q6_K` block as `dequantize_dense_q6_K` reads it: low and
/// high nibbles of `ql` alternate with the 2-bit fields of `qh`, and the `-32`
/// bias is folded before the multiply.
fn dequant_q6_k(block: &BlockQ6K, e: usize) -> f32 {
    let half = e / 128;
    let quarter = (e % 128) / 32;
    let l = e % 32;
    let scale = f32::from(block.scales[half * 8 + quarter * 2 + l / 16]);
    let d = oracle::f16_to_f32(block.d) * scale;
    let ql = block.ql[half * 64 + (quarter % 2) * 32 + l] >> ((quarter / 2) * 4);
    let qh = block.qh[half * 32 + l] >> QH_SHIFTS[quarter];
    let v = u32::from(ql & 0x0F) | (u32::from(qh & 3) << 4);
    d * (v as i32 - 32) as f32
}

/// Left-to-right dot product, the association the matvec accumulator uses.
fn dot(weights: &[f32], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (w, v) in weights.iter().zip(x) {
        acc += w * v;
    }
    acc
}

/// Exact `f16` bit patterns for the small integers the prefill fixture uses.
/// Every one of them round-trips, so the activation matrix adds no rounding to
/// the oracle dot product.
fn f16_of_int(value: i32) -> u16 {
    let (sign, magnitude) = if value < 0 {
        (0x8000u16, -value)
    } else {
        (0u16, value)
    };
    let bits: u16 = match magnitude {
        0 => 0x0000,
        1 => 0x3C00,
        2 => 0x4000,
        3 => 0x4180,
        4 => 0x4400,
        5 => 0x4500,
        6 => 0x4600,
        7 => 0x4700,
        8 => 0x4800,
        other => panic!("{other} is not one of the fixture's exactly representable integers"),
    };
    let _ = bits;
    sign | bits
}

/// Every 6-bit scale field of the prefill fixture decodes to 8, so a decoded
/// weight is `8 * q` in the whole groups and `q / 2` in the half-scale ones:
/// multiples of a half, which `f16` holds exactly.
const GEMM_SCALES: [u8; 12] = [0x08; 12];

/// Nibbles cycling the whole 4-bit range, so every weight a `q4_K` block can hold
/// at this scale pair shows up somewhere in the fixture.
fn q4_values(row: usize) -> [u8; IN] {
    let mut values = [0u8; IN];
    for (e, slot) in values.iter_mut().enumerate() {
        *slot = ((e + row) % 16) as u8;
    }
    values
}

/// The `q4_K` blocks written out in field order.
///
/// Written out byte by byte rather than reinterpreted, so the 144-byte layout the
/// kernel sees is visible in the fixture: two `f16` scales, twelve packed
/// scale/min bytes and 128 weight bytes, which is `BlockQ4K`'s size.
pub(crate) fn q4_k_bytes(blocks: &[oracle::Q4K]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(blocks.len() * size_of::<BlockQ4K>());
    for block in blocks {
        bytes.extend(block.d.to_le_bytes());
        bytes.extend(block.dmin.to_le_bytes());
        bytes.extend(block.scales);
        bytes.extend(block.qs);
    }
    bytes
}

/// Runs one matvec fixture through `entry` and compares every output row with the
/// dot product of the dequantized weight row and the same `x` the device saw.
///
/// One quant block per row and one token, so the grid follows entirely from the
/// row-blocking constants: `(32 / nxpsg) * nsg = 4` output rows per threadgroup,
/// which turns 32 output rows into a grid of 8.
///
/// `module` is a parameter rather than a constant because
/// `shiso_engine::launch::ext_module` is: it maps `q5_K` and `q6_K` to
/// `launch::Q35` and everything else, `q4_K` included, to `launch::GENERIC`. A
/// case that named one module for all three would grade an entry the engine
/// never asks for at that name.
fn matvec_case(
    gpu: &Gpu,
    module: &'static str,
    entry: (&'static str, &'static [FnConst]),
    weight_bytes: &[u8],
    dequant: impl Fn(usize, usize) -> f32,
) {
    let x = oracle::signal(IN, 0x5A5A);
    let got = matvec_run(gpu, module, entry, weight_bytes, &x);
    for row in 0..OUT {
        let dequantized: Vec<f32> = (0..IN).map(|e| dequant(row, e)).collect();
        crate::assert_close!(
            got[row],
            dot(&dequantized, &x),
            oracle::TOL_QUANT,
            "output row {row}",
        );
    }
}

/// Launches one matvec fixture over `x` and hands back the `OUT` output rows.
///
/// Split out of [`matvec_case`] because
/// `the_dequantizers_produce_the_same_bits_they_always_have` drives the same
/// entry with a different activation and grades the result a different way.
fn matvec_run(
    gpu: &Gpu,
    module: &'static str,
    entry: (&'static str, &'static [FnConst]),
    weight_bytes: &[u8],
    x: &[f32],
) -> Vec<f32> {
    let row_bytes = (weight_bytes.len() / OUT) as u64;
    let weights = gpu.upload(weight_bytes);
    let src = gpu.upload(&f32_bytes(x));
    let dst = gpu.upload(&[0u8; OUT * 4]);
    let args = MulMvExtArgs {
        ne00: IN as i32,
        ne01: OUT as i32,
        ne02: 1,
        _pad0: 0,
        nb00: row_bytes,
        nb01: row_bytes,
        nb02: row_bytes * OUT as u64,
        nb03: row_bytes * OUT as u64,
        ne10: IN as i32,
        ne11: 1,
        ne12: 1,
        _pad1: 0,
        nb10: 4,
        nb11: (IN * 4) as u64,
        nb12: (IN * 4) as u64,
        nb13: (IN * 4) as u64,
        ne0: OUT as i32,
        ne1: 1,
        r2: 1,
        r3: 1,
        _pad2: [0; 4],
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(weights.view()),
        KernelArg::Buffer(src.view()),
        KernelArg::Buffer(dst.view()),
    ];
    run(
        gpu,
        KernelRef::specialized(module, entry.0, entry.1),
        &LaunchArgs::new(
            Extent3::new((OUT / ROWS_PER_TG) as u32, 1, 1),
            Extent3::new(32, 2, 1),
            &params,
        ),
    );
    drain(gpu);
    gpu.read_f32(&dst)
}

/// `q5_K` matvec over blocks whose nibbles cycle the whole 0..31 range, so both
/// nibble halves of `qs` and every bit of the fifth-bit plane are exercised,
/// against the dot product of the same row dequantized on the host.
pub fn q5_k_matvec_matches_the_oracle_dot(gpu: &Gpu) {
    let blocks: Vec<BlockQ5K> = (0..OUT).map(|row| pack_q5_k(&q5_values(row))).collect();
    let weight_bytes: Vec<u8> = blocks
        .iter()
        .flat_map(bytemuck::bytes_of)
        .copied()
        .collect();
    matvec_case(
        gpu,
        "qwen35",
        matvec_entry(
            gpu,
            "kernel_mul_mv_ext_q5_K_f32_r1_1_nx16",
            "kernel_mul_mv_ext_q5_K_f32_r1_1",
        ),
        &weight_bytes,
        |row, e| dequant_q5_k(&blocks[row], e),
    );
}

/// The same shape for `q6_K`: the whole 0..63 range across all four quarters and
/// both signs of the per-16 scales, so the symmetric `-32` bias is exercised.
pub fn q6_k_matvec_matches_the_oracle_dot(gpu: &Gpu) {
    let blocks: Vec<BlockQ6K> = (0..OUT).map(|row| pack_q6_k(&q6_values(row))).collect();
    let weight_bytes: Vec<u8> = blocks
        .iter()
        .flat_map(bytemuck::bytes_of)
        .copied()
        .collect();
    matvec_case(
        gpu,
        "qwen35",
        matvec_entry(
            gpu,
            "kernel_mul_mv_ext_q6_K_f32_r1_1_nx16",
            "kernel_mul_mv_ext_q6_K_f32_r1_1",
        ),
        &weight_bytes,
        |row, e| dequant_q6_k(&blocks[row], e),
    );
}

/// Every dequantized weight, compared BY BITS and not by a budget.
///
/// # What this exists to catch
///
/// The three cases above grade a dot product at [`oracle::TOL_QUANT`], which is
/// `1e-3`, and the committed golden grades a decode in nats. Neither can see a
/// last-bit change to a dequantizer, and on 2026-09-21 all three `_t4` unpacks
/// were rewritten - they had been building a whole sixteen-element group and
/// discarding twelve of it. That rewrite was bit-exact *by construction*, and
/// nothing in this tree could have told anyone if it had not been. This case
/// is what makes the next such rewrite gradeable.
///
/// # How a dot product is made to yield one weight
///
/// The activation is one-hot. With `x[e] = 1` and every other element zero,
/// each of the matvec's products is either `0 * w`, which is zero, or `1 * w`,
/// which is `w` exactly; summing one value and 255 zeros - and then folding
/// zeros across the lane group - carries no rounding at all. So output row `r`
/// is the dequantized weight `w[r][e]`, bit for bit, and the loop below reads
/// all `IN * OUT` of them out of the kernel one column at a time.
///
/// # The reference is written in the kernel's own association
///
/// `q4_K` and `q5_K` build a weight as `dl * q - ml`, which nvcc contracts to
/// a single `fma`; the host references above spell it as a separate multiply
/// and subtract, which rounds twice and differs in the last bit. So this case
/// uses [`f32::mul_add`] for those two, and a plain multiply for `q6_K`, whose
/// `d * (v - 32)` has nothing to contract. The one concession is that a zero is
/// compared by value and not by sign, because what comes back is an
/// accumulator and `0.0 + -0.0` is `+0.0`; the same concession
/// `every_e4m3_code_decodes_the_way_the_host_decodes_it` makes, for the same
/// reason. **Any other failure here means the last bit of a dequantized weight
/// moved.** That is usually an edit to
/// `kernels/cuda/generic/generic.cuh`; it can also be a compiler that stopped
/// contracting, which is a real numerics change and is worth failing over
/// either way. Re-capture it the way the plan's numerics rule says: in the
/// commit that moves it, with the measured delta.
///
/// # Panics
///
/// When the launch is refused, or any weight's bits differ from the reference.
pub fn the_dequantizers_produce_the_same_bits_they_always_have(gpu: &Gpu) {
    let q4: Vec<oracle::Q4K> = (0..OUT)
        .map(|row| oracle::Q4K {
            d: ONE_F16,
            dmin: 0,
            scales: GEMM_SCALES,
            qs: oracle::pack_q4_k_dense(&q4_values(row)),
        })
        .collect();
    let q5: Vec<BlockQ5K> = (0..OUT).map(|row| pack_q5_k(&q5_values(row))).collect();
    let q6: Vec<BlockQ6K> = (0..OUT).map(|row| pack_q6_k(&q6_values(row))).collect();
    let q5_bytes: Vec<u8> = q5.iter().flat_map(bytemuck::bytes_of).copied().collect();
    let q6_bytes: Vec<u8> = q6.iter().flat_map(bytemuck::bytes_of).copied().collect();

    bits_case(
        gpu,
        "q4_K",
        "generic",
        matvec_entry(
            gpu,
            "kernel_mul_mv_ext_q4_K_f32_r1_1_nx16",
            "kernel_mul_mv_ext_q4_K_f32_r1_1",
        ),
        &q4_k_bytes(&q4),
        |row, e| exact_q4_k(&q4[row], e),
    );
    bits_case(
        gpu,
        "q5_K",
        "qwen35",
        matvec_entry(
            gpu,
            "kernel_mul_mv_ext_q5_K_f32_r1_1_nx16",
            "kernel_mul_mv_ext_q5_K_f32_r1_1",
        ),
        &q5_bytes,
        |row, e| exact_q5_k(&q5[row], e),
    );
    bits_case(
        gpu,
        "q6_K",
        "qwen35",
        matvec_entry(
            gpu,
            "kernel_mul_mv_ext_q6_K_f32_r1_1_nx16",
            "kernel_mul_mv_ext_q6_K_f32_r1_1",
        ),
        &q6_bytes,
        |row, e| dequant_q6_k(&q6[row], e),
    );
}

/// Reads every weight of one codec's fixture back through its matvec and
/// compares the bits.
fn bits_case(
    gpu: &Gpu,
    codec: &str,
    module: &'static str,
    entry: (&'static str, &'static [FnConst]),
    weight_bytes: &[u8],
    want: impl Fn(usize, usize) -> f32,
) {
    let mut checked = 0usize;
    for e in 0..IN {
        let mut x = vec![0.0f32; IN];
        x[e] = 1.0;
        let got = matvec_run(gpu, module, entry, weight_bytes, &x);
        for (row, value) in got.iter().enumerate().take(OUT) {
            let reference = want(row, e);
            // Zero is compared by value and not by sign, the same concession
            // `every_e4m3_code_decodes_the_way_the_host_decodes_it` makes and
            // for the same reason: what comes back is an accumulator, and
            // `0.0 + -0.0` is `+0.0`, so a sum cannot carry the sign of a zero
            // however faithfully the unpack produced it.
            let same =
                value.to_bits() == reference.to_bits() || (*value == 0.0 && reference == 0.0);
            assert!(
                same,
                "{codec} weight [{row}][{e}]: kernel {value:e} ({:#010x}) vs reference {reference:e} ({:#010x})",
                value.to_bits(),
                reference.to_bits(),
            );
            checked += 1;
        }
    }
    assert_eq!(checked, IN * OUT, "{codec}: every weight was compared");
}

/// `q4_K` element `e`, in the association nvcc gives the kernel.
///
/// [`oracle::dequant_q4_k_dense`] spelled as one `fma`, which is what the
/// device computes; see
/// [`the_dequantizers_produce_the_same_bits_they_always_have`].
pub(crate) fn exact_q4_k(block: &oracle::Q4K, e: usize) -> f32 {
    let chunk = e / 4;
    let il = chunk / 4;
    let i = (chunk % 4) * 4 + e % 4;
    let whole = il % 4 < 2;
    let mask: u16 = if whole { 0x0F } else { 0xF0 };
    let nibble = u16::from(block.qs[(il / 4) * 32 + 16 * (il % 2) + i]) & mask;
    let (sc, smin) = oracle::scale_min_k4((il / 4) * 2, (il % 4) / 2, &block.scales);
    let d = oracle::f16_to_f32(block.d);
    let dl = if whole { d } else { d * (1.0 / 16.0) } * f32::from(sc);
    let ml = oracle::f16_to_f32(block.dmin) * f32::from(smin);
    dl.mul_add(f32::from(nibble), -ml)
}

/// `q5_K` element `e`, in the association nvcc gives the kernel.
fn exact_q5_k(block: &BlockQ5K, e: usize) -> f32 {
    let group = e / 32;
    let l0 = (e / 16 % 2) * 16;
    let i = e % 16;
    let (sc, smin) = oracle::scale_min_k4(group, 0, &block.scales);
    let dl = oracle::f16_to_f32(block.d) * f32::from(sc);
    let ml = oracle::f16_to_f32(block.dmin) * f32::from(smin);
    let nibble = u32::from(block.qs[(group / 2) * 32 + l0 + i] >> ((group % 2) * 4) & 0x0F);
    let fifth = u32::from(block.qh[l0 + i] & (1 << group) != 0) << 4;
    dl.mul_add((nibble | fifth) as f32, -ml)
}

/// A `q6_K` row gathered by id must equal the same block unpacked by the matvec.
///
/// The two kernels were written against the same block layout independently, so
/// this is where a mistake in either unpack shows up as a disagreement instead of
/// cancelling out: the gather walks 256-element blocks with a per-thread stride,
pub fn q6_k_gather_matches_the_dense_unpack(gpu: &Gpu) {
    let blocks: Vec<BlockQ6K> = (0..VOCAB).map(|row| pack_q6_k(&q6_values(row))).collect();
    let table_bytes: Vec<u8> = blocks
        .iter()
        .flat_map(bytemuck::bytes_of)
        .copied()
        .collect();
    // Out of order and repeating, so a gather that ignored the id would fail.
    let ids: [i32; TOK] = [3, 0, 2];
    let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();

    let table = gpu.upload(&table_bytes);
    let tokens = gpu.upload(&id_bytes);
    let dst = gpu.upload(&vec![0u8; TOK * IN * 4]);
    let args = GetRowsQuantArgs {
        n_embd: IN as i32,
        n_vocab: VOCAB as i32,
        n_tokens: TOK as i32,
        _pad0: 0,
        src_row_bytes: size_of::<BlockQ6K>() as u64,
        dst_row_bytes: (IN * 4) as u64,
        token_stride: 4,
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(table.view()),
        KernelArg::Buffer(tokens.view()),
        KernelArg::Buffer(dst.view()),
    ];
    run(
        gpu,
        KernelRef::new("qwen35", "kernel_get_rows_q6_K_f32"),
        // One threadgroup per 256-element block of a row, one row per token.
        &LaunchArgs::new(
            Extent3::new((IN / QK_K) as u32, TOK as u32, 1),
            Extent3::linear(256),
            &params,
        ),
    );
    drain(gpu);
    let got = gpu.read_f32(&dst);
    for tok in 0..TOK {
        let id = usize::try_from(ids[tok]).expect("the fixture ids are not negative");
        for e in 0..IN {
            crate::assert_close!(
                got[tok * IN + e],
                dequant_q6_k(&blocks[id], e),
                oracle::TOL_QUANT,
                "token {tok} element {e}",
            );
        }
    }
}

/// The prefill GEMM against a triple loop over the same dequantized weights.
///
/// The fixture is chosen so the whole computation is exact in binary: the scale
/// pair makes every weight a multiple of a half, the activations are small
/// integers, and no partial sum approaches `f32`'s 24-bit mantissa. The kernel
/// stages both operands as `f16` and accumulates in `f32`, so agreement is
/// bit-for-bit and any difference is the tiling or the tile boundaries.
pub fn q4_k_prefill_gemm_matches_a_triple_loop(gpu: &Gpu) {
    let blocks: Vec<oracle::Q4K> = (0..GEMM_OUT)
        .map(|out| oracle::Q4K {
            d: ONE_F16,
            dmin: 0,
            scales: GEMM_SCALES,
            qs: oracle::pack_q4_k_dense(&q4_values(out)),
        })
        .collect();
    // Small integers, exactly what the `f16` tile the GEMM stages can hold.
    let activations: Vec<f32> = (0..GEMM_TOK * GEMM_IN)
        .map(|i| oracle::f16_to_f32(f16_of_int((i % 9) as i32)))
        .collect();

    let weights = gpu.upload(&q4_k_bytes(&blocks));
    let src = gpu.upload(&f32_bytes(&activations));
    let dst = gpu.upload(&vec![0u8; GEMM_TOK * GEMM_OUT * 4]);
    let args = MulMmArgs {
        ne00: GEMM_IN as i32,
        ne02: 1,
        nb01: size_of::<BlockQ4K>() as u64,
        nb02: (size_of::<BlockQ4K>() * GEMM_OUT) as u64,
        nb03: (size_of::<BlockQ4K>() * GEMM_OUT) as u64,
        ne12: 1,
        _pad0: 0,
        nb10: 4,
        nb11: (GEMM_IN * 4) as u64,
        nb12: (GEMM_IN * GEMM_TOK * 4) as u64,
        nb13: (GEMM_IN * GEMM_TOK * 4) as u64,
        ne0: GEMM_OUT as i32,
        ne1: GEMM_TOK as i32,
        r2: 1,
        r3: 1,
        _pad2: [0; 4],
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(weights.view()),
        KernelArg::Buffer(src.view()),
        KernelArg::Buffer(dst.view()),
    ];
    let mut launch = LaunchArgs::new(
        // The tile is 64 output rows by 32 tokens; grid.x walks tokens, grid.y
        // walks the 64-row output tiles.
        Extent3::new(
            (GEMM_TOK / TILE_COLS) as u32,
            (GEMM_OUT / TILE_ROWS) as u32,
            1,
        ),
        Extent3::linear(128),
        &params,
    );
    launch.smem_bytes = GEMM_SMEM;
    run(
        gpu,
        KernelRef::specialized("qwen35", "kernel_mul_mm_q4_K_f32", gemm_consts(gpu)),
        &launch,
    );
    drain(gpu);
    let got = gpu.read_f32(&dst);
    for tok in 0..GEMM_TOK {
        let x = &activations[tok * GEMM_IN..(tok + 1) * GEMM_IN];
        for out in 0..GEMM_OUT {
            let dequantized: Vec<f32> = (0..GEMM_IN)
                .map(|e| oracle::dequant_q4_k_dense(&blocks[out], e))
                .collect();
            crate::assert_close!(
                got[tok * GEMM_OUT + out],
                dot(&dequantized, x),
                oracle::TOL_QUANT,
                "token {tok} output {out}",
            );
        }
    }
}

/// `q4_K` matvec over blocks whose nibbles cycle the whole 0..15 range, against
/// the dot product of the same row dequantized on the host.
///
/// The codec the bulk of this container is packed at, and the one path any of
/// those 131 tensors takes at one token: the GEMM above serves a prefill chunk
/// of two or more, and a decode step is one. The fixture is the GEMM's own
/// scale pair and value pattern, so the two cases read the same bytes two ways
/// and a disagreement is the matvec's unpack rather than the fixture's.
pub fn q4_k_matvec_matches_the_oracle_dot(gpu: &Gpu) {
    let blocks: Vec<oracle::Q4K> = (0..OUT)
        .map(|row| oracle::Q4K {
            d: ONE_F16,
            dmin: 0,
            scales: GEMM_SCALES,
            qs: oracle::pack_q4_k_dense(&q4_values(row)),
        })
        .collect();
    matvec_case(
        gpu,
        "generic",
        matvec_entry(
            gpu,
            "kernel_mul_mv_ext_q4_K_f32_r1_1_nx16",
            "kernel_mul_mv_ext_q4_K_f32_r1_1",
        ),
        &q4_k_bytes(&blocks),
        |row, e| oracle::dequant_q4_k_dense(&blocks[row], e),
    );
}

/// Blocks per row of the batched fixture. Two, so a lane walks into a second
/// block and, for `q6_K`, reads a block that is only two-byte aligned: 210 is
/// not a multiple of four, and every block after a row's first sits off a
/// four-byte boundary.
const BATCH_BLOCKS: usize = 2;
/// Input width of the batched fixture.
const BATCH_IN: usize = IN * BATCH_BLOCKS;

/// One codec's host dequantizer, by block and element.
type Dequant<'a> = Box<dyn Fn(usize, usize) -> f32 + 'a>;

/// The three lane splits the engine dispatches, with the two-simdgroup block.
const EXT_SPLITS: [(u32, [FnConst; 2]); 3] =
    [(4, ext_split(4)), (8, ext_split(8)), (16, ext_split(16))];

/// The specialization of one lane split.
const fn ext_split(nxpsg: i16) -> [FnConst; 2] {
    [
        FnConst {
            index: 600,
            value: FnConstValue::I16(2),
        },
        FnConst {
            index: 601,
            value: FnConstValue::I16(nxpsg),
        },
    ]
}

/// Every K-quant matvec instantiation the engine reaches, against the oracle.
///
/// The cases above run one token at the sixteen-lane split, and a decode batch
/// of two to five rows or a narrower input reaches the other fourteen
/// instantiations per codec, which share the dispatch but not the row loop.
/// Each row here is two blocks, the `q4_K` minimum is non-zero so the folded
/// `- ml * sum(y)` term is graded, and the four-row blocking is also run over
/// seven tokens so the tail threadgroup's guard is exercised.
///
/// # Panics
///
/// When a launch is refused or any output differs from the dot product of
/// the host-dequantized row by more than [`oracle::TOL_QUANT`].
pub fn k_quant_matvec_batches_match_the_oracle_dot(gpu: &Gpu) {
    let q4: Vec<oracle::Q4K> = (0..OUT * BATCH_BLOCKS)
        .map(|b| oracle::Q4K {
            d: ONE_F16,
            // 0.5: a minimum the kernel has to subtract.
            dmin: 0x3800,
            scales: K_SCALES,
            qs: oracle::pack_q4_k_dense(&q4_values(b)),
        })
        .collect();
    let q5: Vec<BlockQ5K> = (0..OUT * BATCH_BLOCKS)
        .map(|b| pack_q5_k(&q5_values(b)))
        .collect();
    let q6: Vec<BlockQ6K> = (0..OUT * BATCH_BLOCKS)
        .map(|b| pack_q6_k(&q6_values(b)))
        .collect();
    let q5_bytes: Vec<u8> = q5.iter().flat_map(bytemuck::bytes_of).copied().collect();
    let q6_bytes: Vec<u8> = q6.iter().flat_map(bytemuck::bytes_of).copied().collect();

    let codecs: [(&str, &'static str, Vec<u8>, Dequant<'_>); 3] = [
        (
            "q4_K",
            "generic",
            q4_k_bytes(&q4),
            Box::new(|b, e| oracle::dequant_q4_k_dense(&q4[b], e)),
        ),
        (
            "q5_K",
            "qwen35",
            q5_bytes,
            Box::new(|b, e| dequant_q5_k(&q5[b], e)),
        ),
        (
            "q6_K",
            "qwen35",
            q6_bytes,
            Box::new(|b, e| dequant_q6_k(&q6[b], e)),
        ),
    ];
    for (codec, module, bytes, dequant) in &codecs {
        let rows: Vec<Vec<f32>> = (0..OUT)
            .map(|row| {
                (0..BATCH_IN)
                    .map(|e| dequant(row * BATCH_BLOCKS + e / IN, e % IN))
                    .collect()
            })
            .collect();
        for r1 in 1..=5usize {
            let tokens = if r1 == 4 { [4usize, 7] } else { [r1, r1] };
            for (nxpsg, consts) in &EXT_SPLITS {
                let entry = batched_entry(gpu, codec, r1, *nxpsg, consts);
                for &n_tok in tokens.iter().take(if r1 == 4 { 2 } else { 1 }) {
                    let x = oracle::signal(BATCH_IN * n_tok, 0x1234 + n_tok as u32);
                    let got = batched_run(
                        gpu,
                        &BatchedDesc {
                            module,
                            entry,
                            weight_bytes: bytes,
                            x: &x,
                            n_tok,
                            r1,
                            nxpsg: *nxpsg as usize,
                        },
                    );
                    for tok in 0..n_tok {
                        let xt = &x[tok * BATCH_IN..(tok + 1) * BATCH_IN];
                        for (row, weights) in rows.iter().enumerate() {
                            crate::assert_close!(
                                got[tok * OUT + row],
                                dot(weights, xt),
                                oracle::TOL_QUANT,
                                "{codec} r1={r1} nx={nxpsg} tokens={n_tok} token {tok} row {row}",
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The entry for one codec, row blocking and lane split, in this device's
/// spelling.
///
/// Leaked because a kernel reference names its entry for the life of the
/// process and these names are built rather than written out: sixty literals
/// would be a second table of the engine's that could drift from it.
fn batched_entry(
    gpu: &Gpu,
    codec: &str,
    r1: usize,
    nxpsg: u32,
    consts: &'static [FnConst; 2],
) -> (&'static str, &'static [FnConst]) {
    if gpu.device().specializes_by_name() {
        let name = format!("kernel_mul_mv_ext_{codec}_f32_r1_{r1}_nx{nxpsg}");
        (Box::leak(name.into_boxed_str()), &[])
    } else {
        let name = format!("kernel_mul_mv_ext_{codec}_f32_r1_{r1}");
        (Box::leak(name.into_boxed_str()), consts)
    }
}

/// One launch of the batched fixture: the entry, the weights, the activation
/// and the geometry the grid is built from.
struct BatchedDesc<'a> {
    module: &'static str,
    entry: (&'static str, &'static [FnConst]),
    weight_bytes: &'a [u8],
    x: &'a [f32],
    /// Token rows in `x`.
    n_tok: usize,
    /// Token rows one threadgroup takes, which the entry was built for.
    r1: usize,
    /// Lanes one output row's input is split over.
    nxpsg: usize,
}

/// Launches the batched fixture and hands back `n_tok * OUT` outputs, token
/// major.
fn batched_run(gpu: &Gpu, desc: &BatchedDesc<'_>) -> Vec<f32> {
    let BatchedDesc {
        module,
        entry,
        weight_bytes,
        x,
        n_tok,
        r1,
        nxpsg,
    } = *desc;
    let row_bytes = (weight_bytes.len() / OUT) as u64;
    let weights = gpu.upload(weight_bytes);
    let src = gpu.upload(&f32_bytes(x));
    let dst = gpu.upload(&vec![0u8; n_tok * OUT * 4]);
    let args = MulMvExtArgs {
        ne00: BATCH_IN as i32,
        ne01: OUT as i32,
        ne02: 1,
        _pad0: 0,
        nb00: row_bytes,
        nb01: row_bytes,
        nb02: row_bytes * OUT as u64,
        nb03: row_bytes * OUT as u64,
        ne10: BATCH_IN as i32,
        ne11: n_tok as i32,
        ne12: 1,
        _pad1: 0,
        nb10: 4,
        nb11: (BATCH_IN * 4) as u64,
        nb12: (BATCH_IN * 4 * n_tok) as u64,
        nb13: (BATCH_IN * 4 * n_tok) as u64,
        ne0: OUT as i32,
        ne1: n_tok as i32,
        r2: 1,
        r3: 1,
        _pad2: [0; 4],
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(weights.view()),
        KernelArg::Buffer(src.view()),
        KernelArg::Buffer(dst.view()),
    ];
    let rows_per_tg = (32 / nxpsg) * 2;
    run(
        gpu,
        KernelRef::specialized(module, entry.0, entry.1),
        &LaunchArgs::new(
            Extent3::new(
                OUT.div_ceil(rows_per_tg) as u32,
                n_tok.div_ceil(r1) as u32,
                1,
            ),
            Extent3::new(32, 2, 1),
            &params,
        ),
    );
    drain(gpu);
    gpu.read_f32(&dst)
}
