//! `q4_0` matvec and prefill-GEMM tests.
//!
//! Gemma 4's GGUF stores every linear layer as `q4_0`, a format no kernel in
//! this library read before: 18 bytes holding one `f16` scale and 32 four-bit
//! weights, the low nibble of byte `j` carrying element `j` and the high nibble
//! carrying element `j + 16`.
//!
//! The fixture is packed here, dequantized here, and the device is compared
//! against a plain CPU dot product over those same numbers, so a failure names
//! the kernel or its launch geometry rather than a second implementation of the
//! unpack.
//!
//! The `#[test]` wrappers live in each backend's own driver -
//! `crates/shiso-metal/tests/dense_q4_0.rs`,
//! `crates/shiso-cuda/tests/dense_q4_0.rs`.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "kernel argument structs take i32/u32/u64 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use shiso_core::{Extent3, FnConst, FnConstValue, KernelArg, KernelRef, LaunchArgs};

/// Elements in one `q4_0` block.
const QK4_0: usize = 32;
/// Bytes in one `q4_0` block: an `f16` scale plus two weights per byte.
const Q4_0_BYTES: usize = 2 + QK4_0 / 2;

/// Input width of both fixtures. A multiple of 256, which is the width at which
/// the engine picks the widest lane split for one and two tokens.
const IN: usize = 256;
/// Output width of both fixtures. A whole number of the GEMM's 64-row tile and
/// of every matvec row blocking below.
const OUT: usize = 64;

/// Simdgroups per extended-matvec threadgroup, mirroring
/// `shiso_engine::launch::EXT_NSG`.
const EXT_NSG: u32 = 2;
/// Lanes in a simdgroup, mirroring `shiso_engine::launch::WARP`.
const WARP: u32 = 32;
/// Function-constant index of the simdgroup count (`FC_MUL_MV + 0`).
const EXT_NSG_INDEX: u16 = 600;
/// Function-constant index of the lane split (`FC_MUL_MV + 1`).
const EXT_NXPSG_INDEX: u16 = 601;

/// The specialization of a 16-lane input split, which is what a 256-wide input
/// gets at one and two tokens. `KernelRef::consts` is `&'static`, so the two
/// splits this fixture reaches are tables rather than values built per case.
const EXT_CONSTS_16: [FnConst; 2] = [
    FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(2),
    },
    FnConst {
        index: EXT_NXPSG_INDEX,
        value: FnConstValue::I16(16),
    },
];
/// The specialization of a 4-lane input split, which is what an input width
/// that is not a multiple of 128 gets at any token count. No fixture here is
/// that narrow, so it is reached only by the exhaustive case below, which walks
/// every pair the engine can select rather than the pairs one width reaches.
const EXT_CONSTS_4: [FnConst; 2] = [
    FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(2),
    },
    FnConst {
        index: EXT_NXPSG_INDEX,
        value: FnConstValue::I16(4),
    },
];
/// The specialization of an 8-lane input split, which the same input gets at
/// three tokens and above.
const EXT_CONSTS_8: [FnConst; 2] = [
    FnConst {
        index: EXT_NSG_INDEX,
        value: FnConstValue::I16(2),
    },
    FnConst {
        index: EXT_NXPSG_INDEX,
        value: FnConstValue::I16(8),
    },
];

/// The prefill GEMM's two broadcast toggles, as
/// `shiso_engine::launch::mm_consts` sets them: the input tail is off because
/// every fixture width is a whole number of the 32-wide K tile, and the output
/// tile guard is on because 33 tokens is not a whole number of tiles.
const GEMM_CONSTS: [FnConst; 2] = [
    FnConst {
        index: 700,
        value: FnConstValue::Bool(false),
    },
    FnConst {
        index: 701,
        value: FnConstValue::Bool(true),
    },
];
/// The GEMM's constant table, or nothing on a device that spells its
/// specialization into the entry name.
///
/// [`GEMM_CONSTS`] never varies with the launch - the engine's `mm_consts` is
/// one fixed table - so a device that cannot take a constant at launch compiles
/// the one variant and is handed an empty table, exactly as
/// `shiso_engine::launch::mm_consts_for` does. Unlike the matvec, the GEMM
/// therefore needs no second spelling of its name.
fn gemm_consts(gpu: &Gpu) -> &'static [FnConst] {
    if gpu.device().specializes_by_name() {
        &[]
    } else {
        &GEMM_CONSTS
    }
}

/// Threads per prefill-GEMM threadgroup, mirroring
/// `shiso_engine::launch::GEMM_BLOCK`.
const GEMM_BLOCK: u32 = 128;
/// Threadgroup bytes the prefill GEMM asks for, mirroring
/// `shiso_engine::launch::GEMM_SMEM_BYTES`: the partial-tile epilogue's view of
/// the staging allocation is the larger of the two claims.
const GEMM_SMEM_BYTES: u32 = 8_192;
/// Output rows one prefill-GEMM tile covers.
const GEMM_TILE_ROWS: usize = 64;
/// Tokens one prefill-GEMM tile covers.
const GEMM_TILE_TOKENS: usize = 32;
/// Tokens of the prefill-GEMM fixture: one whole tile plus one row, so the
/// partial-tile epilogue the output guard selects is the path under test.
const GEMM_TOK: usize = 33;
/// The prefill-GEMM fixture's scale, a sixteenth. `8 * d` is a half, so the
/// extreme element of every block round-trips through `f16` and the packer
/// recovers this exact value; every dequantized weight is then a multiple of a
/// sixteenth, which `f16` also holds exactly.
const GEMM_SCALE: f32 = 0.0625;

/// The quantized value of element `e` in prefill-GEMM weight row `out`.
///
/// `e / 16` is what keeps the two halves of a block apart. Without it the
/// pattern repeats every sixteen elements, so `row[e] == row[e + 16]` and a
/// `dequantize_q4_0` that read the high nibble where it should read the low one
/// would produce the same answer -- and the GEMM is the only caller of that
/// unpack, the matvec exercising only the `_t4` twin. With it, element `e` and
/// element `e + 16` of a block differ by one step of the cycle while each half
/// still runs the whole `0..=15` range, so the quantized zero (whose `-8 * d` is
/// the block's extreme, and therefore what makes the scale round-trip exactly)
/// is still present in both halves.
fn gemm_quant(e: usize, out: usize) -> f32 {
    ((e + out + e / 16) % 16) as f32 - 8.0
}

/// The extended-matvec argument block (`metal/generic/mul_mv_ext.metal`).
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

/// Input elements each lane takes a turn on, mirroring
/// `shiso_engine::launch::ext_nxpsg`.
const fn ext_nxpsg(in_dim: usize, n_tok: usize) -> u32 {
    if in_dim.is_multiple_of(256) && n_tok < 3 {
        16
    } else if in_dim.is_multiple_of(128) {
        8
    } else {
        4
    }
}

/// Token rows one extended-matvec threadgroup takes, mirroring
/// `shiso_engine::launch::ext_r1ptg` over the widths this fixture launches.
const fn ext_r1ptg(n_tok: usize) -> u32 {
    match n_tok {
        0 | 1 => 1,
        2 => 2,
        3 | 6 => 3,
        5 => 5,
        _ => 4,
    }
}

/// Output rows one extended-matvec threadgroup produces, mirroring
/// `shiso_engine::launch::ext_rows_per_threadgroup`.
const fn ext_rows_per_threadgroup(nxpsg: u32) -> u32 {
    WARP / nxpsg * EXT_NSG
}

/// Packs one row into `q4_0` blocks, following ggml's `quantize_row_q4_0_ref`.
///
/// The scale is the block's largest-magnitude element divided by `-8`, keeping
/// that element's sign, so the extreme of the range lands on the quantized `0`
/// and the whole 4-bit range is reachable from either side of zero. The scale is
/// stored as `f16`, which is what the kernel reads it back as.
///
/// # Panics
///
/// When `row` is not a whole number of 32-element blocks.
pub(crate) fn pack_q4_0(row: &[f32]) -> Vec<u8> {
    assert!(
        row.len().is_multiple_of(QK4_0),
        "a q4_0 row is a whole number of 32-element blocks",
    );
    let mut bytes = Vec::with_capacity(row.len() / QK4_0 * Q4_0_BYTES);
    for block in row.as_chunks::<QK4_0>().0 {
        let mut extreme = 0.0f32;
        for &x in block {
            if x.abs() > extreme.abs() {
                extreme = x;
            }
        }
        let d = extreme / -8.0;
        let d_bits = half::f16::from_f32(d).to_bits();
        // The kernel divides by the `f16` scale it reads, so the packer has to
        // quantize against that same rounded value and not against `d`.
        let rounded = crate::oracle::f16_to_f32(d_bits);
        let inv = if rounded == 0.0 { 0.0 } else { 1.0 / rounded };
        bytes.extend(d_bits.to_le_bytes());
        for j in 0..QK4_0 / 2 {
            let lo = quantize(block[j] * inv);
            let hi = quantize(block[j + QK4_0 / 2] * inv);
            bytes.push(lo | (hi << 4));
        }
    }
    bytes
}

/// One scaled weight as a 4-bit field: `round(x / d) + 8`, clamped to the range
/// the nibble can hold.
fn quantize(scaled: f32) -> u8 {
    (scaled.round() + 8.0).clamp(0.0, 15.0) as u8
}

/// The elements of a packed `q4_0` row, in element order.
///
/// The inverse of [`pack_q4_0`] and the oracle both matvec and GEMM are compared
/// against: element `j` of a block is the low nibble of byte `j` and element
/// `j + 16` the high nibble of the same byte, biased by `-8` and scaled.
///
/// # Panics
///
/// When `bytes` is not a whole number of 18-byte blocks.
pub(crate) fn dequant_q4_0(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len().is_multiple_of(Q4_0_BYTES),
        "a q4_0 row is a whole number of 18-byte blocks",
    );
    let mut out = Vec::with_capacity(bytes.len() / Q4_0_BYTES * QK4_0);
    for block in bytes.as_chunks::<Q4_0_BYTES>().0 {
        let d = crate::oracle::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];
        let mut elems = [0.0f32; QK4_0];
        for (j, &packed) in qs.iter().enumerate() {
            elems[j] = d * (f32::from(packed & 0x0F) - 8.0);
            elems[j + QK4_0 / 2] = d * (f32::from(packed >> 4) - 8.0);
        }
        out.extend(elems);
    }
    out
}

/// Left-to-right dot product, the association the matvec accumulator uses.
fn dot(weights: &[f32], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (w, v) in weights.iter().zip(x) {
        acc += w * v;
    }
    acc
}

/// The weight matrix of both fixtures, as `OUT` packed rows plus the same rows
/// dequantized. `seed` moves the signal so the two fixtures do not share a row.
fn weight_rows(seed: u32) -> (Vec<u8>, Vec<Vec<f32>>) {
    let mut packed = Vec::with_capacity(OUT * IN / QK4_0 * Q4_0_BYTES);
    let mut rows = Vec::with_capacity(OUT);
    for row in 0..OUT {
        let bytes = pack_q4_0(&crate::oracle::signal(
            IN,
            seed ^ (row as u32).wrapping_mul(0x9E37),
        ));
        rows.push(dequant_q4_0(&bytes));
        packed.extend(bytes);
    }
    (packed, rows)
}

/// The `q4_0` matvec instantiations spelled with the lane split folded in,
/// indexed by `[r1ptg - 1][nxpsg slot]`, mirroring
/// `shiso_engine::launch::EXT_Q4_0_BY_NAME`.
///
/// Fifteen names for what is one entry point plus a constant on Metal. A CUDA
/// module holds one symbol per instantiation, so the split that is a pipeline
/// constant there is part of the symbol here.
const EXT_Q4_0_BY_NAME: [[&str; 3]; 5] = [
    [
        "kernel_mul_mv_ext_q4_0_f32_r1_1_nx4",
        "kernel_mul_mv_ext_q4_0_f32_r1_1_nx8",
        "kernel_mul_mv_ext_q4_0_f32_r1_1_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_0_f32_r1_2_nx4",
        "kernel_mul_mv_ext_q4_0_f32_r1_2_nx8",
        "kernel_mul_mv_ext_q4_0_f32_r1_2_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_0_f32_r1_3_nx4",
        "kernel_mul_mv_ext_q4_0_f32_r1_3_nx8",
        "kernel_mul_mv_ext_q4_0_f32_r1_3_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_0_f32_r1_4_nx4",
        "kernel_mul_mv_ext_q4_0_f32_r1_4_nx8",
        "kernel_mul_mv_ext_q4_0_f32_r1_4_nx16",
    ],
    [
        "kernel_mul_mv_ext_q4_0_f32_r1_5_nx4",
        "kernel_mul_mv_ext_q4_0_f32_r1_5_nx8",
        "kernel_mul_mv_ext_q4_0_f32_r1_5_nx16",
    ],
];

/// The unspecialized names, one per row blocking, mirroring
/// `shiso_engine::launch::EXT_Q4_0`.
const EXT_Q4_0: [&str; 5] = [
    "kernel_mul_mv_ext_q4_0_f32_r1_1",
    "kernel_mul_mv_ext_q4_0_f32_r1_2",
    "kernel_mul_mv_ext_q4_0_f32_r1_3",
    "kernel_mul_mv_ext_q4_0_f32_r1_4",
    "kernel_mul_mv_ext_q4_0_f32_r1_5",
];

/// The entry name and constant table for one `(r1ptg, nxpsg)` pair, in whichever
/// of the two spellings this device wants.
///
/// The question is asked of the backend rather than answered by a `cfg`, for the
/// same reason `shiso_engine::launch::ext_name_for` asks it: which spelling a
/// kernel has is a property of the device, and a case that decided it at compile
/// time would be a case that only ever ran one of them.
///
/// # Panics
///
/// When `nxpsg` is not one of the three splits the family is built with.
fn matvec_entry(gpu: &Gpu, r1ptg: u32, nxpsg: u32) -> (&'static str, &'static [FnConst]) {
    let slot = match nxpsg {
        4 => 0,
        8 => 1,
        16 => 2,
        other => panic!("{other} is not a lane split the extended family is built with"),
    };
    let row = r1ptg as usize - 1;
    if gpu.device().specializes_by_name() {
        return (EXT_Q4_0_BY_NAME[row][slot], &[]);
    }
    let consts: &'static [FnConst] = match nxpsg {
        4 => &EXT_CONSTS_4,
        8 => &EXT_CONSTS_8,
        _ => &EXT_CONSTS_16,
    };
    (EXT_Q4_0[row], consts)
}

/// Runs the `q4_0` extended matvec at one token count and compares every output
/// against the dot product of the dequantized weight row and the same
/// activations the device saw.
fn matvec_case(gpu: &Gpu, n_tok: usize) {
    matvec_at(gpu, ext_r1ptg(n_tok), ext_nxpsg(IN, n_tok), n_tok);
}

/// Runs the matvec at one `(r1ptg, nxpsg)` pair over `n_tok` tokens.
fn matvec_at(gpu: &Gpu, r1ptg: u32, nxpsg: u32, n_tok: usize) {
    let (entry, consts) = matvec_entry(gpu, r1ptg, nxpsg);

    let (packed, rows) = weight_rows(0x0F0F);
    let row_bytes = (IN / QK4_0 * Q4_0_BYTES) as u64;
    let x = crate::oracle::signal(IN * n_tok, 0x5A5A);

    let weights = gpu.upload(&packed);
    let src = gpu.upload(&f32_bytes(&x));
    let dst = gpu.upload(&vec![0u8; n_tok * OUT * 4]);
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
        ne11: n_tok as i32,
        ne12: 1,
        _pad1: 0,
        nb10: 4,
        nb11: (IN * 4) as u64,
        nb12: (IN * n_tok * 4) as u64,
        nb13: (IN * n_tok * 4) as u64,
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
    run(
        gpu,
        KernelRef::specialized("generic", entry, consts),
        &LaunchArgs::new(
            Extent3::new(
                (OUT as u32).div_ceil(ext_rows_per_threadgroup(nxpsg)),
                (n_tok as u32).div_ceil(r1ptg),
                1,
            ),
            Extent3::new(WARP, EXT_NSG, 1),
            &params,
        ),
    );
    drain(gpu);
    let got = gpu.read_f32(&dst);
    for tok in 0..n_tok {
        let activations = &x[tok * IN..(tok + 1) * IN];
        for (out, row) in rows.iter().enumerate() {
            crate::assert_close!(
                got[tok * OUT + out],
                dot(row, activations),
                crate::oracle::TOL_QUANT,
                "{entry}: token {tok} output {out}",
            );
        }
    }
}

/// Every `q4_0` matvec instantiation against the oracle dot product.
///
/// One case per row blocking, at the token count the engine picks that blocking
/// for, which is also the two lane splits a 256-wide input reaches: sixteen
/// lanes at one and two tokens, eight above that.
pub fn q4_0_matvec_matches_the_oracle_dot(gpu: &Gpu) {
    for n_tok in 1..=5 {
        matvec_case(gpu, n_tok);
    }
}

/// Every `(r1ptg, nxpsg)` pair `projection::projection` can select resolves and
/// agrees with the oracle.
///
/// The case above reaches five of the fifteen: a 256-wide input only ever asks
/// for two of the three lane splits, and each row blocking only at the token
/// count that selects it. The other ten are reached by a different weight width
/// or a different token count, and on a device where the split is part of the
/// entry name a pair that is merely *absent from the module* is invisible at any
/// single width - it is a model that loads, answers at one prompt length and
/// refuses at another.
///
/// The token count is `r1ptg`, so every row the blocking carries is a real
/// token rather than one of the guarded tail rows the kernel drops.
pub fn every_q4_0_matvec_instantiation_agrees_with_the_oracle(gpu: &Gpu) {
    for r1ptg in 1..=5u32 {
        for nxpsg in [4u32, 8, 16] {
            matvec_at(gpu, r1ptg, nxpsg, r1ptg as usize);
        }
    }
}

/// The `q4_0` prefill GEMM against a triple loop over the same weights.
///
/// The fixture is exact in binary so the comparison is not a tolerance argument:
/// every dequantized weight is a multiple of a sixteenth and every activation a
/// small integer, both of which `f16` holds exactly, and the widest partial sum
/// needs fourteen bits. The scale is recovered exactly by the packer because
/// each half of each block contains the quantized zero, whose magnitude `8d` is
/// the block's largest.
///
/// Odd blocks are negated, which puts that extreme on the positive side and so
/// makes their scale negative. A `q4_0` scale is signed -- the format's `-8`
/// bias is what the sign buys -- and this is the only case that reads one
/// through `dequantize_q4_0`, where the sign rides on three separate constants
/// (`d1`, `d2` and `md`).
pub fn q4_0_prefill_gemm_matches_a_triple_loop(gpu: &Gpu) {
    let mut packed = Vec::with_capacity(OUT * IN / QK4_0 * Q4_0_BYTES);
    let mut rows = Vec::with_capacity(OUT);
    for out in 0..OUT {
        let row: Vec<f32> = (0..IN)
            .map(|e| {
                let sign = if (e / QK4_0).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                sign * GEMM_SCALE * gemm_quant(e, out)
            })
            .collect();
        let bytes = pack_q4_0(&row);
        assert_eq!(dequant_q4_0(&bytes), row, "the q4_0 fixture round-trips");
        rows.push(row);
        packed.extend(bytes);
    }
    let activations: Vec<f32> = (0..GEMM_TOK * IN).map(|i| (i % 9) as f32).collect();

    let row_bytes = (IN / QK4_0 * Q4_0_BYTES) as u64;
    let weights = gpu.upload(&packed);
    let src = gpu.upload(&f32_bytes(&activations));
    let dst = gpu.upload(&vec![0u8; GEMM_TOK * OUT * 4]);
    let args = MulMmArgs {
        ne00: IN as i32,
        ne02: 1,
        nb01: row_bytes,
        nb02: row_bytes * OUT as u64,
        nb03: row_bytes * OUT as u64,
        ne12: 1,
        _pad0: 0,
        nb10: 4,
        nb11: (IN * 4) as u64,
        nb12: (IN * GEMM_TOK * 4) as u64,
        nb13: (IN * GEMM_TOK * 4) as u64,
        ne0: OUT as i32,
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
        Extent3::new(
            GEMM_TOK.div_ceil(GEMM_TILE_TOKENS) as u32,
            OUT.div_ceil(GEMM_TILE_ROWS) as u32,
            1,
        ),
        Extent3::linear(GEMM_BLOCK),
        &params,
    );
    launch.smem_bytes = GEMM_SMEM_BYTES;
    run(
        gpu,
        KernelRef::specialized("qwen35", "kernel_mul_mm_q4_0_f32", gemm_consts(gpu)),
        &launch,
    );
    drain(gpu);
    let got = gpu.read_f32(&dst);
    for tok in 0..GEMM_TOK {
        let x = &activations[tok * IN..(tok + 1) * IN];
        for (out, row) in rows.iter().enumerate() {
            crate::assert_close!(
                got[tok * OUT + out],
                dot(row, x),
                crate::oracle::TOL_QUANT,
                "token {tok} output {out}",
            );
        }
    }
}

/// Width of both sides of the bench matvec: a 4096-wide hidden state.
const BENCH_DIM: usize = 4096;

/// One token through a `BENCH_DIM`-square `q4_0` matvec, at the split and
/// row blocking the engine picks for one token, for [`crate::bench`].
pub(crate) fn bench_matvec(gpu: &Gpu) -> crate::bench::Fixture {
    let (in_dim, out_dim) = (BENCH_DIM, BENCH_DIM);
    let (r1ptg, nxpsg) = (ext_r1ptg(1), ext_nxpsg(in_dim, 1));
    let (entry, consts) = matvec_entry(gpu, r1ptg, nxpsg);
    let mut packed = Vec::with_capacity(out_dim * in_dim / QK4_0 * Q4_0_BYTES);
    for row in 0..out_dim {
        packed.extend(pack_q4_0(&crate::oracle::signal(
            in_dim,
            0x0F0F ^ (row as u32).wrapping_mul(0x9E37),
        )));
    }
    let x = crate::oracle::signal(in_dim, 0x0B0B);
    let weights = gpu.upload(&packed);
    let src = gpu.upload(&f32_bytes(&x));
    let dst = gpu.upload(&vec![0u8; out_dim * 4]);
    let row_bytes = (in_dim / QK4_0 * Q4_0_BYTES) as u64;
    let args = MulMvExtArgs {
        ne00: in_dim as i32,
        ne01: out_dim as i32,
        ne02: 1,
        _pad0: 0,
        nb00: row_bytes,
        nb01: row_bytes,
        nb02: row_bytes * out_dim as u64,
        nb03: row_bytes * out_dim as u64,
        ne10: in_dim as i32,
        ne11: 1,
        ne12: 1,
        _pad1: 0,
        nb10: 4,
        nb11: (in_dim * 4) as u64,
        nb12: (in_dim * 4) as u64,
        nb13: (in_dim * 4) as u64,
        ne0: out_dim as i32,
        ne1: 1,
        r2: 1,
        r3: 1,
        _pad2: [0; 4],
    };
    let grid = Extent3::new(
        (out_dim as u32).div_ceil(ext_rows_per_threadgroup(nxpsg)),
        1,
        1,
    );
    let bytes = (packed.len() + in_dim * 4 + out_dim * 4) as u64;
    let flops = 2 * (in_dim * out_dim) as u64;
    crate::bench::Fixture::new(bytes, flops, move |gpu| {
        run(
            gpu,
            KernelRef::specialized("generic", entry, consts),
            &LaunchArgs::new(
                grid,
                Extent3::new(WARP, EXT_NSG, 1),
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(weights.view()),
                    KernelArg::Buffer(src.view()),
                    KernelArg::Buffer(dst.view()),
                ],
            ),
        );
    })
}
