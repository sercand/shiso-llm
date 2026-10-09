//! Oracle cases for the mixed i-quant matvecs, the `iq1_m` gather, and the
//! codebook tables the nine formats decode through.
//!
//! The Saluki export packs its linear layers in `q2_K`, `iq2_xxs`, `iq2_xs`,
//! `iq2_s`, `iq3_xxs`, `iq3_s`, `iq1_s`, `iq1_m` and `iq4_xs`, and its
//! vocabulary table in `iq1_m`. None of these formats is a small integer times
//! a scale: a weight is a codebook byte, usually times a sign drawn from a
//! second table, so there is no packer here and no quantizer anywhere. The
//! fixture is *random codes* — bytes that are whatever the format says they
//! are — and the oracle is `shiso_modelfile::iq_quant::decode_block`, the host
//! reference that was verified bit-for-bit against the reference decoder. The
//! format is the contract; grading a kernel against a second encoder would
//! only prove the two encoders agree.
//!
//! The matvec cases launch the `kernel_mul_mv_ext_<fmt>_f32_r1_<1..=5>`
//! family on the same geometry `shiso_engine::launch` picks, and the gather
//! launches `kernel_get_rows_iq1_m_f32` with the same argument block and grid
//! as the engine's single gather launch site. The table case reads the nine
//! `constant` arrays of `iq_tables.metal` back off the device through
//! `kernel_iq_table_copy` and byte-compares them against the host tables:
//! those values are format data, and one changed byte silently redefines
//! every checkpoint on disk.
//!
//! The `#[test]` wrappers live in each backend's own driver -
//! `crates/shiso-metal/tests/dense_iq.rs`.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "kernel argument structs take i32/u32/u64 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DType, Extent3, FnConst, FnConstValue, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::iq_quant::decode_block;
use shiso_modelfile::quant_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, IQ4NL_VALUES,
    KSIGNS_IQ2XS, KSIGNS64,
};

use crate::harness::{Gpu, drain, f32_bytes, run};

/// The table comparison reads native-endian bytes off a device that is
/// little-endian; a big-endian host would have to spell the tables out.
const _: () = assert!(cfg!(target_endian = "little"));

/// Elements in one block of every one of the nine formats.
const QK_K: usize = 256;

/// Input width of the matvec fixtures: two whole blocks a row, so the lane
/// walk crosses a block boundary and the `u / 8` block advance is exercised.
/// Also a multiple of 256, the width at which the engine picks the widest
/// lane split for one and two tokens.
const IN: usize = 512;
/// Output width of the matvec fixtures: a whole number of every row blocking
/// the lane splits produce (4, 8 and 16 rows a threadgroup).
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

/// The specialization of a 16-lane input split, which is what a 512-wide
/// input gets at one and two tokens. `KernelRef::consts` is `&'static`, so
/// the three splits this fixture reaches are tables rather than values built
/// per case.
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
/// The specialization of a 4-lane input split, reached only by the exhaustive
/// case below, which walks every pair the engine can select rather than the
/// pairs one width reaches.
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

/// Threads per threadgroup of the gather, mirroring
/// `shiso_engine::launch::GET_ROWS_BLOCK`.
const GET_ROWS_BLOCK: u32 = 256;
/// Threads for the table copy: the word count of the largest table
/// (`IQ1S_GRID`, 2048 `ulong` words). Threads past the selected table's word
/// count exit on the kernel's own bound.
const TABLE_COPY_WORDS: u32 = 2048;
/// Threads per threadgroup of the table copy.
const TABLE_COPY_BLOCK: u32 = 128;

/// One of the nine formats, with the entry points that read it.
#[derive(Debug, Clone, Copy)]
struct Format {
    /// The format, as the oracle's `decode_block` names it.
    dtype: DType,
    /// The spelling the entry point carries the format in. `q2_K` keeps the
    /// capital `K` the kernel was named with, which is not what
    /// `DType::type_name` returns.
    tag: &'static str,
    /// Bytes in one block of the format.
    block_bytes: usize,
    /// The five row-blocked instantiations, indexed by `r1ptg - 1`.
    entries: [&'static str; 5],
}

/// The nine formats the `kq` walk reads, in the order the engine's dispatch
/// tables name them.
const FORMATS: [Format; 9] = [
    Format {
        dtype: DType::Q2_K,
        tag: "q2_K",
        block_bytes: 84,
        entries: [
            "kernel_mul_mv_ext_q2_K_f32_r1_1",
            "kernel_mul_mv_ext_q2_K_f32_r1_2",
            "kernel_mul_mv_ext_q2_K_f32_r1_3",
            "kernel_mul_mv_ext_q2_K_f32_r1_4",
            "kernel_mul_mv_ext_q2_K_f32_r1_5",
        ],
    },
    Format {
        dtype: DType::IQ2_XXS,
        tag: "iq2_xxs",
        block_bytes: 66,
        entries: [
            "kernel_mul_mv_ext_iq2_xxs_f32_r1_1",
            "kernel_mul_mv_ext_iq2_xxs_f32_r1_2",
            "kernel_mul_mv_ext_iq2_xxs_f32_r1_3",
            "kernel_mul_mv_ext_iq2_xxs_f32_r1_4",
            "kernel_mul_mv_ext_iq2_xxs_f32_r1_5",
        ],
    },
    Format {
        dtype: DType::IQ2_XS,
        tag: "iq2_xs",
        block_bytes: 74,
        entries: [
            "kernel_mul_mv_ext_iq2_xs_f32_r1_1",
            "kernel_mul_mv_ext_iq2_xs_f32_r1_2",
            "kernel_mul_mv_ext_iq2_xs_f32_r1_3",
            "kernel_mul_mv_ext_iq2_xs_f32_r1_4",
            "kernel_mul_mv_ext_iq2_xs_f32_r1_5",
        ],
    },
    Format {
        dtype: DType::IQ2_S,
        tag: "iq2_s",
        block_bytes: 82,
        entries: [
            "kernel_mul_mv_ext_iq2_s_f32_r1_1",
            "kernel_mul_mv_ext_iq2_s_f32_r1_2",
            "kernel_mul_mv_ext_iq2_s_f32_r1_3",
            "kernel_mul_mv_ext_iq2_s_f32_r1_4",
            "kernel_mul_mv_ext_iq2_s_f32_r1_5",
        ],
    },
    Format {
        dtype: DType::IQ3_XXS,
        tag: "iq3_xxs",
        block_bytes: 98,
        entries: [
            "kernel_mul_mv_ext_iq3_xxs_f32_r1_1",
            "kernel_mul_mv_ext_iq3_xxs_f32_r1_2",
            "kernel_mul_mv_ext_iq3_xxs_f32_r1_3",
            "kernel_mul_mv_ext_iq3_xxs_f32_r1_4",
            "kernel_mul_mv_ext_iq3_xxs_f32_r1_5",
        ],
    },
    Format {
        dtype: DType::IQ3_S,
        tag: "iq3_s",
        block_bytes: 110,
        entries: [
            "kernel_mul_mv_ext_iq3_s_f32_r1_1",
            "kernel_mul_mv_ext_iq3_s_f32_r1_2",
            "kernel_mul_mv_ext_iq3_s_f32_r1_3",
            "kernel_mul_mv_ext_iq3_s_f32_r1_4",
            "kernel_mul_mv_ext_iq3_s_f32_r1_5",
        ],
    },
    Format {
        dtype: DType::IQ1_S,
        tag: "iq1_s",
        block_bytes: 50,
        entries: [
            "kernel_mul_mv_ext_iq1_s_f32_r1_1",
            "kernel_mul_mv_ext_iq1_s_f32_r1_2",
            "kernel_mul_mv_ext_iq1_s_f32_r1_3",
            "kernel_mul_mv_ext_iq1_s_f32_r1_4",
            "kernel_mul_mv_ext_iq1_s_f32_r1_5",
        ],
    },
    Format {
        dtype: DType::IQ1_M,
        tag: "iq1_m",
        block_bytes: 56,
        entries: [
            "kernel_mul_mv_ext_iq1_m_f32_r1_1",
            "kernel_mul_mv_ext_iq1_m_f32_r1_2",
            "kernel_mul_mv_ext_iq1_m_f32_r1_3",
            "kernel_mul_mv_ext_iq1_m_f32_r1_4",
            "kernel_mul_mv_ext_iq1_m_f32_r1_5",
        ],
    },
    Format {
        dtype: DType::IQ4_XS,
        tag: "iq4_xs",
        block_bytes: 136,
        entries: [
            "kernel_mul_mv_ext_iq4_xs_f32_r1_1",
            "kernel_mul_mv_ext_iq4_xs_f32_r1_2",
            "kernel_mul_mv_ext_iq4_xs_f32_r1_3",
            "kernel_mul_mv_ext_iq4_xs_f32_r1_4",
            "kernel_mul_mv_ext_iq4_xs_f32_r1_5",
        ],
    },
];

/// The format a case names by `DType` rather than by position.
///
/// # Panics
///
/// When `dtype` is not one of the nine; every caller names one.
fn format_of(dtype: DType) -> &'static Format {
    FORMATS
        .iter()
        .find(|f| f.dtype == dtype)
        .unwrap_or_else(|| panic!("{dtype:?} is not one of the nine mixed formats"))
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

/// The row-gather argument block (`metal/generic/get_rows.metal`), mirroring
/// `shiso_engine::launch::GetRowsArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GetRowsArgs {
    n_embd: i32,
    n_vocab: i32,
    n_tokens: i32,
    pad0: i32,
    src_row_bytes: u64,
    dst_row_bytes: u64,
    token_stride: u64,
}
const _: () = assert!(size_of::<GetRowsArgs>() == 40);

/// The table-copy argument block (`metal/generic/iq_tables.metal`). The
/// destination is not in here: the kernel takes it as a bound buffer at
/// index 1, because a Metal buffer carries no address the host could write
/// into a struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct IqTableCopyArgs {
    table: u32,
}
const _: () = assert!(size_of::<IqTableCopyArgs>() == 4);

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

/// `SplitMix64`, the generator behind the seeded fixtures: every seed bit
/// reaches every output bit, so a fixture that is supposed to cover the whole
/// bit width of a code field actually does.
#[derive(Debug, Clone)]
struct Codes(u64);

impl Codes {
    /// A generator seeded with `seed`.
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Fills `out` with random bytes.
    fn fill(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let word = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
    }
}

/// Finite `f16` bit patterns the block scale fields are pinned to. Every
/// other byte of a fixture block is random, and a random `u16` read as `f16`
/// is an `Inf` or a `NaN` about one block in a hundred - which would compare
/// unequal to itself and fail the case for a reason that has nothing to do
/// with a kernel. The scale is the only field with a float interpretation,
/// so it is the only field that is not left random - but pinning it to
/// powers of two would hold every mantissa bit of the scale path at zero,
/// so all four values (1.5, 0.75, 2.5, 1.125) carry a nonzero mantissa.
const D_SCALES: [u16; 4] = [0x3E00, 0x3A00, 0x4100, 0x3C80];
/// The same, for `q2_K`'s second scale field. `dmin` is that format's
/// signed component, so the table mixes signs (-1.5, 1.25, -0.75, 3.0) and
/// the fixture exercises the `f16` sign path as well as the mantissa.
const DMIN_SCALES: [u16; 4] = [0xBE00, 0x3D00, 0xBA00, 0x4200];

/// Pins one block's scale field(s) to finite `f16` patterns, leaving every
/// code, sign and index byte random.
///
/// `q2_K` keeps its two `f16`s at the end of the block. `iq1_m` stores no
/// `f16` at all: it is assembled from the four top nibbles of the odd scale
/// bytes, so only those nibbles are written and every other bit of the
/// scale words stays random - including the multiplier bits, whose top two
/// bits live in the even bytes the pinning does not touch. Every other
/// format stores its `f16` as the block's first two bytes.
fn pin_scales(fmt: &Format, block: &mut [u8], index: usize) {
    let d = D_SCALES[index % D_SCALES.len()];
    match fmt.dtype {
        DType::Q2_K => {
            block[80..82].copy_from_slice(&d.to_le_bytes());
            block[82..84].copy_from_slice(&DMIN_SCALES[index % DMIN_SCALES.len()].to_le_bytes());
        }
        DType::IQ1_M => {
            for word in 0..4 {
                block[48 + 2 * word + 1] =
                    (block[48 + 2 * word + 1] & 0x0F) | (((d >> (4 * word)) & 0xF) as u8) << 4;
            }
        }
        _ => block[..2].copy_from_slice(&d.to_le_bytes()),
    }
}

/// `count` packed blocks of `fmt`: random codes with the scale fields pinned.
///
/// The two assertions are the fixture's own coverage gate. A generator that
/// quietly produced only low grid indices - every code under 128, say - would
/// pass every case here vacuously, because the oracle decodes the same bytes
/// the kernel does. Requiring every bit position of the fixture to take both
/// values is the cheap statement that the codes span the full width of the
/// fields the formats read.
///
/// # Panics
///
/// When the generated bytes leave a bit position constant, which is a broken
/// fixture rather than a broken kernel.
fn random_blocks(fmt: &Format, count: usize, seed: u64) -> Vec<u8> {
    let mut rng = Codes::new(seed);
    let mut bytes = vec![0u8; count * fmt.block_bytes];
    rng.fill(&mut bytes);
    for (index, block) in bytes.chunks_mut(fmt.block_bytes).enumerate() {
        pin_scales(fmt, block, index);
    }
    let or_all = bytes.iter().fold(0u8, |acc, &b| acc | b);
    let and_all = bytes.iter().fold(0xFFu8, |acc, &b| acc & b);
    assert_eq!(or_all, 0xFF, "{} fixture never sets a code bit", fmt.tag);
    assert_eq!(and_all, 0x00, "{} fixture always sets a code bit", fmt.tag);
    bytes
}

/// The oracle row: every block of `packed` decoded through the host
/// reference, concatenated in element order.
///
/// # Panics
///
/// When `packed` is not a whole number of blocks, or the oracle refuses the
/// format - both are fixture errors.
fn decode_row(fmt: &Format, packed: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() / fmt.block_bytes * QK_K);
    let mut block_out = [0.0f32; QK_K];
    for block in packed.chunks_exact(fmt.block_bytes) {
        decode_block(fmt.dtype, block, &mut block_out)
            .unwrap_or_else(|err| panic!("{} block rejected by the oracle: {err}", fmt.tag));
        out.extend_from_slice(&block_out);
    }
    out
}

/// The matvec fixture for one format: `OUT` packed rows plus the same rows
/// dequantized by the oracle.
struct Fixture {
    packed: Vec<u8>,
    rows: Vec<Vec<f32>>,
    row_bytes: u64,
}

/// Builds the fixture: `OUT` rows of two blocks each, random codes.
fn fixture(fmt: &Format, seed: u64) -> Fixture {
    let blocks_per_row = IN / QK_K;
    let packed = random_blocks(fmt, blocks_per_row * OUT, seed);
    let row_bytes = (blocks_per_row * fmt.block_bytes) as u64;
    let rows = packed
        .chunks_exact(row_bytes as usize)
        .map(|row| decode_row(fmt, row))
        .collect();
    Fixture {
        packed,
        rows,
        row_bytes,
    }
}

/// The entry name and constant table for one `(r1ptg, nxpsg)` pair of one
/// format.
///
/// This family has only the constant spelling. `shiso_engine::launch::ext_name_for`
/// returns `None` for these nine on a device that folds the lane split into
/// the entry name, because no such instantiation exists anywhere in the tree,
/// and a case that invented the name would be launching a kernel nobody built.
/// Metal is not such a device, which is why the assertion below has never
/// fired on the device these cases gate.
///
/// # Panics
///
/// When `nxpsg` is not one of the three splits the family is built with, or
/// when the device spells specializations into entry names.
fn matvec_entry(
    gpu: &Gpu,
    fmt: &Format,
    r1ptg: u32,
    nxpsg: u32,
) -> (&'static str, &'static [FnConst]) {
    assert!(
        !gpu.device().specializes_by_name(),
        "the nine mixed formats have no by-name instantiation; {} cannot run on this device",
        fmt.tag,
    );
    assert!(
        (1..=5).contains(&r1ptg),
        "{r1ptg} is not a row blocking the family is built with"
    );
    let consts: &'static [FnConst] = match nxpsg {
        4 => &EXT_CONSTS_4,
        8 => &EXT_CONSTS_8,
        16 => &EXT_CONSTS_16,
        other => panic!("{other} is not a lane split the extended family is built with"),
    };
    (fmt.entries[r1ptg as usize - 1], consts)
}

/// Runs one format's matvec at one `(r1ptg, nxpsg)` pair over `n_tok` tokens
/// and compares every output against the oracle dot product of the same
/// numbers.
fn matvec_at(gpu: &Gpu, fmt: &Format, fx: &Fixture, r1ptg: u32, nxpsg: u32, n_tok: usize) {
    let (entry, consts) = matvec_entry(gpu, fmt, r1ptg, nxpsg);
    let x = crate::oracle::signal(IN * n_tok, 0x5A5A);

    let weights = gpu.upload(&fx.packed);
    let src = gpu.upload(&f32_bytes(&x));
    let dst = gpu.upload(&vec![0u8; n_tok * OUT * 4]);
    let args = MulMvExtArgs {
        ne00: IN as i32,
        ne01: OUT as i32,
        ne02: 1,
        _pad0: 0,
        nb00: fx.row_bytes,
        nb01: fx.row_bytes,
        nb02: fx.row_bytes * OUT as u64,
        nb03: fx.row_bytes * OUT as u64,
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
        for (out, row) in fx.rows.iter().enumerate() {
            crate::assert_close!(
                got[tok * OUT + out],
                crate::oracle::dot(row, activations),
                crate::oracle::TOL_QUANT,
                "{entry}: token {tok} output {out}",
            );
        }
    }
}

/// Every mixed i-quant matvec against the oracle dot product, at the token
/// counts and lane splits the engine actually picks.
///
/// The fixture is random codes, never a quantized signal: these formats are
/// codebooks, and the only statement worth grading is that the device reads
/// the same bytes the host reference decoder reads.
pub fn iq_matvec_matches_the_oracle_dot(gpu: &Gpu) {
    for (index, fmt) in FORMATS.iter().enumerate() {
        let fx = fixture(fmt, 0x1234 ^ (index as u64));
        for n_tok in 1..=5 {
            matvec_at(gpu, fmt, &fx, ext_r1ptg(n_tok), ext_nxpsg(IN, n_tok), n_tok);
        }
    }
}

/// Every `(r1ptg, nxpsg)` pair of all nine formats resolves and agrees with
/// the oracle.
///
/// The case above reaches five of the fifteen pairs per format: a 512-wide
/// input only ever asks for two of the three lane splits, and each row
/// blocking only at the token count that selects it. `r1ptg = 4` in
/// particular is what the engine picks at four tokens and at six and above -
/// the widest blocking the family compiles - and no single width or token
/// count reaches it together with all three splits. The pairs are walked
/// directly instead, with the token count set to `r1ptg` so every row the
/// blocking carries is a real token rather than a guarded tail row.
pub fn every_iq_matvec_instantiation_agrees_with_the_oracle(gpu: &Gpu) {
    for (index, fmt) in FORMATS.iter().enumerate() {
        let fx = fixture(fmt, 0xABCD ^ (index as u64));
        for r1ptg in 1..=5u32 {
            for nxpsg in [4u32, 8, 16] {
                matvec_at(gpu, fmt, &fx, r1ptg, nxpsg, r1ptg as usize);
            }
        }
    }
}

/// Vocabulary rows in the gather fixture.
const VOCAB: usize = 6;
/// Blocks per gathered row: two whole 256-weight blocks.
const GATHER_BLOCKS: usize = 2;
/// Tokens, including a repeat so a kernel that gathered by position rather
/// than by id is caught.
const GATHER_IDS: [i32; 4] = [3, 0, 3, 5];

/// The `iq1_m` gather against the oracle rows, bit for bit.
///
/// The gather and the host decoder are the same expression in the same
/// order - the assembled `f16`, the per-subblock multiplier, the grid byte
/// plus the delta - so this compares bit patterns, not a tolerance. A drift
/// of one ulp here means one of the three read the format differently.
pub fn iq1_m_get_rows_matches_the_oracle(gpu: &Gpu) {
    let fmt = format_of(DType::IQ1_M);
    let packed = random_blocks(fmt, VOCAB * GATHER_BLOCKS, 0x6E77);
    let rows: Vec<Vec<f32>> = packed
        .chunks_exact(GATHER_BLOCKS * fmt.block_bytes)
        .map(|row| decode_row(fmt, row))
        .collect();
    let embd = GATHER_BLOCKS * QK_K;

    let table = gpu.upload(&packed);
    let ids: Vec<i32> = GATHER_IDS.to_vec();
    let d_ids = gpu.upload(bytemuck::cast_slice(&ids));
    let dst = gpu.upload(&vec![0u8; GATHER_IDS.len() * embd * 4]);
    let args = GetRowsArgs {
        n_embd: embd as i32,
        n_vocab: VOCAB as i32,
        n_tokens: GATHER_IDS.len() as i32,
        pad0: 0,
        src_row_bytes: (GATHER_BLOCKS * fmt.block_bytes) as u64,
        dst_row_bytes: (embd * 4) as u64,
        token_stride: 4,
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(table.view()),
        KernelArg::Buffer(d_ids.view()),
        KernelArg::Buffer(dst.view()),
    ];
    run(
        gpu,
        KernelRef::new("generic", "kernel_get_rows_iq1_m_f32"),
        &LaunchArgs::new(
            Extent3::new(GATHER_BLOCKS as u32, GATHER_IDS.len() as u32, 1),
            Extent3::linear(GET_ROWS_BLOCK),
            &params,
        ),
    );
    drain(gpu);
    let got = gpu.read_f32(&dst);
    for (tok, &id) in GATHER_IDS.iter().enumerate() {
        let want = &rows[usize::try_from(id).expect("fixture id is negative")];
        for (elem, (&g, &w)) in got[tok * embd..(tok + 1) * embd]
            .iter()
            .zip(want)
            .enumerate()
        {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "iq1_m gather: token {tok} element {elem} is {g}, oracle {w}",
            );
        }
    }
}

/// The nine device tables, in the order `kernel_iq_table_copy` indexes them,
/// as the native-endian bytes of the host constants they must match.
fn host_tables() -> [(&'static str, &'static [u8]); 9] {
    [
        ("IQ2XXS_GRID", bytemuck::cast_slice(&IQ2XXS_GRID)),
        ("IQ2XS_GRID", bytemuck::cast_slice(&IQ2XS_GRID)),
        ("IQ2S_GRID", bytemuck::cast_slice(&IQ2S_GRID)),
        ("IQ3XXS_GRID", bytemuck::cast_slice(&IQ3XXS_GRID)),
        ("IQ3S_GRID", bytemuck::cast_slice(&IQ3S_GRID)),
        ("IQ1S_GRID", bytemuck::cast_slice(&IQ1S_GRID)),
        ("IQ4NL", bytemuck::cast_slice(&IQ4NL_VALUES)),
        ("KSIGNS_IQ2XS", bytemuck::cast_slice(&KSIGNS_IQ2XS)),
        ("KSIGNS64", bytemuck::cast_slice(&KSIGNS64)),
    ]
}

/// Every codebook table on the device, byte for byte, against the host
/// constant it is a transcription of.
///
/// The destination is pre-filled with a sentinel so a kernel that wrote
/// nothing fails rather than agrees, and it is bound as `buffer(1)` rather
/// than named inside the argument block, which is how the kernel takes it.
pub fn iq_tables_match_the_host_copy(gpu: &Gpu) {
    for (index, (name, want)) in host_tables().iter().enumerate() {
        let dst = gpu.upload(&vec![0xA5u8; want.len()]);
        let args = IqTableCopyArgs {
            table: index as u32,
        };
        let params = [
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(dst.view()),
        ];
        run(
            gpu,
            KernelRef::new("generic", "kernel_iq_table_copy"),
            &LaunchArgs::new(
                Extent3::linear(TABLE_COPY_WORDS),
                Extent3::linear(TABLE_COPY_BLOCK),
                &params,
            ),
        );
        drain(gpu);
        assert_eq!(
            gpu.read(&dst),
            *want,
            "device copy of {name} (table {index}) differs from the host table",
        );
    }
}
