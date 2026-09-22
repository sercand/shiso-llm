//! The vocabulary gather and the three products a Gemma 4 GGUF pass reaches
//! outside the `dense` module.
//!
//! `kernel_get_rows_q6_K_f32` is the first kernel a prompt touches: the QAT
//! export packs the vocabulary table at `q6_K` whatever its linear layers are
//! packed at. `kernel_mul_mv_ext_q6_K_f32_r1_*_nx*` is the output head, which
//! is that same table tied back and therefore has no tiled product to fall
//! through to. `kernel_mul_mm_f16_f32` and `kernel_mul_mv_f16_f32_4` are the
//! per-layer model projection, the one unquantized weight in the container, at
//! its prefill and its decode width.
//!
//! The `q6_K` packer and dequantizer are `crate::dense::ple`'s, re-used rather
//! than rewritten: a second packer would be a second opinion about the format
//! and the point of an oracle is that it is not.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "kernel argument structs take i32/u32/u64 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]
#![expect(
    clippy::similar_names,
    reason = "r1ptg and nxpsg are the engine's own names for the two widths"
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};
use shiso_modelfile::quant::BlockQ6K;

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::{TOL_F32, TOL_QUANT};

/// Every entry point this file's cases launch, the fifteen matvec
/// instantiations written out because a pair that is merely absent is invisible
/// at any single width.
pub const ENTRIES: [&str; 23] = [
    "kernel_get_rows_q6_K_f32",
    // The two entries `qwen35moe` was the first CUDA family to name. The half
    // gather and the single-precision matvec had existed in the Metal set and
    // in `launch.rs`'s own tables for as long as there had been a directory
    // checkpoint, and on this backend neither was there: every CUDA family so
    // far read a `q6_K` vocabulary table and had no unquantized weight that
    // was not half precision. Both refused by name at the first launch that
    // wanted them, which is the twelfth kind of defect and the only one that
    // says what it is.
    "kernel_get_rows_f16_f32",
    "kernel_mul_mv_f32_f32_4",
    "kernel_mul_mm_f16_f32",
    "kernel_mul_mv_f16_f32_4",
    "kernel_mul_mv_ext_q6_K_f32_r1_1_nx4",
    "kernel_mul_mv_ext_q6_K_f32_r1_1_nx8",
    "kernel_mul_mv_ext_q6_K_f32_r1_1_nx16",
    "kernel_mul_mv_ext_q6_K_f32_r1_2_nx4",
    "kernel_mul_mv_ext_q6_K_f32_r1_2_nx8",
    "kernel_mul_mv_ext_q6_K_f32_r1_2_nx16",
    "kernel_mul_mv_ext_q6_K_f32_r1_3_nx4",
    "kernel_mul_mv_ext_q6_K_f32_r1_3_nx8",
    "kernel_mul_mv_ext_q6_K_f32_r1_3_nx16",
    "kernel_mul_mv_ext_q6_K_f32_r1_4_nx4",
    "kernel_mul_mv_ext_q6_K_f32_r1_4_nx8",
    "kernel_mul_mv_ext_q6_K_f32_r1_4_nx16",
    "kernel_mul_mv_ext_q6_K_f32_r1_5_nx4",
    "kernel_mul_mv_ext_q6_K_f32_r1_5_nx8",
    "kernel_mul_mv_ext_q6_K_f32_r1_5_nx16",
    "kernel_mul_mm_q4_0_f32",
    "kernel_mul_mm_q8_0_f32",
    "kernel_mul_mv_q8_0_f32",
];

/// Weights in one `q6_K` block.
pub(crate) const QK_K: usize = 256;
/// Weights in one `q8_0` block.
pub(crate) const QK8_0: usize = 32;
/// Threads per threadgroup of the gather, mirroring
/// `shiso_engine::launch::GET_ROWS_BLOCK`.
const GET_ROWS_BLOCK: u32 = 256;
/// Lanes in a warp, mirroring `shiso_engine::launch::WARP`.
const WARP: u32 = 32;
/// Simdgroups per threadgroup of the extended matvec, mirroring
/// `shiso_engine::launch::EXT_NSG`.
const EXT_NSG: u32 = 2;
/// Threads per threadgroup of the prefill GEMM, mirroring
/// `shiso_engine::launch::GEMM_BLOCK`.
const GEMM_BLOCK: u32 = 128;
/// Threadgroup bytes it stages into, mirroring
/// `shiso_engine::launch::GEMM_SMEM_BYTES`.
const GEMM_SMEM: u32 = 8192;
/// Output rows one GEMM tile covers.
const GEMM_TILE_ROWS: u32 = 64;
/// Tokens one GEMM tile covers.
const GEMM_TILE_TOKENS: u32 = 32;
/// Threadgroup bytes the plain matvec's reduction needs, mirroring
/// `shiso_engine::launch::PLAIN_SMEM_BYTES`.
const PLAIN_SMEM: u32 = WARP * 2 * 4;
/// Output rows one plain-matvec threadgroup produces, mirroring
/// `shiso_engine::launch::PLAIN_ROWS`.
const PLAIN_ROWS: u32 = 2;
/// Simdgroups per threadgroup of the `q8_0` plain matvec, mirroring
/// `shiso_engine::launch::Q8_MV_NSG`. Fixed rather than derived: that codec's
/// entry takes no width-dependent count on either backend.
const Q8_MV_NSG: u32 = 4;

// -- argument blocks --------------------------------------------------------

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

/// The extended matvec's argument block (`metal/generic/mul_mv_ext.metal`),
/// mirroring `shiso_engine::launch::MulMvExtArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MulMvExtArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    pad0: [u8; 4],
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    pad1: [u8; 4],
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
    pad2: [i8; 4],
}

const _: () = assert!(size_of::<MulMvExtArgs>() == 112);

/// The plain matvec's argument block (`metal/generic/mul_mv_dense.metal`),
/// mirroring `shiso_engine::launch::MulMvArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MulMvArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    pad0: [u8; 4],
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    pad1: [u8; 4],
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    nr0: i32,
    r2: i16,
    r3: i16,
}

const _: () = assert!(size_of::<MulMvArgs>() == 112);

/// The prefill GEMM's argument block (`metal/qwen35/gemm_q4k.metal`), mirroring
/// `shiso_engine::launch::MulMmArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MulMmArgs {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    pad0: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
    pad2: [i8; 4],
}

const _: () = assert!(size_of::<MulMmArgs>() == 88);

// -- fixtures ---------------------------------------------------------------

/// `f16` bits for 1.0.
const ONE_F16: u16 = 0x3C00;

/// Signed per-16-element `q6_K` scales, both signs, as `crate::dense::ple` uses.
const Q6_SCALES: [i8; 16] = [
    17, -33, 5, -1, 64, -64, 22, -7, 1, -1, 31, -31, 12, -12, 40, -40,
];

/// The `qh` bit position carrying each 32-element quarter's top two bits.
const QH_SHIFTS: [u32; 4] = [0, 2, 4, 6];

/// Raw six-bit weights for one block, covering the whole `0..64` range.
fn q6_values(block: usize) -> [u8; QK_K] {
    let mut values = [0u8; QK_K];
    for (e, slot) in values.iter_mut().enumerate() {
        *slot = ((e * 3 + block * 5) % 64) as u8;
    }
    values
}

/// Packs 256 raw six-bit values into a `q6_K` block.
fn pack_q6_k(values: &[u8; QK_K]) -> BlockQ6K {
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

/// Element `e` of a `q6_K` block, the `-32` bias folded before the multiply.
fn dequant_q6_k(block: &BlockQ6K, e: usize) -> f32 {
    let half = e / 128;
    let quarter = (e % 128) / 32;
    let l = e % 32;
    let scale = f32::from(block.scales[half * 8 + quarter * 2 + l / 16]);
    let d = crate::oracle::f16_to_f32(block.d) * scale;
    let ql = block.ql[half * 64 + (quarter % 2) * 32 + l] >> ((quarter / 2) * 4);
    let qh = block.qh[half * 32 + l] >> QH_SHIFTS[quarter];
    let v = u32::from(ql & 0x0F) | (u32::from(qh & 3) << 4);
    d * (v as i32 - 32) as f32
}

/// A `rows`-by-`width` `q6_K` matrix, block `i` seeded by its own index.
fn q6_matrix(rows: usize, width: usize) -> (Vec<BlockQ6K>, Vec<f32>) {
    let per_row = width / QK_K;
    let blocks: Vec<BlockQ6K> = (0..rows * per_row)
        .map(|b| pack_q6_k(&q6_values(b)))
        .collect();
    let mut values = vec![0.0f32; rows * width];
    for r in 0..rows {
        for b in 0..per_row {
            let block = &blocks[r * per_row + b];
            for e in 0..QK_K {
                values[r * width + b * QK_K + e] = dequant_q6_k(block, e);
            }
        }
    }
    (blocks, values)
}

/// The bytes of a `q6_K` block run, as the device reads them.
fn q6_bytes(blocks: &[BlockQ6K]) -> Vec<u8> {
    bytemuck::cast_slice::<BlockQ6K, u8>(blocks).to_vec()
}

/// `f32` values rounded to `f16` and back, which is what a half-precision
/// weight holds.
pub(crate) fn to_f16_and_back(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    let mut back = Vec::with_capacity(values.len());
    for &v in values {
        let h = half::f16::from_f32(v);
        bytes.extend_from_slice(&h.to_bits().to_le_bytes());
        back.push(h.to_f32());
    }
    (bytes, back)
}

/// One `q8_0` block as both devices read it: a half scale and 32 signed bytes,
/// in element order.
///
/// Declared here rather than taken from `shiso_modelfile::quant`, which carries the
/// k-quant blocks and not this one. The static assertion is the same guard the
/// CUDA header's is: a block one byte out reads every row from the wrong
/// offset.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct BlockQ80 {
    d: u16,
    qs: [i8; QK8_0],
}

const _: () = assert!(size_of::<BlockQ80>() == 34);

/// `f16` bits for 1/64, which is exact in half and in single.
///
/// The scale is a power of two on purpose. Every dequantized weight is then
/// `q * 2^-6` with `|q| <= 127`, which is exact in half as well as in float, so
/// the GEMM's staging narrows nothing and what the two cases below measure is
/// the product and its launch rather than the codec's rounding.
const ONE_SIXTY_FOURTH_F16: u16 = 0x2400;

/// A `rows`-by-`width` `q8_0` matrix, and the exact values it dequantizes to.
///
/// The bytes walk the whole `-127..=127` range, both signs, so a dequantizer
/// that read `qs` as unsigned is caught on the first row rather than on a
/// particular one.
pub(crate) fn q8_matrix(rows: usize, width: usize) -> (Vec<BlockQ80>, Vec<f32>) {
    let per_row = width / QK8_0;
    let scale = crate::oracle::f16_to_f32(ONE_SIXTY_FOURTH_F16);
    let mut blocks = Vec::with_capacity(rows * per_row);
    let mut values = vec![0.0f32; rows * width];
    for b in 0..rows * per_row {
        let mut qs = [0i8; QK8_0];
        for (e, slot) in qs.iter_mut().enumerate() {
            *slot = (((e * 7 + b * 13) % 255) as i32 - 127) as i8;
            values[b * QK8_0 + e] = f32::from(*slot) * scale;
        }
        blocks.push(BlockQ80 {
            d: ONE_SIXTY_FOURTH_F16,
            qs,
        });
    }
    (blocks, values)
}

// -- helpers ----------------------------------------------------------------

/// Launches one entry of a named module.
fn launch(
    gpu: &Gpu,
    module: &'static str,
    entry: &'static str,
    grid: Extent3,
    block: Extent3,
    smem_bytes: u32,
    args: &[KernelArg<'_>],
) {
    run(
        gpu,
        KernelRef::new(module, entry),
        &LaunchArgs {
            grid,
            block,
            smem_bytes,
            args,
        },
    );
}

/// The row blocking and lane split an extended matvec's name states.
///
/// `..._r1_<r1ptg>_nx<nxpsg>`, which is the whole of the entry's geometry.
/// Parsed rather than derived from a table, because a table has to be kept in
/// step with a list and a name cannot drift from itself.
fn split_of(entry: &str) -> (u32, u32) {
    let (head, nx) = entry.split_once("_nx").unwrap_or_else(|| {
        panic!("{entry} is not an extended matvec name");
    });
    let r1 = head.rsplit_once("_r1_").map_or("", |(_, r)| r);
    (
        r1.parse()
            .unwrap_or_else(|_| panic!("{entry}: row blocking")),
        nx.parse().unwrap_or_else(|_| panic!("{entry}: lane split")),
    )
}

/// An f32 device buffer of `len` zeros.
fn zeros(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![0.0f32; len]))
}

/// Asserts every value agrees within `tol`, scaled by the expected magnitude.
fn close_at(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = tol * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= bound,
            "{what}: element {index} is {g}, expected {w} (tolerance {bound})",
        );
    }
}

// -- the gather -------------------------------------------------------------

/// Vocabulary rows in the gather fixture.
const VOCAB: usize = 6;
/// Width of one vocabulary row: two whole `q6_K` blocks.
const EMBD: usize = 2 * QK_K;
/// Tokens the gather covers, including a repeat so a kernel that gathered by
/// position rather than by id is caught.
const GATHER_IDS: [i32; 4] = [3, 0, 3, 5];

/// The gather dequantizes the `q6_K` rows the ids name, one f32 row per token.
pub fn the_q6_k_gather_dequantizes_the_rows_the_ids_name(gpu: &Gpu) {
    let (blocks, values) = q6_matrix(VOCAB, EMBD);
    let d_table = gpu.upload(&q6_bytes(&blocks));
    let ids: Vec<i32> = GATHER_IDS.to_vec();
    let d_ids = gpu.upload(bytemuck::cast_slice(&ids));
    let dst = zeros(gpu, GATHER_IDS.len() * EMBD);

    let per_row = EMBD / QK_K;
    let args = GetRowsArgs {
        n_embd: EMBD as i32,
        n_vocab: VOCAB as i32,
        n_tokens: GATHER_IDS.len() as i32,
        pad0: 0,
        src_row_bytes: (per_row * size_of::<BlockQ6K>()) as u64,
        dst_row_bytes: (EMBD * 4) as u64,
        token_stride: 4,
    };
    launch(
        gpu,
        "qwen35",
        "kernel_get_rows_q6_K_f32",
        Extent3::new(per_row as u32, GATHER_IDS.len() as u32, 1),
        Extent3::new(GET_ROWS_BLOCK, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_table.view()),
            KernelArg::Buffer(d_ids.view()),
            KernelArg::Buffer(dst.view()),
        ],
    );
    drain(gpu);

    let mut want = vec![0.0f32; GATHER_IDS.len() * EMBD];
    for (t, &id) in GATHER_IDS.iter().enumerate() {
        let row = id as usize;
        want[t * EMBD..(t + 1) * EMBD].copy_from_slice(&values[row * EMBD..(row + 1) * EMBD]);
    }
    close_at(&gpu.read_f32(&dst), &want, TOL_QUANT, "q6_K gather");
}

// -- the q6_K matvec --------------------------------------------------------

/// Input width of the matvec fixture: a whole number of `4 * chpt * nxpsg`
/// elements at every lane split the engine can ask for, which at sixteen lanes
/// is 256.
const MV_IN: usize = 512;
/// Output rows: enough that the grid is more than one block at every row
/// blocking, and not a multiple of them all, so the guarded store is exercised.
const MV_OUT: usize = 30;

/// Every `(r1ptg, nxpsg)` pair of the `q6_K` matvec agrees with the oracle.
///
/// Walked rather than spot-checked: the engine picks the row blocking from the
/// token count and the lane split from the input width, independently, so an
/// instantiation that is merely absent or merely wrong is invisible at any
/// single prompt length.
pub fn every_q6_k_matvec_instantiation_agrees_with_the_oracle(gpu: &Gpu) {
    let (blocks, weights) = q6_matrix(MV_OUT, MV_IN);
    let d_w = gpu.upload(&q6_bytes(&blocks));
    let per_row = MV_IN / QK_K;
    let row_bytes = (per_row * size_of::<BlockQ6K>()) as u64;

    for (index, entry) in ENTRIES.iter().enumerate() {
        if !entry.starts_with("kernel_mul_mv_ext_q6_K") {
            continue;
        }
        // Read off the NAME, not off the position in the list. This was
        // `(index - 3) / 3 + 1` and `(index - 3) % 3`, which encoded where the
        // q6_K run happened to start - so inserting two unrelated entries
        // above it launched every instantiation under another's geometry and
        // the case failed at "element 16 is 0" without a word about why. The
        // name already says both numbers.
        let (r1ptg, nxpsg) = split_of(entry);
        let n_tok = r1ptg as usize;
        let x = crate::oracle::signal(n_tok * MV_IN, 90 + index as u32);
        let d_x = gpu.upload(&f32_bytes(&x));
        let dst = zeros(gpu, n_tok * MV_OUT);

        let act_row = (MV_IN * 4) as u64;
        let args = MulMvExtArgs {
            ne00: MV_IN as i32,
            ne01: MV_OUT as i32,
            ne02: 1,
            pad0: [0; 4],
            nb00: 0,
            nb01: row_bytes,
            nb02: row_bytes * MV_OUT as u64,
            nb03: row_bytes * MV_OUT as u64,
            ne10: MV_IN as i32,
            ne11: n_tok as i32,
            ne12: 1,
            pad1: [0; 4],
            nb10: 4,
            nb11: act_row,
            nb12: act_row * n_tok as u64,
            nb13: act_row * n_tok as u64,
            ne0: MV_OUT as i32,
            ne1: n_tok as i32,
            r2: 1,
            r3: 1,
            pad2: [0; 4],
        };
        // `ext_grid`: the output rows are split across the lane groups of a
        // block as well as across blocks, which is what the `32 / nxpsg` is.
        let nypsg = 32 / nxpsg;
        let grid = Extent3::new(
            (MV_OUT as u32).div_ceil(nypsg * EXT_NSG),
            (n_tok as u32).div_ceil(r1ptg),
            1,
        );
        run(
            gpu,
            KernelRef::new("qwen35", entry),
            &LaunchArgs {
                grid,
                block: Extent3::new(WARP, EXT_NSG, 1),
                smem_bytes: 0,
                args: &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(d_w.view()),
                    KernelArg::Buffer(d_x.view()),
                    KernelArg::Buffer(dst.view()),
                ],
            },
        );
        drain(gpu);

        let mut want = vec![0.0f32; n_tok * MV_OUT];
        for t in 0..n_tok {
            for r in 0..MV_OUT {
                want[t * MV_OUT + r] = crate::oracle::dot(
                    &weights[r * MV_IN..(r + 1) * MV_IN],
                    &x[t * MV_IN..(t + 1) * MV_IN],
                );
            }
        }
        close_at(
            &gpu.read_f32(&dst),
            &want,
            TOL_QUANT,
            &format!("q6_K matvec r1ptg {r1ptg} nxpsg {nxpsg}"),
        );
    }
}

// -- the f16 products -------------------------------------------------------

/// Input width of the f16 fixtures: a whole number of the GEMM's k tile and of
/// the plain matvec's block run.
const F16_IN: usize = 256;
/// Output rows, deliberately not a multiple of the GEMM's 64-row tile nor of
/// the matvec's two-row block, so both guarded epilogues run.
const F16_OUT: usize = 70;

/// The f16 prefill GEMM agrees with a triple loop over the same numbers.
pub fn the_f16_prefill_gemm_matches_a_triple_loop(gpu: &Gpu) {
    let n_tok = 40usize;
    let raw = crate::oracle::signal(F16_OUT * F16_IN, 101);
    let (w_bytes, weights) = to_f16_and_back(&raw);
    let d_w = gpu.upload(&w_bytes);
    // The activations are narrowed to `half` by the oracle as well as by the
    // kernel, because that is the contract rather than an approximation: the
    // Metal template stages both tiles as `half` (`S0`/`S1`) and accumulates in
    // `float`, and a reference that summed f32 activations would be comparing
    // two different products and calling the difference a tolerance.
    let x_raw = crate::oracle::signal(n_tok * F16_IN, 102);
    let (_, x) = to_f16_and_back(&x_raw);
    let d_x = gpu.upload(&f32_bytes(&x_raw));
    let dst = zeros(gpu, n_tok * F16_OUT);

    let row = (F16_IN * 2) as u64;
    let plane = row * F16_OUT as u64;
    let act_row = (F16_IN * 4) as u64;
    let args = MulMmArgs {
        ne00: F16_IN as i32,
        ne02: 1,
        nb01: row,
        nb02: plane,
        nb03: plane,
        ne12: 1,
        pad0: 0,
        nb10: 4,
        nb11: act_row,
        nb12: act_row * n_tok as u64,
        nb13: act_row * n_tok as u64,
        ne0: F16_OUT as i32,
        ne1: n_tok as i32,
        r2: 1,
        r3: 1,
        pad2: [0; 4],
    };
    launch(
        gpu,
        "qwen35",
        "kernel_mul_mm_f16_f32",
        Extent3::new(
            (n_tok as u32).div_ceil(GEMM_TILE_TOKENS),
            (F16_OUT as u32).div_ceil(GEMM_TILE_ROWS),
            1,
        ),
        Extent3::new(GEMM_BLOCK, 1, 1),
        GEMM_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(d_x.view()),
            KernelArg::Buffer(dst.view()),
        ],
    );
    drain(gpu);

    let mut want = vec![0.0f32; n_tok * F16_OUT];
    for t in 0..n_tok {
        for r in 0..F16_OUT {
            want[t * F16_OUT + r] = crate::oracle::dot(
                &weights[r * F16_IN..(r + 1) * F16_IN],
                &x[t * F16_IN..(t + 1) * F16_IN],
            );
        }
    }
    // Both sides narrow the same way, so what is left is summation order: the
    // kernel sums along k in index order over 32-wide tiles and the oracle in
    // one pass, and `TOL_QUANT` is the tolerance the q4_0 GEMM's case already
    // grades that difference at.
    close_at(&gpu.read_f32(&dst), &want, TOL_QUANT, "f16 prefill gemm");
}

/// The f16 decode matvec agrees with the same dot product.
///
/// `nsg` is a launch constant on Metal and the block's second extent on CUDA;
/// the case sets the extent, which is the number the host sets either way.
pub fn the_f16_decode_matvec_matches_the_oracle_dot(gpu: &Gpu) {
    let nsg = (F16_IN as u32).div_ceil(128).min(8);
    let raw = crate::oracle::signal(F16_OUT * F16_IN, 103);
    let (w_bytes, weights) = to_f16_and_back(&raw);
    let d_w = gpu.upload(&w_bytes);
    let x = crate::oracle::signal(F16_IN, 104);
    let d_x = gpu.upload(&f32_bytes(&x));
    let dst = zeros(gpu, F16_OUT);

    let row = (F16_IN * 2) as u64;
    let plane = row * F16_OUT as u64;
    let act_row = (F16_IN * 4) as u64;
    let args = MulMvArgs {
        ne00: F16_IN as i32,
        ne01: F16_OUT as i32,
        ne02: 1,
        pad0: [0; 4],
        nb00: 2,
        nb01: row,
        nb02: plane,
        nb03: plane,
        ne10: F16_IN as i32,
        ne11: 1,
        ne12: 1,
        pad1: [0; 4],
        nb10: 4,
        nb11: act_row,
        nb12: act_row,
        nb13: act_row,
        ne0: F16_OUT as i32,
        ne1: 1,
        nr0: PLAIN_ROWS as i32,
        r2: 1,
        r3: 1,
    };
    launch(
        gpu,
        "generic",
        "kernel_mul_mv_f16_f32_4",
        Extent3::new((F16_OUT as u32).div_ceil(PLAIN_ROWS), 1, 1),
        Extent3::new(WARP, nsg, 1),
        PLAIN_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(d_x.view()),
            KernelArg::Buffer(dst.view()),
        ],
    );
    drain(gpu);

    let want: Vec<f32> = (0..F16_OUT)
        .map(|r| crate::oracle::dot(&weights[r * F16_IN..(r + 1) * F16_IN], &x))
        .collect();
    close_at(&gpu.read_f32(&dst), &want, TOL_F32, "f16 decode matvec");
}

/// The f32 decode matvec agrees with the same dot product.
///
/// The twin of the case above, at the element type `qwen35moe`'s router, its
/// shared expert's scalar gate and the recurrent block's two drivers arrive
/// as. Unlike that one there is no narrowing on either side: the weights are
/// already `f32`, so what is left is summation order alone and the tolerance
/// is the tighter of the two.
pub fn the_f32_decode_matvec_matches_the_oracle_dot(gpu: &Gpu) {
    let nsg = (F16_IN as u32).div_ceil(128).min(8);
    let weights = crate::oracle::signal(F16_OUT * F16_IN, 106);
    let d_w = gpu.upload(&f32_bytes(&weights));
    let x = crate::oracle::signal(F16_IN, 107);
    let d_x = gpu.upload(&f32_bytes(&x));
    let dst = zeros(gpu, F16_OUT);

    let row = (F16_IN * 4) as u64;
    let plane = row * F16_OUT as u64;
    let args = MulMvArgs {
        ne00: F16_IN as i32,
        ne01: F16_OUT as i32,
        ne02: 1,
        pad0: [0; 4],
        // Four bytes an element, not two: the same argument block with the
        // other element width, and a kernel handed the half strides would read
        // every row at half its length.
        nb00: 4,
        nb01: row,
        nb02: plane,
        nb03: plane,
        ne10: F16_IN as i32,
        ne11: 1,
        ne12: 1,
        pad1: [0; 4],
        nb10: 4,
        nb11: row,
        nb12: row,
        nb13: row,
        ne0: F16_OUT as i32,
        ne1: 1,
        nr0: PLAIN_ROWS as i32,
        r2: 1,
        r3: 1,
    };
    launch(
        gpu,
        "generic",
        "kernel_mul_mv_f32_f32_4",
        // 70 output rows over two per block is 35, which is deliberately not a
        // whole number of anything: the last block's guarded epilogue runs.
        Extent3::new((F16_OUT as u32).div_ceil(PLAIN_ROWS), 1, 1),
        Extent3::new(WARP, nsg, 1),
        PLAIN_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(d_x.view()),
            KernelArg::Buffer(dst.view()),
        ],
    );
    drain(gpu);

    let want: Vec<f32> = (0..F16_OUT)
        .map(|r| crate::oracle::dot(&weights[r * F16_IN..(r + 1) * F16_IN], &x))
        .collect();
    close_at(&gpu.read_f32(&dst), &want, TOL_F32, "f32 decode matvec");
}

/// The half vocabulary gather agrees with the table it read.
///
/// The first launch of the first prefill for a family whose table is not
/// `q6_K`, and the entry that was missing on this backend until `qwen35moe`
/// asked for it. An id outside the table leaves its row untouched, which is
/// the guard a fixture with only valid ids cannot see - so one id here is past
/// the end and its row is asserted to still hold what was poisoned into it.
pub fn the_f16_gather_matches_the_table(gpu: &Gpu) {
    const VOCAB: usize = 9;
    // Deliberately not a multiple of the gather's 256-element span: the last
    // span of a row is short and its bound is `n_embd`.
    const EMBD: usize = 300;
    const TOKENS: usize = 4;

    let raw = crate::oracle::signal(VOCAB * EMBD, 108);
    let (table, values) = to_f16_and_back(&raw);
    let d_table = gpu.upload(&table);
    // The third id is past the end of the table.
    let ids: [i32; TOKENS] = [7, 0, i32::try_from(VOCAB).expect("a small table"), 3];
    let d_ids = gpu.upload(
        &ids.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let poison = f32::from_bits(0xA5A5_A5A5);
    let dst = gpu.upload(&f32_bytes(&vec![poison; TOKENS * EMBD]));

    let args = GetRowsArgs {
        n_embd: EMBD as i32,
        n_vocab: VOCAB as i32,
        n_tokens: TOKENS as i32,
        pad0: 0,
        src_row_bytes: (EMBD * 2) as u64,
        dst_row_bytes: (EMBD * 4) as u64,
        token_stride: 4,
    };
    launch(
        gpu,
        "generic",
        "kernel_get_rows_f16_f32",
        Extent3::new((EMBD as u32).div_ceil(256), TOKENS as u32, 1),
        Extent3::new(256, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_table.view()),
            KernelArg::Buffer(d_ids.view()),
            KernelArg::Buffer(dst.view()),
        ],
    );
    drain(gpu);

    let got = gpu.read_f32(&dst);
    for (slot, id) in ids.iter().enumerate() {
        let row = &got[slot * EMBD..(slot + 1) * EMBD];
        if *id < 0 || *id as usize >= VOCAB {
            assert!(
                row.iter().all(|v| v.to_bits() == poison.to_bits()),
                "token {slot} names row {id}, which is not in the table, and its \
                 destination was written anyway"
            );
            continue;
        }
        let want = &values[*id as usize * EMBD..(*id as usize + 1) * EMBD];
        // Bit for bit: a gather widens and copies, with no arithmetic in it at
        // all, so a tolerance here would only hide a row read at the wrong
        // offset by a whole element.
        for (i, (a, b)) in row.iter().zip(want).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "token {slot} (row {id}), element {i}: {a} against {b}"
            );
        }
    }
}

// -- the q8_0 products ------------------------------------------------------

/// Input width of the `q8_0` fixtures: a whole number of the GEMM's k tile,
/// of the codec's own 32-element block, and of the matvec's eight-lane run.
const Q8_IN: usize = 256;
/// Output rows, deliberately not a multiple of the GEMM's 64-row tile nor of
/// the matvec's two-row block, so both guarded epilogues run.
const Q8_OUT: usize = 70;

/// The `q8_0` prefill GEMM agrees with a triple loop over the same numbers.
///
/// This entry is not optional for this family: a `Q4_K_M` qwen35 container keeps
/// `ssm_alpha` and `ssm_beta` at `q8_0`, 24 of each, and `launch::mm_name` hands
/// every chunk of two tokens or more to this name.
pub fn the_q8_0_prefill_gemm_matches_a_triple_loop(gpu: &Gpu) {
    let n_tok = 40usize;
    let (blocks, weights) = q8_matrix(Q8_OUT, Q8_IN);
    let d_w = gpu.upload(bytemuck::cast_slice(&blocks));
    // The activations are narrowed to `half` by the oracle as well as by the
    // kernel, for the reason the f16 case gives: the template stages both tiles
    // as `half` and a reference that summed f32 activations would be comparing
    // a different product.
    let x_raw = crate::oracle::signal(n_tok * Q8_IN, 105);
    let (_, x) = to_f16_and_back(&x_raw);
    let d_x = gpu.upload(&f32_bytes(&x_raw));
    let dst = zeros(gpu, n_tok * Q8_OUT);

    let per_row = Q8_IN / QK8_0;
    let row = (per_row * size_of::<BlockQ80>()) as u64;
    let plane = row * Q8_OUT as u64;
    let act_row = (Q8_IN * 4) as u64;
    let args = MulMmArgs {
        ne00: Q8_IN as i32,
        ne02: 1,
        nb01: row,
        nb02: plane,
        nb03: plane,
        ne12: 1,
        pad0: 0,
        nb10: 4,
        nb11: act_row,
        nb12: act_row * n_tok as u64,
        nb13: act_row * n_tok as u64,
        ne0: Q8_OUT as i32,
        ne1: n_tok as i32,
        r2: 1,
        r3: 1,
        pad2: [0; 4],
    };
    launch(
        gpu,
        "qwen35",
        "kernel_mul_mm_q8_0_f32",
        Extent3::new(
            (n_tok as u32).div_ceil(GEMM_TILE_TOKENS),
            (Q8_OUT as u32).div_ceil(GEMM_TILE_ROWS),
            1,
        ),
        Extent3::new(GEMM_BLOCK, 1, 1),
        GEMM_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(d_x.view()),
            KernelArg::Buffer(dst.view()),
        ],
    );
    drain(gpu);

    let mut want = vec![0.0f32; n_tok * Q8_OUT];
    for t in 0..n_tok {
        for r in 0..Q8_OUT {
            want[t * Q8_OUT + r] = crate::oracle::dot(
                &weights[r * Q8_IN..(r + 1) * Q8_IN],
                &x[t * Q8_IN..(t + 1) * Q8_IN],
            );
        }
    }
    close_at(&gpu.read_f32(&dst), &want, TOL_QUANT, "q8_0 prefill gemm");
}

/// The `q8_0` decode matvec agrees with the same dot product.
///
/// `ext_name_for` returns `None` for this codec on a backend that specializes
/// by name, so on CUDA this entry is the whole of how a `q8_0` weight is read
/// below two tokens - not a fallback from an extended instantiation that does
/// not exist.
pub fn the_q8_0_decode_matvec_matches_the_oracle_dot(gpu: &Gpu) {
    let (blocks, weights) = q8_matrix(Q8_OUT, Q8_IN);
    let d_w = gpu.upload(bytemuck::cast_slice(&blocks));
    let x = crate::oracle::signal(Q8_IN, 106);
    let d_x = gpu.upload(&f32_bytes(&x));
    let dst = zeros(gpu, Q8_OUT);

    let per_row = Q8_IN / QK8_0;
    let row = (per_row * size_of::<BlockQ80>()) as u64;
    let plane = row * Q8_OUT as u64;
    let act_row = (Q8_IN * 4) as u64;
    let args = MulMvArgs {
        ne00: Q8_IN as i32,
        ne01: Q8_OUT as i32,
        ne02: 1,
        pad0: [0; 4],
        nb00: 0,
        nb01: row,
        nb02: plane,
        nb03: plane,
        ne10: Q8_IN as i32,
        ne11: 1,
        ne12: 1,
        pad1: [0; 4],
        nb10: 4,
        nb11: act_row,
        nb12: act_row,
        nb13: act_row,
        ne0: Q8_OUT as i32,
        ne1: 1,
        nr0: PLAIN_ROWS as i32,
        r2: 1,
        r3: 1,
    };
    launch(
        gpu,
        "generic",
        "kernel_mul_mv_q8_0_f32",
        Extent3::new((Q8_OUT as u32).div_ceil(PLAIN_ROWS), 1, 1),
        Extent3::new(WARP, Q8_MV_NSG, 1),
        PLAIN_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(d_x.view()),
            KernelArg::Buffer(dst.view()),
        ],
    );
    drain(gpu);

    let want: Vec<f32> = (0..Q8_OUT)
        .map(|r| crate::oracle::dot(&weights[r * Q8_IN..(r + 1) * Q8_IN], &x))
        .collect();
    close_at(&gpu.read_f32(&dst), &want, TOL_F32, "q8_0 decode matvec");
}
