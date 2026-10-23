//! The `q8_0` vocabulary gather and the two-row extended matvec.
//!
//! Unlike the rest of this module, both cases name their Metal function
//! constants themselves (`KernelRef::specialized`) instead of leaving them to
//! the engine's pipeline cache, so a backend that honours a specialization table
//! can run them and a backend that reads the geometry off the launch can ignore
//! it. The `q8_0` fixture is `product::q8_matrix`: a power-of-two scale, so the
//! dequantized values are exact and the gather is compared bit for bit.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "kernel argument structs take i32/u32/u64 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]

use shiso_core::{Extent3, FnConst, FnConstValue, KernelArg, KernelRef, LaunchArgs};

use super::product::{
    BlockQ80, EXT_NSG, GetRowsArgs, MulMvExtArgs, QK8_0, WARP, launch, q8_matrix,
};
use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::{TOL_F32, dot, gather, signal};

/// Every entry point this file's cases launch.
pub const ENTRIES: [&str; 2] = [
    "kernel_get_rows_q8_0_f32",
    "kernel_mul_mv_ext_q8_0_f32_r1_2",
];

/// Function-constant index of the simdgroup count, mirroring the engine's
/// `EXT_NSG_INDEX`.
const NSG_INDEX: u16 = 600;
/// Function-constant index of the lane split, mirroring `EXT_NXPSG_INDEX`.
const NXPSG_INDEX: u16 = 601;

/// The constant table for one lane split: two simdgroups, `nxpsg` lanes a row.
macro_rules! ext_consts {
    ($nxpsg:expr) => {
        &[
            FnConst {
                index: NSG_INDEX,
                value: FnConstValue::I16(2),
            },
            FnConst {
                index: NXPSG_INDEX,
                value: FnConstValue::I16($nxpsg),
            },
        ]
    };
}

static CONSTS_NX4: [FnConst; 2] = *ext_consts!(4);
static CONSTS_NX8: [FnConst; 2] = *ext_consts!(8);
static CONSTS_NX16: [FnConst; 2] = *ext_consts!(16);

/// The `(input width, lane split, constant table)` triples the matvec walks.
///
/// The engine picks 16 lanes when the width is a multiple of 256 and fewer than
/// three tokens are in flight, so 512 and 1024 would both get 16 from it. The
/// case forces 8 at 512 on purpose, to cover that specialization too; 1024 is the
/// one the engine itself would hand 16.
pub const EXT_SHAPES: [(usize, i16, &[FnConst]); 3] = [
    (576, 4, &CONSTS_NX4),
    (512, 8, &CONSTS_NX8),
    (1024, 16, &CONSTS_NX16),
];

/// Rows in the gather table.
const VOCAB: usize = 9;
/// Width of a gather row: eighteen blocks.
const EMBD: usize = 576;
/// Token ids, with a repeat and both ends of the table.
const IDS: [i32; 4] = [3, 0, 8, 3];

/// The gather dequantizes the `q8_0` rows the ids name, one f32 row per token.
pub fn the_q8_0_gather_dequantizes_the_rows_the_ids_name(gpu: &Gpu) {
    let (blocks, values) = q8_matrix(VOCAB, EMBD);
    let d_table = gpu.upload(bytemuck::cast_slice::<BlockQ80, u8>(&blocks));
    let d_ids = gpu.upload(bytemuck::cast_slice(&IDS));
    // One row more than the ids need, NaN everywhere: a kernel that wrote past
    // its last token, or never wrote a token's row, shows up as a NaN or a
    // changed canary rather than as a plausible zero.
    let dst = gpu.upload(&f32_bytes(&vec![f32::NAN; (IDS.len() + 1) * EMBD]));

    let per_row = EMBD / QK8_0;
    let args = GetRowsArgs {
        n_embd: EMBD as i32,
        n_vocab: VOCAB as i32,
        n_tokens: IDS.len() as i32,
        pad0: 0,
        src_row_bytes: (per_row * size_of::<BlockQ80>()) as u64,
        dst_row_bytes: (EMBD * 4) as u64,
        token_stride: 4,
    };
    launch(
        gpu,
        "generic",
        ENTRIES[0],
        Extent3::new(per_row as u32, IDS.len() as u32, 1),
        Extent3::new(WARP, 1, 1),
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
    let want = gather(&values, EMBD, &IDS);
    let (rows, canary) = got.split_at(IDS.len() * EMBD);
    // Exact: `q * 2^-6` with `|q| <= 127` is representable, so any difference
    // is a wrong byte read, not rounding.
    for (i, (g, w)) in rows.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "q8_0 gather element {i}");
    }
    assert!(
        canary.iter().all(|v| v.is_nan()),
        "q8_0 gather wrote past its last token's row",
    );
}

/// Tokens in the matvec fixture, which is the entry's row blocking.
const N_TOK: usize = 2;
/// Output rows: 70 leaves a short last block at 4, 8 and 16 rows a block
/// (the three lane splits), so the guarded store runs at each.
const OUT: usize = 70;
/// NaN slots after the output, to catch a store past `ne0 * ne1`.
const CANARY: usize = 4;

/// The two-row `q8_0` extended matvec agrees with the oracle dot product at
/// every lane split.
pub fn the_two_row_q8_0_ext_matvec_matches_the_oracle_dot(gpu: &Gpu) {
    for (seed, &(in_dim, nxpsg, consts)) in EXT_SHAPES.iter().enumerate() {
        let (blocks, weights) = q8_matrix(OUT, in_dim);
        let d_w = gpu.upload(bytemuck::cast_slice::<BlockQ80, u8>(&blocks));
        // Activations on a 1/16 grid. Every product `q * 2^-6 * k/16` and every
        // partial sum of up to 1024 of them is then exact in f32 (22 bits at
        // most), so the lane split's summation order cannot move the result and
        // the tolerance is spent on the kernel, not on rounding. Raw `signal`
        // values left a 1024-wide row 1.1e-5 off the oracle with 16 lanes, which
        // is reassociation of the sum rather than a wrong read.
        let x: Vec<f32> = signal(N_TOK * in_dim, 200 + seed as u32)
            .iter()
            .map(|v| (v * 16.0).round() / 16.0)
            .collect();
        let d_x = gpu.upload(&f32_bytes(&x));
        let dst = gpu.upload(&f32_bytes(&vec![f32::NAN; N_TOK * OUT + CANARY]));

        let row = ((in_dim / QK8_0) * size_of::<BlockQ80>()) as u64;
        let plane = row * OUT as u64;
        let act_row = (in_dim * 4) as u64;
        let act_plane = act_row * N_TOK as u64;
        let args = MulMvExtArgs {
            ne00: in_dim as i32,
            ne01: OUT as i32,
            ne02: 1,
            pad0: [0; 4],
            nb00: row,
            nb01: row,
            nb02: plane,
            nb03: plane,
            ne10: in_dim as i32,
            ne11: N_TOK as i32,
            ne12: 1,
            pad1: [0; 4],
            nb10: 4,
            nb11: act_row,
            nb12: act_plane,
            nb13: act_plane,
            ne0: OUT as i32,
            ne1: N_TOK as i32,
            r2: 1,
            r3: 1,
            pad2: [0; 4],
        };
        // Each simdgroup covers `32 / nxpsg` rows, and a block has `EXT_NSG`
        // simdgroups; the two tokens are one `r1_2` group, so y is one.
        let rows_per_block = (32 / nxpsg as u32) * EXT_NSG;
        run(
            gpu,
            KernelRef::specialized("generic", ENTRIES[1], consts),
            &LaunchArgs {
                grid: Extent3::new((OUT as u32).div_ceil(rows_per_block), 1, 1),
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

        let got = gpu.read_f32(&dst);
        for t in 0..N_TOK {
            for r in 0..OUT {
                let want = dot(
                    &weights[r * in_dim..(r + 1) * in_dim],
                    &x[t * in_dim..(t + 1) * in_dim],
                );
                crate::assert_close!(
                    got[t * OUT + r],
                    want,
                    TOL_F32,
                    "q8_0 ext matvec in {in_dim} nxpsg {nxpsg} token {t} row {r}"
                );
            }
        }
        assert!(
            got[N_TOK * OUT..].iter().all(|v| v.is_nan()),
            "q8_0 ext matvec in {in_dim} nxpsg {nxpsg} wrote past its output",
        );
    }
}
