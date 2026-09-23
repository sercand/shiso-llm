//! The indexer's two block scorers, over a page table the pool handed out of
//! order.
//!
//! Both scorers stage a tile of pooled key rows behind one page-table lookup,
//! which is only right while the tile lies inside one page. The scalar tile is
//! exactly one page wide. The tensor-core tile was twice that, so its second
//! half read the page physically after the first rather than the one the table
//! names - and when the first was the pool's last page, it read past the end of
//! the plane. A fresh process hands a session its pages in ascending order, so
//! "physically next" and "logically next" coincide and every single-request run
//! answered correctly; a replay that had been through sixteen long sessions
//! faulted in the scorer. The table here is scattered on purpose, and its third
//! slot is the pool's last page, so either failure shows.
//!
//! The fixture values are multiples of 1/8 in `[-1, 1]`, so every product is a
//! multiple of 1/64 and every sum over 128 dims and four heads is exact in
//! `f32`. The oracle and both kernels therefore agree to the bit, whatever
//! order each one sums in, and the tolerance is only there to name the row.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "kernel arguments are u32 while the fixtures index in usize and store \
              positions and page ids as the kernel's i32; every value here is a small \
              non-negative literal"
)]

use half::f16;
use shiso_core::page::PAGE_TOKENS;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The module both scorers live in, mirroring `shiso_engine::sparse::kernels::MOD_IDX`.
const MODULE: &str = "q4e_idx";
/// Positions one pooled block stands for, mirroring `kernels::IDX_R`.
const R: u32 = 4;
/// Width of an indexer key row, mirroring `kernels::IDX_DIM`.
const DIM: usize = 128;
/// Heads the tensor-core scorer is written for, mirroring `kernels::IDX_TC_HEADS`.
const HEADS: u32 = 4;
/// Query rows one scorer block covers, mirroring `kernels::IDX_QT`.
const QT: u32 = 16;
/// Key blocks one scalar scorer block covers, mirroring `kernels::IDX_BT`.
const SCALAR_BT: u32 = 64;
/// Key blocks one tensor-core scorer block covers, mirroring `kernels::IDX_TC_BT`.
const TC_BT: u32 = 128;
/// Threads per scorer block, mirroring `kernels::BLOCK_LINEAR`.
const BLOCK: u32 = 256;

/// Pages in the fixture pool.
const POOL_PAGES: u32 = 8;
/// The session's page table: logical page `i` lives in pool page `TABLE[i]`.
///
/// No two neighbours are physically adjacent, and the third entry is the last
/// page of the pool, so a tile that runs off the end of its first page either
/// reads the wrong rows or reads past the plane.
const TABLE: [i32; 4] = [6, 1, 7, 2];
/// The query rows' positions: the last three of the four pages, so the last
/// block is visible to one row only and the one before it to two.
const POSITIONS: [i32; 3] = [1_021, 1_022, 1_023];
/// The context the score rows are strided for; `ctx / R + 1` blocks.
const CTX: u32 = PAGE_TOKENS * TABLE.len() as u32;

/// Pooled rows one page holds.
const fn rows_per_page() -> u32 {
    PAGE_TOKENS / R
}

/// Score slots per query row, mirroring `IndexerShape::score_blocks`.
const fn stride() -> u32 {
    CTX / R + 1
}

/// Fixture values: multiples of 1/8 in `[-1, 1]`, as `f16` bits.
fn eighths(len: usize, seed: u32) -> Vec<f16> {
    signal(len, seed)
        .into_iter()
        .map(|v| f16::from_f32((v * 8.0).round() / 8.0))
        .collect()
}

/// `f16` values as little-endian bytes.
fn f16_bytes(values: &[f16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `i32` values as little-endian bytes.
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The inputs both scorers read, on the device and on the host.
struct Fixture {
    /// Normed queries, `[t][h][DIM]`.
    qn: Vec<f16>,
    /// The whole pool's pooled plane, `[pool row][DIM]`.
    pooled: Vec<f16>,
    qn_dev: DeviceBuffer,
    pooled_dev: DeviceBuffer,
    pos_dev: DeviceBuffer,
    pages_dev: DeviceBuffer,
}

impl Fixture {
    fn new(gpu: &Gpu) -> Self {
        let n_tok = POSITIONS.len();
        let qn = eighths(n_tok * HEADS as usize * DIM, 11);
        // Sized to the pool exactly, so a read past the last page is a read
        // past the allocation rather than into slack.
        let pooled = eighths((POOL_PAGES * rows_per_page()) as usize * DIM, 29);
        Self {
            qn_dev: gpu.upload(&f16_bytes(&qn)),
            pooled_dev: gpu.upload(&f16_bytes(&pooled)),
            pos_dev: gpu.upload(&i32_bytes(&POSITIONS)),
            pages_dev: gpu.upload(&i32_bytes(&TABLE)),
            qn,
            pooled,
        }
    }

    /// `score[t][b]`: the sum over heads of `relu(<q_th, k_b>)` for a block the
    /// row has fully seen, `-inf` for one it has not.
    fn oracle(&self) -> Vec<Vec<f32>> {
        let n_blocks = (POSITIONS[POSITIONS.len() - 1] as u32 + 1) / R;
        POSITIONS
            .iter()
            .enumerate()
            .map(|(t, &pos)| {
                (0..n_blocks)
                    .map(|b| {
                        if b * R + R - 1 > pos as u32 {
                            return f32::NEG_INFINITY;
                        }
                        let page = TABLE[(b / rows_per_page()) as usize] as u32;
                        let row = (page * rows_per_page() + b % rows_per_page()) as usize;
                        let key = &self.pooled[row * DIM..(row + 1) * DIM];
                        (0..HEADS as usize)
                            .map(|h| {
                                let base = (t * HEADS as usize + h) * DIM;
                                let q = &self.qn[base..base + DIM];
                                let dot: f32 = q
                                    .iter()
                                    .zip(key)
                                    .map(|(a, k)| a.to_f32() * k.to_f32())
                                    .sum();
                                dot.max(0.0)
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}

/// Reads the device's score rows back, trimmed to the blocks the oracle covers.
fn read(gpu: &Gpu, score: &DeviceBuffer, n_blocks: usize) -> Vec<Vec<f32>> {
    let all = gpu.read_f32(score);
    all.chunks(stride() as usize)
        .map(|row| row[..n_blocks].to_vec())
        .collect()
}

/// Asserts two score blocks agree, naming the first row and block that does
/// not.
fn agree(got: &[Vec<f32>], want: &[Vec<f32>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: rows");
    for (t, (g, w)) in got.iter().zip(want).enumerate() {
        for (b, (&gv, &wv)) in g.iter().zip(w).enumerate() {
            let page = b as u32 / rows_per_page();
            // The only infinite score is an invisible block's `-inf`.
            let same = if wv.is_infinite() {
                gv.is_infinite() && gv.is_sign_negative()
            } else {
                (gv - wv).abs() <= 1e-4 * wv.abs().max(1.0)
            };
            assert!(
                same,
                "{what}: row {t} block {b} (logical page {page}, pool page {}) scored {gv}, \
                 expected {wv}",
                TABLE[page as usize],
            );
        }
    }
}

/// Launches one scorer entry over the fixture.
fn score(gpu: &Gpu, fixture: &Fixture, entry: &'static str) -> DeviceBuffer {
    let n_tok = POSITIONS.len() as u32;
    let sentinel = vec![f32::NAN; (n_tok * stride()) as usize];
    let score = gpu.upload(&f32_bytes(&sentinel));
    let last = fixture
        .pos_dev
        .slice(u64::from(n_tok - 1) * 4, 4)
        .expect("the last position is inside the position row");
    let row_blocks = n_tok.div_ceil(QT);
    let (tile, smem, head_arg) = if entry == TC_ENTRY {
        (TC_BT, (16 * HEADS + TC_BT) * DIM as u32 * 2, None)
    } else {
        (
            SCALAR_BT,
            (QT * HEADS + SCALAR_BT) * DIM as u32 * 2,
            Some(HEADS),
        )
    };
    let mut args = vec![
        KernelArg::Buffer(score.view()),
        KernelArg::Buffer(fixture.qn_dev.view()),
        KernelArg::Buffer(fixture.pooled_dev.view()),
        KernelArg::Buffer(fixture.pos_dev.view()),
        KernelArg::Buffer(fixture.pages_dev.view()),
    ];
    if let Some(heads) = head_arg {
        args.push(KernelArg::U32(heads));
    }
    args.extend([
        KernelArg::U32(n_tok),
        KernelArg::U32(stride()),
        KernelArg::Buffer(last),
    ]);
    run(
        gpu,
        KernelRef::new(MODULE, entry),
        &LaunchArgs {
            grid: Extent3::new(stride().div_ceil(tile), row_blocks, 1),
            block: Extent3::new(BLOCK, 1, 1),
            smem_bytes: smem,
            args: &args,
        },
    );
    drain(gpu);
    score
}

/// The tensor-core scorer's entry.
const TC_ENTRY: &str = "q4e_idx_score_tc_kernel";
/// The scalar scorer's entry.
const SCALAR_ENTRY: &str = "q4e_idx_score_kernel";

/// The scalar scorer reads each key block through the page the table names.
pub fn the_scalar_scorer_follows_a_scattered_page_table(gpu: &Gpu) {
    let fixture = Fixture::new(gpu);
    let want = fixture.oracle();
    let got = score(gpu, &fixture, SCALAR_ENTRY);
    agree(&read(gpu, &got, want[0].len()), &want, "scalar scorer");
}

/// The tensor-core scorer reads each key block through the page the table
/// names, including the blocks in the second page a tile covers.
pub fn the_tensor_core_scorer_follows_a_scattered_page_table(gpu: &Gpu) {
    let fixture = Fixture::new(gpu);
    let want = fixture.oracle();
    let got = score(gpu, &fixture, TC_ENTRY);
    agree(&read(gpu, &got, want[0].len()), &want, "tensor-core scorer");
}
