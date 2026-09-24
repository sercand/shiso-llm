//! The indexer's storing and selecting entries, graded against a host
//! restatement of each kernel body.
//!
//! [`super::indexer`] covers the block scorers' page-table walk; this file
//! covers the rest of the tower: the raw key store, the pooled key pass, the
//! query norm, a scorer launch that spans more than one query tile, the top-k
//! selection and the expansion back into positions.
//!
//! Every case that touches the cache runs over a page table the pool handed out
//! of order, because a kernel that treats a session's pages as contiguous only
//! goes wrong once the pool has recycled pages - never on a fresh process.
//!
//! The selection runs over three thousand candidate blocks against the shipped
//! keep count of 512, so it actually selects: a context under the budget keeps
//! every visible block, and every selection kernel answers that correctly.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::missing_panics_doc,
    reason = "every case asserts by panicking, which is the whole of what a case does; \
              kernel arguments are u32 and the kernels store positions, page ids and block \
              indices as i32 while the fixtures index in usize; every value here is a small \
              non-negative literal or a position far below 2^24"
)]

use half::f16;
use shiso_core::page::PAGE_TOKENS;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The module the indexer entries live in, mirroring `kernels::MOD_IDX`.
const MODULE: &str = "q4e_idx";
/// Positions one pooled block stands for, mirroring `kernels::IDX_R`.
const R: u32 = 4;
/// Width of an indexer key row, mirroring `kernels::IDX_DIM`.
const DIM: usize = 128;
/// The shipped indexer's query heads.
const HEADS: u32 = 4;
/// Threads in the narrow per-row kernels, mirroring `kernels::BLOCK_NARROW`.
const BLOCK_NARROW: u32 = 128;
/// Threads in the scorer and the expansion, mirroring `kernels::BLOCK_LINEAR`.
const BLOCK_LINEAR: u32 = 256;
/// Threads in the selection, mirroring `kernels::BLOCK_WIDE`.
const BLOCK_WIDE: u32 = 1_024;
/// Query rows one scorer block covers, mirroring `kernels::IDX_QT`.
const QT: u32 = 16;
/// Key blocks one scalar scorer block covers, mirroring `kernels::IDX_BT`.
const BT: u32 = 64;
/// Blocks the selection keeps: the shipped 2,048-position budget over `R`.
const KEEP: u32 = 2_048 / R;
/// Slots one expanded selection row holds, mirroring `kernels::IDX_WIDTH`.
const WIDTH: u32 = 2_048 + 3;
/// Rows of the rotary table in front of a chunk's first token, mirroring
/// `inputs::MROPE_BACK`.
const MROPE_BACK: u32 = 4;
/// Rotated dims of the shipped model.
const N_ROT: u32 = 64;
/// The shipped rotary base.
const ROPE_BASE: f32 = 1.0e7;
/// The shipped multimodal rotary sections: time, height, width.
const SECTIONS: [u32; 3] = [11, 11, 10];
/// The shipped norm epsilon.
const EPS: f32 = 1.0e-6;

/// Pages in the fixture pool.
const POOL_PAGES: u32 = 8;
/// The session's page table: no two neighbours physically adjacent, and the
/// third entry the pool's last page.
const TABLE: [i32; 5] = [6, 1, 7, 2, 4];
/// A sentinel an untouched `f16` slot holds: a NaN no kernel writes.
const F16_SENTINEL: u16 = 0x7E5A;

/// `f16` values as little-endian bytes.
fn f16_bytes(values: &[f16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `u16` bit patterns as little-endian bytes.
fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `i32` values as little-endian bytes.
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A device buffer read back as `u16` bit patterns.
fn read_u16(gpu: &Gpu, buffer: &DeviceBuffer) -> Vec<u16> {
    gpu.read(buffer)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|w| u16::from_le_bytes(*w))
        .collect()
}

/// A device buffer read back as `i32` values.
fn read_i32(gpu: &Gpu, buffer: &DeviceBuffer) -> Vec<i32> {
    gpu.read(buffer)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| i32::from_le_bytes(*w))
        .collect()
}

/// `q4e_kv_row`: the physical cache row of logical position `p`.
fn kv_row(p: u32) -> u32 {
    TABLE[(p / PAGE_TOKENS) as usize] as u32 * PAGE_TOKENS + p % PAGE_TOKENS
}

/// Rows of the whole pool's raw key plane.
const fn pool_rows() -> u32 {
    POOL_PAGES * PAGE_TOKENS
}

/// Fixture values: multiples of 1/8 in `[-1, 1]`, as `f16`.
fn eighths(len: usize, seed: u32) -> Vec<f16> {
    signal(len, seed)
        .into_iter()
        .map(|v| f16::from_f32((v * 8.0).round() / 8.0))
        .collect()
}

/// A rotary table row for position `p`: three distinct axis positions.
fn mrope_row(p: i32) -> [i32; 3] {
    [p, p / 2 + 7, p / 3 + 100]
}

/// `q4e_rope_mrope` in double precision, except for the angle, which is
/// formed in `f32` as the device forms it: the angle's own rounding at a
/// position in the thousands is a shift the device really makes.
fn rope(v: &mut [f64], at: [i32; 3]) {
    let half = (N_ROT / 2) as usize;
    for i in 0..half {
        let mut axis = i % 3;
        let used = (i / 3) as u32;
        if used >= SECTIONS[axis] {
            axis = 0;
        }
        let p = at[axis] as u32 as f32;
        let theta = f64::from(p * ROPE_BASE.powf(-((2 * i) as f32) / N_ROT as f32));
        let (s, c) = theta.sin_cos();
        let (x0, x1) = (v[i], v[i + half]);
        v[i] = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
}

/// `RMSNorm` with a weight, in double precision.
fn norm(x: &[f64], w: &[f32]) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let scale = 1.0 / (ms + f64::from(EPS)).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, w)| v * scale * f64::from(*w))
        .collect()
}

/// Asserts a device `f16` agrees with the oracle to about one `f16` step.
fn close_f16(got: u16, want: f64, what: &str) {
    let g = f64::from(f16::from_bits(got).to_f32());
    assert!(
        (g - want).abs() <= 4e-3 * want.abs().max(1.0),
        "{what}: got {g}, expected {want}"
    );
}

/// The raw key store writes each row at the page its position names, and
/// nothing else.
pub fn the_key_store_follows_a_scattered_page_table(gpu: &Gpu) {
    // Nine rows across the first page boundary.
    let positions: Vec<i32> = (250..259).collect();
    let n_tok = positions.len() as u32;
    let k = signal(n_tok as usize * DIM, 3);
    let cache = gpu.upload(&u16_bytes(&vec![F16_SENTINEL; pool_rows() as usize * DIM]));
    let k_dev = gpu.upload(&f32_bytes(&k));
    let pos_dev = gpu.upload(&i32_bytes(&positions));
    let pages_dev = gpu.upload(&i32_bytes(&TABLE));
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_idx_store_k_kernel"),
        &LaunchArgs {
            grid: Extent3::new(n_tok, 1, 1),
            block: Extent3::new(BLOCK_NARROW, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(cache.view()),
                KernelArg::Buffer(k_dev.view()),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(pages_dev.view()),
                KernelArg::U32(n_tok),
            ],
        },
    );
    drain(gpu);
    let mut want = vec![F16_SENTINEL; pool_rows() as usize * DIM];
    for (t, &p) in positions.iter().enumerate() {
        let row = kv_row(p as u32) as usize;
        for i in 0..DIM {
            want[row * DIM + i] = f16::from_f32(k[t * DIM + i]).to_bits();
        }
    }
    let got = read_u16(gpu, &cache);
    for (slot, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g,
            w,
            "key store: pool row {} dim {} holds {g:#06x}, expected {w:#06x}",
            slot / DIM,
            slot % DIM
        );
    }
}

/// The pooled pass averages, norms and rotates every block the chunk
/// completes, writes it at its page's pooled row, and leaves the partial block
/// and every other row alone - over a chunk that starts mid-block, so the
/// first block's rotary row reaches back into the table's lookback.
pub fn the_pooled_pass_covers_the_blocks_a_chunk_completes(gpu: &Gpu) {
    // 509 is one past a block start, so block 127 starts at 508 and its rotary
    // row is lookback row 3; 1030 leaves block 257 incomplete.
    let positions: Vec<i32> = (509..=1_030).collect();
    let n_tok = positions.len() as u32;
    let cache = eighths(pool_rows() as usize * DIM, 5);
    let w: Vec<f32> = signal(DIM, 7).iter().map(|v| 1.0 + 0.5 * v).collect();
    let mut mrope = Vec::new();
    for m in 0..(MROPE_BACK + n_tok) {
        mrope.extend(mrope_row(positions[0] - MROPE_BACK as i32 + m as i32));
    }
    let pooled_rows = pool_rows() / R;
    let pooled = gpu.upload(&u16_bytes(&vec![F16_SENTINEL; pooled_rows as usize * DIM]));
    let cache_dev = gpu.upload(&f16_bytes(&cache));
    let w_dev = gpu.upload(&f32_bytes(&w));
    let pos_dev = gpu.upload(&i32_bytes(&positions));
    let pages_dev = gpu.upload(&i32_bytes(&TABLE));
    let mrope_dev = gpu.upload(&i32_bytes(&mrope));
    // `kernels::idx_pool`: a truncating divide plus one, not a ceiling.
    let grid = n_tok / R + 1;
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_idx_pool_kernel"),
        &LaunchArgs {
            grid: Extent3::new(grid, 1, 1),
            block: Extent3::new(BLOCK_NARROW, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(pooled.view()),
                KernelArg::Buffer(cache_dev.view()),
                KernelArg::Buffer(w_dev.view()),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(pages_dev.view()),
                KernelArg::U32(n_tok),
                KernelArg::U32(N_ROT),
                KernelArg::F32(ROPE_BASE),
                KernelArg::Buffer(mrope_dev.view()),
                KernelArg::U32(SECTIONS[0]),
                KernelArg::U32(SECTIONS[1]),
                KernelArg::U32(SECTIONS[2]),
                KernelArg::F32(EPS),
            ],
        },
    );
    drain(gpu);
    let got = read_u16(gpu, &pooled);
    let b0 = positions[0] as u32 / R;
    let b1 = (*positions.last().expect("a chunk has rows") as u32 + 1) / R;
    let mut written = vec![false; pooled_rows as usize];
    for b in b0..b1 {
        let krow = kv_row(b * R) as usize;
        let brow = krow / R as usize;
        written[brow] = true;
        let mean: Vec<f64> = (0..DIM)
            .map(|i| {
                let mut acc = 0.0f32;
                for j in 0..R as usize {
                    acc += cache[(krow + j) * DIM + i].to_f32();
                }
                f64::from(acc * (1.0 / R as f32))
            })
            .collect();
        let mut v = norm(&mean, &w);
        let mrow = (MROPE_BACK as i32 + (b * R) as i32 - positions[0]) as usize;
        rope(
            &mut v,
            [mrope[3 * mrow], mrope[3 * mrow + 1], mrope[3 * mrow + 2]],
        );
        for i in 0..DIM {
            close_f16(
                got[brow * DIM + i],
                v[i],
                &format!("pooled block {b} (pool row {brow}) dim {i}"),
            );
        }
    }
    for (row, done) in written.iter().enumerate() {
        if !done {
            assert!(
                got[row * DIM..(row + 1) * DIM]
                    .iter()
                    .all(|&g| g == F16_SENTINEL),
                "pooled row {row} belongs to no block this chunk completes and was written"
            );
        }
    }
}

/// The query pass norms each (row, head) and rotates it at the row's own
/// rotary row.
pub fn the_query_pass_norms_and_rotates_every_head(gpu: &Gpu) {
    let positions: [i32; 5] = [2_047, 2_048, 9_001, 13_337, 20_000];
    let n_tok = positions.len() as u32;
    let q = signal(n_tok as usize * HEADS as usize * DIM, 13);
    let w: Vec<f32> = signal(DIM, 17).iter().map(|v| 1.0 + 0.5 * v).collect();
    // The engine hands the kernel the table past its lookback prefix, so the
    // buffer carries the prefix and the launch binds a slice past it.
    let mut mrope = vec![-1i32; 3 * MROPE_BACK as usize];
    for &p in &positions {
        mrope.extend(mrope_row(p));
    }
    let qn = gpu.upload(&u16_bytes(&vec![F16_SENTINEL; q.len()]));
    let q_dev = gpu.upload(&f32_bytes(&q));
    let w_dev = gpu.upload(&f32_bytes(&w));
    let mrope_dev = gpu.upload(&i32_bytes(&mrope));
    let prefix = u64::from(MROPE_BACK) * 3 * 4;
    let table = mrope_dev
        .slice(prefix, mrope_dev.len() - prefix)
        .expect("the table extends past its prefix");
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_idx_q_kernel"),
        &LaunchArgs {
            grid: Extent3::new(HEADS, n_tok, 1),
            block: Extent3::new(BLOCK_NARROW, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(qn.view()),
                KernelArg::Buffer(q_dev.view()),
                KernelArg::Buffer(w_dev.view()),
                KernelArg::Buffer(table),
                KernelArg::U32(SECTIONS[0]),
                KernelArg::U32(SECTIONS[1]),
                KernelArg::U32(SECTIONS[2]),
                KernelArg::U32(HEADS),
                KernelArg::U32(N_ROT),
                KernelArg::F32(ROPE_BASE),
                KernelArg::F32(EPS),
            ],
        },
    );
    drain(gpu);
    let got = read_u16(gpu, &qn);
    for (t, &p) in positions.iter().enumerate() {
        for h in 0..HEADS as usize {
            let base = (t * HEADS as usize + h) * DIM;
            let x: Vec<f64> = q[base..base + DIM].iter().map(|&v| f64::from(v)).collect();
            let mut v = norm(&x, &w);
            rope(&mut v, mrope_row(p));
            for i in 0..DIM {
                close_f16(
                    got[base + i],
                    v[i],
                    &format!("query row {t} head {h} dim {i}"),
                );
            }
        }
    }
}

/// The scalar scorer over two query tiles, the second one partial, and six
/// key tiles each behind its own page lookup.
///
/// Values are multiples of 1/8, so every sum is exact and the scores agree to
/// the bit whatever order either side sums in.
pub fn the_scalar_scorer_covers_a_partial_second_query_tile(gpu: &Gpu) {
    // Twenty rows at the end of five pages: the second query tile holds four.
    let positions: Vec<i32> = (1_260..1_280).collect();
    let n_tok = positions.len() as u32;
    let ctx = PAGE_TOKENS * TABLE.len() as u32;
    let stride = ctx / R + 1;
    let qn = eighths(n_tok as usize * HEADS as usize * DIM, 19);
    let pooled = eighths((pool_rows() / R) as usize * DIM, 23);
    let qn_dev = gpu.upload(&f16_bytes(&qn));
    let pooled_dev = gpu.upload(&f16_bytes(&pooled));
    let pos_dev = gpu.upload(&i32_bytes(&positions));
    let pages_dev = gpu.upload(&i32_bytes(&TABLE));
    let score = gpu.upload(&f32_bytes(&vec![f32::NAN; (n_tok * stride) as usize]));
    let last = pos_dev
        .slice(u64::from(n_tok - 1) * 4, 4)
        .expect("the last position is inside the position row");
    // `kernels::idx_score_smem`.
    let smem = (QT * HEADS + BT) * DIM as u32 * 2;
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_idx_score_kernel"),
        &LaunchArgs {
            grid: Extent3::new(stride.div_ceil(BT), n_tok.div_ceil(QT), 1),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: smem,
            args: &[
                KernelArg::Buffer(score.view()),
                KernelArg::Buffer(qn_dev.view()),
                KernelArg::Buffer(pooled_dev.view()),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(pages_dev.view()),
                KernelArg::U32(HEADS),
                KernelArg::U32(n_tok),
                KernelArg::U32(stride),
                KernelArg::Buffer(last),
            ],
        },
    );
    drain(gpu);
    let got = gpu.read_f32(&score);
    let n_blocks = (*positions.last().expect("rows") as u32 + 1) / R;
    for (t, &p) in positions.iter().enumerate() {
        for b in 0..n_blocks {
            let want = if b * R + R - 1 > p as u32 {
                f32::NEG_INFINITY
            } else {
                let row = (kv_row(b * R) / R) as usize;
                let key = &pooled[row * DIM..(row + 1) * DIM];
                (0..HEADS as usize)
                    .map(|h| {
                        let base = (t * HEADS as usize + h) * DIM;
                        let dot: f32 = qn[base..base + DIM]
                            .iter()
                            .zip(key)
                            .map(|(a, k)| a.to_f32() * k.to_f32())
                            .sum();
                        dot.max(0.0)
                    })
                    .sum()
            };
            let g = got[t * stride as usize + b as usize];
            assert!(
                g.to_bits() == want.to_bits(),
                "scorer: row {t} (position {p}) block {b} scored {g}, expected {want}"
            );
        }
    }
}

/// The key of a float's bit pattern the selection orders by.
fn key_of(f: f32) -> u32 {
    let u = f.to_bits();
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

/// The selection a row should produce: every visible block above the `k`-th
/// largest key, then the ties at that key lowest index first, in index order.
fn select(row: &[f32], k: u32) -> Vec<i32> {
    let neg_inf = key_of(f32::NEG_INFINITY);
    let mut keys: Vec<u32> = row
        .iter()
        .map(|&v| key_of(v))
        .filter(|&u| u != neg_inf)
        .collect();
    if keys.len() < k as usize {
        return row
            .iter()
            .enumerate()
            .filter(|&(_, &v)| key_of(v) != neg_inf)
            .map(|(i, _)| i as i32)
            .collect();
    }
    keys.sort_unstable_by(|a, b| b.cmp(a));
    let threshold = keys[k as usize - 1];
    let above = keys.iter().filter(|&&u| u > threshold).count();
    let mut ties = k as usize - above;
    let mut out = Vec::new();
    for (i, &v) in row.iter().enumerate() {
        let u = key_of(v);
        if u == neg_inf {
            continue;
        }
        if u > threshold {
            out.push(i as i32);
        } else if u == threshold && ties > 0 {
            ties -= 1;
            out.push(i as i32);
        }
    }
    out
}

/// The selection keeps the best 512 of three thousand blocks per row: with
/// heavy ties at the threshold, with distinct scores of both signs, and with
/// fewer visible blocks than it keeps.
pub fn the_selection_keeps_the_best_blocks_in_index_order(gpu: &Gpu) {
    const N_BLOCKS: u32 = 3_000;
    const ROWS: usize = 3;
    // The score rows are strided by the whole context, past the blocks the
    // last position completes; the slack is filled with scores that would win
    // if the kernel ranked it.
    let ctx = 12_288u32;
    let stride = (ctx / R + 1) as usize;
    let pos_last = [(N_BLOCKS * R - 1) as i32];
    let mut score = vec![1.0e30f32; ROWS * stride];
    // Row 0: sixteenths, so a few dozen distinct values over three thousand
    // blocks and a threshold bin full of ties; every seventh block invisible.
    for (i, v) in signal(N_BLOCKS as usize, 31).into_iter().enumerate() {
        score[i] = if i % 7 == 3 {
            f32::NEG_INFINITY
        } else {
            (v * 16.0).round() / 16.0
        };
    }
    // Row 1: distinct scores of both signs.
    for (i, v) in signal(N_BLOCKS as usize, 37).into_iter().enumerate() {
        score[stride + i] = v * 5.0;
    }
    // Row 2: three hundred visible blocks, so everything visible is kept.
    for (i, v) in signal(N_BLOCKS as usize, 41).into_iter().enumerate() {
        score[2 * stride + i] = if i % 10 == 0 { v } else { f32::NEG_INFINITY };
    }
    let score_dev = gpu.upload(&f32_bytes(&score));
    let sel = gpu.upload(&i32_bytes(&vec![-7; ROWS * KEEP as usize]));
    let cnt = gpu.upload(&i32_bytes(&[-7; ROWS]));
    let last = gpu.upload(&i32_bytes(&pos_last));
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_idx_topk_kernel"),
        &LaunchArgs {
            grid: Extent3::new(ROWS as u32, 1, 1),
            block: Extent3::new(BLOCK_WIDE, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(sel.view()),
                KernelArg::Buffer(cnt.view()),
                KernelArg::Buffer(score_dev.view()),
                KernelArg::U32(stride as u32),
                KernelArg::Buffer(last.view()),
                KernelArg::U32(KEEP),
            ],
        },
    );
    drain(gpu);
    let got_sel = read_i32(gpu, &sel);
    let got_cnt = read_i32(gpu, &cnt);
    for t in 0..ROWS {
        let row = &score[t * stride..t * stride + N_BLOCKS as usize];
        let want = select(row, KEEP);
        let n = want.len();
        assert_eq!(got_cnt[t] as usize, n, "selection row {t}: count");
        let got = &got_sel[t * KEEP as usize..(t + 1) * KEEP as usize];
        assert_eq!(&got[..n], &want[..], "selection row {t}: blocks");
        assert!(
            got[n..].iter().all(|&g| g == -7),
            "selection row {t}: wrote past its count"
        );
    }
    // The fixture has to exercise what it claims to.
    assert_eq!(got_cnt[0] as u32, KEEP, "row 0 selects");
    assert_eq!(got_cnt[2], 300, "row 2 keeps everything visible");
}

/// The expansion turns each kept block into its four positions and appends
/// the incomplete block's tail, for every tail length, and a full selection
/// with a three-position tail fills the row's width exactly.
pub fn the_expansion_appends_every_tail_length(gpu: &Gpu) {
    // Tails of 3, 0, 1 and 2 positions.
    let positions: [i32; 4] = [8_194, 8_195, 9_000, 12_001];
    let counts: [i32; 4] = [KEEP as i32, 0, 37, 200];
    let rows = positions.len();
    let mut sel = vec![-5i32; rows * KEEP as usize];
    for t in 0..rows {
        let complete = (positions[t] as u32 + 1) / R;
        // An ascending spread of complete blocks.
        for j in 0..counts[t] as u32 {
            sel[t * KEEP as usize + j as usize] = (j * complete / KEEP) as i32 + (j % 2) as i32;
        }
    }
    let tokens = gpu.upload(&i32_bytes(&vec![-9; rows * WIDTH as usize]));
    let n_sel = gpu.upload(&i32_bytes(&[-9; 4]));
    let sel_dev = gpu.upload(&i32_bytes(&sel));
    let cnt_dev = gpu.upload(&i32_bytes(&counts));
    let pos_dev = gpu.upload(&i32_bytes(&positions));
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_idx_expand_kernel"),
        &LaunchArgs {
            grid: Extent3::new(rows as u32, 1, 1),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(tokens.view()),
                KernelArg::Buffer(n_sel.view()),
                KernelArg::Buffer(sel_dev.view()),
                KernelArg::Buffer(cnt_dev.view()),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::U32(KEEP),
                KernelArg::U32(WIDTH),
            ],
        },
    );
    drain(gpu);
    let got = read_i32(gpu, &tokens);
    let got_n = read_i32(gpu, &n_sel);
    for t in 0..rows {
        let mut want = vec![-9i32; WIDTH as usize];
        let c = counts[t] as usize;
        for i in 0..c * R as usize {
            want[i] = sel[t * KEEP as usize + i / R as usize] * R as i32 + (i % R as usize) as i32;
        }
        let p = positions[t] as u32;
        let tail0 = (p + 1) / R * R;
        let n_tail = p + 1 - tail0;
        for j in 0..n_tail {
            want[c * R as usize + j as usize] = (tail0 + j) as i32;
        }
        assert_eq!(
            got_n[t],
            c as i32 * R as i32 + n_tail as i32,
            "expansion row {t}: count"
        );
        let row = &got[t * WIDTH as usize..(t + 1) * WIDTH as usize];
        for (i, (&g, &w)) in row.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "expansion row {t} slot {i}");
        }
    }
    assert_eq!(got_n[0] as u32, WIDTH, "a full selection fills the width");
}
