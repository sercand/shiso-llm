//! The indexer's pooled pass over a per-session raw key ring
//! (`q4e_idx_ring_pool_kernel`), graded bit for bit against the layout it
//! replaced: a raw key row stored for every position (`q4e_idx_store_k_kernel`)
//! and the ported pooled pass reading it (`q4e_idx_pool_kernel`).
//!
//! The reference here is not a host oracle but the old device path itself, run
//! side by side over the same keys, positions and scattered page table: the
//! claim being tested is that removing the raw plane moves no bit of any
//! pooled row, so the comparison is `==` on every pooled row after every pass.
//!
//! Every sequence of passes a session can make is covered, because each one
//! leaves a different set of raw keys live between passes: a prompt cut at
//! arbitrary chunk boundaries, a decode one row at a time, a speculative verify
//! that settles anywhere inside its rows (so later passes overwrite rejected
//! positions with different keys), the draft head's chain and catch-up (which
//! starts behind what it wrote), a resume from a checkpoint's canonical ring,
//! the tap a grid checkpoint takes, and two sessions interleaved one row at a
//! time with their own rings and tables.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::missing_panics_doc,
    reason = "every case asserts by panicking, which is the whole of what a case does; \
              kernel arguments are u32 and the kernels store positions and page ids as i32 \
              while the fixtures index in usize; every value here is a small non-negative \
              position far below 2^24"
)]

use half::f16;
use shiso_core::page::PAGE_TOKENS;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The ported indexer unit, mirroring `kernels::MOD_IDX`.
const MODULE_IDX: &str = "q4e_idx";
/// The ring unit, mirroring `kernels::MOD_IDXRING`.
const MODULE_RING: &str = "q4e_idxring";
/// Positions one pooled block averages over, mirroring `kernels::IDX_R`.
const R: u32 = 4;
/// Width of an indexer key row, mirroring `kernels::IDX_DIM`.
const DIM: usize = 128;
/// Rows of the raw key ring, mirroring `kernels::IDX_RING`.
const RING: u32 = 16;
/// The tap row that asks for no copy, mirroring `kernels::IDX_NO_TAP`.
const NO_TAP: u32 = u32::MAX;
/// Threads in the narrow per-row kernels, mirroring `kernels::BLOCK_NARROW`.
const BLOCK_NARROW: u32 = 128;
/// Rows of the rotary table in front of a pass's first token.
const MROPE_BACK: u32 = 4;
/// Rotated dims of the shipped model.
const N_ROT: u32 = 64;
/// The shipped rotary base.
const ROPE_BASE: f32 = 1.0e7;
/// The shipped multimodal rotary sections.
const SECTIONS: [u32; 3] = [11, 11, 10];
/// The shipped norm epsilon.
const EPS: f32 = 1.0e-6;
/// Pages in the fixture pool.
const POOL_PAGES: u32 = 8;
/// A session's table: no two neighbours adjacent, the pool's last page third.
const TABLE: [i32; 5] = [6, 1, 7, 2, 4];
/// A second session's table, disjoint from the first.
const TABLE_B: [i32; 3] = [3, 0, 5];
/// A sentinel an untouched `f16` slot holds: a NaN no kernel writes.
const F16_SENTINEL: u16 = 0x7E5A;

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_u16(gpu: &Gpu, buffer: &DeviceBuffer) -> Vec<u16> {
    gpu.read(buffer)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|w| u16::from_le_bytes(*w))
        .collect()
}

/// A rotary table row for position `p`: three distinct axis positions.
fn mrope_row(p: i32) -> [i32; 3] {
    [p, p / 2 + 7, p / 3 + 100]
}

/// Rows of the whole pool.
const fn pool_rows() -> usize {
    (POOL_PAGES * PAGE_TOKENS) as usize
}

/// One session under both layouts: the old one's raw plane and the new one's
/// ring, each feeding its own pooled plane, over one page table.
struct Session {
    table: Vec<i32>,
    pages: DeviceBuffer,
    raw: DeviceBuffer,
    ring: DeviceBuffer,
}

/// The pool both layouts write: one pooled plane each, shared by every session
/// (they own disjoint pages), and the norm weight.
struct Pools {
    pooled_old: DeviceBuffer,
    pooled_new: DeviceBuffer,
    w: DeviceBuffer,
}

impl Pools {
    fn new(gpu: &Gpu) -> Self {
        let rows = pool_rows() / R as usize;
        let w: Vec<f32> = signal(DIM, 7).iter().map(|v| 1.0 + 0.5 * v).collect();
        Self {
            pooled_old: gpu.upload(&u16_bytes(&vec![F16_SENTINEL; rows * DIM])),
            pooled_new: gpu.upload(&u16_bytes(&vec![F16_SENTINEL; rows * DIM])),
            w: gpu.upload(&f32_bytes(&w)),
        }
    }

    /// Both pooled planes are the same bytes, every row, written or not.
    fn assert_same(&self, gpu: &Gpu, what: &str) {
        let old = read_u16(gpu, &self.pooled_old);
        let new = read_u16(gpu, &self.pooled_new);
        for (slot, (o, n)) in old.iter().zip(&new).enumerate() {
            assert_eq!(
                o,
                n,
                "{what}: pooled row {} dim {} is {n:#06x} over the ring and {o:#06x} over the \
                 raw plane",
                slot / DIM,
                slot % DIM
            );
        }
    }
}

impl Session {
    fn new(gpu: &Gpu, table: &[i32]) -> Self {
        Self {
            table: table.to_vec(),
            pages: gpu.upload(&i32_bytes(table)),
            raw: gpu.upload(&u16_bytes(&vec![F16_SENTINEL; pool_rows() * DIM])),
            ring: gpu.upload(&u16_bytes(&vec![0u16; RING as usize * DIM])),
        }
    }

    fn kv_row(&self, p: u32) -> usize {
        (self.table[(p / PAGE_TOKENS) as usize] as u32 * PAGE_TOKENS + p % PAGE_TOKENS) as usize
    }

    /// One pass over positions `[start, start + len)` with keys from `seed`,
    /// through both layouts. Returns the tap's buffer when `tap` asks for one.
    fn pass(
        &self,
        gpu: &Gpu,
        pools: &Pools,
        start: u32,
        len: u32,
        seed: u32,
        tap: Option<u32>,
    ) -> Option<Vec<u16>> {
        let positions: Vec<i32> = (start..start + len).map(|p| p as i32).collect();
        let k = signal(len as usize * DIM, seed);
        let mut mrope = Vec::new();
        for m in 0..(MROPE_BACK + len) {
            mrope.extend(mrope_row(start as i32 - MROPE_BACK as i32 + m as i32));
        }
        let k_dev = gpu.upload(&f32_bytes(&k));
        let pos_dev = gpu.upload(&i32_bytes(&positions));
        let mrope_dev = gpu.upload(&i32_bytes(&mrope));
        let grid = len / R + 1;
        // The old layout: every raw key into the plane, then the ported pool.
        run(
            gpu,
            KernelRef::new(MODULE_IDX, "q4e_idx_store_k_kernel"),
            &LaunchArgs {
                grid: Extent3::new(len, 1, 1),
                block: Extent3::new(BLOCK_NARROW, 1, 1),
                smem_bytes: 0,
                args: &[
                    KernelArg::Buffer(self.raw.view()),
                    KernelArg::Buffer(k_dev.view()),
                    KernelArg::Buffer(pos_dev.view()),
                    KernelArg::Buffer(self.pages.view()),
                    KernelArg::U32(len),
                ],
            },
        );
        let mut old_args = vec![
            KernelArg::Buffer(pools.pooled_old.view()),
            KernelArg::Buffer(self.raw.view()),
            KernelArg::Buffer(pools.w.view()),
        ];
        old_args.extend([
            KernelArg::Buffer(pos_dev.view()),
            KernelArg::Buffer(self.pages.view()),
            KernelArg::U32(len),
            KernelArg::U32(N_ROT),
            KernelArg::F32(ROPE_BASE),
            KernelArg::Buffer(mrope_dev.view()),
            KernelArg::U32(SECTIONS[0]),
            KernelArg::U32(SECTIONS[1]),
            KernelArg::U32(SECTIONS[2]),
            KernelArg::F32(EPS),
        ]);
        run(
            gpu,
            KernelRef::new(MODULE_IDX, "q4e_idx_pool_kernel"),
            &LaunchArgs {
                grid: Extent3::new(grid, 1, 1),
                block: Extent3::new(BLOCK_NARROW, 1, 1),
                smem_bytes: 0,
                args: &old_args,
            },
        );
        // The new layout: one launch over the ring and the pass's own keys.
        let snap = gpu.upload(&u16_bytes(&vec![F16_SENTINEL; RING as usize * DIM]));
        let (snap_view, snap_row) = match tap {
            Some(row) => (snap.view(), row),
            None => (self.ring.view(), NO_TAP),
        };
        run(
            gpu,
            KernelRef::new(MODULE_RING, "q4e_idx_ring_pool_kernel"),
            &LaunchArgs {
                grid: Extent3::new(grid, 1, 1),
                block: Extent3::new(BLOCK_NARROW, 1, 1),
                smem_bytes: 0,
                args: &[
                    KernelArg::Buffer(pools.pooled_new.view()),
                    KernelArg::Buffer(self.ring.view()),
                    KernelArg::Buffer(k_dev.view()),
                    KernelArg::Buffer(pools.w.view()),
                    KernelArg::Buffer(pos_dev.view()),
                    KernelArg::Buffer(self.pages.view()),
                    KernelArg::U32(len),
                    KernelArg::U32(N_ROT),
                    KernelArg::F32(ROPE_BASE),
                    KernelArg::Buffer(mrope_dev.view()),
                    KernelArg::U32(SECTIONS[0]),
                    KernelArg::U32(SECTIONS[1]),
                    KernelArg::U32(SECTIONS[2]),
                    KernelArg::F32(EPS),
                    KernelArg::Buffer(snap_view),
                    KernelArg::U32(snap_row),
                ],
            },
        );
        drain(gpu);
        tap.map(|_| read_u16(gpu, &snap))
    }

    /// The ring holds, at each live slot for a pass starting at `next`, exactly
    /// the raw key the plane holds for that position.
    fn assert_live_rows(&self, gpu: &Gpu, next: u32, what: &str) {
        let raw = read_u16(gpu, &self.raw);
        let ring = read_u16(gpu, &self.ring);
        for p in (next - next % R)..next {
            let slot = (p % RING) as usize;
            let row = self.kv_row(p);
            assert_eq!(
                &ring[slot * DIM..(slot + 1) * DIM],
                &raw[row * DIM..(row + 1) * DIM],
                "{what}: ring slot {slot} does not hold position {p}'s raw key"
            );
        }
    }

    /// What a checkpoint at `q` carries and a resume restores: the ring with
    /// every row a resume does not read zeroed. Restored into the same buffer,
    /// which is what a resumed session's ring is: the checkpoint's bytes and
    /// nothing of the session that wrote it.
    fn resume_from_canonical(&self, gpu: &Gpu, q: u32) {
        let mut ring = read_u16(gpu, &self.ring);
        for slot in 0..RING {
            let live = ((q - q % R)..q).any(|p| p % RING == slot);
            if !live {
                ring[slot as usize * DIM..(slot as usize + 1) * DIM].fill(0);
            }
        }
        gpu.device()
            .copy_h2d(self.ring.view(), &u16_bytes(&ring))
            .expect("host-to-device copy");
    }
}

/// Runs `(start, len)` passes in order with fresh keys each, asserting both
/// pooled planes agree after every one and the ring's live rows match the
/// plane before the next.
fn run_passes(
    gpu: &Gpu,
    pools: &Pools,
    session: &Session,
    passes: &[(u32, u32)],
    seed: u32,
    what: &str,
) {
    for (i, (start, len)) in passes.iter().enumerate() {
        session.pass(
            gpu,
            pools,
            *start,
            *len,
            seed.wrapping_add(i as u32 * 7919),
            None,
        );
        pools.assert_same(gpu, &format!("{what}, pass {i} at {start}+{len}"));
        if let Some((next, _)) = passes.get(i + 1) {
            session.assert_live_rows(gpu, *next, &format!("{what}, before pass {}", i + 1));
        }
    }
}

/// Consecutive passes of the given widths from `start`.
fn chunks(start: u32, widths: &[u32]) -> Vec<(u32, u32)> {
    let mut at = start;
    widths
        .iter()
        .map(|w| {
            let pass = (at, *w);
            at += w;
            pass
        })
        .collect()
}

/// A prompt cut at arbitrary chunk boundaries - mid-block, on block
/// boundaries, across page boundaries, one row and a few hundred - pools every
/// block bit for bit as the raw plane did.
pub fn chunked_prefill_at_any_boundary_pools_as_the_raw_plane(gpu: &Gpu) {
    let pools = Pools::new(gpu);
    let session = Session::new(gpu, &TABLE);
    let widths = [1, 2, 3, 5, 7, 13, 64, 250, 3, 300, 1, 1, 1, 17, 255, 256, 9];
    run_passes(
        gpu,
        &pools,
        &session,
        &chunks(0, &widths),
        11,
        "chunked prefill",
    );
}

/// A decode one row at a time across two page boundaries.
pub fn decode_row_by_row_pools_as_the_raw_plane(gpu: &Gpu) {
    let pools = Pools::new(gpu);
    let session = Session::new(gpu, &TABLE);
    let mut passes = chunks(0, &[250]);
    passes.extend((250..530).map(|p| (p, 1)));
    run_passes(gpu, &pools, &session, &passes, 23, "decode");
}

/// A speculative verify of up to nine rows settling anywhere inside them: the
/// next pass starts right after the last kept row and overwrites the rejected
/// positions with different keys. The ring must still hold the kept rows'
/// keys of the block the next pass starts inside.
pub fn verify_rollbacks_pool_as_the_raw_plane(gpu: &Gpu) {
    let pools = Pools::new(gpu);
    let session = Session::new(gpu, &TABLE);
    let mut passes = chunks(0, &[37]);
    let mut frontier = 37u32;
    // (rows, kept) - kept includes the committed token, so 1..=rows.
    for (rows, kept) in [
        (9, 1),
        (9, 9),
        (5, 2),
        (9, 4),
        (3, 3),
        (9, 7),
        (2, 1),
        (9, 5),
        (9, 1),
        (6, 6),
        (9, 3),
        (1, 1),
        (9, 8),
    ] {
        passes.push((frontier, rows));
        frontier += kept;
    }
    passes.push((frontier, 1));
    run_passes(gpu, &pools, &session, &passes, 31, "verify rollback");
}

/// The draft head's pattern: a chain of one-row passes running ahead of the
/// frontier from the position before it, then a catch-up pass rewriting the
/// committed rows from the frontier - a pass that starts up to eight
/// positions behind the highest one written.
pub fn draft_chain_and_catch_up_pool_as_the_raw_plane(gpu: &Gpu) {
    let pools = Pools::new(gpu);
    let session = Session::new(gpu, &TABLE);
    // The head has written up to 298 after a 300-token prompt's warm-up.
    let mut passes = vec![(0, 299)];
    let mut frontier = 300u32;
    for (budget, committed) in [
        (8, 3),
        (8, 9),
        (4, 1),
        (8, 5),
        (2, 2),
        (8, 1),
        (8, 8),
        (6, 4),
    ] {
        // Chain: positions frontier - 1 .. frontier + budget - 2.
        for i in 0..budget {
            passes.push((frontier - 1 + i, 1));
        }
        // Catch-up over the committed rows bar the last: frontier ..
        // frontier + committed - 2.
        if committed > 1 {
            passes.push((frontier, committed - 1));
        }
        frontier += committed;
    }
    run_passes(gpu, &pools, &session, &passes, 41, "draft chain");
}

/// A checkpoint mid-block, restored from its canonical ring - every row a
/// resume does not read zeroed - continues bit for bit: with a decode, a
/// verify and a prompt chunk.
pub fn resume_from_a_canonical_ring_pools_as_the_raw_plane(gpu: &Gpu) {
    for (q, rest) in [
        (515u32, vec![(515u32, 1u32), (516, 1), (517, 3)]),
        (514, vec![(514, 9), (515, 1)]),
        (513, vec![(513, 300)]),
        (512, vec![(512, 2), (514, 5)]),
    ] {
        let pools = Pools::new(gpu);
        let session = Session::new(gpu, &TABLE);
        // Something that leaves stale rows all over the ring first: a verify
        // that ran past q and was rejected back to it.
        session.pass(gpu, &pools, 0, q - 3, 51, None);
        session.pass(gpu, &pools, q - 3, 3, 52, None);
        session.pass(gpu, &pools, q, 9, 53, None);
        pools.assert_same(gpu, "before the checkpoint");
        session.resume_from_canonical(gpu, q);
        session.assert_live_rows(gpu, q, "the restored ring");
        run_passes(gpu, &pools, &session, &rest, 57, &format!("resume at {q}"));
    }
}

/// The tap copies the raw keys of the block holding row `row`'s position, up
/// to it, into the snapshot section at their ring slots - including keys from
/// before the pass - and nothing else.
pub fn the_tap_copies_the_frontier_blocks_raw_keys(gpu: &Gpu) {
    let pools = Pools::new(gpu);
    let session = Session::new(gpu, &TABLE);
    session.pass(gpu, &pools, 0, 62, 61, None);
    // Rows of a pass from 62: row 1 is position 63 (keys 60, 61, 62 from the
    // ring and the pass), row 3 is position 65 (keys 64 from the pass), row 2
    // is position 64 (a block boundary: nothing).
    for (row, want) in [
        (1u32, vec![60u32, 61, 62]),
        (3, vec![64]),
        (2, vec![]),
        (0, vec![60, 61]),
    ] {
        // Eight rows, so each round starts within the ring's reach of the
        // last: the rounds rewind to 62 as a verify would.
        let snap = session
            .pass(gpu, &pools, 62, 8, 62 + row, Some(row))
            .expect("the tap was asked for");
        let raw = read_u16(gpu, &session.raw);
        for slot in 0..RING {
            let got = &snap[slot as usize * DIM..(slot as usize + 1) * DIM];
            match want.iter().find(|p| *p % RING == slot) {
                Some(p) => {
                    let r = session.kv_row(*p);
                    assert_eq!(
                        got,
                        &raw[r * DIM..(r + 1) * DIM],
                        "tap row {row}, position {p}"
                    );
                }
                None => assert!(
                    got.iter().all(|v| *v == F16_SENTINEL),
                    "tap row {row} wrote slot {slot}, which no live position holds"
                ),
            }
        }
        pools.assert_same(gpu, &format!("the tapped pass at row {row}"));
    }
}

/// Two sessions decoding one row each, interleaved, each over its own ring and
/// table: neither reads the other's raw keys. What a batch's per-row launches
/// are.
pub fn interleaved_sessions_keep_their_own_rings(gpu: &Gpu) {
    let pools = Pools::new(gpu);
    let a = Session::new(gpu, &TABLE);
    let b = Session::new(gpu, &TABLE_B);
    a.pass(gpu, &pools, 0, 253, 71, None);
    b.pass(gpu, &pools, 0, 130, 72, None);
    for step in 0..40u32 {
        a.pass(gpu, &pools, 253 + step, 1, 1_000 + step, None);
        b.pass(gpu, &pools, 130 + step, 1, 2_000 + step, None);
        pools.assert_same(gpu, &format!("interleaved step {step}"));
    }
    a.assert_live_rows(gpu, 293, "session a");
    b.assert_live_rows(gpu, 170, "session b");
    // And the ring is the pass's raw keys rounded as the plane rounds them.
    let ring = read_u16(gpu, &a.ring);
    let k = signal(DIM, 1_000 + 39);
    let slot = (292 % RING) as usize;
    for i in 0..DIM {
        assert_eq!(
            ring[slot * DIM + i],
            f16::from_f32(k[i]).to_bits(),
            "dim {i}"
        );
    }
}
