//! The resident row cache, and the three passes one fetch makes over it.
//!
//! Rows are Zipf distributed over natural text, so a cache well short of the
//! whole table still absorbs most lookups. The index is open addressed and the
//! replacement policy is CLOCK: a second-chance sweep costs one bit per slot
//! and avoids the whole-arena victim scans a bigger-granularity cache can
//! afford but this one cannot - there are millions of slots.
//!
//! # Why a fetch is three passes and not one loop
//!
//! A batch routinely asks for the same row twice, and the two must not become
//! two reads. So the first pass claims a slot for every miss and puts it in the
//! index *before* any read is issued: a later entry for the same row then finds
//! that slot, sees it is still in flight, and defers its copy instead of
//! queueing the row again. The second pass reads. The third serves the deferred
//! copies, which is the only point at which the claimed slots hold anything.
//!
//! # Why the reads land in a staging buffer first
//!
//! The reference points each reader straight at its destination - a cache slot
//! or the caller's output - which is one copy cheaper and asks the reader
//! threads to write into the same arena the cache is indexing. Here the batch
//! reads into a contiguous staging buffer and the arena is filled from it
//! afterwards, on the one thread that holds the cache. That buys the whole
//! parallel reader path freedom from aliasing at a cost of one extra copy of a
//! hundred-odd bytes per miss, which is four orders of magnitude below the
//! device round trip it overlaps.

use std::time::Duration;

use crate::handle::Counters;
use crate::{PleError, PleResult};

/// The index entry that means "nothing here".
///
/// A sentinel inside a private table rather than an `Option<u32>`: the table is
/// two entries per slot and millions of slots, and the niche `Option` would
/// need does not exist for `u32`.
const EMPTY: u32 = u32::MAX;

/// Where a batch's rows come from once the cache has decided it needs them.
///
/// The cache never opens a file. It hands a reader the identifiers it missed
/// and a buffer to fill, and the reader decides whether that means a ring, a
/// pool of threads, or one blocking read after another.
pub(crate) trait RowSource {
    /// Fills `staging` with one row per identifier, in order.
    ///
    /// # Errors
    ///
    /// Whatever the underlying reader reports. A partial fill is still an
    /// error: the cache unwinds its claims rather than publishing slots whose
    /// contents are unknown.
    fn read_rows(&mut self, rows: &[u64], staging: &mut [u8], stats: &Counters) -> PleResult<()>;
}

/// A fixed set of row slots, an index over them, and one batch's scratch.
#[derive(Debug)]
pub(crate) struct RowCache {
    row_bytes: usize,
    /// Slot storage, `n_slots * row_bytes`.
    arena: Vec<u8>,
    /// Which row is resident in each slot.
    slot_row: Vec<u64>,
    /// The CLOCK second-chance bit.
    slot_ref: Vec<bool>,
    /// Slots holding a read this batch has issued but not completed. They are
    /// already in the index so repeats coalesce onto them, which means they
    /// must not be evicted or read from until the batch lands.
    slot_pending: Vec<bool>,
    n_slots: u32,
    /// Slots handed out before the first eviction, so a cold cache costs no
    /// sweeping.
    next_free: u32,
    /// The CLOCK hand.
    hand: u32,
    /// Row identifier to slot, power-of-two sized and at most half full.
    index: Vec<u32>,
    index_mask: u32,

    /// Per-batch scratch, kept between fetches so a steady stream of
    /// same-sized batches allocates once.
    miss_row: Vec<u64>,
    miss_slot: Vec<u32>,
    pend: Vec<u32>,
    staging: Vec<u8>,
}

impl RowCache {
    /// Sizes a cache to fit `cache_bytes`, counting the index against it.
    ///
    /// The per-slot cost is the row itself plus its identifier, its CLOCK bit
    /// and two index entries - two because the table is kept at most half full,
    /// which is what keeps a linear probe short. A budget too small for one
    /// slot gives a cache of none, which is the documented way to turn caching
    /// off.
    pub(crate) fn new(row_bytes: u32, n_rows: u64, cache_bytes: u64) -> Self {
        let per_slot = u64::from(row_bytes) + 8 + 1 + 2 * 4;
        let mut slots = cache_bytes / per_slot;
        slots = slots.min(n_rows).min(u64::from(u32::MAX / 4));
        let n_slots = u32::try_from(slots).unwrap_or(0);

        let (index, index_mask) = if n_slots == 0 {
            (Vec::new(), 0)
        } else {
            let mut cap = 1u32;
            while cap < n_slots.saturating_mul(2) {
                cap = cap.saturating_mul(2);
            }
            (vec![EMPTY; cap as usize], cap - 1)
        };
        let stride = row_bytes as usize;
        Self {
            row_bytes: stride,
            arena: vec![0; stride * n_slots as usize],
            slot_row: vec![0; n_slots as usize],
            slot_ref: vec![false; n_slots as usize],
            slot_pending: vec![false; n_slots as usize],
            n_slots,
            next_free: 0,
            hand: 0,
            index,
            index_mask,
            miss_row: Vec::new(),
            miss_slot: Vec::new(),
            pend: Vec::new(),
            staging: Vec::new(),
        }
    }

    /// Slots the cache holds.
    pub(crate) const fn slots(&self) -> u64 {
        self.n_slots as u64
    }

    /// Serves `rows` into `out`, reading whatever is not resident.
    ///
    /// # Errors
    ///
    /// [`PleError::BufferTooSmall`] when `out` cannot hold the rows,
    /// [`PleError::RowOutOfRange`] for an identifier past the table, and
    /// whatever the reader reports. Any of them leaves the cache exactly as it
    /// was: slots claimed for this batch are dropped from the index, so a later
    /// fetch reads the row again rather than trusting a cell that was never
    /// filled.
    pub(crate) fn fetch(
        &mut self,
        rows: &[u64],
        out: &mut [u8],
        n_rows: u64,
        src: &mut dyn RowSource,
        stats: &Counters,
    ) -> PleResult<()> {
        let want = rows.len() as u64 * self.row_bytes as u64;
        if (out.len() as u64) < want {
            return Err(PleError::BufferTooSmall {
                what: "fetched row output",
                want,
                got: out.len() as u64,
            });
        }
        self.miss_row.clear();
        self.miss_slot.clear();
        self.pend.clear();
        self.pend.resize(rows.len(), EMPTY);

        if let Err(err) = self.classify(rows, out, n_rows, stats) {
            self.unwind();
            return Err(err);
        }

        let n_miss = self.miss_row.len();
        self.staging.clear();
        self.staging.resize(n_miss * self.row_bytes, 0);
        if n_miss > 0
            && let Err(err) = src.read_rows(&self.miss_row, &mut self.staging, stats)
        {
            self.unwind();
            return Err(err);
        }

        self.publish(out);
        Ok(())
    }

    /// Pass one: serve hits, claim a slot and queue a read for every miss.
    fn classify(
        &mut self,
        rows: &[u64],
        out: &mut [u8],
        n_rows: u64,
        stats: &Counters,
    ) -> PleResult<()> {
        let stride = self.row_bytes;
        for (i, &row) in rows.iter().enumerate() {
            if row >= n_rows {
                return Err(PleError::RowOutOfRange { row, n_rows });
            }
            stats.lookups.bump(1);

            if self.n_slots == 0 {
                // No cache: every entry is its own read, straight into the
                // caller's buffer by way of the staging chunk it owns.
                stats.misses.bump(1);
                self.miss_row.push(row);
                self.miss_slot.push(EMPTY);
                continue;
            }

            if let Some(slot) = self.find(row) {
                self.slot_ref[slot as usize] = true;
                if self.slot_pending[slot as usize] {
                    self.pend[i] = slot;
                } else {
                    let at = slot as usize * stride;
                    out[i * stride..(i + 1) * stride].copy_from_slice(&self.arena[at..at + stride]);
                }
                stats.hits.bump(1);
                continue;
            }

            stats.misses.bump(1);
            let Some(victim) = self.claim(stats) else {
                return Err(PleError::CacheTooSmall {
                    distinct: self.miss_row.len() as u64 + 1,
                    slots: u64::from(self.n_slots),
                });
            };
            self.slot_row[victim as usize] = row;
            self.slot_ref[victim as usize] = true;
            self.slot_pending[victim as usize] = true;
            self.insert(row, victim);
            self.miss_row.push(row);
            self.miss_slot.push(victim);
            self.pend[i] = victim;
        }
        Ok(())
    }

    /// Pass three: the reads have landed, so the claims can be published and
    /// the deferred copies served.
    fn publish(&mut self, out: &mut [u8]) {
        let stride = self.row_bytes;
        if self.n_slots == 0 {
            // Uncached: miss `j` is entry `j`, because nothing was deduplicated.
            out[..self.staging.len()].copy_from_slice(&self.staging);
            return;
        }
        for (j, &slot) in self.miss_slot.iter().enumerate() {
            let at = slot as usize * stride;
            self.arena[at..at + stride]
                .copy_from_slice(&self.staging[j * stride..(j + 1) * stride]);
            self.slot_pending[slot as usize] = false;
        }
        for (i, &slot) in self.pend.iter().enumerate() {
            if slot == EMPTY {
                continue;
            }
            let at = slot as usize * stride;
            out[i * stride..(i + 1) * stride].copy_from_slice(&self.arena[at..at + stride]);
        }
    }

    /// Drops every slot this batch claimed.
    ///
    /// A failed batch leaves claimed slots holding nothing, and they are
    /// already in the index. Leaving them there would make the next fetch of
    /// those rows a hit on an empty cell - the one failure mode a cache must
    /// never have.
    fn unwind(&mut self) {
        for j in 0..self.miss_slot.len() {
            let slot = self.miss_slot[j];
            if slot == EMPTY {
                continue;
            }
            self.slot_pending[slot as usize] = false;
            self.remove(self.slot_row[slot as usize]);
        }
        self.miss_row.clear();
        self.miss_slot.clear();
    }

    /// The first index bucket a row probes.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the mask is a u32, so a value masked with it fits one exactly"
    )]
    fn bucket(&self, row: u64) -> u32 {
        (mix64(row) & u64::from(self.index_mask)) as u32
    }

    /// The slot holding `row`, if any.
    fn find(&self, row: u64) -> Option<u32> {
        let mut i = self.bucket(row);
        loop {
            let slot = self.index[i as usize];
            if slot == EMPTY {
                return None;
            }
            if self.slot_row[slot as usize] == row {
                return Some(slot);
            }
            i = (i + 1) & self.index_mask;
        }
    }

    /// Puts `slot` in the index under `row`.
    fn insert(&mut self, row: u64, slot: u32) {
        let mut i = self.bucket(row);
        while self.index[i as usize] != EMPTY {
            i = (i + 1) & self.index_mask;
        }
        self.index[i as usize] = slot;
    }

    /// Takes `row` out of the index, closing the probe chain behind it.
    ///
    /// Backward-shift deletion rather than a tombstone: linear probing cannot
    /// tombstone without eventually filling the table with them, and this cache
    /// evicts continuously, so the tombstones would never stop accumulating.
    fn remove(&mut self, row: u64) {
        let mut i = self.bucket(row);
        loop {
            let slot = self.index[i as usize];
            if slot == EMPTY {
                return;
            }
            if self.slot_row[slot as usize] == row {
                break;
            }
            i = (i + 1) & self.index_mask;
        }
        self.index[i as usize] = EMPTY;
        let mut j = (i + 1) & self.index_mask;
        while self.index[j as usize] != EMPTY {
            let moved = self.index[j as usize];
            self.index[j as usize] = EMPTY;
            self.insert(self.slot_row[moved as usize], moved);
            j = (j + 1) & self.index_mask;
        }
    }

    /// The next slot to hand out, evicting if the arena is full.
    ///
    /// `None` when no slot can be had: every one of them was claimed earlier in
    /// this same batch and has a read in flight into it. The hand is bounded
    /// rather than unbounded because that case is reachable - a fetch of more
    /// distinct rows than the cache has slots - and an unbounded clock hand
    /// over an all-pending array does not evict, does not fail and does not
    /// return. It spins, at one whole processor, having issued no I/O, which is
    /// the least diagnosable shape a wrong answer can take.
    fn claim(&mut self, stats: &Counters) -> Option<u32> {
        if self.next_free < self.n_slots {
            let slot = self.next_free;
            self.next_free += 1;
            return Some(slot);
        }
        // Two passes over the wheel at most: the first clears a reference bit
        // and the second evicts the slot it cleared, which is the whole of what
        // the clock can do. Anything still unclaimable after that is pending,
        // and pending does not clear until this batch's reads land.
        for _ in 0..u64::from(self.n_slots) * 2 {
            let slot = self.hand;
            self.hand = (self.hand + 1) % self.n_slots;
            // A slot claimed earlier in this same batch has a read in flight
            // into it; handing it out twice would make both entries wrong.
            if self.slot_pending[slot as usize] {
                continue;
            }
            if self.slot_ref[slot as usize] {
                self.slot_ref[slot as usize] = false;
                continue;
            }
            self.remove(self.slot_row[slot as usize]);
            stats.evictions.bump(1);
            return Some(slot);
        }
        None
    }
}

/// Spreads a row identifier over the whole index.
///
/// The identifiers are already a hash, but they are a hash modulo a per-head
/// prime plus a per-head offset, which leaves them clustered by head. Probing
/// an open-addressed table with those directly would pile every head's rows
/// into its own stretch of the table.
const fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

/// One reader's contribution to the shared counters.
///
/// Accumulated per worker and merged once, so the atomics are touched a handful
/// of times per batch instead of once per row.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Tally {
    /// Positional reads issued, after coalescing.
    pub(crate) reads: u64,
    /// Bytes actually asked of the device, block padding included.
    pub(crate) bytes: u64,
    /// Time those reads took.
    pub(crate) elapsed: Duration,
}

impl Tally {
    /// Folds another reader's numbers in.
    pub(crate) fn merge(&mut self, other: Self) {
        self.reads += other.reads;
        self.bytes += other.bytes;
        self.elapsed += other.elapsed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source that answers every row with a byte pattern derived from its
    /// identifier, and counts what it was asked for.
    #[derive(Debug, Default)]
    struct Synthetic {
        asked: Vec<u64>,
    }

    fn pattern(row: u64, j: usize) -> u8 {
        u8::try_from((row.wrapping_mul(31) + j as u64 * 7 + 11) & 0xFF).unwrap_or(0)
    }

    impl RowSource for Synthetic {
        fn read_rows(
            &mut self,
            rows: &[u64],
            staging: &mut [u8],
            stats: &Counters,
        ) -> PleResult<()> {
            let stride = staging.len() / rows.len().max(1);
            for (j, &row) in rows.iter().enumerate() {
                for k in 0..stride {
                    staging[j * stride + k] = pattern(row, k);
                }
                self.asked.push(row);
            }
            stats.reads.bump(rows.len() as u64);
            Ok(())
        }
    }

    /// A source that refuses, to prove the unwind.
    #[derive(Debug)]
    struct Broken;

    impl RowSource for Broken {
        fn read_rows(&mut self, _: &[u64], _: &mut [u8], _: &Counters) -> PleResult<()> {
            Err(PleError::ShortRead {
                row: 0,
                got: 0,
                want: 8,
            })
        }
    }

    fn expect_rows(out: &[u8], rows: &[u64], stride: usize) {
        for (i, &row) in rows.iter().enumerate() {
            for k in 0..stride {
                assert_eq!(out[i * stride + k], pattern(row, k), "row {row} byte {k}");
            }
        }
    }

    #[test]
    fn a_cache_budget_buys_slots_after_the_index_is_paid_for() {
        // Eight rows of eight bytes would be sixty-four bytes of payload, but a
        // slot also carries its identifier, its CLOCK bit and two index cells.
        let cache = RowCache::new(8, 1_000, 8 * (8 + 8 + 1 + 8));
        assert_eq!(cache.slots(), 8);
        // A budget under one slot is a cache of none, which is how caching is
        // turned off.
        assert_eq!(RowCache::new(8, 1_000, 10).slots(), 0);
        // Never more slots than the table has rows.
        assert_eq!(RowCache::new(8, 3, 1 << 20).slots(), 3);
    }

    #[test]
    fn a_repeated_row_inside_one_batch_becomes_one_read() {
        let mut cache = RowCache::new(4, 1_000, 1 << 16);
        let stats = Counters::default();
        let mut src = Synthetic::default();
        let rows = [7, 9, 7, 7, 9, 12];
        let mut out = vec![0u8; rows.len() * 4];
        cache
            .fetch(&rows, &mut out, 1_000, &mut src, &stats)
            .expect("the synthetic source always answers");
        assert_eq!(src.asked, vec![7, 9, 12], "repeats coalesce onto the claim");
        expect_rows(&out, &rows, 4);
    }

    #[test]
    fn a_second_pass_over_the_same_rows_is_served_without_reading() {
        let mut cache = RowCache::new(4, 1_000, 1 << 16);
        let stats = Counters::default();
        let mut src = Synthetic::default();
        let rows = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut out = vec![0u8; rows.len() * 4];
        cache
            .fetch(&rows, &mut out, 1_000, &mut src, &stats)
            .expect("first pass");
        let after_first = src.asked.len();
        cache
            .fetch(&rows, &mut out, 1_000, &mut src, &stats)
            .expect("second pass");
        assert_eq!(src.asked.len(), after_first, "nothing was read twice");
        assert!(stats.snapshot().hits >= rows.len() as u64);
        expect_rows(&out, &rows, 4);
    }

    #[test]
    fn a_cache_of_no_slots_reads_every_entry_every_time() {
        let mut cache = RowCache::new(4, 1_000, 0);
        let stats = Counters::default();
        let mut src = Synthetic::default();
        let rows = [3, 3, 4];
        let mut out = vec![0u8; rows.len() * 4];
        for _ in 0..2 {
            cache
                .fetch(&rows, &mut out, 1_000, &mut src, &stats)
                .expect("uncached fetch");
        }
        assert_eq!(src.asked.len(), 6, "no coalescing and no residency");
        assert_eq!(stats.snapshot().hits, 0);
        expect_rows(&out, &rows, 4);
    }

    #[test]
    fn the_clock_hand_skips_a_referenced_slot_once_before_taking_one() {
        // Two slots, both referenced, then a third row arrives. The sweep
        // clears both bits on its way past and takes the first slot it comes
        // back to - so the row behind the hand survives this eviction, which is
        // the second chance. A policy without the bit would have taken it.
        let mut cache = RowCache::new(4, 1_000, 2 * (4 + 8 + 1 + 8));
        assert_eq!(cache.slots(), 2);
        let stats = Counters::default();
        let mut src = Synthetic::default();
        let mut out = vec![0u8; 4];
        for row in [1u64, 2] {
            cache
                .fetch(&[row], &mut out, 1_000, &mut src, &stats)
                .expect("fetch");
        }
        cache
            .fetch(&[3], &mut out, 1_000, &mut src, &stats)
            .expect("fetch");
        assert_eq!(stats.snapshot().evictions, 1, "exactly one row made room");

        let before = src.asked.len();
        cache
            .fetch(&[2], &mut out, 1_000, &mut src, &stats)
            .expect("fetch");
        assert_eq!(src.asked.len(), before, "row 2 was skipped, not taken");
        expect_rows(&out, &[2], 4);

        // Row 1 is the one that went, so it has to be read again.
        cache
            .fetch(&[1], &mut out, 1_000, &mut src, &stats)
            .expect("fetch");
        assert_eq!(src.asked.len(), before + 1, "row 1 was the victim");
    }

    #[test]
    fn backward_shift_deletion_keeps_a_displaced_row_findable() {
        // Small table, many rows: every row is evicted at least once, so the
        // probe chains are rebuilt constantly. If deletion broke a chain, some
        // resident row would stop being found and the read count would climb.
        let mut cache = RowCache::new(4, 4_096, 32 * (4 + 8 + 1 + 8));
        let stats = Counters::default();
        let mut src = Synthetic::default();
        let mut out = vec![0u8; 4];
        for round in 0..64u64 {
            for row in 0..32u64 {
                cache
                    .fetch(&[row + round % 3], &mut out, 4_096, &mut src, &stats)
                    .expect("fetch");
                expect_rows(&out, &[row + round % 3], 4);
            }
        }
        let stats = stats.snapshot();
        assert!(stats.hits > stats.misses, "{stats:?} should mostly hit");
    }

    #[test]
    fn a_row_past_the_end_of_the_table_is_refused_by_name() {
        let mut cache = RowCache::new(4, 10, 1 << 16);
        let stats = Counters::default();
        let mut src = Synthetic::default();
        let mut out = vec![0u8; 8];
        let err = cache
            .fetch(&[3, 11], &mut out, 10, &mut src, &stats)
            .expect_err("row 11 is past the table");
        assert!(
            matches!(
                err,
                PleError::RowOutOfRange {
                    row: 11,
                    n_rows: 10
                }
            ),
            "{err:?}"
        );
        assert!(src.asked.is_empty(), "the batch never ran");
    }

    #[test]
    fn a_failed_batch_leaves_no_slot_claiming_a_row_it_never_read() {
        let mut cache = RowCache::new(4, 1_000, 1 << 16);
        let stats = Counters::default();
        let mut out = vec![0u8; 8];
        cache
            .fetch(&[5, 6], &mut out, 1_000, &mut Broken, &stats)
            .expect_err("the source refuses");
        // The same rows through a working source must read, not hit.
        let mut src = Synthetic::default();
        cache
            .fetch(&[5, 6], &mut out, 1_000, &mut src, &stats)
            .expect("second source answers");
        assert_eq!(src.asked, vec![5, 6], "the failed claims were dropped");
        expect_rows(&out, &[5, 6], 4);
    }

    #[test]
    fn a_short_output_buffer_is_refused_rather_than_partly_filled() {
        let mut cache = RowCache::new(4, 1_000, 1 << 16);
        let stats = Counters::default();
        let mut out = vec![0u8; 4];
        let err = cache
            .fetch(&[1, 2], &mut out, 1_000, &mut Synthetic::default(), &stats)
            .expect_err("two rows need eight bytes");
        assert!(matches!(err, PleError::BufferTooSmall { .. }), "{err:?}");
    }
}
