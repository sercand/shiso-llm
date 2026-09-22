//! Checkpoints: the state a recurrent stack cannot rewind.
//!
//! Attention can be rebuilt by replaying the tokens it consumed, so the span tree
//! is enough for it. A gated delta-rule layer cannot be rewound — its state is a
//! summary of everything it has folded in — so returning to a position partway
//! through a sequence needs a snapshot of that state. A checkpoint is everything the
//! recurrent stack cannot reconstruct at a position: the conv windows, the
//! state matrices, the convolution windows, and optionally the prompt-final logits
//! row.
//!
//! # What a checkpoint is worth
//!
//! A checkpoint is worth the prefill between it and the **next nearest checkpoint
//! below**, not from the start of the sequence: recovering to the next nearest
//! checkpoint below and replaying the difference is always available, so the value
//! of a checkpoint is only the work between it and the next nearest one below.
//! That value, in seconds, is divided by the megabytes the slot costs and decayed
//! by how long ago the slot was touched:
//!
//! ```text
//! (pos - lower_ckpt_pos) * prefill_s_per_token
//!     ------------------------------------------------
//!     (ckpt_bytes / MiB) * (1 + age / decay_s)
//! ```
//!
//! # Admission, in order
//!
//! 1. A **speculative** slot — taken because a boundary looked likely to pay off,
//!    not because a rule asked — is always evicted before an earned one.
//! 2. Among earned slots, the least useful goes first.
//! 3. A slot the caller wants kept is priced above the scale in the price table it
//!    hands in, so it never loses a comparison it was not supposed to enter. There is
//!    no separate pinned flag, because a price is the same mechanism with fewer ways to
//!    disagree with itself.
//!
//! Above all three: **a chunk boundary never evicts.** The caller states whether
//! eviction is allowed; when it is not, a full store is a refusal rather than a
//! victim, because the boundary will come around again and a checkpoint bought with
//! a better one is a net loss nobody agreed to.
//!
//! # Where the bytes live
//!
//! A slot holds an opaque host blob the engine snapshots in and copies back out.
//! That costs one device-to-host transfer per admission and buys a store that has
//! never seen a device pointer.

use crate::error::KvError;
use crate::geom::CacheGeometry;
use crate::ids::to_u32;
use crate::ids::{NodeId, SlotId};

/// One slot's contents.
#[derive(Debug)]
struct Slot {
    /// The state blob, or `None` when the slot is free or has been evicted.
    state: Option<Vec<u8>>,
    /// The prompt-final logits row, kept only when it was still fresh.
    logits: Option<Vec<u8>>,
    /// The node this state was taken at, which is what makes a slot findable again.
    node: Option<NodeId>,
    /// Taken on speculation rather than because a rule asked.
    speculative: bool,
    /// Monotonic time of the last touch.
    hit: f64,
}

/// The slot table.
#[derive(Debug)]
pub(crate) struct CkptStore {
    slots: Vec<Slot>,
    geometry: CacheGeometry,
    /// Slots this store may hold. The caller derives it from how much host memory
    /// the engine can spare for checkpoints; it is a ceiling on the table, not on
    /// the bytes, because a slot's blob is allocated when it is first filled.
    cap: u32,
    /// Slots dropped by eviction since the store was built.
    evictions: u64,
}

impl CkptStore {
    /// A store that holds at most `cap` checkpoints.
    ///
    /// Slots are allocated lazily, so `cap` costs nothing until the store is
    /// actually filled; a cache opened with a generous ceiling and a cold tree holds
    /// no blobs at all.
    pub(crate) fn new(geometry: CacheGeometry, cap: u32) -> Self {
        Self {
            slots: Vec::new(),
            geometry,
            cap,
            evictions: 0,
        }
    }

    /// The ceiling this store was opened with.
    #[must_use]
    pub(crate) const fn capacity(&self) -> u32 {
        self.cap
    }

    /// Whether a checkpoint could be admitted without evicting one.
    ///
    /// Either a table slot holds nothing, or the table has room to grow. This is what
    /// gates a stride position in [`crate::next_admission_pos`]: a chunk boundary is
    /// only worth stopping at while a checkpoint is free to take.
    #[must_use]
    pub(crate) fn any_free(&self) -> bool {
        self.slots.len() < usize::try_from(self.cap).unwrap_or(usize::MAX)
            || self.slots.iter().any(|entry| entry.state.is_none())
    }

    /// Slots holding state, which is what a caller compares against capacity.
    pub(crate) fn filled(&self) -> u32 {
        to_u32(self.slots.iter().filter(|s| s.state.is_some()).count())
    }

    /// Slots dropped by eviction.
    pub(crate) fn evictions(&self) -> u64 {
        self.evictions
    }

    /// The reference a caller holds for a slot, or `None` if it holds nothing.
    pub(crate) fn reference(&self, slot: SlotId, pos: u32) -> Option<CkptRef> {
        let entry = self.slots.get(slot.index())?;
        entry.state.as_ref().map(|_| CkptRef {
            slot,
            pos,
            has_logits: entry.logits.is_some(),
        })
    }

    /// Whether a slot holds state at all.
    pub(crate) fn is_live(&self, slot: SlotId) -> bool {
        self.slots
            .get(slot.index())
            .is_some_and(|s| s.state.is_some())
    }

    /// Whether a slot holds speculative state.
    ///
    /// Test-only: the store reads the flag itself when it picks a victim, and a caller
    /// has nothing to do with the answer -- it either allowed eviction or did not.
    #[cfg(test)]
    pub(crate) fn is_speculative(&self, slot: SlotId) -> bool {
        self.slots.get(slot.index()).is_some_and(|s| s.speculative)
    }

    /// Whether a slot's logits row is present alongside its state.
    ///
    /// A checkpoint at the very end of a prompt is only usable with its logits row,
    /// because resuming there means answering the next token from that row.
    #[must_use]
    pub(crate) fn has_logits(&self, slot: SlotId) -> bool {
        self.slots
            .get(slot.index())
            .is_some_and(|s| s.state.is_some() && s.logits.is_some())
    }

    /// The slot's last-touch stamp, which the caller needs in order to price it.
    #[must_use]
    pub(crate) fn hit_of(&self, slot: SlotId) -> f64 {
        self.slots.get(slot.index()).map_or(0.0, |s| s.hit)
    }

    /// Bytes in the slot's logits row, or `None` when it has none.
    #[must_use]
    pub(crate) fn logits_len(&self, slot: SlotId) -> Option<usize> {
        self.slots
            .get(slot.index())
            .and_then(|s| s.logits.as_ref())
            .map(Vec::len)
    }

    /// Every slot that holds state, in table order.
    ///
    /// The caller prices these (see [`Self::alloc`]) because pricing reads the span
    /// tree, and the tree is not this store's to borrow while it is being mutated.
    #[must_use]
    pub(crate) fn priceable(&self) -> Vec<SlotId> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.state.is_some())
            .map(|(index, _)| SlotId::new(to_u32(index)))
            .collect()
    }

    /// The node a slot's state was taken at.
    pub(crate) fn node_of(&self, slot: SlotId) -> Option<NodeId> {
        self.slots.get(slot.index()).and_then(|s| s.node)
    }

    /// Records that a slot was read, resetting its decay clock.
    pub(crate) fn touch(&mut self, slot: SlotId, now: f64) {
        if let Some(entry) = self.slots.get_mut(slot.index()) {
            entry.hit = now;
        }
    }

    /// Copies a slot's state blob into a caller-owned buffer.
    ///
    /// The buffer length is checked rather than resized: the blob is tens of
    /// megabytes and the engine keeps a reusable staging buffer precisely so that a
    /// restore never allocates. An evicted slot is an error, not an empty buffer —
    /// silently handing back zero bytes would resume a sequence from a zeroed state.
    pub(crate) fn state_into(&self, ckpt: CkptRef, buf: &mut [u8]) -> Result<(), KvError> {
        let entry = self
            .slots
            .get(ckpt.slot.index())
            .and_then(|s| s.state.as_ref())
            .ok_or(KvError::StaleCheckpoint { slot: ckpt.slot })?;
        let want = usize::try_from(self.geometry.ckpt_bytes).unwrap_or(usize::MAX);
        if buf.len() != want {
            return Err(KvError::BufferLen {
                want,
                got: buf.len(),
            });
        }
        buf.copy_from_slice(entry);
        Ok(())
    }

    /// Copies a slot's logits row into a caller-owned buffer.
    pub(crate) fn logits_into(&self, ckpt: CkptRef, buf: &mut [u8]) -> Result<(), KvError> {
        let entry = self
            .slots
            .get(ckpt.slot.index())
            .and_then(|s| s.logits.as_ref())
            .ok_or(KvError::StaleCheckpoint { slot: ckpt.slot })?;
        if buf.len() != entry.len() {
            return Err(KvError::BufferLen {
                want: entry.len(),
                got: buf.len(),
            });
        }
        buf.copy_from_slice(entry);
        Ok(())
    }
}

impl CkptStore {
    /// Takes a slot for a new checkpoint, evicting only when allowed to.
    ///
    /// `allow_evict` is the caller's answer to "did a rule ask for this slot?". A
    /// chunk boundary passes `false`, and then a full store is a `SlotRefused`
    /// instead of a victim.
    ///
    /// `prices` is the caller's valuation of each of [`Self::priceable`]'s slots,
    /// which is how the tree's knowledge of lower checkpoints reaches the store
    /// without the store having to know a tree exists -- and why it is a table of
    /// numbers rather than a callback: pricing reads the span tree, and this method
    /// needs the slot table to itself while it is evicting from it.
    ///
    /// A slot missing from `prices` is priced at zero and so becomes the first victim;
    /// a caller that means to protect a slot prices it high rather than leaving it
    /// out.
    ///
    /// The returned [`Admission`] names the node whose checkpoint was displaced, if
    /// there was one, because only the tree can un-point at it.
    pub(crate) fn alloc(
        &mut self,
        now: f64,
        allow_evict: bool,
        speculative: bool,
        prices: &[(SlotId, f64)],
    ) -> Option<Admission> {
        // A slot that has never held state is always the cheapest admission, and an
        // unused table entry is cheaper still than growing the table.
        let mut displaced = None;
        let slot = if let Some(index) = self.slots.iter().position(|e| e.state.is_none()) {
            SlotId::new(to_u32(index))
        } else if self.slots.len() < usize::try_from(self.cap).unwrap_or(usize::MAX) {
            self.slots.push(Slot {
                state: None,
                logits: None,
                node: None,
                speculative: false,
                hit: now,
            });
            SlotId::new(to_u32(self.slots.len() - 1))
        } else if allow_evict {
            // The victim is picked before it is released: `victim` only borrows the
            // slots to price them, and releasing one mutates them.
            let victim = self.victim(prices)?;
            displaced = self.evict(victim);
            victim
        } else {
            // A chunk boundary does not get to take someone else's checkpoint; the
            // caller turns `None` into a `SlotRefused` and retries at the next one.
            return None;
        };
        // The slot is claimed here and written by `fill`. Stamping it here is what
        // makes an abandoned admission look like a taken slot rather than a slot
        // that was never taken, and it is what makes `used` mean "ever taken".
        if let Some(entry) = self.slots.get_mut(slot.index()) {
            entry.hit = now;
            entry.speculative = speculative;
        }
        Some(Admission { slot, displaced })
    }

    /// Stores `state` in `slot` and returns the reference the caller holds.
    ///
    /// The blob must be exactly [`CacheGeometry::ckpt_bytes`] long, which is checked
    /// here rather than at read time: [`Self::state_into`] copies a slot's blob into
    /// a caller buffer of that exact length, so a short blob accepted now becomes a
    /// failed resume many requests later, on a path the caller cannot anticipate.
    ///
    /// The pricing flags are stamped by [`Self::alloc`] when it claims the slot,
    /// so a fill cannot re-price a slot it never admitted.
    ///
    /// # Errors
    ///
    /// [`KvError::BufferLen`] if `state` is not a whole checkpoint blob.
    pub(crate) fn fill(
        &mut self,
        slot: SlotId,
        node: NodeId,
        pos: u32,
        state: &[u8],
        logits: Option<&[u8]>,
        now: f64,
    ) -> Result<CkptRef, KvError> {
        let want = usize::try_from(self.geometry.ckpt_bytes).unwrap_or(usize::MAX);
        if state.len() != want {
            return Err(KvError::BufferLen {
                want,
                got: state.len(),
            });
        }
        let entry = &mut self.slots[slot.index()];
        entry.state = Some(state.to_vec());
        entry.logits = logits.map(<[u8]>::to_vec);
        entry.node = Some(node);
        entry.hit = now;
        Ok(CkptRef {
            slot,
            pos,
            has_logits: logits.is_some(),
        })
    }

    /// Clears a slot and reports whose checkpoint it was.
    ///
    /// The owner comes back because the tree points at the slot too, and a node left
    /// pointing at a cleared slot advertises a checkpoint that is not there. That is
    /// not merely untidy: the eviction scan prices such a node as if it still had
    /// something to lose, and a drop that refuses a node for its stale pointer makes
    /// an eviction loop pick the same victim forever.
    pub(crate) fn evict(&mut self, slot: SlotId) -> Option<NodeId> {
        let entry = self.slots.get_mut(slot.index())?;
        if entry.state.is_some() {
            self.evictions += 1;
        }
        let owner = entry.node;
        entry.state = None;
        entry.logits = None;
        entry.node = None;
        entry.speculative = false;
        owner
    }

    /// The victim eviction would take, or `None` when nothing may be taken.
    ///
    /// Speculative slots are offered first, then earned ones. `prices` is the caller's
    /// valuation of each candidate on the
    /// shared utility scale, which is how the tree's knowledge of neighbouring
    /// checkpoints reaches the store without the store having to know what a tree is.
    pub(crate) fn victim(&self, prices: &[(SlotId, f64)]) -> Option<SlotId> {
        let filled: Vec<SlotId> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.state.is_some())
            .map(|(index, _)| SlotId::new(to_u32(index)))
            .collect();
        if filled.is_empty() {
            return None;
        }
        let speculative: Vec<SlotId> = filled
            .iter()
            .copied()
            .filter(|slot| self.slots[slot.index()].speculative)
            .collect();
        let pool = if speculative.is_empty() {
            &filled
        } else {
            &speculative
        };
        pool.iter().copied().min_by(|a, b| {
            price_of(prices, *a)
                .partial_cmp(&price_of(prices, *b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// One slot's valuation, defaulting to the cheapest possible.
fn price_of(prices: &[(SlotId, f64)], slot: SlotId) -> f64 {
    prices
        .iter()
        .find(|(priced, _)| *priced == slot)
        .map_or(0.0, |(_, utility)| *utility)
}

/// A slot handed to a new checkpoint.
///
/// The displaced node travels with it because evicting a slot cannot clear the
/// tree's pointer at it, and a stale pointer there is what makes an eviction loop
/// spin: the victim scan is deterministic, so the same node is picked again on every
/// pass.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Admission {
    /// The slot now available for a new checkpoint.
    pub(crate) slot: SlotId,
    /// The node whose checkpoint was evicted to make room, if any.
    pub(crate) displaced: Option<NodeId>,
}

/// A held reference to a checkpoint slot: the slot, its position, and what came with it.
/// `pos` is carried so a caller can price a resume without a second lookup, and
/// `has_logits` is carried because a prompt-final checkpoint is worth more than one
/// without the logits row: resuming at that position needs the row the sampler read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CkptRef {
    /// The slot holding the state.
    pub slot: SlotId,
    /// The position the state is valid at.
    pub pos: u32,
    /// Whether the prompt-final logits row came with it.
    pub has_logits: bool,
}

/// Admission is the part of this store a caller can get wrong, so every rule in the
/// module doc gets its own case here. The facade's integration tests cover what a
/// *cache* does with these answers; these cover what the store promises on its own.
#[cfg(test)]
mod tests {
    use super::{CkptStore, KvError};
    use crate::geom::CacheGeometry;
    use crate::ids::{NodeId, SlotId};

    /// A geometry small enough that a test can fill the store with byte vectors.
    fn geometry() -> CacheGeometry {
        CacheGeometry::new(256, 4096, 1024, 0.001, 600.0)
    }

    fn blob() -> Vec<u8> {
        vec![7u8; 1024]
    }

    /// Fills one slot with a distinct owner so a displacement can be recognised.
    fn fill(store: &mut CkptStore, slot: SlotId, node: u32, now: f64) {
        assert!(
            store
                .fill(slot, NodeId::new(node), node * 100, &blob(), None, now)
                .is_ok()
        );
    }

    #[test]
    fn a_slot_is_handed_out_without_evicting_while_the_table_has_room() {
        let mut store = CkptStore::new(geometry(), 2);
        let first = store.alloc(1.0, false, false, &[]).expect("room");
        fill(&mut store, first.slot, 1, 1.0);
        let second = store.alloc(2.0, false, false, &[]).expect("room");
        assert_eq!(second.displaced, None);
        assert_eq!(store.filled(), 1);
        assert_eq!(store.evictions(), 0);
    }

    /// The rule a chunk boundary depends on: a full store is a refusal, not a victim.
    #[test]
    fn a_boundary_that_may_not_evict_gets_nothing_from_a_full_store() {
        let mut store = CkptStore::new(geometry(), 1);
        let taken = store.alloc(1.0, true, false, &[]).expect("room");
        fill(&mut store, taken.slot, 1, 1.0);
        assert!(store.alloc(2.0, false, false, &[]).is_none());
        assert!(store.is_live(taken.slot), "the earned slot survived");
        assert_eq!(store.evictions(), 0);
    }

    #[test]
    fn a_speculative_slot_is_evicted_before_an_earned_one() {
        let mut store = CkptStore::new(geometry(), 2);
        let earned = store.alloc(1.0, true, false, &[]).expect("room");
        fill(&mut store, earned.slot, 1, 1.0);
        let guessed = store.alloc(1.0, true, true, &[]).expect("room");
        fill(&mut store, guessed.slot, 2, 1.0);
        assert!(store.is_speculative(guessed.slot));
        assert!(!store.is_speculative(earned.slot));

        let prices = vec![(earned.slot, 9.0), (guessed.slot, 1.0)];
        let next = store.alloc(5.0, true, false, &prices).expect("a victim");
        assert_eq!(next.slot, guessed.slot, "the speculative one went");
        assert_eq!(next.displaced, Some(NodeId::new(2)));
        assert!(store.is_live(earned.slot), "the earned one stayed");
    }

    /// Among earned slots the price decides, and the displaced owner comes back so the
    /// tree can stop pointing at it.
    #[test]
    fn the_least_useful_earned_slot_goes_when_no_speculative_one_is_left() {
        let mut store = CkptStore::new(geometry(), 2);
        let cheap = store.alloc(1.0, true, false, &[]).expect("room");
        fill(&mut store, cheap.slot, 7, 1.0);
        let dear = store.alloc(1.0, true, false, &[]).expect("room");
        fill(&mut store, dear.slot, 8, 1.0);
        let prices = vec![(cheap.slot, 0.01), (dear.slot, 50.0)];
        let next = store.alloc(9.0, true, false, &prices).expect("a victim");
        assert_eq!(next.slot, cheap.slot);
        assert_eq!(next.displaced, Some(NodeId::new(7)));
        assert!(store.is_live(dear.slot));
    }

    /// The third admission rule: a slot the caller prices above the scale never loses a
    /// comparison it was not supposed to enter.
    #[test]
    fn a_slot_priced_above_the_scale_is_not_the_one_that_goes() {
        let mut store = CkptStore::new(geometry(), 1);
        let kept = store.alloc(1.0, true, false, &[]).expect("room");
        fill(&mut store, kept.slot, 1, 1.0);
        let prices = vec![(kept.slot, f64::MAX)];
        assert_eq!(
            store.victim(&prices),
            Some(kept.slot),
            "it is the only candidate"
        );
        let next = store.alloc(2.0, true, false, &prices).expect("a victim");
        assert_eq!(next.slot, kept.slot);
        assert_eq!(store.evictions(), 1);
    }

    #[test]
    fn a_free_slot_is_reported_while_the_table_can_still_grow() {
        let mut store = CkptStore::new(geometry(), 2);
        assert!(store.any_free());
        let first = store.alloc(1.0, true, false, &[]).expect("room");
        fill(&mut store, first.slot, 1, 1.0);
        assert!(store.any_free(), "the table has one slot left to grow into");
        let second = store.alloc(1.0, true, false, &[]).expect("room");
        fill(&mut store, second.slot, 2, 1.0);
        assert!(!store.any_free(), "the cap is reached and nothing is free");
    }

    #[test]
    fn a_blob_of_the_wrong_size_is_refused_rather_than_stored() {
        let mut store = CkptStore::new(geometry(), 1);
        let slot = store.alloc(1.0, true, false, &[]).expect("room");
        let short = [0u8; 512];
        let error = store
            .fill(slot.slot, NodeId::new(1), 100, &short, None, 1.0)
            .expect_err("a half blob must be refused");
        assert!(matches!(
            error,
            KvError::BufferLen {
                want: 1024,
                got: 512
            }
        ));
        assert!(!store.is_live(slot.slot));
    }

    #[test]
    fn evicting_clears_the_owner_so_it_cannot_be_advertised_again() {
        let mut store = CkptStore::new(geometry(), 1);
        let slot = store.alloc(1.0, true, false, &[]).expect("room");
        fill(&mut store, slot.slot, 3, 1.0);
        assert_eq!(store.node_of(slot.slot), Some(NodeId::new(3)));
        assert_eq!(store.evict(slot.slot), Some(NodeId::new(3)));
        assert_eq!(store.node_of(slot.slot), None);
        assert!(!store.is_live(slot.slot));
        assert_eq!(store.evictions(), 1);
        // A second evict is a no-op rather than a second eviction counted.
        assert_eq!(store.evict(slot.slot), None);
        assert_eq!(store.evictions(), 1);
    }
}
