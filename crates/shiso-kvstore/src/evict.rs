//! Eviction: what an entry is worth, and which one goes first.
//!
//! Both policies price an entry the same way: prefill seconds saved per mebibyte,
//! decayed by idle time. That is Marconi's point -- LRU throws away the large
//! recurrent-state entries that are worth the most, precisely because they are large.
//!
//! A page saves the prefill of the positions it holds, so every page is worth the
//! same wherever it is and the page ranking reduces to recency plus how much of the
//! last page a span actually fills. Interior nodes are unreachable here, which is the
//! "leaves before interior" rule: dropping a leaf span is the only drop that cannot
//! orphan anything.
//!
//! A checkpoint saves the prefill from the nearest lower checkpoint on its path to
//! its own position, so one 300 tokens above its neighbour is nearly worthless and a
//! lone one at 20k is worth tens of seconds; ranking checkpoints by recency instead
//! would evict exactly the ones worth keeping.
//!
//! Everything here is a pure function of the tree plus a `now` the caller passes in,
//! so the eviction tests are deterministic and never sleep.

use crate::geom::CacheGeometry;
use crate::ids::NodeId;
use crate::ids::to_u32;
use crate::tree::{SpanTree, TreeNode};

/// What a checkpoint with no recurrent state is priced at, in mebibytes.
///
/// A model with no recurrent state has `ckpt_bytes == 0`, and dividing by that
/// would make every such checkpoint infinitely valuable -- or, at distance zero,
/// a NaN -- so both victim scans, which keep the first candidate when no
/// comparison is true, would stop being an ordering at all. A stateless
/// checkpoint is priced as one MiB so its utility stays finite and its ranking is
/// the prefill it saves; the value only has to be positive and equal across such
/// checkpoints, because every one of them costs the same -- a logits row, which
/// the store does not size.
const STATELESS_CKPT_MIB: f64 = 1.0;

/// Seconds since `hit`, clamped at zero so a clock that went backwards reads as
/// "just touched" rather than as a negative age.
#[must_use]
pub(crate) fn age(hit: f64, now: f64) -> f64 {
    if now > hit { now - hit } else { 0.0 }
}

/// Seconds of prefill one node's pages save per mebibyte, decayed by idle time.
///
/// `pages.max(1)` because a span always holds at least the page its last row sits
/// in, and a node that has not been given pages yet would otherwise divide by zero
/// and rank as infinitely valuable.
#[must_use]
pub(crate) fn node_utility(node: &TreeNode, geometry: &CacheGeometry, now: f64) -> f64 {
    let pages = node.pages.len().max(1);
    let pages = f64::from(to_u32(pages));
    let saved = f64::from(node.end - node.start) * geometry.prefill_s_per_token;
    saved / (pages * geometry.page_mib()) / (1.0 + age(node.hit, now) / geometry.decay_s)
}

/// What a checkpoint is worth: the prefill between the nearest lower checkpoint on
/// its path and its own position, per mebibyte, decayed.
///
/// `distance` is `pos - lower_ckpt_pos`, which is the whole of the ranking: a
/// checkpoint 300 tokens above its neighbour saves 300 tokens of prefill and a lone
/// one at 20k saves 20k tokens' worth, so the lone one survives page pressure that
/// takes the crowded one first.
///
/// The cost term is [`CacheGeometry::ckpt_mib`], except where the model has no
/// recurrent state at all: a zero-byte blob is priced at [`STATELESS_CKPT_MIB`],
/// which leaves the prefill saved as the only thing separating two such
/// checkpoints. The substitution is made only at zero rather than as a floor over
/// every size, so no geometry that does carry a blob is re-priced by it.
#[must_use]
pub(crate) fn ckpt_utility(distance: u32, geometry: &CacheGeometry, hit: f64, now: f64) -> f64 {
    let saved = f64::from(distance) * geometry.prefill_s_per_token;
    let cost = if geometry.ckpt_bytes == 0 {
        STATELESS_CKPT_MIB
    } else {
        geometry.ckpt_mib()
    };
    saved / cost / (1.0 + age(hit, now) / geometry.decay_s)
}

impl SpanTree {
    /// The least useful leaf whose pages nothing is reading: no children, and no
    /// execution context on its path.
    ///
    /// `extra` prices whatever else the caller would lose with the node -- a
    /// checkpoint at its end -- so page pressure cannot silently spend a valuable
    /// one. It is a closure rather than a function pointer because the pricing of a
    /// checkpoint needs the checkpoint store, and in Rust that is a closure over the
    /// checkpoint store rather than a function pointer plus `void *`.
    ///
    /// `extra` is handed the id as well as the node, because pricing what hangs off a
    /// span -- its checkpoint, measured against the next checkpoint below -- means walking
    /// up from it, and the arena is not reachable from a `&TreeNode`.
    ///
    /// Ties keep the earlier node, because the comparison is a strict `<` and not a
    /// utility the walk order decides, and the walk order is the tree's.
    pub(crate) fn pick_victim(
        &self,
        now: f64,
        extra: &dyn Fn(NodeId, &TreeNode) -> f64,
    ) -> Option<NodeId> {
        let geometry = self.geometry();
        let mut best: Option<(NodeId, f64)> = None;
        for id in self.leaves() {
            // Leaves are already childless and never the root; the live test is the
            // other half of "nothing is reading this path".
            if self.holders(id) != 0 {
                continue;
            }
            let node = self.node(id);
            let utility = node_utility(node, geometry, now) + extra(id, node);
            if best.is_none_or(|(_, best_u)| utility < best_u) {
                best = Some((id, utility));
            }
        }
        best.map(|(id, _)| id)
    }
}

/// The next position worth admitting to: the next position worth committing to.
///
/// The stop is the earliest of three candidates ahead of `pos`: a chat-template
/// boundary the caller passed as a `hint`, the branch point where this path stops
/// matching the tree, and -- only while checkpoint slots are going spare -- the next
/// multiple of `stride`. The earliest of the three wins; failing all three, the
/// request itself ends at `len`. Stride positions are prefill chunk boundaries, so
/// they are only worth stopping at while checkpoint slots are going spare; a chunk
/// boundary is never a checkpoint candidate, which is why `spare_slot` gates it.
///
/// `stride` is the caller's chunk size, never a constant in this crate.
#[must_use]
pub fn next_admission_pos(
    len: u32,
    pos: u32,
    hints: &[u32],
    branch: u32,
    stride: u32,
    spare_slot: bool,
) -> u32 {
    let mut stop = len;
    for &hint in hints {
        if hint > pos && hint < stop {
            stop = hint;
        }
    }
    if branch > pos && branch <= len && branch < stop {
        stop = branch;
    }
    if spare_slot && stride > 0 {
        // Absolute multiples of the chunk size, never a multiple of what is left.
        let next = pos / stride * stride + stride;
        if next > pos && next <= len && next < stop {
            stop = next;
        }
    }
    stop
}

#[cfg(test)]
mod tests {
    use super::{CacheGeometry, ckpt_utility, next_admission_pos};

    /// A model with no recurrent state: its checkpoint blob is zero bytes.
    fn stateless() -> CacheGeometry {
        CacheGeometry::new(256, 4_096, 0, 0.004, 600.0)
    }

    /// qwen35's numbers: 51.2 MiB of recurrent state per checkpoint.
    fn stateful() -> CacheGeometry {
        CacheGeometry::new(256, 8_388_608, 53_684_224, 0.004, 600.0)
    }

    #[test]
    fn a_stateless_checkpoints_utility_is_finite_at_every_distance() {
        let geometry = stateless();
        for distance in [0, 1, 300, 20_000] {
            let utility = ckpt_utility(distance, &geometry, 1.0, 1.0);
            assert!(
                utility.is_finite(),
                "a zero-byte blob must not divide {distance} tokens of prefill by zero"
            );
        }
        assert!(
            ckpt_utility(0, &geometry, 1.0, 1.0).abs() < 1e-9,
            "a checkpoint that saves no prefill is worth nothing, not a NaN"
        );
        assert!(ckpt_utility(1, &geometry, 1.0, 1.0) > 0.0);
    }

    #[test]
    fn stateless_checkpoints_rank_by_the_prefill_they_save() {
        // Every stateless checkpoint costs the same -- a logits row the store does
        // not size -- so the distance is the whole of the ranking.
        let geometry = stateless();
        let crowded = ckpt_utility(300, &geometry, 1.0, 1.0);
        let lone = ckpt_utility(20_000, &geometry, 1.0, 1.0);
        assert!(
            lone > crowded,
            "the lone checkpoint at 20k saves more prefill than one 300 above its neighbour"
        );
    }

    #[test]
    fn a_geometry_that_carries_a_blob_is_still_priced_by_its_own_size() {
        // The stateless price is substituted at zero, never applied as a floor: a
        // checkpoint that really costs 51.2 MiB is worth 51.2 times less than a
        // stateless one that saves the same prefill.
        let with_blob = ckpt_utility(20_000, &stateful(), 1.0, 1.0);
        let without = ckpt_utility(20_000, &stateless(), 1.0, 1.0);
        assert!((without / with_blob - 51.2).abs() < 0.01);
    }

    #[test]
    fn stops_at_the_end_with_nothing_to_stop_for() {
        assert_eq!(next_admission_pos(1000, 0, &[], 0, 0, false), 1000);
    }

    #[test]
    fn hints_beat_the_end_and_earlier_hints_beat_later_ones() {
        assert_eq!(next_admission_pos(1000, 0, &[400, 700], 0, 0, false), 400);
        assert_eq!(next_admission_pos(1000, 500, &[400, 700], 0, 0, false), 700);
    }

    #[test]
    fn a_hint_at_or_behind_the_cursor_is_not_a_stop() {
        // A hint at or behind the current position is already spent; taking it would
        // make the position never advance and the caller's loop would never advance.
        assert_eq!(next_admission_pos(1000, 500, &[400], 0, 0, false), 1000);
    }

    #[test]
    fn a_branch_point_ahead_stops_before_the_branch() {
        // Step 7 makes the branch position itself a stop: the commit ends where the
        // path stops matching the tree, so nothing is thrown away and the next
        // request can still resume from it.
        assert_eq!(next_admission_pos(1000, 100, &[], 250, 0, false), 250);
        // A branch already behind the frontier cannot stop anything.
        assert_eq!(next_admission_pos(1000, 500, &[], 300, 0, false), 1000);
    }

    #[test]
    fn stride_positions_only_while_a_slot_is_spare() {
        assert_eq!(next_admission_pos(9000, 0, &[], 0, 2048, true), 2048);
        assert_eq!(next_admission_pos(9000, 0, &[], 0, 2048, false), 9000);
        // Absolute multiples, not multiples of the remainder.
        assert_eq!(next_admission_pos(9000, 3000, &[], 0, 2048, true), 4096);
        // A stride past the end is not a stop.
        assert_eq!(next_admission_pos(3000, 2048, &[], 0, 2048, true), 3000);
    }

    #[test]
    fn the_earliest_of_the_three_rules_wins() {
        // Step 7: the stop is the earliest of the hint, the branch point, and -- only
        // while a checkpoint slot is spare -- the next stride multiple.
        assert_eq!(next_admission_pos(9000, 0, &[5000], 3000, 2048, true), 2048);
        assert_eq!(next_admission_pos(9000, 0, &[1000], 3000, 2048, true), 1000);
        assert_eq!(
            next_admission_pos(9000, 2100, &[5000], 3000, 2048, true),
            3000
        );
        // With no slot going spare the stride stop is gone and the branch wins.
        assert_eq!(
            next_admission_pos(9000, 0, &[5000], 3000, 2048, false),
            3000
        );
    }

    #[test]
    fn a_branch_behind_the_cursor_is_not_a_stop() {
        assert_eq!(next_admission_pos(9000, 4000, &[], 3000, 0, false), 9000);
    }
}
