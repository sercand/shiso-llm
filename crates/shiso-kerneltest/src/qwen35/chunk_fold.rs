//! The chunk kernel's lane handout, replayed as index arithmetic on the host.
//!
//! `gdn_chunk.metal` compiles one body at four lane splits. Two of them,
//! `_c4` and `_c8`, need 512 and 1024 threads, and no machine this project has
//! been built on will launch either: they compile, they resolve as pipelines,
//! and every device case that would exercise them skips. Their fold order and
//! their delta ownership were settled by an argument on paper, and an argument
//! on paper is not a regression test - a later edit to
//! `q35_gdn_chunk_fold` or to the `usub` handout would break exactly the two
//! widths nothing runs, on a machine that has one.
//!
//! So the arithmetic is replayed here. Not the kernel's floats: a second float
//! implementation would be a copy of the thing it is checking, and the two
//! would be wrong together. What is replayed is the *partition* - which lane
//! owns which unit, which deltas a lane's strided chains reach, and which
//! order the partials are combined in - because that is what the parameter
//! `CSPLIT` actually varies, and every one of those questions has an answer
//! that is a set or a tree rather than a number.
//!
//! Pure host Rust, so it runs on every machine and in every build. That is the
//! point: the widths this cannot cover on a device are the widths it exists
//! for.
//!
//! **This module is about the Metal body and nothing else, and there is no CUDA
//! driver for it.** `CSPLIT` exists because Apple caps threadgroup memory at
//! 32768 B and the CUDA chunk kernel this set was ported from wants 99072 B; a
//! `sm_121` block may opt into 101376 B, so the CUDA side carries the original's
//! single shape and has no lane handout to partition. A CUDA driver here would
//! report five passes and grade nothing on that device.
//!
//! The cases run twice: as this crate's own unit tests, on every machine and in
//! every build, and through `crates/shiso-metal/tests/q35_gdn_chunk_fold.rs`
//! beside the Metal kernel they describe. They were only compiled at all on
//! macOS before the move.

/// The canonical split: four-slot units in one 128-float state row, and
/// residue classes in the delta chains. `Q35_GDN_CSPLIT` in `q35_common.h`.
const UNITS: usize = 8;

/// Slots in one unit, `Q35_GDN_UNIT_SLOTS`. Each slot is a `float4`, so a unit
/// is sixteen of the row's floats and eight units tile the row.
const UNIT_SLOTS: usize = 4;

/// Floats in one state row, `Q35_GDN_D`.
const HEAD_DIM: usize = UNITS * UNIT_SLOTS * 4;

/// Tokens in one run, `Q35_GDN_CHUNK`.
const CHUNK: usize = 64;

/// The lane splits the body is instantiated at, and the only ones it admits.
///
/// The `static_assert` in `q35_gdn_chunk_impl` refuses any other value, and
/// the reason is the same one this file is testing: at 16 a lane would own
/// half a unit and there would be no canonical order left to reproduce.
const SPLITS: [usize; 4] = [1, 2, 4, 8];

/// The units lane `sub` owns at a given split, in the order it walks them.
///
/// `usub = ui * CSPLIT + sub`, verbatim from the three phases and the state
/// load. Interleaved, not blocked: `sub` lands in the low bits of the unit
/// index, which is what makes the cross-lane butterfly pair by the same bits
/// the reference tree pairs by.
fn owned_units(csplit: usize, sub: usize) -> Vec<usize> {
    (0..UNITS / csplit).map(|ui| ui * csplit + sub).collect()
}

/// The floats of the state row unit `usub` covers, in the rotated order the
/// kernel walks them.
///
/// `slot = (n4 + urot) & (UNIT_SLOTS - 1)` with `urot = usub / 2`, and each
/// slot is the four floats at `4 * (usub * UNIT_SLOTS + slot)`.
fn unit_floats(usub: usize) -> Vec<usize> {
    let urot = usub / 2;
    (0..UNIT_SLOTS)
        .flat_map(|n4| {
            let slot = (n4 + urot) & (UNIT_SLOTS - 1);
            let base = 4 * (usub * UNIT_SLOTS + slot);
            (base..base + 4).collect::<Vec<_>>()
        })
        .collect()
}

/// Every unit belongs to exactly one lane, and the units tile the state row.
///
/// The three claims that make `CSPLIT` a free parameter rather than a rewrite:
/// a lane owns a whole number of units, the lanes between them own each unit
/// once, and `sub` is the low bits of every unit index a lane holds. The last
/// is the one a "tidier" blocked handout would break, and it would break it
/// silently - the units would still partition, and the deltas would move to a
/// lane that no longer satisfies `sub == t % CSPLIT`.
pub fn the_units_partition_the_row_at_every_split() {
    for csplit in SPLITS {
        let mut seen = vec![0usize; UNITS];
        for sub in 0..csplit {
            let owned = owned_units(csplit, sub);
            assert_eq!(
                owned.len(),
                UNITS / csplit,
                "CSPLIT {csplit} lane {sub} owns a whole number of units",
            );
            for usub in owned {
                assert!(usub < UNITS, "CSPLIT {csplit} lane {sub} owns unit {usub}");
                assert_eq!(
                    usub % csplit,
                    sub,
                    "CSPLIT {csplit}: unit {usub} carries the lane in its low bits",
                );
                seen[usub] += 1;
            }
        }
        assert!(
            seen.iter().all(|count| *count == 1),
            "CSPLIT {csplit} covers every unit exactly once: {seen:?}",
        );
    }

    // And the units, however they are handed out, tile the row: every float of
    // the state row is read by exactly one unit, through exactly one slot.
    let mut floats = vec![0usize; HEAD_DIM];
    for usub in 0..UNITS {
        let walked = unit_floats(usub);
        assert_eq!(
            walked.len(),
            UNIT_SLOTS * 4,
            "unit {usub} walks sixteen floats"
        );
        for at in walked {
            floats[at] += 1;
        }
    }
    assert!(
        floats.iter().all(|count| *count == 1),
        "the eight units tile the 128-float row: {floats:?}",
    );
}

/// The rotation reorders a unit's slots and never leaves or repeats one.
///
/// `urot` exists so that the sequence belongs to the unit rather than to the
/// lane holding it, which is the whole reason a lane may hold several. It is
/// only sound while it is a permutation: a rotation that dropped a slot would
/// read a stale register and a partial state row, and both would look like a
/// numerical problem rather than an indexing one.
pub fn the_slot_rotation_is_a_permutation_of_every_unit() {
    for usub in 0..UNITS {
        let urot = usub / 2;
        let mut slots: Vec<usize> = (0..UNIT_SLOTS)
            .map(|n4| (n4 + urot) & (UNIT_SLOTS - 1))
            .collect();
        slots.sort_unstable();
        assert_eq!(
            slots,
            (0..UNIT_SLOTS).collect::<Vec<_>>(),
            "unit {usub} walks each of its slots once",
        );
    }
}

/// Which lane holds token `t`'s delta, and at which index of its own store.
///
/// `sub == t & (CSPLIT - 1)` writes it, `dreg[t / CSPLIT]` is where; the
/// hybrid scales the index by a row pitch instead of by one, which changes the
/// address and not the ownership.
const fn delta_owner(csplit: usize, t: usize) -> (usize, usize) {
    (t & (csplit - 1), t / csplit)
}

/// Every token's delta has one home, and the stores are exactly full.
///
/// `dreg` is declared `float[Q35_GDN_CHUNK / CSPLIT]`, so an index past that is
/// a register array overrun the compiler will not catch - the index is a
/// runtime value. The bound is the claim being checked here.
pub fn every_delta_has_one_lane_and_one_slot() {
    for csplit in SPLITS {
        let depth = CHUNK / csplit;
        let mut homes = vec![vec![0usize; depth]; csplit];
        for t in 0..CHUNK {
            let (sub, slot) = delta_owner(csplit, t);
            assert!(
                slot < depth,
                "CSPLIT {csplit}: token {t} lands at dreg[{slot}] of {depth}",
            );
            homes[sub][slot] += 1;
        }
        for (sub, slots) in homes.iter().enumerate() {
            assert!(
                slots.iter().all(|count| *count == 1),
                "CSPLIT {csplit} lane {sub} fills each of its slots once: {slots:?}",
            );
        }
    }
}

/// A lane's strided chains read its own deltas, and each of them once.
///
/// The substitution and the output phase both walk `for u = usub; u < t; u +=
/// 8` per owned unit - the CSPLIT = 8 chains, never one long sweep over the
/// lane's whole residue class. Two things have to hold at once and they pull in
/// opposite directions: every `u` a lane reaches must be a delta that lane
/// *owns*, since there is no shared array to read someone else's from, and the
/// lanes together must reach every `u` below `t` exactly once, since the sum is
/// over all of them.
///
/// That pair is what fixes the stride at `Q35_GDN_CSPLIT` rather than at
/// `CSPLIT`: a chain strided by the lane count would still cover the tokens,
/// and would cover them from the wrong lanes.
pub fn a_lanes_chains_cover_its_own_deltas_exactly_once() {
    for csplit in SPLITS {
        // `t` is the row being substituted; the output phase is the same walk
        // with `u <= t`, so both bounds are checked at every row.
        for t in 0..CHUNK {
            for inclusive in [false, true] {
                let last = if inclusive { t } else { t.saturating_sub(1) };
                let reach = if inclusive || t > 0 { last + 1 } else { 0 };
                let mut seen = vec![0usize; reach];
                for sub in 0..csplit {
                    for usub in owned_units(csplit, sub) {
                        let mut u = usub;
                        while u < reach {
                            assert_eq!(
                                delta_owner(csplit, u).0,
                                sub,
                                "CSPLIT {csplit} lane {sub} reads the delta of token {u}, \
                                 which lane {} owns",
                                delta_owner(csplit, u).0,
                            );
                            seen[u] += 1;
                            u += UNITS;
                        }
                    }
                }
                assert!(
                    seen.iter().all(|count| *count == 1),
                    "CSPLIT {csplit}, t {t}, inclusive {inclusive}: {seen:?}",
                );
            }
        }
    }
}

/// One addition, as a tree over the unit indices that reached it.
///
/// Floating-point addition is commutative to the bit and associative to
/// nothing, so this is what "the same order" has to mean: the same bracketing,
/// read up to the order of each pair's operands. [`Sum::canonical`] is what
/// throws away the part that does not matter, and it throws away nothing else.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Sum {
    /// One unit's partial, before anything is combined.
    Unit(usize),
    /// Two partials added. Which is the left operand is not significant.
    Add(Box<Sum>, Box<Sum>),
}

impl Sum {
    /// The same tree with every pair's operands in a fixed order.
    ///
    /// Sorted on the smallest unit index below each operand, which is a total
    /// order here because the leaves are distinct and each appears once.
    fn canonical(&self) -> Self {
        match self {
            Self::Unit(unit) => Self::Unit(*unit),
            Self::Add(left, right) => {
                let (left, right) = (left.canonical(), right.canonical());
                if left.first() <= right.first() {
                    Self::Add(Box::new(left), Box::new(right))
                } else {
                    Self::Add(Box::new(right), Box::new(left))
                }
            }
        }
    }

    /// The smallest unit index this subtree covers.
    fn first(&self) -> usize {
        match self {
            Self::Unit(unit) => *unit,
            Self::Add(left, right) => left.first().min(right.first()),
        }
    }
}

/// The tree one lane holds after the fold, at a given split.
///
/// `q35_gdn_chunk_fold` verbatim: the cross-lane butterfly over each slot of
/// the partial array first, pairing by bit 0 then bit 1 then bit 2 of the lane
/// index, and the lane's own slots folded afterwards by doubling stride. Both
/// loops are written out rather than reasoned about, because the claim is that
/// this code and not an idealization of it builds one tree.
fn fold_tree(csplit: usize, sub: usize) -> Sum {
    let units = UNITS / csplit;
    // `part[ui]` across every lane at once: the butterfly reads a neighbour's
    // slot, so one lane's array is not enough to simulate it.
    let mut part: Vec<Vec<Sum>> = (0..csplit)
        .map(|lane| {
            owned_units(csplit, lane)
                .into_iter()
                .map(Sum::Unit)
                .collect()
        })
        .collect();

    for ui in 0..units {
        let mut offset = 1;
        while offset < csplit {
            // Every lane exchanges at once, so the step is read from the state
            // before any of it is written.
            let before: Vec<Sum> = part.iter().map(|lane| lane[ui].clone()).collect();
            for (lane, slots) in part.iter_mut().enumerate() {
                slots[ui] = Sum::Add(
                    Box::new(before[lane].clone()),
                    Box::new(before[lane ^ offset].clone()),
                );
            }
            offset <<= 1;
        }
    }

    let slots = &mut part[sub];
    let mut step = 1;
    while step < units {
        let mut i = 0;
        while i + step < units {
            slots[i] = Sum::Add(
                Box::new(slots[i].clone()),
                Box::new(slots[i + step].clone()),
            );
            i += 2 * step;
        }
        step <<= 1;
    }
    slots[0].clone()
}

/// All four splits combine the eight units in one order, and it is the
/// reference one.
///
/// This is the property `chunk_variants_are_bit_identical` checks on a device
/// and can only check for the widths that device will launch. Here it is
/// checked for all four, which is the only place `_c4` and `_c8` are covered at
/// all on this hardware.
///
/// The reference is `CSPLIT = 8`'s own tree - three shuffles, pairing by bit 0,
/// then bit 1, then bit 2 - and it is written out as a literal rather than
/// taken from `fold_tree(8, ..)`, so that a change to the butterfly cannot move
/// the reference and the widths together.
pub fn all_four_splits_build_the_same_combination_tree() {
    fn add(left: Sum, right: Sum) -> Sum {
        Sum::Add(Box::new(left), Box::new(right))
    }
    let pair = |a: usize, b: usize| add(Sum::Unit(a), Sum::Unit(b));
    let reference = add(add(pair(0, 1), pair(2, 3)), add(pair(4, 5), pair(6, 7))).canonical();

    for csplit in SPLITS {
        for sub in 0..csplit {
            assert_eq!(
                fold_tree(csplit, sub).canonical(),
                reference,
                "CSPLIT {csplit} lane {sub} folds in the reference order",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    //! The five cases, run as unit tests of this crate.
    //!
    //! They need no device and no backend, so they run wherever `cargo test`
    //! does. The Metal driver runs the same five beside the kernel; that is a
    //! duplicate run of a millisecond of index arithmetic, and the alternative
    //! is a file that a Linux build compiles and never executes, which is what
    //! it was.

    #[test]
    fn the_units_partition_the_row_at_every_split() {
        super::the_units_partition_the_row_at_every_split();
    }

    #[test]
    fn the_slot_rotation_is_a_permutation_of_every_unit() {
        super::the_slot_rotation_is_a_permutation_of_every_unit();
    }

    #[test]
    fn every_delta_has_one_lane_and_one_slot() {
        super::every_delta_has_one_lane_and_one_slot();
    }

    #[test]
    fn a_lanes_chains_cover_its_own_deltas_exactly_once() {
        super::a_lanes_chains_cover_its_own_deltas_exactly_once();
    }

    #[test]
    fn all_four_splits_build_the_same_combination_tree() {
        super::all_four_splits_build_the_same_combination_tree();
    }
}
