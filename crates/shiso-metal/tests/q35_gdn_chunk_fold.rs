//! Metal's run of the shared `q35_gdn_chunk_fold` cases.
//!
//! The cases themselves live in `shiso_kerneltest::qwen35::chunk_fold`, so this
//! file is the backend's half only. There is nothing to open: the cases are
//! pure host index arithmetic about `gdn_chunk.metal`'s lane handout, which is
//! why they are not `#[ignore]`-gated and why the crate that owns them runs
//! them as unit tests on every machine as well.
//!
//! **There is deliberately no CUDA sibling of this file.** `CSPLIT` is a Metal
//! answer to a 32768 B threadgroup-memory cap; a `sm_121` block may opt into
//! 101376 B, so the CUDA chunk kernel carries the original single shape and has
//! no lane handout to partition. A driver here would report five passes and
//! grade nothing on that device.
#![cfg(all(feature = "metal", target_os = "macos"))]

use shiso_kerneltest::qwen35::chunk_fold as cases;

#[test]
fn the_units_partition_the_row_at_every_split() {
    cases::the_units_partition_the_row_at_every_split();
}

#[test]
fn the_slot_rotation_is_a_permutation_of_every_unit() {
    cases::the_slot_rotation_is_a_permutation_of_every_unit();
}

#[test]
fn every_delta_has_one_lane_and_one_slot() {
    cases::every_delta_has_one_lane_and_one_slot();
}

#[test]
fn a_lanes_chains_cover_its_own_deltas_exactly_once() {
    cases::a_lanes_chains_cover_its_own_deltas_exactly_once();
}

#[test]
fn all_four_splits_build_the_same_combination_tree() {
    cases::all_four_splits_build_the_same_combination_tree();
}
