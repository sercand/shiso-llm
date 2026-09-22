//! Tests for the multimodal position rule and the splice map beside it.
//!
//! A position mistake here is silent. The request succeeds, every buffer is the
//! right size, the model stays fluent, and it answers about a picture that has
//! been rearranged or about the placeholder embedding it was left holding. There
//! is no downstream check that catches it, so the arithmetic is pinned directly.
//!
//! The cases the image rule is held to are computed from the pre-temporal rule
//! written out again inside the test, not from remembered constants: an image is
//! the one-group case, and the whole claim is that introducing the temporal axis
//! moved nothing.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "the fixtures are small literal grids and the pre-temporal rule is transcribed with the widths it was written in; a conversion at every index would make it a worse transcription of the rule it is here to reproduce"
)]

use super::{MediaPositions, PadTokens, Streams, VisionGrid, Walk};
use shiso_core::ids::TokenId;

const IMG: TokenId = TokenId::new(248_056);
const VID: TokenId = TokenId::new(248_057);
const TXT: TokenId = TokenId::new(7);

fn pads() -> PadTokens {
    PadTokens::new(Some(IMG), Some(VID))
}

/// One whole-prompt walk over the three-stream rule: every token in a single
/// chunk.
fn run(tokens: &[TokenId], grids: &[VisionGrid], start: u32) -> Streams {
    let mut out = Streams::default();
    Walk::new(grids, pads(), start, MediaPositions::Mrope3D).chunk(tokens, &mut out);
    out
}

/// The same walk under the one-position-per-token rule.
fn run_linear(tokens: &[TokenId], grids: &[VisionGrid], start: u32) -> Streams {
    let mut out = Streams::default();
    Walk::new(grids, pads(), start, MediaPositions::Linear).chunk(tokens, &mut out);
    out
}

/// The same walk, split at every offset, proving the split is invisible.
fn run_split(tokens: &[TokenId], grids: &[VisionGrid], start: u32, at: usize) -> Streams {
    let mut out = Streams::default();
    let mut walk = Walk::new(grids, pads(), start, MediaPositions::Mrope3D);
    let (head, tail) = tokens.split_at(at);
    walk.chunk(head, &mut out);
    walk.chunk(tail, &mut out);
    out
}

fn grid(t: u32, h: u32, w: u32) -> VisionGrid {
    VisionGrid::new(t, h, w)
}

// -- text --------------------------------------------------------------------

#[test]
fn text_only_is_the_identity_ramp_on_all_three_streams() {
    let out = run(&[TXT; 5], &[], 0);
    assert_eq!(out.t, vec![0, 1, 2, 3, 4]);
    assert_eq!(out.h, out.t);
    assert_eq!(out.w, out.t);
    assert_eq!(out.map, vec![-1; 5], "no text token takes an encoder row");
}

#[test]
fn the_streams_start_where_the_caller_says() {
    let out = run(&[TXT; 3], &[], 100);
    assert_eq!(out.t, vec![100, 101, 102]);
    assert_eq!(out.h, out.t);
    assert_eq!(out.w, out.t);
}

// -- images ------------------------------------------------------------------

/// A 2x3 image: the temporal stream is constant over the whole run, the row
/// stream steps by grid row and the column stream by grid column, and the
/// running position then advances by the long side.
#[test]
fn an_image_holds_its_temporal_stream_constant_and_advances_by_its_long_side() {
    let tokens = [TXT, IMG, IMG, IMG, IMG, IMG, IMG, TXT];
    let out = run(&tokens, &[grid(1, 2, 3)], 0);

    assert_eq!(out.t, vec![0, 1, 1, 1, 1, 1, 1, 4]);
    assert_eq!(out.h, vec![0, 1, 1, 1, 2, 2, 2, 4]);
    assert_eq!(out.w, vec![0, 1, 2, 3, 1, 2, 3, 4]);
    // The trailing text token sits at 1 + max(2, 3).
    assert_eq!(*out.t.last().expect("a trailing text token"), 4);
    assert_eq!(out.map, vec![-1, 0, 1, 2, 3, 4, 5, -1]);
}

/// The independent statement of the same guarantee: an image is the one-group
/// case, so introducing the temporal axis must not have moved a position. The
/// expected values come from the pre-temporal rule, written out here, rather
/// than from a constant somebody could have updated to match a regression.
#[test]
fn the_image_rule_is_unchanged_by_the_temporal_axis() {
    fn old_rule(tokens: &[TokenId], grids: &[(u32, u32)]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
        let (mut pos, mut idx, mut i) = (0i32, 0usize, 0usize);
        while i < tokens.len() {
            if tokens[i] == IMG && idx < grids.len() {
                let (gh, gw) = grids[idx];
                let run = (gh * gw) as usize;
                let base = pos;
                for k in 0..run {
                    t.push(base);
                    h.push(base + (k as u32 / gw.max(1)) as i32);
                    w.push(base + (k as u32 % gw.max(1)) as i32);
                }
                pos += gh.max(gw) as i32;
                i += run;
                idx += 1;
            } else {
                t.push(pos);
                h.push(pos);
                w.push(pos);
                pos += 1;
                i += 1;
            }
        }
        (t, h, w)
    }

    for (gh, gw) in [(1u32, 1u32), (2, 3), (3, 2), (4, 4), (1, 7), (7, 1)] {
        let mut tokens = vec![TXT, TXT];
        tokens.extend(std::iter::repeat_n(IMG, (gh * gw) as usize));
        tokens.extend([TXT, TXT]);
        let out = run(&tokens, &[grid(1, gh, gw)], 0);
        assert_eq!(
            (out.t, out.h, out.w),
            old_rule(&tokens, &[(gh, gw)]),
            "the {gh}x{gw} image moved when the temporal axis was added"
        );
    }
}

#[test]
fn two_images_each_advance_the_running_position_and_take_consecutive_rows() {
    let tokens = [IMG, IMG, IMG, IMG, IMG, TXT];
    // 1x2 based at 0 (advances 2), then 1x3 based at 2 (advances 3): text at 5.
    let out = run(&tokens, &[grid(1, 1, 2), grid(1, 1, 3)], 0);
    assert_eq!(out.t, vec![0, 0, 2, 2, 2, 5]);
    assert_eq!(out.h, vec![0, 0, 2, 2, 2, 5]);
    assert_eq!(out.w, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(out.map, vec![0, 1, 2, 3, 4, -1], "one packed row block");
}

// -- video -------------------------------------------------------------------

/// The rule video exists for. Three temporal groups over a 2x2 grid: the
/// temporal stream advances once per GROUP while the two spatial streams restart
/// in every group, and the item advances the running position by max(3, 2, 2).
#[test]
fn a_clip_advances_its_temporal_stream_once_per_group() {
    let tokens = [
        TXT, VID, VID, VID, VID, VID, VID, VID, VID, VID, VID, VID, VID, TXT,
    ];
    let out = run(&tokens, &[grid(3, 2, 2)], 0);

    assert_eq!(
        out.t,
        vec![0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4],
        "the temporal stream must step per group"
    );
    assert_eq!(out.h, vec![0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 4]);
    assert_eq!(out.w, vec![0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 4]);
    assert_eq!(*out.t.last().expect("a trailing text token"), 4);
}

/// A clip longer than it is wide advances by its TEMPORAL extent. If the advance
/// assumed the spatial pair dominates, a long clip would overlap the text after
/// it, every following token sharing a position with the tail of the clip.
#[test]
fn a_long_clip_advances_by_its_temporal_extent() {
    let t_len = 10u32;
    let mut tokens = vec![TXT];
    tokens.extend(std::iter::repeat_n(VID, t_len as usize));
    tokens.push(TXT);
    let out = run(&tokens, &[grid(t_len, 1, 1)], 0);
    assert_eq!(out.t[0], 0);
    assert_eq!(
        &out.t[1..=t_len as usize],
        &(1..=t_len as i32).collect::<Vec<_>>()[..]
    );
    assert_eq!(
        *out.t.last().expect("a trailing text token"),
        1 + t_len as i32,
        "the clip's temporal extent did not advance the running position"
    );
}

/// A clip and a still in the same request, which is the case where a mis-scan
/// shows up as everything after the clip being shifted.
#[test]
fn a_clip_and_an_image_interleave_correctly() {
    let mut tokens = vec![TXT];
    tokens.extend(std::iter::repeat_n(VID, 2));
    tokens.push(TXT);
    tokens.extend(std::iter::repeat_n(IMG, 4));
    tokens.push(TXT);
    let out = run(&tokens, &[grid(2, 1, 1), grid(1, 2, 2)], 0);
    // text 0 | clip based at 1, groups 1 and 2, advance max(2,1,1) -> 3
    // text 3 | still based at 4, advance max(2,2) -> 6 | text 6
    assert_eq!(out.t, vec![0, 1, 2, 3, 4, 4, 4, 4, 6]);
    assert_eq!(out.map, vec![-1, 0, 1, -1, 2, 3, 4, 5, -1]);
}

/// A one-group clip is positionally identical to a still of the same grid. It
/// should be: one temporal group IS a still.
#[test]
fn a_one_group_clip_matches_the_equivalent_image() {
    let image = [TXT, IMG, IMG, IMG, IMG, TXT];
    let clip = [TXT, VID, VID, VID, VID, TXT];
    let a = run(&image, &[grid(1, 2, 2)], 0);
    let b = run(&clip, &[grid(1, 2, 2)], 0);
    assert_eq!((a.t, a.h, a.w, a.map), (b.t, b.h, b.w, b.map));
}

/// Matching only the image placeholder walks a clip's run one text token at a
/// time. This is that failure, stated as the difference a vocabulary with no
/// clip placeholder produces, so the case fails if the match is ever narrowed.
#[test]
fn a_clip_placeholder_the_vocabulary_does_not_declare_is_walked_as_text() {
    let tokens = [TXT, VID, VID, VID, VID, TXT];
    let grids = [grid(1, 2, 2)];
    let mut narrow = Streams::default();
    Walk::new(
        &grids,
        PadTokens::new(Some(IMG), None),
        0,
        MediaPositions::Mrope3D,
    )
    .chunk(&tokens, &mut narrow);
    assert_eq!(
        narrow.t,
        vec![0, 1, 2, 3, 4, 5],
        "an unrecognized run is walked a token at a time"
    );
    assert_eq!(narrow.map, vec![-1; 6], "and no encoder row is spliced");
    assert_ne!(narrow.t, run(&tokens, &grids, 0).t);
}

// -- linear media positions --------------------------------------------------

/// The family that positions a picture the way it positions text: a placeholder
/// token takes the next sequence position like any other token, and the three
/// streams are that one number repeated.
///
/// What is also the token index is the bound: attention is causal, so every
/// token reads up to its own sequence position.
#[test]
fn linear_positions_are_the_token_index() {
    let mut tokens = vec![TXT; 4];
    tokens.extend(std::iter::repeat_n(IMG, 6));
    let out = run_linear(&tokens, &[grid(1, 2, 3)], 0);

    let ramp: Vec<i32> = (0..10).collect();
    assert_eq!(out.t, ramp, "a placeholder advances the position by one");
    assert_eq!(out.h, ramp, "and the three streams are that one number");
    assert_eq!(out.w, ramp);
    assert_eq!(
        out.map,
        vec![-1, -1, -1, -1, 0, 1, 2, 3, 4, 5],
        "the encoder rows are handed out exactly as they are under the grid rule"
    );
    assert_eq!(
        out.bound, ramp,
        "every token sees up to its own sequence position"
    );
}

/// A second item starts its own span. The first image's tokens must not be
/// allowed to read the second's, which is what a bound taken over the whole
/// prompt rather than over the item would produce.
#[test]
fn each_item_bounds_its_own_span() {
    let mut tokens = vec![TXT];
    tokens.extend(std::iter::repeat_n(IMG, 4));
    tokens.push(TXT);
    tokens.extend(std::iter::repeat_n(VID, 4));
    tokens.push(TXT);
    let out = run_linear(&tokens, &[grid(1, 2, 2), grid(2, 2, 1)], 0);
    assert_eq!(out.t, (0..11).collect::<Vec<_>>());
    assert_eq!(out.bound, (0..11).collect::<Vec<_>>());
}

/// Under the three-stream rule the bound is the token's SEQUENCE position, not
/// its stream position: a cache row is addressed by the linear position, so a
/// bound that carried the grid coordinate would name a row belonging to another
/// token. The two numbers part company at the first image, which is why this is
/// a case rather than a comment.
#[test]
fn mrope_bounds_are_sequence_positions() {
    let mut tokens = vec![TXT, TXT];
    tokens.extend(std::iter::repeat_n(IMG, 6));
    tokens.extend([TXT, TXT]);
    let out = run(&tokens, &[grid(1, 2, 3)], 0);
    assert_eq!(
        out.bound,
        (0..10).collect::<Vec<_>>(),
        "every token may read every key up to its own position"
    );
    assert_ne!(
        out.t, out.bound,
        "and the stream position is a different number"
    );
}

// -- where a chunk may be cut ------------------------------------------------

/// The engine splits a prompt wider than its scratch, and a placeholder run is
/// the unit the bound is stated in: a cut inside one would give the first half
/// of an item a bound past the end of the chunk. The split is therefore pulled
/// back to the last token before the run starts.
#[test]
fn split_point_never_cuts_an_item() {
    let mut tokens = vec![TXT; 500];
    tokens.extend(std::iter::repeat_n(IMG, 30));
    tokens.extend(std::iter::repeat_n(TXT, 10));
    let grids = [grid(1, 5, 6)];
    let walk = Walk::new(&grids, pads(), 0, MediaPositions::Linear);
    assert_eq!(
        walk.split_point(&tokens, 512),
        500,
        "the chunk stops where the item starts"
    );
}

/// An item wider than the whole chunk has no cut that clears it, so the width
/// wins: taking nothing would be a prefill that never advances.
#[test]
fn an_item_wider_than_the_chunk_falls_back_to_the_width() {
    let mut tokens: Vec<TokenId> = std::iter::repeat_n(IMG, 30).collect();
    tokens.extend(std::iter::repeat_n(TXT, 10));
    let grids = [grid(1, 5, 6)];
    let walk = Walk::new(&grids, pads(), 0, MediaPositions::Linear);
    assert_eq!(walk.split_point(&tokens, 20), 20);
}

/// A chunk that fits takes every token; a chunk that does not stops before the
/// run; and a run a fallback already cut into is measured from what it has left.
#[test]
fn split_point_takes_what_fits_and_measures_a_run_in_flight() {
    let tokens = [TXT, IMG, IMG, IMG, IMG, TXT];
    let grids = [grid(1, 2, 2)];
    let walk = Walk::new(&grids, pads(), 0, MediaPositions::Linear);
    assert_eq!(walk.split_point(&tokens, 16), tokens.len());
    // The run starts at 1 and ends at 4, so 1 is the last cut that clears it.
    assert_eq!(walk.split_point(&tokens, 3), 1);

    let mut part = Streams::default();
    let mut walk = Walk::new(&grids, pads(), 0, MediaPositions::Linear);
    walk.chunk(&tokens[..2], &mut part);
    assert_eq!(
        walk.split_point(&tokens[2..], 3),
        3,
        "the three tokens the run has left are the cut that clears it"
    );
}

/// A run in flight is the item the walk is still ON, not one it has finished:
/// [`Walk::chunk`] advances the item index only when a run completes. So the
/// next placeholder run a split sees is the item AFTER the one in flight, and a
/// scan that started from the walk's own index measures it - and every item
/// after it - by the in-flight item's count.
///
/// Here the in-flight item is the SHORTER of the two, so the wrong count lands
/// the cut inside the second item's run: exactly the thing this function
/// exists to prevent, and undetectable downstream.
#[test]
fn a_split_after_a_resumed_run_measures_the_next_item_by_its_own_count() {
    // An 8-pad item and a 64-pad one. The walk has already emitted four of the
    // first item's tokens, so four are owed.
    let grids = [grid(1, 2, 4), grid(1, 8, 8)];
    let mut out = Streams::default();
    let mut walk = Walk::new(&grids, pads(), 0, MediaPositions::Linear);
    walk.chunk(&[IMG; 4], &mut out);

    let mut tokens = vec![IMG; 4];
    tokens.extend([TXT, TXT]);
    tokens.extend(std::iter::repeat_n(IMG, 64));
    // The four owed tokens, then two text tokens, and the second item's run
    // starts at 6 and is far wider than the width. Six is the only cut.
    assert_eq!(
        walk.split_point(&tokens, 20),
        6,
        "the cut has to clear the second item, which is 64 wide and not 8"
    );
}

/// The mirror: the item in flight is the LONGER of the two, so the wrong count
/// overshoots and the scan stops seeing the boundaries past it. Nothing is cut
/// inside a run this time - the chunk is simply far shorter than it could be,
/// which is a prefill that costs more launches for no reason.
#[test]
fn a_split_after_a_resumed_run_still_sees_the_boundaries_past_a_shorter_item() {
    let grids = [grid(1, 8, 8), grid(1, 2, 4)];
    let mut out = Streams::default();
    let mut walk = Walk::new(&grids, pads(), 0, MediaPositions::Linear);
    walk.chunk(&[IMG; 60], &mut out);

    let mut tokens = vec![IMG; 4];
    tokens.extend([TXT, TXT]);
    tokens.extend(std::iter::repeat_n(IMG, 8));
    tokens.extend(std::iter::repeat_n(TXT, 10));
    // Four owed, two text, the 8-pad item's whole run, then text to the width.
    assert_eq!(walk.split_point(&tokens, 20), 20);
}

// -- chunk boundaries --------------------------------------------------------

/// Splitting the prompt anywhere produces the same four streams. The split is
/// the engine's own, invisible to the caller, and a run that straddles one has
/// to keep its base position and keep handing out consecutive encoder rows.
#[test]
fn a_split_anywhere_produces_the_same_streams() {
    let mut tokens = vec![TXT, TXT];
    tokens.extend(std::iter::repeat_n(VID, 12));
    tokens.push(TXT);
    tokens.extend(std::iter::repeat_n(IMG, 6));
    tokens.extend([TXT, TXT]);
    let grids = [grid(3, 2, 2), grid(1, 2, 3)];
    let whole = run(&tokens, &grids, 5);
    for at in 0..=tokens.len() {
        let split = run_split(&tokens, &grids, 5, at);
        assert_eq!(split.t, whole.t, "temporal stream, split at {at}");
        assert_eq!(split.h, whole.h, "row stream, split at {at}");
        assert_eq!(split.w, whole.w, "column stream, split at {at}");
        assert_eq!(split.map, whole.map, "splice map, split at {at}");
    }
}

// -- bounds and degenerate input ---------------------------------------------

/// A request owns its own slice of a shared grid list, so there is no base index
/// to get wrong: the window IS the slice it is handed.
#[test]
fn the_grid_slice_bounds_which_items_are_consumed() {
    let tokens = [IMG, IMG, IMG, IMG];
    let grids = [grid(1, 1, 4), grid(1, 2, 2), grid(1, 4, 1)];
    let out = run(&tokens, &grids[1..2], 0);
    assert_eq!(out.t, vec![0, 0, 0, 0], "consumed its one owned item");
    assert_eq!(out.h, vec![0, 0, 1, 1]);
    assert_eq!(out.w, vec![0, 1, 0, 1]);
}

/// Placeholders with no item left to describe them fall through to the text rule
/// rather than indexing past the end.
#[test]
fn placeholders_beyond_the_owned_grids_do_not_panic() {
    let tokens = [IMG, IMG, IMG, IMG, IMG];
    let out = run(&tokens, &[grid(1, 2, 2)], 0);
    assert_eq!(out.t.len(), 5, "one token in, one position out");
    assert_eq!(out.t, vec![0, 0, 0, 0, 2]);
    assert_eq!(out.map, vec![0, 1, 2, 3, -1]);
}

/// Every input length produces exactly that many rows on every stream. A stream
/// shorter than the chunk is an upload that reads past its staging.
#[test]
fn every_stream_always_matches_the_token_count() {
    for grids in [
        vec![],
        vec![grid(1, 2, 2)],
        vec![grid(3, 2, 2)],
        vec![grid(2, 1, 1), grid(1, 3, 3)],
    ] {
        let total: u32 = grids.iter().map(VisionGrid::pad_count).sum();
        let mut tokens = vec![TXT, TXT];
        tokens.extend(std::iter::repeat_n(IMG, total as usize));
        tokens.push(TXT);
        for out in [run(&tokens, &grids, 0), run_linear(&tokens, &grids, 0)] {
            assert_eq!(out.t.len(), tokens.len(), "temporal, grids={grids:?}");
            assert_eq!(out.h.len(), tokens.len(), "row, grids={grids:?}");
            assert_eq!(out.w.len(), tokens.len(), "column, grids={grids:?}");
            assert_eq!(out.map.len(), tokens.len(), "splice map, grids={grids:?}");
            assert_eq!(out.bound.len(), tokens.len(), "bound, grids={grids:?}");
        }
    }
}

/// A zero extent must not divide by zero or loop forever. It is clamped to one,
/// so the item costs one placeholder and one position.
#[test]
fn a_degenerate_grid_is_survivable() {
    let out = run(&[IMG, TXT], &[grid(0, 0, 0)], 7);
    assert_eq!(out.t, vec![7, 8]);
    assert_eq!(out.h, out.t);
    assert_eq!(out.w, out.t);
    assert_eq!(out.map, vec![0, -1]);
}

/// The splice map is a permutation of the encoder's rows, in order, with the
/// text tokens removed. Anything else re-uses or skips a row.
#[test]
fn the_splice_map_hands_out_every_row_once_and_in_order() {
    let grids = [grid(2, 2, 2), grid(1, 3, 3)];
    let total: u32 = grids.iter().map(VisionGrid::pad_count).sum();
    let mut tokens = vec![TXT];
    tokens.extend(std::iter::repeat_n(VID, 8));
    tokens.push(TXT);
    tokens.extend(std::iter::repeat_n(IMG, 9));
    tokens.push(TXT);
    let out = run(&tokens, &grids, 0);
    let used: Vec<i32> = out.map.into_iter().filter(|row| *row >= 0).collect();
    assert_eq!(used, (0..total as i32).collect::<Vec<_>>());
}

// -- the bound the uploader actually sends -----------------------------------

/// The split's fallback case, end to end through the clamp the uploader applies.
///
/// Every bound a chunk emits is its own token's sequence position, so none of
/// them can name a cache row that chunk has not written.
///
/// This is the invariant `clamp_bounds` used to enforce at the upload, and the
/// reason it no longer exists. An earlier revision widened a media item's tokens
/// to the item's LAST position, which for an item wider than a whole chunk -
/// the one case [`Walk::split_point`] cannot cut cleanly - put the first half's
/// bound past what that half had written; the clamp pulled it back. Attention is
/// causal on both position rules now, so the widening is gone and the clamp was
/// pure identity. `HybridStack` carries a `debug_assert!` saying so.
///
/// A twelve-token item in an eight-token chunk is the exact shape that used to
/// need the clamp, which is why it is still the fixture here.
#[test]
fn every_bound_a_chunk_emits_is_inside_that_chunk() {
    let tokens: Vec<TokenId> = std::iter::repeat_n(IMG, 12).collect();
    let grids = [grid(1, 3, 4)];
    let mut walk = Walk::new(&grids, pads(), 0, MediaPositions::Linear);
    assert_eq!(
        walk.split_point(&tokens, 8),
        8,
        "no cut clears the item, so the width wins"
    );

    let mut head = Streams::default();
    walk.chunk(&tokens[..8], &mut head);
    assert_eq!(
        head.bound,
        (0..8).collect::<Vec<_>>(),
        "the first half is bounded per token, not at the item's last position",
    );

    let mut tail = Streams::default();
    walk.chunk(&tokens[8..], &mut tail);
    assert_eq!(
        tail.bound,
        (8..12).collect::<Vec<_>>(),
        "and so is the second, which runs with the first already in the cache",
    );
}

/// The same holds for a chunk that mixes text and an item it holds whole, which
/// is every chunk the split can cut cleanly.
#[test]
fn a_chunk_holding_an_item_whole_is_bounded_per_token_too() {
    let tokens = [TXT, IMG, IMG, IMG, IMG, TXT];
    let grids = [grid(1, 2, 2)];
    let mut out = Streams::default();
    Walk::new(&grids, pads(), 0, MediaPositions::Linear).chunk(&tokens, &mut out);
    assert_eq!(out.bound, (0..6).collect::<Vec<_>>());
}
