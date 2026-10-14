//! Where a prompt carrying media may be cut without changing its answer.
//!
//! What is pinned is that every reader of one prompt - a cold read, a resume
//! from a sibling's checkpoint, a batched tail - agrees on the cut points: the
//! grid restarts at every item's end, and no cut lands inside an item.

use shiso_core::media::{MediaSpan, VisionGrid, run_floor, run_origin};

const GRID: u32 = 64;

/// One still image of `rows` placeholder tokens starting at `start`.
fn picture(start: u32, rows: u32) -> MediaSpan {
    MediaSpan::new(start, VisionGrid::new(1, 1, rows))
}

#[test]
fn a_text_prompt_is_cut_on_the_plain_grid() {
    assert_eq!(run_origin(490, &[]), 0);
    assert_eq!(run_floor(490, &[], GRID), 448);
    assert_eq!(run_floor(448, &[], GRID), 448);
    assert_eq!(run_floor(63, &[], GRID), 0);
}

#[test]
fn the_grid_restarts_where_a_picture_ends() {
    // A screenshot at 307..487, its question parting three tokens later: the
    // plain grid point 448 lies inside it, so the cut is the picture's end.
    let spans = [picture(307, 180)];
    assert_eq!(run_origin(490, &spans), 487);
    assert_eq!(run_floor(490, &spans, GRID), 487);
    assert_eq!(run_floor(487, &spans, GRID), 487);
    // Past the end the runs count from it, not from the prompt's start.
    assert_eq!(run_floor(487 + 64 + 5, &spans, GRID), 487 + 64);
}

#[test]
fn a_point_inside_a_picture_falls_below_its_start() {
    let spans = [picture(307, 180)];
    assert_eq!(run_floor(400, &spans, GRID), 256);
    // The picture's own start is a cut only when it is on the grid below it.
    assert_eq!(run_floor(307, &spans, GRID), 256);
    assert_eq!(run_floor(320, &[picture(320, 10)], GRID), 320);
}

#[test]
fn a_picture_right_after_another_counts_from_the_first_ones_end() {
    let spans = [picture(100, 50), picture(160, 40)];
    assert_eq!(run_origin(170, &spans), 150);
    // 170 is inside the second picture: below its start the grid counts from
    // the first picture's end, which is the highest cut there.
    assert_eq!(run_floor(170, &spans, GRID), 150);
    assert_eq!(run_floor(205, &spans, GRID), 200);
}
