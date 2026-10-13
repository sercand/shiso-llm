//! Sample plans over decode-ordered timestamps.

use std::num::NonZeroU32;

use super::SamplePlan;
use crate::demux::Pts;
use crate::video::sample_indices;

fn per_second(ticks: u32) -> NonZeroU32 {
    NonZeroU32::new(ticks).expect("non-zero")
}

/// Thirty frames at 30 fps in `I P B B` decode order: each reference is
/// decoded ahead of the two frames shown before it.
fn reordered() -> Vec<Pts> {
    let mut order = vec![0];
    let mut anchor = 0;
    while anchor + 3 < 30 {
        order.extend([anchor + 3, anchor + 1, anchor + 2]);
        anchor += 3;
    }
    order.extend((anchor + 1)..30);
    assert_eq!(order.len(), 30);
    order.into_iter().map(Pts::new).collect()
}

/// The presentation times of the samples `plan` keeps, sorted.
fn kept_pts(plan: &SamplePlan, pts: &[Pts]) -> Vec<i64> {
    let mut kept: Vec<i64> = (0..pts.len())
        .filter(|&i| plan.keep(i))
        .map(|i| pts[i].ticks())
        .collect();
    kept.sort_unstable();
    kept
}

#[test]
fn keeps_display_positions_not_decode_positions() {
    let pts = reordered();
    let plan = SamplePlan::from_pts(&pts, per_second(30), 2.0, 768);
    assert!((plan.native_fps() - 30.0).abs() < 1e-9);
    // One second at 2 fps is two frames, raised to the four-frame floor.
    let positions = sample_indices(30, 30.0, 2.0, crate::DEFAULT_MIN_FRAMES, 768, 1);
    assert_eq!(positions, [0, 10, 19, 29]);
    // With one tick per frame, a display position IS its timestamp.
    let expected: Vec<i64> = positions
        .iter()
        .map(|&p| i64::try_from(p).expect("small"))
        .collect();
    assert_eq!(kept_pts(&plan, &pts), expected);
    assert_eq!(plan.kept(), 4);
    // Keeping by decode index instead would have kept sample 10, which shows
    // at tick 12 in this order; the frame shown at tick 10 is sample 11.
    assert_eq!(pts[10].ticks(), 12);
    assert!(!plan.keep(10));
    assert_eq!(pts[11].ticks(), 10);
    assert!(plan.keep(11));
}

#[test]
fn every_frame_at_or_above_the_native_rate() {
    let pts = reordered();
    let plan = SamplePlan::from_pts(&pts, per_second(30), 60.0, 768);
    assert_eq!(plan.kept(), 30);
    assert!((0..30).all(|i| plan.keep(i)));
    assert!(!plan.keep(30), "past the last sample");
}

#[test]
fn frame_cap() {
    let pts = reordered();
    let plan = SamplePlan::from_pts(&pts, per_second(30), 30.0, 8);
    assert_eq!(plan.kept(), 8);
    let kept = kept_pts(&plan, &pts);
    assert_eq!(kept.first(), Some(&0));
    assert_eq!(kept.last(), Some(&29), "the span is covered end to end");
}

#[test]
fn timescale_and_offset_set_the_rate() {
    // 10 fps in a 90 kHz timescale, starting at a non-zero time.
    let pts: Vec<Pts> = (0..20).map(|i| Pts::new(1_000 + i * 9_000)).collect();
    let plan = SamplePlan::from_pts(&pts, per_second(90_000), 5.0, 768);
    assert!((plan.native_fps() - 10.0).abs() < 1e-9);
    assert_eq!(plan.kept(), 10);
}

#[test]
fn no_measurable_rate_falls_back() {
    // Every timestamp equal: the rate is unknown, the sampler's default
    // applies, and the plan is still deterministic and in range.
    let pts = vec![Pts::new(7); 12];
    let plan = SamplePlan::from_pts(&pts, per_second(1000), 2.0, 768);
    assert!(plan.native_fps().abs() < f64::EPSILON);
    assert!(plan.kept() >= 1 && plan.kept() <= 12);

    let single = SamplePlan::from_pts(&[Pts::new(0)], per_second(1000), 2.0, 768);
    assert_eq!(single.kept(), 1);
    assert!(single.keep(0));

    let empty = SamplePlan::from_pts(&[], per_second(1000), 2.0, 768);
    assert_eq!(empty.kept(), 0);
    assert!(!empty.keep(0));
}

#[test]
fn extreme_timestamps_do_not_overflow() {
    let pts = [Pts::new(i64::MIN), Pts::new(i64::MAX), Pts::new(0)];
    let plan = SamplePlan::from_pts(&pts, per_second(1), 2.0, 768);
    assert!(plan.native_fps().is_finite());
    assert!(plan.kept() <= 3);
}
