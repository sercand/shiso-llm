//! The schedule against its formula and the reference's `timesteps`.
#![expect(
    clippy::cast_precision_loss,
    reason = "a loop index below five, exact in f64"
)]

use super::scheduler::{ALPHA, DEFAULT_STEPS, schedule};

#[test]
fn schedule_16_alpha_6_93() {
    assert_eq!(DEFAULT_STEPS, 16);
    let s = schedule(16, 6.93);
    assert_eq!(s.len(), 17);
    assert_eq!(s[0].to_bits(), 1.0_f32.to_bits(), "starts at pure noise");
    assert_eq!(
        s[16].to_bits(),
        0.0_f32.to_bits(),
        "ends at the clean image"
    );
    let t = 15.0_f64 / 16.0;
    let want = 6.93 * t / (1.0 + 5.93 * t);
    assert!(
        (f64::from(s[1]) - want).abs() < 1e-6,
        "step 1 is {} not {want}",
        s[1]
    );
    for pair in s.windows(2) {
        assert!(pair[0] > pair[1], "strictly decreasing: {pair:?}");
    }
    assert!((ALPHA - 6.93).abs() < f32::EPSILON);
}

#[test]
fn the_bend_spends_steps_near_noise() {
    // With alpha > 1 the first step is a small drop from 1: the uniform
    // ladder would have dropped 1/16.
    let s = schedule(16, ALPHA);
    assert!(s[0] - s[1] < 1.0 / 16.0, "first step {}", s[0] - s[1]);
    // alpha = 1 is the plain uniform ladder.
    let flat = schedule(4, 1.0);
    for (i, v) in flat.iter().enumerate() {
        assert!(
            (f64::from(*v) - (1.0 - i as f64 / 4.0)).abs() < 1e-7,
            "{flat:?}"
        );
    }
}

#[test]
fn a_zero_step_walk_is_just_the_start() {
    assert_eq!(schedule(0, ALPHA), vec![1.0]);
}
