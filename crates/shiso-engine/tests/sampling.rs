//! Bit-exactness tests for the ported sampler (task-06 "Tests" items 1-5).
//!
//! Expected values come from the algorithm the port copies - xorshift64\* for the
//! stream, a cumulative walk for the pick - and not from the Rust, so a change
//! that would alter a sampled token fails here rather than surfacing as a
//! different generation forty tokens later.

use std::num::NonZeroU32;

use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_engine::{argmax, build_probabilities, rng_f32, rng_next, sample_logits};

/// State replacement for a zero seed: the golden-ratio constant.
const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;
/// xorshift64\* output multiplier.
const MULTIPLIER: u64 = 0x2545_f491_4f6c_dd1d;
/// `2^24`, the resolution of the `[0, 1)` draw.
const DRAW_SCALE: u32 = 16_777_216;

/// The state one advance produces, written out here so the stream assertions do
/// not lean on `shiso_core::Seed`'s own implementation.
fn next_state(state: u64) -> u64 {
    let mut x = if state == 0 { GOLDEN } else { state };
    x ^= x >> 12;
    x ^= x.wrapping_shl(25);
    x ^= x >> 27;
    x
}

/// The `[0, 1)` draw of one state, as a count of `2^-24` steps.
///
/// An integer count is the stronger check: every such value is exactly
/// representable in `f32`, so multiplying the draw back by `2^24` recovers it
/// with no rounding at all.
fn draw_steps(state: u64) -> u32 {
    let scrambled = next_state(state).wrapping_mul(MULTIPLIER);
    u32::try_from((scrambled >> 40) & 0x00ff_ffff).expect("the mask leaves 24 bits")
}

/// A request with no stage filters: temperature 1, no top-k, no cutoffs.
fn neutral() -> SamplingParams {
    SamplingParams::default()
}

/// Greedy is a temperature of zero, which the port treats as "no division at
/// all" rather than a division by a very small number.
fn greedy() -> SamplingParams {
    let mut params = neutral();
    params.temperature = 0.0;
    params
}

/// A row narrowed to `k` candidates by the top-k stage alone.
fn narrowed(k: u32) -> SamplingParams {
    let mut params = neutral();
    params.top_k = NonZeroU32::new(k);
    params
}

#[test]
fn the_stream_is_xorshift64_star_over_its_state() {
    for state in [1_u64, 42, 12_345] {
        let mut seed = Seed::new(state);
        let drawn = rng_next(&mut seed);
        let scrambled = next_state(state);
        assert_eq!(
            seed.get(),
            scrambled,
            "state {state} advances by the xorshift"
        );
        assert_eq!(
            drawn,
            scrambled.wrapping_mul(MULTIPLIER),
            "the output is the scrambled state, not the state itself"
        );
    }
}

#[test]
fn a_zero_seed_is_replaced_rather_than_stuck_at_zero() {
    let mut seed = Seed::new(0);
    let first = rng_next(&mut seed);
    // A zero state would stay zero forever, so the port substitutes the golden
    // ratio before scrambling: the draw of seed 0 is the draw of .
    assert_eq!(
        seed.get(),
        next_state(0),
        "a zero seed is replaced, not stuck"
    );
    assert_eq!(first, next_state(0).wrapping_mul(MULTIPLIER));
}

#[expect(
    clippy::float_cmp,
    reason = "exactness is the assertion: the draw is 24 bits scaled by 2^24, so the product is an integer the float path holds exactly, and an epsilon here would hide the low-bit drift this test exists to catch"
)]
#[test]
fn the_float_draw_is_the_top_24_bits_over_two_to_the_24th() {
    for state in [1_u64, 42, 12_345, 0] {
        let mut seed = Seed::new(state);
        let drawn = rng_f32(&mut seed);
        assert!((0.0..1.0).contains(&drawn), "the draw is in [0, 1)");
        assert_eq!(
            f64::from(drawn) * f64::from(DRAW_SCALE),
            f64::from(draw_steps(state)),
            "state {state} draws exactly the 24 high bits of its output"
        );
    }
}

#[test]
fn a_uniform_row_picks_the_bucket_the_draw_lands_in() {
    // Equal logits at temperature 1 make every probability exactly 1/n, so the
    // walk is `ceil(u * n) - 1` with no rounding wiggle room anywhere.
    const N: u32 = 8;
    for state in [1_u64, 42, 12_345] {
        let logits = vec![0.5_f32; N as usize];
        let picked = sample_logits(&logits, &neutral(), &mut Seed::new(state));
        let steps = draw_steps(state);
        assert!(steps > 0, "the fixture draws must not land exactly on zero");
        let expected = (steps * N).div_ceil(DRAW_SCALE) - 1;
        assert_eq!(
            picked,
            TokenId::new(expected),
            "state {state} must land in the bucket its own draw names"
        );
    }
}

#[test]
fn a_stochastic_sample_advances_the_stream_exactly_once() {
    for state in [1_u64, 42, 12_345] {
        let logits = vec![0.5_f32; 8];
        let mut seed = Seed::new(state);
        let _ = sample_logits(&logits, &neutral(), &mut seed);
        assert_eq!(
            seed.get(),
            next_state(state),
            "one sample, one draw: the state is one xorshift along and no further"
        );
    }
}

#[test]
fn greedy_is_argmax_and_does_not_touch_the_stream() {
    let logits = vec![0.25_f32, 9.5, 0.75];
    for state in [7_u64, 99] {
        let mut seed = Seed::new(state);
        assert_eq!(
            sample_logits(&logits, &greedy(), &mut seed),
            TokenId::new(1),
            "the largest logit wins outright"
        );
        assert_eq!(seed.get(), state, "greedy sampling must not draw at all");
    }
}

#[test]
fn top_k_keeps_the_k_largest_and_zeroes_the_rest() {
    let logits = vec![5.0_f32, 4.0, 3.0, 2.0, 1.0];
    let mut probs = vec![0_f32; logits.len()];
    assert!(build_probabilities(&logits, &narrowed(2), &mut probs));
    assert_eq!(
        &probs[2..],
        &[0.0, 0.0, 0.0],
        "everything past top_k is gone"
    );
    assert!(
        probs[0] > probs[1] && probs[1] > 0.0,
        "the larger logit keeps the larger share"
    );
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "the surviving mass is renormalised, got {sum}"
    );
}

#[test]
fn a_top_k_beyond_the_vocabulary_is_clamped_not_wrapped() {
    let logits = vec![1.0_f32, 2.0, 3.0];
    let mut probs = vec![0_f32; logits.len()];
    assert!(build_probabilities(&logits, &narrowed(4_096), &mut probs));
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "a clamped top_k still sums to 1, got {sum}"
    );
    assert!(probs.iter().all(|p| *p >= 0.0), "no entry goes negative");
}

#[expect(
    clippy::float_cmp,
    reason = "a removed entry is written rather than computed, so it is exactly zero; a denormal or a surviving NaN is the failure this asserts"
)]
#[test]
fn non_finite_entries_are_removed_from_the_row() {
    let logits = vec![f32::NAN, f32::INFINITY, 1.0, 0.5];
    let mut probs = vec![0_f32; logits.len()];
    assert!(build_probabilities(&logits, &neutral(), &mut probs));
    assert_eq!(probs[0], 0.0, "a NaN entry is removed from the row");
    assert_eq!(probs[1], 0.0, "an infinite entry is removed from the row");
    assert!(
        probs[2] > 0.0 && probs[3] > 0.0,
        "the finite pair survives and carries the whole mass"
    );
}

#[test]
fn ties_resolve_to_the_lower_id_and_a_row_of_neg_infinity_still_picks() {
    assert_eq!(argmax(&[1.0, 7.0, 7.0, 3.0]), TokenId::new(1));
    assert_eq!(argmax(&[f32::NEG_INFINITY; 5]), TokenId::new(0));
}

#[test]
fn the_argmax_scan_covers_a_row_that_is_not_a_multiple_of_eight() {
    let mut logits = vec![0_f32; 11];
    logits[9] = 1.5;
    assert_eq!(argmax(&logits), TokenId::new(9));
    logits[10] = 2.5;
    assert_eq!(
        argmax(&logits),
        TokenId::new(10),
        "the final element of the tail is not dropped"
    );
}
