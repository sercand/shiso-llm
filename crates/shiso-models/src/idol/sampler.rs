//! The flow sampler: the time grid, the guidance combination and the Euler
//! step.
//!
//! Three pieces of arithmetic that are cheap to get subtly wrong and expensive
//! to notice, because a wrong grid still produces a plausible object. They live
//! here, on the host, away from the device graph, so they can be tested against
//! hand-computed values.

/// The model's timestep is this many times the sampler's `t`.
///
/// The denoiser was trained with timesteps in `0..=1000` while the flow
/// sampler integrates `t` in `0..=1`; the factor is the whole of the bridge.
pub const TIMESTEP_SCALE: f32 = 1000.0;

/// The time grid: `steps + 1` points from 1 down to 0, shifted towards high
/// noise by `rescale_t`.
///
/// The shift is `rescale_t * t / (1 + (rescale_t - 1) * t)`, which maps
/// `[0, 1]` onto itself, fixes both ends, and spends more of the steps where
/// the object is still mostly noise. `rescale_t == 1` is the plain grid.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "the step index and count are bounded by the request's 1..=100 step ceiling"
)]
pub fn time_grid(steps: u32, rescale_t: f32) -> Vec<f32> {
    let steps = steps as usize;
    (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f32 / steps as f32;
            rescale_t * t / (1.0 + (rescale_t - 1.0) * t)
        })
        .collect()
}

/// The guided velocity: `(1 + cfg) * v_cond - cfg * v_neg`.
///
/// With `cfg == 0` this is the conditional velocity alone, which is why the
/// caller does not need a branch: a guidance strength of zero is a legitimate
/// request, not a missing one.
///
/// # Panics
///
/// Panics when the two passes or the output disagree on shape, which is a
/// graph bug rather than a request's.
pub fn guided_velocity(v_cond: &[f32], v_neg: &[f32], cfg: f32, out: &mut [f32]) {
    assert_eq!(
        v_cond.len(),
        v_neg.len(),
        "the two velocity passes must be the same shape"
    );
    assert_eq!(
        v_cond.len(),
        out.len(),
        "the output must be the size of one velocity pass"
    );
    let (positive, negative) = (1.0 + cfg, cfg);
    for ((out, cond), neg) in out.iter_mut().zip(v_cond).zip(v_neg) {
        *out = positive * cond - negative * neg;
    }
}

/// One Euler step: `sample -= velocity * (t - t_prev)`.
///
/// `t` is the current grid point and `t_prev` the next one, which is smaller,
/// so the subtraction moves the sample towards the data.
///
/// # Panics
///
/// Panics when the velocity is not the sample's shape, which is a graph bug
/// rather than a request's.
pub fn euler_step(sample: &mut [f32], velocity: &[f32], t: f32, t_prev: f32) {
    assert_eq!(
        sample.len(),
        velocity.len(),
        "the velocity must be the sample's shape"
    );
    let dt = t - t_prev;
    for (value, velocity) in sample.iter_mut().zip(velocity) {
        *value -= velocity * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_spans_one_to_zero_inclusive() {
        let grid = time_grid(50, 3.0);
        assert_eq!(grid.len(), 51);
        assert!((grid[0] - 1.0).abs() < 1e-6);
        assert!(grid[50].abs() < 1e-6);
        assert!(grid.windows(2).all(|pair| pair[0] > pair[1]), "descending");
    }

    #[test]
    fn the_shift_matches_the_reference_grid() {
        // The reference's own numbers for 50 steps at rescale 3, read from
        // `np.linspace(1, 0, 51)` through `3t / (1 + 2t)`.
        let grid = time_grid(50, 3.0);
        for (index, want) in [(0usize, 1.0f32), (1, 0.993_243_2), (25, 0.75), (50, 0.0)] {
            assert!(
                (grid[index] - want).abs() < 1e-6,
                "grid[{index}] = {}, want {want}",
                grid[index]
            );
        }
    }

    #[test]
    fn a_unit_rescale_is_the_plain_grid() {
        let grid = time_grid(4, 1.0);
        let expected = [1.0, 0.75, 0.5, 0.25, 0.0];
        for (got, want) in grid.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn the_shift_moves_points_towards_high_noise() {
        // Every interior point of a shifted grid is later (closer to 1) than
        // the plain grid's, which is what "more steps at high noise" means.
        let shifted = time_grid(10, 3.0);
        let plain = time_grid(10, 1.0);
        for (s, p) in shifted.iter().zip(&plain).skip(1).take(9) {
            assert!(s > p, "shifted {s} should be above plain {p}");
        }
    }

    #[test]
    fn zero_guidance_is_the_conditional_velocity() {
        let cond = [1.0, -2.0, 3.0];
        let neg = [9.0, 9.0, 9.0];
        let mut out = [0.0; 3];
        guided_velocity(&cond, &neg, 0.0, &mut out);
        for (got, want) in out.iter().zip(cond) {
            assert!((got - want).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn guidance_is_the_reference_combination() {
        let cond = [1.0, 2.0];
        let neg = [0.5, 0.5];
        let mut out = [0.0; 2];
        guided_velocity(&cond, &neg, 3.0, &mut out);
        // (1 + 3) * cond - 3 * neg
        assert!((out[0] - (4.0 * 1.0 - 3.0 * 0.5)).abs() < 1e-6);
        assert!((out[1] - (4.0 * 2.0 - 3.0 * 0.5)).abs() < 1e-6);
    }

    #[test]
    fn an_euler_step_moves_against_the_velocity() {
        let mut sample = [1.0, 2.0];
        euler_step(&mut sample, &[0.5, -0.5], 1.0, 0.5);
        // dt = 0.5: sample -= velocity * 0.5
        assert!((sample[0] - 0.75).abs() < 1e-6);
        assert!((sample[1] - 2.25).abs() < 1e-6);
    }

    #[test]
    fn the_whole_grid_integrates_a_constant_velocity_exactly() {
        // A constant velocity of 2 moves the sample by 2 over the full grid,
        // whatever the shift: the steps sum to t_last - t_first = -1.
        let grid = time_grid(7, 3.0);
        let mut sample = [0.0];
        for pair in grid.windows(2) {
            euler_step(&mut sample, &[2.0], pair[0], pair[1]);
        }
        assert!((sample[0] + 2.0).abs() < 1e-5, "got {}", sample[0]);
    }

    #[test]
    fn the_timestep_scale_is_a_thousand() {
        assert!((TIMESTEP_SCALE * 0.5 - 500.0).abs() < f32::EPSILON);
    }
}
