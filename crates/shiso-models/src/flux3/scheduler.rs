//! The flow-map time schedule.
//!
//! A distilled FLUX.3 walks `steps` Euler steps along times that start as a
//! uniform ladder from 1 to 0 and are bent by `a*t / (1 + (a-1)*t)`, which
//! spends more of the steps near noise. Each step also tells the model the
//! time it is jumping to, which is simply the next entry of this schedule.

/// The bend the distilled checkpoint was trained with.
pub(crate) const ALPHA: f32 = 6.93;

/// The default number of Euler steps of the distilled checkpoint.
pub(crate) const DEFAULT_STEPS: u32 = 16;

/// The `steps + 1` times of a walk, from `1.0` (pure noise) to `0.0`.
///
/// Computed in `f64` and rounded once, so the endpoints are exact and the
/// values do not drift with the step count. A zero step count answers just
/// `[1.0]`; refusing it is the caller's job, since the request, not this
/// table, is what is wrong.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the schedule is an f32 quantity; the f64 intermediate only avoids accumulated rounding"
)]
pub(crate) fn schedule(steps: u32, alpha: f32) -> Vec<f32> {
    let alpha = f64::from(alpha);
    (0..=steps)
        .map(|i| {
            let t = if steps == 0 {
                1.0
            } else {
                1.0 - f64::from(i) / f64::from(steps)
            };
            (alpha * t / (1.0 + (alpha - 1.0) * t)) as f32
        })
        .collect()
}
