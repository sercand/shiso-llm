//! The flow-map Euler walk and its starting noise.
//!
//! Each step asks the transformer for a velocity at time `t` while telling
//! it the time `t_ref` the step lands on, then moves the state by
//! `dt * velocity` with `dt = t_ref - t`. The state stays f32 across steps
//! while the starting noise carries only bf16 precision, as the reference's
//! does (it draws in f32 and rounds to bf16 once, then widens).
//!
//! [`run_euler`] is the loop the pipeline runs, parameterised over the
//! velocity so a test can drive it with a function it can integrate by hand;
//! [`euler_host`] is its host twin.

use half::bf16;
use shiso_core::ids::Seed;
use shiso_core::{DeviceBuffer, DeviceView};
use shiso_vision::generator::ImageGenError;

use crate::diffusion::noise::gaussian;
use crate::diffusion::ops::{Ops, Scratch};

/// The times of one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StepTimes {
    /// The time the velocity is asked at.
    pub t: f32,
    /// The time the step lands on, passed to the transformer as its target.
    pub t_ref: f32,
    /// `t_ref - t`, in f32 (negative: the walk runs from 1 down to 0).
    pub dt: f32,
}

/// The steps of a schedule of `n + 1` times.
#[must_use]
pub(crate) fn step_times(schedule: &[f32]) -> Vec<StepTimes> {
    schedule
        .windows(2)
        .map(|w| StepTimes {
            t: w[0],
            t_ref: w[1],
            dt: w[1] - w[0],
        })
        .collect()
}

/// `v` rounded to the nearest bf16 (ties to even) and widened back.
#[must_use]
pub(crate) fn round_bf16(v: f32) -> f32 {
    bf16::from_f32(v).to_f32()
}

/// `count` starting-noise values for `seed`: Gaussian, rounded to bf16, held
/// as f32.
#[must_use]
pub(crate) fn initial_noise(seed: Seed, count: usize) -> Vec<f32> {
    gaussian(seed, count).into_iter().map(round_bf16).collect()
}

/// Walks `latents` (`count` f32 values, updated in place) along `schedule`.
///
/// `velocity` returns the step's `count` velocities for the current state;
/// `after` runs once the step's update is queued (progress, a drain).
///
/// # Errors
/// Whatever `velocity`, `after` or the device refuse with.
pub(crate) fn run_euler<'o>(
    ops: &'o Ops,
    latents: &DeviceBuffer,
    count: usize,
    schedule: &[f32],
    velocity: &mut dyn FnMut(
        usize,
        &StepTimes,
        DeviceView<'_>,
    ) -> Result<Scratch<'o>, ImageGenError>,
    after: &mut dyn FnMut(usize, &StepTimes) -> Result<(), ImageGenError>,
) -> Result<(), ImageGenError> {
    for (index, step) in step_times(schedule).iter().enumerate() {
        let v = velocity(index, step, latents.view())?;
        ops.axpy(
            "flux3 euler",
            latents.view(),
            latents.view(),
            v.view(),
            count,
            step.dt,
        )?;
        drop(v);
        after(index, step)?;
    }
    Ok(())
}

#[cfg(test)]
/// The host twin of [`run_euler`]: `x += dt * v(x, step)` over the schedule.
pub(crate) fn euler_host(
    x: &mut [f32],
    schedule: &[f32],
    velocity: &mut dyn FnMut(&[f32], &StepTimes) -> Vec<f32>,
) {
    for step in step_times(schedule) {
        let v = velocity(x, &step);
        for (xi, vi) in x.iter_mut().zip(&v) {
            *xi += step.dt * vi;
        }
    }
}
