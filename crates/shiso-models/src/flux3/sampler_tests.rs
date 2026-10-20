//! The flow-map Euler walk: its host twin against a hand integration, the
//! device loop against the twin, and the noise's precision contract.

#![expect(
    clippy::float_cmp,
    reason = "the schedule's entries are copied through, not recomputed, so they compare exactly"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "loop indices in a synthetic state are small"
)]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_core::ids::Seed;

use super::sampler::{StepTimes, euler_host, initial_noise, round_bf16, run_euler, step_times};
use super::scheduler::{ALPHA, schedule};
use crate::diffusion::noise::gaussian;
use crate::diffusion::ops::Ops;

#[test]
fn sampler_applies_flow_map_euler() {
    let times = schedule(16, ALPHA);
    let steps = step_times(&times);
    assert_eq!(steps.len(), 16);
    // `t_ref` is the next schedule entry, `t` the current one, and `dt`
    // their difference: the last step lands exactly on zero.
    for (i, step) in steps.iter().enumerate() {
        assert_eq!(step.t, times[i]);
        assert_eq!(step.t_ref, times[i + 1]);
        assert_eq!(step.dt, times[i + 1] - times[i]);
        assert!(step.dt < 0.0, "the walk runs from noise down to the image");
    }
    assert_eq!(steps[0].t, 1.0);
    assert_eq!(steps[15].t_ref, 0.0);

    // A velocity that does not depend on the state integrates to
    // `c * (t_end - t_start) = -c`, however the steps are spaced.
    let mut x = vec![0.5f32, -2.0, 3.25];
    euler_host(&mut x, &times, &mut |_, _| vec![2.0, 2.0, 2.0]);
    for (got, want) in x.iter().zip([-1.5f32, -4.0, 1.25]) {
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }

    // The model is asked at the current `t` with the next one as its target,
    // in order, and sees the state the previous step left.
    let mut seen: Vec<(f32, f32)> = Vec::new();
    let mut x = vec![1.0f32];
    euler_host(&mut x, &times, &mut |state, step| {
        seen.push((step.t, step.t_ref));
        vec![state[0]]
    });
    assert_eq!(
        seen,
        times.windows(2).map(|w| (w[0], w[1])).collect::<Vec<_>>()
    );
    // dx = x dt integrates to x0 * exp(t_end - t_start) = e^-1 (Euler's own
    // error aside).
    assert!((x[0] - (-1.0f32).exp()).abs() < 0.05, "{}", x[0]);
}

#[test]
fn noise_is_bf16_rounded_f32_state() {
    let noise = initial_noise(Seed::new(42), 4096);
    assert_eq!(noise.len(), 4096);
    let raw = gaussian(Seed::new(42), 4096);
    for (n, r) in noise.iter().zip(&raw) {
        assert_eq!(n.to_bits() & 0xFFFF, 0, "{n} carries more than bf16's bits");
        assert_eq!(*n, round_bf16(*r));
    }
    assert!(
        noise.iter().zip(&raw).any(|(n, r)| n != r),
        "rounding must change something"
    );
    // The state is f32: one Euler step with a velocity leaves the bf16 grid.
    let mut x = noise.clone();
    euler_host(&mut x, &schedule(2, ALPHA), &mut |s, _| {
        s.iter().map(|v| v * 0.37).collect()
    });
    assert!(
        x.iter().any(|v| v.to_bits() & 0xFFFF != 0),
        "the state must stay f32 across steps"
    );
}

#[test]
fn same_seed_same_noise_and_other_seeds_differ() {
    assert_eq!(
        initial_noise(Seed::new(7), 256),
        initial_noise(Seed::new(7), 256)
    );
    assert_ne!(
        initial_noise(Seed::new(7), 256),
        initial_noise(Seed::new(8), 256)
    );
}

#[expect(
    clippy::print_stderr,
    reason = "the device cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn device() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the sampler device case");
        return None;
    }
    let backend: Arc<dyn Backend> =
        shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    Some(backend)
}

#[test]
#[ignore = "requires a Metal device"]
fn the_device_loop_matches_its_host_twin() {
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let times = schedule(16, ALPHA);
    let start: Vec<f32> = (0..1000)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    // A velocity that depends on both the time and the state, so a wrong
    // `dt`, a stale state or a wrong `t` each change the answer.
    let law = |x: f32, step: &StepTimes| x * (0.5 + step.t) - step.t_ref;

    let mut want = start.clone();
    euler_host(&mut want, &times, &mut |state, step| {
        state.iter().map(|&x| law(x, step)).collect()
    });

    let latents = ops.upload_f32("test latents", &start).expect("upload");
    let calls = std::cell::Cell::new(0usize);
    run_euler(
        &ops,
        &latents,
        start.len(),
        &times,
        &mut |index, step, state| {
            assert_eq!(index, calls.get(), "steps arrive in order");
            let host = ops.download("test state", state, start.len())?;
            let v: Vec<f32> = host.iter().map(|&x| law(x, step)).collect();
            ops.upload_scratch("test velocity", &v)
        },
        &mut |index, _| {
            assert_eq!(index, calls.get());
            calls.set(calls.get() + 1);
            Ok(())
        },
    )
    .expect("the loop runs");
    assert_eq!(calls.get(), 16);
    let got = ops
        .download("test result", latents.view(), start.len())
        .expect("download");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 1e-4 * w.abs().max(1.0), "[{i}] {g} vs {w}");
    }
}
