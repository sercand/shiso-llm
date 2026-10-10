use super::scheduler::*;

#[test]
#[expect(
    clippy::float_cmp,
    reason = "the two mu values come from different interpolation branches; equality would mean the shift is length-independent"
)]
fn empirical_mu_calculation() {
    let mu_1024 = compute_empirical_mu(4096, 4);
    assert!(mu_1024 > 0.0);
    let mu_512 = compute_empirical_mu(1024, 4);
    assert!(mu_512 > 0.0);
    assert_ne!(mu_1024, mu_512);
}

#[test]
fn scheduler_4_steps_monotonic_timesteps() {
    let scheduler = new_scheduler(4096, 4);
    assert_eq!(scheduler.timesteps.len(), 4);
    for i in 1..4 {
        assert!(scheduler.timesteps[i] < scheduler.timesteps[i - 1]);
    }
}
