//! The 3D splat generation contract.

#[cfg(test)]
mod tests {
    use crate::splat::{
        DEFAULT_GAUSSIANS, DEFAULT_STEPS, MAX_GAUSSIANS, MIN_GAUSSIANS, SplatFormat, SplatGenError,
        SplatJob, SplatLimits, SplatMode,
    };
    use shiso_core::ids::Seed;

    fn job() -> SplatJob {
        SplatJob {
            image: vec![0x89, b'P', b'N', b'G'],
            mode: SplatMode::Synthetic,
            seed: Seed::new(42),
            steps: 50,
            cfg_strength: 3.0,
            rescale_t: 3.0,
            num_gaussians: 262_144,
            crop_margin: 1.2,
            format: SplatFormat::Ply,
            refresh_every: 1,
        }
    }

    fn job_with(f: impl FnOnce(&mut SplatJob)) -> SplatJob {
        let mut job = job();
        f(&mut job);
        job
    }

    #[test]
    fn the_reference_defaults_are_accepted() {
        assert!(job().validate().is_ok());
    }

    #[test]
    fn gaussian_counts_are_a_power_grid_not_a_range() {
        for count in [MIN_GAUSSIANS, 65_536, MAX_GAUSSIANS] {
            assert!(
                job_with(|j| j.num_gaussians = count).validate().is_ok(),
                "{count}"
            );
        }
        for count in [MIN_GAUSSIANS - 32, 32_769, MAX_GAUSSIANS + 32, 0] {
            assert!(
                matches!(
                    job_with(|j| j.num_gaussians = count).validate(),
                    Err(SplatGenError::InvalidGaussians { .. })
                ),
                "{count}"
            );
        }
    }

    #[test]
    fn steps_outside_one_to_hundred_are_refused() {
        assert!(matches!(
            job_with(|j| j.steps = 0).validate(),
            Err(SplatGenError::InvalidSteps { steps: 0 })
        ));
        assert!(matches!(
            job_with(|j| j.steps = 101).validate(),
            Err(SplatGenError::InvalidSteps { steps: 101 })
        ));
    }

    #[test]
    fn non_finite_or_negative_knobs_are_refused() {
        for bad in [-0.1, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    job_with(|j| j.cfg_strength = bad).validate(),
                    Err(SplatGenError::InvalidCfg { .. })
                ),
                "cfg {bad}"
            );
            assert!(
                matches!(
                    job_with(|j| j.rescale_t = bad).validate(),
                    Err(SplatGenError::InvalidRescale { .. })
                ),
                "rescale {bad}"
            );
            assert!(
                matches!(
                    job_with(|j| j.crop_margin = bad).validate(),
                    Err(SplatGenError::InvalidCropMargin { .. })
                ),
                "margin {bad}"
            );
        }
    }

    #[test]
    fn a_model_with_no_defaults_named_still_has_them() {
        assert_eq!(SplatLimits::default().default_steps, DEFAULT_STEPS);
        assert_eq!(
            SplatLimits::default().default_num_gaussians,
            DEFAULT_GAUSSIANS
        );
    }

    #[test]
    fn the_mode_picks_the_crop_margin() {
        assert!((SplatMode::Synthetic.default_crop_margin() - 1.2).abs() < f32::EPSILON);
        assert!((SplatMode::Real.default_crop_margin() - 1.7).abs() < f32::EPSILON);
    }

    #[test]
    fn a_refresh_interval_is_one_to_the_ceiling() {
        use crate::splat::MAX_REFRESH_EVERY;
        for value in [1, 2, MAX_REFRESH_EVERY] {
            job_with(|job| job.refresh_every = value)
                .validate()
                .expect("in range");
        }
        for value in [0, MAX_REFRESH_EVERY + 1] {
            assert!(matches!(
                job_with(|job| job.refresh_every = value).validate(),
                Err(SplatGenError::InvalidRefresh { value: got }) if got == value
            ));
        }
    }
}
