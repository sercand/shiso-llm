//! Metal's run of the shared `splat` module cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens the device or says
//! why the cases did not run. The kernels are `kernels/metal/splat/splat.metal`.

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::splat;
use shiso_metal::MetalBackend;

/// Opens the device for the shared cases, or says on stderr why they did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open_gpu() -> Option<Gpu> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the splat GPU cases");
        return None;
    }
    let metal = MetalBackend::new().expect("the embedded kernel library compiles");
    let device: Arc<dyn Backend> = metal.clone();
    Some(Gpu::new(device, Box::new(metal)))
}

#[test]
#[ignore = "requires a Metal device"]
fn every_splat_entry_resolves_as_a_pipeline() {
    let Some(gpu) = open_gpu() else { return };
    for entry in splat::ENTRIES {
        assert!(
            gpu.device()
                .max_block_threads(KernelRef::new("splat", entry))
                .is_some(),
            "{entry} does not resolve"
        );
    }
}

/// One ignored wrapper per shared case.
macro_rules! splat_cases {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            #[ignore = "requires a Metal device"]
            fn $name() {
                let Some(gpu) = open_gpu() else { return };
                splat::$name(&gpu);
            }
        )*
    };
}

splat_cases!(
    activations_match_their_formulas,
    resize_bilinear_aligns_corners,
    merge2x2_takes_the_reference_order,
    window_attention_pads_with_the_pad_rows,
    window_attention_shifts_and_masks_the_regions,
    window_attention_at_the_production_window,
    deform_im2col_samples_bilinearly_with_the_mask,
    attention_matches_the_softmax_oracle,
    attention_survives_large_scores,
    attention_at_the_joint_shape,
);
