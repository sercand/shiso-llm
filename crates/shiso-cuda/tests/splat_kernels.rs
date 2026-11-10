//! CUDA's run of the shared `splat` module cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens the device or says
//! why the cases did not run. The kernels are `kernels/cuda/splat` (one
//! translation unit, module `splat`); a case that fails here against a Metal
//! pass is a CUDA port bug, not a case bug.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! splat_kernels -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::splat;

/// Opens device 0, or says on stderr why the cases did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<Gpu> {
    match CudaBackend::new(0, CudaOptions::default()) {
        Ok(cuda) => {
            let device: Arc<dyn Backend> = cuda.clone();
            Some(Gpu::new(device, Box::new(cuda)))
        }
        Err(err) => {
            eprintln!("note: no CUDA device on this machine, skipping the GPU cases: {err}");
            None
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_splat_entry_resolves_on_the_device() {
    let Some(gpu) = open() else { return };
    let missing: Vec<&&str> = splat::ENTRIES
        .iter()
        .filter(|entry| {
            gpu.device()
                .max_block_threads(KernelRef::new("splat", entry))
                .is_none()
        })
        .collect();
    assert!(
        missing.is_empty(),
        "the splat image does not export {missing:?}"
    );
}

/// One ignored wrapper per shared case.
macro_rules! cases {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            #[ignore = "requires a CUDA device"]
            fn $name() {
                let Some(gpu) = open() else { return };
                splat::$name(&gpu);
            }
        )*
    };
}

cases!(
    activations_match_their_formulas,
    resize_bilinear_aligns_corners,
    merge2x2_takes_the_reference_order,
    window_attention_pads_with_the_pad_rows,
    window_attention_shifts_and_masks_the_regions,
    window_attention_at_the_production_window,
    deform_im2col_samples_bilinearly_with_the_mask,
    attention_matches_the_softmax_oracle,
    attention_weighs_a_heavy_key_as_its_copies,
    attention_survives_large_scores,
    attention_at_the_joint_shape,
    rope_norm_turns_and_scales_each_head,
);
