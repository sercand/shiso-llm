//! CUDA's run of the shared `diffusion` module cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run. The
//! kernels are `kernels/cuda/diffusion` (one translation unit, module
//! `diffusion`); a case that fails here against a Metal pass is a CUDA port
//! bug, not a case bug.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! diffusion_kernels -- --ignored`. Without `SHISO_CUDA_COMPILE=1` the kernel
//! crate embeds an empty registry and every case refuses at the first launch.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::diffusion::{int8, kquant, prefix, vae};

/// The module every diffusion entry lives under.
const MODULE: &str = "diffusion";

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

/// Asserts every entry resolves in the compiled `diffusion` image, naming the
/// ones that do not: the cheapest first thing to run on a new build.
fn assert_entries_resolve(gpu: &Gpu, entries: &[&'static str]) {
    let missing: Vec<&&str> = entries
        .iter()
        .filter(|entry| {
            gpu.device()
                .max_block_threads(KernelRef::new(MODULE, entry))
                .is_none()
        })
        .collect();
    assert!(
        missing.is_empty(),
        "the diffusion image does not export {missing:?}"
    );
}

/// One ignored wrapper per shared case of one case module.
macro_rules! cases {
    ($module:ident: $($name:ident),* $(,)?) => {
        $(
            #[test]
            #[ignore = "requires a CUDA device"]
            fn $name() {
                let Some(gpu) = open() else { return };
                $module::$name(&gpu);
            }
        )*
    };
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_shared_case_entry_resolves_on_the_device() {
    let Some(gpu) = open() else { return };
    assert_entries_resolve(&gpu, &int8::ENTRIES);
    assert_entries_resolve(&gpu, &kquant::ENTRIES);
    assert_entries_resolve(&gpu, &vae::ENTRIES);
    assert_entries_resolve(&gpu, &prefix::ENTRIES);
}

cases!(int8:
    hadamard256_matches_the_explicit_matrix,
    hadamard256_is_its_own_inverse,
    hadamard256_rotates_the_swiglu_row_it_forms,
    gemm_i8_matches_the_dequantized_product,
    gemm_i8_scales_each_output_row_not_each_input_column,
    fused_i8_is_the_unfused_composition,
);

cases!(kquant:
    gemm64_q2k_matches_dequant_oracle,
    gemm64_q3k_matches_dequant_oracle,
    gemm64_q4k_matches_dequant_oracle,
    gemm64_q5k_matches_dequant_oracle,
    gemm64_q6k_matches_dequant_oracle,
    gemm64_q8_0_matches_dequant_oracle,
    gemm64_fused_q2k_is_the_unfused_composition,
    gemm64_fused_q3k_is_the_unfused_composition,
    gemm64_fused_q4k_is_the_unfused_composition,
    gemm64_fused_q5k_is_the_unfused_composition,
    gemm64_fused_q6k_is_the_unfused_composition,
    gemm64_fused_q8_0_is_the_unfused_composition,
);

cases!(vae:
    channel_rms_inv_is_the_scaled_inverse_pixel_norm,
    conv2d_rms_is_the_conv_of_the_normalized_activated_field,
    conv2d_up2_is_the_conv_of_the_upsampled_field,
    dup_up_add_matches_the_reshaped_shortcut,
);

cases!(prefix:
    flash_sdpa_causal_masks_every_later_key,
    qk_norm_rope_normalizes_then_turns_adjacent_pairs,
    tanh_and_gelu_tanh_match_their_formulas,
    gqa_valid_keys_masks_pad_keys,
    rope_half_pins_trailing_rows_to_one_position,
);
