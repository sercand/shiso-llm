//! CUDA's run of the shared `dense_attn` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! These fail until the `dense` module exists, and they are supposed to. A
//! refusal naming the missing module is the baseline every kernel task in this
//! phase moves off; a *pass* here would mean the cases asserted nothing.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test dense_attn
//! -- --ignored`
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::attn;

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
fn rope_pairs_over_the_full_head() {
    let Some(gpu) = open() else { return };
    attn::rope_pairs_over_the_full_head(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn rope_over_the_rotated_slice() {
    let Some(gpu) = open() else { return };
    attn::rope_over_the_rotated_slice(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_value_norm_is_unweighted() {
    let Some(gpu) = open() else { return };
    attn::the_value_norm_is_unweighted(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn rope_without_a_norm_rotates_the_raw_head() {
    let Some(gpu) = open() else { return };
    attn::rope_without_a_norm_rotates_the_raw_head(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_rope_only_store_keeps_the_value_raw() {
    let Some(gpu) = open() else { return };
    attn::the_rope_only_store_keeps_the_value_raw(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn window_and_bound_select_the_keys() {
    let Some(gpu) = open() else { return };
    attn::window_and_bound_select_the_keys(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn scale_is_an_argument() {
    let Some(gpu) = open() else { return };
    attn::scale_is_an_argument(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn an_empty_span_contributes_zeros_rather_than_nan() {
    let Some(gpu) = open() else { return };
    attn::an_empty_span_contributes_zeros_rather_than_nan(&gpu);
}
