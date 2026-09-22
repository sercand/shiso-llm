//! CUDA's run of the shared `dense_misc` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! These fail until the `dense` module exists, and they are supposed to. A
//! refusal naming the missing module is the baseline every kernel task in this
//! phase moves off; a *pass* here would mean the cases asserted nothing.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test dense_misc
//! -- --ignored`. Without `SHISO_CUDA_COMPILE=1` the kernel crate embeds an
//! empty registry and the refusal says so, which is a different message from
//! the one a real build gives.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::misc;

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
fn the_gated_gelu_is_the_tanh_approximation_times_the_up_row() {
    let Some(gpu) = open() else { return };
    misc::the_gated_gelu_is_the_tanh_approximation_times_the_up_row(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_gated_gelu_saturates_on_a_large_gate_instead_of_overflowing_to_nan() {
    let Some(gpu) = open() else { return };
    misc::the_gated_gelu_saturates_on_a_large_gate_instead_of_overflowing_to_nan(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_scaled_close_scales_the_sum_before_it_stores_and_before_it_reduces() {
    let Some(gpu) = open() else { return };
    misc::the_scaled_close_scales_the_sum_before_it_stores_and_before_it_reduces(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_softcap_squashes_a_row_into_the_cap() {
    let Some(gpu) = open() else { return };
    misc::the_softcap_squashes_a_row_into_the_cap(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_row_scale_multiplies_in_place_and_stops_at_its_bound() {
    let Some(gpu) = open() else { return };
    misc::the_row_scale_multiplies_in_place_and_stops_at_its_bound(&gpu);
}
