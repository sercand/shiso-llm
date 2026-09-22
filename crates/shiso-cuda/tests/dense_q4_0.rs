//! CUDA's run of the shared `dense_q4_0` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! These fail until the `generic` module exists, and they are supposed to. The
//! matvec is also the one path in the set that asks for a function constant,
//! which this backend refuses by design rather than launching the wrong
//! variant - so a refusal here may name either the missing module or the
//! constants, and the two are different work.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test dense_q4_0
//! -- --ignored`
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::q4_0;

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
fn q4_0_matvec_matches_the_oracle_dot() {
    let Some(gpu) = open() else { return };
    q4_0::q4_0_matvec_matches_the_oracle_dot(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn q4_0_prefill_gemm_matches_a_triple_loop() {
    let Some(gpu) = open() else { return };
    q4_0::q4_0_prefill_gemm_matches_a_triple_loop(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_q4_0_matvec_instantiation_agrees_with_the_oracle() {
    let Some(gpu) = open() else { return };
    q4_0::every_q4_0_matvec_instantiation_agrees_with_the_oracle(&gpu);
}
