//! CUDA's run of the shared mixed i-quant cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! The matvec cases reach the device through the by-name spellings - one
//! symbol per `(r1ptg, nxpsg)` pair, launched with an empty constant table -
//! so a missing instantiation here fails in the module's symbol lookup rather
//! than in a refused constant.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test dense_iq
//! -- --ignored`
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::iq;

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
fn iq_matvec_matches_the_oracle_dot() {
    let Some(gpu) = open() else { return };
    iq::iq_matvec_matches_the_oracle_dot(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_iq_matvec_instantiation_agrees_with_the_oracle() {
    let Some(gpu) = open() else { return };
    iq::every_iq_matvec_instantiation_agrees_with_the_oracle(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn iq1_m_get_rows_matches_the_oracle() {
    let Some(gpu) = open() else { return };
    iq::iq1_m_get_rows_matches_the_oracle(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn iq_tables_match_the_host_copy() {
    let Some(gpu) = open() else { return };
    iq::iq_tables_match_the_host_copy(&gpu);
}
