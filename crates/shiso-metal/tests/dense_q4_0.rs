//! Metal's run of the shared `dense_q4_0` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --features metal --test dense_q4_0 --
//! --ignored`
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::q4_0;
use shiso_metal::MetalBackend;

/// Opens the device, or says on stderr why the cases did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<Gpu> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the GPU cases");
        return None;
    }
    let metal = MetalBackend::new().expect("the embedded kernel library compiles");
    let device: Arc<dyn Backend> = metal.clone();
    Some(Gpu::new(device, Box::new(metal)))
}

#[test]
#[ignore = "requires a Metal device"]
fn q4_0_matvec_matches_the_oracle_dot() {
    let Some(gpu) = open() else { return };
    q4_0::q4_0_matvec_matches_the_oracle_dot(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn q4_0_prefill_gemm_matches_a_triple_loop() {
    let Some(gpu) = open() else { return };
    q4_0::q4_0_prefill_gemm_matches_a_triple_loop(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn every_q4_0_matvec_instantiation_agrees_with_the_oracle() {
    let Some(gpu) = open() else { return };
    q4_0::every_q4_0_matvec_instantiation_agrees_with_the_oracle(&gpu);
}
