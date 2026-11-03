//! CUDA's run of the shared `qwen4exp` low-precision prompt cases: the W4A4
//! routed expert stages (`q4e_nvfp4_a4`) and the W8A8 dense projections
//! (`q4e_mxfp8_a8`), on the block-scaled tensor-core instructions.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run. They need a kernel set built for a family-specific
//! architecture, which the `cuda/q4e_lowp` directory's manifest pins
//! (`sm_121a`); a device that cannot load that module, or a build without
//! kernels, is a skip, said on stderr.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! q4e_lowp -- --ignored --nocapture`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::lowp;

/// Opens device 0 if its kernel set carries the low-precision entries, or
/// says on stderr why the cases did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<Gpu> {
    match CudaBackend::new(0, CudaOptions::default()) {
        Ok(cuda) => {
            let device: Arc<dyn Backend> = cuda.clone();
            let gpu = Gpu::new(device, Box::new(cuda));
            if lowp::available(&gpu) {
                Some(gpu)
            } else {
                eprintln!(
                    "note: this kernel set or device has no block-scaled prompt kernels \
                     (needs SHISO_CUDA_COMPILE=1 and a family-specific architecture), skipping"
                );
                None
            }
        }
        Err(err) => {
            eprintln!("note: no CUDA device on this machine, skipping the GPU cases: {err}");
            None
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_nvfp4_quantizer_matches_the_host_bit_for_bit() {
    let Some(gpu) = open() else { return };
    lowp::the_nvfp4_quantizer_matches_the_host_bit_for_bit(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_mxfp8_quantizer_matches_the_host_bit_for_bit() {
    let Some(gpu) = open() else { return };
    lowp::the_mxfp8_quantizer_matches_the_host_bit_for_bit(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
#[expect(
    clippy::print_stderr,
    reason = "the measured errors are the numbers the tolerances and the quality call rest on"
)]
fn the_a4_stages_match_the_host_product_of_their_operands() {
    let Some(gpu) = open() else { return };
    let r = lowp::the_a4_stages_match_the_host_product_of_their_operands(&gpu);
    eprintln!(
        "w4a4: worst accumulation error {:e} of the absolute sum; against the weight-only \
         stages: silu(g)*u {:.4e}, down {:.4e} (relative L2)",
        r.accumulation, r.mid_vs_a16, r.down_vs_a16
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_a4_columns_do_not_depend_on_the_chunk() {
    let Some(gpu) = open() else { return };
    lowp::the_a4_columns_do_not_depend_on_the_chunk(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
#[expect(
    clippy::print_stderr,
    reason = "the measured errors are the numbers the tolerances and the quality call rest on"
)]
fn the_a8_gemm_matches_the_host_and_holds_its_rows() {
    let Some(gpu) = open() else { return };
    let r = lowp::the_a8_gemm_matches_the_host_and_holds_its_rows(&gpu);
    eprintln!(
        "w8a8: worst accumulation error {:e} of the absolute sum; against the weight-only \
         kernel {:.4e} (relative L2)",
        r.accumulation, r.vs_a16
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_a4_combine_sums_in_slot_order() {
    let Some(gpu) = open() else { return };
    lowp::the_a4_combine_sums_in_slot_order(&gpu);
}
