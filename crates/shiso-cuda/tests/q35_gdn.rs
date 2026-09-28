//! CUDA's run of the shared `q35_gdn` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test q35_gdn --
//! --ignored`.
//!
//! One test here is NOT a thin wrapper, and it guards the one way this file can
//! report nine passes having launched nothing. Two of the shared cases open
//! with `RECURRENT_VARIANTS.iter().find(viable) else { return }`, and `viable`
//! asks `Backend::max_block_threads`. The trait's default answers `None`, which
//! means "no opinion", so a backend that does not implement it makes both
//! cases return early and pass. `CudaBackend` implements it now; this asserts
//! that it does, by name, so the day it stops the failure is here rather than
//! in two cases that go quiet.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen35::gdn as cases;

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
fn conv_resumes_from_the_window_a_previous_call_left() {
    let Some(gpu) = open() else { return };
    cases::conv_resumes_from_the_window_a_previous_call_left(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn l2norm_normalizes_each_head_row_in_place() {
    let Some(gpu) = open() else { return };
    cases::l2norm_normalizes_each_head_row_in_place(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn gates_apply_the_softplus_guard_and_the_sigmoid() {
    let Some(gpu) = open() else { return };
    cases::gates_apply_the_softplus_guard_and_the_sigmoid(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn out_gate_normalizes_one_head_and_scales_it_by_the_gate() {
    let Some(gpu) = open() else { return };
    cases::out_gate_normalizes_one_head_and_scales_it_by_the_gate(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn recurrent_matches_a_naive_delta_rule() {
    let Some(gpu) = open() else { return };
    cases::recurrent_matches_a_naive_delta_rule(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_viable_recurrent_variant_matches_the_oracle() {
    let Some(gpu) = open() else { return };
    cases::every_viable_recurrent_variant_matches_the_oracle(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn recurrent_variants_are_bit_identical() {
    let Some(gpu) = open() else { return };
    cases::recurrent_variants_are_bit_identical(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn chunk_variants_are_bit_identical() {
    let Some(gpu) = open() else { return };
    cases::chunk_variants_are_bit_identical(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_viable_chunk_variant_matches_the_recurrence() {
    let Some(gpu) = open() else { return };
    cases::every_viable_chunk_variant_matches_the_recurrence(&gpu);
}

/// Every GDN variant this module compiles reports a block limit, and it is at
/// least the lane count the variant's index arithmetic assumes.
///
/// The names and the lane counts are `crates/shiso-models/src/qwen35/kernels/
/// gdn_tune.rs`'s shape table, restated here rather than imported because that
/// table is a host-side private and because a test that shares its source with
/// the thing it checks checks nothing.
#[test]
#[ignore = "requires a CUDA device"]
fn every_gdn_variant_reports_the_block_limit_its_arithmetic_assumes() {
    const VARIANTS: [(&str, u32); 9] = [
        ("kernel_q35_gdn_recurrent_s1", 128),
        ("kernel_q35_gdn_recurrent_s2", 256),
        ("kernel_q35_gdn_recurrent_s4", 512),
        ("kernel_q35_gdn_recurrent_hybrid", 128),
        ("kernel_q35_gdn_chunk_c1", 128),
        ("kernel_q35_gdn_chunk_c2", 256),
        ("kernel_q35_gdn_chunk_c4", 512),
        ("kernel_q35_gdn_chunk_c8", 1024),
        ("kernel_q35_gdn_chunk_hybrid", 128),
    ];
    let Some(gpu) = open() else { return };
    for (entry, lanes) in VARIANTS {
        let limit = gpu
            .device()
            .max_block_threads(shiso_core::KernelRef::new("qwen35", entry));
        assert_eq!(
            limit,
            Some(lanes),
            "{entry} should report exactly the lane count it was compiled for",
        );
    }
    // And an entry that is not there reports nothing, rather than a number off
    // the device: that is what lets a variant simply be absent.
    assert_eq!(
        gpu.device().max_block_threads(shiso_core::KernelRef::new(
            "qwen35",
            "kernel_q35_no_such_entry"
        )),
        None,
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn recurrent_trail_records_are_the_per_row_walk() {
    let Some(gpu) = open() else { return };
    cases::recurrent_trail_records_are_the_per_row_walk(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn conv_trail_records_are_the_per_row_walk() {
    let Some(gpu) = open() else { return };
    cases::conv_trail_records_are_the_per_row_walk(&gpu);
}
