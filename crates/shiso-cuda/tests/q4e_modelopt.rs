//! CUDA's run of the shared `qwen4exp` `modelopt` cases: the NVFP4 routed
//! expert stages and the FP8 and NVFP4 n-gram row decodes, all in the
//! `q4e_modelopt` module, and the engine's resident NVFP4 bank read through
//! those stages.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --test
//! q4e_modelopt -- --ignored`.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::{fp8, nvfp4, ple_nvfp4};

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
fn the_nvfp4_expert_stages_follow_the_slot_map() {
    let Some(gpu) = open() else { return };
    nvfp4::the_nvfp4_expert_stages_follow_the_slot_map(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_fp8_ngram_rows_decode_under_the_table_scale() {
    let Some(gpu) = open() else { return };
    fp8::the_fp8_ngram_rows_decode_under_the_table_scale(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_fp8_expert_stages_follow_the_slot_map() {
    let Some(gpu) = open() else { return };
    fp8::the_fp8_expert_stages_follow_the_slot_map(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_nvfp4_prompt_stages_follow_the_slot_map() {
    let Some(gpu) = open() else { return };
    nvfp4::the_nvfp4_prompt_stages_follow_the_slot_map(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_nvfp4_step_rows_do_not_depend_on_the_pass() {
    let Some(gpu) = open() else { return };
    nvfp4::the_nvfp4_step_rows_do_not_depend_on_the_pass(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_nvfp4_step_rows_are_the_lane_order_bit_for_bit() {
    let Some(gpu) = open() else { return };
    nvfp4::the_nvfp4_step_rows_are_the_lane_order_bit_for_bit(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_activated_gate_up_is_the_activation_of_its_rows() {
    let Some(gpu) = open() else { return };
    nvfp4::the_activated_gate_up_is_the_activation_of_its_rows(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_nvfp4_prompt_columns_do_not_depend_on_the_chunk() {
    let Some(gpu) = open() else { return };
    nvfp4::the_nvfp4_prompt_columns_do_not_depend_on_the_chunk(&gpu);
}

/// The engine's resident NVFP4 bank, a draft head's, read through the step
/// stages exactly as a streamed slot pool is.
#[test]
#[ignore = "requires a CUDA device"]
fn the_nvfp4_stages_read_a_resident_bank() {
    use shiso_engine::sparse::weights::{Weight, WeightPart};
    let Some(gpu) = open() else { return };
    nvfp4::the_nvfp4_stages_read_a_placed_bank(
        &gpu,
        |gpu, bank| {
            Weight::nvfp4(
                bank.in_dim,
                bank.out_dim,
                bank.slots,
                [
                    gpu.upload(bank.codes),
                    gpu.upload(bank.scales),
                    gpu.upload(bank.globals),
                ],
            )
            .expect("the arrays are the bank's own lengths")
        },
        |weight| {
            [
                WeightPart::Whole,
                WeightPart::GroupScales,
                WeightPart::GlobalScale,
            ]
            .map(|part| {
                weight
                    .part(part)
                    .expect("a resident NVFP4 bank has this part")
            })
        },
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn nvfp4_ple_row_decode_equals_host_decode() {
    let Some(gpu) = open() else { return };
    ple_nvfp4::nvfp4_ple_row_decode_equals_host_decode(&gpu);
}

#[test]
#[ignore = "requires a CUDA device"]
fn an_nvfp4_ple_row_under_the_nan_scale_decodes_to_nan() {
    let Some(gpu) = open() else { return };
    ple_nvfp4::an_nvfp4_ple_row_under_the_nan_scale_decodes_to_nan(&gpu);
}
