//! The CPU device's run of the shared kernel cases.
//!
//! The cases live in `shiso-kerneltest`, so this file is the backend's half
//! only: open the device and name the cases it supports. Unlike the GPU
//! backends' cases these are not `#[ignore]`d: the CPU is always present.
//! Later kernel tasks add their cases to the `cpu_cases!` lists below.

use std::sync::Arc;

use shiso_core::Backend;
use shiso_cpu::{CpuBackend, CpuOptions};
use shiso_kerneltest::Gpu;

/// Opens the CPU device. The keepalive is the backend itself.
fn open() -> Gpu {
    let cpu = CpuBackend::new(CpuOptions::default()).expect("the CPU device always opens");
    let device: Arc<dyn Backend> = cpu.clone();
    Gpu::new(device, Box::new(cpu))
}

/// One `#[test]` per case, named after it, each on a fresh device. A case is
/// written `module::file::case`, its path under `shiso_kerneltest`.
macro_rules! cpu_cases {
    ($($m:ident :: $f:ident :: $case:ident),+ $(,)?) => {
        $(
            #[test]
            fn $case() {
                shiso_kerneltest::$m::$f::$case(&open());
            }
        )+
    };
}

cpu_cases!(
    generic::norm::the_plain_norm_scales_the_row_and_multiplies_by_the_weight,
    generic::norm::the_fused_close_writes_both_the_sum_and_the_normalized_row,
    generic::glu::the_gated_silu_is_the_sigmoid_weighted_gate_times_the_up_row,
    generic::glu::the_clamp_is_one_sided_on_the_gate_and_two_sided_on_the_up_row,
    generic::rows_q8::the_q8_0_gather_dequantizes_the_rows_the_ids_name,
);
