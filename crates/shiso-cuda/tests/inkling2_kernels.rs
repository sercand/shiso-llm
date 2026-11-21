//! CUDA's run of the shared `inkling2` module cases.

#![cfg(target_os = "linux")]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_cuda::CudaBackend;
use shiso_kerneltest::Gpu;
use shiso_kerneltest::inkling2;

#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open_gpu() -> Option<Gpu> {
    let cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("note: no CUDA device available ({err}), skipping inkling2 GPU cases");
            return None;
        }
    };
    let device: Arc<dyn Backend> = cuda.clone();
    Some(Gpu::new(device, Box::new(cuda)))
}

#[test]
#[ignore = "requires a CUDA device"]
fn every_inkling2_entry_resolves_as_a_pipeline() {
    let Some(gpu) = open_gpu() else { return };
    for entry in inkling2::ENTRIES {
        assert!(
            gpu.device()
                .max_block_threads(KernelRef::new(inkling2::MODULE, entry))
                .is_some(),
            "{entry} does not resolve"
        );
    }
}

macro_rules! inkling2_cases {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            #[ignore = "requires a CUDA device"]
            fn $name() {
                let Some(gpu) = open_gpu() else { return };
                inkling2::$name(&gpu);
            }
        )*
    };
}

inkling2_cases!(
    attn_res_matches_oracle,
    loop_control_matches_oracle,
    short_convs_match_oracle,
    moe_nvfp4_and_mxfp8_match_oracle,
    rel_attn_with_sink_matches_oracle,
    route_topk_matches_oracle,
    bf16_linear_and_norms_match_oracle,
    qkvr_gate_sink_matches_oracle,
    vit_kernels_match_oracle,
    mtp_kernels_match_oracle,
);
