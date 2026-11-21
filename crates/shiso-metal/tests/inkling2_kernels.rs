//! Metal's run of the shared `inkling2` module cases.

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::inkling2;
use shiso_metal::MetalBackend;

#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open_gpu() -> Option<Gpu> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the inkling2 GPU cases");
        return None;
    }
    let metal = MetalBackend::new().expect("the embedded kernel library compiles");
    let device: Arc<dyn Backend> = metal.clone();
    Some(Gpu::new(device, Box::new(metal)))
}

#[test]
#[ignore = "requires a Metal device"]
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
            #[ignore = "requires a Metal device"]
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
