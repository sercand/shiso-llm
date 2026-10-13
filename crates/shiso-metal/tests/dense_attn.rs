//! Metal's run of the shared `dense_attn` cases.
//!
//! The cases themselves live in `shiso-kerneltest`, so this file is the
//! backend's half only: open a device, or say why the cases did not run.
//!
//! One case is not ignore-gated: the pipelines have to resolve, and a missing or
//! misnamed entry point should fail the default gate rather than hide behind the
//! GPU gate. The rest are `#[ignore]`d: `cargo test -p shiso-metal --features
//! metal --test dense_attn -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, KernelRef};
use shiso_kerneltest::Gpu;
use shiso_kerneltest::dense::{attn, attn_flash};
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
#[expect(
    clippy::print_stderr,
    reason = "a machine with no Metal device skips rather than fails, and the reason belongs on stderr"
)]
fn every_dense_attention_entry_point_resolves_as_a_pipeline() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the pipeline check");
        return;
    }
    let backend = MetalBackend::new().expect("the embedded kernel library compiles");
    for entry in attn::ENTRIES
        .into_iter()
        .chain(attn_flash::ENTRIES.map(|(entry, _)| entry))
    {
        backend
            .prepare(KernelRef::new("dense", entry))
            .unwrap_or_else(|err| panic!("{entry} does not resolve: {err}"));
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn rope_pairs_over_the_full_head() {
    let Some(gpu) = open() else { return };
    attn::rope_pairs_over_the_full_head(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn rope_over_the_rotated_slice() {
    let Some(gpu) = open() else { return };
    attn::rope_over_the_rotated_slice(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn the_value_norm_is_unweighted() {
    let Some(gpu) = open() else { return };
    attn::the_value_norm_is_unweighted(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn window_and_bound_select_the_keys() {
    let Some(gpu) = open() else { return };
    attn::window_and_bound_select_the_keys(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn scale_is_an_argument() {
    let Some(gpu) = open() else { return };
    attn::scale_is_an_argument(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn an_empty_span_contributes_zeros_rather_than_nan() {
    let Some(gpu) = open() else { return };
    attn::an_empty_span_contributes_zeros_rather_than_nan(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn flash_long_causal_spans_match_the_oracle() {
    let Some(gpu) = open() else { return };
    attn_flash::long_causal_spans_match_the_oracle(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn flash_a_forward_bound_reads_past_the_query() {
    let Some(gpu) = open() else { return };
    attn_flash::a_forward_bound_reads_past_the_query(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn flash_empty_spans_write_zeros_beside_live_ones() {
    let Some(gpu) = open() else { return };
    attn_flash::empty_spans_write_zeros_beside_live_ones(&gpu);
}
