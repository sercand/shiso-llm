//! Metal's run of the shared `qwen4exp` qsa cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --test q4e_qsa -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_kerneltest::Gpu;
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

use shiso_kerneltest::qwen4exp::qsa;

#[test]
#[ignore = "requires a Metal device"]
fn q4e_qsa_q_norm_rope() {
    let Some(gpu) = open() else { return };
    qsa::the_query_norm_rope_splits_rotates_and_gates(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn q4e_qsa_store_kv() {
    let Some(gpu) = open() else { return };
    qsa::the_kv_store_follows_the_page_table_and_refuses_past_the_pool(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn q4e_qsa_attention_decode() {
    let Some(gpu) = open() else { return };
    qsa::the_dense_attention_decodes_across_scattered_pages(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn q4e_qsa_attention_prefill() {
    let Some(gpu) = open() else { return };
    qsa::the_dense_attention_prefills_a_chunk_causally(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn q4e_qsa_attention_split_decode() {
    let Some(gpu) = open() else { return };
    qsa::the_split_attention_decodes_and_folds(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn q4e_qsa_attention_split_verify() {
    let Some(gpu) = open() else { return };
    qsa::the_split_attention_verifies_near_the_start(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn q4e_qsa_attention_split_direct() {
    let Some(gpu) = open() else { return };
    qsa::the_split_attention_writes_directly_at_one_split(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn q4e_qsa_attention_gather() {
    let Some(gpu) = open() else { return };
    qsa::the_gathered_attention_reads_the_selection(&gpu);
}

#[test]
#[ignore = "requires a Metal device"]
fn q4e_qsa_gate() {
    let Some(gpu) = open() else { return };
    qsa::the_output_gate_is_a_sigmoid_and_stops_at_n(&gpu);
}
