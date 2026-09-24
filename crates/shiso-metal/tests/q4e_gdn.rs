//! Metal's run of the shared `qwen4exp` gdn cases.
//!
//! The cases live in `shiso-kerneltest`; this file opens a device or says why
//! the cases did not run.
//!
//! GPU cases: `cargo test -p shiso-metal --test q4e_gdn -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_kerneltest::Gpu;
use shiso_kerneltest::qwen4exp::gdn;
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

/// `shiso_kerneltest::qwen4exp::gdn::conv_decode` on this device.
#[test]
#[ignore = "requires a Metal device"]
fn conv_decode() {
    if let Some(gpu) = open() {
        gdn::conv_decode(&gpu);
    }
}

/// `shiso_kerneltest::qwen4exp::gdn::conv_prefill` on this device.
#[test]
#[ignore = "requires a Metal device"]
fn conv_prefill() {
    if let Some(gpu) = open() {
        gdn::conv_prefill(&gpu);
    }
}

/// `shiso_kerneltest::qwen4exp::gdn::conv_records_rollback_windows` on this device.
#[test]
#[ignore = "requires a Metal device"]
fn conv_records_rollback_windows() {
    if let Some(gpu) = open() {
        gdn::conv_records_rollback_windows(&gpu);
    }
}

/// `shiso_kerneltest::qwen4exp::gdn::l2norm_prefill` on this device.
#[test]
#[ignore = "requires a Metal device"]
fn l2norm_prefill() {
    if let Some(gpu) = open() {
        gdn::l2norm_prefill(&gpu);
    }
}

/// `shiso_kerneltest::qwen4exp::gdn::gates_prefill` on this device.
#[test]
#[ignore = "requires a Metal device"]
fn gates_prefill() {
    if let Some(gpu) = open() {
        gdn::gates_prefill(&gpu);
    }
}

/// `shiso_kerneltest::qwen4exp::gdn::out_gate_prefill` on this device.
#[test]
#[ignore = "requires a Metal device"]
fn out_gate_prefill() {
    if let Some(gpu) = open() {
        gdn::out_gate_prefill(&gpu);
    }
}

/// `shiso_kerneltest::qwen4exp::gdn::recurrent_decode` on this device.
#[test]
#[ignore = "requires a Metal device"]
fn recurrent_decode() {
    if let Some(gpu) = open() {
        gdn::recurrent_decode(&gpu);
    }
}

/// `shiso_kerneltest::qwen4exp::gdn::recurrent_prefill` on this device.
#[test]
#[ignore = "requires a Metal device"]
fn recurrent_prefill() {
    if let Some(gpu) = open() {
        gdn::recurrent_prefill(&gpu);
    }
}

/// `shiso_kerneltest::qwen4exp::gdn::recurrent_rollback_and_replay` on this device.
#[test]
#[ignore = "requires a Metal device"]
fn recurrent_rollback_and_replay() {
    if let Some(gpu) = open() {
        gdn::recurrent_rollback_and_replay(&gpu);
    }
}
