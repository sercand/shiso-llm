//! Metal's segmented-execution capability: one command buffer with a host
//! handshake cut between two segments.
//!
//! GPU cases: `cargo test -p shiso-metal --features metal --test segments -- --ignored`.
#![cfg(all(feature = "metal", target_os = "macos"))]
#![expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "fixture dimensions are small literals (256 rows) well inside f32 and usize"
)]

use std::sync::Arc;

use shiso_core::backend::{Backend, KernelArg, LaunchArgs};
use shiso_core::buffer::{DeviceBuffer, alloc};
use shiso_core::{Extent3, KernelRef};
use shiso_metal::MetalBackend;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = 1e-5 * w.abs().max(1.0);
        assert!((g - w).abs() <= tol, "{what}: element {i} got {g} want {w}");
    }
}

/// A cut pauses the GPU between two segments so the host can read what the
/// first wrote and write what the second reads, all in one committed command
/// buffer. The kernel is the in-place `x = silu(x * inv)`; segment A runs on
/// the initial rows, the host replaces them at the cut, segment B runs on the
/// replacement. If the cut did not order the GPU, B would run on A's output
/// (or the initial rows) instead of the replacement, and the final read would
/// not be `silu(replacement)`.
#[test]
#[ignore = "requires a Metal device"]
fn a_cut_orders_the_host_between_two_segments() {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device, skipping");
        return;
    }
    let metal = MetalBackend::new().expect("the embedded kernel library compiles");
    let backend: Arc<dyn Backend> = metal.clone();
    assert!(backend.supports_segments(), "Metal must support segments");

    let n = 256u64;
    let x: DeviceBuffer = alloc(&backend, n * 4).expect("alloc");
    let initial: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 1.0).collect();
    backend
        .copy_h2d(x.view(), &f32_bytes(&initial))
        .expect("h2d");
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let after_a: Vec<f32> = initial.iter().map(|&v| silu(v)).collect();
    let replacement: Vec<f32> = (0..n).map(|i| i as f32 * 0.02 - 2.0).collect();
    let after_b: Vec<f32> = replacement.iter().map(|&v| silu(v)).collect();

    let kernel = KernelRef::new("q4e_hc", "q4e_scale_silu_kernel");
    let args = [
        KernelArg::Buffer(x.view()),
        KernelArg::F32(1.0),
        KernelArg::U64(n),
    ];
    let launch = LaunchArgs::new(Extent3::new(1, 1, 1), Extent3::new(256, 1, 1), &args);

    // Segment A, a cut, then segment B — all encoded before one commit.
    backend.run(kernel, &launch).expect("run A");
    let cut = backend.encode_cut().expect("encode_cut");
    backend.run(kernel, &launch).expect("run B");
    backend.commit_segments().expect("commit_segments");

    // The GPU has run A and paused at the cut: its output is readable.
    backend.wait_cut(cut).expect("wait_cut");
    let mut got_a = vec![0u8; (n * 4) as usize];
    backend
        .read_shared(x.view(), &mut got_a)
        .expect("read_shared");
    assert_close(&read_f32(&got_a), &after_a, "after segment A");

    // The host replaces the rows, then releases the GPU into segment B.
    backend
        .write_shared(&[x.view()], &mut |slices| {
            slices[0].copy_from_slice(&f32_bytes(&replacement));
            Ok(())
        })
        .expect("write_shared");
    backend.resume_cut(cut).expect("resume_cut");
    backend.sync().expect("sync");

    let mut got_b = vec![0u8; (n * 4) as usize];
    backend.copy_d2h(x.view(), &mut got_b).expect("d2h");
    assert_close(&read_f32(&got_b), &after_b, "after segment B");
}
