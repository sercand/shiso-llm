//! Host-side operations of the CPU device, exercised through `Arc<dyn Backend>`
//! only: what an engine can see of it, and nothing it cannot.

use std::sync::Arc;

use shiso_core::{Backend, CoreError, DeviceView, Extent3, KernelRef, LaunchArgs, MemTag, alloc};
use shiso_cpu::{CpuBackend, CpuOptions};

fn backend() -> Arc<dyn Backend> {
    CpuBackend::new(CpuOptions::default()).expect("the CPU device always opens")
}

#[test]
fn a_copy_round_trips() {
    let backend = backend();
    let buffer = alloc(&backend, 64).expect("alloc");
    let payload: Vec<u8> = (1..=37).collect();
    backend
        .copy_h2d(buffer.slice(3, 37).expect("slice"), &payload)
        .expect("h2d");
    let mut all = vec![0xAAu8; 64];
    backend.copy_d2h(buffer.view(), &mut all).expect("d2h");
    assert_eq!(&all[3..40], payload.as_slice());
    assert!(all[..3].iter().all(|&b| b == 0));
    assert!(all[40..].iter().all(|&b| b == 0));
}

#[test]
fn fill_zero_clears_only_its_view() {
    let backend = backend();
    let buffer = alloc(&backend, 32).expect("alloc");
    backend.copy_h2d(buffer.view(), &[7u8; 32]).expect("h2d");
    backend
        .fill_zero(buffer.slice(8, 8).expect("slice"))
        .expect("fill");
    let mut all = vec![0u8; 32];
    backend.copy_d2h(buffer.view(), &mut all).expect("d2h");
    for (i, b) in all.iter().enumerate() {
        assert_eq!(*b, if (8..16).contains(&i) { 0 } else { 7 }, "byte {i}");
    }
}

#[test]
fn copy_d2d_moves_bytes_between_and_within_buffers() {
    let backend = backend();
    let a = alloc(&backend, 32).expect("alloc");
    let b = alloc(&backend, 32).expect("alloc");
    let data: Vec<u8> = (0..32).collect();
    backend.copy_h2d(a.view(), &data).expect("h2d");
    backend
        .copy_d2d(b.slice(0, 16).expect("dst"), a.slice(16, 16).expect("src"))
        .expect("across buffers");
    // Overlapping inside one buffer behaves as memmove.
    backend
        .copy_d2d(a.slice(4, 16).expect("dst"), a.slice(0, 16).expect("src"))
        .expect("within a buffer");
    let mut out = vec![0u8; 32];
    backend.copy_d2h(b.view(), &mut out).expect("d2h");
    assert_eq!(&out[..16], &data[16..]);
    backend.copy_d2h(a.view(), &mut out).expect("d2h");
    assert_eq!(&out[4..20], &data[..16]);
}

#[test]
fn copy_d2d_refuses_a_length_mismatch() {
    let backend = backend();
    let a = alloc(&backend, 32).expect("alloc");
    let b = alloc(&backend, 32).expect("alloc");
    let err = backend
        .copy_d2d(a.slice(0, 16).expect("dst"), b.slice(0, 8).expect("src"))
        .expect_err("lengths differ");
    assert!(
        matches!(
            err,
            CoreError::ShapeMismatch {
                field: "copy_d2d len",
                expected: 8,
                got: 16
            }
        ),
        "{err:?}"
    );
}

#[test]
fn a_view_past_the_allocation_is_out_of_bounds() {
    let backend = backend();
    let small = alloc(&backend, 16).expect("alloc");
    let big = alloc(&backend, 64).expect("alloc");
    // A view cut from the larger buffer, replayed against the smaller handle,
    // is what a caller that mixes handles would hand the device.
    let wide = big.slice(0, 48).expect("slice");
    let forged = DeviceView::from_raw(
        shiso_core::RawBuffer::new(small.view().id(), 16),
        wide.offset(),
        wide.len(),
    );
    let mut out = vec![0u8; 48];
    let err = backend
        .copy_d2h(forged, &mut out)
        .expect_err("past the end");
    assert!(matches!(err, CoreError::OutOfBounds { .. }), "{err:?}");
    let err = backend
        .copy_h2d(forged, &[0u8; 48])
        .expect_err("past the end");
    assert!(matches!(err, CoreError::OutOfBounds { .. }), "{err:?}");
}

#[test]
fn an_unknown_kernel_is_unsupported() {
    let backend = backend();
    let args = LaunchArgs::new(Extent3::linear(1), Extent3::linear(1), &[]);
    let err = backend
        .run(KernelRef::new("generic", "kernel_nope"), &args)
        .expect_err("no such kernel");
    assert!(
        matches!(
            err,
            CoreError::Unsupported {
                what: "kernel_nope",
                backend: "cpu"
            }
        ),
        "{err:?}"
    );
    assert!(!shiso_cpu::has_entry("generic", "kernel_nope"));
}

#[test]
fn memory_info_reports_host_ram() {
    let info = backend().memory_info().expect("the host can say");
    assert!(info.total > 0);
    assert!(info.free <= info.total);
}

#[test]
fn the_ledger_counts_live_buffers() {
    let backend = backend();
    let ledger = backend.ledger().expect("the CPU device keeps a ledger");
    let before = ledger.live(MemTag::Untagged);
    let buffer = alloc(&backend, 4096).expect("alloc");
    assert_eq!(ledger.live(MemTag::Untagged), before + 4096);
    drop(buffer);
    assert_eq!(ledger.live(MemTag::Untagged), before);
}

#[test]
fn numerics_id_is_versioned() {
    let backend = backend();
    assert_eq!(backend.numerics_id(), "cpu-v1");
    assert_eq!(backend.backend_name(), "cpu");
}
