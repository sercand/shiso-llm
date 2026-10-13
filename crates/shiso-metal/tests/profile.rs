//! Per-launch GPU timing on the device: a region times every launch on its
//! own, and closing it - finished or abandoned - restores batching.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer, ProfileRegion, alloc};
use shiso_metal::MetalBackend;

#[expect(
    clippy::print_stderr,
    reason = "a skipped device case says why on stderr, where the test runner shows it"
)]
fn open() -> Option<(Arc<MetalBackend>, Arc<dyn Backend>)> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the GPU cases");
        return None;
    }
    let metal = MetalBackend::new().expect("the embedded kernel library compiles");
    let device: Arc<dyn Backend> = metal.clone();
    Some((metal, device))
}

fn buffer(device: &Arc<dyn Backend>) -> DeviceBuffer {
    alloc(device, 1 << 20).expect("a 1 MiB buffer allocates")
}

#[test]
#[ignore = "requires a Metal device"]
fn every_launch_in_a_region_is_timed_under_its_kernel() {
    let Some((_metal, device)) = open() else {
        return;
    };
    let dst = buffer(&device);
    let profiler = device.as_profiler().expect("Metal times launches");
    let region = ProfileRegion::open(profiler).expect("no region is open yet");
    for _ in 0..8 {
        device.fill_zero(dst.view()).expect("the fill launches");
    }
    let report = region.finish().expect("the region closes");
    assert_eq!(report.launches, 8, "{}", report.table(10));
    assert_eq!(report.kernels.len(), 1, "{}", report.table(10));
    let row = &report.kernels[0];
    assert_eq!(row.key.entry, "kernel_fill_zero_u8");
    assert_eq!(row.launches, 8);
    // Zero would mean the end time was read before the completion
    // notification set it.
    assert!(row.min_ns > 0, "{}", report.table(10));
    assert!(report.serialized_ns >= row.max_ns);
    assert!(
        report.wall_ns >= report.serialized_ns,
        "{}",
        report.table(10)
    );
}

#[test]
#[ignore = "requires a Metal device"]
fn a_blit_copy_is_its_own_row() {
    let Some((_metal, device)) = open() else {
        return;
    };
    let (src, dst) = (buffer(&device), buffer(&device));
    let profiler = device.as_profiler().expect("Metal times launches");
    let region = ProfileRegion::open(profiler).expect("opens");
    device
        .copy_d2d(dst.view(), src.view())
        .expect("the copy encodes");
    let report = region.finish().expect("closes");
    assert_eq!(report.kernels.len(), 1, "{}", report.table(10));
    assert_eq!(report.kernels[0].key.entry, "copy_d2d");
}

#[test]
#[ignore = "requires a Metal device"]
fn a_second_region_is_refused() {
    let Some((_metal, device)) = open() else {
        return;
    };
    let profiler = device.as_profiler().expect("Metal times launches");
    let first = ProfileRegion::open(profiler).expect("opens");
    assert!(ProfileRegion::open(profiler).is_err());
    first.finish().expect("the first region still closes");
}

#[test]
#[ignore = "requires a Metal device"]
fn batching_resumes_after_a_region() {
    let Some((metal, device)) = open() else {
        return;
    };
    let dst = buffer(&device);
    let profiler = device.as_profiler().expect("Metal times launches");

    let before = metal.commits();
    let region = ProfileRegion::open(profiler).expect("opens");
    for _ in 0..4 {
        device.fill_zero(dst.view()).expect("fills");
    }
    region.finish().expect("closes");
    assert_eq!(metal.commits() - before, 4, "one commit per launch inside");

    let before = metal.commits();
    for _ in 0..4 {
        device.fill_zero(dst.view()).expect("fills");
    }
    device.sync().expect("drains");
    assert_eq!(
        metal.commits() - before,
        1,
        "one commit for the batch outside"
    );
}

#[test]
#[ignore = "requires a Metal device"]
fn batching_resumes_after_an_abandoned_region() {
    let Some((metal, device)) = open() else {
        return;
    };
    let dst = buffer(&device);
    let profiler = device.as_profiler().expect("Metal times launches");
    drop(ProfileRegion::open(profiler).expect("opens"));

    let before = metal.commits();
    for _ in 0..4 {
        device.fill_zero(dst.view()).expect("fills");
    }
    device.sync().expect("drains");
    assert_eq!(metal.commits() - before, 1);
}
