//! Every kernel bench case builds, launches and times on the device, so a
//! case cannot rot between the times somebody benchmarks with it.
#![cfg(all(feature = "metal", target_os = "macos"))]

use std::sync::Arc;

use shiso_core::Backend;
use shiso_kerneltest::Gpu;
use shiso_kerneltest::bench::{CASES, time};
use shiso_metal::MetalBackend;

#[expect(
    clippy::print_stderr,
    reason = "a skipped device case says why on stderr"
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
#[ignore = "requires a Metal device"]
fn every_bench_case_times_one_kernel() {
    let Some(gpu) = open() else { return };
    assert!(!CASES.is_empty());
    for case in CASES {
        let fixture = (case.build)(&gpu);
        let stat = time(&gpu, &fixture, 2).unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(stat.launches, 2, "{}", case.name);
        assert!(stat.min_ns > 0, "{}", case.name);
        assert!(fixture.bytes > 0, "{}", case.name);
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn time_refuses_zero_reps() {
    let Some(gpu) = open() else { return };
    let fixture = (CASES[0].build)(&gpu);
    assert!(time(&gpu, &fixture, 0).is_err());
}
