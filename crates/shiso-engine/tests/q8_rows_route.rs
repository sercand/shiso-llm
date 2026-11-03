//! Which kernel a few `q8_0` rows go through: the row kernel only for a weight
//! marked row-exact, the extended family for every other weight, so a family
//! that never asked for row exactness answers what it was checked against.
//!
//! GPU case: `cargo test -p shiso-engine --test q8_rows_route -- --ignored`.
#![cfg(target_os = "macos")]

use std::sync::Arc;

use shiso_core::profile::ProfileRegion;
use shiso_core::{Backend, DType, alloc};
use shiso_engine::projection::{Weight, projection};

const IN_DIM: u32 = 64;
const OUT_DIM: u32 = 16;
const ROWS: u32 = 2;

/// The entries one two-row projection through `weight` launches.
fn entries(backend: &Arc<dyn Backend>, row_exact: bool) -> Vec<&'static str> {
    // 34-byte blocks of 32: a zero scale and zero quants are a valid matrix.
    let blocks = alloc(backend, u64::from(OUT_DIM * IN_DIM / 32 * 34)).expect("alloc");
    backend.fill_zero(blocks.view()).expect("fill");
    let weight = Weight::from_buffer(DType::Q8_0, IN_DIM, OUT_DIM, blocks);
    let weight = if row_exact {
        weight.row_exact()
    } else {
        weight
    };
    let src = alloc(backend, u64::from(ROWS * IN_DIM * 4)).expect("alloc");
    backend.fill_zero(src.view()).expect("fill");
    let dst = alloc(backend, u64::from(ROWS * OUT_DIM * 4)).expect("alloc");
    backend.sync().expect("sync");
    let region = ProfileRegion::open(backend.as_profiler().expect("the device times launches"))
        .expect("open");
    projection(backend.as_ref(), &weight, ROWS, src.view(), dst.view()).expect("projection");
    let report = region.finish().expect("finish");
    report.kernels.iter().map(|k| k.key.entry).collect()
}

#[test]
#[ignore = "requires a Metal device"]
fn only_a_row_exact_weight_takes_the_row_kernel() {
    let backend: Arc<dyn Backend> =
        shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let exact = entries(&backend, true);
    assert!(exact.iter().any(|e| e.contains("q8_0_rows")), "{exact:?}");
    let plain = entries(&backend, false);
    assert!(plain.iter().all(|e| !e.contains("q8_0_rows")), "{plain:?}");
}
