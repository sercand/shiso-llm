//! What one picture costs the Qwen tower on a CUDA device.
//!
//! The CUDA twin of `vision_cost.rs`, driving the tower the server opens for a
//! `--vision` GGUF sidecar. It records durations and never a logit. The pixel
//! values are generated, because the tower's cost is a function of the grid
//! and the widths and not of what the picture shows.
//!
//! ```text
//! SHISO_TEST_MODEL_VISION=/path/to/vision.gguf \
//!   cargo test -p shiso-bench --release --test vision_cost_cuda -- --ignored --nocapture
//! ```

#![cfg(target_os = "linux")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use shiso_core::Backend;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_models::qwen35::vision::{
    Qwen35Tower, VisionEncoder, VisionWeights, load_vision_weights,
};
use shiso_vision::{VisionItem, VisionSource, VisionTower};

/// Square edges, in pixels: a small still and the size agents screenshot at.
const EDGES: [usize; 2] = [448, 896];

/// Encodes timed per edge, after one discarded.
const TIMED: usize = 3;

#[expect(clippy::print_stdout, reason = "the measurement is a test's output")]
fn report(what: &str) {
    println!("{what}");
}

fn tower_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_VISION").ok()?);
    path.is_file().then_some(path)
}

/// One item over a square image of `edge` pixels at the tower's geometry.
fn item_for(tower: &Qwen35Tower, edge: usize) -> VisionItem {
    let geom = tower.preprocess().geometry;
    let merge = geom.spatial_merge_size();
    let side = (edge / geom.patch_size() / merge) * merge;
    let row_len = geom.patch_dim();
    let mut state: u32 = 0x9E37_79B9;
    let pixels: Vec<f32> = (0..side * side * row_len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from(state.to_le_bytes()[3]) / 127.5 - 1.0
        })
        .collect();
    VisionItem::new(pixels.into_boxed_slice(), row_len, 1, side, side)
        .expect("the generated buffer matches its grid")
}

/// One encode, readback included, in milliseconds and as host floats: the
/// readback is what forces the device to have finished.
fn encode(
    backend: &Arc<dyn Backend>,
    encoder: VisionEncoder,
    weights: &VisionWeights,
    item: &VisionItem,
) -> (f64, Vec<f32>) {
    let started = Instant::now();
    let out = encoder
        .encode(backend, weights, item)
        .expect("the tower encodes the item");
    let mut host = vec![0u8; usize::try_from(out.view().len()).expect("fits")];
    backend
        .copy_d2h(out.view(), &mut host)
        .expect("the rows read back");
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    let values = host
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    (ms, values)
}

/// The median of `TIMED` encodes after one discarded, and the last output.
fn timed(
    backend: &Arc<dyn Backend>,
    encoder: VisionEncoder,
    weights: &VisionWeights,
    item: &VisionItem,
) -> (f64, Vec<f32>) {
    encode(backend, encoder, weights, item);
    let mut out = Vec::new();
    let mut times: Vec<f64> = (0..TIMED)
        .map(|_| {
            let (ms, values) = encode(backend, encoder, weights, item);
            out = values;
            ms
        })
        .collect();
    times.sort_by(f64::total_cmp);
    (times[TIMED / 2], out)
}

#[test]
#[ignore = "needs the local tower sidecar and a CUDA device"]
fn qwen_tower_encode_cost_on_cuda() {
    let Some(path) = tower_path() else {
        report("SKIP: SHISO_TEST_MODEL_VISION does not name a file");
        return;
    };
    let backend: Arc<dyn Backend> =
        CudaBackend::new(0, CudaOptions::default()).expect("a CUDA device");
    let tower = Qwen35Tower::open(&backend, &path).expect("the tower opens");
    let weights =
        load_vision_weights(&backend, &VisionSource::SidecarGguf(path)).expect("the weights load");
    for edge in EDGES {
        let item = item_for(&tower, edge);
        let (fast_ms, fast) = timed(&backend, VisionEncoder::new(), &weights, &item);
        let (scalar_ms, scalar) = timed(
            &backend,
            VisionEncoder::new().with_scalar_kernels(),
            &weights,
            &item,
        );
        let worst = fast
            .iter()
            .zip(&scalar)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let largest = scalar.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let (t, h, w) = item.grid();
        report(&format!(
            "vision encode edge={edge} patches={} fast_ms={fast_ms:.1} scalar_ms={scalar_ms:.1} \
             worst_abs_delta={worst:e} largest={largest:e}",
            t * h * w,
        ));
    }
}
