//! The logits stages every stack runs before a pick: cases graded against a
//! host restatement of the kernel bodies.
//!
//! The suppress stage writes `-inf` and copies everything else, so every
//! comparison here is bitwise.
#![expect(
    clippy::cast_possible_truncation,
    reason = "kernel arguments are u32 while the fixtures index in usize; every size and index \
              here is a small literal"
)]

use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The module, mirroring `shiso_engine::sparse::kernels::MOD_SAMPLE`.
const MOD_SAMPLE: &str = "sample";
/// Threads per row, mirroring the engine's launch.
const BLOCK: u32 = 256;
/// The shipped Qwen vocabulary, so a listed id sits at the far end of a row.
const VOCAB: u32 = 248_320;

/// `u32` values as little-endian bytes.
fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Runs the stage over `rows` rows of `logits` with `list` (count first).
fn suppress(gpu: &Gpu, logits: &[f32], rows: u32, list: &[u32]) -> Vec<f32> {
    let logits_dev = gpu.upload(&f32_bytes(logits));
    let list_dev = gpu.upload(&u32_bytes(list));
    run(
        gpu,
        KernelRef::new(MOD_SAMPLE, "sample_suppress_rows_kernel"),
        &LaunchArgs {
            grid: Extent3::new(rows, 1, 1),
            block: Extent3::new(BLOCK, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(logits_dev.view()),
                KernelArg::Buffer(list_dev.view()),
                KernelArg::U32(VOCAB),
            ],
        },
    );
    drain(gpu);
    gpu.read_f32(&logits_dev)
}

/// Asserts two `f32` runs are bitwise equal, naming the first that is not.
fn same(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "lengths");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "element {i}: {g} vs {w}");
    }
}

/// `sample_suppress_rows_kernel` writes `-inf` at every listed id of every
/// row and leaves every other element as it was - including an id listed
/// twice, one past a block's worth of threads, and one past the vocabulary,
/// which is skipped rather than written.
///
/// # Panics
///
/// When an element differs from the oracle's.
pub fn the_suppress_stage_masks_every_listed_id_in_every_row(gpu: &Gpu) {
    let v = VOCAB as usize;
    let rows = 3u32;
    let logits = signal(rows as usize * v, 5);
    let mut ids: Vec<u32> = (0..300).map(|i| 248_044 + (i % 276)).collect();
    ids.extend([0, VOCAB - 1, 7, 7, VOCAB, u32::MAX]);
    let mut list = vec![ids.len() as u32];
    list.extend(&ids);
    let mut want = logits.clone();
    for row in want.chunks_mut(v) {
        for &id in &ids {
            if id < VOCAB {
                row[id as usize] = f32::NEG_INFINITY;
            }
        }
    }
    same(&suppress(gpu, &logits, rows, &list), &want);
}

/// An empty list - the count zero, which is what a stack starts with - leaves
/// the rows untouched, so a server that suppresses nothing pays one launch and
/// no change in what it picks.
///
/// # Panics
///
/// When any element changed.
pub fn an_empty_list_leaves_the_rows_alone(gpu: &Gpu) {
    let logits = signal(2 * VOCAB as usize, 9);
    // A stale id behind a zero count must not be read.
    same(&suppress(gpu, &logits, 2, &[0, 5, 6]), &logits);
}
