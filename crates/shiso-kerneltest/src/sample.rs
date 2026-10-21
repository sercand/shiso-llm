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

/// Entries the top-of-row kernel keeps, mirroring
/// `shiso_engine::sparse::kernels::ROW_TOP`.
const ROW_TOP: u32 = 64;

/// Runs the top-of-row kernel over `rows` rows of `n_vocab` entries and
/// returns each row's `(ids, values)`, `min(ROW_TOP, n_vocab)` of them.
fn top_rows(gpu: &Gpu, logits: &[f32], rows: u32, n_vocab: u32) -> Vec<(Vec<u32>, Vec<f32>)> {
    let logits_dev = gpu.upload(&f32_bytes(logits));
    let entries = rows as usize * ROW_TOP as usize;
    let values_dev = gpu.upload(&vec![0u8; entries * 4]);
    let ids_dev = gpu.upload(&vec![0u8; entries * 4]);
    run(
        gpu,
        KernelRef::new(MOD_SAMPLE, "sample_top_rows_kernel"),
        &LaunchArgs {
            grid: Extent3::new(rows, 1, 1),
            block: Extent3::new(1_024, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(logits_dev.view()),
                KernelArg::U32(n_vocab),
                KernelArg::U32(ROW_TOP),
                KernelArg::Buffer(values_dev.view()),
                KernelArg::Buffer(ids_dev.view()),
            ],
        },
    );
    drain(gpu);
    let values = gpu.read_f32(&values_dev);
    let ids: Vec<u32> = gpu
        .read(&ids_dev)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let kept = ROW_TOP.min(n_vocab) as usize;
    (0..rows as usize)
        .map(|r| {
            let at = r * ROW_TOP as usize..r * ROW_TOP as usize + kept;
            (ids[at.clone()].to_vec(), values[at].to_vec())
        })
        .collect()
}

/// The oracle: a row's top `ROW_TOP` by the host's total order, larger first
/// and ties to the lower id, then ascending by id.
fn top_oracle(row: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|a, b| row[*b].total_cmp(&row[*a]).then(a.cmp(b)));
    order.truncate(ROW_TOP as usize);
    order.sort_unstable();
    (
        order.iter().map(|i| *i as u32).collect(),
        order.iter().map(|i| row[*i]).collect(),
    )
}

/// `sample_top_rows_kernel` keeps each row's top entries, ascending by id,
/// exactly the oracle's set: on full-vocabulary rows with a wide tie at the
/// cut, with `-inf` (a suppressed id) above it in value order and a negative
/// row, and on a row shorter than the kept count.
///
/// # Panics
///
/// When a row's ids or values differ from the oracle's.
pub fn the_top_of_a_row_is_its_highest_entries_by_id(gpu: &Gpu) {
    let v = VOCAB as usize;
    let mut logits = signal(4 * v, 11);
    // Row 1: a tie of 300 entries straddling the cut, so which ties are kept
    // is the lower ids' and not an arrival order's.
    for i in 0..300 {
        logits[v + i * 800] = 5.0;
    }
    for i in 0..40 {
        logits[v + 7 + i * 900] = 6.0;
    }
    // Row 2: the top masked to -inf, as a suppressed id would be.
    for i in 0..10 {
        logits[2 * v + i] = 9.0;
        logits[2 * v + 100 + i] = f32::NEG_INFINITY;
    }
    // Row 3: every entry negative.
    for value in &mut logits[3 * v..] {
        *value = -1.0 - value.abs();
    }
    let got = top_rows(gpu, &logits, 4, VOCAB);
    for (r, row) in logits.chunks(v).enumerate() {
        let want = top_oracle(row);
        assert_eq!(got[r].0, want.0, "row {r}: ids");
        same(&got[r].1, &want.1);
    }
    let short = signal(40, 3);
    let got = top_rows(gpu, &short, 1, 40);
    let want = top_oracle(&short);
    assert_eq!(got[0].0, want.0, "the short row keeps every entry");
    same(&got[0].1, &want.1);
}
