//! The sparse family's misc kernels: cases graded against a host restatement of the CUDA bodies.
//!
//! The leftovers of `q4e_misc` - the n-gram block's two-term residual add,
//! the on-device greedy pick and the block drafter's stream mean - and the half-precision row gathers of
//! `q4e_embed`. Each runs at the engine's grid: `elementwise` (256 threads) for
//! the add and the gathers, `argmax_rows` (one 1024-thread block per row) for
//! the pick. Every output here is an exact restatement - a copy, an f32 add, an
//! index - so every comparison is bitwise.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "kernel arguments are u32/i32 while the fixtures index in usize; every size and \
              index here is a small literal"
)]

use half::f16;
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The add and the pick's module, mirroring `kernels::MOD_MISC`.
const MOD_MISC: &str = "q4e_misc";
/// The gathers' module, mirroring `kernels::MOD_EMBED`.
const MOD_EMBED: &str = "q4e_embed";
/// Threads in an elementwise launch, mirroring `kernels::BLOCK_LINEAR`.
const BLOCK_LINEAR: u32 = 256;
/// Threads in the pick's one block per row, mirroring `kernels::BLOCK_WIDE`.
const BLOCK_WIDE: u32 = 1_024;
/// The residual width of one stream.
const HIDDEN: u32 = 2_560;
/// Residual streams.
const STREAMS: u32 = 4;
/// The shipped vocabulary, which the pick reads whole.
const VOCAB: u32 = 248_320;
/// Rows in the gather fixture's table: the shipped vocabulary would be a
/// 1.2 GiB table, and a gather reads nothing but the rows it is asked for.
const TABLE_ROWS: u32 = 64;

/// Launches one entry and waits for it.
fn launch(gpu: &Gpu, kernel: KernelRef, grid: Extent3, block: u32, args: &[KernelArg<'_>]) {
    run(
        gpu,
        kernel,
        &LaunchArgs {
            grid,
            block: Extent3::new(block, 1, 1),
            smem_bytes: 0,
            args,
        },
    );
    drain(gpu);
}

/// The engine's `elementwise(n)` grid.
fn flat(n: u64) -> Extent3 {
    Extent3::new(n.div_ceil(u64::from(BLOCK_LINEAR)) as u32, 1, 1)
}

/// `f16` values as little-endian bytes.
fn f16_bytes(values: &[f16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `i32` values as little-endian bytes.
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Asserts two `f32` runs are bitwise equal, naming the first that is not.
fn same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: element {i} is {g}, expected {w}"
        );
    }
}

/// `q4e_add2_kernel` adds both n-gram branches into the residual, the two
/// branches summed first.
pub fn the_ngram_add_sums_both_branches_into_the_residual(gpu: &Gpu) {
    let n = (3 * STREAMS * HIDDEN) as usize;
    let res = signal(n, 3);
    let a = signal(n, 5);
    let b = signal(n, 7);
    let res_dev = gpu.upload(&f32_bytes(&res));
    let a_dev = gpu.upload(&f32_bytes(&a));
    let b_dev = gpu.upload(&f32_bytes(&b));
    launch(
        gpu,
        KernelRef::new(MOD_MISC, "q4e_add2_kernel"),
        flat(n as u64),
        BLOCK_LINEAR,
        &[
            KernelArg::Buffer(res_dev.view()),
            KernelArg::Buffer(a_dev.view()),
            KernelArg::Buffer(b_dev.view()),
            KernelArg::U64(n as u64),
        ],
    );
    let want: Vec<f32> = (0..n).map(|i| res[i] + (a[i] + b[i])).collect();
    same(&gpu.read_f32(&res_dev), &want, "add2");
}

/// `q4e_stream_mean_rows_kernel` writes each row's mean over the streams into
/// its slot of a wider row and touches nothing else of that row: what a block
/// drafter reads as one tapped layer. The streams are summed in order and
/// divided once, so the oracle is that arithmetic, bitwise.
///
/// # Panics
///
/// When an element is not the host's.
pub fn the_stream_mean_lands_in_its_tap_slot(gpu: &Gpu) {
    const ROWS: u32 = 5;
    const TAPS: u32 = 3;
    const SLOT: u32 = 1;
    let n = (ROWS * STREAMS * HIDDEN) as usize;
    let res = signal(n, 13);
    let width = (ROWS * TAPS * HIDDEN) as usize;
    let fill = signal(width, 17);
    let res_dev = gpu.upload(&f32_bytes(&res));
    let dst_dev = gpu.upload(&f32_bytes(&fill));
    launch(
        gpu,
        KernelRef::new(MOD_MISC, "q4e_stream_mean_rows_kernel"),
        flat(u64::from(ROWS * HIDDEN)),
        BLOCK_LINEAR,
        &[
            KernelArg::Buffer(dst_dev.view()),
            KernelArg::Buffer(res_dev.view()),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(STREAMS),
            KernelArg::U32(ROWS),
            KernelArg::U32(TAPS * HIDDEN),
            KernelArg::U32(SLOT * HIDDEN),
        ],
    );
    let (h, s) = (HIDDEN as usize, STREAMS as usize);
    let mut want = fill;
    for t in 0..ROWS as usize {
        for e in 0..h {
            let mut sum = 0.0f32;
            for k in 0..s {
                sum += res[(t * s + k) * h + e];
            }
            want[t * TAPS as usize * h + SLOT as usize * h + e] = sum / STREAMS as f32;
        }
    }
    same(&gpu.read_f32(&dst_dev), &want, "stream mean");
}

/// `q4e_top_slices_kernel` takes each slice's `k` best entries, best first and
/// ties to the lower id, which merged on the host are the row's own top `k`.
/// Ties are planted inside a slice and across a slice boundary; a short last
/// slice pads with -inf and an id past the row.
///
/// # Panics
///
/// When a slice's entry is not the host's.
pub fn the_sliced_top_keeps_each_slice_best_first(gpu: &Gpu) {
    const ROWS: usize = 3;
    const K: u32 = 16;
    const SLICES: u32 = 32;
    let v = VOCAB as usize;
    let mut logits = signal(ROWS * v, 23);
    logits[100] = 4.0;
    logits[101] = 4.0;
    logits[7_759] = 3.5;
    logits[7_760] = 3.5;
    logits[v + v - 1] = 5.0;
    let logits_dev = gpu.upload(&f32_bytes(&logits));
    let entries = ROWS * (SLICES * K) as usize;
    let values_dev = gpu.upload(&vec![0u8; entries * 4]);
    let ids_dev = gpu.upload(&vec![0u8; entries * 4]);
    run(
        gpu,
        KernelRef::new(MOD_MISC, "q4e_top_slices_kernel"),
        &LaunchArgs {
            grid: Extent3::new(SLICES, ROWS as u32, 1),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(logits_dev.view()),
                KernelArg::U32(VOCAB),
                KernelArg::U32(K),
                KernelArg::U32(SLICES),
                KernelArg::Buffer(values_dev.view()),
                KernelArg::Buffer(ids_dev.view()),
            ],
        },
    );
    drain(gpu);
    let values = gpu.read_f32(&values_dev);
    let ids: Vec<u32> = gpu.read_f32(&ids_dev).iter().map(|f| f.to_bits()).collect();
    let span = (v as u32).div_ceil(SLICES) as usize;
    for row in 0..ROWS {
        for slice in 0..SLICES as usize {
            let lo = slice * span;
            let hi = (lo + span).min(v);
            let mut want: Vec<(f32, u32)> =
                (lo..hi).map(|i| (logits[row * v + i], i as u32)).collect();
            want.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for t in 0..K as usize {
                let at = (row * SLICES as usize + slice) * K as usize + t;
                let (wv, wi) = want
                    .get(t)
                    .copied()
                    .unwrap_or((f32::NEG_INFINITY, u32::MAX));
                assert_eq!(ids[at], wi, "row {row} slice {slice} rank {t}: id");
                assert_eq!(
                    values[at].to_bits(),
                    wv.to_bits(),
                    "row {row} slice {slice} rank {t}: value"
                );
            }
        }
    }
}

/// `q4e_argmax_rows_kernel` picks each row's maximum over the whole
/// vocabulary, a tie going to the lower index even when the two sit in
/// different threads and different warps.
///
/// # Panics
///
/// When a row's pick is not the oracle's.
pub fn the_greedy_pick_takes_the_lowest_index_of_the_maximum(gpu: &Gpu) {
    greedy_pick(gpu, VOCAB);
}

/// The same picks over a width that is not a multiple of four, which the
/// kernel reads one float at a time rather than four.
///
/// # Panics
///
/// When a row's pick is not the oracle's.
pub fn the_greedy_pick_reads_an_odd_width_one_float_at_a_time(gpu: &Gpu) {
    greedy_pick(gpu, VOCAB + 1);
}

/// Six rows of `vocab` logits with the maximum placed where each way of
/// walking a row could get the tie wrong, graded against a host scan.
fn greedy_pick(gpu: &Gpu, vocab: u32) {
    const ROWS: usize = 6;
    let v = vocab as usize;
    let mut logits = signal(ROWS * v, 11);
    // Row 0: the maximum at the last index, the tail of the strided loop.
    logits[v - 1] = 3.0;
    // Row 1: a tie between two threads of different warps; the lower wins.
    logits[v + 70_001] = 2.5;
    logits[v + 5_000] = 2.5;
    // Row 2: a tie between two elements one thread visits a block apart.
    logits[2 * v + 7] = 2.0;
    logits[2 * v + 7 + BLOCK_WIDE as usize] = 2.0;
    // Row 4: a tie inside one sixteen-byte load.
    logits[4 * v + 14] = 2.0;
    logits[4 * v + 13] = 2.0;
    // Row 5: a tie between two loads one thread holds in the same round.
    logits[5 * v + 4 * (3 + BLOCK_WIDE as usize) + 1] = 2.0;
    logits[5 * v + 13] = 2.0;
    // Row 3: left as fixture data, whose maximum is wherever it lands.
    let want: Vec<i32> = logits
        .chunks(v)
        .map(|row| {
            let mut best = (f32::NEG_INFINITY, -1_i32);
            for (i, &x) in row.iter().enumerate() {
                if x > best.0 {
                    best = (x, i as i32);
                }
            }
            best.1
        })
        .collect();
    assert_eq!(
        [want[0], want[1], want[2], want[4], want[5]],
        [v as i32 - 1, 5_000, 7, 13, 13],
        "the oracle's own picks"
    );
    let logits_dev = gpu.upload(&f32_bytes(&logits));
    let out = gpu.upload(&i32_bytes(&[i32::MIN; ROWS]));
    launch(
        gpu,
        KernelRef::new(MOD_MISC, "q4e_argmax_rows_kernel"),
        Extent3::new(ROWS as u32, 1, 1),
        BLOCK_WIDE,
        &[
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(logits_dev.view()),
            KernelArg::U32(vocab),
        ],
    );
    let got: Vec<i32> = gpu
        .read(&out)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    assert_eq!(got, want, "argmax rows");
}

/// An `f16` vocabulary table and its widened copy.
fn table() -> (Vec<f16>, Vec<f32>) {
    let t: Vec<f16> = signal((TABLE_ROWS * HIDDEN) as usize, 13)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let wide = t.iter().map(|v| v.to_f32()).collect();
    (t, wide)
}

/// `embed_tokens_hc_kernel` at one stream count: ids below zero or past the
/// vocabulary read row zero.
fn gather_case(gpu: &Gpu, n_hc: u32) {
    let (t, wide) = table();
    let ids = [5_i32, -1, 63, 64, 0, 17, i32::MAX];
    let h = HIDDEN as usize;
    let n = ids.len() * n_hc as usize * h;
    let table_dev = gpu.upload(&f16_bytes(&t));
    let ids_dev = gpu.upload(&i32_bytes(&ids));
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; n]));
    launch(
        gpu,
        KernelRef::new(MOD_EMBED, "embed_tokens_hc_kernel"),
        flat(n as u64),
        BLOCK_LINEAR,
        &[
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(ids_dev.view()),
            KernelArg::Buffer(table_dev.view()),
            KernelArg::U32(TABLE_ROWS),
            KernelArg::U32(ids.len() as u32),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(n_hc),
        ],
    );
    let want: Vec<f32> = ids
        .iter()
        .flat_map(|&id| {
            // An id below zero fails the conversion, as it fails the kernel's
            // sign test.
            let row = match u32::try_from(id) {
                Ok(row) if row < TABLE_ROWS => row as usize,
                _ => 0,
            };
            (0..n_hc).flat_map(move |_| (0..h).map(move |e| row * h + e))
        })
        .map(|i| wide[i])
        .collect();
    same(
        &gpu.read_f32(&out),
        &want,
        &format!("embed_tokens_hc n_hc {n_hc}"),
    );
}

/// The gather the stack launches: one copy of a row per token.
pub fn the_half_gather_reads_one_row_per_token(gpu: &Gpu) {
    gather_case(gpu, 1);
}

/// The gather's fan-out across four streams.
pub fn the_half_gather_fans_a_row_out_across_streams(gpu: &Gpu) {
    gather_case(gpu, STREAMS);
}

/// `embed_token_hc_kernel`: one token's row tiled across four streams.
pub fn the_single_token_gather_tiles_its_row(gpu: &Gpu) {
    let (t, wide) = table();
    let token = 41_u32;
    let h = HIDDEN as usize;
    let n = STREAMS as usize * h;
    let table_dev = gpu.upload(&f16_bytes(&t));
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; n]));
    launch(
        gpu,
        KernelRef::new(MOD_EMBED, "embed_token_hc_kernel"),
        flat(n as u64),
        BLOCK_LINEAR,
        &[
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(table_dev.view()),
            KernelArg::U32(token),
            KernelArg::U32(HIDDEN),
            KernelArg::U32(STREAMS),
        ],
    );
    let want: Vec<f32> = (0..n).map(|i| wide[token as usize * h + i % h]).collect();
    same(&gpu.read_f32(&out), &want, "embed_token_hc");
}
