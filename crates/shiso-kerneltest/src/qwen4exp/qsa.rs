//! The sparse family's qsa kernels: cases graded against a host restatement of the CUDA bodies.
//!
//! The attention block at the real geometry - 24 query heads over 2 key heads,
//! head width 256, a 64-wide partial rotary in three mRoPE sections at base
//! 1e7 - reading a paged f16 cache through a page table the pool handed out of
//! order, whose second entry is the pool's last page. A kernel that walks the
//! cache as if a session's pages were contiguous reads the wrong keys from the
//! first page boundary on, and a decode at position 3000 crosses eleven.
//!
//! The oracles are `f64` loops over the same stored values: the cache as the
//! f16 it holds, and for the split and gathered kernels the query as the f16
//! they stage it as (scaled first, then rounded), so what is left between the
//! device and the oracle is the order of `f32` sums and the fast `exp`.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "kernel arguments are u32 while the fixtures index in usize and store \
              positions and page ids as the kernel's i32; every value here is a small \
              non-negative literal, and positions below 2^24 are exact in f32"
)]
#![expect(
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "the oracles restate kernel bodies whose names are single letters (q, k, v, \
              m, l) and paired key/value buffers; renaming them would hide the \
              correspondence being graded"
)]

use half::f16;
use shiso_core::page::PAGE_TOKENS;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The dense module, mirroring `shiso_engine::sparse::kernels::MOD_QSA`.
const MODULE: &str = "q4e_qsa";
/// The selected-block module, mirroring `kernels::MOD_QSA_SPARSE`.
const MODULE_SPARSE: &str = "q4e_qsa_sparse";

/// Query heads.
const HEADS: u32 = 24;
/// Key/value heads.
const KV_HEADS: u32 = 2;
/// Head width, which is also `kernels::ATTN_SPLIT_HD`.
const HD: u32 = 256;
/// Rotated dims per head (partial rotary 0.25).
const N_ROT: u32 = 64;
/// mRoPE sections (time, height, width), in rotary pairs.
const SECTIONS: [u32; 3] = [11, 11, 10];
/// Rotary base.
const THETA: f32 = 1.0e7;
/// `RMSNorm` epsilon.
const EPS: f32 = 1.0e-6;

/// Threads in the norm and store kernels, mirroring `kernels::BLOCK_NARROW`.
const BLOCK_NARROW: u32 = 128;
/// Threads in the elementwise and sparse kernels, mirroring `kernels::BLOCK_LINEAR`.
const BLOCK_LINEAR: u32 = 256;
/// Warps in the dense attention block, mirroring `kernels::ATTN_WARPS`.
const ATTN_WARPS: u32 = 8;
/// Key splits of the split path, mirroring `kernels::ATTN_SPLITS`.
const ATTN_SPLITS: u32 = 48;
/// Splits of the gathered path, mirroring `kernels::ATTN_GSPLITS`.
const ATTN_GSPLITS: u32 = 8;
/// Floats per partial slab: `m, l`, six pad, 256 accumulators.
const PART_STRIDE: u32 = HD + 8;
/// Slots per selection row, mirroring `kernels::IDX_WIDTH`.
const IDX_WIDTH: u32 = 2_048 + 3;
/// Threadgroup memory the engine requests for the split and gathered kernels,
/// `kernels::qsa_sparse_smem()`: the CUDA layout's key tile, value tile, the
/// group's f16 queries and the weight rows.
///
/// Past a 32 KiB device. The Metal port shares the K and V tiles in a static
/// array and declares no dynamic binding, which is how that backend knows not
/// to apply this; the case asks what the engine asks.
const SPARSE_SMEM: u32 = HD * 32 * 2 * 2 + 16 * HD * 2 + 8 * 32 * 4;

/// Pages in the fixture pool.
const POOL_PAGES: u32 = 20;
/// The session's page table: logical page `i` lives in pool page `TABLE[i]`.
/// No neighbours are adjacent and the second entry is the pool's last page.
const TABLE: [i32; 16] = [13, 19, 2, 7, 0, 11, 5, 17, 9, 3, 15, 1, 18, 6, 12, 4];

/// `f16` values as little-endian bytes.
fn f16_bytes(values: &[f16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `i32` values as little-endian bytes.
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `f16` values from little-endian bytes.
fn f16_values(bytes: &[u8]) -> Vec<f16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|w| f16::from_le_bytes(*w))
        .collect()
}

/// Physical cache row of logical position `p` through `table`.
fn kv_row(table: &[i32], p: u32) -> usize {
    table[(p / PAGE_TOKENS) as usize] as usize * PAGE_TOKENS as usize + (p % PAGE_TOKENS) as usize
}

/// Asserts every element agrees within `tol` absolute plus `tol` relative,
/// naming the first that does not.
fn agree(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = tol * (1.0 + w.abs());
        assert!(
            (f64::from(g) - w).abs() <= bound,
            "{what}: element {i} (head-row {}, dim {}) is {g}, expected {w}",
            i / HD as usize,
            i % HD as usize,
        );
    }
}

// ---------------------------------------------------------------------------
// query / key norm and rotary
// ---------------------------------------------------------------------------

/// `RMSNorm` with weight `w`, then the interleaved mRoPE of `q4e_rope_mrope`
/// over the first `N_ROT` dims, the angle formed in `f32` as the kernel forms
/// it.
fn norm_rope(x: &[f32], w: &[f32], mrope: [i32; 3]) -> Vec<f64> {
    let ss: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let scale = 1.0 / (ss / f64::from(HD) + f64::from(EPS)).sqrt();
    let mut v: Vec<f64> = x
        .iter()
        .zip(w)
        .map(|(&a, &b)| f64::from(a) * scale * f64::from(b))
        .collect();
    let half = (N_ROT / 2) as usize;
    for i in 0..half {
        let mut axis = i % 3;
        if (i / 3) as u32 >= SECTIONS[axis] {
            axis = 0;
        }
        let p = mrope[axis] as u32 as f32;
        let theta = p * THETA.powf(-((2 * i) as f32) / N_ROT as f32);
        let (s, c) = f64::from(theta).sin_cos();
        let (x0, x1) = (v[i], v[i + half]);
        v[i] = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
    v
}

/// `mrope` rows for `positions`: height and width offset from time, as an
/// image token's would be, so a rotary that used the wrong axis shows.
fn mrope_rows(positions: &[i32]) -> Vec<[i32; 3]> {
    positions.iter().map(|&p| [p, p - 7, p - 3]).collect()
}

/// The query norm and rotary splits a strided `[q | gate]` row per head,
/// normalizes and rotates the query and passes the gate through.
pub fn the_query_norm_rope_splits_rotates_and_gates(gpu: &Gpu) {
    let positions = [3_000, 3_001, 3_002, 3_003];
    let n_tok = positions.len() as u32;
    let mrope = mrope_rows(&positions);
    let qkv = signal((n_tok * HEADS * 2 * HD) as usize, 3);
    let w: Vec<f32> = signal(HD as usize, 5)
        .iter()
        .map(|v| 1.0 + 0.5 * v)
        .collect();
    let out_len = (n_tok * HEADS * HD) as usize;
    let q_out = gpu.upload(&f32_bytes(&vec![f32::NAN; out_len]));
    let gate_out = gpu.upload(&f32_bytes(&vec![f32::NAN; out_len]));
    let qkv_dev = gpu.upload(&f32_bytes(&qkv));
    let w_dev = gpu.upload(&f32_bytes(&w));
    let flat: Vec<i32> = mrope.iter().flatten().copied().collect();
    let mrope_dev = gpu.upload(&i32_bytes(&flat));
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_qsa_q_norm_rope_kernel"),
        &LaunchArgs {
            grid: Extent3::new(HEADS, n_tok, 1),
            block: Extent3::new(BLOCK_NARROW, 1, 1),
            smem_bytes: HD * 4,
            args: &[
                KernelArg::Buffer(q_out.view()),
                KernelArg::Buffer(gate_out.view()),
                KernelArg::Buffer(qkv_dev.view()),
                KernelArg::Buffer(w_dev.view()),
                KernelArg::U32(HD),
                KernelArg::U32(HEADS),
                KernelArg::U32(N_ROT),
                KernelArg::F32(THETA),
                KernelArg::Buffer(mrope_dev.view()),
                KernelArg::U32(SECTIONS[0]),
                KernelArg::U32(SECTIONS[1]),
                KernelArg::U32(SECTIONS[2]),
                KernelArg::F32(EPS),
            ],
        },
    );
    drain(gpu);
    let mut want_q = Vec::with_capacity(out_len);
    let mut want_gate = Vec::with_capacity(out_len);
    let hd = HD as usize;
    for (t, row) in mrope.iter().enumerate() {
        for h in 0..HEADS as usize {
            let src = (t * HEADS as usize + h) * 2 * hd;
            want_q.extend(norm_rope(&qkv[src..src + hd], &w, *row));
            want_gate.extend(qkv[src + hd..src + 2 * hd].iter().map(|&v| f64::from(v)));
        }
    }
    agree(&gpu.read_f32(&q_out), &want_q, 2e-3, "q_norm_rope query");
    agree(
        &gpu.read_f32(&gate_out),
        &want_gate,
        0.0,
        "q_norm_rope gate",
    );
}

/// The store writes each normed, rotated key and its value at the row the page
/// table names - across a page boundary - and refuses a row past the pool.
///
/// # Panics
///
/// When a stored row disagrees with the oracle or a refused row was written.
pub fn the_kv_store_follows_the_page_table_and_refuses_past_the_pool(gpu: &Gpu) {
    // Logical page 2 maps to pool page 19, and the pool is declared as 19
    // pages, so the last row's write must be refused.
    let table = [3, 7, 19];
    let pool_slots = 19 * PAGE_TOKENS;
    let positions = [254, 255, 256, 257, 512];
    let n_tok = positions.len() as u32;
    let mrope = mrope_rows(&positions);
    let kv_len = (n_tok * KV_HEADS * HD) as usize;
    let k = signal(kv_len, 7);
    let v = signal(kv_len, 9);
    let kw: Vec<f32> = signal(HD as usize, 13)
        .iter()
        .map(|x| 1.0 + 0.25 * x)
        .collect();
    let plane = (POOL_PAGES * PAGE_TOKENS * KV_HEADS * HD) as usize;
    let sentinel = f16::from_f32(-7.5);
    let k_cache = gpu.upload(&f16_bytes(&vec![sentinel; plane]));
    let v_cache = gpu.upload(&f16_bytes(&vec![sentinel; plane]));
    let k_dev = gpu.upload(&f32_bytes(&k));
    let v_dev = gpu.upload(&f32_bytes(&v));
    let kw_dev = gpu.upload(&f32_bytes(&kw));
    let pos_dev = gpu.upload(&i32_bytes(&positions));
    let flat: Vec<i32> = mrope.iter().flatten().copied().collect();
    let mrope_dev = gpu.upload(&i32_bytes(&flat));
    let pages_dev = gpu.upload(&i32_bytes(&table));
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_qsa_store_kv_kernel"),
        &LaunchArgs {
            grid: Extent3::new(KV_HEADS, n_tok, 1),
            block: Extent3::new(BLOCK_NARROW, 1, 1),
            smem_bytes: HD * 4,
            args: &[
                KernelArg::Buffer(k_cache.view()),
                KernelArg::Buffer(v_cache.view()),
                KernelArg::Buffer(k_dev.view()),
                KernelArg::Buffer(v_dev.view()),
                KernelArg::Buffer(kw_dev.view()),
                KernelArg::U32(HD),
                KernelArg::U32(KV_HEADS),
                KernelArg::U32(N_ROT),
                KernelArg::F32(THETA),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(mrope_dev.view()),
                KernelArg::U32(SECTIONS[0]),
                KernelArg::U32(SECTIONS[1]),
                KernelArg::U32(SECTIONS[2]),
                KernelArg::Buffer(pages_dev.view()),
                KernelArg::U32(pool_slots),
                KernelArg::F32(EPS),
            ],
        },
    );
    drain(gpu);
    let got_k = f16_values(&gpu.read(&k_cache));
    let got_v = f16_values(&gpu.read(&v_cache));
    let hd = HD as usize;
    let mut written = 0usize;
    for (t, &p) in positions.iter().enumerate() {
        let row = kv_row(&table, p as u32);
        for h in 0..KV_HEADS as usize {
            let src = (t * KV_HEADS as usize + h) * hd;
            let slot = (row * KV_HEADS as usize + h) * hd;
            let got_k_row: Vec<f32> = got_k[slot..slot + hd].iter().map(|x| x.to_f32()).collect();
            let got_v_row: Vec<f32> = got_v[slot..slot + hd].iter().map(|x| x.to_f32()).collect();
            if row >= pool_slots as usize {
                assert!(
                    got_k[slot..slot + hd]
                        .iter()
                        .chain(&got_v[slot..slot + hd])
                        .all(|x| x.to_bits() == sentinel.to_bits()),
                    "store_kv: position {p} maps past the pool and must not be written",
                );
                continue;
            }
            written += 1;
            let want_k = norm_rope(&k[src..src + hd], &kw, mrope[t]);
            // The cache is f16: 2^-11 relative from the rounding alone.
            agree(
                &got_k_row,
                &want_k,
                3e-3,
                &format!("store_kv key, position {p} head {h}"),
            );
            let want_v: Vec<f64> = v[src..src + hd]
                .iter()
                .map(|&x| f64::from(f16::from_f32(x).to_f32()))
                .collect();
            agree(
                &got_v_row,
                &want_v,
                0.0,
                &format!("store_kv value, position {p} head {h}"),
            );
        }
    }
    assert_eq!(
        written, 8,
        "store_kv: four in-pool positions over two heads"
    );
}

// ---------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------

/// A resident cache over the whole fixture pool, and its page table.
struct Cache {
    k: Vec<f16>,
    v: Vec<f16>,
    k_dev: DeviceBuffer,
    v_dev: DeviceBuffer,
    pages_dev: DeviceBuffer,
}

impl Cache {
    fn new(gpu: &Gpu) -> Self {
        let plane = (POOL_PAGES * PAGE_TOKENS * KV_HEADS * HD) as usize;
        let k: Vec<f16> = signal(plane, 17).into_iter().map(f16::from_f32).collect();
        let v: Vec<f16> = signal(plane, 19).into_iter().map(f16::from_f32).collect();
        Self {
            k_dev: gpu.upload(&f16_bytes(&k)),
            v_dev: gpu.upload(&f16_bytes(&v)),
            pages_dev: gpu.upload(&i32_bytes(&TABLE)),
            k,
            v,
        }
    }

    /// Softmax attention of query `q` (already scaled) over the listed
    /// positions of key head `hkv`, in `f64`.
    fn attend(&self, q: &[f64], hkv: usize, keys: impl Iterator<Item = u32>) -> Vec<f64> {
        let hd = HD as usize;
        let rows: Vec<usize> = keys
            .map(|p| (kv_row(&TABLE, p) * KV_HEADS as usize + hkv) * hd)
            .collect();
        let scores: Vec<f64> = rows
            .iter()
            .map(|&b| {
                q.iter()
                    .zip(&self.k[b..b + hd])
                    .map(|(a, k)| a * f64::from(k.to_f32()))
                    .sum()
            })
            .collect();
        let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
        let den: f64 = weights.iter().sum();
        let mut out = vec![0.0f64; hd];
        for (&b, w) in rows.iter().zip(&weights) {
            for (o, v) in out.iter_mut().zip(&self.v[b..b + hd]) {
                *o += w * f64::from(v.to_f32());
            }
        }
        out.iter().map(|o| o / den).collect()
    }
}

/// Queries for `n_tok` rows, `[t][h][HD]`, spread to four so the softmax is
/// peaked enough that a wrong key moves the answer.
fn queries(n_tok: u32, seed: u32) -> Vec<f32> {
    signal((n_tok * HEADS * HD) as usize, seed)
        .into_iter()
        .map(|v| 4.0 * v)
        .collect()
}

/// The query of row `t`, head `h`, scaled by `1/sqrt(HD)` - and rounded to
/// f16 after scaling when `staged`, as the split and gathered kernels stage it.
fn scaled_query(q: &[f32], t: usize, h: usize, staged: bool) -> Vec<f64> {
    let hd = HD as usize;
    let base = (t * HEADS as usize + h) * hd;
    let scale = 1.0f32 / (HD as f32).sqrt();
    q[base..base + hd]
        .iter()
        .map(|&x| {
            let s = x * scale;
            if staged {
                f64::from(f16::from_f32(s).to_f32())
            } else {
                f64::from(x) / f64::from(HD).sqrt()
            }
        })
        .collect()
}

/// The causal oracle for every (row, head) of a pass at `positions`.
fn causal_oracle(cache: &Cache, q: &[f32], positions: &[i32], staged: bool) -> Vec<f64> {
    let group = (HEADS / KV_HEADS) as usize;
    let mut want = Vec::new();
    for (t, &p) in positions.iter().enumerate() {
        for h in 0..HEADS as usize {
            let qs = scaled_query(q, t, h, staged);
            want.extend(cache.attend(&qs, h / group, 0..=p as u32));
        }
    }
    want
}

/// Runs the dense attention over `positions` and grades it.
fn dense_case(gpu: &Gpu, positions: &[i32], seed: u32, what: &str) {
    let cache = Cache::new(gpu);
    let n_tok = positions.len() as u32;
    let q = queries(n_tok, seed);
    let out_len = (n_tok * HEADS * HD) as usize;
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; out_len]));
    let q_dev = gpu.upload(&f32_bytes(&q));
    let pos_dev = gpu.upload(&i32_bytes(positions));
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_qsa_attention_kernel"),
        &LaunchArgs {
            grid: Extent3::new(HEADS, n_tok, 1),
            block: Extent3::new(32 * ATTN_WARPS, 1, 1),
            smem_bytes: ATTN_WARPS * HD * 4,
            args: &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(cache.k_dev.view()),
                KernelArg::Buffer(cache.v_dev.view()),
                KernelArg::Buffer(q_dev.view()),
                KernelArg::U32(HD),
                KernelArg::U32(HEADS),
                KernelArg::U32(KV_HEADS),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(cache.pages_dev.view()),
                KernelArg::U32(n_tok),
            ],
        },
    );
    drain(gpu);
    let want = causal_oracle(&cache, &q, positions, false);
    agree(&gpu.read_f32(&out), &want, 1e-4, what);
}

/// One decode row at position 3000 reads twelve pages through the scattered
/// table.
pub fn the_dense_attention_decodes_across_scattered_pages(gpu: &Gpu) {
    dense_case(gpu, &[3_000], 23, "dense attention decode");
}

/// A short prefill chunk that crosses a page boundary, each row masked at its
/// own position; the first rows leave some simdgroups with no position.
pub fn the_dense_attention_prefills_a_chunk_causally(gpu: &Gpu) {
    let positions: Vec<i32> = (250..259).chain([3, 5]).collect();
    dense_case(gpu, &positions, 29, "dense attention prefill");
}

/// Launches the combine over `n_tok` rows' partial slabs.
fn combine(gpu: &Gpu, out: &DeviceBuffer, part: &DeviceBuffer, n_tok: u32, splits: u32) {
    run(
        gpu,
        KernelRef::new(MODULE_SPARSE, "q4e_qsa_attention_combine_kernel"),
        &LaunchArgs {
            grid: Extent3::new(HEADS * n_tok, 1, 1),
            block: Extent3::new(HD, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(part.view()),
                KernelArg::U32(splits),
            ],
        },
    );
}

/// Runs the split attention at `splits` over `positions` (folding when there
/// is more than one split) and grades it.
fn split_case(gpu: &Gpu, positions: &[i32], splits: u32, seed: u32, what: &str) {
    let cache = Cache::new(gpu);
    let n_tok = positions.len() as u32;
    let q = queries(n_tok, seed);
    let out_len = (n_tok * HEADS * HD) as usize;
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; out_len]));
    let part_len = (n_tok * HEADS * splits * PART_STRIDE) as usize;
    let part = gpu.upload(&f32_bytes(&vec![f32::NAN; part_len]));
    let q_dev = gpu.upload(&f32_bytes(&q));
    let pos_dev = gpu.upload(&i32_bytes(positions));
    run(
        gpu,
        KernelRef::new(MODULE_SPARSE, "q4e_qsa_attention_split_kernel"),
        &LaunchArgs {
            grid: Extent3::new(n_tok, splits, KV_HEADS),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: SPARSE_SMEM,
            args: &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(part.view()),
                KernelArg::Buffer(cache.k_dev.view()),
                KernelArg::Buffer(cache.v_dev.view()),
                KernelArg::Buffer(q_dev.view()),
                KernelArg::U32(HEADS),
                KernelArg::U32(KV_HEADS),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(cache.pages_dev.view()),
                KernelArg::U32(n_tok),
                KernelArg::U32(splits),
            ],
        },
    );
    if splits > 1 {
        combine(gpu, &out, &part, n_tok, splits);
    }
    drain(gpu);
    let want = causal_oracle(&cache, &q, positions, true);
    agree(&gpu.read_f32(&out), &want, 1e-4, what);
}

/// The split path at the engine's 48 splits, one decode row at 3000: 94 key
/// tiles, two a split, the last split empty; then the fold.
pub fn the_split_attention_decodes_and_folds(gpu: &Gpu) {
    split_case(gpu, &[3_000], ATTN_SPLITS, 31, "split attention decode");
}

/// A verify-shaped pass at the start of a session: every row fits one tile, so
/// 47 of the 48 splits leave empty partials the fold must skip.
pub fn the_split_attention_verifies_near_the_start(gpu: &Gpu) {
    split_case(
        gpu,
        &[0, 1, 2, 3],
        ATTN_SPLITS,
        37,
        "split attention verify",
    );
}

/// At one split the kernel normalizes and writes the destination itself; the
/// rows straddle a page boundary.
pub fn the_split_attention_writes_directly_at_one_split(gpu: &Gpu) {
    split_case(
        gpu,
        &[510, 511, 512, 513],
        1,
        41,
        "split attention, one split",
    );
}

/// A selection row: positions in `0..=last` whose pooled block is picked by a
/// fixed pattern, plus the row's own last few, capped at `cap`.
fn selection(last: u32, cap: usize) -> Vec<i32> {
    let mut list: Vec<i32> = (0..=last)
        .filter(|p| (p / 4) % 11 == 3 || (p / 4) % 29 == 0 || last - p < 5)
        .map(|p| p as i32)
        .collect();
    list.truncate(cap);
    list
}

/// The gathered attention reads exactly the listed positions through the page
/// table, across several of its eight slices, and the fold merges them.
///
/// # Panics
///
/// When an output disagrees with the oracle.
pub fn the_gathered_attention_reads_the_selection(gpu: &Gpu) {
    let cache = Cache::new(gpu);
    let lists = [selection(3_000, 2_048), selection(700, 37)];
    let n_tok = lists.len() as u32;
    let q = queries(n_tok, 43);
    let mut tokens = vec![-1i32; (n_tok * IDX_WIDTH) as usize];
    let mut counts = Vec::new();
    for (t, list) in lists.iter().enumerate() {
        let base = t * IDX_WIDTH as usize;
        tokens[base..base + list.len()].copy_from_slice(list);
        counts.push(list.len() as i32);
    }
    let out_len = (n_tok * HEADS * HD) as usize;
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; out_len]));
    let part_len = (n_tok * HEADS * ATTN_GSPLITS * PART_STRIDE) as usize;
    let part = gpu.upload(&f32_bytes(&vec![f32::NAN; part_len]));
    let q_dev = gpu.upload(&f32_bytes(&q));
    let tokens_dev = gpu.upload(&i32_bytes(&tokens));
    let counts_dev = gpu.upload(&i32_bytes(&counts));
    run(
        gpu,
        KernelRef::new(MODULE_SPARSE, "q4e_qsa_attention_gather_kernel"),
        &LaunchArgs {
            grid: Extent3::new(ATTN_GSPLITS, KV_HEADS, n_tok),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: SPARSE_SMEM,
            args: &[
                KernelArg::Buffer(part.view()),
                KernelArg::Buffer(cache.k_dev.view()),
                KernelArg::Buffer(cache.v_dev.view()),
                KernelArg::Buffer(q_dev.view()),
                KernelArg::Buffer(tokens_dev.view()),
                KernelArg::Buffer(counts_dev.view()),
                KernelArg::Buffer(cache.pages_dev.view()),
                KernelArg::U32(IDX_WIDTH),
                KernelArg::U32(HEADS),
                KernelArg::U32(KV_HEADS),
                KernelArg::U32(ATTN_GSPLITS),
            ],
        },
    );
    combine(gpu, &out, &part, n_tok, ATTN_GSPLITS);
    drain(gpu);
    let group = (HEADS / KV_HEADS) as usize;
    let mut want = Vec::new();
    for (t, list) in lists.iter().enumerate() {
        for h in 0..HEADS as usize {
            let qs = scaled_query(&q, t, h, true);
            want.extend(cache.attend(&qs, h / group, list.iter().map(|&p| p as u32)));
        }
    }
    assert!(
        lists[0].len() > 8 * 32,
        "the long row spans all eight slices"
    );
    agree(&gpu.read_f32(&out), &want, 1e-4, "gathered attention");
}

/// The output gate multiplies by `sigmoid(gate)` and stops at `n`.
pub fn the_output_gate_is_a_sigmoid_and_stops_at_n(gpu: &Gpu) {
    let n: u64 = 3 * u64::from(HEADS * HD) - 100;
    let len = 3 * (HEADS * HD) as usize;
    let x = signal(len, 47);
    let gate: Vec<f32> = signal(len, 53).into_iter().map(|g| 6.0 * g).collect();
    let x_dev = gpu.upload(&f32_bytes(&x));
    let gate_dev = gpu.upload(&f32_bytes(&gate));
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_qsa_gate_kernel"),
        &LaunchArgs {
            grid: Extent3::new(n.div_ceil(u64::from(BLOCK_LINEAR)) as u32, 1, 1),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(x_dev.view()),
                KernelArg::Buffer(gate_dev.view()),
                KernelArg::U64(n),
            ],
        },
    );
    drain(gpu);
    let want: Vec<f64> = x
        .iter()
        .zip(&gate)
        .enumerate()
        .map(|(i, (&a, &g))| {
            if (i as u64) < n {
                f64::from(a) / (1.0 + (-f64::from(g)).exp())
            } else {
                f64::from(a)
            }
        })
        .collect();
    agree(&gpu.read_f32(&x_dev), &want, 1e-5, "output gate");
}

// ---------------------------------------------------------------------------
// the tensor-core attention
// ---------------------------------------------------------------------------

/// The module of the exact tensor-core attention, mirroring
/// `kernels::MOD_EXACT`.
const MODULE_EXACT: &str = "q4e_exact";
/// Keys per tile of the tensor-core attention, `kernels::TC_KT`.
const TC_KT: u32 = 64;
/// Shared memory of the ported tensor-core attention, `kernels::TC_SMEM`: the
/// query, key and value tiles, the score rows, the f16 weights and the rescale
/// factors.
const TC_SMEM: u32 = (16 + 2 * TC_KT) * HD * 2 + 16 * TC_KT * 4 + 16 * TC_KT * 2 + 16 * 4;
/// Shared memory of the exact one, `kernels::TC_EXACT_SMEM`: the same plus the
/// low half of every weight.
const TC_EXACT_SMEM: u32 = TC_SMEM + 16 * TC_KT * 2;
/// Splits the sparse entry takes at a row count, `kernels::tc_splits_sparse`.
const fn tc_splits_sparse(n_tok: u32) -> u32 {
    if n_tok * KV_HEADS >= 64 {
        1
    } else {
        ATTN_GSPLITS
    }
}

/// The error budget of the tensor-core attention against the `f64` oracle.
///
/// The ported kernel rounds each softmax weight to f16 before the P V product
/// while the denominator sums the f32 weights, a relative error up to 2^-11 a
/// weight that nothing cancels; over a short key list that is ~1e-4 on the
/// output. The exact kernel carries the weight as two f16 halves, which leaves
/// the f32 sum order and the fast `exp`. Measured on a GB10 over the cases
/// below (gathered, dense): ported 1.2e-4 and 1.6e-4, exact 3.5e-7 and 6.0e-7.
/// The budget sits between the two, so it says which kernel ran.
const TC_EXACT_TOL: f64 = 2e-5;

/// Which tensor-core attention entry a case runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TcEntry {
    /// `q4e_qsa_attention_tc_kernel`, the ported kernel the engine no longer
    /// routes to.
    Ported,
    /// `q4e_qsa_attention_tc_exact_kernel`.
    Exact,
}

impl TcEntry {
    fn kernel(self) -> KernelRef {
        match self {
            Self::Ported => KernelRef::new(MODULE_SPARSE, "q4e_qsa_attention_tc_kernel"),
            Self::Exact => KernelRef::new(MODULE_EXACT, "q4e_qsa_attention_tc_exact_kernel"),
        }
    }

    const fn smem(self) -> u32 {
        match self {
            Self::Ported => TC_SMEM,
            Self::Exact => TC_EXACT_SMEM,
        }
    }
}

/// The largest `|got - want| / (1 + |want|)` over a pass, and where it is.
fn worst(got: &[f32], want: &[f64]) -> (f64, usize) {
    assert_eq!(got.len(), want.len(), "lengths");
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (f64::from(g) - w).abs() / (1.0 + w.abs()))
        .enumerate()
        .fold((0.0, 0), |acc, (i, e)| if e > acc.0 { (e, i) } else { acc })
}

/// Runs a tensor-core attention over the selection `lists`, sliced the way the
/// sparse entry slices it, and returns the folded output with its oracle.
fn tc_gathered(gpu: &Gpu, entry: TcEntry, lists: &[Vec<i32>], seed: u32) -> (Vec<f32>, Vec<f64>) {
    let cache = Cache::new(gpu);
    let n_tok = lists.len() as u32;
    let splits = tc_splits_sparse(n_tok);
    let q = queries(n_tok, seed);
    let mut tokens = vec![-1i32; (n_tok * IDX_WIDTH) as usize];
    let mut counts = Vec::new();
    for (t, list) in lists.iter().enumerate() {
        let base = t * IDX_WIDTH as usize;
        tokens[base..base + list.len()].copy_from_slice(list);
        counts.push(list.len() as i32);
    }
    // Zero positions: the gathered entry reads its list, never a position.
    let positions = vec![0i32; n_tok as usize];
    let out_len = (n_tok * HEADS * HD) as usize;
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; out_len]));
    let part_len = (n_tok * HEADS * splits * PART_STRIDE) as usize;
    let part = gpu.upload(&f32_bytes(&vec![f32::NAN; part_len]));
    let q_dev = gpu.upload(&f32_bytes(&q));
    let tokens_dev = gpu.upload(&i32_bytes(&tokens));
    let counts_dev = gpu.upload(&i32_bytes(&counts));
    let pos_dev = gpu.upload(&i32_bytes(&positions));
    // One slice writes the destination; more write slabs the combine folds.
    let (out_arg, part_arg) = if splits == 1 {
        (KernelArg::Buffer(out.view()), KernelArg::U64(0))
    } else {
        (KernelArg::U64(0), KernelArg::Buffer(part.view()))
    };
    run(
        gpu,
        entry.kernel(),
        &LaunchArgs {
            grid: Extent3::new(splits, KV_HEADS, n_tok),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: entry.smem(),
            args: &[
                out_arg,
                part_arg,
                KernelArg::Buffer(cache.k_dev.view()),
                KernelArg::Buffer(cache.v_dev.view()),
                KernelArg::Buffer(q_dev.view()),
                KernelArg::Buffer(tokens_dev.view()),
                KernelArg::Buffer(counts_dev.view()),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(cache.pages_dev.view()),
                KernelArg::U32(IDX_WIDTH),
                KernelArg::U32(HEADS),
                KernelArg::U32(KV_HEADS),
                KernelArg::U32(splits),
            ],
        },
    );
    if splits > 1 {
        combine(gpu, &out, &part, n_tok, splits);
    }
    drain(gpu);
    let group = (HEADS / KV_HEADS) as usize;
    let mut want = Vec::new();
    for (t, list) in lists.iter().enumerate() {
        for h in 0..HEADS as usize {
            let qs = scaled_query(&q, t, h, true);
            want.extend(cache.attend(&qs, h / group, list.iter().map(|&p| p as u32)));
        }
    }
    (gpu.read_f32(&out), want)
}

/// Runs a tensor-core attention over the dense causal range of each row - no
/// selection list, one slice, the destination written directly - and returns
/// the output with its oracle.
fn tc_dense(gpu: &Gpu, entry: TcEntry, positions: &[i32], seed: u32) -> (Vec<f32>, Vec<f64>) {
    let cache = Cache::new(gpu);
    let n_tok = positions.len() as u32;
    let q = queries(n_tok, seed);
    let out_len = (n_tok * HEADS * HD) as usize;
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; out_len]));
    let q_dev = gpu.upload(&f32_bytes(&q));
    let pos_dev = gpu.upload(&i32_bytes(positions));
    run(
        gpu,
        entry.kernel(),
        &LaunchArgs {
            grid: Extent3::new(1, KV_HEADS, n_tok),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: entry.smem(),
            args: &[
                KernelArg::Buffer(out.view()),
                KernelArg::U64(0),
                KernelArg::Buffer(cache.k_dev.view()),
                KernelArg::Buffer(cache.v_dev.view()),
                KernelArg::Buffer(q_dev.view()),
                KernelArg::U64(0),
                KernelArg::U64(0),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(cache.pages_dev.view()),
                KernelArg::U32(IDX_WIDTH),
                KernelArg::U32(HEADS),
                KernelArg::U32(KV_HEADS),
                KernelArg::U32(1),
            ],
        },
    );
    drain(gpu);
    (
        gpu.read_f32(&out),
        causal_oracle(&cache, &q, positions, true),
    )
}

/// The selection rows of the gathered cases: a long row over every slice and
/// a short one whose 37 keys leave each weight's rounding nowhere to average
/// out.
fn tc_lists() -> [Vec<i32>; 2] {
    [selection(3_000, 2_048), selection(700, 37)]
}

/// Sixteen rows (the dense entry's floor) at positions from the first key to
/// past a page boundary: the short rows are where a rounded weight shows.
fn tc_positions() -> Vec<i32> {
    [0, 1, 2, 3, 5, 8, 13, 21]
        .into_iter()
        .chain(250..258)
        .collect()
}

/// The exact tensor-core attention over a selection agrees with the `f64`
/// oracle's f32-weighted softmax within [`TC_EXACT_TOL`].
pub fn the_tensor_core_attention_matches_fp32_weights(gpu: &Gpu) {
    let (got, want) = tc_gathered(gpu, TcEntry::Exact, &tc_lists(), 43);
    agree(
        &got,
        &want,
        TC_EXACT_TOL,
        "exact tensor-core attention, gathered",
    );
}

/// The same over the dense causal range, the prefill path.
pub fn the_dense_tensor_core_attention_matches_fp32_weights(gpu: &Gpu) {
    let (got, want) = tc_dense(gpu, TcEntry::Exact, &tc_positions(), 47);
    agree(
        &got,
        &want,
        TC_EXACT_TOL,
        "exact tensor-core attention, dense",
    );
}

/// The budget the two cases above grade at is one the ported kernel fails,
/// over both paths: without this, a budget loose enough to pass the rounded
/// weights would make those cases say nothing about which kernel ran.
///
/// # Panics
///
/// When the ported kernel meets the exact kernel's budget.
pub fn the_ported_tensor_core_attention_misses_that_budget(gpu: &Gpu) {
    let (got, want) = tc_gathered(gpu, TcEntry::Ported, &tc_lists(), 43);
    let (err, at) = worst(&got, &want);
    assert!(
        err > TC_EXACT_TOL,
        "ported gathered: worst error {err:e} at {at} is inside {TC_EXACT_TOL:e}"
    );
    let (got, want) = tc_dense(gpu, TcEntry::Ported, &tc_positions(), 47);
    let (err, at) = worst(&got, &want);
    assert!(
        err > TC_EXACT_TOL,
        "ported dense: worst error {err:e} at {at} is inside {TC_EXACT_TOL:e}"
    );
}
