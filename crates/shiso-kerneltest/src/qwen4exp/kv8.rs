//! The FP8 key/value cache kernels (`q4e_kv8`): the codec against a host
//! restatement, and every reader against its f16 twin.
//!
//! The module's whole claim is that the store is the only place it rounds:
//! every reader decodes a byte to exactly the f16 value it stands for and then
//! runs the f16 kernel's arithmetic. So the cases split the same way.
//!
//! - The encode law - round to nearest even on the E4M3 grid, saturate at
//!   448, never write the not-a-number code - is graded against a host
//!   reference that finds the nearest code by brute force, over values chosen
//!   to sit on every tie, every saturation edge, the subnormal floor, signed
//!   zero and the non-finite inputs. The values reach the device through the
//!   store's value plane, which stores its input unprocessed.
//! - The decode is graded for all 256 codes, through the dense reader at a
//!   single position, whose output is then exactly the stored value row.
//! - Every reader is then run twice over the same keys: once over an FP8
//!   cache, once - the f16 kernel - over an f16 cache holding the values those
//!   bytes decode to. The two outputs must be identical to the bit. That is a
//!   stronger statement than any tolerance against an oracle, and it is what
//!   keeps the f16 path's equivalences (dense vs gathered, verify vs single
//!   step) true of the FP8 path without restating them.
//!
//! The fixture geometry is the attention block's real one, over the same
//! scattered page table the f16 cases use, at a few MB of device memory.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "kernel arguments are u32 while the fixtures index in usize and store \
              positions and page ids as the kernel's i32; every value here is a small \
              non-negative literal"
)]
#![expect(
    clippy::many_single_char_names,
    clippy::similar_names,
    reason = "the oracles restate kernel bodies whose names are single letters (q, k, v, \
              m, l) and paired key/value buffers; renaming them would hide the \
              correspondence being graded"
)]
#![expect(
    clippy::float_cmp,
    reason = "exactness is what these cases assert: a decode either lands on the code's \
              value or it is wrong, and a tie is a tie only at equality"
)]

use half::f16;
use shiso_core::page::PAGE_TOKENS;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The FP8 module, mirroring `shiso_engine::sparse::kernels::MOD_KV8`.
const MODULE_KV8: &str = "q4e_kv8";
/// The f16 dense module, `kernels::MOD_QSA`.
const MODULE: &str = "q4e_qsa";
/// The f16 selected-block module, `kernels::MOD_QSA_SPARSE`.
const MODULE_SPARSE: &str = "q4e_qsa_sparse";
/// The f16 exact tensor-core module, `kernels::MOD_EXACT`.
const MODULE_EXACT: &str = "q4e_exact";

/// Query heads.
const HEADS: u32 = 24;
/// Key/value heads.
const KV_HEADS: u32 = 2;
/// Head width.
const HD: u32 = 256;
/// Rotated dims per head.
const N_ROT: u32 = 64;
/// mRoPE sections, in rotary pairs.
const SECTIONS: [u32; 3] = [11, 11, 10];
/// Rotary base.
const THETA: f32 = 1.0e7;
/// `RMSNorm` epsilon.
const EPS: f32 = 1.0e-6;
/// Threads in the store, `kernels::BLOCK_NARROW`.
const BLOCK_NARROW: u32 = 128;
/// Threads in the sparse and tensor-core readers, `kernels::BLOCK_LINEAR`.
const BLOCK_LINEAR: u32 = 256;
/// Warps of the dense reader, `kernels::ATTN_WARPS`.
const ATTN_WARPS: u32 = 8;
/// Queries per tile of the tiled reader, `kernels::ATTN_QT`.
const ATTN_QT: u32 = 16;
/// Keys per tile, `kernels::ATTN_KT`.
const ATTN_KT: u32 = 32;
/// Splits of the split reader, `kernels::ATTN_SPLITS`.
const ATTN_SPLITS: u32 = 48;
/// Splits of the gathered reader, `kernels::ATTN_GSPLITS`.
const ATTN_GSPLITS: u32 = 8;
/// Floats per partial slab.
const PART_STRIDE: u32 = HD + 8;
/// Slots per selection row, `kernels::IDX_WIDTH`.
const IDX_WIDTH: u32 = 2_048 + 3;
/// `kernels::qsa_tiled_smem(HD)`.
const TILED_SMEM: u32 = 2 * HD * ATTN_KT * 2 + ATTN_QT * HD * 4 + 8 * ATTN_KT * 4;
/// `kernels::qsa_sparse_smem()`.
const SPARSE_SMEM: u32 = HD * ATTN_KT * 2 * 2 + 16 * HD * 2 + 8 * ATTN_KT * 4;
/// Keys per tile of the tensor-core reader, `kernels::TC_KT`.
const TC_KT: u32 = 64;
/// `kernels::TC_EXACT_SMEM`.
const TC_EXACT_SMEM: u32 =
    (16 + 2 * TC_KT) * HD * 2 + 16 * TC_KT * 4 + 16 * TC_KT * 2 + 16 * 4 + 16 * TC_KT * 2;

/// Pages in the fixture pool.
const POOL_PAGES: u32 = 20;
/// The session's page table: no neighbours adjacent, the second entry the
/// pool's last page.
const TABLE: [i32; 16] = [13, 19, 2, 7, 0, 11, 5, 17, 9, 3, 15, 1, 18, 6, 12, 4];
/// The largest finite E4M3 magnitude.
const E4M3_MAX: f32 = 448.0;

// ---------------------------------------------------------------------------
// the host codec
// ---------------------------------------------------------------------------

/// The value of an E4M3 code: one sign, four exponent bits biased by seven,
/// three mantissa bits, subnormals at exponent zero, and the all-ones pattern
/// not a number. Restated from the format's definition, not from the kernel.
#[must_use]
pub fn e4m3_value(code: u8) -> f32 {
    let e = i32::from((code >> 3) & 0xF);
    let m = f32::from(code & 7);
    let magnitude = if e == 0 {
        m * 2f32.powi(-9)
    } else if e == 15 && code & 7 == 7 {
        f32::NAN
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if code & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// What a reader decodes a code to. The format's value for every finite code;
/// for the not-a-number code, which the store never writes, the grid step past
/// 448 that the kernels' shift-and-scale decode lands on.
#[must_use]
pub fn e4m3_decoded(code: u8) -> f32 {
    if code & 0x7F == 0x7F {
        if code & 0x80 == 0 { 480.0 } else { -480.0 }
    } else {
        e4m3_value(code)
    }
}

/// The code the store must write for `x`: the nearest finite code by brute
/// force, ties to the even code, anything at or past 448 (and NaN) saturated,
/// the sign carried even onto zero.
#[must_use]
pub fn e4m3_encode_reference(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = f64::from(x.abs());
    if x.is_nan() || a >= f64::from(E4M3_MAX) {
        return sign | 0x7E;
    }
    let mut best = 0u8;
    let mut best_err = f64::INFINITY;
    for code in 0..=0x7Eu8 {
        let err = (f64::from(e4m3_value(code)) - a).abs();
        if err < best_err || (err == best_err && code & 1 == 0) {
            best = code;
            best_err = err;
        }
    }
    sign | best
}

/// `f16` values as little-endian bytes.
fn f16_bytes(values: &[f16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `i32` values as little-endian bytes.
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Physical cache row of logical position `p` through `table`.
fn kv_row(table: &[i32], p: u32) -> usize {
    table[(p / PAGE_TOKENS) as usize] as usize * PAGE_TOKENS as usize + (p % PAGE_TOKENS) as usize
}

/// `mrope` rows for `positions`, height and width offset from time.
fn mrope_rows(positions: &[i32]) -> Vec<i32> {
    positions.iter().flat_map(|&p| [p, p - 7, p - 3]).collect()
}

// ---------------------------------------------------------------------------
// the store
// ---------------------------------------------------------------------------

/// Which store entry a case runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Store {
    /// One sequence, one page table.
    Scalar,
    /// One page table per row, every row handed the same one.
    Batch,
}

/// Runs an FP8 store of `k` and `v` rows at `positions` through `table`, into
/// planes of `POOL_PAGES` pages prefilled with `sentinel`, and returns the two
/// planes.
fn store_fp8(
    gpu: &Gpu,
    which: Store,
    positions: &[i32],
    table: &[i32],
    pool_slots: u32,
    rows: (&[f32], &[f32], &[f32]),
) -> (Vec<u8>, Vec<u8>) {
    let (k, v, kw) = rows;
    let n_tok = positions.len() as u32;
    let plane = (POOL_PAGES * PAGE_TOKENS * KV_HEADS * HD) as usize;
    let sentinel = 0x5Au8;
    let k_cache = gpu.upload(&vec![sentinel; plane]);
    let v_cache = gpu.upload(&vec![sentinel; plane]);
    let k_dev = gpu.upload(&f32_bytes(k));
    let v_dev = gpu.upload(&f32_bytes(v));
    let kw_dev = gpu.upload(&f32_bytes(kw));
    let pos_dev = gpu.upload(&i32_bytes(positions));
    let mrope_dev = gpu.upload(&i32_bytes(&mrope_rows(positions)));
    let pages_dev = gpu.upload(&i32_bytes(table));
    let pointers = pointer_array(gpu, &vec![&pages_dev; positions.len()]);
    let (entry, pages_arg) = match which {
        Store::Scalar => ("q4e_qsa_store_kv_fp8_kernel", pages_dev.view()),
        Store::Batch => ("q4e_qsa_store_kv_batch_fp8_kernel", pointers.view()),
    };
    run(
        gpu,
        KernelRef::new(MODULE_KV8, entry),
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
                KernelArg::Buffer(pages_arg),
                KernelArg::U32(pool_slots),
                KernelArg::F32(EPS),
            ],
        },
    );
    drain(gpu);
    (gpu.read(&k_cache), gpu.read(&v_cache))
}

/// The values the encode cases put through the store: every code's own value,
/// every midpoint between neighbouring codes (the ties) and one f32 step to
/// either side of it, the saturation edge and past it, the floor below the
/// smallest subnormal, signed zeros, f32 subnormals and the non-finite inputs -
/// each with both signs.
fn encode_probes() -> Vec<f32> {
    let mut out = Vec::new();
    let finite: Vec<f32> = (0..=0x7Eu8).map(e4m3_value).collect();
    for pair in finite.windows(2) {
        let (lo, hi) = (pair[0], pair[1]);
        let mid = (f64::from(lo) + f64::from(hi)) / 2.0;
        let mid = mid as f32;
        assert_eq!(
            f64::from(mid),
            (f64::from(lo) + f64::from(hi)) / 2.0,
            "a midpoint is exact"
        );
        out.extend([lo, mid, mid.next_down(), mid.next_up()]);
    }
    out.extend([
        E4M3_MAX,
        E4M3_MAX.next_up(),
        464.0,
        479.9,
        480.0,
        1.0e6,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
        2f32.powi(-10),
        2f32.powi(-10).next_up(),
        2f32.powi(-10).next_down(),
        2f32.powi(-11),
        2f32.powi(-6).next_down(),
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.0,
    ]);
    let negated: Vec<f32> = out.iter().map(|x| -x).collect();
    out.extend(negated);
    out
}

/// Asserts the store's key plane holds the encode law's code of each rotated
/// key - wherever the host's f64 restatement of the norm and rotary is not
/// within reach of a rounding boundary, where the two can disagree by the f32
/// arithmetic's last bit - and that the bytes outside the written rows are
/// untouched.
fn assert_keys(
    got_k: &[u8],
    k: &[f32],
    kw: &[f32],
    positions: &[i32],
    table: &[i32],
    pool_slots: u32,
) {
    let hd = HD as usize;
    let mut graded = 0usize;
    for (t, &p) in positions.iter().enumerate() {
        let row = kv_row(table, p as u32);
        if row >= pool_slots as usize {
            continue;
        }
        let mrope = [p, p - 7, p - 3];
        for h in 0..KV_HEADS as usize {
            let src = (t * KV_HEADS as usize + h) * hd;
            let slot = (row * KV_HEADS as usize + h) * hd;
            let want = norm_rope(&k[src..src + hd], kw, mrope);
            for (i, &w) in want.iter().enumerate() {
                let code = got_k[slot + i];
                let near_tie = near_boundary(w);
                if !near_tie {
                    assert_eq!(
                        code,
                        e4m3_encode_reference(w as f32),
                        "store key: position {p} head {h} dim {i}: {w} stored as {code:#04x}"
                    );
                    graded += 1;
                }
                // Even at a boundary, the code is one of the two neighbours.
                let step = (f64::from(e4m3_decoded(code)) - w).abs();
                assert!(
                    step <= w.abs() / 8.0 + 2f64.powi(-9),
                    "store key: position {p} head {h} dim {i}: {w} decoded as {}",
                    e4m3_decoded(code)
                );
            }
        }
    }
    assert!(
        graded > 1_000,
        "only {graded} key elements were away from a boundary"
    );
}

/// Whether `w` is within a relative 1e-5 of the midpoint between two
/// neighbouring codes, where the device's f32 and the host's f64 can round to
/// different sides.
fn near_boundary(w: f64) -> bool {
    let a = w.abs();
    (0..0x7Eu8).any(|code| {
        let mid = (f64::from(e4m3_value(code)) + f64::from(e4m3_value(code + 1))) / 2.0;
        (a - mid).abs() <= 1e-5 * mid.max(2f64.powi(-10))
    })
}

/// `RMSNorm` with weight `w`, then the interleaved mRoPE over the first `N_ROT`
/// dims, in f64 with the angle formed in f32 as the kernel forms it.
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
        #[expect(
            clippy::cast_precision_loss,
            reason = "positions below 2^24 are exact in f32, as the kernel forms them"
        )]
        let p = mrope[axis] as u32 as f32;
        #[expect(
            clippy::cast_precision_loss,
            reason = "small rotary indices, exact in f32"
        )]
        let theta = p * THETA.powf(-((2 * i) as f32) / N_ROT as f32);
        let (s, c) = f64::from(theta).sin_cos();
        let (x0, x1) = (v[i], v[i + half]);
        v[i] = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
    v
}

/// The store writes the encode law's code for every value it is handed: the
/// probes go in through the value plane, which stores its input as is, and the
/// rotated keys beside them are graded wherever the host can say which side of
/// a boundary they fall on. Rows are written where the page table says, and a
/// row mapped past the pool is refused, leaving its bytes untouched.
///
/// # Panics
///
/// When a stored byte disagrees with the encode law or a refused row moved.
pub fn the_store_writes_the_e4m3_code_of_each_value(gpu: &Gpu) {
    store_case(gpu, Store::Scalar);
}

/// The batched store writes, row for row, what the scalar store writes.
///
/// # Panics
///
/// As [`the_store_writes_the_e4m3_code_of_each_value`].
pub fn the_batched_store_writes_the_e4m3_code_of_each_value(gpu: &Gpu) {
    store_case(gpu, Store::Batch);
}

/// The store case at either entry.
fn store_case(gpu: &Gpu, which: Store) {
    let probes = encode_probes();
    let row_elems = (KV_HEADS * HD) as usize;
    // Every probe placed, then padded out with ordinary values to whole rows;
    // the last row is one the pool refuses, to show the refusal.
    let n_written = probes.len().div_ceil(row_elems) + 1;
    let mut positions: Vec<i32> = (0..n_written as i32).map(|i| 250 + 3 * i).collect();
    positions.push(512);
    let table = [3, 7, 19];
    let pool_slots = 19 * PAGE_TOKENS;
    let kv_len = positions.len() * row_elems;
    let mut v: Vec<f32> = signal(kv_len, 9).into_iter().map(|x| 3.0 * x).collect();
    v[..probes.len()].copy_from_slice(&probes);
    let k: Vec<f32> = signal(kv_len, 7);
    let kw: Vec<f32> = signal(HD as usize, 13)
        .iter()
        .map(|x| 1.0 + 0.25 * x)
        .collect();
    let (got_k, got_v) = store_fp8(gpu, which, &positions, &table, pool_slots, (&k, &v, &kw));
    let hd = HD as usize;
    for (t, &p) in positions.iter().enumerate() {
        let row = kv_row(&table, p as u32);
        for h in 0..KV_HEADS as usize {
            let src = (t * KV_HEADS as usize + h) * hd;
            let slot = (row * KV_HEADS as usize + h) * hd;
            if row >= pool_slots as usize {
                assert!(
                    got_k[slot..slot + hd]
                        .iter()
                        .chain(&got_v[slot..slot + hd])
                        .all(|&b| b == 0x5A),
                    "store: position {p} maps past the pool and must not be written"
                );
                continue;
            }
            for i in 0..hd {
                let x = v[src + i];
                assert_eq!(
                    got_v[slot + i],
                    e4m3_encode_reference(x),
                    "store value: position {p} head {h} dim {i}: {x:e} ({:#010x})",
                    x.to_bits()
                );
            }
        }
    }
    assert!(
        got_v.iter().all(|&b| b & 0x7F != 0x7F),
        "store: a not-a-number code was written"
    );
    assert_keys(&got_k, &k, &kw, &positions, &table, pool_slots);
}

// ---------------------------------------------------------------------------
// the readers
// ---------------------------------------------------------------------------

/// The same cache twice: as FP8 codes, and as the f16 values they decode to.
struct Caches {
    k8: DeviceBuffer,
    v8: DeviceBuffer,
    k16: DeviceBuffer,
    v16: DeviceBuffer,
    pages: DeviceBuffer,
}

impl Caches {
    /// Codes for an ordinary spread of keys and values, with one element in
    /// sixty-one a code drawn from the whole finite range so that subnormals
    /// and the large binades are read too. Keys draw only up to 8 there, so
    /// that no one key's score swamps every softmax.
    fn new(gpu: &Gpu) -> Self {
        let plane = (POOL_PAGES * PAGE_TOKENS * KV_HEADS * HD) as usize;
        let codes = |seed: u32, spread: f32, cap: u8| -> Vec<u8> {
            let wild = signal(plane, seed ^ 0x55);
            signal(plane, seed)
                .into_iter()
                .zip(wild)
                .enumerate()
                .map(|(i, (x, w))| {
                    if i % 61 == 0 {
                        let code = ((w + 1.0) * 63.5) as u8 % cap;
                        if w < 0.0 { code | 0x80 } else { code }
                    } else {
                        e4m3_encode_reference(spread * x)
                    }
                })
                .collect()
        };
        let k = codes(17, 3.0, 0x50);
        let v = codes(19, 2.0, 0x7F);
        let decode = |bytes: &[u8]| -> Vec<f16> {
            bytes
                .iter()
                .map(|&b| {
                    let value = e4m3_decoded(b);
                    let h = f16::from_f32(value);
                    assert_eq!(h.to_f32(), value, "every E4M3 value is an f16 value");
                    h
                })
                .collect()
        };
        Self {
            k16: gpu.upload(&f16_bytes(&decode(&k))),
            v16: gpu.upload(&f16_bytes(&decode(&v))),
            k8: gpu.upload(&k),
            v8: gpu.upload(&v),
            pages: gpu.upload(&i32_bytes(&TABLE)),
        }
    }

    /// The key and value planes of one element.
    const fn planes(&self, fp8: bool) -> (&DeviceBuffer, &DeviceBuffer) {
        if fp8 {
            (&self.k8, &self.v8)
        } else {
            (&self.k16, &self.v16)
        }
    }
}

/// Queries for `n_tok` rows, spread to four so the softmax is peaked enough
/// that a wrong key moves the answer.
fn queries(n_tok: u32, seed: u32) -> Vec<f32> {
    signal((n_tok * HEADS * HD) as usize, seed)
        .into_iter()
        .map(|v| 4.0 * v)
        .collect()
}

/// Writes the device addresses of `buffers` into a fresh pointer array.
fn pointer_array(gpu: &Gpu, buffers: &[&DeviceBuffer]) -> DeviceBuffer {
    let array = gpu.upload(&vec![0u8; buffers.len().max(1) * 8]);
    let views: Vec<_> = buffers.iter().map(|b| b.view()).collect();
    gpu.device()
        .as_pointers()
        .expect("this backend packs pointer arrays")
        .pack_pointers(array.view(), &views)
        .expect("the pointer array packs");
    array
}

/// Asserts two outputs are identical to the bit, naming the first element that
/// is not.
fn bit_equal(fp8: &[f32], f16: &[f32], what: &str) {
    assert_eq!(fp8.len(), f16.len(), "{what}: lengths");
    for (i, (a, b)) in fp8.iter().zip(f16).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{what}: element {i} (head-row {}, dim {}) is {a:e} over the FP8 cache and {b:e} \
             over the f16 one",
            i / HD as usize,
            i % HD as usize,
        );
    }
    assert!(
        fp8.iter().all(|x| x.is_finite()),
        "{what}: a non-finite output - the pass did not write every element"
    );
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

/// A dense-family reader: the plain, batched or tiled entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dense {
    /// One block per (head, row).
    Plain,
    /// The same, one page table per row.
    Batch,
    /// Query tiles of sixteen rows.
    Tiled,
}

impl Dense {
    fn kernel(self, fp8: bool) -> KernelRef {
        match (self, fp8) {
            (Self::Plain, false) => KernelRef::new(MODULE, "q4e_qsa_attention_kernel"),
            (Self::Plain, true) => KernelRef::new(MODULE_KV8, "q4e_qsa_attention_fp8_kernel"),
            (Self::Batch, false) => KernelRef::new(MODULE, "q4e_qsa_attention_batch_kernel"),
            (Self::Batch, true) => KernelRef::new(MODULE_KV8, "q4e_qsa_attention_batch_fp8_kernel"),
            (Self::Tiled, false) => KernelRef::new(MODULE, "q4e_qsa_attention_tiled_kernel"),
            (Self::Tiled, true) => KernelRef::new(MODULE_KV8, "q4e_qsa_attention_tiled_fp8_kernel"),
        }
    }
}

/// Runs a dense-family reader over `positions` on one element's cache. The
/// batched entry gets one page table per row, every one a different
/// permutation of the pool, so a row that read another row's table would read
/// other keys.
fn dense_run(
    gpu: &Gpu,
    caches: &Caches,
    which: Dense,
    fp8: bool,
    positions: &[i32],
    seed: u32,
) -> Vec<f32> {
    let n_tok = positions.len() as u32;
    let q = queries(n_tok, seed);
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; (n_tok * HEADS * HD) as usize]));
    let q_dev = gpu.upload(&f32_bytes(&q));
    let pos_dev = gpu.upload(&i32_bytes(positions));
    let tables: Vec<DeviceBuffer> = (0..positions.len())
        .map(|r| {
            let mut table = TABLE;
            table.rotate_left(r % TABLE.len());
            gpu.upload(&i32_bytes(&table))
        })
        .collect();
    let pointers = pointer_array(gpu, &tables.iter().collect::<Vec<_>>());
    let (k, v) = caches.planes(fp8);
    let (grid, block, smem, pages) = match which {
        Dense::Plain => (
            Extent3::new(HEADS, n_tok, 1),
            32 * ATTN_WARPS,
            ATTN_WARPS * HD * 4,
            caches.pages.view(),
        ),
        Dense::Batch => (
            Extent3::new(HEADS, n_tok, 1),
            32 * ATTN_WARPS,
            ATTN_WARPS * HD * 4,
            pointers.view(),
        ),
        Dense::Tiled => (
            Extent3::new(HEADS, n_tok.div_ceil(ATTN_QT), 1),
            32 * 8,
            TILED_SMEM,
            caches.pages.view(),
        ),
    };
    run(
        gpu,
        which.kernel(fp8),
        &LaunchArgs {
            grid,
            block: Extent3::new(block, 1, 1),
            smem_bytes: smem,
            args: &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(k.view()),
                KernelArg::Buffer(v.view()),
                KernelArg::Buffer(q_dev.view()),
                KernelArg::U32(HD),
                KernelArg::U32(HEADS),
                KernelArg::U32(KV_HEADS),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(pages),
                KernelArg::U32(n_tok),
            ],
        },
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// One dense-family reader, both elements, bit for bit.
fn dense_pair(gpu: &Gpu, which: Dense, positions: &[i32], seed: u32, what: &str) {
    let caches = Caches::new(gpu);
    let fp8 = dense_run(gpu, &caches, which, true, positions, seed);
    let f16 = dense_run(gpu, &caches, which, false, positions, seed);
    bit_equal(&fp8, &f16, what);
}

/// Every code decodes to its value: all 256 of them stored as one value row
/// at position 0, where the dense reader's softmax has one key and its output
/// is that row exactly. The not-a-number codes, which no store writes, read as
/// +-480.
///
/// # Panics
///
/// When a code decodes to anything else.
pub fn every_code_decodes_to_its_value(gpu: &Gpu) {
    let plane = (POOL_PAGES * PAGE_TOKENS * KV_HEADS * HD) as usize;
    let mut k = vec![0x38u8; plane];
    let mut v = vec![0u8; plane];
    let row = kv_row(&TABLE, 0);
    for h in 0..KV_HEADS as usize {
        let slot = (row * KV_HEADS as usize + h) * HD as usize;
        for code in 0..=255u8 {
            // Head 1 reads the codes in reverse, so a decode that depended on
            // the lane would show.
            let c = if h == 0 { code } else { 255 - code };
            v[slot + usize::from(code)] = c;
            k[slot + usize::from(code)] = 0x30;
        }
    }
    let k8 = gpu.upload(&k);
    let v8 = gpu.upload(&v);
    let pages = gpu.upload(&i32_bytes(&TABLE));
    let q = queries(1, 3);
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; (HEADS * HD) as usize]));
    let q_dev = gpu.upload(&f32_bytes(&q));
    let pos_dev = gpu.upload(&i32_bytes(&[0]));
    run(
        gpu,
        Dense::Plain.kernel(true),
        &LaunchArgs {
            grid: Extent3::new(HEADS, 1, 1),
            block: Extent3::new(32 * ATTN_WARPS, 1, 1),
            smem_bytes: ATTN_WARPS * HD * 4,
            args: &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(k8.view()),
                KernelArg::Buffer(v8.view()),
                KernelArg::Buffer(q_dev.view()),
                KernelArg::U32(HD),
                KernelArg::U32(HEADS),
                KernelArg::U32(KV_HEADS),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(pages.view()),
                KernelArg::U32(1),
            ],
        },
    );
    drain(gpu);
    let got = gpu.read_f32(&out);
    let group = (HEADS / KV_HEADS) as usize;
    for h in 0..HEADS as usize {
        for code in 0..=255u8 {
            let c = if h / group == 0 { code } else { 255 - code };
            let g = got[h * HD as usize + usize::from(code)];
            let want = e4m3_decoded(c);
            // The reader's accumulation `0 * corr + 1 * v` adds a +0 to the
            // value, which turns the negative-zero code's -0 into +0 - in the
            // f16 kernel exactly as here - so zero is graded by value and
            // every other code by its bits.
            assert!(
                g.to_bits() == want.to_bits() || (want == 0.0 && g == 0.0),
                "code {c:#04x} (head {h}) decoded to {g:e}, its value is {want:e}"
            );
        }
    }
}

/// The dense reader decodes one row at position 3000, across twelve pages of
/// the scattered table, exactly as the f16 reader reads the decoded cache.
pub fn the_dense_reader_matches_f16_at_decode(gpu: &Gpu) {
    dense_pair(gpu, Dense::Plain, &[3_000], 23, "dense decode");
}

/// The same over a prefill chunk that crosses a page boundary.
pub fn the_dense_reader_matches_f16_over_a_chunk(gpu: &Gpu) {
    let positions: Vec<i32> = (250..259).chain([3, 5]).collect();
    dense_pair(gpu, Dense::Plain, &positions, 29, "dense prefill");
}

/// The batched reader, each row its own sequence and page table.
pub fn the_batched_reader_matches_f16(gpu: &Gpu) {
    dense_pair(
        gpu,
        Dense::Batch,
        &[3_000, 700, 5, 2_049],
        31,
        "batched decode",
    );
}

/// The tiled reader over twenty rows: one full query tile and a partial one.
pub fn the_tiled_reader_matches_f16(gpu: &Gpu) {
    let positions: Vec<i32> = (240..260).collect();
    dense_pair(gpu, Dense::Tiled, &positions, 37, "tiled prefill");
}

/// Runs a split reader over `positions` at `splits` on one element's cache.
fn split_run(
    gpu: &Gpu,
    caches: &Caches,
    fp8: bool,
    positions: &[i32],
    splits: u32,
    seed: u32,
) -> Vec<f32> {
    let n_tok = positions.len() as u32;
    let q = queries(n_tok, seed);
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; (n_tok * HEADS * HD) as usize]));
    let part = gpu.upload(&f32_bytes(&vec![
        f32::NAN;
        (n_tok * HEADS * splits * PART_STRIDE)
            as usize
    ]));
    let q_dev = gpu.upload(&f32_bytes(&q));
    let pos_dev = gpu.upload(&i32_bytes(positions));
    let (k, v) = caches.planes(fp8);
    let kernel = if fp8 {
        KernelRef::new(MODULE_KV8, "q4e_qsa_attention_split_fp8_kernel")
    } else {
        KernelRef::new(MODULE_SPARSE, "q4e_qsa_attention_split_kernel")
    };
    run(
        gpu,
        kernel,
        &LaunchArgs {
            grid: Extent3::new(n_tok, splits, KV_HEADS),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: SPARSE_SMEM,
            args: &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(part.view()),
                KernelArg::Buffer(k.view()),
                KernelArg::Buffer(v.view()),
                KernelArg::Buffer(q_dev.view()),
                KernelArg::U32(HEADS),
                KernelArg::U32(KV_HEADS),
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(caches.pages.view()),
                KernelArg::U32(n_tok),
                KernelArg::U32(splits),
            ],
        },
    );
    if splits > 1 {
        combine(gpu, &out, &part, n_tok, splits);
    }
    drain(gpu);
    gpu.read_f32(&out)
}

/// The split reader at decode (48 splits, then the fold), at verify width near
/// the start of a session (empty splits), and at one split writing directly.
pub fn the_split_reader_matches_f16(gpu: &Gpu) {
    let caches = Caches::new(gpu);
    for (positions, splits, seed, what) in [
        (&[3_000][..], ATTN_SPLITS, 41, "split decode"),
        (&[0, 1, 2, 3][..], ATTN_SPLITS, 43, "split verify"),
        (&[510, 511, 512, 513][..], 1, 47, "split, one split"),
    ] {
        let fp8 = split_run(gpu, &caches, true, positions, splits, seed);
        let f16 = split_run(gpu, &caches, false, positions, splits, seed);
        bit_equal(&fp8, &f16, what);
    }
}

/// A selection row: positions in `0..=last` whose pooled block is picked by a
/// fixed pattern, plus the row's own last few, capped at `cap`.
fn selection(last: u32, cap: usize) -> Vec<i32> {
    let mut list: Vec<i32> = (0..=last)
        .filter(|p| (p / 4) % 11 == 3 || (p / 4) % 29 == 0 || (p / 4) % 2 == 1 || last - p < 5)
        .map(|p| p as i32)
        .collect();
    list.truncate(cap);
    list
}

/// Selection rows packed `IDX_WIDTH` apart, and their counts.
fn pack_lists(lists: &[Vec<i32>]) -> (Vec<i32>, Vec<i32>) {
    let mut tokens = vec![-1i32; lists.len() * IDX_WIDTH as usize];
    let mut counts = Vec::new();
    for (t, list) in lists.iter().enumerate() {
        let base = t * IDX_WIDTH as usize;
        tokens[base..base + list.len()].copy_from_slice(list);
        counts.push(list.len() as i32);
    }
    (tokens, counts)
}

/// Runs the gathered reader over `lists` on one element's cache.
fn gather_run(gpu: &Gpu, caches: &Caches, fp8: bool, lists: &[Vec<i32>], seed: u32) -> Vec<f32> {
    let n_tok = lists.len() as u32;
    let q = queries(n_tok, seed);
    let (tokens, counts) = pack_lists(lists);
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; (n_tok * HEADS * HD) as usize]));
    let part = gpu.upload(&f32_bytes(&vec![
        f32::NAN;
        (n_tok * HEADS * ATTN_GSPLITS * PART_STRIDE)
            as usize
    ]));
    let q_dev = gpu.upload(&f32_bytes(&q));
    let tokens_dev = gpu.upload(&i32_bytes(&tokens));
    let counts_dev = gpu.upload(&i32_bytes(&counts));
    let (k, v) = caches.planes(fp8);
    let kernel = if fp8 {
        KernelRef::new(MODULE_KV8, "q4e_qsa_attention_gather_fp8_kernel")
    } else {
        KernelRef::new(MODULE_SPARSE, "q4e_qsa_attention_gather_kernel")
    };
    run(
        gpu,
        kernel,
        &LaunchArgs {
            grid: Extent3::new(ATTN_GSPLITS, KV_HEADS, n_tok),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: SPARSE_SMEM,
            args: &[
                KernelArg::Buffer(part.view()),
                KernelArg::Buffer(k.view()),
                KernelArg::Buffer(v.view()),
                KernelArg::Buffer(q_dev.view()),
                KernelArg::Buffer(tokens_dev.view()),
                KernelArg::Buffer(counts_dev.view()),
                KernelArg::Buffer(caches.pages.view()),
                KernelArg::U32(IDX_WIDTH),
                KernelArg::U32(HEADS),
                KernelArg::U32(KV_HEADS),
                KernelArg::U32(ATTN_GSPLITS),
            ],
        },
    );
    combine(gpu, &out, &part, n_tok, ATTN_GSPLITS);
    drain(gpu);
    gpu.read_f32(&out)
}

/// The selection rows of the gathered cases: a full 2048-key selection out of
/// a 4000-token context - past the budget, which is the only time the
/// gathered readers run - and a short one.
fn lists() -> [Vec<i32>; 2] {
    let long = selection(4_000, 2_048);
    assert_eq!(long.len(), 2_048, "the long row is a full selection");
    [long, selection(700, 37)]
}

/// The gathered reader over a selection past the budget.
pub fn the_gathered_reader_matches_f16(gpu: &Gpu) {
    let caches = Caches::new(gpu);
    let fp8 = gather_run(gpu, &caches, true, &lists(), 53);
    let f16 = gather_run(gpu, &caches, false, &lists(), 53);
    bit_equal(&fp8, &f16, "gathered");
}

/// Runs the exact tensor-core reader on one element's cache: over `lists` when
/// given, sliced as the sparse entry slices them, else over the dense causal
/// range of each of `positions` at one slice.
fn tc_run(
    gpu: &Gpu,
    caches: &Caches,
    fp8: bool,
    lists: Option<&[Vec<i32>]>,
    positions: &[i32],
    seed: u32,
) -> Vec<f32> {
    let n_tok = positions.len() as u32;
    let splits = match lists {
        // `kernels::tc_splits_sparse`.
        Some(_) if n_tok * KV_HEADS < 64 => ATTN_GSPLITS,
        _ => 1,
    };
    let q = queries(n_tok, seed);
    let (tokens, counts) = pack_lists(lists.unwrap_or(&[]));
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; (n_tok * HEADS * HD) as usize]));
    let part = gpu.upload(&f32_bytes(&vec![
        f32::NAN;
        (n_tok * HEADS * splits * PART_STRIDE)
            as usize
    ]));
    let q_dev = gpu.upload(&f32_bytes(&q));
    let tokens_dev = gpu.upload(&i32_bytes(
        &tokens.clone().into_iter().chain([0]).collect::<Vec<_>>(),
    ));
    let counts_dev = gpu.upload(&i32_bytes(
        &counts.clone().into_iter().chain([0]).collect::<Vec<_>>(),
    ));
    let pos_dev = gpu.upload(&i32_bytes(positions));
    let (k, v) = caches.planes(fp8);
    let kernel = if fp8 {
        KernelRef::new(MODULE_KV8, "q4e_qsa_attention_tc_exact_fp8_kernel")
    } else {
        KernelRef::new(MODULE_EXACT, "q4e_qsa_attention_tc_exact_kernel")
    };
    let (out_arg, part_arg) = if splits == 1 {
        (KernelArg::Buffer(out.view()), KernelArg::U64(0))
    } else {
        (KernelArg::U64(0), KernelArg::Buffer(part.view()))
    };
    let (tokens_arg, counts_arg) = if lists.is_some() {
        (
            KernelArg::Buffer(tokens_dev.view()),
            KernelArg::Buffer(counts_dev.view()),
        )
    } else {
        (KernelArg::U64(0), KernelArg::U64(0))
    };
    run(
        gpu,
        kernel,
        &LaunchArgs {
            grid: Extent3::new(splits, KV_HEADS, n_tok),
            block: Extent3::new(BLOCK_LINEAR, 1, 1),
            smem_bytes: TC_EXACT_SMEM,
            args: &[
                out_arg,
                part_arg,
                KernelArg::Buffer(k.view()),
                KernelArg::Buffer(v.view()),
                KernelArg::Buffer(q_dev.view()),
                tokens_arg,
                counts_arg,
                KernelArg::Buffer(pos_dev.view()),
                KernelArg::Buffer(caches.pages.view()),
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
    gpu.read_f32(&out)
}

/// The exact tensor-core reader over a selection past the budget (eight
/// slices, folded), over thirty-two selections at once (one slice, written
/// directly - the prefill shape of the sparse entry), and over the dense
/// causal range of sixteen rows (the dense entry).
pub fn the_tensor_core_reader_matches_f16(gpu: &Gpu) {
    let caches = Caches::new(gpu);
    let few = lists();
    let few_pos = vec![0i32; few.len()];
    let many: Vec<Vec<i32>> = (0..32).map(|t| selection(3_700 + 11 * t, 2_048)).collect();
    let many_pos = vec![0i32; many.len()];
    let dense_pos: Vec<i32> = [0, 1, 2, 3, 5, 8, 13, 21]
        .into_iter()
        .chain(250..258)
        .collect();
    for (lists, positions, seed, what) in [
        (
            Some(&few[..]),
            &few_pos[..],
            59,
            "tensor-core gathered, eight slices",
        ),
        (
            Some(&many[..]),
            &many_pos[..],
            61,
            "tensor-core gathered, one slice",
        ),
        (None, &dense_pos[..], 67, "tensor-core dense"),
    ] {
        let fp8 = tc_run(gpu, &caches, true, lists, positions, seed);
        let f16 = tc_run(gpu, &caches, false, lists, positions, seed);
        bit_equal(&fp8, &f16, what);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_encoder_inverts_every_finite_code() {
        for code in (0..=0xFFu8).filter(|c| c & 0x7F != 0x7F) {
            assert_eq!(e4m3_encode_reference(e4m3_value(code)), code, "{code:#04x}");
        }
    }

    #[test]
    fn the_reference_encoder_ties_to_even_and_saturates() {
        // 1.0 is 0x38 and the next code up is 1.125 (0x39): the midpoint goes
        // to the even code below it, and the midpoint above 0x39 to 0x3A.
        assert_eq!(e4m3_encode_reference(1.0625), 0x38);
        assert_eq!(e4m3_encode_reference(1.1875), 0x3A);
        assert_eq!(e4m3_encode_reference(464.0), 0x7E);
        assert_eq!(e4m3_encode_reference(f32::INFINITY), 0x7E);
        assert_eq!(e4m3_encode_reference(f32::NEG_INFINITY), 0xFE);
        assert_eq!(e4m3_encode_reference(2f32.powi(-10)), 0x00);
        assert_eq!(e4m3_encode_reference(2f32.powi(-10).next_up()), 0x01);
        assert_eq!(e4m3_encode_reference(-0.0), 0x80);
    }
}
