//! Device cases for the dense attention family: the two norm-and-rotate kernels,
//! their two rotate-only twins and the two softmax entry points.
//!
//! These are the parametric attention kernels Gemma 4 needs and qwen35 does not:
//! a rotation whose frequency ladder is computed over a width of its own rather
//! than over the rotated slice, a value projection normalized before it is
//! cached, and a softmax bounded from below by a sliding window, from above by a
//! per-token bound, and scaled by a per-layer number.
//!
//! Every case compares the device against the simplest possible reading of the
//! formula, written as a plain Rust `f32` loop, so a disagreement points at the
//! kernel or at its launch geometry rather than at a second, equally subtle
//! implementation.
//!
//! The `#[test]` wrappers live in each backend's own driver -
//! `crates/shiso-metal/tests/dense_attn.rs`,
//! `crates/shiso-cuda/tests/dense_attn.rs`. [`ENTRIES`] is here rather than
//! there because the twin list is a property of the kernel set, not of a
//! device.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "kernel argument structs take i32/u32 dimensions while the fixtures are written in usize, and every fixture dimension and position here is a small non-negative literal"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use half::f16;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::oracle::{TOL_F32, TOL_QUANT};

/// Threads per threadgroup of the two norm-and-rotate kernels, mirroring
/// `shiso_engine::launch::BLOCK_128`.
const BLOCK_128: u32 = 128;
/// Threads per threadgroup of the two softmax entry points, mirroring
/// `shiso_engine::launch::BLOCK_LINEAR`: eight simdgroups.
pub(super) const BLOCK_256: u32 = 256;
/// Threadgroup bytes the norm-and-rotate kernels ask for, mirroring
/// `shiso_engine::launch::DENSE_NORM_ROPE_SMEM_BYTES`: the block reduction's
/// slots plus the widest head this set carries.
const NORM_ROPE_SMEM: u32 = 4 * (33 + 512);
/// Threadgroup bytes the decode softmax asks for, mirroring
/// `shiso_engine::launch::DENSE_ATTN_DECODE_SMEM_BYTES`: one accumulator row per
/// simdgroup plus the two partial arrays.
const DECODE_SMEM: u32 = 8 * 512 * 4 + 2 * 8 * 4;
/// Threadgroup bytes the prefill softmax asks for, mirroring
/// `shiso_engine::launch::DENSE_ATTN_PREFILL_SMEM_BYTES`: the transposed key
/// tile at the widest head this set carries.
const PREFILL_SMEM: u32 = 512 * 16 * 2;
/// Queries per prefill threadgroup, mirroring
/// `shiso_engine::launch::DENSE_ATTN_PREFILL_TILE`.
const QT: usize = 16;
/// The decode entry point's name, which is also how [`check_softmax`] tells the
/// two launches apart.
const DECODE: &str = "kernel_dense_attn_decode";
/// The prefill entry point's name.
const PREFILL: &str = "kernel_dense_attn_prefill";

/// Every entry point this module's cases launch.
pub const ENTRIES: [&str; 6] = [
    "kernel_dense_q_norm_rope",
    "kernel_dense_kv_norm_rope_store",
    "kernel_dense_q_rope",
    "kernel_dense_kv_rope_store",
    DECODE,
    PREFILL,
];

/// Epsilon inside every per-head mean square below.
const EPS: f32 = 1e-6;

// -- argument blocks --------------------------------------------------------

/// `dense_args_q_norm_rope`, mirroring `shiso_engine::launch::DenseQNormRopeArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct QNormRopeArgs {
    head_dim: u32,
    n_head: u32,
    n_rot: u32,
    freq_dims: u32,
    n_tok: u32,
    rope_base: f32,
    eps: f32,
}

const _: () = assert!(size_of::<QNormRopeArgs>() == 28);

/// `dense_args_kv_norm_rope`, mirroring
/// `shiso_engine::launch::DenseKvNormRopeArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct KvNormRopeArgs {
    head_dim: u32,
    n_head_kv: u32,
    n_rot: u32,
    freq_dims: u32,
    n_tok: u32,
    rope_base: f32,
    eps: f32,
    page_tokens: u32,
    v_norm: u32,
    /// Entries between one row's page table and the next; `0` when every row
    /// of the launch reads the table the pointer names, which is what these
    /// single-session fixtures want.
    page_stride: u32,
}

const _: () = assert!(size_of::<KvNormRopeArgs>() == 40);

/// `dense_args_attn`, mirroring `shiso_engine::launch::DenseAttnArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct AttnArgs {
    pub(super) head_dim: u32,
    pub(super) n_head: u32,
    pub(super) n_head_kv: u32,
    pub(super) n_tok: u32,
    pub(super) page_tokens: u32,
    pub(super) window: u32,
    pub(super) scale: f32,
    /// Entries between one row's page table and the next; `0` when every row
    /// of the launch reads the table the pointer names, which is what these
    /// single-session fixtures want.
    pub(super) page_stride: u32,
}

const _: () = assert!(size_of::<AttnArgs>() == 32);

// -- helpers ----------------------------------------------------------------

/// Launches one dense entry point, naming the geometry if the driver refuses it.
pub(super) fn launch(
    gpu: &Gpu,
    entry: &'static str,
    grid: Extent3,
    block: Extent3,
    smem_bytes: u32,
    args: &[KernelArg<'_>],
) {
    run(
        gpu,
        KernelRef::new("dense", entry),
        &LaunchArgs {
            grid,
            block,
            smem_bytes,
            args,
        },
    );
}

/// An f32 device buffer of `len` zeros.
fn zeros(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![0.0f32; len]))
}

/// An f16 device buffer of `len` zeros.
fn zeros_f16(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&vec![0u8; len * 2])
}

/// Asserts every value agrees within `tol`, scaled by the expected magnitude.
pub(super) fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (index, (&have, &expect)) in got.iter().zip(want).enumerate() {
        let bound = tol * expect.abs().max(1.0);
        assert!(
            (have - expect).abs() <= bound,
            "{what}: element {index} is {have}, expected {expect} (tolerance {bound})",
        );
    }
}

/// Little-endian bytes of an `int32` array, which is how positions, bounds and
/// page tables reach a kernel.
pub(super) fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().copied().flat_map(i32::to_le_bytes).collect()
}

/// Little-endian bytes of a whole fixture as `f16`, through the same conversion
/// the loader uses.
pub(super) fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}

/// `f16` values from little-endian bytes, read back through the oracle's own
/// conversion so nothing in the fixture path re-implements one.
fn f16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| crate::oracle::f16_to_f32(u16::from_le_bytes(*pair)))
        .collect()
}

/// Fixture values exact in `f16`: multiples of an eighth in `[-1, 1.75]`, so a
/// cache round trip is exact and any difference is arithmetic. The period is
/// coprime with the head, position and dimension strides, so no two heads and no
/// two positions read the same row.
pub(super) fn fixture(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7 + salt * 31) % 23) as f32 * 0.125 - 1.0)
        .collect()
}

// -- the rotation -----------------------------------------------------------

/// One layer's rotary shape: the head it covers, the slice it rotates, the width
/// its frequency ladder is derived over, and the base of that ladder.
///
/// The four travel together because three of them are what the rotation reads
/// and no two of them are derivable from the others.
#[derive(Debug, Clone, Copy)]
struct RopeShape {
    head_dim: usize,
    n_rot: usize,
    freq_dims: usize,
    base: f32,
}

/// Gemma 4's global layers: a 512-wide head whose frequency ladder is derived
/// over all 512 dimensions but only whose leading 128 are rotated, which pairs
/// `i` with `i + 256` for `i < 64`.
const FULL: RopeShape = RopeShape {
    head_dim: 512,
    n_rot: 128,
    freq_dims: 512,
    base: 1e6,
};

/// The other shape a dense layer takes: the whole head is the ladder and the
/// whole head is rotated, which pairs `i` with `i + 128`.
const SLICE: RopeShape = RopeShape {
    head_dim: 256,
    n_rot: 256,
    freq_dims: 256,
    base: 1e4,
};

/// Query heads of the two rotary cases.
const ROPE_HEADS: usize = 2;

/// The naive reading of a weighted per-head RMS norm followed by the rotation:
/// pair `(i, i + freq_dims / 2)` for `i < n_rot / 2`, at the angle
/// `pos * base^(-2i / freq_dims)`.
///
/// The frequency width is a separate argument from the rotated width on purpose.
/// A head whose ladder is derived over the whole head but only whose leading
/// slice is rotated pairs dimension `i` with one a quarter of a head away, not
/// with `i + n_rot / 2`, and a kernel that conflated the two would be plausible
/// and wrong on every global layer.
fn norm_rope(row: &[f32], weight: &[f32], shape: RopeShape, pos: f32) -> Vec<f32> {
    rope(norm_only(row, weight, shape.head_dim), shape, pos)
}

/// The rotation alone, over a row as it is.
fn rope(mut out: Vec<f32>, shape: RopeShape, pos: f32) -> Vec<f32> {
    let half = shape.freq_dims / 2;
    for i in 0..shape.n_rot / 2 {
        let theta = pos * shape.base.powf(-2.0 * i as f32 / shape.freq_dims as f32);
        let (sin_t, cos_t) = (theta.sin(), theta.cos());
        let lo = out[i];
        let hi = out[i + half];
        out[i] = lo * cos_t - hi * sin_t;
        out[i + half] = lo * sin_t + hi * cos_t;
    }
    out
}

/// The plain weighted norm, which is what the untouched tail of a partially
/// rotated head is.
fn norm_only(row: &[f32], weight: &[f32], head_dim: usize) -> Vec<f32> {
    let mean: f32 = row.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
    let inv = 1.0 / (mean + EPS).sqrt();
    row.iter()
        .zip(weight)
        .map(|(&v, &scale)| v * inv * scale)
        .collect()
}

/// Runs the query kernel over [`ROPE_HEADS`] heads of `pos.len()` tokens and
/// reads the rotated rows back.
fn q_norm_rope(gpu: &Gpu, row: &[f32], weight: &[f32], shape: RopeShape, pos: &[i32]) -> Vec<f32> {
    let n_tok = pos.len();
    let d_row = gpu.upload(&f32_bytes(row));
    let d_weight = gpu.upload(&f32_bytes(weight));
    let d_pos = gpu.upload(&i32_bytes(pos));
    let out = zeros(gpu, n_tok * ROPE_HEADS * shape.head_dim);
    let args = QNormRopeArgs {
        head_dim: shape.head_dim as u32,
        n_head: ROPE_HEADS as u32,
        n_rot: shape.n_rot as u32,
        freq_dims: shape.freq_dims as u32,
        n_tok: n_tok as u32,
        rope_base: shape.base,
        eps: EPS,
    };
    launch(
        gpu,
        "kernel_dense_q_norm_rope",
        Extent3::new(ROPE_HEADS as u32, n_tok as u32, 1),
        Extent3::new(BLOCK_128, 1, 1),
        NORM_ROPE_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_row.view()),
            KernelArg::Buffer(d_weight.view()),
            KernelArg::Buffer(d_pos.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// The rotation at `freq_dims == head_dim`, where the pair stride and the
/// rotated width agree and every reading of the arguments produces the same
/// answer. The baseline the next case is told apart from.
pub fn rope_pairs_over_the_full_head(gpu: &Gpu) {
    let pos = [7i32, 3];
    let rows = crate::oracle::signal(pos.len() * ROPE_HEADS * FULL.head_dim, 61);
    // Ones, so this case is the rotation alone: the weighted half of the norm is
    // what the key case below separates from the unweighted value norm.
    let weight = vec![1.0f32; FULL.head_dim];
    let got = q_norm_rope(gpu, &rows, &weight, FULL, &pos);

    for (tok, &at) in pos.iter().enumerate() {
        for head in 0..ROPE_HEADS {
            let base = (tok * ROPE_HEADS + head) * FULL.head_dim;
            let row = &rows[base..base + FULL.head_dim];
            let want = norm_rope(row, &weight, FULL, at as f32);
            close(
                &got[base..base + FULL.head_dim],
                &want,
                TOL_F32,
                &format!("token {tok} head {head}"),
            );

            // The dimensions between the rotated slice and its partner, and
            // everything past the partner, carry the norm and nothing else. A
            // kernel that paired `i` with `i + n_rot / 2` would disturb 64..128
            // and leave 256..320 alone, so this is the assertion that separates
            // the two pairings.
            let plain = norm_only(row, &weight, FULL.head_dim);
            let untouched = (FULL.n_rot / 2..FULL.freq_dims / 2)
                .chain(FULL.freq_dims / 2 + FULL.n_rot / 2..FULL.head_dim);
            for i in untouched {
                crate::assert_close!(
                    got[base + i],
                    plain[i],
                    TOL_F32,
                    "token {tok} head {head} dimension {i} is outside every rotated pair"
                );
            }
            // And the rotated dimensions are not: at position 7 with this base
            // the leading angle is 7 radians, which moves the pair by far more
            // than rounding.
            let moved = (0..FULL.n_rot / 2)
                .filter(|&i| (got[base + i] - plain[i]).abs() > 1e-3)
                .count();
            if at != 0 {
                assert!(
                    moved > FULL.n_rot / 8,
                    "token {tok} head {head}: the fixture does not move the rotated slice",
                );
            }
        }
    }
}

/// The rotation at `freq_dims != n_rot`, which is the case that tells a right
/// kernel from a fluent wrong one.
///
/// With `half = freq_dims / 2`, every `i` below `n_rot / 2` rotates the pair
/// `(v[i], v[i + half])`. At `head_dim` 512, `n_rot` 128, `freq_dims` 512 the
/// pairs are `(i, i + 256)`; a kernel that reads the pair stride off `n_rot`
/// pairs `(i, i + 64)` instead - every head still normalized, still rotated,
/// and wrong.
pub fn rope_over_the_rotated_slice(gpu: &Gpu) {
    let pos = [11i32, 0];
    let rows = crate::oracle::signal(pos.len() * ROPE_HEADS * SLICE.head_dim, 62);
    let weight = vec![1.0f32; SLICE.head_dim];
    let got = q_norm_rope(gpu, &rows, &weight, SLICE, &pos);

    for (tok, &at) in pos.iter().enumerate() {
        for head in 0..ROPE_HEADS {
            let base = (tok * ROPE_HEADS + head) * SLICE.head_dim;
            let want = norm_rope(
                &rows[base..base + SLICE.head_dim],
                &weight,
                SLICE,
                at as f32,
            );
            close(
                &got[base..base + SLICE.head_dim],
                &want,
                TOL_F32,
                &format!("token {tok} head {head}"),
            );
        }
    }
}

// -- the key and value store ------------------------------------------------

/// Head width of the cache cases.
const KV_HD: usize = 256;
/// Key/value heads of the cache cases.
const KV_HEADS: usize = 2;
/// Query heads of the softmax cases: four queries per key head.
const Q_HEADS: usize = 8;
/// Logical cache positions the cases write and read.
const N_POS: usize = 40;
/// The rotary shape the cache cases use.
const KV_SHAPE: RopeShape = RopeShape {
    head_dim: KV_HD,
    n_rot: KV_HD,
    freq_dims: KV_HD,
    base: 1e4,
};
/// Tokens the store case writes.
const KV_TOK: usize = 3;

/// One session's page table: how a logical position becomes a physical row.
///
/// Carried as a value because it is a property of the CASE rather than of the
/// kernel - the same launch reads a one-page prefix and a scattered one - and
/// because the oracle has to perform the same translation from the same two
/// numbers rather than from a second copy of the rule.
#[derive(Debug, Clone, Copy)]
pub(super) struct Paging {
    pub(super) page_tokens: u32,
    pub(super) pages: &'static [i32],
}

impl Paging {
    /// Physical cache rows this table addresses, which is what the fixture and
    /// the cache allocation are sized by.
    pub(super) const fn rows(self) -> usize {
        self.pages.len() * self.page_tokens as usize
    }

    /// The translation `dense_kv_row` performs:
    /// `pages[p / page_tokens] * page_tokens + p % page_tokens`.
    pub(super) fn row(self, position: u32) -> usize {
        let page = self.pages[(position / self.page_tokens) as usize] as usize;
        page * self.page_tokens as usize + (position % self.page_tokens) as usize
    }
}

/// One page wide enough for the whole fixture, so the translation is the
/// identity and a case using it isolates everything that is not paging.
const ONE_PAGE: Paging = Paging {
    page_tokens: 64,
    pages: &[0],
};

/// Three short pages handed out in an order no walk over positions would
/// produce: logical 0..15 live at rows 32..47, 16..31 at 0..15 and 32..39 at
/// 16..23.
///
/// Two consecutive positions are two different pages' rows across each
/// boundary, which is what makes a tile that walks a page edge -- and a decode
/// span that crosses one -- an assertion about `dense_kv_row` rather than about
/// contiguous memory.
const SHUFFLED_PAGES: Paging = Paging {
    page_tokens: 16,
    pages: &[2, 0, 1],
};

const _: () = assert!(ONE_PAGE.rows() >= N_POS);
const _: () = assert!(SHUFFLED_PAGES.rows() >= N_POS);

/// Uploads one page table.
pub(super) fn page_table(gpu: &Gpu, paging: Paging) -> DeviceBuffer {
    gpu.upload(&i32_bytes(paging.pages))
}

/// The value half of the store is normalized without a weight, which is the one
/// thing about it that is not the key half.
pub fn the_value_norm_is_unweighted(gpu: &Gpu) {
    let keys = crate::oracle::signal(KV_TOK * KV_HEADS * KV_HD, 71);
    let values = crate::oracle::signal(KV_TOK * KV_HEADS * KV_HD, 72);
    // Deliberately not ones: if the value half reached for this weight the
    // comparison below would fail on every dimension it is not 1 at.
    let k_weight: Vec<f32> = crate::oracle::signal(KV_HD, 73)
        .iter()
        .map(|v| v + 1.5)
        .collect();
    // Starting at 1, so position 0 is a row nothing in this chunk names.
    let pos: Vec<i32> = (0..KV_TOK as i32).map(|t| t * 3 + 1).collect();

    let d_k = gpu.upload(&f32_bytes(&keys));
    let d_v = gpu.upload(&f32_bytes(&values));
    let d_kw = gpu.upload(&f32_bytes(&k_weight));
    let d_pos = gpu.upload(&i32_bytes(&pos));
    // Through the scattered table, so the rows below are rows the translation
    // chose rather than rows the token index happened to sit at.
    let pages = page_table(gpu, SHUFFLED_PAGES);
    let rows = SHUFFLED_PAGES.rows();
    let k_cache = zeros_f16(gpu, rows * KV_HEADS * KV_HD);
    let v_cache = zeros_f16(gpu, rows * KV_HEADS * KV_HD);

    let args = KvNormRopeArgs {
        head_dim: KV_HD as u32,
        n_head_kv: KV_HEADS as u32,
        n_rot: KV_SHAPE.n_rot as u32,
        freq_dims: KV_SHAPE.freq_dims as u32,
        n_tok: KV_TOK as u32,
        rope_base: KV_SHAPE.base,
        eps: EPS,
        page_tokens: SHUFFLED_PAGES.page_tokens,
        v_norm: 1,
        page_stride: 0,
    };
    launch(
        gpu,
        "kernel_dense_kv_norm_rope_store",
        Extent3::new(KV_HEADS as u32, KV_TOK as u32, 1),
        Extent3::new(BLOCK_128, 1, 1),
        NORM_ROPE_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(k_cache.view()),
            KernelArg::Buffer(v_cache.view()),
            KernelArg::Buffer(d_k.view()),
            KernelArg::Buffer(d_v.view()),
            KernelArg::Buffer(d_kw.view()),
            KernelArg::Buffer(d_pos.view()),
            KernelArg::Buffer(pages.view()),
        ],
    );
    drain(gpu);

    let got_k = f16_values(&gpu.read(&k_cache));
    let got_v = f16_values(&gpu.read(&v_cache));
    let ones = vec![1.0f32; KV_HD];
    for (tok, &at) in pos.iter().enumerate() {
        for head in 0..KV_HEADS {
            let src = (tok * KV_HEADS + head) * KV_HD;
            let slot = (SHUFFLED_PAGES.row(at as u32) * KV_HEADS + head) * KV_HD;
            // The key half is the weighted norm and the rotation, in the cache
            // row the position names.
            let want_k = norm_rope(&keys[src..src + KV_HD], &k_weight, KV_SHAPE, at as f32);
            close(
                &got_k[slot..slot + KV_HD],
                &want_k,
                TOL_QUANT,
                &format!("key token {tok} head {head}"),
            );
            // The value half is the same mean square with no weight and no
            // rotation.
            let want_v = norm_only(&values[src..src + KV_HD], &ones, KV_HD);
            close(
                &got_v[slot..slot + KV_HD],
                &want_v,
                TOL_QUANT,
                &format!("value token {tok} head {head}"),
            );
        }
    }

    // Row 0 belongs to logical position 16, which this chunk never names - and
    // it is also the row a kernel that skipped the translation would have
    // written for its first token. It is still zero.
    assert!(
        got_v[..KV_HD].iter().all(|v| *v == 0.0),
        "row 0 was written, so the store went by token index rather than through the page table",
    );
}

// -- rotation without a norm ----------------------------------------------

/// A Llama-style head: 64 wide, every dimension rotated over a ladder of its
/// own width, pairing `i` with `i + 32`.
const NARROW: RopeShape = RopeShape {
    head_dim: 64,
    n_rot: 64,
    freq_dims: 64,
    base: 1e5,
};

/// How far the rotate-only fixtures sit from unit root mean square.
///
/// The signal is uniform over `[-1, 1]`, a root mean square near 0.58, which a
/// normalizing kernel would scale by about 1.7; tripled, it would scale by about
/// 0.58 instead. Either way every dimension moves far past the tolerance, but
/// tripled is the side where "close to unit" cannot be argued.
const RAW_AMPLITUDE: f32 = 3.0;

/// Runs the rotate-only query kernel over [`ROPE_HEADS`] heads of `pos.len()`
/// tokens and reads the rows back.
fn q_rope(gpu: &Gpu, row: &[f32], shape: RopeShape, pos: &[i32]) -> Vec<f32> {
    let n_tok = pos.len();
    let d_row = gpu.upload(&f32_bytes(row));
    let d_pos = gpu.upload(&i32_bytes(pos));
    let out = zeros(gpu, n_tok * ROPE_HEADS * shape.head_dim);
    let args = QNormRopeArgs {
        head_dim: shape.head_dim as u32,
        n_head: ROPE_HEADS as u32,
        n_rot: shape.n_rot as u32,
        freq_dims: shape.freq_dims as u32,
        n_tok: n_tok as u32,
        rope_base: shape.base,
        // Unread: a rotate-only slot has no norm to put an epsilon in.
        eps: 0.0,
    };
    launch(
        gpu,
        "kernel_dense_q_rope",
        Extent3::new(ROPE_HEADS as u32, n_tok as u32, 1),
        Extent3::new(BLOCK_128, 1, 1),
        NORM_ROPE_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_row.view()),
            KernelArg::Buffer(d_pos.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// The rotate-only query kernel is the rotation of the raw head: no division
/// by its root mean square, at a Llama head's width and at the partially
/// rotated Gemma shape whose pair stride is not `n_rot / 2`.
pub fn rope_without_a_norm_rotates_the_raw_head(gpu: &Gpu) {
    for (salt, shape) in [(63, NARROW), (64, FULL)] {
        let pos = [9i32, 0, 21];
        let rows: Vec<f32> = crate::oracle::signal(pos.len() * ROPE_HEADS * shape.head_dim, salt)
            .iter()
            .map(|v| v * RAW_AMPLITUDE)
            .collect();
        let got = q_rope(gpu, &rows, shape, &pos);
        for (tok, &at) in pos.iter().enumerate() {
            for head in 0..ROPE_HEADS {
                let base = (tok * ROPE_HEADS + head) * shape.head_dim;
                let raw = &rows[base..base + shape.head_dim];
                let want = rope(raw.to_vec(), shape, at as f32);
                close(
                    &got[base..base + shape.head_dim],
                    &want,
                    TOL_F32,
                    &format!("{}-wide head, token {tok} head {head}", shape.head_dim),
                );
                // Position 0 rotates by nothing, so the row is the input to
                // the last bit: the kernel added no scale of its own.
                if at == 0 {
                    close(
                        &got[base..base + shape.head_dim],
                        raw,
                        0.0,
                        "position 0 is the identity",
                    );
                }
            }
        }
    }
}

/// The rotate-only store: the key rotated, the value raw, both in the cache row
/// the page table names and neither normalized.
pub fn the_rope_only_store_keeps_the_value_raw(gpu: &Gpu) {
    let keys: Vec<f32> = crate::oracle::signal(KV_TOK * KV_HEADS * KV_HD, 74)
        .iter()
        .map(|v| v * RAW_AMPLITUDE)
        .collect();
    let values: Vec<f32> = crate::oracle::signal(KV_TOK * KV_HEADS * KV_HD, 75)
        .iter()
        .map(|v| v * RAW_AMPLITUDE)
        .collect();
    let pos: Vec<i32> = (0..KV_TOK as i32).map(|t| t * 5 + 2).collect();

    let d_k = gpu.upload(&f32_bytes(&keys));
    let d_v = gpu.upload(&f32_bytes(&values));
    let d_pos = gpu.upload(&i32_bytes(&pos));
    let pages = page_table(gpu, SHUFFLED_PAGES);
    let rows = SHUFFLED_PAGES.rows();
    let k_cache = zeros_f16(gpu, rows * KV_HEADS * KV_HD);
    let v_cache = zeros_f16(gpu, rows * KV_HEADS * KV_HD);

    let args = KvNormRopeArgs {
        head_dim: KV_HD as u32,
        n_head_kv: KV_HEADS as u32,
        n_rot: KV_SHAPE.n_rot as u32,
        freq_dims: KV_SHAPE.freq_dims as u32,
        n_tok: KV_TOK as u32,
        rope_base: KV_SHAPE.base,
        eps: 0.0,
        page_tokens: SHUFFLED_PAGES.page_tokens,
        // Set on purpose: the kernel must not read it.
        v_norm: 1,
        page_stride: 0,
    };
    launch(
        gpu,
        "kernel_dense_kv_rope_store",
        Extent3::new(KV_HEADS as u32, KV_TOK as u32, 1),
        Extent3::new(BLOCK_128, 1, 1),
        NORM_ROPE_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(k_cache.view()),
            KernelArg::Buffer(v_cache.view()),
            KernelArg::Buffer(d_k.view()),
            KernelArg::Buffer(d_v.view()),
            KernelArg::Buffer(d_pos.view()),
            KernelArg::Buffer(pages.view()),
        ],
    );
    drain(gpu);

    let got_k = f16_values(&gpu.read(&k_cache));
    let got_v = f16_values(&gpu.read(&v_cache));
    for (tok, &at) in pos.iter().enumerate() {
        for head in 0..KV_HEADS {
            let src = (tok * KV_HEADS + head) * KV_HD;
            let slot = (SHUFFLED_PAGES.row(at as u32) * KV_HEADS + head) * KV_HD;
            let want_k = rope(keys[src..src + KV_HD].to_vec(), KV_SHAPE, at as f32);
            close(
                &got_k[slot..slot + KV_HD],
                &want_k,
                TOL_QUANT,
                &format!("key token {tok} head {head}"),
            );
            close(
                &got_v[slot..slot + KV_HD],
                &values[src..src + KV_HD],
                TOL_QUANT,
                &format!("value token {tok} head {head}"),
            );
        }
    }
    // Row 0 belongs to logical position 16, which this chunk never names.
    assert!(
        got_k[..KV_HD].iter().all(|v| *v == 0.0),
        "row 0 was written, so the store went by token index rather than through the page table",
    );
}

// -- the softmax ------------------------------------------------------------

/// The scale the query fixture is multiplied by before it reaches the device.
///
/// Without it the fixture is one-hot. A 256-wide dot of two rows in `[-1, 1.75]`
/// spans about 264, so at a score scale of 1 the largest key in a span carries a
/// probability of 1 to within `f32`, every other key contributes below the
/// tolerance, and a kernel that dropped or duplicated a NON-maximal key would
/// agree with the oracle anyway. A power of two, so the f16 keys and values stay
/// exact and only the f32 query row moves; `MIN_IN_SPAN_WEIGHT` is what keeps it
/// honest from here on.
const QUERY_AMPLITUDE: f32 = 1.0 / 256.0;

/// The smallest share of a softmax any key inside a span must carry.
///
/// Asserted on the ORACLE, not on the device: it is a statement about the
/// fixture rather than about the kernel. Every in-span key has to matter by more
/// than [`TOL_QUANT`] by a wide margin, or the comparison stops being able to
/// see a key the kernel skipped. 0.02 is twenty times the tolerance, and the
/// spans below run between 0.031 and 0.125 at every scale they are launched at.
const MIN_IN_SPAN_WEIGHT: f32 = 0.02;

/// The keys one query at `pos` with `bound` may read under `window`.
///
/// `0` is no window at all, which makes the span everything up to the bound -
/// and the bound is not the query's own position, so a span reaching forward is
/// a span this kernel has to serve.
pub(super) fn key_span(pos: u32, bound: u32, window: u32) -> std::ops::RangeInclusive<u32> {
    let lo = if window != 0 && pos + 1 > window {
        pos + 1 - window
    } else {
        0
    };
    lo..=bound
}

/// One softmax case: where the queries sit, how far each may read, the window,
/// the score scale and the page table the cache is reached through.
#[derive(Debug, Clone, Copy)]
struct Span<'a> {
    pos: &'a [i32],
    bound: &'a [i32],
    window: u32,
    scale: f32,
    paging: Paging,
}

/// The straight-through masked softmax the two entry points replace: score every
/// key in the span, run one softmax over those scores, blend the values by it.
///
/// Returns the blended rows and the smallest probability any in-span key was
/// given, which is what [`check_softmax`] holds the fixture to.
fn attn_oracle(query: &[f32], keys: &[f32], values: &[f32], span: Span<'_>) -> (Vec<f32>, f32) {
    let n_tok = span.pos.len();
    let mut out = vec![0.0f32; n_tok * Q_HEADS * KV_HD];
    let mut smallest = 1.0f32;
    for tok in 0..n_tok {
        let range = key_span(span.pos[tok] as u32, span.bound[tok] as u32, span.window);
        for head in 0..Q_HEADS {
            let hkv = head / (Q_HEADS / KV_HEADS);
            let qb = (tok * Q_HEADS + head) * KV_HD;
            let scores: Vec<f32> = range
                .clone()
                .map(|at| {
                    let kb = (span.paging.row(at) * KV_HEADS + hkv) * KV_HD;
                    let dot: f32 = (0..KV_HD).map(|dim| query[qb + dim] * keys[kb + dim]).sum();
                    dot * span.scale
                })
                .collect();
            // An empty span contributes a row of zeros - which is what both
            // entry points do, and what `KERNELS.md` requires of them. The row
            // is already zero, so there is nothing left to normalize; falling
            // through would divide it by a denominator of zero and grade a
            // correct kernel against NaN. No case fed this reference an empty
            // span until one was written for it, which is why it read as
            // arithmetic that could not happen.
            if scores.is_empty() {
                continue;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let den: f32 = weights.iter().sum();
            for (index, at) in range.clone().enumerate() {
                let vb = (span.paging.row(at) * KV_HEADS + hkv) * KV_HD;
                smallest = smallest.min(weights[index] / den);
                for dim in 0..KV_HD {
                    out[qb + dim] += weights[index] * values[vb + dim];
                }
            }
            for dim in 0..KV_HD {
                out[qb + dim] /= den;
            }
        }
    }
    (out, smallest)
}

/// Runs one softmax entry point over the fixture and compares it with the oracle.
///
/// Both entry points are driven through this one helper because they are one
/// softmax over the same rows: a windowed span the decode kernel gets right and
/// the tiled one does not is exactly the bug a shared fixture catches.
fn check_softmax(gpu: &Gpu, entry: &'static str, span: Span<'_>) {
    let n_tok = span.pos.len();
    let rows = span.paging.rows();
    let query: Vec<f32> = fixture(n_tok * Q_HEADS * KV_HD, 1)
        .iter()
        .map(|v| v * QUERY_AMPLITUDE)
        .collect();
    let keys = fixture(rows * KV_HEADS * KV_HD, 2);
    let values = fixture(rows * KV_HEADS * KV_HD, 3);

    let d_q = gpu.upload(&f32_bytes(&query));
    let d_k = gpu.upload(&f16_bytes(&keys));
    let d_v = gpu.upload(&f16_bytes(&values));
    let d_pos = gpu.upload(&i32_bytes(span.pos));
    let d_bound = gpu.upload(&i32_bytes(span.bound));
    let pages = page_table(gpu, span.paging);
    let out = zeros(gpu, n_tok * Q_HEADS * KV_HD);

    let args = AttnArgs {
        head_dim: KV_HD as u32,
        n_head: Q_HEADS as u32,
        n_head_kv: KV_HEADS as u32,
        n_tok: n_tok as u32,
        page_tokens: span.paging.page_tokens,
        window: span.window,
        scale: span.scale,
        page_stride: 0,
    };
    let (grid, smem) = if entry == PREFILL {
        (
            Extent3::new(Q_HEADS as u32, (n_tok as u32).div_ceil(QT as u32), 1),
            PREFILL_SMEM,
        )
    } else {
        (Extent3::new(Q_HEADS as u32, n_tok as u32, 1), DECODE_SMEM)
    };
    launch(
        gpu,
        entry,
        grid,
        Extent3::new(BLOCK_256, 1, 1),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_q.view()),
            KernelArg::Buffer(d_k.view()),
            KernelArg::Buffer(d_v.view()),
            KernelArg::Buffer(d_pos.view()),
            KernelArg::Buffer(d_bound.view()),
            KernelArg::Buffer(pages.view()),
        ],
    );
    drain(gpu);

    let (want, smallest) = attn_oracle(&query, &keys, &values, span);
    let what = format!(
        "{entry} {} queries window {} scale {}",
        n_tok, span.window, span.scale,
    );
    // The fixture, not the device: a span whose softmax has collapsed onto one
    // key cannot see a kernel that dropped any of the others.
    assert!(
        smallest > MIN_IN_SPAN_WEIGHT,
        "{what}: the softmax has collapsed - the smallest in-span weight is \
         {smallest}, at or below {MIN_IN_SPAN_WEIGHT}, so most of the span \
         contributes less than the comparison tolerance",
    );
    close(&gpu.read_f32(&out), &want, TOL_QUANT, &what);
}

/// The spans both entry points are held to, at a given score scale.
///
/// A sliding window of 8 at position 39 leaves keys 32..=39 and nothing before.
/// No window with a bound ahead of the query's own position leaves keys 0..=20,
/// a span that reaches forward, which is what a bidirectional prefix needs. Each
/// is run through the decode entry with one query and through the tiled entry
/// with a full tile of sixteen, one bound per query; the last case runs the
/// tiled entry short.
fn check_every_span(gpu: &Gpu, scale: f32) {
    check_softmax(
        gpu,
        DECODE,
        Span {
            pos: &[39],
            bound: &[39],
            window: 8,
            scale,
            paging: ONE_PAGE,
        },
    );
    // Through the scattered table: keys 0..=20 are two pages, and the second
    // page's rows sit below the first's.
    check_softmax(
        gpu,
        DECODE,
        Span {
            pos: &[10],
            bound: &[20],
            window: 0,
            scale,
            paging: SHUFFLED_PAGES,
        },
    );

    // Sixteen queries at sixteen positions, each with its own window start: the
    // tile stages keys 17..=39 and every query masks it down to its own eight.
    let windowed: Vec<i32> = (24..24 + QT as i32).collect();
    check_softmax(
        gpu,
        PREFILL,
        Span {
            pos: &windowed,
            bound: &windowed,
            window: 8,
            scale,
            paging: ONE_PAGE,
        },
    );
    let spread: Vec<i32> = (5..5 + QT as i32).collect();
    let bounds = vec![20i32; QT];
    check_softmax(
        gpu,
        PREFILL,
        Span {
            pos: &spread,
            bound: &bounds,
            window: 0,
            scale,
            paging: ONE_PAGE,
        },
    );

    // Nineteen queries: a full tile and then a short one of three, which is the
    // only case that reaches the `qi >= n_q` breaks, the tile-bound scan over a
    // partial tile and the final store loop with simdgroups that own no query.
    // Through the scattered table, and with a window of 12, so the first tile's
    // key range (10..=37) walks two page edges.
    let short: Vec<i32> = (21..21 + SHORT_TILE as i32).collect();
    let short_bounds: Vec<i32> = short.iter().map(|&p| (p + 1).min(39)).collect();
    check_softmax(
        gpu,
        PREFILL,
        Span {
            pos: &short,
            bound: &short_bounds,
            window: 12,
            scale,
            paging: SHUFFLED_PAGES,
        },
    );
}

/// Queries in the case that runs the prefill entry's last tile short: one full
/// tile of sixteen and one of three.
const SHORT_TILE: usize = 19;

const _: () = assert!(SHORT_TILE > QT && !SHORT_TILE.is_multiple_of(QT));

/// The two ends of the span, through both entry points.
///
/// The window bounds the span from below and the per-token bound from above, and
/// neither is derivable from the other: a kernel that walked `0..=pos` would pass
/// nothing here.
pub fn window_and_bound_select_the_keys(gpu: &Gpu) {
    check_every_span(gpu, 1.0);
}

/// The score scale is an argument rather than `1 / sqrt(head_dim)`: this
/// family's layers pass 1.0 and qwen35's pass the reciprocal root, and only a
/// kernel that reads the number can serve both.
///
/// Two of them, because the scale has to change the ANSWER and not only the
/// arithmetic: 1/sqrt(256) leaves a span noticeably peaked and a hundredth
/// leaves it nearly flat, so a kernel that ignored the argument disagrees with
/// one of these however it was launched.
pub fn scale_is_an_argument(gpu: &Gpu) {
    // 1/sqrt(256), which is what a layer of this width would carry.
    check_every_span(gpu, 0.0625);
    check_every_span(gpu, 0.01);
}

/// What an untouched output element still holds when a kernel returns.
///
/// The engine's attention output is scratch: on a real step it carries the
/// previous chunk's rows rather than zeros. A case starting from zeros could not
/// tell "wrote a row of zeros" from "wrote nothing at all", and the contract
/// asks for the first. Finite and enormous, so a failure reads as untouched
/// rather than as arithmetic.
pub(super) const POISON: f32 = -1.0e30;

/// An `f32` device buffer of `len` copies of [`POISON`].
pub(super) fn poisoned(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![POISON; len]))
}

/// One empty-span case: launch over a poisoned output and hold every element to
/// the oracle, which leaves an empty query's row at zero.
fn check_empty_span(gpu: &Gpu, entry: &'static str, span: Span<'_>) {
    let n_tok = span.pos.len();
    let rows = span.paging.rows();
    let query: Vec<f32> = fixture(n_tok * Q_HEADS * KV_HD, 1)
        .iter()
        .map(|v| v * QUERY_AMPLITUDE)
        .collect();
    let keys = fixture(rows * KV_HEADS * KV_HD, 2);
    let values = fixture(rows * KV_HEADS * KV_HD, 3);

    let d_q = gpu.upload(&f32_bytes(&query));
    let d_k = gpu.upload(&f16_bytes(&keys));
    let d_v = gpu.upload(&f16_bytes(&values));
    let d_pos = gpu.upload(&i32_bytes(span.pos));
    let d_bound = gpu.upload(&i32_bytes(span.bound));
    let pages = page_table(gpu, span.paging);
    let out = poisoned(gpu, n_tok * Q_HEADS * KV_HD);

    let args = AttnArgs {
        head_dim: KV_HD as u32,
        n_head: Q_HEADS as u32,
        n_head_kv: KV_HEADS as u32,
        n_tok: n_tok as u32,
        page_tokens: span.paging.page_tokens,
        window: span.window,
        scale: span.scale,
        page_stride: 0,
    };
    let (grid, smem) = if entry == PREFILL {
        (
            Extent3::new(Q_HEADS as u32, (n_tok as u32).div_ceil(QT as u32), 1),
            PREFILL_SMEM,
        )
    } else {
        (Extent3::new(Q_HEADS as u32, n_tok as u32, 1), DECODE_SMEM)
    };
    launch(
        gpu,
        entry,
        grid,
        Extent3::new(BLOCK_256, 1, 1),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(d_q.view()),
            KernelArg::Buffer(d_k.view()),
            KernelArg::Buffer(d_v.view()),
            KernelArg::Buffer(d_pos.view()),
            KernelArg::Buffer(d_bound.view()),
            KernelArg::Buffer(pages.view()),
        ],
    );
    drain(gpu);

    let got = gpu.read_f32(&out);
    let (want, _) = attn_oracle(&query, &keys, &values, span);
    let what = format!("{entry} {n_tok} queries window {}", span.window);

    // Named separately from the comparison below, because these are the two
    // wrong answers this case exists for and neither reads as a tolerance miss.
    // A NaN row is what an unguarded `-inf` softmax over nothing produces, and
    // it reaches the residual stream and makes the whole answer plausible and
    // wrong; an untouched row is a kernel that returned early and left the
    // previous step's values in place.
    for (index, &have) in got.iter().enumerate() {
        assert!(
            !have.is_nan(),
            "{what}: element {index} is NaN - an empty span took the softmax over \
             no keys instead of contributing a row of zeros",
        );
        assert!(
            have != POISON,
            "{what}: element {index} is untouched - the kernel returned without \
             writing it, so a real step would keep whatever the last one left",
        );
    }
    close(&got, &want, TOL_QUANT, &what);
}

/// A span with no keys in it contributes a row of ZEROS, and does so by writing
/// them.
///
/// `KERNELS.md` states it: "An empty span (a bound below the window's low end)
/// contributes a row of zeros rather than a row of NaN." A sliding window of 8
/// at position 39 opens at key 32, and a bound of 20 closes below it, so the
/// query has nothing to read - which happens whenever a bidirectional prefix's
/// bound sits behind a window that has already moved past it.
///
/// Three shapes, because the ways to get this wrong differ between the entries.
/// The decode entry runs one threadgroup per query and can only fail alone. The
/// prefill entry stages a key tile for sixteen queries at once and masks each by
/// its own span, so a tile whose queries are ALL empty exercises the guard with
/// no in-span key anywhere to rescue it, and a MIXED tile catches the sharper
/// bug: a running maximum or a denominator shared across the tile, where one
/// empty query's `-inf` poisons the fifteen beside it that were perfectly fine.
pub fn an_empty_span_contributes_zeros_rather_than_nan(gpu: &Gpu) {
    // Decode, one query, nothing to read.
    check_empty_span(
        gpu,
        DECODE,
        Span {
            pos: &[39],
            bound: &[20],
            window: 8,
            scale: 1.0,
            paging: ONE_PAGE,
        },
    );

    // Prefill, a whole tile of them.
    let empty_pos = vec![39i32; QT];
    let empty_bound = vec![20i32; QT];
    check_empty_span(
        gpu,
        PREFILL,
        Span {
            pos: &empty_pos,
            bound: &empty_bound,
            window: 8,
            scale: 1.0,
            paging: ONE_PAGE,
        },
    );

    // Prefill, half the tile empty and half of it not. The live queries sit at
    // 24..31 reading 16..=their own position; the empty ones are the same
    // 39/20 pair as above, interleaved so neither half is a contiguous block.
    let mixed_pos: Vec<i32> = (0..QT)
        .map(|i| if i % 2 == 0 { 39 } else { 24 + i as i32 })
        .collect();
    let mixed_bound: Vec<i32> = (0..QT)
        .map(|i| if i % 2 == 0 { 20 } else { 24 + i as i32 })
        .collect();
    check_empty_span(
        gpu,
        PREFILL,
        Span {
            pos: &mixed_pos,
            bound: &mixed_bound,
            window: 8,
            scale: 1.0,
            paging: ONE_PAGE,
        },
    );
}

/// Cache positions the bench decode reads: a 4096-token context.
const BENCH_CTX: usize = 4096;

/// One decode query per head over a `BENCH_CTX`-long, contiguously paged f16
/// cache, at the launch the engine uses for one token, for [`crate::bench`].
pub(crate) fn bench_decode(gpu: &Gpu) -> crate::bench::Fixture {
    const PAGE: u32 = 64;
    let pages: Vec<i32> = (0..(BENCH_CTX as u32 / PAGE) as i32).collect();
    let cache_len = BENCH_CTX * KV_HEADS * KV_HD;
    let d_q = gpu.upload(&f32_bytes(&fixture(Q_HEADS * KV_HD, 1)));
    let d_k = gpu.upload(&f16_bytes(&fixture(cache_len, 2)));
    let d_v = gpu.upload(&f16_bytes(&fixture(cache_len, 3)));
    let last = BENCH_CTX as i32 - 1;
    let d_pos = gpu.upload(&i32_bytes(&[last]));
    let d_bound = gpu.upload(&i32_bytes(&[last]));
    let d_pages = gpu.upload(&i32_bytes(&pages));
    let out = zeros(gpu, Q_HEADS * KV_HD);
    let args = AttnArgs {
        head_dim: KV_HD as u32,
        n_head: Q_HEADS as u32,
        n_head_kv: KV_HEADS as u32,
        n_tok: 1,
        page_tokens: PAGE,
        window: 0,
        scale: 1.0 / (KV_HD as f32).sqrt(),
        page_stride: 0,
    };
    let bytes = (2 * cache_len * 2 + 2 * Q_HEADS * KV_HD * 4) as u64;
    let flops = (4 * Q_HEADS * BENCH_CTX * KV_HD) as u64;
    crate::bench::Fixture::new(bytes, flops, move |gpu| {
        launch(
            gpu,
            DECODE,
            Extent3::new(Q_HEADS as u32, 1, 1),
            Extent3::new(BLOCK_256, 1, 1),
            DECODE_SMEM,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(d_q.view()),
                KernelArg::Buffer(d_k.view()),
                KernelArg::Buffer(d_v.view()),
                KernelArg::Buffer(d_pos.view()),
                KernelArg::Buffer(d_bound.view()),
                KernelArg::Buffer(d_pages.view()),
            ],
        );
    })
}
