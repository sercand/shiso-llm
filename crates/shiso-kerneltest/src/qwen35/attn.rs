//! Device cases for the qwen35 gated-attention kernels.
//!
//! Part of task-05 (docs/superpowers/plans/tasks/task-05-qwen35-msl-kernels.md).
//!
//! The oracle is the straight-through masked softmax the kernels replace: every
//! position a query may see is scored, one softmax runs over those scores, and the
//! values are blended by the result.  Both kernels instead keep a running max and
//! rescale as they walk positions, and the prefill kernel splits positions over
//! lanes as well as warps, so the three shapes agree only if every rescaling is
//! right.  That is what these cases pin down on a device.
//!
//! The `#[test]` wrappers live in each backend's own driver -
//! `crates/shiso-metal/tests/q35_attn.rs`, `crates/shiso-cuda/tests/q35_attn.rs`.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "kernel argument structs take u32 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]

use crate::oracle;

use crate::harness::{Gpu, drain, f32_bytes, run};
use half::f16;
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

/// Attention `head_dim`, not the 128-wide GDN heads.
const HD: usize = 256;
const N_HEAD: usize = 16;
const N_HEAD_KV: usize = 4;
/// Cache slots.  A token may read every slot up to its own position, so the cache
/// is deeper than the token count and a slot is exercised as a position rather
/// than as a token index.
const N_POS: usize = 64;
const N_TOK: usize = 64;
/// Both attention kernels run 8 simdgroups of 32 lanes.
const BLOCK: u32 = 256;
/// Queries per prefill threadgroup (`Q35_ATTN_QT`).
const QT: usize = 16;
/// Threadgroup memory each kernel asks for, per `KERNELS.md`: `s_v[8][256]` plus
/// the two 8-slot partial arrays for decode, and the transposed key tile for
/// prefill.
const DECODE_SMEM: u32 = 8_256;
const PREFILL_SMEM: u32 = 16_384;

/// Fixture values: multiples of an eighth in [-1, 1.75], which are exact in
/// `f16`, so a cache round-trip is exact and any difference is arithmetic.  The
/// period is coprime with the head, position and dimension strides, so no two
/// heads and no two positions read the same row.
fn fixture(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7 + salt * 31) % 23) as f32 * 0.125 - 1.0)
        .collect()
}

/// `f16` bits for one fixture value, through the same conversion the loader uses.
fn f16_bit(value: f32) -> u16 {
    u16::from_le_bytes(f16::from_f32(value).to_le_bytes())
}

/// `f16` bits for a whole fixture.
fn f16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().copied().map(f16_bit).collect()
}

/// Little-endian bytes of `f16` bits, for upload.
fn f16_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().copied().flat_map(u16::to_le_bytes).collect()
}

/// The position array as `int32_t` little-endian, which is what the kernels index
/// the cache by.
fn pos_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().copied().flat_map(i32::to_le_bytes).collect()
}

/// The shared fixture for `n_tok` tokens: the query rows, the two cache halves as
/// `f16` bits, and the positions.  Token `t` reads slots `0..=t`.
fn fixtures(n_tok: usize) -> (Vec<f32>, Vec<u16>, Vec<u16>, Vec<i32>) {
    let q = fixture(n_tok * N_HEAD * HD, 1);
    let k = fixture(N_POS * N_HEAD_KV * HD, 2);
    let v = fixture(N_POS * N_HEAD_KV * HD, 3);
    let pos: Vec<i32> = (0..n_tok)
        .map(|t| i32::try_from(t).expect("the fixture is far shorter than i32 holds"))
        .collect();
    (q, f16_bits(&k), f16_bits(&v), pos)
}

/// `q35_args_attn_decode` and `q35_args_attn_prefill` are the same six fields.
///
/// `page_stride` is the last of them: entries between one row's page table and
/// the next inside one packed block, and `0` for every launch whose rows share
/// one table. Zero is the shared case rather than a sentinel, because
/// `pages + t * 0` is `pages` for every row.
fn attn_args(n_head: u32, n_head_kv: u32, n_tok: u32, page_stride: u32) -> [u8; 24] {
    let mut out = [0u8; 24];
    for (slot, field) in [
        HD as u32,
        n_head,
        n_head_kv,
        n_tok,
        PAGE_TOKENS,
        page_stride,
    ]
    .iter()
    .enumerate()
    {
        out[slot * 4..(slot + 1) * 4].copy_from_slice(&field.to_le_bytes());
    }
    out
}

/// `q35_args_attn_gate {head_dim, n_head, n_tok}`.
fn gate_args(n_head: u32, n_tok: u32) -> [u8; 12] {
    let mut out = [0u8; 12];
    for (slot, field) in [HD as u32, n_head, n_tok].iter().enumerate() {
        out[slot * 4..(slot + 1) * 4].copy_from_slice(&field.to_le_bytes());
    }
    out
}

/// One decode launch over rows that share one page table.
fn decode(gpu: &Gpu, out: &DeviceBuffer, inputs: [&DeviceBuffer; 5], n_tok: u32) {
    decode_rows(gpu, out, inputs, n_tok, 0);
}

/// One decode launch: a threadgroup per (head, token), positions walked by that
/// threadgroup's 8 simdgroups.
///
/// `page_stride` is entries between one row's page table and the next; `0`
/// means every row reads the table the pointer names.
fn decode_rows(
    gpu: &Gpu,
    out: &DeviceBuffer,
    inputs: [&DeviceBuffer; 5],
    n_tok: u32,
    page_stride: u32,
) {
    let [q, k_cache, v_cache, pos, pages] = inputs;

    let args = attn_args(N_HEAD as u32, N_HEAD_KV as u32, n_tok, page_stride);
    let params = [
        KernelArg::Bytes(&args),
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(q.view()),
        KernelArg::Buffer(k_cache.view()),
        KernelArg::Buffer(v_cache.view()),
        KernelArg::Buffer(pos.view()),
        KernelArg::Buffer(pages.view()),
    ];
    run(
        gpu,
        KernelRef::new("qwen35", "kernel_q35_attn_decode"),
        &LaunchArgs {
            grid: Extent3::new(N_HEAD as u32, n_tok, 1),
            block: Extent3::linear(BLOCK),
            smem_bytes: DECODE_SMEM,
            args: &params,
        },
    );
    drain(gpu);
}

/// Straight-through causal GQA for one (token, head): score every allowed
/// position, softmax them once, then blend the values.
fn attn_oracle(
    q_rows: &[f32],
    k_bits: &[u16],
    v_bits: &[u16],
    last: i32,
    tok: usize,
    head: usize,
) -> Vec<f32> {
    // 16 query heads share 4 key/value heads: an integer divide, as in the kernel.
    let hkv = head / (N_HEAD / N_HEAD_KV);
    let allowed = usize::try_from(last).expect("a fixture position is never negative");
    let q_row = &q_rows[(tok * N_HEAD + head) * HD..][..HD];
    let scores: Vec<f32> = (0..=allowed)
        .map(|pos| {
            let slot = &k_bits[(pos * N_HEAD_KV + hkv) * HD..][..HD];
            q_row
                .iter()
                .zip(slot)
                .map(|(q_val, k_val)| q_val * oracle::f16_to_f32(*k_val))
                .sum()
        })
        .collect();
    // The kernel folds this into q before the dot; scaling the score afterwards is
    // the same number, and 1/sqrt(256) is exact.
    let w = oracle::softmax_row(&scores, 1.0 / (HD as f32).sqrt());
    (0..HD)
        .map(|dim| {
            w.iter()
                .enumerate()
                .map(|(pos, weight)| {
                    let value = v_bits[(pos * N_HEAD_KV + hkv) * HD + dim];
                    weight * oracle::f16_to_f32(value)
                })
                .sum()
        })
        .collect()
}

/// The fixture is only exact if the `f16` round-trip is exact, so the claim is
/// checked rather than assumed.
pub fn fixture_values_survive_the_f16_cache() {
    for salt in 0..4 {
        for value in fixture(23 * 8, salt) {
            // Compared as bits on purpose: the claim is exactness, not agreement
            // within a tolerance.
            assert_eq!(
                oracle::f16_to_f32(f16_bit(value)).to_bits(),
                value.to_bits(),
                "{value} is not exact in f16",
            );
        }
    }
}

/// Decode against the naive masked softmax, over a cache deeper than the token
/// count so a position is read as a slot rather than as a token index.
pub fn decode_matches_a_naive_masked_softmax(gpu: &Gpu) {
    let (q, k_bits, v_bits, pos) = fixtures(N_TOK);
    let out = gpu.upload(&f32_bytes(&vec![0.0; N_TOK * N_HEAD * HD]));
    let q_buf = gpu.upload(&f32_bytes(&q));
    let k_cache = gpu.upload(&f16_bytes(&k_bits));
    let v_cache = gpu.upload(&f16_bytes(&v_bits));
    let pos_buf = gpu.upload(&pos_bytes(&pos));
    let pages = pages_buf(gpu);

    decode(
        gpu,
        &out,
        [&q_buf, &k_cache, &v_cache, &pos_buf, &pages],
        N_TOK as u32,
    );

    let got = gpu.read_f32(&out);
    for t in 0..N_TOK {
        for h in 0..N_HEAD {
            let expected = attn_oracle(&q, &k_bits, &v_bits, pos[t], t, h);
            for d in 0..HD {
                crate::assert_close!(
                    got[(t * N_HEAD + h) * HD + d],
                    expected[d],
                    oracle::TOL_QUANT,
                    "token {t} head {h} dim {d}",
                );
            }
        }
    }
}

/// Prefill against 32 sequential decode calls over the same cache.  The two shapes
/// share nothing: decode hands positions to simdgroups, prefill scores one key per
/// lane and folds each tile with a max butterfly.
pub fn prefill_matches_token_by_token_decode(gpu: &Gpu) {
    const N: usize = 32;
    let (q, k_bits, v_bits, pos) = fixtures(N);
    let seq_out = gpu.upload(&f32_bytes(&vec![0.0; N * N_HEAD * HD]));
    let pf_out = gpu.upload(&f32_bytes(&vec![0.0; N * N_HEAD * HD]));
    let q_buf = gpu.upload(&f32_bytes(&q));
    let k_cache = gpu.upload(&f16_bytes(&k_bits));
    let v_cache = gpu.upload(&f16_bytes(&v_bits));
    let pos_buf = gpu.upload(&pos_bytes(&pos));
    let pages = pages_buf(gpu);

    // One launch per token, each covering the tokens up to and including it, so the
    // last call holds every token and no call sees a wider cache than a step-at-a
    // time decoder would.
    for n_tok in 1..=N {
        decode(
            gpu,
            &seq_out,
            [&q_buf, &k_cache, &v_cache, &pos_buf, &pages],
            n_tok as u32,
        );
    }

    let args = attn_args(N_HEAD as u32, N_HEAD_KV as u32, N as u32, 0);
    let params = [
        KernelArg::Bytes(&args),
        KernelArg::Buffer(pf_out.view()),
        KernelArg::Buffer(q_buf.view()),
        KernelArg::Buffer(k_cache.view()),
        KernelArg::Buffer(v_cache.view()),
        KernelArg::Buffer(pos_buf.view()),
        KernelArg::Buffer(pages.view()),
    ];
    run(
        gpu,
        KernelRef::new("qwen35", "kernel_q35_attn_prefill"),
        &LaunchArgs {
            grid: Extent3::new(N_HEAD as u32, N.div_ceil(QT) as u32, 1),
            block: Extent3::linear(BLOCK),
            smem_bytes: PREFILL_SMEM,
            args: &params,
        },
    );
    drain(gpu);

    let seq = gpu.read_f32(&seq_out);
    let pf = gpu.read_f32(&pf_out);
    for (i, (got, expected)) in pf.iter().zip(&seq).enumerate() {
        crate::assert_close!(*got, *expected, oracle::TOL_QUANT, "flat index {i}");
    }
}

/// Token 0 may see exactly one position, so its output is that position's value
/// row for its shared key/value head: no weight, no blend.  Anything else means
/// the causal bound leaked.
pub fn token_zero_returns_only_its_own_value_row(gpu: &Gpu) {
    let (q, k_bits, v_bits, pos) = fixtures(N_TOK);
    let out = gpu.upload(&f32_bytes(&vec![0.0; N_TOK * N_HEAD * HD]));
    let q_buf = gpu.upload(&f32_bytes(&q));
    let k_cache = gpu.upload(&f16_bytes(&k_bits));
    let v_cache = gpu.upload(&f16_bytes(&v_bits));
    let pos_buf = gpu.upload(&pos_bytes(&pos));
    let pages = pages_buf(gpu);

    decode(gpu, &out, [&q_buf, &k_cache, &v_cache, &pos_buf, &pages], 1);

    let got = gpu.read_f32(&out);
    for h in 0..N_HEAD {
        let hkv = h / (N_HEAD / N_HEAD_KV);
        let row = &v_bits[hkv * HD..][..HD];
        for d in 0..HD {
            crate::assert_close!(
                got[h * HD + d],
                oracle::f16_to_f32(row[d]),
                oracle::TOL_F32,
                "head {h} dim {d} of token 0",
            );
        }
    }
}

/// The output gate is `x *= sigmoid(gate)` with nothing normalized in front of it,
/// so it is held to the `f32` tolerance rather than the cache one.
pub fn gate_mul_applies_the_sigmoid_gate(gpu: &Gpu) {
    const N_GATE: usize = 3;
    let n = N_GATE * N_HEAD * HD;
    let x = fixture(n, 5);
    let gate = fixture(n, 6);
    let x_buf = gpu.upload(&f32_bytes(&x));
    let gate_buf = gpu.upload(&f32_bytes(&gate));

    let args = gate_args(N_HEAD as u32, N_GATE as u32);
    let params = [
        KernelArg::Bytes(&args),
        KernelArg::Buffer(x_buf.view()),
        KernelArg::Buffer(gate_buf.view()),
    ];
    run(
        gpu,
        KernelRef::new("qwen35", "kernel_q35_attn_gate_mul"),
        &LaunchArgs {
            grid: Extent3::linear(n.div_ceil(BLOCK as usize) as u32),
            block: Extent3::linear(BLOCK),
            smem_bytes: 0,
            args: &params,
        },
    );
    drain(gpu);

    let got = gpu.read_f32(&x_buf);
    for i in 0..n {
        crate::assert_close!(
            got[i],
            x[i] * (1.0 / (1.0 + (-gate[i]).exp())),
            oracle::TOL_F32,
            "flat index {i}",
        );
    }
}

// Both attention kernels translate a position through a page table before they
// touch a row, so every launch needs a table and a page size. These cases want
// the rows to land where the oracle thinks they are, so their table is the
// identity: physical page i holds logical page i.
const PAGE_TOKENS: u32 = 16;

fn identity_pages(n: usize) -> Vec<i32> {
    (0..n)
        .map(|p| i32::try_from(p).expect("a fixture page index fits an i32"))
        .collect()
}

fn page_bytes(pages: &[i32]) -> Vec<u8> {
    pages.iter().copied().flat_map(i32::to_le_bytes).collect()
}

/// The identity table as a device buffer, sized for the whole fixture cache.
fn pages_buf(gpu: &Gpu) -> DeviceBuffer {
    let n = N_POS.div_ceil(PAGE_TOKENS as usize);
    gpu.upload(&page_bytes(&identity_pages(n)))
}

/// Two rows in two different page tables agree with the same two rows run
/// alone.
///
/// The defect this is built to catch is a single `pages` pointer shared by
/// every row of a batch. With both rows in one table that defect is invisible:
/// the shared table happens to be the right table. So the fixture gives row one
/// the identity table and row two a rotated one, puts row one at a position
/// inside its first page and row two past its own page boundary, and the only
/// way both answers are right is if each row read its own table.
///
/// The budget is **0.0**, not a tolerance, and the comparison is on bits.
/// Batching changes which rows are issued together, not the arithmetic any one
/// row does: the same query row folds the same keys in the same order whether
/// it is `blockIdx.y = 0` of a one-row launch or `blockIdx.y = 1` of a two-row
/// one. A non-zero delta here is a defect, never noise.
pub fn two_rows_in_two_pages_match_the_rows_run_alone(gpu: &Gpu) {
    const ROWS: usize = 2;
    let (q, k_bits, v_bits, _pos) = fixtures(ROWS);
    // Row zero sits inside its table's first page; row one sits past its own
    // table's first page boundary, which is the only reason a rotated table
    // reaches a different physical row for it.
    let page = PAGE_TOKENS.cast_signed();
    let positions = [page - 1, page + 3];
    let pages = N_POS.div_ceil(PAGE_TOKENS as usize);
    let identity = identity_pages(pages);
    // Rotated by half the table, so no logical page maps to the page the
    // identity table would have given it.
    let rotated: Vec<i32> = (0..pages)
        .map(|p| identity[(p + pages / 2) % pages])
        .collect();

    let k_cache = gpu.upload(&f16_bytes(&k_bits));
    let v_cache = gpu.upload(&f16_bytes(&v_bits));
    let q_buf = gpu.upload(&f32_bytes(&q));

    // The batched launch: both rows, one packed table of two rows, `stride`
    // entries apart.
    let mut packed = page_bytes(&identity);
    packed.extend(page_bytes(&rotated));
    let batched_out = gpu.upload(&f32_bytes(&vec![0.0; ROWS * N_HEAD * HD]));
    decode_rows(
        gpu,
        &batched_out,
        [
            &q_buf,
            &k_cache,
            &v_cache,
            &gpu.upload(&pos_bytes(&positions)),
            &gpu.upload(&packed),
        ],
        ROWS as u32,
        pages as u32,
    );
    let batched = gpu.read_f32(&batched_out);

    // The same two rows alone. Each one's query row is copied to row zero of a
    // one-row buffer, so the lone launch reads the identical bytes the batched
    // one read for that row - anything else would be comparing two different
    // sums.
    for (row, table) in [&identity, &rotated].into_iter().enumerate() {
        let lone_q = q[row * N_HEAD * HD..][..N_HEAD * HD].to_vec();
        let out = gpu.upload(&f32_bytes(&vec![0.0; N_HEAD * HD]));
        decode_rows(
            gpu,
            &out,
            [
                &gpu.upload(&f32_bytes(&lone_q)),
                &k_cache,
                &v_cache,
                &gpu.upload(&pos_bytes(&positions[row..=row])),
                &gpu.upload(&page_bytes(table)),
            ],
            1,
            0,
        );
        let alone = gpu.read_f32(&out);
        for (i, expected) in alone.iter().enumerate() {
            let got = batched[row * N_HEAD * HD + i];
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "row {row} element {i}: batched {got} against {expected} run alone",
            );
        }
    }
}
