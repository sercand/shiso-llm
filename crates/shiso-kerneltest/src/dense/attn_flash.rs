//! Device cases for the simdgroup-matrix prefill softmax,
//! `kernel_dense_attn_flash_hd*`.
//!
//! The same span contract the scalar entries in [`super::attn`] are held to - a
//! sliding window from below, a per-token bound from above, a scale that is an
//! argument, an empty span that writes zeros - at the head widths this entry is
//! instantiated for, and over spans long enough that the online softmax runs
//! across several key blocks. The oracle is the same plain masked softmax.
//!
//! Metal is the only backend that carries the entry: the engine asks a device
//! for it and runs the scalar tile where it is absent, so the wrapper lives in
//! `crates/shiso-metal/tests/dense_attn.rs` alone.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "kernel argument structs take i32/u32 dimensions while the fixtures are written in usize, and every fixture dimension and position here is a small non-negative literal"
)]

use shiso_core::{Extent3, KernelArg};

use super::attn::{
    AttnArgs, BLOCK_256, POISON, Paging, close, f16_bytes, fixture, i32_bytes, key_span, launch,
    page_table, poisoned,
};
use crate::harness::{Gpu, drain, f32_bytes};
use crate::oracle::TOL_QUANT;

/// Every entry point this module's cases launch, with the head width each is
/// instantiated for.
pub const ENTRIES: [(&str, usize); 2] = [
    ("kernel_dense_attn_flash_hd64", 64),
    ("kernel_dense_attn_flash_hd128", 128),
];

/// Query rows one threadgroup covers, `DENSE_FLASH_QROWS`.
const ROWS: usize = 64;
/// Query heads of the fixture.
const Q_HEADS: usize = 8;
/// Key/value heads of the fixture: four query heads read each.
const KV_HEADS: usize = 2;

/// Six 32-row pages handed out out of order, so a span of a hundred and more
/// keys crosses page edges inside one key block as well as between blocks.
const LONG_PAGES: Paging = Paging {
    page_tokens: 32,
    pages: &[3, 0, 5, 1, 4, 2],
};

/// Queries in the long cases: one full tile of 64 and a short one of 19, so a
/// tile's last simdgroups own no query.
const N_TOK: usize = ROWS + 19;

/// The scale the query fixture is multiplied by. Small enough that a span of
/// a hundred keys keeps every key's share of the softmax well above the
/// comparison tolerance, so a skipped key or block shows.
const QUERY_AMPLITUDE: f32 = 1.0 / 64.0;

/// One case: where the queries sit, how far each may read, the window and the
/// score scale.
#[derive(Debug, Clone, Copy)]
struct Case<'a> {
    pos: &'a [i32],
    bound: &'a [i32],
    window: u32,
    scale: f32,
}

/// The masked softmax the entry replaces, at head width `hd`, through the
/// page table: score every key in the span, one softmax, blend the values. An
/// empty span leaves its row at zero.
fn oracle(hd: usize, query: &[f32], keys: &[f32], values: &[f32], case: Case<'_>) -> Vec<f32> {
    let n_tok = case.pos.len();
    let mut out = vec![0.0f32; n_tok * Q_HEADS * hd];
    for tok in 0..n_tok {
        let range = key_span(case.pos[tok] as u32, case.bound[tok] as u32, case.window);
        for head in 0..Q_HEADS {
            let hkv = head / (Q_HEADS / KV_HEADS);
            let qb = (tok * Q_HEADS + head) * hd;
            let scores: Vec<f32> = range
                .clone()
                .map(|at| {
                    let kb = (LONG_PAGES.row(at) * KV_HEADS + hkv) * hd;
                    let dot: f32 = (0..hd).map(|d| query[qb + d] * keys[kb + d]).sum();
                    dot * case.scale
                })
                .collect();
            if scores.is_empty() {
                continue;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let den: f32 = weights.iter().sum();
            for (index, at) in range.clone().enumerate() {
                let vb = (LONG_PAGES.row(at) * KV_HEADS + hkv) * hd;
                for d in 0..hd {
                    out[qb + d] += weights[index] * values[vb + d];
                }
            }
            for d in 0..hd {
                out[qb + d] /= den;
            }
        }
    }
    out
}

/// Runs one entry over a poisoned output and holds every element to the
/// oracle: no NaN, nothing left untouched, every value within tolerance.
fn check(gpu: &Gpu, entry: &'static str, hd: usize, case: Case<'_>) {
    let n_tok = case.pos.len();
    let rows = LONG_PAGES.rows();
    let query: Vec<f32> = fixture(n_tok * Q_HEADS * hd, 1)
        .iter()
        .map(|v| v * QUERY_AMPLITUDE)
        .collect();
    let keys = fixture(rows * KV_HEADS * hd, 2);
    let values = fixture(rows * KV_HEADS * hd, 3);

    let d_q = gpu.upload(&f32_bytes(&query));
    let d_k = gpu.upload(&f16_bytes(&keys));
    let d_v = gpu.upload(&f16_bytes(&values));
    let d_pos = gpu.upload(&i32_bytes(case.pos));
    let d_bound = gpu.upload(&i32_bytes(case.bound));
    let pages = page_table(gpu, LONG_PAGES);
    let out = poisoned(gpu, n_tok * Q_HEADS * hd);

    let args = AttnArgs {
        head_dim: hd as u32,
        n_head: Q_HEADS as u32,
        n_head_kv: KV_HEADS as u32,
        n_tok: n_tok as u32,
        page_tokens: LONG_PAGES.page_tokens,
        window: case.window,
        scale: case.scale,
        page_stride: 0,
    };
    launch(
        gpu,
        entry,
        Extent3::new(Q_HEADS as u32, n_tok.div_ceil(ROWS) as u32, 1),
        Extent3::new(BLOCK_256, 1, 1),
        0,
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
    let what = format!(
        "{entry} {n_tok} queries window {} scale {}",
        case.window, case.scale
    );
    for (index, &have) in got.iter().enumerate() {
        assert!(!have.is_nan(), "{what}: element {index} is NaN");
        assert!(have != POISON, "{what}: element {index} is untouched");
    }
    close(
        &got,
        &oracle(hd, &query, &keys, &values, case),
        TOL_QUANT,
        &what,
    );
}

/// Causal spans of a hundred keys and more, with and without a window, at a
/// scale of one and at the head's reciprocal root.
///
/// Positions 100..183 against bounds at their own position read up to 184
/// keys: several key blocks per query, so the running maximum and denominator
/// are rescaled block over block, through six scattered pages. A window of 40
/// moves every query's low end separately inside one tile.
pub fn long_causal_spans_match_the_oracle(gpu: &Gpu) {
    let pos: Vec<i32> = (100..100 + N_TOK as i32).collect();
    for (entry, hd) in ENTRIES {
        let root = 1.0 / (hd as f32).sqrt();
        for (window, scale) in [(0, 1.0), (0, root), (40, root)] {
            check(
                gpu,
                entry,
                hd,
                Case {
                    pos: &pos,
                    bound: &pos,
                    window,
                    scale,
                },
            );
        }
    }
}

/// A bound past the query's own position, as a bidirectional prefix carries,
/// with a scale small enough that the span is nearly flat.
pub fn a_forward_bound_reads_past_the_query(gpu: &Gpu) {
    let pos: Vec<i32> = (0..N_TOK as i32).collect();
    let bound = vec![150i32; N_TOK];
    for (entry, hd) in ENTRIES {
        check(
            gpu,
            entry,
            hd,
            Case {
                pos: &pos,
                bound: &bound,
                window: 0,
                scale: 0.01,
            },
        );
    }
}

/// A tile where every other query's span is empty - a bound below its
/// window's low end - beside queries whose spans are not: the empty rows write
/// zeros, and their -inf maxima do not reach the rows beside them.
pub fn empty_spans_write_zeros_beside_live_ones(gpu: &Gpu) {
    let pos: Vec<i32> = (0..N_TOK).map(|i| 100 + i as i32).collect();
    let bound: Vec<i32> = pos
        .iter()
        .enumerate()
        .map(|(i, &p)| if i % 2 == 0 { 20 } else { p })
        .collect();
    for (entry, hd) in ENTRIES {
        check(
            gpu,
            entry,
            hd,
            Case {
                pos: &pos,
                bound: &bound,
                window: 16,
                scale: 1.0 / (hd as f32).sqrt(),
            },
        );
    }
}
