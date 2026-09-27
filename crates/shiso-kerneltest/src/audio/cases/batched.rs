//! A batch of independent causal-decoder sequences in one launch, each entry
//! against the single-sequence entry it batches, run on each sequence alone:
//! a batched launch must give that entry's values exactly, so a decoder may
//! batch its sequences and keep its ids. The single entries' own cases hold
//! them to their oracles; the attention cases check the oracle here as well.
//!
//! The sequences' caches share one key and one value buffer, each sequence's
//! `[kv_heads][cap][hd]` plane starting at its own row with its own capacity,
//! as a table of eight words per sequence names them. Every row between the
//! planes, and every stacked row no sequence owns, is a sentinel no launch
//! may read or write.
#![expect(
    clippy::too_many_lines,
    reason = "a batched case lays out its sequences, launches once and checks each sequence against its oracle and the single entry, and reads best in one piece"
)]

use shiso_core::{DeviceBuffer, Extent3, KernelArg};

use super::fused::{QkRopeArgs, same};
use super::{
    ATTN_QUERY_BLOCK, CacheAttnArgs, CacheAttnOracle, DEC_COUNT, DEC_DONE, DEC_POS, DEC_STOP,
    DEC_STOP_CAP, DEC_STOP_EOS, DEC_TOKEN, DECODE_GROUP_TILE, DECODE_MERGE_THREADS,
    DECODE_SPLITS_PER_TG, EmbedTokenArgs, LINEAR_BLOCK, affine8_fixture, affine8_value, blocks,
    cache_attn_oracle, cache_plane, close, decode_batch, decode_state, launch, read_u32, u32_bytes,
    weight_fixture,
};
use crate::{Gpu, drain, f32_bytes, oracle};

/// `audio_args_batch_attn`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchAttnArgs {
    seqs: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    scale: f32,
    splits: u32,
}
const _: () = assert!(size_of::<BatchAttnArgs>() == 24);

/// `audio_args_qk_rope_seq`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QkRopeSeqArgs {
    rows: u32,
    seqs: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    eps: f32,
    theta: f32,
}
const _: () = assert!(size_of::<QkRopeSeqArgs>() == 28);

/// `audio_args_decode_step_batch`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DecodeStepBatchArgs {
    eos0: u32,
    eos1: u32,
    max_tokens: u32,
    advance: u32,
    slot: u32,
    seqs: u32,
}
const _: () = assert!(size_of::<DecodeStepBatchArgs>() == 24);

/// Words of one sequence's table entry, and of one decoder state.
const SEQ_WORDS: usize = 8;
const STATE_WORDS: usize = 8;
/// Threads of one batched-step threadgroup.
const STEP_BLOCK: u32 = 32;
/// The end ids [`decode_batch`] scripts.
const EOS: [u32; 2] = [151_643, 151_645];

/// One sequence of a batch: its cache plane's first row and capacity, and
/// its stacked query rows `[row0, row0 + rows)` at positions from `pos0`.
#[derive(Debug, Clone, Copy)]
struct Span {
    plane: usize,
    cap: usize,
    row0: usize,
    rows: usize,
    pos0: usize,
}

fn seq_table(gpu: &Gpu, spans: &[Span]) -> DeviceBuffer {
    let mut words = vec![0u32; spans.len() * SEQ_WORDS];
    for (s, sp) in spans.iter().enumerate() {
        let e = &mut words[s * SEQ_WORDS..(s + 1) * SEQ_WORDS];
        e[..5].copy_from_slice(&[
            sp.plane as u32,
            sp.cap as u32,
            sp.row0 as u32,
            sp.rows as u32,
            sp.pos0 as u32,
        ]);
    }
    gpu.upload(&u32_bytes(&words))
}

/// Lays the sequences' planes out from row 3 with five sentinel rows after
/// each; returns the spans' plane rows and the whole buffer's rows.
fn lay_planes(caps: &[usize], kv_heads: usize) -> (Vec<usize>, usize) {
    let mut at = 3;
    let mut planes = Vec::new();
    for &cap in caps {
        planes.push(at);
        at += kv_heads * cap + 5;
    }
    (planes, at)
}

/// The shared buffer: `fill` everywhere, each plane at its row.
fn stack_planes(
    planes: &[Vec<f32>],
    spans: &[Span],
    total_rows: usize,
    hd: usize,
    fill: f32,
) -> Vec<f32> {
    let mut all = vec![fill; total_rows * hd];
    for (p, sp) in planes.iter().zip(spans) {
        all[sp.plane * hd..sp.plane * hd + p.len()].copy_from_slice(p);
    }
    all
}

/// Sequence `sp`'s plane out of the shared buffer.
fn plane_of<'a>(all: &'a [f32], sp: &Span, kv_heads: usize, hd: usize) -> &'a [f32] {
    &all[sp.plane * hd..(sp.plane + kv_heads * sp.cap) * hd]
}

/// Every value outside the planes is still the sentinel.
fn outside_planes_untouched(all: &[f32], spans: &[Span], kv_heads: usize, hd: usize, what: &str) {
    let mut owned = vec![false; all.len() / hd];
    for sp in spans {
        owned[sp.plane..sp.plane + kv_heads * sp.cap].fill(true);
    }
    for (r, &mine) in owned.iter().enumerate() {
        if !mine {
            assert!(
                all[r * hd..(r + 1) * hd].iter().all(|v| v.is_nan()),
                "{what}: row {r} between the planes was written"
            );
        }
    }
}

/// The single-sequence causal attention over one plane of its own.
struct Single<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    n_q: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    pos0: usize,
    cap: usize,
    scale: f32,
}

fn single_causal(gpu: &Gpu, s: &Single<'_>) -> Vec<f32> {
    let (qb, kb, vb) = (
        gpu.upload(&f32_bytes(s.q)),
        gpu.upload(&f32_bytes(s.k)),
        gpu.upload(&f32_bytes(s.v)),
    );
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; s.n_q * s.heads * s.hd]));
    let args = CacheAttnArgs {
        n_q: s.n_q as u32,
        heads: s.heads as u32,
        kv_heads: s.kv_heads as u32,
        head_dim: s.hd as u32,
        pos0: s.pos0 as u32,
        cap: s.cap as u32,
        scale: s.scale,
        splits: 0,
    };
    launch(
        gpu,
        "kernel_audio_causal_attn_f32",
        Extent3::new(s.heads as u32, blocks(s.n_q, ATTN_QUERY_BLOCK), 1),
        Extent3::linear(32),
        ATTN_QUERY_BLOCK * s.hd as u32 * 4,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(qb.view()),
            KernelArg::Buffer(kb.view()),
            KernelArg::Buffer(vb.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// Segmented prefill attention: three sequences of 13, 1 and 40 stacked
/// query rows, at positions from 0, 5 and 37, each over its own plane of its
/// own capacity in one shared buffer. Each sequence's rows are its oracle's
/// and, bit for bit, the single-sequence entry's over that plane alone; the
/// stacked row between two sequences is never written, and the rows past each
/// sequence's last position (NaN) are never read.
///
/// # Panics
///
/// When an output differs from the oracle by more than 2e-5, from the single
/// entry at all, or a row no sequence owns is written.
pub fn segmented_causal_attention_matches_each_sequence_alone(gpu: &Gpu) {
    let (heads, kv_heads, hd) = (4usize, 2usize, 128usize);
    let w = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    // (rows, pos0, cap)
    let seqs = [(13usize, 0usize, 20usize), (1, 5, 9), (40, 37, 90)];
    let caps: Vec<usize> = seqs.iter().map(|s| s.2).collect();
    let (planes, total) = lay_planes(&caps, kv_heads);
    // Stacked query rows from row 1, one unowned row between sequences.
    let mut spans = Vec::new();
    let mut row = 1;
    for (&(rows, pos0, cap), &plane) in seqs.iter().zip(&planes) {
        spans.push(Span {
            plane,
            cap,
            row0: row,
            rows,
            pos0,
        });
        row += rows + 1;
    }
    let n_rows = row;
    let mut q = vec![f32::NAN; n_rows * w];
    let (mut ks, mut vs) = (Vec::new(), Vec::new());
    for (s, sp) in spans.iter().enumerate() {
        let qs = oracle::signal(sp.rows * w, 401 + s as u32);
        q[sp.row0 * w..(sp.row0 + sp.rows) * w].copy_from_slice(&qs);
        let valid = sp.pos0 + sp.rows;
        ks.push(cache_plane(kv_heads, sp.cap, hd, valid, 411 + s as u32));
        vs.push(cache_plane(kv_heads, sp.cap, hd, valid, 421 + s as u32));
    }
    let k_all = stack_planes(&ks, &spans, total, hd, f32::NAN);
    let v_all = stack_planes(&vs, &spans, total, hd, f32::NAN);
    let (qb, kb, vb) = (
        gpu.upload(&f32_bytes(&q)),
        gpu.upload(&f32_bytes(&k_all)),
        gpu.upload(&f32_bytes(&v_all)),
    );
    let table = seq_table(gpu, &spans);
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; n_rows * w]));
    let args = BatchAttnArgs {
        seqs: spans.len() as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: hd as u32,
        scale,
        splits: 0,
    };
    let most = spans.iter().map(|s| s.rows).max().unwrap_or(0);
    launch(
        gpu,
        "kernel_audio_causal_attn_seg_f32",
        Extent3::new(
            heads as u32,
            blocks(most, ATTN_QUERY_BLOCK),
            spans.len() as u32,
        ),
        Extent3::linear(32),
        ATTN_QUERY_BLOCK * hd as u32 * 4,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(qb.view()),
            KernelArg::Buffer(kb.view()),
            KernelArg::Buffer(vb.view()),
            KernelArg::Buffer(table.view()),
        ],
    );
    drain(gpu);
    let got = gpu.read_f32(&out);
    let mut owned = vec![false; n_rows];
    for (s, sp) in spans.iter().enumerate() {
        owned[sp.row0..sp.row0 + sp.rows].fill(true);
        let mine = &got[sp.row0 * w..(sp.row0 + sp.rows) * w];
        let qs = &q[sp.row0 * w..(sp.row0 + sp.rows) * w];
        let one = Single {
            q: qs,
            k: &ks[s],
            v: &vs[s],
            n_q: sp.rows,
            heads,
            kv_heads,
            hd,
            pos0: sp.pos0,
            cap: sp.cap,
            scale,
        };
        let want = cache_attn_oracle(&CacheAttnOracle {
            q: qs,
            k: &ks[s],
            v: &vs[s],
            n_q: sp.rows,
            heads,
            kv_heads,
            hd,
            pos0: sp.pos0,
            cap: sp.cap,
            scale,
        });
        let what = format!("segmented attention, sequence {s}");
        close(mine, &want, 2e-5, &what);
        same(mine, &single_causal(gpu, &one), &what);
    }
    for (r, &mine) in owned.iter().enumerate() {
        if !mine {
            assert!(
                got[r * w..(r + 1) * w].iter().all(|v| v.is_nan()),
                "segmented attention wrote stacked row {r}, which no sequence owns"
            );
        }
    }
}

/// Decode attention's two launches over one sequence's plane and state, the
/// single-sequence path.
fn single_decode(
    gpu: &Gpu,
    (q, k, v): (&[f32], &[f32], &[f32]),
    (heads, kv_heads, hd, cap): (usize, usize, usize, usize),
    state: &DeviceBuffer,
    splits: usize,
) -> Vec<f32> {
    let scale = 1.0 / (hd as f32).sqrt();
    let (qb, kb, vb) = (
        gpu.upload(&f32_bytes(q)),
        gpu.upload(&f32_bytes(k)),
        gpu.upload(&f32_bytes(v)),
    );
    let partials = gpu.upload(&f32_bytes(&vec![f32::NAN; heads * splits * (hd + 2)]));
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; heads * hd]));
    let args = CacheAttnArgs {
        n_q: 1,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: hd as u32,
        pos0: 0,
        cap: cap as u32,
        scale,
        splits: splits as u32,
    };
    launch(
        gpu,
        "kernel_audio_decode_attn_f32",
        Extent3::new(
            kv_heads as u32,
            blocks(splits, DECODE_SPLITS_PER_TG),
            blocks(heads / kv_heads, DECODE_GROUP_TILE),
        ),
        Extent3::linear(32 * DECODE_SPLITS_PER_TG),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(partials.view()),
            KernelArg::Buffer(qb.view()),
            KernelArg::Buffer(kb.view()),
            KernelArg::Buffer(vb.view()),
            KernelArg::Buffer(state.view()),
        ],
    );
    launch(
        gpu,
        "kernel_audio_decode_attn_merge_f32",
        Extent3::linear(heads as u32),
        Extent3::linear(DECODE_MERGE_THREADS),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(partials.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// Batched decode attention: five sequences of 1, 7, 600 and 1038 cached rows
/// and one finished (its state done, its plane all NaN), each length read
/// from its own state and each over its own plane of its own capacity. Each
/// live sequence's row is its oracle's and, bit for bit, the single-sequence
/// launches'; the finished one sees no rows and answers zeros. The merge is
/// the single-sequence merge over the batch's heads.
///
/// # Panics
///
/// When an output differs from the oracle by more than 2e-5, from the single
/// launches at all, or the finished row is not zeros.
pub fn batched_decode_attention_reads_each_length(gpu: &Gpu) {
    for (heads, kv_heads, hd) in [(16usize, 8usize, 128usize), (3, 3, 64)] {
        batched_decode_case(gpu, heads, kv_heads, hd);
    }
}

fn batched_decode_case(gpu: &Gpu, heads: usize, kv_heads: usize, hd: usize) {
    let w = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    // (length, cap); length 0 is the finished sequence.
    let seqs = [
        (1usize, 4usize),
        (7, 16),
        (600, 640),
        (0, 320),
        (1038, 1100),
    ];
    let b = seqs.len();
    let caps: Vec<usize> = seqs.iter().map(|s| s.1).collect();
    let (planes, total) = lay_planes(&caps, kv_heads);
    let spans: Vec<Span> = seqs
        .iter()
        .zip(&planes)
        .map(|(&(_, cap), &plane)| Span {
            plane,
            cap,
            row0: 0,
            rows: 0,
            pos0: 0,
        })
        .collect();
    let q = oracle::signal(b * w, 431);
    let (mut ks, mut vs, mut words) = (Vec::new(), Vec::new(), Vec::new());
    for (s, &(len, cap)) in seqs.iter().enumerate() {
        ks.push(cache_plane(kv_heads, cap, hd, len, 441 + s as u32));
        vs.push(cache_plane(kv_heads, cap, hd, len, 451 + s as u32));
        let mut st = [0u32; STATE_WORDS];
        if len == 0 {
            st[DEC_POS] = 300;
            st[DEC_DONE] = 1;
            st[DEC_STOP] = DEC_STOP_EOS;
        } else {
            st[DEC_POS] = (len - 1) as u32;
        }
        words.extend_from_slice(&st);
    }
    let states = gpu.upload(&u32_bytes(&words));
    let k_all = stack_planes(&ks, &spans, total, hd, f32::NAN);
    let v_all = stack_planes(&vs, &spans, total, hd, f32::NAN);
    let (qb, kb, vb) = (
        gpu.upload(&f32_bytes(&q)),
        gpu.upload(&f32_bytes(&k_all)),
        gpu.upload(&f32_bytes(&v_all)),
    );
    let table = seq_table(gpu, &spans);
    for splits in [5usize, 32] {
        let partials = gpu.upload(&f32_bytes(&vec![f32::NAN; b * heads * splits * (hd + 2)]));
        let out = gpu.upload(&f32_bytes(&vec![f32::NAN; b * w]));
        let args = BatchAttnArgs {
            seqs: b as u32,
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            head_dim: hd as u32,
            scale,
            splits: splits as u32,
        };
        launch(
            gpu,
            "kernel_audio_decode_attn_batch_f32",
            Extent3::new(
                kv_heads as u32,
                blocks(splits, DECODE_SPLITS_PER_TG),
                blocks(heads / kv_heads, DECODE_GROUP_TILE) * b as u32,
            ),
            Extent3::linear(32 * DECODE_SPLITS_PER_TG),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(partials.view()),
                KernelArg::Buffer(qb.view()),
                KernelArg::Buffer(kb.view()),
                KernelArg::Buffer(vb.view()),
                KernelArg::Buffer(states.view()),
                KernelArg::Buffer(table.view()),
            ],
        );
        // The merge closes the batch as one sequence of `b * heads` heads.
        let merge = CacheAttnArgs {
            n_q: 1,
            heads: (b * heads) as u32,
            kv_heads: kv_heads as u32,
            head_dim: hd as u32,
            pos0: 0,
            cap: 0,
            scale,
            splits: splits as u32,
        };
        launch(
            gpu,
            "kernel_audio_decode_attn_merge_f32",
            Extent3::linear((b * heads) as u32),
            Extent3::linear(DECODE_MERGE_THREADS),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&merge)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(partials.view()),
            ],
        );
        drain(gpu);
        let got = gpu.read_f32(&out);
        for (s, &(len, cap)) in seqs.iter().enumerate() {
            let mine = &got[s * w..(s + 1) * w];
            let what = format!(
                "batched decode attention, {heads}/{kv_heads} heads of {hd}, sequence {s} of {len} rows, {splits} splits"
            );
            if len == 0 {
                close(mine, &vec![0.0; w], 0.0, &what);
                continue;
            }
            let qs = &q[s * w..(s + 1) * w];
            let want = cache_attn_oracle(&CacheAttnOracle {
                q: qs,
                k: &ks[s],
                v: &vs[s],
                n_q: 1,
                heads,
                kv_heads,
                hd,
                pos0: len - 1,
                cap,
                scale,
            });
            close(mine, &want, 2e-5, &what);
            let state = decode_state(gpu, (len - 1) as u32, 0);
            let single = single_decode(
                gpu,
                (qs, &ks[s], &vs[s]),
                (heads, kv_heads, hd, cap),
                &state,
                splits,
            );
            same(mine, &single, &what);
        }
    }
}

/// The batched step over five sequences, each scripted as a case of the
/// single step's: one stopping on either end id, one at the token cap, one
/// still running and one done before the batch. Each sequence's slots and
/// state are the single step's run on that sequence alone, with no read
/// between the steps, over two batches.
///
/// # Panics
///
/// When a slot or a state word differs from the single step's.
pub fn batched_step_stops_each_sequence_alone(gpu: &Gpu) {
    let none = u32::MAX;
    let max_tokens = 4u32;
    // (pos, token, done) and each step's argmax.
    let starts: [(u32, u32, bool); 5] = [
        (10, 0, false),
        (3, 42, false),
        (0, 42, false),
        (7, 5, true),
        (0, 1, false),
    ];
    let script: [[u32; 5]; 5] = [
        [5, 6, 151_645, 7, 8],
        [9, 151_643, 1, 2, 3],
        [1, 2, 3, 4, 5],
        [4, 4, 4, 4, 4],
        [11, 12, 13, 14, 15],
    ];
    let b = starts.len();
    let steps = script[0].len();
    let mut words = Vec::new();
    for &(pos, token, done) in &starts {
        let mut st = [0u32; STATE_WORDS];
        st[DEC_POS] = pos;
        st[DEC_TOKEN] = token;
        if done {
            st[DEC_DONE] = 1;
            st[DEC_STOP] = DEC_STOP_EOS;
        }
        words.extend_from_slice(&st);
    }
    let states = gpu.upload(&u32_bytes(&words));
    // Two batches over the same states: the first steps from the prefill's
    // argmax (advance 0), the second carries on.
    let halves = [(0usize, 3usize), (3, steps)];
    let mut got_out = vec![Vec::new(); b];
    for (h, &(from, to)) in halves.iter().enumerate() {
        let n = to - from;
        let out = gpu.upload(&u32_bytes(&vec![7777; n * b]));
        let ids: Vec<DeviceBuffer> = (from..to)
            .map(|t| {
                let row: Vec<u32> = script.iter().map(|s| s[t]).collect();
                gpu.upload(&u32_bytes(&row))
            })
            .collect();
        for (slot, idb) in ids.iter().enumerate() {
            let args = DecodeStepBatchArgs {
                eos0: EOS[0],
                eos1: EOS[1],
                max_tokens,
                advance: u32::from(!(h == 0 && slot == 0)),
                slot: slot as u32,
                seqs: b as u32,
            };
            launch(
                gpu,
                "kernel_audio_decode_step_batch_u32",
                Extent3::linear(blocks(b, STEP_BLOCK)),
                Extent3::linear(STEP_BLOCK),
                0,
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(states.view()),
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(idb.view()),
                ],
            );
        }
        drain(gpu);
        let slots = read_u32(gpu, &out);
        for (s, col) in got_out.iter_mut().enumerate() {
            col.extend((0..n).map(|j| slots[j * b + s]));
        }
    }
    let got_states = read_u32(gpu, &states);
    for (s, &(pos, token, done)) in starts.iter().enumerate() {
        let st = if done {
            gpu.upload(&u32_bytes(&[pos, 0, 1, token, 0, DEC_STOP_EOS, 0, 0]))
        } else {
            decode_state(gpu, pos, token)
        };
        let (a, _) = decode_batch(gpu, &st, &script[s][..3], max_tokens, 0);
        let (c, alone) = decode_batch(gpu, &st, &script[s][3..], max_tokens, 1);
        let want: Vec<u32> = a.into_iter().chain(c).collect();
        assert_eq!(got_out[s], want, "sequence {s}: slots");
        assert_eq!(
            &got_states[s * STATE_WORDS..(s + 1) * STATE_WORDS],
            alone.as_slice(),
            "sequence {s}: state"
        );
    }
    // The rules themselves, as the single step's case states them.
    assert_eq!(got_out[0], vec![5, 6, none, none, none]);
    assert_eq!(got_out[1], vec![9, none, none, none, none]);
    assert_eq!(got_out[2], vec![1, 2, 3, 4, none]);
    assert_eq!(got_out[3], vec![none; 5]);
    let st = |s: usize, w: usize| got_states[s * STATE_WORDS + w];
    assert_eq!((st(2, DEC_DONE), st(2, DEC_STOP)), (1, DEC_STOP_CAP));
    assert_eq!((st(3, DEC_POS), st(3, DEC_COUNT)), (7, 0), "done stays put");
    assert_eq!((st(4, DEC_COUNT), st(4, DEC_TOKEN)), (4, 14));
}

/// The single fused norm, rotation and append over `rows` rows of one
/// sequence at `pos`, on a plane of its own; returns the query rows and the
/// two planes.
fn single_qk(
    gpu: &Gpu,
    args: QkRopeArgs,
    (q, k, v): (&[f32], &[f32], &[f32]),
    (qn, kn): (&DeviceBuffer, &DeviceBuffer),
    (kp, vp): (&[f32], &[f32]),
    pos: u32,
) -> [Vec<f32>; 3] {
    let (qb, kb, vb) = (
        gpu.upload(&f32_bytes(q)),
        gpu.upload(&f32_bytes(k)),
        gpu.upload(&f32_bytes(v)),
    );
    let (kc, vc) = (gpu.upload(&f32_bytes(kp)), gpu.upload(&f32_bytes(vp)));
    let state = decode_state(gpu, pos, 0);
    launch(
        gpu,
        "kernel_audio_qk_norm_rope_append_f32",
        Extent3::new(args.heads + args.kv_heads, args.rows, 1),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(qb.view()),
            KernelArg::Buffer(kb.view()),
            KernelArg::Buffer(vb.view()),
            KernelArg::Buffer(qn.view()),
            KernelArg::Buffer(kn.view()),
            KernelArg::Buffer(kc.view()),
            KernelArg::Buffer(vc.view()),
            KernelArg::Buffer(state.view()),
        ],
    );
    drain(gpu);
    [gpu.read_f32(&qb), gpu.read_f32(&kc), gpu.read_f32(&vc)]
}

/// Geometry shared by the two norm-rotation-append cases.
const QK_HEADS: usize = 16;
const QK_KV_HEADS: usize = 8;
const QK_HD: usize = 128;
const QK_THETA: f32 = 1.0e6;

/// Stacked inputs of the norm-rotation-append cases, and the planes' buffers.
struct QkBatch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    ks: Vec<Vec<f32>>,
    vs: Vec<Vec<f32>>,
    k_all: DeviceBuffer,
    v_all: DeviceBuffer,
    qb: DeviceBuffer,
    qn: DeviceBuffer,
    kn: DeviceBuffer,
}

fn qk_batch(gpu: &Gpu, rows: usize, spans: &[Span], total: usize, seed: u32) -> QkBatch {
    let (qw, kw) = (QK_HEADS * QK_HD, QK_KV_HEADS * QK_HD);
    let q = oracle::signal(rows * qw, seed);
    let k = oracle::signal(rows * kw, seed + 1);
    let v = oracle::signal(rows * kw, seed + 2);
    let ks: Vec<Vec<f32>> = spans
        .iter()
        .enumerate()
        .map(|(s, sp)| oracle::signal(QK_KV_HEADS * sp.cap * QK_HD, seed + 10 + s as u32))
        .collect();
    let vs: Vec<Vec<f32>> = spans
        .iter()
        .enumerate()
        .map(|(s, sp)| oracle::signal(QK_KV_HEADS * sp.cap * QK_HD, seed + 20 + s as u32))
        .collect();
    let k_all = gpu.upload(&f32_bytes(&stack_planes(
        &ks,
        spans,
        total,
        QK_HD,
        f32::NAN,
    )));
    let v_all = gpu.upload(&f32_bytes(&stack_planes(
        &vs,
        spans,
        total,
        QK_HD,
        f32::NAN,
    )));
    QkBatch {
        qb: gpu.upload(&f32_bytes(&q)),
        qn: gpu.upload(&f32_bytes(&oracle::signal(QK_HD, seed + 3))),
        kn: gpu.upload(&f32_bytes(&oracle::signal(QK_HD, seed + 4))),
        q,
        k,
        v,
        ks,
        vs,
        k_all,
        v_all,
    }
}

fn qk_args(rows: usize) -> QkRopeArgs {
    QkRopeArgs {
        rows: rows as u32,
        heads: QK_HEADS as u32,
        kv_heads: QK_KV_HEADS as u32,
        head_dim: QK_HD as u32,
        cap: 0,
        eps: 1e-6,
        theta: QK_THETA,
    }
}

/// The batched decode step's norms, rotation and append: row `s` of four is
/// sequence `s`'s, at its own state's position, into its own plane - a
/// position inside the plane, one at its capacity (dropped), one far along,
/// and a finished sequence, which appends nothing. Every row is the single
/// fused entry's on that sequence alone, bit for bit.
///
/// # Panics
///
/// When a value differs from the single entry's, a finished sequence's plane
/// changes, or a row between the planes is written.
pub fn batched_qk_norm_rope_append_is_each_row_alone(gpu: &Gpu) {
    let (qw, kw) = (QK_HEADS * QK_HD, QK_KV_HEADS * QK_HD);
    // (pos, cap, done)
    let seqs = [
        (5u32, 20usize, false),
        (19, 19, false),
        (1000, 1100, false),
        (7, 12, true),
    ];
    let b = seqs.len();
    let caps: Vec<usize> = seqs.iter().map(|s| s.1).collect();
    let (planes, total) = lay_planes(&caps, QK_KV_HEADS);
    let spans: Vec<Span> = seqs
        .iter()
        .zip(&planes)
        .map(|(&(_, cap, _), &plane)| Span {
            plane,
            cap,
            row0: 0,
            rows: 0,
            pos0: 0,
        })
        .collect();
    let fx = qk_batch(gpu, b, &spans, total, 461);
    let mut words = Vec::new();
    for &(pos, _, done) in &seqs {
        words.extend_from_slice(&[pos, 0, u32::from(done), 0, 0, 0, 0, 0]);
    }
    let states = gpu.upload(&u32_bytes(&words));
    let table = seq_table(gpu, &spans);
    let kb = gpu.upload(&f32_bytes(&fx.k));
    let vb = gpu.upload(&f32_bytes(&fx.v));
    let args = QkRopeSeqArgs {
        rows: b as u32,
        seqs: b as u32,
        heads: QK_HEADS as u32,
        kv_heads: QK_KV_HEADS as u32,
        head_dim: QK_HD as u32,
        eps: 1e-6,
        theta: QK_THETA,
    };
    launch(
        gpu,
        "kernel_audio_qk_norm_rope_append_batch_f32",
        Extent3::new((QK_HEADS + QK_KV_HEADS) as u32, b as u32, 1),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(fx.qb.view()),
            KernelArg::Buffer(kb.view()),
            KernelArg::Buffer(vb.view()),
            KernelArg::Buffer(fx.qn.view()),
            KernelArg::Buffer(fx.kn.view()),
            KernelArg::Buffer(fx.k_all.view()),
            KernelArg::Buffer(fx.v_all.view()),
            KernelArg::Buffer(states.view()),
            KernelArg::Buffer(table.view()),
        ],
    );
    drain(gpu);
    let (q_got, k_got, v_got) = (
        gpu.read_f32(&fx.qb),
        gpu.read_f32(&fx.k_all),
        gpu.read_f32(&fx.v_all),
    );
    for (s, (&(pos, cap, done), sp)) in seqs.iter().zip(&spans).enumerate() {
        let mut one = qk_args(1);
        one.cap = cap as u32;
        let [q1, k1, v1] = single_qk(
            gpu,
            one,
            (
                &fx.q[s * qw..(s + 1) * qw],
                &fx.k[s * kw..(s + 1) * kw],
                &fx.v[s * kw..(s + 1) * kw],
            ),
            (&fx.qn, &fx.kn),
            (&fx.ks[s], &fx.vs[s]),
            pos,
        );
        let what = format!("batched norm-rotate-append, sequence {s} at {pos}");
        same(&q_got[s * qw..(s + 1) * qw], &q1, &format!("{what}: q"));
        let (kw_want, vw_want) = if done {
            (fx.ks[s].clone(), fx.vs[s].clone())
        } else {
            (k1, v1)
        };
        same(
            plane_of(&k_got, sp, QK_KV_HEADS, QK_HD),
            &kw_want,
            &format!("{what}: k plane"),
        );
        same(
            plane_of(&v_got, sp, QK_KV_HEADS, QK_HD),
            &vw_want,
            &format!("{what}: v plane"),
        );
    }
    outside_planes_untouched(&k_got, &spans, QK_KV_HEADS, QK_HD, "batched append, k");
    outside_planes_untouched(&v_got, &spans, QK_KV_HEADS, QK_HD, "batched append, v");
}

/// The segmented prefill's norms, rotation and append: three sequences of 3,
/// 1 and 5 stacked rows from positions 0, 7 and 2, into planes of 8, 8 and 6
/// rows, so the last two rows of the third fall past its capacity and are
/// dropped. Each sequence's rows and plane are the single fused entry's over
/// that sequence alone, bit for bit, and the stacked row between two
/// sequences is left as it was.
///
/// # Panics
///
/// When a value differs from the single entry's, or a row no sequence owns
/// changes.
pub fn segmented_qk_norm_rope_append_is_each_sequence_alone(gpu: &Gpu) {
    let (qw, kw) = (QK_HEADS * QK_HD, QK_KV_HEADS * QK_HD);
    // (rows, pos0, cap)
    let seqs = [(3usize, 0usize, 8usize), (1, 7, 8), (5, 2, 6)];
    let caps: Vec<usize> = seqs.iter().map(|s| s.2).collect();
    let (planes, total) = lay_planes(&caps, QK_KV_HEADS);
    let mut spans = Vec::new();
    let mut row = 0;
    for (&(rows, pos0, cap), &plane) in seqs.iter().zip(&planes) {
        spans.push(Span {
            plane,
            cap,
            row0: row,
            rows,
            pos0,
        });
        row += rows + 1;
    }
    let n_rows = row;
    let fx = qk_batch(gpu, n_rows, &spans, total, 481);
    let table = seq_table(gpu, &spans);
    let kb = gpu.upload(&f32_bytes(&fx.k));
    let vb = gpu.upload(&f32_bytes(&fx.v));
    let args = QkRopeSeqArgs {
        rows: n_rows as u32,
        seqs: spans.len() as u32,
        heads: QK_HEADS as u32,
        kv_heads: QK_KV_HEADS as u32,
        head_dim: QK_HD as u32,
        eps: 1e-6,
        theta: QK_THETA,
    };
    launch(
        gpu,
        "kernel_audio_qk_norm_rope_append_seg_f32",
        Extent3::new((QK_HEADS + QK_KV_HEADS) as u32, n_rows as u32, 1),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(fx.qb.view()),
            KernelArg::Buffer(kb.view()),
            KernelArg::Buffer(vb.view()),
            KernelArg::Buffer(fx.qn.view()),
            KernelArg::Buffer(fx.kn.view()),
            KernelArg::Buffer(fx.k_all.view()),
            KernelArg::Buffer(fx.v_all.view()),
            KernelArg::Buffer(table.view()),
        ],
    );
    drain(gpu);
    let (q_got, k_got, v_got) = (
        gpu.read_f32(&fx.qb),
        gpu.read_f32(&fx.k_all),
        gpu.read_f32(&fx.v_all),
    );
    let mut owned = vec![false; n_rows];
    for (s, sp) in spans.iter().enumerate() {
        owned[sp.row0..sp.row0 + sp.rows].fill(true);
        let (r0, r1) = (sp.row0, sp.row0 + sp.rows);
        let mut one = qk_args(sp.rows);
        one.cap = sp.cap as u32;
        let [q1, k1, v1] = single_qk(
            gpu,
            one,
            (
                &fx.q[r0 * qw..r1 * qw],
                &fx.k[r0 * kw..r1 * kw],
                &fx.v[r0 * kw..r1 * kw],
            ),
            (&fx.qn, &fx.kn),
            (&fx.ks[s], &fx.vs[s]),
            sp.pos0 as u32,
        );
        let what = format!("segmented norm-rotate-append, sequence {s}");
        same(&q_got[r0 * qw..r1 * qw], &q1, &format!("{what}: q"));
        same(
            plane_of(&k_got, sp, QK_KV_HEADS, QK_HD),
            &k1,
            &format!("{what}: k plane"),
        );
        same(
            plane_of(&v_got, sp, QK_KV_HEADS, QK_HD),
            &v1,
            &format!("{what}: v plane"),
        );
    }
    for (r, &mine) in owned.iter().enumerate() {
        if !mine {
            same(
                &q_got[r * qw..(r + 1) * qw],
                &fx.q[r * qw..(r + 1) * qw],
                &format!("segmented norm-rotate-append: unowned stacked row {r}"),
            );
        }
    }
    outside_planes_untouched(&k_got, &spans, QK_KV_HEADS, QK_HD, "segmented append, k");
    outside_planes_untouched(&v_got, &spans, QK_KV_HEADS, QK_HD, "segmented append, v");
}

/// The batched embedding rows: row `s` is sequence `s`'s last token's, read
/// from its own state, for every storage the text embedding comes in; zeros
/// for a token past the table.
///
/// # Panics
///
/// When a value differs from the table's.
pub fn batched_embed_token_takes_each_state_token_row(gpu: &Gpu) {
    let (vocab, width, group) = (9usize, 320usize, 64usize);
    let tokens = [0u32, 7, 9, 3];
    let b = tokens.len();
    let mut words = Vec::new();
    for &t in &tokens {
        words.extend_from_slice(&[0, 0, 0, t, 0, 0, 0, 0]);
    }
    let states = gpu.upload(&u32_bytes(&words));
    let args = EmbedTokenArgs {
        width: width as u32,
        vocab: vocab as u32,
        group: group as u32,
    };
    let grid = Extent3::new(blocks(width, LINEAR_BLOCK), b as u32, 1);
    let want_of = |values: &[f32]| -> Vec<f32> {
        tokens
            .iter()
            .flat_map(|&t| {
                let k = t as usize;
                if k < vocab {
                    values[k * width..(k + 1) * width].to_vec()
                } else {
                    vec![0.0; width]
                }
            })
            .collect()
    };
    let dense: [(&str, &'static str); 4] = [
        ("f32w", "kernel_audio_embed_token_batch_f32w"),
        ("bf16w", "kernel_audio_embed_token_batch_bf16w"),
        ("f16w", "kernel_audio_embed_token_batch_f16w"),
        ("q8_0w", "kernel_audio_embed_token_batch_q8_0w"),
    ];
    for (kind, entry) in dense {
        let (bytes, values) = weight_fixture(kind, vocab * width, 491);
        let tb = gpu.upload(&bytes);
        let out = gpu.upload(&f32_bytes(&vec![f32::NAN; b * width]));
        launch(
            gpu,
            entry,
            grid,
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(tb.view()),
                KernelArg::Buffer(states.view()),
            ],
        );
        drain(gpu);
        close(
            &gpu.read_f32(&out),
            &want_of(&values),
            0.0,
            &format!("batched embed token {kind}"),
        );
    }
    let fx = affine8_fixture(vocab, width, group, 492);
    let values: Vec<f32> = (0..vocab * width)
        .map(|i| affine8_value(&fx, group, i))
        .collect();
    let (wb, sb, bb) = (gpu.upload(&fx[0]), gpu.upload(&fx[1]), gpu.upload(&fx[2]));
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; b * width]));
    launch(
        gpu,
        "kernel_audio_embed_token_batch_affine8w",
        grid,
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(wb.view()),
            KernelArg::Buffer(sb.view()),
            KernelArg::Buffer(bb.view()),
            KernelArg::Buffer(states.view()),
        ],
    );
    drain(gpu);
    close(
        &gpu.read_f32(&out),
        &want_of(&values),
        0.0,
        "batched embed token affine8w",
    );
}
