//! The batched launches through [`Ops`], end to end on the device: a
//! segmented prefill and a run of batched decode steps over three sequences
//! must give each sequence's values bit for bit as the single-sequence
//! launches give them on that sequence alone. That pins each batched
//! descriptor's buffer order against its kernel's, and the table's and the
//! states' layouts between the kernels that share them.
#![expect(
    clippy::too_many_lines,
    reason = "each path runs the prefill and every step of all three sequences in order, and reads best as one sequence of launches"
)]

use std::time::Instant;

use shiso_core::{DeviceBuffer, DeviceView};

use super::load::{download_f32, upload_f32};
use super::{
    BatchPlanes, CachePlanes, CausalAttnDesc, CausalAttnSegDesc, DECODE_ATTN_SPLITS, DECODE_DONE,
    DECODE_POS, DECODE_STATE_WORDS, DecodeAttnBatchDesc, DecodeAttnDesc, DecodeStepBatchDesc,
    DecodeStepDesc, EmbedTable, EmbedTokenBatchDesc, EmbedTokenDesc, GemmType, Ops,
    QkRopeAppendBatchDesc, QkRopeAppendDesc, QkRopeAppendSegDesc, SeqSpan, SeqTable,
    decode_attention_partials,
};

const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HD: usize = 128;
const QW: usize = HEADS * HD;
const KW: usize = KV_HEADS * HD;
/// `1 / sqrt(HD)`.
const SCALE: f32 = 0.088_388_35;
const EPS: f32 = 1e-6;
const THETA: f32 = 1.0e6;
const EOS: [u32; 2] = [7, 8];
/// Each sequence's prompt rows and plane capacity.
const SEQS: [(usize, usize); 3] = [(20, 40), (7, 12), (13, 32)];
/// Decode steps after the prefill.
const STEPS: usize = 3;

#[expect(
    clippy::print_stderr,
    reason = "the timing test reports its measurement on stderr"
)]
fn say(line: &str) {
    eprintln!("{line}");
}

/// Deterministic values in `[-1, 1)`.
fn signal(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 0x9E37_79B9;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[expect(clippy::cast_precision_loss, reason = "a 24-bit value is exact in f32")]
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit * 2.0 - 1.0
        })
        .collect()
}

fn bytes(floats: usize) -> u64 {
    u64::try_from(floats * 4).expect("fits")
}

fn rows(buffer: &DeviceBuffer, first: usize, count: usize, width: usize) -> DeviceView<'_> {
    buffer
        .slice(bytes(first * width), bytes(count * width))
        .expect("in range")
}

fn u32s(ops: &Ops, words: &[u32]) -> DeviceBuffer {
    let floats: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();
    upload_f32(ops, &floats).expect("uploads")
}

fn read(ops: &Ops, view: DeviceView<'_>, floats: usize) -> Vec<u32> {
    download_f32(ops, view, floats)
        .expect("reads")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn word(n: usize) -> u32 {
    u32::try_from(n).expect("fits")
}

/// Bit for bit, or both zero.
fn same(got: &[u32], want: &[u32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g == w || (g << 1 == 0 && w << 1 == 0),
            "{what}[{i}]: batched {:e}, single {:e}",
            f32::from_bits(g),
            f32::from_bits(w)
        );
    }
}

/// Every stacked input of the fixture.
struct Fixture {
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    /// Each step's `[seqs]` rows of q, k and v.
    step_q: DeviceBuffer,
    step_k: DeviceBuffer,
    step_v: DeviceBuffer,
    /// Each step's argmax, `[STEPS][seqs]`; sequence 1 ends on the last.
    ids: Vec<u32>,
    embed: DeviceBuffer,
}

/// Stacked rows of the prompts, back to back.
fn prompt_rows() -> usize {
    SEQS.iter().map(|s| s.0).sum()
}

fn fixture(ops: &Ops) -> Fixture {
    let n = prompt_rows();
    let b = SEQS.len();
    let mut ids = Vec::new();
    for t in 0..STEPS {
        for s in 0..b {
            ids.push(if s == 1 && t == STEPS - 1 {
                EOS[1]
            } else {
                word(100 + 10 * s + t)
            });
        }
    }
    Fixture {
        q: upload_f32(ops, &signal(n * QW, 11)).expect("uploads"),
        k: upload_f32(ops, &signal(n * KW, 12)).expect("uploads"),
        v: upload_f32(ops, &signal(n * KW, 13)).expect("uploads"),
        q_norm: upload_f32(ops, &signal(HD, 14)).expect("uploads"),
        k_norm: upload_f32(ops, &signal(HD, 15)).expect("uploads"),
        step_q: upload_f32(ops, &signal(STEPS * b * QW, 16)).expect("uploads"),
        step_k: upload_f32(ops, &signal(STEPS * b * KW, 17)).expect("uploads"),
        step_v: upload_f32(ops, &signal(STEPS * b * KW, 18)).expect("uploads"),
        ids,
        embed: upload_f32(ops, &signal(256 * QW, 19)).expect("uploads"),
    }
}

/// What a run leaves: the prefill's attention rows, each step's attention
/// rows and embedding rows, the slots, the states and the caches.
#[derive(Default)]
struct Run {
    prefill: Vec<Vec<u32>>,
    steps: Vec<Vec<u32>>,
    embeds: Vec<Vec<u32>>,
    slots: Vec<Vec<u32>>,
    states: Vec<Vec<u32>>,
    caches: Vec<Vec<u32>>,
}

/// Each sequence on its own planes, through the single-sequence launches.
fn single(ops: &Ops, fx: &Fixture) -> Run {
    let b = SEQS.len();
    let mut run = Run::default();
    let mut row0 = 0;
    for (s, &(n, cap)) in SEQS.iter().enumerate() {
        let (kc, vc) = (
            ops.scratch(KV_HEADS * cap * HD).expect("allocates"),
            ops.scratch(KV_HEADS * cap * HD).expect("allocates"),
        );
        let cache = CachePlanes {
            k: kc.view(),
            v: vc.view(),
            kv_heads: KV_HEADS,
            cap,
            head_dim: HD,
        };
        let st = ops.scratch(DECODE_STATE_WORDS).expect("allocates");
        let q = ops.scratch(n * QW).expect("allocates");
        ops.backend()
            .copy_d2d(q.view(), rows(&fx.q, row0, n, QW))
            .expect("copies");
        ops.qk_norm_rope_append(&QkRopeAppendDesc {
            q: q.view(),
            k: rows(&fx.k, row0, n, KW),
            v: rows(&fx.v, row0, n, KW),
            q_norm: fx.q_norm.view(),
            k_norm: fx.k_norm.view(),
            cache,
            heads: HEADS,
            rows: n,
            eps: EPS,
            theta: THETA,
            state: st.view(),
        })
        .expect("appends");
        let att = ops.scratch(n * QW).expect("allocates");
        ops.causal_attention(&CausalAttnDesc {
            out: att.view(),
            q: q.view(),
            cache,
            n_q: n,
            heads: HEADS,
            pos0: 0,
            scale: SCALE,
        })
        .expect("attends");
        run.prefill.push(read(ops, att.view(), n * QW));
        ops.backend()
            .copy_h2d(st.slice(0, 4).expect("a word"), &word(n).to_le_bytes())
            .expect("hands over");
        let partials = ops
            .scratch(decode_attention_partials(HEADS, HD, DECODE_ATTN_SPLITS))
            .expect("allocates");
        let out = ops.scratch(STEPS).expect("allocates");
        let (mut steps, mut embeds) = (Vec::new(), Vec::new());
        for t in 0..STEPS {
            let at = t * b + s;
            let qs = ops.scratch(QW).expect("allocates");
            ops.backend()
                .copy_d2d(qs.view(), rows(&fx.step_q, at, 1, QW))
                .expect("copies");
            ops.qk_norm_rope_append(&QkRopeAppendDesc {
                q: qs.view(),
                k: rows(&fx.step_k, at, 1, KW),
                v: rows(&fx.step_v, at, 1, KW),
                q_norm: fx.q_norm.view(),
                k_norm: fx.k_norm.view(),
                cache,
                heads: HEADS,
                rows: 1,
                eps: EPS,
                theta: THETA,
                state: st.view(),
            })
            .expect("appends");
            let att = ops.scratch(QW).expect("allocates");
            ops.decode_attention(&DecodeAttnDesc {
                out: att.view(),
                partials: partials.view(),
                splits: DECODE_ATTN_SPLITS,
                q: qs.view(),
                cache,
                heads: HEADS,
                state: st.view(),
                scale: SCALE,
            })
            .expect("attends");
            let id = u32s(ops, &[fx.ids[at]]);
            ops.decode_step(&DecodeStepDesc {
                state: st.view(),
                out: out.view(),
                ids: id.view(),
                eos: EOS,
                max_tokens: 100,
                advance: 1,
                slot: t,
            })
            .expect("steps");
            let emb = ops.scratch(QW).expect("allocates");
            ops.embed_token(&EmbedTokenDesc {
                out: emb.view(),
                table: EmbedTable::Dense {
                    dtype: GemmType::F32,
                    rows: fx.embed.view(),
                },
                state: st.view(),
                width: QW,
                vocab: 256,
            })
            .expect("fetches");
            steps.push(read(ops, att.view(), QW));
            embeds.push(read(ops, emb.view(), QW));
        }
        run.steps.push(steps.concat());
        run.embeds.push(embeds.concat());
        run.slots.push(read(ops, out.view(), STEPS));
        run.states.push(read(ops, st.view(), DECODE_STATE_WORDS));
        let plane = KV_HEADS * cap * HD;
        run.caches
            .push([read(ops, kc.view(), plane), read(ops, vc.view(), plane)].concat());
        row0 += n;
    }
    run
}

/// The same sequences through the batched launches, their planes in shared
/// buffers with a gap between each.
fn batched(ops: &Ops, fx: &Fixture) -> Run {
    let b = SEQS.len();
    let n = prompt_rows();
    let (mut spans, mut plane, mut row0) = (Vec::new(), 2, 0);
    for &(rows, cap) in &SEQS {
        spans.push(SeqSpan {
            plane,
            cap,
            row0,
            rows,
            pos0: 0,
        });
        plane += KV_HEADS * cap + 3;
        row0 += rows;
    }
    let table = SeqTable::new(ops, &spans, KV_HEADS).expect("uploads");
    let (kc, vc) = (
        ops.scratch(plane * HD).expect("allocates"),
        ops.scratch(plane * HD).expect("allocates"),
    );
    let cache = BatchPlanes {
        k: kc.view(),
        v: vc.view(),
        table: &table,
        kv_heads: KV_HEADS,
        head_dim: HD,
    };
    let q = ops.scratch(n * QW).expect("allocates");
    ops.backend()
        .copy_d2d(q.view(), fx.q.view())
        .expect("copies");
    ops.qk_norm_rope_append_seg(&QkRopeAppendSegDesc {
        q: q.view(),
        k: fx.k.view(),
        v: fx.v.view(),
        q_norm: fx.q_norm.view(),
        k_norm: fx.k_norm.view(),
        cache,
        heads: HEADS,
        rows: n,
        eps: EPS,
        theta: THETA,
    })
    .expect("appends");
    let att = ops.scratch(n * QW).expect("allocates");
    ops.causal_attention_seg(&CausalAttnSegDesc {
        out: att.view(),
        q: q.view(),
        cache,
        heads: HEADS,
        scale: SCALE,
    })
    .expect("attends");
    let prefill = read(ops, att.view(), n * QW);
    let mut words = vec![0u32; b * DECODE_STATE_WORDS];
    for (s, sp) in spans.iter().enumerate() {
        words[s * DECODE_STATE_WORDS + DECODE_POS] = word(sp.rows);
    }
    let states = u32s(ops, &words);
    let partials = ops
        .scratch(decode_attention_partials(b * HEADS, HD, DECODE_ATTN_SPLITS))
        .expect("allocates");
    let out = ops.scratch(STEPS * b).expect("allocates");
    let (mut steps, mut embeds) = (Vec::new(), Vec::new());
    for t in 0..STEPS {
        let qs = ops.scratch(b * QW).expect("allocates");
        ops.backend()
            .copy_d2d(qs.view(), rows(&fx.step_q, t * b, b, QW))
            .expect("copies");
        ops.qk_norm_rope_append_batch(&QkRopeAppendBatchDesc {
            q: qs.view(),
            k: rows(&fx.step_k, t * b, b, KW),
            v: rows(&fx.step_v, t * b, b, KW),
            q_norm: fx.q_norm.view(),
            k_norm: fx.k_norm.view(),
            cache,
            heads: HEADS,
            eps: EPS,
            theta: THETA,
            states: states.view(),
        })
        .expect("appends");
        let att = ops.scratch(b * QW).expect("allocates");
        ops.decode_attention_batch(&DecodeAttnBatchDesc {
            out: att.view(),
            partials: partials.view(),
            splits: DECODE_ATTN_SPLITS,
            q: qs.view(),
            cache,
            heads: HEADS,
            states: states.view(),
            scale: SCALE,
        })
        .expect("attends");
        let ids = u32s(ops, &fx.ids[t * b..(t + 1) * b]);
        ops.decode_step_batch(&DecodeStepBatchDesc {
            states: states.view(),
            out: out.view(),
            ids: ids.view(),
            eos: EOS,
            max_tokens: 100,
            advance: 1,
            slot: t,
            seqs: b,
        })
        .expect("steps");
        let emb = ops.scratch(b * QW).expect("allocates");
        ops.embed_token_batch(&EmbedTokenBatchDesc {
            out: emb.view(),
            table: EmbedTable::Dense {
                dtype: GemmType::F32,
                rows: fx.embed.view(),
            },
            states: states.view(),
            width: QW,
            vocab: 256,
            seqs: b,
        })
        .expect("fetches");
        steps.push(read(ops, att.view(), b * QW));
        embeds.push(read(ops, emb.view(), b * QW));
    }
    let slots = read(ops, out.view(), STEPS * b);
    let st = read(ops, states.view(), b * DECODE_STATE_WORDS);
    let (kall, vall) = (
        read(ops, kc.view(), plane * HD),
        read(ops, vc.view(), plane * HD),
    );
    let mut run = Run::default();
    for (s, sp) in spans.iter().enumerate() {
        run.prefill
            .push(prefill[sp.row0 * QW..(sp.row0 + sp.rows) * QW].to_vec());
        run.steps.push(
            (0..STEPS)
                .flat_map(|t| steps[t][s * QW..(s + 1) * QW].to_vec())
                .collect(),
        );
        run.embeds.push(
            (0..STEPS)
                .flat_map(|t| embeds[t][s * QW..(s + 1) * QW].to_vec())
                .collect(),
        );
        run.slots
            .push((0..STEPS).map(|t| slots[t * b + s]).collect());
        run.states
            .push(st[s * DECODE_STATE_WORDS..(s + 1) * DECODE_STATE_WORDS].to_vec());
        let span = sp.plane * HD..(sp.plane + KV_HEADS * sp.cap) * HD;
        run.caches.push([&kall[span.clone()], &vall[span]].concat());
    }
    run
}

#[test]
#[ignore = "opens a device backend"]
fn batched_launches_give_each_sequence_its_single_values() {
    let Some(backend) = super::testing::device() else {
        return;
    };
    let ops = Ops::new(backend);
    let fx = fixture(&ops);
    let want = single(&ops, &fx);
    let got = batched(&ops, &fx);
    for s in 0..SEQS.len() {
        let what = |part: &str| format!("sequence {s}: {part}");
        same(
            &got.prefill[s],
            &want.prefill[s],
            &what("prefill attention"),
        );
        same(&got.steps[s], &want.steps[s], &what("decode attention"));
        same(&got.embeds[s], &want.embeds[s], &what("embedding rows"));
        assert_eq!(got.slots[s], want.slots[s], "{}", what("slots"));
        assert_eq!(got.states[s], want.states[s], "{}", what("state"));
        same(&got.caches[s], &want.caches[s], &what("cache planes"));
    }
    assert_eq!(
        got.states[1][DECODE_DONE], 1,
        "sequence 1 ended on an end id"
    );
}

/// Batched decode attention over eight sequences of 600 cached rows each,
/// against eight single launches over the same planes; says both times.
#[test]
#[ignore = "opens a device backend; a measurement, not a check"]
fn batched_decode_attention_time_against_single_launches() {
    let Some(backend) = super::testing::device() else {
        return;
    };
    let ops = Ops::new(backend);
    let (b, len, cap, reps) = (8usize, 600usize, 640usize, 200usize);
    let plane = KV_HEADS * cap;
    let spans: Vec<SeqSpan> = (0..b)
        .map(|s| SeqSpan {
            plane: s * plane,
            cap,
            ..SeqSpan::default()
        })
        .collect();
    let table = SeqTable::new(&ops, &spans, KV_HEADS).expect("uploads");
    let kc = upload_f32(&ops, &signal(b * plane * HD, 21)).expect("uploads");
    let vc = upload_f32(&ops, &signal(b * plane * HD, 22)).expect("uploads");
    let q = upload_f32(&ops, &signal(b * QW, 23)).expect("uploads");
    let mut words = vec![0u32; b * DECODE_STATE_WORDS];
    for s in 0..b {
        words[s * DECODE_STATE_WORDS + DECODE_POS] = word(len - 1);
    }
    let states = u32s(&ops, &words);
    let out = ops.scratch(b * QW).expect("allocates");
    let partials = ops
        .scratch(decode_attention_partials(b * HEADS, HD, DECODE_ATTN_SPLITS))
        .expect("allocates");
    let cache = BatchPlanes {
        k: kc.view(),
        v: vc.view(),
        table: &table,
        kv_heads: KV_HEADS,
        head_dim: HD,
    };
    let batch = |splits: usize| {
        ops.decode_attention_batch(&DecodeAttnBatchDesc {
            out: out.view(),
            partials: partials.view(),
            splits,
            q: q.view(),
            cache,
            heads: HEADS,
            states: states.view(),
            scale: SCALE,
        })
        .expect("attends");
    };
    let singles = || {
        for s in 0..b {
            ops.decode_attention(&DecodeAttnDesc {
                out: rows(&out, s, 1, QW),
                partials: rows(
                    &partials,
                    0,
                    1,
                    decode_attention_partials(HEADS, HD, DECODE_ATTN_SPLITS),
                ),
                splits: DECODE_ATTN_SPLITS,
                q: rows(&q, s, 1, QW),
                cache: CachePlanes {
                    k: rows(&kc, s * plane, plane, HD),
                    v: rows(&vc, s * plane, plane, HD),
                    kv_heads: KV_HEADS,
                    cap,
                    head_dim: HD,
                },
                heads: HEADS,
                state: rows(&states, s, 1, DECODE_STATE_WORDS),
                scale: SCALE,
            })
            .expect("attends");
        }
    };
    let time = |f: &dyn Fn()| {
        f();
        ops.backend().sync().expect("syncs");
        let start = Instant::now();
        for _ in 0..reps {
            f();
        }
        ops.backend().sync().expect("syncs");
        start.elapsed().as_secs_f64() * 1e6 / f64::from(word(reps))
    };
    let ts = time(&singles);
    say(&format!(
        "decode attention, {b} sequences of {len} rows, {DECODE_ATTN_SPLITS} splits: {b} single launches {ts:.1} us"
    ));
    // Fewer splits a sequence may do as well once the sequences fill the
    // device; the measurement says whether they do.
    for splits in [8, 16, DECODE_ATTN_SPLITS] {
        let tb = time(&|| batch(splits));
        say(&format!("  batched, {splits} splits: {tb:.1} us"));
    }
}
