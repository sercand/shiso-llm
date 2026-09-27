//! The decoder launches through [`Ops`], end to end on the device: a prefill
//! and a run of queued decode steps over the same rows must agree, which pins
//! each descriptor's buffer order against the kernels' and the state words'
//! meaning between the kernels that share them.

use shiso_core::{DeviceBuffer, DeviceView};

use super::load::{download_f32, upload_f32};
use super::{
    CachePlanes, CausalAttnDesc, DECODE_ATTN_SPLITS, DECODE_COUNT, DECODE_POS, DECODE_STATE_WORDS,
    DECODE_TOKEN, DecodeAttnDesc, DecodeStepDesc, EmbedTable, EmbedTokenDesc, GemmType,
    KvAppendDesc, Ops, decode_attention_partials,
};

const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HD: usize = 128;
const CAP: usize = 64;
/// Rows the reference prefills, and rows the decode path prefills before it
/// steps through the rest.
const ROWS: usize = 20;
const FIRST: usize = 10;
/// `1 / sqrt(HD)`.
const SCALE: f32 = 0.088_388_35;
/// Widths of a query row and of a key or value row.
const QW: usize = HEADS * HD;
const KW: usize = KV_HEADS * HD;

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

fn state(ops: &Ops, words: [u32; DECODE_STATE_WORDS]) -> DeviceBuffer {
    let buffer = ops.scratch(DECODE_STATE_WORDS).expect("allocates");
    let raw: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ops.backend()
        .copy_h2d(buffer.view(), &raw)
        .expect("uploads");
    buffer
}

fn words(ops: &Ops, buffer: &DeviceBuffer, count: usize) -> Vec<u32> {
    download_f32(ops, buffer.view(), count)
        .expect("reads")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Queries, keys and values of every row.
struct Rows {
    q: DeviceBuffer,
    keys: DeviceBuffer,
    values: DeviceBuffer,
}

fn fixture(ops: &Ops) -> Rows {
    Rows {
        q: upload_f32(ops, &signal(ROWS * QW, 1)).expect("uploads"),
        keys: upload_f32(ops, &signal(ROWS * KW, 2)).expect("uploads"),
        values: upload_f32(ops, &signal(ROWS * KW, 3)).expect("uploads"),
    }
}

fn planes<'a>(keys: &'a DeviceBuffer, values: &'a DeviceBuffer) -> CachePlanes<'a> {
    CachePlanes {
        k: keys.view(),
        v: values.view(),
        kv_heads: KV_HEADS,
        cap: CAP,
        head_dim: HD,
    }
}

fn cache_buffers(ops: &Ops) -> (DeviceBuffer, DeviceBuffer) {
    let plane = KV_HEADS * CAP * HD;
    (
        ops.scratch(plane).expect("allocates"),
        ops.scratch(plane).expect("allocates"),
    )
}

/// Every row appended at position 0 and attended causally in one launch.
fn prefill_reference(ops: &Ops, fx: &Rows) -> Vec<f32> {
    let (kc, vc) = cache_buffers(ops);
    let cache = planes(&kc, &vc);
    let st = state(ops, [0; DECODE_STATE_WORDS]);
    ops.kv_append(&KvAppendDesc {
        cache,
        k: fx.keys.view(),
        v: fx.values.view(),
        state: st.view(),
        rows: ROWS,
    })
    .expect("appends");
    let want = ops.scratch(ROWS * QW).expect("allocates");
    ops.causal_attention(&CausalAttnDesc {
        out: want.view(),
        q: fx.q.view(),
        cache,
        n_q: ROWS,
        heads: HEADS,
        pos0: 0,
        scale: SCALE,
    })
    .expect("attends");
    download_f32(ops, want.view(), ROWS * QW).expect("reads")
}

/// The first rows prefilled, then one queued decode step per row after,
/// with no read between them. Returns the attention rows, the state words
/// and the emitted slots.
fn decoded(ops: &Ops, fx: &Rows) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
    let (kc, vc) = cache_buffers(ops);
    let cache = planes(&kc, &vc);
    let st = state(ops, [0; DECODE_STATE_WORDS]);
    ops.kv_append(&KvAppendDesc {
        cache,
        k: rows(&fx.keys, 0, FIRST, KW),
        v: rows(&fx.values, 0, FIRST, KW),
        state: st.view(),
        rows: FIRST,
    })
    .expect("appends");
    ops.backend()
        .copy_h2d(
            st.slice(0, 4).expect("one word"),
            &u32::try_from(FIRST).expect("fits").to_le_bytes(),
        )
        .expect("sets the position");
    let got = ops.scratch(ROWS * QW).expect("allocates");
    let partials = ops
        .scratch(decode_attention_partials(HEADS, HD, DECODE_ATTN_SPLITS))
        .expect("allocates");
    let out = ops.scratch(ROWS - FIRST).expect("allocates");
    // The ids travel as raw words through the f32 upload.
    let ids: Vec<f32> = (0..ROWS - FIRST)
        .map(|j| f32::from_bits(1000 + u32::try_from(j).expect("fits")))
        .collect();
    let ids = upload_f32(ops, &ids).expect("uploads");
    for j in FIRST..ROWS {
        ops.kv_append(&KvAppendDesc {
            cache,
            k: rows(&fx.keys, j, 1, KW),
            v: rows(&fx.values, j, 1, KW),
            state: st.view(),
            rows: 1,
        })
        .expect("appends");
        ops.decode_attention(&DecodeAttnDesc {
            out: rows(&got, j, 1, QW),
            partials: partials.view(),
            splits: DECODE_ATTN_SPLITS,
            q: rows(&fx.q, j, 1, QW),
            cache,
            heads: HEADS,
            state: st.view(),
            scale: SCALE,
        })
        .expect("attends");
        ops.decode_step(&DecodeStepDesc {
            state: st.view(),
            out: out.view(),
            ids: rows(&ids, j - FIRST, 1, 1),
            eos: [7, 8],
            max_tokens: 100,
            advance: 1,
            slot: j - FIRST,
        })
        .expect("steps");
    }
    (
        download_f32(ops, got.view(), ROWS * QW).expect("reads"),
        words(ops, &st, DECODE_STATE_WORDS),
        words(ops, &out, ROWS - FIRST),
    )
}

#[test]
#[ignore = "opens a device backend"]
fn queued_decode_steps_match_the_prefill_over_the_same_rows() {
    let Some(backend) = super::testing::device() else {
        return;
    };
    let ops = Ops::new(backend);
    let fx = fixture(&ops);
    let want = prefill_reference(&ops, &fx);
    let (got, st, emitted) = decoded(&ops, &fx);
    for i in FIRST * QW..ROWS * QW {
        assert!(
            (want[i] - got[i]).abs() <= 2e-6,
            "row {} dim {}: prefill {} decode {}",
            i / QW,
            i % QW,
            want[i],
            got[i]
        );
    }
    let steps = u32::try_from(ROWS - FIRST).expect("fits");
    assert_eq!(st[DECODE_POS], u32::try_from(ROWS).expect("fits"));
    assert_eq!(st[DECODE_COUNT], steps);
    assert_eq!(st[DECODE_TOKEN], 1000 + steps - 1);
    assert_eq!(emitted, (1000..1000 + steps).collect::<Vec<u32>>());
}

#[test]
#[ignore = "opens a device backend"]
fn the_embedding_row_is_the_state_tokens() {
    let Some(backend) = super::testing::device() else {
        return;
    };
    let ops = Ops::new(backend);
    let (vocab, width, token) = (1024usize, 256usize, 1009usize);
    let mut words = [0u32; DECODE_STATE_WORDS];
    words[DECODE_TOKEN] = u32::try_from(token).expect("fits");
    let st = state(&ops, words);
    let table = signal(vocab * width, 4);
    let tb = upload_f32(&ops, &table).expect("uploads");
    let row = ops.scratch(width).expect("allocates");
    ops.embed_token(&EmbedTokenDesc {
        out: row.view(),
        table: EmbedTable::Dense {
            dtype: GemmType::F32,
            rows: tb.view(),
        },
        state: st.view(),
        width,
        vocab,
    })
    .expect("fetches");
    assert_eq!(
        download_f32(&ops, row.view(), width).expect("reads"),
        table[token * width..(token + 1) * width]
    );
}
