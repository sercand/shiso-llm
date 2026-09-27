//! The batched decoder against the single-sequence one.
//!
//! Needs `SHISO_TEST_MODEL_QWEN3ASR`, `SHISO_TEST_MODEL_R2T2`, the reference
//! encoder fixtures under `test-vectors/references/audio/` and a GPU device;
//! run in release. The gate is exactness: every batch - one chunk, three
//! mixed ones, eight with repeats so the short ones finish first - must give
//! each chunk the ids [`Decoder::transcribe`] gives it alone, which on the HF
//! exports are the reference's own.

use std::collections::HashMap;
use std::time::Duration;

use super::{BatchMemory, BatchTiming, BatchTranscriber, POOL_BATCHES, give_back};
use crate::audio_ops::Ops;
use crate::qwen3_asr::config::TextConfig;
use crate::qwen3_asr::config::tests::say;
use crate::qwen3_asr::decoder::tests::{
    CLIPS, DEVICE, Loaded, load, ops, per_pass, reference_rows, want,
};
use crate::qwen3_asr::decoder::{Decoder, KvCache, POOLED_CAPACITY};
use crate::qwen3_asr::encoder::Encoded;

/// Three mixed clips, long first.
const MIXED: [&str; 3] = ["two-speakers", "weather-es", "speech-en-odd"];
/// Eight chunks, repeated and mixed so that short ones finish first and the
/// long ones run on alone.
const EIGHT: [&str; 8] = [
    "speech-en",
    "weather-es",
    "two-speakers",
    "speech-en-odd",
    "weather-es",
    "speech-en",
    "speech-en-odd",
    "two-speakers",
];

fn turn() -> std::sync::MutexGuard<'static, ()> {
    DEVICE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Each clip's encoder fixture on the device, or `None` having said which is
/// absent.
fn clips(ops: &Ops, model: &str) -> Option<HashMap<&'static str, Encoded>> {
    let mut out = HashMap::new();
    for clip in CLIPS {
        want(model, clip)?;
        out.insert(clip, reference_rows(ops, model, clip));
    }
    Some(out)
}

/// The single path's ids of every clip with `forced`.
fn singles(
    ops: &Ops,
    m: &Loaded,
    rows: &HashMap<&'static str, Encoded>,
    forced: Option<&str>,
) -> HashMap<&'static str, Vec<u32>> {
    let decoder = Decoder::new(ops, &m.weights.text, &m.config.text);
    let mut cache = KvCache::new(&m.config.text);
    CLIPS
        .iter()
        .map(|&clip| {
            let ids = decoder
                .transcribe(&mut cache, &rows[clip], forced)
                .expect("decodes");
            (clip, ids)
        })
        .collect()
}

fn line(t: &BatchTiming) -> String {
    format!(
        "prefill {} rows {:.1} ms, decode {} tokens in {} passes {:.1} ms ({:.2} ms/pass)",
        t.prompt,
        t.prefill.as_secs_f64() * 1e3,
        t.tokens,
        t.passes,
        t.decode.as_secs_f64() * 1e3,
        per_pass_of(t.decode, t.passes)
    )
}

fn per_pass_of(decode: Duration, passes: usize) -> f64 {
    decode.as_secs_f64() * 1e3 / crate::audio_ops::f64_of(passes.max(1))
}

/// `names` through one batch with `forced`, against `want`; says the timing.
fn check(
    batch: &BatchTranscriber<'_>,
    memory: &mut BatchMemory,
    rows: &HashMap<&'static str, Encoded>,
    names: &[&'static str],
    forced: Option<&str>,
    want: &HashMap<&'static str, Vec<u32>>,
    what: &str,
) {
    let chunks: Vec<&Encoded> = names.iter().map(|n| &rows[n]).collect();
    let (got, timing) = batch.timed_with(memory, &chunks, forced).expect("decodes");
    let mode = forced.unwrap_or("auto");
    say(&format!(
        "{what} B={} {mode}: {}",
        names.len(),
        line(&timing)
    ));
    assert_eq!(got.len(), names.len(), "{what}: one id list per chunk");
    for (i, (ids, name)) in got.iter().zip(names).enumerate() {
        assert_eq!(
            ids,
            &want[name],
            "{what} B={} {mode}: chunk {i} ({name}) ids",
            names.len()
        );
    }
}

/// Every batch shape of one export, auto and forced, against its own single
/// path; returns the batches compared.
fn every_shape(ops: &Ops, export: &str, model: &str) -> usize {
    let Some(m) = load(ops, export) else {
        return 0;
    };
    let Some(rows) = clips(ops, model) else {
        return 0;
    };
    let batch = BatchTranscriber::new(ops, &m.weights.text, &m.config.text);
    let mut memory = BatchMemory::new(&m.config.text);
    let mut compared = 0;
    for forced in [None, Some("English")] {
        let want = singles(ops, &m, &rows, forced);
        // One untimed batch, so the timings leave out pipeline compiles.
        drop(batch.timed_with(&mut memory, &[&rows["weather-es"]], forced));
        for clip in CLIPS {
            check(&batch, &mut memory, &rows, &[clip], forced, &want, export);
            compared += 1;
        }
        check(&batch, &mut memory, &rows, &MIXED, forced, &want, export);
        let rotated = [MIXED[2], MIXED[0], MIXED[1]];
        check(&batch, &mut memory, &rows, &rotated, forced, &want, export);
        check(&batch, &mut memory, &rows, &EIGHT, forced, &want, export);
        compared += 3;
    }
    // A fixture's own forced language on its own clip, which the reference
    // ran: Spanish for weather-es.
    let spanish = singles(ops, &m, &rows, Some("Spanish"));
    check(
        &batch,
        &mut memory,
        &rows,
        &["weather-es", "speech-en", "weather-es"],
        Some("Spanish"),
        &spanish,
        export,
    );
    say(&format!("{export}: {} batches exact", compared + 1));
    compared + 1
}

#[test]
#[ignore = "requires a GPU device, SHISO_TEST_MODEL_QWEN3ASR and SHISO_TEST_MODEL_R2T2"]
fn batched_ids_equal_single_ids_for_every_fixture() {
    let _turn = turn();
    let Some(ops) = ops() else {
        return;
    };
    let compared = every_shape(&ops, "qwen hf", "qwen3asr") + every_shape(&ops, "r2t2 hf", "r2t2");
    assert!(compared > 0, "no export was compared");
}

#[test]
#[ignore = "requires a GPU device and the real checkpoints"]
fn quantised_batches_match_their_single_path() {
    let _turn = turn();
    let Some(ops) = ops() else {
        return;
    };
    let mut compared = 0;
    for (export, model) in [
        ("qwen gguf q8_0", "qwen3asr"),
        ("qwen mlx", "qwen3asr"),
        ("r2t2 gguf q8_0", "r2t2"),
    ] {
        let Some(m) = load(&ops, export) else {
            continue;
        };
        let Some(rows) = clips(&ops, model) else {
            continue;
        };
        let batch = BatchTranscriber::new(&ops, &m.weights.text, &m.config.text);
        let mut memory = BatchMemory::new(&m.config.text);
        for forced in [None, Some("English")] {
            let want = singles(&ops, &m, &rows, forced);
            check(&batch, &mut memory, &rows, &MIXED, forced, &want, export);
            check(&batch, &mut memory, &rows, &EIGHT, forced, &want, export);
            compared += 2;
        }
    }
    say(&format!("quantised: {compared} batches exact"));
    if ["SHISO_TEST_MODEL_QWEN3ASR", "SHISO_TEST_MODEL_R2T2"]
        .iter()
        .any(|var| std::env::var_os(var).is_some())
    {
        assert!(
            compared > 0,
            "a checkpoint root is set, but nothing was compared"
        );
    }
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_QWEN3ASR"]
fn a_finished_sequence_writes_nothing_more() {
    let _turn = turn();
    let Some(ops) = ops() else {
        return;
    };
    let Some(m) = load(&ops, "qwen hf") else {
        return;
    };
    let Some(rows) = clips(&ops, "qwen3asr") else {
        return;
    };
    let c = &m.config.text;
    let want = singles(&ops, &m, &rows, None);
    let names = ["speech-en", "weather-es", "speech-en-odd", "weather-es"];
    let chunks: Vec<&Encoded> = names.iter().map(|n| &rows[n]).collect();
    for compact in [false, true] {
        let batch =
            BatchTranscriber::new(&ops, &m.weights.text, &m.config.text).compacting(compact);
        // No headroom: every plane holds exactly its prompt, so the first
        // batch of steps grows them all, and the NaN sentinel fills every
        // row no append has reached.
        let mut memory = BatchMemory::new(c).with_sentinel().with_headroom(0);
        let got = batch
            .transcribe_with(&mut memory, &chunks, None)
            .expect("decodes");
        assert!(memory.cache.grown > 0, "the planes grew while decoding");
        for (s, name) in names.iter().enumerate() {
            assert_eq!(got[s], want[name], "compact {compact}: {name} ids");
            let prompt = crate::qwen3_asr::prompt::ids(rows[name].tokens, None)
                .expect("a prompt")
                .len();
            let end = memory.cache.ends[s];
            // The last pass that appended ran over the last emitted token,
            // and its argmax was the end id.
            assert_eq!(end, prompt + got[s].len(), "{name}: rows appended");
            let cap = memory.cache.spans[s].cap;
            for layer in [0, c.layers / 2, c.layers - 1] {
                for which in 0..2 {
                    let plane = memory.cache.plane(&ops, layer, s, which);
                    for h in 0..c.kv_heads {
                        let head = &plane[h * cap * c.head_dim..(h + 1) * cap * c.head_dim];
                        let (filled, past) = head.split_at(end * c.head_dim);
                        assert!(
                            filled.iter().all(|v| !v.is_nan()),
                            "compact {compact}: {name} layer {layer} head {h}: a row before {end} is unwritten"
                        );
                        assert!(
                            past.iter().all(|v| v.is_nan()),
                            "compact {compact}: {name} layer {layer} head {h}: a row at or past {end} (of {cap}) was written"
                        );
                    }
                }
            }
        }
        say(&format!(
            "compact {compact}: {} chunks end at {:?}, planes {:?}, grown {} times",
            names.len(),
            memory.cache.ends,
            memory.cache.spans.iter().map(|s| s.cap).collect::<Vec<_>>(),
            memory.cache.grown
        ));
    }
}

/// A geometry for host-side pool bookkeeping.
fn tiny() -> TextConfig {
    TextConfig {
        hidden: 8,
        layers: 1,
        heads: 1,
        kv_heads: 2,
        head_dim: 8,
        ffn: 8,
        vocab: 8,
        rms_eps: 1e-6,
        rope_theta: 1e6,
        max_positions: 1 << 16,
    }
}

#[test]
fn pooled_memory_is_bounded() {
    let c = tiny();
    let rows = |seqs: usize, cap: usize| seqs * c.kv_heads * cap;
    let mut pool = None;
    // A set whose planes grew past the default - a degenerate chunk run to
    // the token cap - is dropped, not kept for the life of the process.
    give_back(
        &mut pool,
        BatchMemory::claiming(&c, 8, rows(8, POOLED_CAPACITY) + 1),
    );
    assert!(pool.is_none());
    // A batch-sized one is kept.
    give_back(
        &mut pool,
        BatchMemory::claiming(&c, 8, rows(8, POOLED_CAPACITY)),
    );
    assert_eq!(
        pool.as_ref().map(BatchMemory::rows),
        Some(rows(8, POOLED_CAPACITY))
    );
    // And the pool holds no more than one: a second ordinary set is dropped.
    for _ in 0..3 {
        give_back(&mut pool, BatchMemory::claiming(&c, 4, rows(4, 300)));
    }
    assert_eq!(POOL_BATCHES, 1);
    assert_eq!(
        pool.as_ref().map(BatchMemory::rows),
        Some(rows(8, POOLED_CAPACITY))
    );
    // A set sized for a few sequences is judged by the most it ever held.
    let mut pool = None;
    give_back(
        &mut pool,
        BatchMemory::claiming(&c, 2, rows(3, POOLED_CAPACITY)),
    );
    assert!(
        pool.is_none(),
        "two sequences' set holding three planes' rows"
    );
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_QWEN3ASR; a measurement, not a check"]
fn batched_step_and_prefill_times() {
    let _turn = turn();
    let Some(ops) = ops() else {
        return;
    };
    let Some(m) = load(&ops, "qwen hf") else {
        return;
    };
    let Some(rows) = clips(&ops, "qwen3asr") else {
        return;
    };
    let decoder = Decoder::new(&ops, &m.weights.text, &m.config.text);
    let mut cache = KvCache::new(&m.config.text);
    let clip = &rows["two-speakers"];
    drop(decoder.timed(&mut cache, clip, None));
    let (_, single) = decoder.timed(&mut cache, clip, None).expect("decodes");
    say(&format!(
        "single two-speakers: prefill {} rows {:.1} ms, {:.2} ms/pass",
        single.prompt,
        single.prefill.as_secs_f64() * 1e3,
        per_pass(&single)
    ));
    for compact in [false, true] {
        let batch =
            BatchTranscriber::new(&ops, &m.weights.text, &m.config.text).compacting(compact);
        let mut memory = BatchMemory::new(&m.config.text);
        for b in [1usize, 4, 8] {
            let chunks: Vec<&Encoded> = (0..b).map(|_| clip).collect();
            drop(batch.timed_with(&mut memory, &chunks, None));
            let (_, t) = batch
                .timed_with(&mut memory, &chunks, None)
                .expect("decodes");
            say(&format!(
                "compact {compact} B={b} same clip: {} ({:.2} ms/pass/sequence)",
                line(&t),
                per_pass_of(t.decode, t.passes) / crate::audio_ops::f64_of(b)
            ));
        }
        let chunks: Vec<&Encoded> = EIGHT.iter().map(|n| &rows[n]).collect();
        drop(batch.timed_with(&mut memory, &chunks, None));
        let (_, t) = batch
            .timed_with(&mut memory, &chunks, None)
            .expect("decodes");
        say(&format!("compact {compact} B=8 mixed: {}", line(&t)));
        let mut singles = Duration::ZERO;
        for name in EIGHT {
            let (_, t) = decoder
                .timed(&mut cache, &rows[name], None)
                .expect("decodes");
            singles += t.prefill + t.decode;
        }
        say(&format!(
            "  the same eight one at a time: {:.1} ms; batched {:.1} ms",
            singles.as_secs_f64() * 1e3,
            (t.prefill + t.decode).as_secs_f64() * 1e3
        ));
    }
}
