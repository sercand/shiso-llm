//! The decoder against the reference's own greedy token ids.
//!
//! Needs `SHISO_TEST_MODEL_QWEN3ASR`, `SHISO_TEST_MODEL_R2T2`, the reference
//! fixtures under `test-vectors/references/audio/` and a GPU device; run in
//! release. The reference's own tower output is fed in, not ours, so the
//! decoder is all that is compared: the reference ran in f32 from the HF bf16
//! weights, so the two HF exports must give its ids exactly, auto and with the
//! language forced. The quantised exports are held to the word differences
//! measured on them. One case runs the whole path - the reference's mel,
//! our tower, our decoder - to see what the tower's own error does to the
//! ids.
//!
//! Every case loads checkpoints of several gigabytes, so they take turns on
//! one lock rather than loading side by side.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use shiso_audio::{Detokenizer, MelFrames, PieceVocab};

use super::{Decoder, KvCache, Timing};
use crate::audio_ops::Ops;
use crate::audio_ops::load::upload_f32;
use crate::qwen3_asr::config::tests::{Export, exports, say};
use crate::qwen3_asr::config::{IM_END, Qwen3AsrConfig, locate, read_vocab};
use crate::qwen3_asr::encoder::{Encoded, Tower};
use crate::qwen3_asr::prompt::parse;
use crate::qwen3_asr::weights::QwenAsrWeights;

/// Width of a tower row.
pub(crate) const OUT: usize = 2048;

pub(crate) const CLIPS: [&str; 4] = ["speech-en", "speech-en-odd", "weather-es", "two-speakers"];

/// One device case at a time.
pub(crate) static DEVICE: Mutex<()> = Mutex::new(());

pub(crate) fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/audio")
}

fn f32s(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .expect("json")
}

fn u32s(value: &serde_json::Value) -> Vec<u32> {
    value
        .as_array()
        .expect("an array of ids")
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("an id")).expect("fits"))
        .collect()
}

/// What the reference generated for one clip in one mode.
pub(crate) struct Want {
    /// The ids before the end id, which the decoder never emits.
    pub(crate) ids: Vec<u32>,
    /// The forced language, or `None` for the auto prompt.
    pub(crate) forced: Option<String>,
    /// The parsed language and text.
    language: String,
    text: String,
}

/// The reference's two runs of `clip` (auto, then forced), or `None` having
/// said the fixture is absent.
pub(crate) fn want(model: &str, clip: &str) -> Option<[Want; 2]> {
    let path = fixtures().join(format!("{model}-{clip}-tokens.json"));
    if !path.exists() {
        say(&format!("SKIP {}: absent", path.display()));
        return None;
    }
    let j = json(&path);
    let one = |mode: &str| {
        let m = &j[mode];
        let mut ids = u32s(&m["generated_ids"]);
        assert_eq!(
            ids.pop(),
            Some(IM_END),
            "{model} {clip} {mode}: ends at the end id"
        );
        Want {
            ids,
            forced: m["language_forced"].as_str().map(str::to_owned),
            language: m["language"].as_str().expect("a language").to_owned(),
            text: m["text"].as_str().expect("a text").to_owned(),
        }
    };
    Some([one("auto"), one("forced")])
}

/// The reference's tower output for `clip`, on the device.
pub(crate) fn reference_rows(ops: &Ops, model: &str, clip: &str) -> Encoded {
    let rows = f32s(&fixtures().join(format!("{model}-{clip}-encoder.f32")));
    assert_eq!(rows.len() % OUT, 0, "{model} {clip}: whole rows");
    Encoded {
        tokens: rows.len() / OUT,
        buffer: upload_f32(ops, &rows).expect("uploads"),
    }
}

pub(crate) fn ops() -> Option<Ops> {
    crate::audio_ops::testing::device().map(Ops::new)
}

/// One export on the device, with its vocabulary.
pub(crate) struct Loaded {
    pub(crate) config: Qwen3AsrConfig,
    pub(crate) weights: QwenAsrWeights,
    vocab: Arc<PieceVocab>,
}

/// The export named `name`, or `None` having said why.
pub(crate) fn load(ops: &Ops, name: &str) -> Option<Loaded> {
    let Some(export): Option<Export> = exports().into_iter().find(|e| e.name == name) else {
        say(&format!("SKIP {name}: its root variable is unset"));
        return None;
    };
    if !export.path.exists() {
        say(&format!("SKIP {name}: {} is absent", export.path.display()));
        return None;
    }
    let located = locate(&export.path, None)
        .expect("readable")
        .expect("this family");
    let stores = located.stores().expect("opens");
    let config = Qwen3AsrConfig::read(&located, &stores).expect("reads");
    let weights = QwenAsrWeights::load(ops, &located, &config, &stores).expect("loads");
    let vocab = Arc::new(read_vocab(&located, &stores).expect("a vocabulary"));
    Some(Loaded {
        config,
        weights,
        vocab,
    })
}

fn text_of(vocab: &Arc<PieceVocab>, ids: &[u32]) -> String {
    let mut detok = Detokenizer::new(vocab.clone());
    for &id in ids {
        detok.push(id);
    }
    detok.flush();
    detok.text().to_owned()
}

/// Word-level edit distance.
fn word_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<&str>, Vec<&str>) = (
        a.split_whitespace().collect(),
        b.split_whitespace().collect(),
    );
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, wa) in a.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, wb) in b.iter().enumerate() {
            let here = row[j + 1];
            row[j + 1] = (prev + usize::from(wa != wb)).min(row[j] + 1).min(here + 1);
            prev = here;
        }
    }
    row[b.len()]
}

/// Milliseconds per decode pass, read-backs included.
pub(crate) fn per_pass(t: &Timing) -> f64 {
    t.decode.as_secs_f64() * 1e3 / crate::audio_ops::f64_of(t.passes.max(1))
}

fn line(t: &Timing) -> String {
    format!(
        "prefill {} rows {:.1} ms, decode {} tokens in {} passes {:.1} ms ({:.2} ms/pass)",
        t.prompt,
        t.prefill.as_secs_f64() * 1e3,
        t.tokens,
        t.passes,
        t.decode.as_secs_f64() * 1e3,
        per_pass(t)
    )
}

/// Every clip of `model`, auto and forced, through `export`: the ids must be
/// the reference's.
fn exact(export: &str, model: &str) {
    let _turn = DEVICE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(ops) = ops() else {
        return;
    };
    let Some(m) = load(&ops, export) else {
        return;
    };
    let decoder = Decoder::new(&ops, &m.weights.text, &m.config.text);
    let mut cache = KvCache::new(&m.config.text);
    // One untimed pass, so the timings leave out pipeline compiles.
    drop(decoder.transcribe(&mut cache, &reference_rows(&ops, model, "weather-es"), None));
    let mut compared = 0;
    for clip in CLIPS {
        let Some(wants) = want(model, clip) else {
            continue;
        };
        let rows = reference_rows(&ops, model, clip);
        for w in &wants {
            let (got, timing) = decoder
                .timed(&mut cache, &rows, w.forced.as_deref())
                .expect("decodes");
            let mode = w.forced.as_deref().unwrap_or("auto");
            say(&format!("{export} {clip} {mode}: {}", line(&timing)));
            assert_eq!(got, w.ids, "{export} {clip} {mode}: ids");
            let parsed = parse(&text_of(&m.vocab, &got), w.forced.as_deref());
            assert_eq!(parsed.text, w.text, "{export} {clip} {mode}: text");
            assert_eq!(
                parsed.language.as_deref().unwrap_or(""),
                w.language,
                "{export} {clip} {mode}: language"
            );
            compared += 1;
        }
    }
    say(&format!("{export}: {compared} runs exact"));
    assert!(compared > 0, "{export}: no fixture was compared");
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_QWEN3ASR"]
fn hf_tokens_match_the_reference_exactly() {
    exact("qwen hf", "qwen3asr");
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_R2T2"]
fn r2t2_tokens_match_its_reference_exactly() {
    exact("r2t2 hf", "r2t2");
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_QWEN3ASR"]
fn a_growing_cache_changes_nothing() {
    let _turn = DEVICE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(ops) = ops() else {
        return;
    };
    let Some(m) = load(&ops, "qwen hf") else {
        return;
    };
    let decoder = Decoder::new(&ops, &m.weights.text, &m.config.text);
    let Some([auto, _]) = want("qwen3asr", "speech-en") else {
        return;
    };
    let rows = reference_rows(&ops, "qwen3asr", "speech-en");
    // Room for the prompt and ten decoded rows: the first batch of seven
    // steps fits, and the second grows the cache with those seven decoded
    // rows in it, which the copy must carry over.
    let prompt = crate::qwen3_asr::prompt::ids(rows.tokens, None)
        .expect("a prompt")
        .len();
    let mut tight = KvCache::tight(&ops, &m.config.text, prompt + 10).expect("allocates");
    let (got, timing) = decoder.timed(&mut tight, &rows, None).expect("decodes");
    say(&format!(
        "tight cache: capacity {} after {} prompt rows and {} tokens, grown with {:?} rows filled",
        tight.capacity(),
        timing.prompt,
        got.len(),
        tight.copied()
    ));
    assert_eq!(timing.prompt, prompt);
    assert!(got.len() > 10, "the decode runs past the first capacity");
    assert!(
        tight.copied().iter().any(|&filled| filled > prompt),
        "a growth while decoding copied decoded rows: {:?}",
        tight.copied()
    );
    assert_eq!(tight.copied().first(), Some(&(prompt + 7)));
    let mut roomy = KvCache::new(&m.config.text);
    let (again, _) = decoder.timed(&mut roomy, &rows, None).expect("decodes");
    assert_eq!(got, again, "the same ids either way");
    assert_eq!(got, auto.ids, "and the reference's");
}

/// Word differences from the reference text, measured 2026-09-27 on M1 Max
/// with the reference's tower rows fed in, per clip in [`CLIPS`] order, auto
/// then forced. The budget is each measured value plus two words.
///
/// Every run but one gave the reference's ids exactly, including MLX's 8-bit
/// affine decoder: the one is the base model's `Q8_0` decoder on weather-es
/// without a forced language, which writes the numbers as digits ("35",
/// "20") where the reference spells them out - two words.
const MEASURED: [(&str, &str, [usize; 8]); 4] = [
    ("qwen gguf q8_0", "qwen3asr", [0, 0, 0, 0, 2, 0, 0, 0]),
    ("qwen gguf bf16", "qwen3asr", [0; 8]),
    ("qwen mlx", "qwen3asr", [0; 8]),
    ("r2t2 gguf q8_0", "r2t2", [0; 8]),
];

#[test]
#[ignore = "requires a GPU device and the real checkpoints"]
fn quantised_exports_stay_close() {
    let _turn = DEVICE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(ops) = ops() else {
        return;
    };
    let (mut measured, mut compared) = (0, 0);
    for (export, model, budget) in MEASURED {
        let Some(m) = load(&ops, export) else {
            continue;
        };
        let decoder = Decoder::new(&ops, &m.weights.text, &m.config.text);
        let mut cache = KvCache::new(&m.config.text);
        let mut got_words = [0usize; 8];
        let mut report = format!("{export}:");
        for (c, clip) in CLIPS.iter().enumerate() {
            let Some(wants) = want(model, clip) else {
                continue;
            };
            let rows = reference_rows(&ops, model, clip);
            for (k, w) in wants.iter().enumerate() {
                let (got, timing) = decoder
                    .timed(&mut cache, &rows, w.forced.as_deref())
                    .expect("decodes");
                let parsed = parse(&text_of(&m.vocab, &got), w.forced.as_deref());
                let d = word_distance(&parsed.text, &w.text);
                got_words[c * 2 + k] = d;
                compared += 1;
                let same = parsed.language.as_deref().unwrap_or("") == w.language;
                write!(
                    report,
                    " {clip}/{} {d}w{}{} ({:.2} ms/pass)",
                    if k == 0 { "auto" } else { "forced" },
                    if got == w.ids { " exact" } else { "" },
                    if same { "" } else { " LANG" },
                    per_pass(&timing)
                )
                .expect("a String takes it");
                if got != w.ids {
                    say(&format!("  {export} {clip}: {:?}", parsed.text));
                }
            }
        }
        say(&report);
        for (i, (&d, &b)) in got_words.iter().zip(&budget).enumerate() {
            assert!(
                d <= b + 2,
                "{export} {} {}: {d} words differ, measured {b}",
                CLIPS[i / 2],
                if i % 2 == 0 { "auto" } else { "forced" }
            );
        }
        measured += 1;
    }
    say(&format!("measured {measured} exports, {compared} decodes"));
    // With a checkpoint root set, a run that compared nothing is a broken
    // setup - absent exports or fixtures - not a pass.
    if ["SHISO_TEST_MODEL_QWEN3ASR", "SHISO_TEST_MODEL_R2T2"]
        .iter()
        .any(|var| std::env::var_os(var).is_some())
    {
        assert!(
            compared > 0,
            "a checkpoint root is set, but no export was compared"
        );
    }
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_QWEN3ASR"]
fn hf_end_to_end_from_the_mel() {
    let _turn = DEVICE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(ops) = ops() else {
        return;
    };
    let Some(m) = load(&ops, "qwen hf") else {
        return;
    };
    let clip = "speech-en";
    let Some([auto, forced]) = want("qwen3asr", clip) else {
        return;
    };
    let meta = json(&fixtures().join(format!("qwen3asr-{clip}-mel.json")));
    let frames = usize::try_from(meta["frames"].as_u64().expect("frames")).expect("fits");
    let mel = f32s(&fixtures().join(format!("qwen3asr-{clip}-mel.f32")));
    assert_eq!(mel.len(), frames * 128);
    let mel = MelFrames::from_rows(mel, 128).expect("whole rows");
    let tower = Tower::new(&ops, &m.weights.tower, &m.config.audio);
    let rows = tower.encode(&mel).expect("encodes");
    let decoder = Decoder::new(&ops, &m.weights.text, &m.config.text);
    let mut cache = KvCache::new(&m.config.text);
    for w in [&auto, &forced] {
        let (got, _) = decoder
            .timed(&mut cache, &rows, w.forced.as_deref())
            .expect("decodes");
        let mode = w.forced.as_deref().unwrap_or("auto");
        let first = got.iter().zip(&w.ids).position(|(a, b)| a != b);
        say(&format!(
            "end to end {clip} {mode}: {} ids against {}, first difference {first:?}",
            got.len(),
            w.ids.len()
        ));
        assert_eq!(got, w.ids, "end to end {clip} {mode}");
    }
}
