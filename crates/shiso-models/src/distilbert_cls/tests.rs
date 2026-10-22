//! Host-only tests of the config reader and the weight validation, plus an
//! ignored check of a real checkpoint. Nothing here opens a device.

use std::path::Path;

use super::weights::{expected, validate};
use super::{ClassifierConfig, ClassifyError, describes};
use crate::audio_ops::store::TensorStore;

const GOLDENS_CONFIG: &str = r#"{"model_type":"distilbert","architectures":["DistilBertForSequenceClassification"],"n_layers":6,"n_heads":12,"dim":768,"hidden_dim":3072,"vocab_size":30522,"max_position_embeddings":512,"num_labels":3,"id2label":{"0":"safe","1":"needs-approval","2":"dangerous"},"label2id":{"safe":0,"needs-approval":1,"dangerous":2},"layer_norm_eps":1e-12,"calibration":{"temperature":1.3366203308105469,"energy_p95":-2.543365240097046,"safe_threshold":0.9,"max_length":128}}"#;

fn config_with(from: &str, to: &str) -> String {
    assert!(GOLDENS_CONFIG.contains(from), "{from} is not in the config");
    GOLDENS_CONFIG.replacen(from, to, 1)
}

/// A small config, so a complete checkpoint is a few hundred bytes.
fn tiny() -> ClassifierConfig {
    ClassifierConfig {
        layers: 1,
        heads: 2,
        dim: 4,
        hidden_dim: 8,
        vocab: 5,
        max_positions: 6,
        labels: vec!["a".into(), "b".into(), "c".into()],
        eps: 1e-12,
    }
}

/// A tensor to write: name, safetensors dtype, shape.
type Spec = (String, &'static str, Vec<u64>);

/// Writes a safetensors file by hand: an 8-byte little-endian header length,
/// the JSON header, then zeroed payloads laid end to end.
fn write_safetensors(path: &Path, tensors: &[Spec]) {
    let mut header = serde_json::Map::new();
    let mut at = 0_usize;
    for (name, dtype, shape) in tensors {
        let width = match *dtype {
            "F32" => 4,
            "F16" | "BF16" => 2,
            _ => 1,
        };
        #[expect(clippy::cast_possible_truncation, reason = "tiny test shapes")]
        let len = shape.iter().product::<u64>() as usize * width;
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [at, at + len]}),
        );
        at += len;
    }
    let header = serde_json::Value::Object(header).to_string().into_bytes();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.resize(file.len() + at, 0);
    std::fs::write(path, file).expect("write the checkpoint");
}

/// Every tensor `cfg` implies, matrices F16 and the rest F32, as a converter
/// would write them.
fn complete(cfg: &ClassifierConfig) -> Vec<Spec> {
    expected(cfg)
        .into_iter()
        .map(|(name, shape)| {
            let dtype = if shape.len() == 2 { "F16" } else { "F32" };
            (name, dtype, shape)
        })
        .collect()
}

fn store_of(dir: &Path, tensors: &[Spec]) -> TensorStore {
    let path = dir.join("model.safetensors");
    write_safetensors(&path, tensors);
    TensorStore::open_safetensors(&path).expect("open the checkpoint")
}

#[test]
fn config_parses_goldens_config_json() {
    let cfg = ClassifierConfig::from_json(GOLDENS_CONFIG).expect("the goldens config parses");
    assert_eq!(cfg.layers, 6);
    assert_eq!(cfg.heads, 12);
    assert_eq!(cfg.dim, 768);
    assert_eq!(cfg.hidden_dim, 3072);
    assert_eq!(cfg.vocab, 30522);
    assert_eq!(cfg.max_positions, 512);
    assert_eq!(cfg.head_dim(), 64);
    assert_eq!(cfg.labels, ["safe", "needs-approval", "dangerous"]);
    assert_eq!(cfg.eps.to_bits(), 1e-12_f32.to_bits());
}

#[test]
fn config_rejects_wrong_arch() {
    // A different depth is a different stack but the same family.
    let deeper = config_with(r#""n_layers":6"#, r#""n_layers":12"#);
    assert_eq!(
        ClassifierConfig::from_json(&deeper).expect("parses").layers,
        12
    );

    let ragged = config_with(r#""dim":768"#, r#""dim":100"#);
    match ClassifierConfig::from_json(&ragged) {
        Err(ClassifyError::BadConfig(why)) => assert!(why.contains("dim not divisible by heads")),
        other => panic!("expected BadConfig, got {other:?}"),
    }
    let two = config_with(r#""num_labels":3"#, r#""num_labels":2"#);
    assert!(matches!(
        ClassifierConfig::from_json(&two),
        Err(ClassifyError::BadConfig(_))
    ));
    let bert = config_with(r#""model_type":"distilbert""#, r#""model_type":"bert""#);
    assert!(matches!(
        ClassifierConfig::from_json(&bert),
        Err(ClassifyError::BadConfig(_))
    ));
    let twin = config_with(r#""2":"dangerous""#, r#""2":"safe""#);
    assert!(matches!(
        ClassifierConfig::from_json(&twin),
        Err(ClassifyError::BadConfig(why)) if why.contains("repeats")
    ));
    let gap = config_with(r#""2":"dangerous""#, r#""7":"dangerous""#);
    assert!(matches!(
        ClassifierConfig::from_json(&gap),
        Err(ClassifyError::BadConfig(_))
    ));
}

#[test]
fn describes_requires_both_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert!(!describes(dir.path()));
    std::fs::write(dir.path().join("config.json"), GOLDENS_CONFIG).expect("write");
    assert!(
        !describes(dir.path()),
        "config.json alone is not a checkpoint"
    );
    std::fs::write(dir.path().join("model.safetensors"), b"").expect("write");
    assert!(describes(dir.path()));
    let bert = config_with(r#""model_type":"distilbert""#, r#""model_type":"bert""#);
    std::fs::write(dir.path().join("config.json"), bert).expect("write");
    assert!(!describes(dir.path()));
    let encoder = config_with(
        r#""architectures":["DistilBertForSequenceClassification"]"#,
        r#""architectures":["DistilBertModel"]"#,
    );
    std::fs::write(dir.path().join("config.json"), encoder).expect("write");
    assert!(!describes(dir.path()), "a bare encoder has no head");
}

#[test]
fn weights_report_missing_tensor() {
    let dir = tempfile::tempdir().expect("tempdir");
    let only = vec![("classifier.weight".to_owned(), "F16", vec![3_u64, 4])];
    let store = store_of(dir.path(), &only);
    match validate(&store, &tiny()) {
        Err(ClassifyError::MissingTensor(name)) => {
            assert_eq!(name, "distilbert.embeddings.word_embeddings.weight");
        }
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

#[test]
fn weights_report_missing_layer_tensor() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = tiny();
    let name = "distilbert.transformer.layer.0.ffn.lin2.bias";
    let all: Vec<Spec> = complete(&cfg).into_iter().filter(|s| s.0 != name).collect();
    match validate(&store_of(dir.path(), &all), &cfg) {
        Err(ClassifyError::MissingTensor(got)) => assert_eq!(got, name),
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

#[test]
fn weights_report_bad_shape() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = tiny();
    let all: Vec<Spec> = complete(&cfg)
        .into_iter()
        .map(|mut s| {
            if s.0 == "classifier.weight" {
                s.2 = vec![2, 4];
            }
            s
        })
        .collect();
    match validate(&store_of(dir.path(), &all), &cfg) {
        Err(ClassifyError::BadTensor { name, why }) => {
            assert_eq!(name, "classifier.weight");
            assert!(why.contains("shape"), "{why}");
        }
        other => panic!("expected BadTensor, got {other:?}"),
    }
}

#[test]
fn weights_report_bad_dtype() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = tiny();
    let all: Vec<Spec> = complete(&cfg)
        .into_iter()
        .map(|mut s| {
            if s.0 == "pre_classifier.bias" {
                s.1 = "U8";
            }
            s
        })
        .collect();
    match validate(&store_of(dir.path(), &all), &cfg) {
        Err(ClassifyError::BadTensor { name, .. }) => assert_eq!(name, "pre_classifier.bias"),
        other => panic!("expected BadTensor, got {other:?}"),
    }
}

#[test]
fn host_table_refuses_short_buffer_and_names_tensor() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = tiny();
    let store = store_of(dir.path(), &complete(&cfg));
    let table =
        super::weights::host_table(&store, "distilbert.embeddings.word_embeddings.weight", 4)
            .expect("table");
    let mut short = [0.0_f32; 3];
    match table.row(0, &mut short) {
        Err(ClassifyError::BadTensor { name, why }) => {
            assert_eq!(name, "distilbert.embeddings.word_embeddings.weight");
            assert!(why.contains("row buffer"), "{why}");
        }
        other => panic!("expected BadTensor, got {other:?}"),
    }
    let mut ok = [1.0_f32; 4];
    table.row(4, &mut ok).expect("last row");
    assert!(ok.iter().all(|v| v.to_bits() == 0));
    assert!(table.row(5, &mut ok).is_err());
}

#[test]
fn complete_checkpoint_validates() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = tiny();
    validate(&store_of(dir.path(), &complete(&cfg)), &cfg).expect("a complete checkpoint passes");
}

/// Host-side check of a converted checkpoint; needs
/// `SHISO_TEST_MODEL_DISTILBERT_CLS` (the model directory).
#[test]
#[ignore = "needs SHISO_TEST_MODEL_DISTILBERT_CLS"]
#[expect(clippy::print_stderr, reason = "a skipped case says why")]
fn real_checkpoint_validates() {
    let Ok(dir) = std::env::var("SHISO_TEST_MODEL_DISTILBERT_CLS") else {
        eprintln!("SKIPPED: SHISO_TEST_MODEL_DISTILBERT_CLS is not set");
        return;
    };
    let dir = Path::new(&dir);
    assert!(describes(dir));
    let text = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = ClassifierConfig::from_json(&text).expect("config parses");
    let store = TensorStore::open_safetensors(&dir.join("model.safetensors")).expect("open");
    validate(&store, &cfg).expect("the converted checkpoint validates");
}

// ---- Classifier::open's refusals, which need no device ----

use std::sync::Arc;

use shiso_core::Backend;

use super::{Classifier, ClassifierDesc, head, tokens};

/// A backend that does nothing: `open` must refuse these cases before any
/// device work.
#[derive(Debug)]
struct NoDevice;

fn refusal(op: &'static str) -> shiso_core::CoreError {
    shiso_core::CoreError::Backend {
        backend: "none",
        op,
        source: Box::new(std::io::Error::other("this backend does nothing")),
    }
}

impl Backend for NoDevice {
    fn backend_name(&self) -> &'static str {
        "none"
    }
    fn alloc_raw(&self, _bytes: u64) -> shiso_core::CoreResult<shiso_core::RawBuffer> {
        Err(refusal("alloc_raw"))
    }
    fn free_raw(&self, _raw: shiso_core::RawBuffer) -> shiso_core::CoreResult<()> {
        Ok(())
    }
    fn copy_h2d(&self, _d: shiso_core::DeviceView<'_>, _s: &[u8]) -> shiso_core::CoreResult<()> {
        Err(refusal("copy_h2d"))
    }
    fn copy_d2h(
        &self,
        _s: shiso_core::DeviceView<'_>,
        _d: &mut [u8],
    ) -> shiso_core::CoreResult<()> {
        Err(refusal("copy_d2h"))
    }
    fn copy_d2d(
        &self,
        _d: shiso_core::DeviceView<'_>,
        _s: shiso_core::DeviceView<'_>,
    ) -> shiso_core::CoreResult<()> {
        Err(refusal("copy_d2d"))
    }
    fn run(
        &self,
        _kernel: shiso_core::KernelRef,
        _args: &shiso_core::LaunchArgs<'_>,
    ) -> shiso_core::CoreResult<()> {
        Err(refusal("run"))
    }
    fn sync(&self) -> shiso_core::CoreResult<()> {
        Err(refusal("sync"))
    }
}

/// A word-piece tokenizer with six ids.
const TINY_TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"~~","max_input_chars_per_word":100,"vocab":{"[UNK]":0,"a":1,"b":2,"c":3,"d":4,"e":5}}}"#;

/// Opens a directory holding the tiny tokenizer and `config`, and no weights.
fn open_without_weights(config: &str) -> Result<Classifier, ClassifyError> {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("tokenizer.json"), TINY_TOKENIZER).expect("write");
    std::fs::write(dir.path().join("config.json"), config).expect("write");
    let backend: Arc<dyn Backend> = Arc::new(NoDevice);
    Classifier::open(&ClassifierDesc {
        dir: dir.path(),
        backend,
    })
}

#[test]
fn open_refuses_a_position_table_shorter_than_the_window() {
    let cfg = config_with(
        r#""max_position_embeddings":512"#,
        r#""max_position_embeddings":64"#,
    );
    match open_without_weights(&cfg) {
        Err(ClassifyError::BadConfig(why)) => assert!(why.contains("max_position_embeddings")),
        other => panic!("expected BadConfig, got {other:?}"),
    }
    // Exactly the window is enough: the failure moves on to the missing weights.
    let cfg = config_with(
        r#""max_position_embeddings":512"#,
        r#""max_position_embeddings":128"#,
    );
    assert!(matches!(
        open_without_weights(&cfg),
        Err(ClassifyError::Io(_))
    ));
}

#[test]
fn open_refuses_a_tokenizer_wider_than_the_word_table() {
    let cfg = config_with(r#""vocab_size":30522"#, r#""vocab_size":5"#);
    match open_without_weights(&cfg) {
        Err(ClassifyError::BadConfig(why)) => assert!(why.contains("vocab_size")),
        other => panic!("expected BadConfig, got {other:?}"),
    }
}

#[test]
fn open_reports_missing_weights_as_io() {
    assert!(matches!(
        open_without_weights(GOLDENS_CONFIG),
        Err(ClassifyError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound
    ));
}

#[test]
fn decide_follows_the_calibration() {
    // Confidently safe, confidently dangerous, an uncertain mix, and a safe
    // class whose energy is too high to trust.
    assert_eq!(decide(&[6.0, -2.0, -4.0]), "approve");
    assert_eq!(decide(&[-4.0, -2.0, 6.0]), "deny");
    assert_eq!(decide(&[1.0, 1.2, 0.8]), "defer");
    assert_eq!(decide(&[0.2, -0.3, -0.1]), "defer");
}

// ---- tokenization, head and the device pass ----

const MODEL_VAR: &str = "SHISO_TEST_MODEL_DISTILBERT_CLS";

/// A golden row: the text, the ids and logits the reference produced, its
/// label and its calibrated decision.
struct Golden {
    text: String,
    ids: Vec<u32>,
    logits: Vec<f32>,
    label: String,
    decision: String,
}

#[expect(clippy::print_stderr, reason = "a skipped case says why on stderr")]
fn model_dir() -> Option<std::path::PathBuf> {
    let dir = std::env::var_os(MODEL_VAR).map(std::path::PathBuf::from);
    if dir.is_none() {
        eprintln!("SKIP: {MODEL_VAR} is unset");
    }
    dir
}

fn goldens(dir: &Path, file: &str) -> Option<Vec<Golden>> {
    let text = std::fs::read_to_string(dir.join(file)).ok()?;
    let rows: Vec<serde_json::Value> = serde_json::from_str(&text).expect("goldens are json");
    Some(
        rows.iter()
            .map(|r| {
                #[expect(clippy::cast_possible_truncation, reason = "f32 goldens")]
                let f = |v: &serde_json::Value| v.as_f64().expect("number") as f32;
                Golden {
                    text: r["text"].as_str().expect("text").to_owned(),
                    ids: r["ids"]
                        .as_array()
                        .expect("ids")
                        .iter()
                        .map(|v| u32::try_from(v.as_u64().expect("id")).expect("u32"))
                        .collect(),
                    logits: r["logits"]
                        .as_array()
                        .expect("logits")
                        .iter()
                        .map(f)
                        .collect(),
                    label: r["label"].as_str().expect("label").to_owned(),
                    decision: r["decision"].as_str().expect("decision").to_owned(),
                }
            })
            .collect(),
    )
}

fn tokenizer() -> Option<tokens::WordPiece> {
    let dir = model_dir()?;
    Some(tokens::WordPiece::load(&dir.join("tokenizer.json")).expect("tokenizer.json loads"))
}

#[test]
fn tokenizer_matches_reference_ids() {
    let Some(dir) = model_dir() else { return };
    let wp = tokens::WordPiece::load(&dir.join("tokenizer.json")).expect("tokenizer.json loads");
    let rows = goldens(&dir, "goldens.json").expect("goldens.json");
    assert!(!rows.is_empty());
    for g in &rows {
        assert_eq!(wp.ids(&g.text).expect("encodes"), g.ids, "{:?}", g.text);
    }
}

#[test]
fn truncates_to_128_with_trailing_sep() {
    let Some(wp) = tokenizer() else { return };
    let long = "echo hello world and then some more words ".repeat(40);
    let ids = wp.ids(&long).expect("encodes");
    assert_eq!(ids.len(), tokens::MAX_TOKENS);
    assert_eq!(ids.first(), Some(&101));
    assert_eq!(ids.last(), Some(&102));
}

#[test]
fn empty_and_whitespace_text() {
    let Some(wp) = tokenizer() else { return };
    assert_eq!(wp.ids("").expect("encodes"), [101, 102]);
    assert_eq!(wp.ids("  \t\n ").expect("encodes"), [101, 102]);
}

#[test]
fn head_math_matches_reference() {
    // dim 4, 3 labels. pre: rows pick x0, x1, x0 - x2, x3 + 1.
    let pre_w = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        1.0, 0.0, -1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    let pre_b = [0.0, 0.0, 0.0, 1.0];
    let x = [2.0, -3.0, 5.0, 0.5];
    // relu(pre x) = [2, 0, 0, 1.5]
    let cls_w = [
        1.0, 1.0, 1.0, 1.0, //
        0.0, 0.0, 0.0, 2.0, //
        -1.0, 0.0, 0.0, 0.0,
    ];
    let cls_b = [0.5, 0.0, 0.25];
    let out = head::apply(&pre_w, &pre_b, &cls_w, &cls_b, &x, (4, 3)).expect("shapes agree");
    assert!(head::apply(&pre_w, &pre_b, &cls_w[..11], &cls_b, &x, (4, 3)).is_err());
    // [2 + 1.5 + 0.5, 3.0, -2 + 0.25]
    for (got, want) in out.iter().zip([4.0_f32, 3.0, -1.75]) {
        assert!((got - want).abs() < 1e-6, "{out:?}");
    }
}

/// The caller's policy, reproduced here to compare decisions: a score above
/// the calibration's 95th-percentile energy defers; otherwise the class with
/// the top temperature-scaled probability decides: `safe` over 0.90 approves,
/// `dangerous` denies, and everything else defers.
fn decide(logits: &[f32]) -> &'static str {
    const TEMPERATURE: f64 = 1.336_620_330_810_546_9;
    const ENERGY_P95: f64 = -2.543_365_240_097_046;
    const SAFE_THRESHOLD: f64 = 0.90;
    let z: Vec<f64> = logits.iter().map(|l| f64::from(*l) / TEMPERATURE).collect();
    let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = z.iter().map(|v| (v - top).exp()).sum();
    let energy = -(TEMPERATURE * (top + sum.ln()));
    if energy > ENERGY_P95 {
        return "defer";
    }
    let probs: Vec<f64> = z.iter().map(|v| (v - top).exp() / sum).collect();
    let best = probs
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |acc, (i, p)| {
            if *p > acc.1 { (i, *p) } else { acc }
        })
        .0;
    match best {
        0 if probs[0] > SAFE_THRESHOLD => "approve",
        2 => "deny",
        _ => "defer",
    }
}

fn open_classifier(dir: &Path, backend: Arc<dyn Backend>) -> Classifier {
    Classifier::open(&ClassifierDesc { dir, backend }).expect("the classifier opens")
}

#[test]
#[ignore = "requires a Metal or CUDA device and SHISO_TEST_MODEL_DISTILBERT_CLS"]
fn real_checkpoint_loads_on_device() {
    let (Some(dir), Some(backend)) = (model_dir(), crate::audio_ops::testing::device()) else {
        return;
    };
    let text = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = ClassifierConfig::from_json(&text).expect("config");
    let store = TensorStore::open_safetensors(&dir.join("model.safetensors")).expect("open");
    let ops = crate::audio_ops::Ops::new(backend);
    let weights = super::ClassifierWeights::load(&ops, &store, &cfg).expect("load");
    assert_eq!(weights.layers.len(), cfg.layers);
}

#[test]
#[ignore = "requires a Metal or CUDA device and SHISO_TEST_MODEL_DISTILBERT_CLS"]
#[expect(clippy::print_stderr, reason = "the measured numbers are the report")]
fn logits_match_reference_goldens() {
    let (Some(dir), Some(backend)) = (model_dir(), crate::audio_ops::testing::device()) else {
        return;
    };
    let rows = goldens(&dir, "goldens.json").expect("goldens.json");
    let clf = open_classifier(&dir, backend);
    assert_eq!(clf.labels(), ["safe", "needs-approval", "dangerous"]);
    // Logits of the f32 host reference run on the converted weights.
    let host = goldens(&dir, "goldens_numpy.json");
    let host_note = |worst: f32| {
        if host.is_some() {
            format!("{worst:.4}")
        } else {
            "skipped (no goldens_numpy.json)".to_owned()
        }
    };
    let (mut worst_int8, mut worst_host) = (0.0_f32, 0.0_f32);
    let (mut agree, mut false_approvals) = (0, 0);
    for (i, g) in rows.iter().enumerate() {
        let got = clf.logits(&g.text).expect("logits");
        let d_int8 = got
            .iter()
            .zip(&g.logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        // The reference is the int8 model; this runs the fp16 conversion of it.
        // The conversion alone moves logits by up to 0.51 (the f32 host reference run on the
        // converted weights vs the int8 goldens), so this bound is loose on
        // purpose. The 0.05 bound against the f32 host reference below is the
        // check on the port.
        assert!(
            d_int8 < 0.6,
            "{:?}: max |dlogit| {d_int8} vs the int8 reference",
            g.text
        );
        worst_int8 = worst_int8.max(d_int8);
        if let Some(n) = host.as_ref().map(|n| &n[i]) {
            assert_eq!(n.text, g.text);
            let d = got
                .iter()
                .zip(&n.logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(
                d < 0.05,
                "{:?}: max |dlogit| {d} vs the f32 host reference",
                g.text
            );
            worst_host = worst_host.max(d);
        }
        let ours = decide(&got);
        agree += usize::from(ours == g.decision);
        if ours == "approve" && g.decision != "approve" && g.label != "safe" && g.label != "edge" {
            false_approvals += 1;
        }
    }
    let pct = 100.0 * f64::from(u32::try_from(agree).expect("small"))
        / f64::from(u32::try_from(rows.len()).expect("small"));
    eprintln!(
        "rows {} max|dlogit| int8 {worst_int8:.4} host {} decisions {agree}/{} ({pct:.2}%) false approvals {false_approvals}",
        rows.len(),
        host_note(worst_host),
        rows.len()
    );
    assert!(pct >= 97.0, "3-way decision agreement {pct:.2}%");
    assert_eq!(false_approvals, 0);
}

#[test]
#[ignore = "requires a Metal or CUDA device and SHISO_TEST_MODEL_DISTILBERT_CLS"]
fn concurrent_logits_equal_serial() {
    let (Some(dir), Some(backend)) = (model_dir(), crate::audio_ops::testing::device()) else {
        return;
    };
    let clf = Arc::new(open_classifier(&dir, backend));
    let texts: Vec<String> = goldens(&dir, "goldens.json")
        .expect("goldens.json")
        .into_iter()
        .take(40)
        .map(|g| g.text)
        .collect();
    assert_eq!(texts.len(), 40);
    let serial: Vec<_> = texts
        .iter()
        .map(|t| clf.logits(t).expect("logits"))
        .collect();
    // The work runs on its own thread so a deadlock fails the test after 60 s
    // instead of hanging the run.
    let (done, finished) = std::sync::mpsc::channel();
    let worker = {
        let clf = Arc::clone(&clf);
        std::thread::spawn(move || {
            std::thread::scope(|s| {
                for _ in 0..4 {
                    s.spawn(|| {
                        for (t, want) in texts.iter().zip(&serial) {
                            let got = clf.logits(t).expect("logits");
                            assert!(
                                got.iter()
                                    .zip(want)
                                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                                "{t:?}: {got:?} differs from the serial {want:?}"
                            );
                        }
                    });
                }
            });
            let _ = done.send(());
        })
    };
    finished
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("four threads finish within 60 s (no deadlock, no panic)");
    worker.join().expect("no worker panicked");
}

#[test]
#[ignore = "requires a Metal or CUDA device and SHISO_TEST_MODEL_DISTILBERT_CLS"]
fn empty_text_runs() {
    let (Some(dir), Some(backend)) = (model_dir(), crate::audio_ops::testing::device()) else {
        return;
    };
    let clf = open_classifier(&dir, backend);
    assert_eq!(clf.token_ids("").expect("encodes"), [101, 102]);
    let empty = clf.logits("").expect("empty text runs");
    assert!(empty.iter().all(|l| l.is_finite()), "{empty:?}");
    // Whitespace tokenizes to the same two markers, so it answers the same.
    let blank = clf.logits("  \t\n ").expect("blank text runs");
    assert_eq!(empty.map(f32::to_bits), blank.map(f32::to_bits));
}

#[test]
#[ignore = "requires a Metal or CUDA device and SHISO_TEST_MODEL_DISTILBERT_CLS"]
fn very_long_text_is_truncated() {
    let (Some(dir), Some(backend)) = (model_dir(), crate::audio_ops::testing::device()) else {
        return;
    };
    let clf = open_classifier(&dir, backend);
    let long = "echo hello world and then some more words ".repeat(250);
    assert!(long.len() >= 10_000);
    let ids = clf.token_ids(&long).expect("encodes");
    assert_eq!(ids.len(), tokens::MAX_TOKENS);
    assert_eq!(ids.last(), Some(&102));
    let got = clf.logits(&long).expect("a 10 000 character text runs");
    assert!(got.iter().all(|l| l.is_finite()), "{got:?}");
    // Truncation is the tokenizer's, so ids cannot be decoded back to an exact
    // prefix text (WordPiece does not round-trip). The equivalent check: a
    // different text with the same first 126 content tokens has identical ids,
    // hence identical logits, and the dropped tail has no influence.
    let other = format!("{long}rm -rf / --no-preserve-root");
    assert_eq!(clf.token_ids(&other).expect("encodes"), ids);
    let again = clf.logits(&other).expect("logits");
    assert_eq!(got.map(f32::to_bits), again.map(f32::to_bits));
    assert_eq!(
        got.map(f32::to_bits),
        clf.logits(&long).expect("logits").map(f32::to_bits)
    );
}

/// Median of `calls` timed runs of `f`, in milliseconds.
fn median_ms(calls: usize, mut f: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..calls)
        .map(|_| {
            let start = std::time::Instant::now();
            f();
            start.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[calls / 2]
}

/// Startup and steady-state cost, recorded in the module docs. Run with
/// `--release`; a debug build measures nothing useful.
#[test]
#[ignore = "requires a Metal or CUDA device and SHISO_TEST_MODEL_DISTILBERT_CLS; run with --release -- --nocapture"]
#[expect(clippy::print_stderr, reason = "the measured numbers are the report")]
fn measure_latency() {
    let Some(dir) = model_dir() else { return };
    let start = std::time::Instant::now();
    let Some(backend) = crate::audio_ops::testing::device() else {
        return;
    };
    let clf = open_classifier(&dir, backend);
    let short = "grep -rn TODO src | sort | head -n 5";
    let long = "echo hello world and then some more words ".repeat(40);
    let first = clf.logits(short).expect("logits");
    let startup = start.elapsed().as_secs_f64();
    assert!(first.iter().all(|l| l.is_finite()));
    let n_short = clf.token_ids(short).expect("encodes").len();
    let n_long = clf.token_ids(&long).expect("encodes").len();
    let ms_short = median_ms(100, || {
        clf.logits(short).expect("logits");
    });
    let ms_long = median_ms(100, || {
        clf.logits(&long).expect("logits");
    });
    eprintln!(
        "backend creation to first logits {startup:.2} s; median of 100 calls: {n_short} ids {ms_short:.2} ms, {n_long} ids {ms_long:.2} ms"
    );
}
