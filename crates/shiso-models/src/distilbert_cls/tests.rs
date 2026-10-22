//! Host-only tests of the config reader and the weight validation, plus an
//! ignored check of a real checkpoint. Nothing here opens a device.

use std::path::Path;

use super::weights::expected;
use super::{ClassifierConfig, ClassifyError, describes, validate};
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
    assert_eq!(ok, [0.0; 4]);
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
