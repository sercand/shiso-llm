//! Pins the container dispatch, with no model on disk.
//!
//! [`ModelFile::open`] decides which loader to run from the path alone, and the
//! decision has to be wrong in a legible way when the path is wrong: a
//! directory that is not a checkpoint must complain about the file it is
//! missing, and a file that is not a checkpoint must complain about the header
//! it does not have. A dispatch that fell back to the other loader on failure
//! would turn both of those into the same unhelpful message.

use shiso_modelfile::{GgufError, ModelFile};

#[test]
fn a_file_that_is_not_a_checkpoint_is_rejected_by_its_header() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("not-a-model.gguf");
    std::fs::write(&path, vec![0_u8; 64]).expect("the fixture is writable");

    let err = ModelFile::open(&path).expect_err("sixty-four zero bytes are not a model");
    assert!(
        matches!(err, GgufError::NotGguf),
        "expected a header rejection, got {err}"
    );
}

#[test]
fn a_file_too_short_to_hold_a_header_is_rejected() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("stub.gguf");
    std::fs::write(&path, b"GGUF").expect("the fixture is writable");

    let err = ModelFile::open(&path).expect_err("four bytes are not a model");
    assert!(
        matches!(err, GgufError::Truncated { .. }),
        "expected a truncation, got {err}"
    );
}

#[test]
fn a_directory_without_a_config_names_the_file_it_wanted() {
    let dir = tempfile::tempdir().expect("a temporary directory");

    let err = ModelFile::open(dir.path()).expect_err("an empty directory is not a checkpoint");
    let message = err.to_string();
    assert!(
        message.contains("config.json"),
        "the error should name the missing file, got {message}"
    );
    assert!(
        message.contains(&dir.path().display().to_string()),
        "the error should name the directory, got {message}"
    );
}

#[test]
fn a_missing_path_is_reported_rather_than_guessed_at() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = dir.path().join("nothing-here.gguf");

    let err = ModelFile::open(&path).expect_err("there is no such file");
    assert!(
        matches!(err, GgufError::Io(_)),
        "expected an I/O failure, got {err}"
    );
}

fn write_shard(path: &std::path::Path, tensors: &[(&str, &str, &[u64], &[u64])], payload: &[u8]) {
    let mut header_map = serde_json::Map::new();
    header_map.insert(
        "__metadata__".to_string(),
        serde_json::json!({ "format": "mlx" }),
    );
    for &(name, dtype, shape, offsets) in tensors {
        header_map.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": offsets,
            }),
        );
    }
    let header_json = serde_json::to_string(&header_map).expect("serialize header");
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;
    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&header_len.to_le_bytes());
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.extend_from_slice(payload);
    std::fs::write(path, file_bytes).expect("write shard");
}

#[test]
fn affine_triple_returns_the_three_payloads_and_their_layout() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    // Shard 1: a.weight (U32 [1, 8]), a.scales (BF16 [1, 1]), a.biases (BF16 [1, 1])
    // a.weight: 1 * 8 = 8 u32 words = 32 bytes (data_offsets: [0, 32])
    // a.scales: 1 * 1 = 1 bf16 = 2 bytes (data_offsets: [32, 34])
    // a.biases: 1 * 1 = 1 bf16 = 2 bytes (data_offsets: [34, 36])
    let mut payload1 = vec![0_u8; 36];
    payload1[0..4].copy_from_slice(&0x1234_5678_u32.to_le_bytes());
    write_shard(
        &dir.path().join("model-00001-of-00002.safetensors"),
        &[
            ("a.weight", "U32", &[1, 8], &[0, 32]),
            ("a.scales", "BF16", &[1, 1], &[32, 34]),
            ("a.biases", "BF16", &[1, 1], &[34, 36]),
        ],
        &payload1,
    );

    // Shard 2: b.weight (BF16 [1, 4]) -> 8 bytes (data_offsets: [0, 8])
    let payload2 = vec![0_u8; 8];
    write_shard(
        &dir.path().join("model-00002-of-00002.safetensors"),
        &[("b.weight", "BF16", &[1, 4], &[0, 8])],
        &payload2,
    );

    // Index file
    let index_json = serde_json::json!({
        "metadata": { "total_size": 56 },
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "a.scales": "model-00001-of-00002.safetensors",
            "a.biases": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors"
        }
    });
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        index_json.to_string(),
    )
    .expect("write index");

    // Config file with quantization map
    let config_json = serde_json::json!({
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "x.mlp.gate_proj": {
                "bits": 8,
                "group_size": 64
            }
        }
    });
    std::fs::write(dir.path().join("config.json"), config_json.to_string()).expect("write config");

    let file = shiso_modelfile::SafetensorsFile::open(dir.path()).expect("the checkpoint opens");
    assert_eq!(file.bits_for("x.mlp.gate_proj"), 8);
    assert_eq!(file.bits_for("other"), 4);

    let triple = file
        .affine_triple("a.weight")
        .expect("query succeeds")
        .expect("a is affine");
    assert_eq!(triple.layout.group_size, 64);
    assert_eq!(triple.layout.bits, 4);
    assert_eq!(triple.words.len(), 32);

    let b = file.affine_triple("b.weight").expect("query succeeds");
    assert!(b.is_none());
}
