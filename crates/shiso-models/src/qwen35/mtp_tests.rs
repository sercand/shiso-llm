//! What the sidecar says about the draft head, and what the head's container
//! turns out to hold.

use std::path::{Path, PathBuf};

use super::*;

/// The sidecar directory this family's head is read from, or a skip.
///
/// A skipped case and a passing one look identical in cargo's output, so every
/// skip here says so and the reader counts cases rather than reading the
/// colour.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn sidecar() -> Option<PathBuf> {
    let Ok(dir) = std::env::var("SHISO_TEST_MODEL_QWEN35_SIDECAR") else {
        println!("SKIP: SHISO_TEST_MODEL_QWEN35_SIDECAR names no directory");
        return None;
    };
    Some(PathBuf::from(dir))
}

/// The sidecar's own document names the head and its codec.
///
/// The five keys are read off `$SHISO_TEST_MODEL_OPTIQ/config.json`
/// on 2026-09-21, verbatim.
#[test]
fn the_sidecar_document_names_the_head() {
    let Some(dir) = sidecar() else { return };
    let manifest = read_manifest(&dir)
        .expect("the document reads")
        .expect("this checkpoint ships a head");
    assert_eq!(manifest.file, dir.join("optiq/mtp.safetensors"));
    assert_eq!(manifest.layers, 1);
    assert_eq!(manifest.quant.bits, 4);
    assert_eq!(manifest.quant.group_size, 64);
    assert!(manifest.quant.prequantized);
}

/// A checkpoint with no head is not an error.
#[test]
fn a_document_with_no_head_reads_as_none() {
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"qwen3_5"}"#,
    )
    .expect("the document writes");
    assert!(
        read_manifest(dir.path())
            .expect("the document reads")
            .is_none()
    );
}

/// A head the document names and the directory does not hold is a refusal that
/// says which path was looked at.
///
/// Not `None`: `None` means "this checkpoint ships no head", and answering it
/// for a checkpoint that says it does would turn a missing 86.7 MB file into a
/// server that quietly decodes one token a step.
#[test]
fn a_named_head_that_is_absent_names_the_path_it_looked_at() {
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"qwen3_5","mtp_file":"optiq/mtp.safetensors"}"#,
    )
    .expect("the document writes");
    let error = read_manifest(dir.path()).expect_err("an absent head refuses");
    let text = error.to_string();
    assert!(
        text.contains("optiq/mtp.safetensors"),
        "the refusal names the path it looked at, got {text:?}"
    );
}

/// A head whose codec the document does not describe falls back to the
/// container's own default, which is what `mtp_policy` already spells.
#[test]
fn a_document_with_no_codec_subtree_takes_the_family_default() {
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::create_dir_all(dir.path().join("optiq")).expect("the subdirectory");
    std::fs::write(dir.path().join("optiq/mtp.safetensors"), [0u8; 8])
        .expect("a placeholder head writes");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"qwen3_5","mtp_file":"optiq/mtp.safetensors"}"#,
    )
    .expect("the document writes");
    let manifest = read_manifest(dir.path())
        .expect("the document reads")
        .expect("this document ships a head");
    assert_eq!(manifest.layers, 1);
    assert_eq!(manifest.quant, AffineQuant::default());
}

/// Every tensor the head ships is bound, and none of the target's is.
#[test]
#[ignore = "requires the sidecar's 86.7 MB head"]
fn the_head_binds_twenty_nine_tensors() {
    let Some(dir) = sidecar() else { return };
    let manifest = read_manifest(&dir).expect("reads").expect("ships a head");
    let bound = MtpWeights::bind(&manifest).expect("the head binds");
    assert_eq!(bound.tensor_count(), 29);
    assert_eq!(bound.fc.logical_shape().expect("rank two"), [2560, 5120]);
    assert_eq!(bound.layer.mlp.gate_proj.packed_shape(), &[9216, 320]);
    assert_eq!(bound.layer.self_attn.q_norm.shape(), &[256]);
}

/// The head's affine triples are read at the codec the manifest states, and the
/// layout each one derives agrees with the packed word count in the file.
///
/// The check that matters is `MlxAffineLayout::elements`: a triple whose scales
/// row count does not divide its input width is a group size read from the
/// wrong place, and every dequantization after it succeeds and is wrong.
#[test]
#[ignore = "requires the sidecar's 86.7 MB head"]
fn every_packed_tensor_agrees_with_its_declared_group_size() {
    let Some(dir) = sidecar() else { return };
    let manifest = read_manifest(&dir).expect("reads").expect("ships a head");
    let bound = MtpWeights::bind(&manifest).expect("the head binds");
    for packed in bound.packed() {
        let layout = packed.layout();
        assert_eq!(
            layout.group_size,
            usize::try_from(manifest.quant.group_size).expect("a group size that fits"),
            "{} was read at another group size",
            packed.name()
        );
        assert!(
            layout.elements().is_some(),
            "{} has a layout that names no element count",
            packed.name()
        );
    }
}

/// Writes a safetensors container of zeros: `(name, dtype, shape)` per tensor.
fn write_container(path: &Path, tensors: &[(String, &str, Vec<u64>)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, dtype, shape) in tensors {
        let width = if *dtype == "U32" { 4 } else { 2 };
        let bytes = shape.iter().product::<u64>() * width;
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        offset += bytes;
    }
    let mut text = serde_json::Value::Object(header).to_string();
    while !text.len().is_multiple_of(8) {
        text.push(' ');
    }
    let mut file = Vec::new();
    file.extend_from_slice(&u64::try_from(text.len()).expect("fits").to_le_bytes());
    file.extend_from_slice(text.as_bytes());
    file.resize(file.len() + usize::try_from(offset).expect("fits"), 0);
    std::fs::write(path, file).expect("the container writes");
}

/// The tensors of a miniature head, `hidden` 64 wide, at 4 bits and group 64.
///
/// `prefix` is the namespace the names are stored under, and `packed_fc`
/// chooses between the two forms the pair projection ships in.
fn miniature_head(prefix: &str, packed_fc: bool) -> Vec<(String, &'static str, Vec<u64>)> {
    let hidden = 64u64;
    let mut out = Vec::new();
    let mut triple = |name: &str, rows: u64, cols: u64| {
        out.push((
            format!("{prefix}{name}.weight"),
            "U32",
            vec![rows, cols / 8],
        ));
        out.push((
            format!("{prefix}{name}.scales"),
            "BF16",
            vec![rows, cols / 64],
        ));
        out.push((
            format!("{prefix}{name}.biases"),
            "BF16",
            vec![rows, cols / 64],
        ));
    };
    if packed_fc {
        triple("fc", hidden, hidden * 2);
    }
    for (name, rows, cols) in [
        ("layers.0.self_attn.q_proj", hidden * 2, hidden),
        ("layers.0.self_attn.k_proj", hidden, hidden),
        ("layers.0.self_attn.v_proj", hidden, hidden),
        ("layers.0.self_attn.o_proj", hidden, hidden),
        ("layers.0.mlp.gate_proj", hidden * 2, hidden),
        ("layers.0.mlp.up_proj", hidden * 2, hidden),
        ("layers.0.mlp.down_proj", hidden, hidden * 2),
    ] {
        triple(name, rows, cols);
    }
    if !packed_fc {
        out.push((
            format!("{prefix}fc.weight"),
            "BF16",
            vec![hidden, hidden * 2],
        ));
    }
    for name in [
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
        "norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
    ] {
        out.push((format!("{prefix}{name}"), "BF16", vec![hidden]));
    }
    out
}

/// Binds a miniature head written under `prefix`, with `fc` in either form.
fn bind_miniature(prefix: &str, packed_fc: bool) -> CoreResult<MtpWeights> {
    let dir = tempfile::tempdir().expect("a temp dir");
    let file = dir.path().join("head.safetensors");
    write_container(&file, &miniature_head(prefix, packed_fc));
    MtpWeights::bind(&MtpManifest {
        file,
        layers: 1,
        quant: AffineQuant::family_default(),
    })
}

/// The sidecar's layout: `mtp.` names and a BF16 `fc`, 29 tensors.
#[test]
fn a_prefixed_head_with_a_half_fc_binds() {
    let bound = bind_miniature("mtp.", false).expect("the head binds");
    assert_eq!(bound.tensor_count(), 29);
    assert!(matches!(bound.fc, FcMatrix::Half(_)));
    assert_eq!(bound.fc.logical_shape().expect("rank two"), [64, 128]);
}

/// A head published on its own: bare names and a packed `fc`, 31 tensors.
#[test]
fn a_bare_head_with_a_packed_fc_binds() {
    let bound = bind_miniature("", true).expect("the head binds");
    assert_eq!(bound.tensor_count(), 31);
    let FcMatrix::Packed(fc) = &bound.fc else {
        panic!("fc should bind packed");
    };
    assert_eq!(fc.name(), "fc.weight");
    assert_eq!(bound.fc.logical_shape().expect("rank two"), [64, 128]);
}

/// A namespace other than the two this family ships under refuses by name.
#[test]
fn a_head_under_another_namespace_refuses() {
    let error = bind_miniature("model.mtp.", true).expect_err("refuses");
    assert!(
        error.to_string().contains("fc.weight"),
        "the refusal names a tensor it looked for, got {error}"
    );
}
