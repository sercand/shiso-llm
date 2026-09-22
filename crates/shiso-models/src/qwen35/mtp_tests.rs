//! What the sidecar says about the draft head, and what the head's container
//! turns out to hold.

use std::path::PathBuf;

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
    assert_eq!(bound.fc.shape(), &[2560, 5120]);
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
