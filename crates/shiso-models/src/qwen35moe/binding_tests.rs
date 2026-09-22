//! The expected-tensor list, and the export check over the real checkpoint.
//!
//! The list cases need no checkpoint: the list is derived from the shape, and
//! the shape comes from a config fixture that is checked in. Only the two cases
//! that ask what the real export CONTAINS need the 23.4 GB snapshot, and they
//! say out loud when they do not have it; the refusals that fire before any
//! tensor is consulted run against a synthetic one-shard container instead, so
//! that a refusal is not checked on one machine only.

use std::path::{Path, PathBuf};

use shiso_core::config::HfConfig;
use shiso_modelfile::modelopt::ModeloptFile;

use super::{TEXT_ROOTS, check_export, expected_tensors};
use crate::qwen35moe::{QuantConfig, Qwen35MoeConfig};

/// The checked-in fixtures.
fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors")
}

/// The family's config, read from the checked-in documents.
fn config() -> Qwen35MoeConfig {
    let doc =
        HfConfig::from_file(&vectors().join("loader/qwen35moe-config.json")).expect("the document");
    let quant = QuantConfig::from_file(&vectors().join("loader/qwen35moe-quant-config.json"))
        .expect("the quant");
    Qwen35MoeConfig::from_hf(&doc, quant).expect("a checked config")
}

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// The real checkpoint, or `None` when it is not on this machine.
fn snapshot() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_QWEN35MOE").ok()?);
    path.join("model.safetensors.index.json")
        .is_file()
        .then_some(path)
}

#[test]
fn the_list_is_the_geometry_counted_out() {
    let cfg = config();
    let names = expected_tensors(cfg.shape());

    // 30 recurrent slots at 784 tensors each, 10 attention slots at 781, plus
    // the embedding table, the final norm and the output projection. The two
    // slot figures differ by three because an attention slot carries six
    // tensors where a recurrent one carries nine.
    assert_eq!(2 + 9 + 2 + 3 + 256 * 3, 784);
    assert_eq!(2 + 6 + 2 + 3 + 256 * 3, 781);
    assert_eq!(30 * 784 + 10 * 781 + 3, 31_333);
    assert_eq!(names.len(), 31_333);

    // Every name is distinct. At 31,333 names built from a template, a
    // transposed index is the mistake that keeps the count.
    let distinct: std::collections::HashSet<&String> = names.iter().collect();
    assert_eq!(distinct.len(), names.len());
}

#[test]
fn the_list_names_each_kind_of_slot_the_way_the_export_does() {
    let names = expected_tensors(config().shape());
    let has = |name: &str| names.iter().any(|kept| kept == name);

    // Slot 0 is recurrent, slot 3 is attention: the interval is four and the
    // attention slot is the last of each run.
    assert!(has("model.language_model.layers.0.linear_attn.in_proj_qkv"));
    assert!(!has("model.language_model.layers.0.self_attn.q_proj"));
    assert!(has("model.language_model.layers.3.self_attn.q_proj"));
    assert!(!has(
        "model.language_model.layers.3.linear_attn.in_proj_qkv"
    ));

    // The two recurrent parameters that carry no `.weight` in the export, and
    // which an earlier draft of the container silently dropped.
    assert!(has("model.language_model.layers.0.linear_attn.A_log"));
    assert!(has("model.language_model.layers.0.linear_attn.dt_bias"));

    // The feed-forward, which every slot has whatever block sits above it.
    assert!(has("model.language_model.layers.3.mlp.gate"));
    assert!(has("model.language_model.layers.3.mlp.shared_expert_gate"));
    assert!(has(
        "model.language_model.layers.3.mlp.shared_expert.down_proj"
    ));
    assert!(has(
        "model.language_model.layers.39.mlp.experts.255.down_proj"
    ));
    assert!(!has(
        "model.language_model.layers.39.mlp.experts.256.down_proj"
    ));

    // The three that sit outside every slot.
    assert!(has("model.language_model.embed_tokens"));
    assert!(has("model.language_model.norm"));
    assert!(has("lm_head"));

    // And nothing from the two towers this family does not load.
    assert!(!names.iter().any(|name| name.starts_with("model.visual.")));
    assert!(!names.iter().any(|name| name.starts_with("mtp")));
}

#[test]
fn the_real_export_is_complete_and_stored_as_its_own_document_claims() {
    let Some(dir) = snapshot() else {
        skip("SHISO_TEST_MODEL_QWEN35MOE names no checkpoint directory");
        return;
    };
    let file = ModeloptFile::open(&dir, &TEXT_ROOTS).expect("the language model");
    check_export(&file, &config()).expect("a complete export stored as declared");
}

/// A one-shard modelopt directory holding a single irrelevant tensor.
///
/// Enough of a container to call [`check_export`] with, and no more: the
/// refusal under test fires before a single tensor is consulted, so this needs
/// no checkpoint and runs everywhere. The header is the shape
/// `crates/shiso-modelfile/src/modelopt_tests.rs` writes its own fixtures in - an
/// eight-byte little-endian length, the JSON, then the payload.
fn empty_export() -> tempfile::TempDir {
    let header = r#"{"__metadata__":{"format":"pt"},
        "unrelated.weight":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}"#;
    let dir = tempfile::tempdir().expect("a temporary directory");
    let mut bytes = u64::try_from(header.len())
        .expect("fits")
        .to_le_bytes()
        .to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0u8; 8]);
    std::fs::write(dir.path().join("model.safetensors"), bytes).expect("the shard");
    dir
}

#[test]
fn an_export_cannot_be_checked_against_a_config_that_carries_no_quant_document() {
    // A config read from a Splash package has no `hf_quant_config.json` behind
    // it. Before this was an `Option` it was an empty map, which answered
    // "stored at its declared element type" for all 31,333 tensors and passed
    // every one of them unchecked.
    let text =
        std::fs::read_to_string(vectors().join("loader/qwen35moe-config.json")).expect("fixture");
    let document: serde_json::Value = serde_json::from_str(&text).expect("json");
    let without = Qwen35MoeConfig::from_json(&document).expect("the same widths");
    assert!(without.quant().is_none());

    // The synthetic container rather than the 23.4 GB snapshot, because this
    // refusal is reached before any tensor is looked up: a case that can only
    // run on one machine is a refusal that is checked on no machine.
    let dir = empty_export();
    let file = ModeloptFile::open(dir.path(), &TEXT_ROOTS).expect("a container with no text roots");
    match check_export(&file, &without).expect_err("nothing says how these are stored") {
        shiso_core::error::CoreError::ConfigMissing { key } => {
            assert_eq!(key, "hf_quant_config.json");
        }
        other => panic!("expected ConfigMissing, got {other:?}"),
    }

    // And the same container against a config that DOES carry the document gets
    // past the quantization check to the tensor it is actually missing, which
    // is what says the refusal above is about the document and not about the
    // fixture being empty.
    match check_export(&file, &config()).expect_err("the fixture holds no model") {
        shiso_core::error::CoreError::InvalidValue { key, .. } => {
            assert_eq!(key, "missing tensor");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

#[test]
fn an_export_short_one_tensor_is_refused_by_that_tensor_s_name() {
    let Some(dir) = snapshot() else {
        skip("SHISO_TEST_MODEL_QWEN35MOE names no checkpoint directory");
        return;
    };
    // The whole point of the check: a checkpoint missing a single expert binds
    // cleanly under any per-tensor lookup and answers wrongly. Simulated by
    // asking the container for a narrower slice than the family needs, which is
    // the same thing from the check's point of view.
    let file =
        ModeloptFile::open(&dir, &["model.language_model.layers.0."]).expect("one slot's worth");
    let err = check_export(&file, &config()).expect_err("one slot is not forty");
    match err {
        shiso_core::error::CoreError::InvalidValue { key, value } => {
            assert_eq!(key, "missing tensor");
            assert_eq!(value, "model.language_model.embed_tokens");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

#[test]
fn the_declared_fp8_cache_ships_without_the_scales_it_would_need() {
    let Some(dir) = snapshot() else {
        skip("SHISO_TEST_MODEL_QWEN35MOE names no checkpoint directory");
        return;
    };
    // A fact Phase 8 Part C's Task C5 did not have when it ruled on this, and
    // one that strengthens the ruling rather than reopening it.
    //
    // `hf_quant_config.json` declares `kv_cache_quant_algo: FP8`. An engine
    // that honoured it needs a scale to quantize by, and the usual place for
    // one is a `k_scale`/`v_scale` tensor per attention slot. **This export
    // ships none** - the only `*_scale` tensors under `self_attn` are the four
    // projections' own weight and activation scales. So the declaration does
    // not determine what the reference does: an engine honouring it has to
    // calibrate at runtime or default to a scale of one, and those are two
    // different references.
    //
    // This build serves `f16`, which is a strict refinement of either.
    let file = ModeloptFile::open(&dir, &TEXT_ROOTS).expect("the language model");
    let cache_scales: Vec<&str> = file
        .tensors()
        .map(shiso_modelfile::modelopt::ModeloptTensor::name)
        .filter(|name| {
            name.ends_with(".k_scale") || name.ends_with(".v_scale") || name.contains("kv_scale")
        })
        .collect();
    assert!(
        cache_scales.is_empty(),
        "the export ships cache scales after all: {cache_scales:?}"
    );

    // And the four that DO exist under an attention slot are the projections'.
    let slot: Vec<&str> = file
        .tensors()
        .map(shiso_modelfile::modelopt::ModeloptTensor::name)
        .filter(|name| name.contains("layers.3.self_attn"))
        .collect();
    assert_eq!(slot.len(), 6, "{slot:?}");
}
