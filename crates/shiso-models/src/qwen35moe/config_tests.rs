//! The config reader, over the real documents.
//!
//! The two fixtures are `Qwen3.6-35B-A3B-NVFP4`'s own
//! `config.json` and `hf_quant_config.json`, copied verbatim into
//! `test-vectors/`. They are checked in rather than read out of the 23.4 GB
//! snapshot for one reason: a case that skips when a checkpoint is absent and a
//! case that passes print the same colour, and this plan's own environment note
//! records three lanes on this branch that were found silently skipping.

use std::path::{Path, PathBuf};

use shiso_core::config::HfConfig;
use shiso_core::error::CoreError;
use shiso_engine::shape::LayerKind;

use super::splash::Manifest;
use super::{QuantConfig, Qwen35MoeConfig};

/// The checked-in fixtures.
fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors")
}

/// The family's own document.
fn fixture() -> HfConfig {
    let path = vectors().join("loader/qwen35moe-config.json");
    HfConfig::from_file(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// The family's own quantization document.
fn quant() -> QuantConfig {
    let path = vectors().join("loader/qwen35moe-quant-config.json");
    QuantConfig::from_file(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// The config, read and checked.
fn config() -> Qwen35MoeConfig {
    Qwen35MoeConfig::from_hf(&fixture(), quant()).expect("the family's own document")
}

#[test]
fn the_document_is_recognised_and_the_dense_family_refuses_it() {
    let cfg = config();
    assert_eq!(cfg.shape().n_layers(), 40);

    // The dense reader must still say no, by name. It asserts 32 layers and
    // hidden 2560, so it would refuse on a width anyway; what this pins is that
    // it refuses on the architecture, before any width is consulted.
    let err = crate::qwen35::Qwen35Config::from_hf(&fixture())
        .expect_err("a routed hybrid is not the dense one");
    match err {
        CoreError::UnknownArch { model_type } => assert_eq!(model_type, "qwen3_5_moe"),
        other => panic!("expected UnknownArch, got {other:?}"),
    }
}

#[test]
fn every_measured_number_is_what_the_checkpoint_says() {
    let cfg = config();
    let shape = cfg.shape();

    // The stack.
    assert_eq!(shape.n_layers(), 40);
    assert_eq!(shape.hidden, 2048);
    assert_eq!(shape.vocab, 248_320);
    assert_eq!(cfg.full_attention_interval(), 4);

    // Three recurrent slots to one attention slot, and the attention slots are
    // the fourth of each run of four.
    assert_eq!(shape.count(LayerKind::Gdn), 30);
    assert_eq!(shape.count(LayerKind::Attn), 10);

    // The recurrent block, which is byte-for-byte the dense family's geometry
    // and is why this module dispatches that family's entries.
    assert_eq!(shape.gdn.v_heads, 32);
    assert_eq!(shape.gdn.qk_heads, 16);
    assert_eq!(shape.gdn.head_dim, 128);
    assert_eq!(shape.gdn.conv_kernel, 4);

    // The attention block: one descriptor per attention slot, all identical.
    assert_eq!(shape.attn.len(), 10);
    let attn = shape.attn[0];
    assert_eq!(attn.q_heads, 16);
    assert_eq!(attn.kv_heads, 2);
    assert_eq!(attn.head_dim, 256);
    // `partial_rotary_factor` 0.25 over a 256-wide head.
    assert_eq!(attn.n_rot, 64);
    assert!(shape.attn.iter().all(|slot| *slot == attn));
    assert!(shape.attn_output_gate);

    // The router, which is the whole reason this module exists.
    let moe = cfg.moe();
    assert_eq!(moe.experts, 256);
    assert_eq!(moe.used, 8);
    assert_eq!(moe.shared, 1);
    assert_eq!(moe.intermediate, 512);
    // The shared expert IS the slot's dense feed-forward, so its width is the
    // shape's dense one. The two are equal in this checkpoint and are not the
    // same field.
    assert_eq!(shape.intermediate, 512);

    // One routed column per (token, slot).
    assert_eq!(moe.slot_rows(1), 8);
    assert_eq!(moe.slot_rows(512), 4_096);
}

#[test]
fn the_shape_this_reader_builds_validates() {
    config().shape().validate().expect("a coherent shape");
}

#[test]
fn a_turn_ends_on_im_end_and_the_document_s_own_id_rides_behind_it() {
    let cfg = config();
    // `text_config.eos_token_id` is `<|endoftext|>`, which is where a base
    // model stops; a chat turn ends on `<|im_end|>`.
    assert_eq!(cfg.eos_token().get(), 248_044);
    let stops: Vec<u32> = cfg.stop_tokens().iter().map(|id| id.get()).collect();
    assert_eq!(stops, vec![248_046, 248_044]);
}

#[test]
fn the_dense_family_s_document_is_refused_by_name() {
    let path = vectors().join("loader/qwen35-config.json");
    let dense = HfConfig::from_file(&path).expect("the dense family's document");
    let err =
        Qwen35MoeConfig::from_hf(&dense, quant()).expect_err("the dense hybrid is not this family");
    match err {
        CoreError::UnknownArch { model_type } => assert_eq!(model_type, "qwen3_5"),
        other => panic!("expected UnknownArch, got {other:?}"),
    }
}

/// The fixture with one key of `text_config` replaced.
fn with_text_key(key: &str, value: serde_json::Value) -> HfConfig {
    let path = vectors().join("loader/qwen35moe-config.json");
    let text = std::fs::read_to_string(&path).expect("the fixture");
    let mut doc: serde_json::Value = serde_json::from_str(&text).expect("json");
    doc["text_config"][key] = value;
    HfConfig::from_json_str(&serde_json::to_string(&doc).expect("json")).expect("a document")
}

#[test]
fn a_width_the_kernels_were_not_written_for_is_refused_by_name() {
    let doc = with_text_key("num_experts_per_tok", serde_json::json!(10));
    match Qwen35MoeConfig::from_hf(&doc, quant()).expect_err("eight, not ten") {
        CoreError::ShapeMismatch {
            field,
            expected,
            got,
        } => {
            assert_eq!(field, "num_experts_per_tok");
            assert_eq!((expected, got), (8, 10));
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn a_rotary_triple_the_entry_points_do_not_deal_out_is_refused() {
    // `q35_common.cuh` picks a pair's axis with `i % 3`, which at 32 pairs IS
    // [11, 11, 10]. Any other triple rotates every head against the wrong axis
    // and produces a fluent answer to a different prompt, so it is refused here
    // rather than discovered in the logits.
    let doc = with_text_key(
        "rope_parameters",
        serde_json::json!({
            "mrope_interleaved": true,
            "mrope_section": [16, 8, 8],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default"
        }),
    );
    match Qwen35MoeConfig::from_hf(&doc, quant()).expect_err("not the triple i % 3 deals out") {
        CoreError::InvalidValue { key, value } => {
            assert_eq!(key, "rope_parameters.mrope_section");
            assert!(value.contains("[16, 8, 8]"), "{value}");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

#[test]
fn sections_that_are_not_interleaved_are_refused() {
    let doc = with_text_key(
        "rope_parameters",
        serde_json::json!({
            "mrope_interleaved": false,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default"
        }),
    );
    match Qwen35MoeConfig::from_hf(&doc, quant()).expect_err("the entries interleave") {
        CoreError::ShapeMismatch { field, .. } => {
            assert_eq!(field, "rope_parameters.mrope_interleaved");
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

// -- the document alone, with no quantization document beside it ------------

/// The family's own document, as a bare value.
fn document() -> serde_json::Value {
    let text =
        std::fs::read_to_string(vectors().join("loader/qwen35moe-config.json")).expect("fixture");
    serde_json::from_str(&text).expect("json")
}

#[test]
fn the_checkpoints_own_widths_are_accepted_without_a_quantization_document() {
    let cfg = Qwen35MoeConfig::from_json(&document()).expect("the real widths load");
    assert_eq!(cfg.shape().hidden, 2048);
    assert_eq!(cfg.shape().n_layers(), 40);
    assert_eq!(cfg.moe().experts, 256);
    assert_eq!(cfg.moe().used, 8);
    assert_eq!(cfg.moe().intermediate, 512);
    // The same geometry the two-document path builds, and the one difference
    // between them stated rather than assumed.
    assert_eq!(cfg.shape(), config().shape());
    assert!(cfg.quant().is_none());
    assert!(config().quant().is_some());
}

#[test]
fn an_expert_width_the_tile_cannot_divide_is_refused_by_name() {
    let mut doc = document();
    doc["text_config"]["moe_intermediate_size"] = serde_json::json!(500);
    let err = Qwen35MoeConfig::from_json(&doc).expect_err("500 is not a tile");
    let text = format!("{err}");
    assert!(text.contains("moe_intermediate_size"), "{text}");
    assert!(text.contains("500"), "{text}");
}

#[test]
fn more_experts_than_the_sort_can_hold_is_refused() {
    let mut doc = document();
    doc["text_config"]["num_experts"] = serde_json::json!(512);
    let err = Qwen35MoeConfig::from_json(&doc).expect_err("512 experts");
    assert!(format!("{err}").contains("num_experts"), "{err}");
}

#[test]
fn a_document_of_another_family_is_refused_before_its_subtree_is_read() {
    let mut doc = document();
    doc["model_type"] = serde_json::json!("qwen3_5");
    match Qwen35MoeConfig::from_json(&doc).expect_err("the dense hybrid is not this family") {
        CoreError::UnknownArch { model_type } => assert_eq!(model_type, "qwen3_5"),
        other => panic!("expected UnknownArch, got {other:?}"),
    }
}

// -- the Splash package's two documents, against each other -----------------

/// The shipped `manifest.json`'s account of the target, with the artifact list
/// and the parts no config reading consults reduced to one entry each.
///
/// Verbatim numbers from `Qwen3.6-35B-A3B-Splash`. It is built here rather than
/// checked in because what these cases vary is one field at a time, and a
/// fixture that had to be edited on disk to do that would be six fixtures.
fn manifest_json() -> serde_json::Value {
    let mut kinds = vec!["gdn"; 40];
    for slot in (3..40).step_by(4) {
        kinds[slot] = "attention";
    }
    serde_json::json!({
        "schema_version": 4,
        "model": "Qwen3.6-35B-A3B",
        "artifact_set_sha256": "0c465f96066fe62f185273a5c58f6de0f493d9bc64b0e43f77d1229eead2440a",
        "artifacts": [],
        "format": {
            "name": "splash-packed-q4-moe", "quant_group_size": 64, "storage_n": 256,
            "section_alignment_bytes": 16384, "q4_bits": 4, "q8_bits": 8,
            "target_layer_magic": "MDFM0001", "draft_layer_magic": "MDFD0004",
            "vision_magic": "MDFV0001"
        },
        "target": {
            "architecture": "qwen3_5_moe", "hidden_size": 2048, "layers": 40,
            "experts": 256, "experts_per_token": 8, "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512, "vocabulary_size": 248_320,
            "layer_types": kinds, "gdn_actual_width": 12352,
            "gdn_packed_width": 12544, "attention_packed_width": 9216
        },
        "draft": {
            "architecture": "DFlash2DraftModel", "block_size": 8,
            "dynamic_conv_group_size": 16, "dynamic_conv_kernel_size": 2,
            "hidden_size": 2048, "intermediate_size": 6144, "is_causal": false,
            "layers": 6, "selector_rank": 256, "selector_top_k": 16,
            "sliding_window": 2048, "target_capture_layers": [1, 6, 11, 16, 22, 27, 32, 37]
        },
        "execution_geometry": {
            "allocation_extent_target_bytes": 134_217_728, "attention_history_group": 128,
            "draft_proposal_tokens": 7, "draft_query_rows": 8, "draft_sliding_window": 2048,
            "maximum_batch_width": 4, "prefill_token_budget": 2048,
            "target_kv_block_tokens": 32, "target_verify_rows": 8
        },
        "converter": {
            "source_path": "install/convert.py", "source_git_commit": "d441aba",
            "source_sha256": "dd4f4b6", "adapter_sha256": "d8a306e",
            "mlx_version": "0.32.2"
        },
        "upstream": {
            "target": {"repo_id": "mlx-community/Qwen3.6-35B-A3B-4bit", "revision": "38740b8"},
            "draft": {"repo_id": "incoai/Qwen3.6-35B-A3B-DFlash2", "revision": "8e71350"},
            "tokenizer": {"repo_id": "Qwen/Qwen3.6-35B-A3B", "revision": "0000000"},
            "vision": {"repo_id": "Qwen/Qwen3.6-35B-A3B", "revision": "0000000"}
        }
    })
}

/// That manifest with one key of `target` replaced.
fn manifest_with(key: &str, value: serde_json::Value) -> Manifest {
    let mut doc = manifest_json();
    doc["target"][key] = value;
    Manifest::from_json(&serde_json::to_string(&doc).expect("json")).expect("a manifest")
}

/// The manifest as shipped.
fn manifest() -> Manifest {
    Manifest::from_json(&serde_json::to_string(&manifest_json()).expect("json"))
        .expect("a manifest")
}

#[test]
fn a_package_whose_two_documents_agree_reads_as_one_geometry() {
    let cfg = Qwen35MoeConfig::from_splash(&manifest(), &document()).expect("the two agree");
    // Nothing is merged: the two documents state the same numbers, and this is
    // the geometry either one of them alone would have produced.
    assert_eq!(cfg.shape(), config().shape());
    assert_eq!(cfg.moe().experts, 256);
    // A Splash package carries no quantization document. The storage of each
    // section is on the section, in `layout.json`.
    assert!(cfg.quant().is_none());
}

#[test]
fn a_width_the_two_documents_disagree_on_is_refused_with_both_numbers() {
    let manifest = manifest_with("hidden_size", serde_json::json!(4096));
    match Qwen35MoeConfig::from_splash(&manifest, &document()).expect_err("2048, not 4096") {
        CoreError::ShapeMismatch {
            field,
            expected,
            got,
        } => {
            assert_eq!(field, "manifest.target.hidden_size");
            // `expected` is the manifest's, which describes the bytes on disk.
            assert_eq!((expected, got), (4096, 2048));
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn every_width_the_two_documents_share_is_actually_compared() {
    // One case per overlapping field, because a cross-check that silently
    // compares four of seven looks exactly like one that compares all seven.
    for (key, field) in [
        ("hidden_size", "manifest.target.hidden_size"),
        ("layers", "manifest.target.layers"),
        ("experts", "manifest.target.experts"),
        ("experts_per_token", "manifest.target.experts_per_token"),
        (
            "moe_intermediate_size",
            "manifest.target.moe_intermediate_size",
        ),
        (
            "shared_expert_intermediate_size",
            "manifest.target.shared_expert_intermediate_size",
        ),
        ("vocabulary_size", "manifest.target.vocabulary_size"),
    ] {
        // `layers` is also the length of the layer roll, so moving it alone
        // would be caught by the roll's own length check first; the roll is
        // trimmed with it so that the width is what refuses.
        let mut doc = manifest_json();
        doc["target"][key] = serde_json::json!(7);
        if key == "layers" {
            doc["target"]["layer_types"] = serde_json::json!(vec!["gdn"; 7]);
        }
        let manifest =
            Manifest::from_json(&serde_json::to_string(&doc).expect("json")).expect("a manifest");
        match Qwen35MoeConfig::from_splash(&manifest, &document()).expect_err(key) {
            CoreError::ShapeMismatch {
                field: named,
                expected,
                ..
            } => {
                assert_eq!(named, field);
                assert_eq!(expected, 7);
            }
            other => panic!("{key}: expected ShapeMismatch, got {other:?}"),
        }
    }
}

#[test]
fn a_layer_roll_that_disagrees_at_one_slot_is_refused_by_that_slot() {
    // Every width still checks out and the count is still forty: one recurrent
    // slot and one attention slot have swapped places, which is a stack that
    // runs a recurrent block over an attention block's weights.
    let mut kinds = vec!["gdn"; 40];
    for slot in (3..40).step_by(4) {
        kinds[slot] = "attention";
    }
    kinds.swap(7, 8);
    let manifest = manifest_with("layer_types", serde_json::json!(kinds));
    match Qwen35MoeConfig::from_splash(&manifest, &document()).expect_err("slot 7 moved") {
        CoreError::InvalidValue { key, value } => {
            assert_eq!(key, "manifest.target.layer_types");
            assert!(value.contains("slot 7"), "{value}");
            assert!(value.contains("Gdn") && value.contains("Attn"), "{value}");
        }
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

#[test]
fn a_manifest_naming_another_architecture_is_refused_by_name() {
    let manifest = manifest_with("architecture", serde_json::json!("qwen3_5"));
    match Qwen35MoeConfig::from_splash(&manifest, &document()).expect_err("not this family") {
        CoreError::UnknownArch { model_type } => assert_eq!(model_type, "qwen3_5"),
        other => panic!("expected UnknownArch, got {other:?}"),
    }
}

#[test]
fn a_package_directory_without_the_two_documents_names_the_one_it_wanted() {
    let empty = tempfile::tempdir().expect("a temporary directory");
    match Qwen35MoeConfig::from_splash_dir(empty.path()).expect_err("nothing is there") {
        CoreError::ConfigMissing { key } => assert!(key.ends_with("manifest.json"), "{key}"),
        other => panic!("expected ConfigMissing, got {other:?}"),
    }
    // And the second document, which is the one at a path nobody guesses.
    std::fs::write(
        empty.path().join("manifest.json"),
        serde_json::to_string(&manifest_json()).expect("json"),
    )
    .expect("the manifest");
    match Qwen35MoeConfig::from_splash_dir(empty.path()).expect_err("no model config") {
        CoreError::ConfigMissing { key } => {
            assert!(key.ends_with("tokenizer/config.json"), "{key}");
        }
        other => panic!("expected ConfigMissing, got {other:?}"),
    }
}

#[test]
fn a_package_read_off_disk_is_the_package_read_from_the_two_values() {
    let root = tempfile::tempdir().expect("a temporary directory");
    std::fs::write(
        root.path().join("manifest.json"),
        serde_json::to_string(&manifest_json()).expect("json"),
    )
    .expect("the manifest");
    std::fs::create_dir(root.path().join("tokenizer")).expect("the tokenizer directory");
    std::fs::write(
        root.path().join("tokenizer").join("config.json"),
        serde_json::to_string(&document()).expect("json"),
    )
    .expect("the model config");
    let from_disk = Qwen35MoeConfig::from_splash_dir(root.path()).expect("both documents");
    assert_eq!(
        from_disk,
        Qwen35MoeConfig::from_splash(&manifest(), &document()).expect("the two agree")
    );
}

#[test]
fn a_layer_list_that_disagrees_with_the_interval_is_refused() {
    let mut types: Vec<&str> = vec!["linear_attention"; 40];
    for slot in (3..40).step_by(4) {
        types[slot] = "full_attention";
    }
    // One slot moved: every individual width is still correct.
    types[7] = "linear_attention";
    types[8] = "full_attention";
    let doc = with_text_key("layer_types", serde_json::json!(types));
    match Qwen35MoeConfig::from_hf(&doc, quant()).expect_err("two slots disagree") {
        CoreError::ShapeMismatch { field, got, .. } => {
            assert_eq!(field, "layer_types.interval_mismatch_count");
            assert_eq!(got, 2);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}
