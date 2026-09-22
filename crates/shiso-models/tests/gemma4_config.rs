//! What the Gemma 4 document says, and whether the container agrees with it.
//!
//! The fixture is `shiso/test-vectors/loader/gemma4-config.json`, copied verbatim from
//! the local sidecar directory. Nothing here hand-writes a config: a document
//! typed out by hand would encode wrong constants into every assertion that
//! reads it.
//!
//! # Why one case compares two readers rather than checking a number
//!
//! This family is described twice - by a `config.json` beside the weights and by
//! the container's own `gemma4.*` metadata - and either can be the only one a
//! deployment has. Two readers that each pass their own assertions but disagree
//! by one head width would give the same checkpoint two different stacks
//! depending on which file happened to be present, and nothing downstream could
//! notice. So the last case asserts the only thing that rules that out: the two
//! shapes are equal.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use shiso_core::config::HfConfig;
use shiso_core::error::CoreError;
use shiso_core::ids::{LayerIdx, TokenId};
use shiso_core::model::StopTokens;
use shiso_engine::shape::{Activation, AttnScale, KvSource, LayerKind, MediaPositions, NormScheme};
use shiso_modelfile::GgufFile;
use shiso_models::gemma4::Gemma4Config;
use shiso_models::qwen35::Qwen35Config;

/// The checked-in fixture directory, reached from this crate's manifest.
fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors")
}

/// The Gemma 4 E4B document.
fn fixture() -> HfConfig {
    let path = vectors().join("loader/gemma4-config.json");
    HfConfig::from_file(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// The fixture with one value replaced, as a parsed document.
///
/// The edit is made on the parsed tree rather than on the text: three of the
/// four cases below change a float, and a float's spelling in the file is not
/// something a test should have to match.
fn mutated(edit: impl FnOnce(&mut serde_json::Value)) -> HfConfig {
    let path = vectors().join("loader/gemma4-config.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let mut document: serde_json::Value = serde_json::from_str(&text).expect("the fixture parses");
    edit(&mut document);
    HfConfig::from_json_str(&document.to_string()).expect("the edited fixture parses")
}

/// One entry of the text subtree, for an edit to overwrite.
fn text_entry<'a>(document: &'a mut serde_json::Value, path: &[&str]) -> &'a mut serde_json::Value {
    let mut node = document
        .get_mut("text_config")
        .expect("the fixture has a text subtree");
    for key in path {
        node = node
            .get_mut(*key)
            .unwrap_or_else(|| panic!("the text subtree has no {key}"));
    }
    node
}

/// The refusal a tampered scale factor produces, or a panic naming what came
/// back instead.
fn refused_value(config: &HfConfig) -> (String, String) {
    let err = Gemma4Config::from_hf(config, Path::new("/nonexistent"))
        .expect_err("a scale factor the kernels were not written for cannot be adopted");
    match err {
        CoreError::InvalidValue { key, value } => (key.to_owned(), value),
        other => panic!("expected InvalidValue, got {other:?}"),
    }
}

/// The Qwen3.5 document, which this family must refuse.
fn other_family() -> HfConfig {
    let path = vectors().join("loader/qwen35-config.json");
    HfConfig::from_file(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// The local container, overridable, or `None` when it is not on this machine.
fn checkpoint() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4").ok()?);
    path.exists().then_some(path)
}

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

#[test]
fn the_document_is_recognised() {
    assert!(Gemma4Config::describes(&fixture()));
    let err = Qwen35Config::from_hf(&fixture()).expect_err("a gemma is not a qwen");
    match err {
        CoreError::UnknownArch { model_type } => assert_eq!(model_type, "gemma4"),
        other => panic!("expected UnknownArch, got {other:?}"),
    }
}

#[expect(
    clippy::float_cmp,
    reason = "JSON constants narrowed from f64 to f32, compared bit-exact on purpose"
)]
#[test]
fn the_shape_is_e4b() {
    // The document carries `text_config`, so the container is never opened and
    // the path handed in is never read.
    let cfg = Gemma4Config::from_hf(&fixture(), Path::new("/nonexistent")).expect("the e4b shape");
    let s = cfg.shape();

    assert_eq!(s.n_layers(), 42);
    assert_eq!(s.attn.len(), 42);
    assert!(s.layers.iter().all(|kind| *kind == LayerKind::Attn));
    assert_eq!(s.hidden, 2560);
    assert_eq!(s.intermediate, 10240);
    assert_eq!(s.vocab, 262_144);
    assert_eq!(cfg.ple_vocab(), 262_144);

    // Slot 5 is the first full-attention slot: the wide head, a quarter of it
    // rotated, over a ladder as wide as the head itself.
    assert_eq!(s.attn[5].head_dim, 512);
    assert_eq!(s.attn[5].n_rot, 128);
    assert_eq!(s.attn[5].freq_dims, 512);
    assert_eq!(s.attn[5].window, None);
    assert_eq!(s.attn[5].rope_theta, 1.0e6);
    assert_eq!(s.attn[5].scale, AttnScale::One);
    assert_eq!(s.attn[5].kv, KvSource::Own);

    // Slot 4 is a sliding slot: the narrow head, fully rotated, windowed.
    assert_eq!(s.attn[4].head_dim, 256);
    assert_eq!(s.attn[4].window, NonZeroU32::new(512));
    assert_eq!(s.attn[4].n_rot, 256);
    assert_eq!(s.attn[4].freq_dims, 256);
    assert_eq!(s.attn[4].rope_theta, 1.0e4);
    assert_eq!(s.attn[4].q_heads, 8);
    assert_eq!(s.attn[4].kv_heads, 2);

    // Eighteen slots share: a sliding one reads slot 22's runs, a full one reads
    // slot 23's, and twenty-four slots own runs of their own.
    assert_eq!(s.kv_owner_rank(LayerIdx::new(24)), Some(22));
    assert_eq!(s.kv_owner_rank(LayerIdx::new(29)), Some(23));
    assert_eq!(s.kv_owning_attn().len(), 24);
    assert_eq!(
        s.attn[24].kv,
        KvSource::Shared {
            owner: LayerIdx::new(22)
        }
    );
    assert_eq!(
        s.attn[29].kv,
        KvSource::Shared {
            owner: LayerIdx::new(23)
        }
    );

    // The family switches.
    assert_eq!(s.final_softcap, Some(30.0));
    assert!(s.layer_scalar);
    assert_eq!(s.ple.map(|p| p.dim), Some(256));
    assert_eq!(s.ple.map(|p| p.vocab), Some(262_144));
    assert_eq!(s.activation, Activation::GeGluTanh);
    assert_eq!(s.norms, NormScheme::PrePost);
    assert_eq!(s.media_positions, MediaPositions::Linear);
    assert!(s.v_norm);
    assert!(!s.attn_output_gate);
    assert_eq!(s.embed_scale, Some(2560_f32.sqrt()));
    assert_eq!(s.rms_eps, 1.0e-6);
    s.validate().expect("the parsed shape is valid");

    assert_eq!(
        cfg.stop_tokens(),
        [TokenId::new(106), TokenId::new(1), TokenId::new(50)]
    );
}

/// The document lists the three ids `<eos>`, `<turn|>`, `<|tool_response>` in
/// that order, and the reader does not adopt it: the id a caller with room for
/// exactly one records is the one that ends an *assistant turn*, which is
/// `<turn|>` and not the `<eos>` that ends a document.
#[test]
fn the_turn_ending_id_leads_the_set() {
    let cfg = Gemma4Config::from_hf(&fixture(), Path::new("/nonexistent"))
        .expect("the fixture is this family");
    assert_eq!(
        cfg.stop_tokens().first(),
        Some(&TokenId::new(106)),
        "the document's own order reached the stop set"
    );
    let stop = StopTokens::new(cfg.stop_tokens()).expect("three ids fit");
    assert_eq!(stop.primary(), TokenId::new(106));
    for id in [TokenId::new(1), TokenId::new(50)] {
        assert!(
            stop.contains(id),
            "{id:?} is declared and is not in the set"
        );
    }
}

#[test]
fn a_qwen_document_is_not_this_family() {
    assert!(!Gemma4Config::describes(&other_family()));
    let err = Gemma4Config::from_hf(&other_family(), Path::new("/nonexistent"))
        .expect_err("a qwen is not a gemma");
    match err {
        CoreError::UnknownArch { model_type } => assert_eq!(model_type, "qwen3_5"),
        other => panic!("expected UnknownArch, got {other:?}"),
    }
}

#[test]
fn another_sliding_rope_base_is_refused() {
    let config = mutated(|document| {
        *text_entry(
            document,
            &["rope_parameters", "sliding_attention", "rope_theta"],
        ) = serde_json::json!(5000.0);
    });
    let (key, value) = refused_value(&config);
    assert_eq!(key, "rope_parameters.sliding_attention.rope_theta");
    assert!(value.starts_with("5000"), "got {value}");
}

#[test]
fn another_global_rope_base_is_refused() {
    let config = mutated(|document| {
        *text_entry(
            document,
            &["rope_parameters", "full_attention", "rope_theta"],
        ) = serde_json::json!(500_000.0);
    });
    let (key, value) = refused_value(&config);
    assert_eq!(key, "rope_parameters.full_attention.rope_theta");
    assert!(value.starts_with("500000"), "got {value}");
}

#[test]
fn another_normalization_epsilon_is_refused() {
    let config = mutated(|document| {
        *text_entry(document, &["rms_norm_eps"]) = serde_json::json!(1.0e-5);
    });
    let (key, _) = refused_value(&config);
    assert_eq!(key, "rms_norm_eps");
}

#[test]
fn another_attention_schedule_is_refused() {
    // Slot 4 is windowed in every checkpoint of this family; the list is the
    // only statement of that, and its length is unchanged by the edit.
    let config = mutated(|document| {
        let types = text_entry(document, &["layer_types"])
            .as_array_mut()
            .expect("the list is an array");
        assert_eq!(types.len(), 42);
        types[4] = serde_json::json!("full_attention");
    });
    let err = Gemma4Config::from_hf(&config, Path::new("/nonexistent"))
        .expect_err("a schedule that windows the wrong slots is not this family");
    match err {
        CoreError::ShapeMismatch { field, got, .. } => {
            assert_eq!(field, "layer_types.schedule_mismatch_count");
            assert_eq!(got, 1);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
#[ignore = "needs the local GGUF"]
fn the_gguf_metadata_agrees_with_the_document() {
    let Some(path) = checkpoint() else {
        skip("the gemma 4 E4B container is not on this machine");
        return;
    };
    let file = GgufFile::open_shared(&path).expect("the container opens");
    let from_gguf = Gemma4Config::from_gguf(&file).expect("the container states the e4b shape");
    let from_doc = Gemma4Config::from_hf(&fixture(), &path).expect("the document states it too");
    assert_eq!(from_gguf.shape(), from_doc.shape());
    assert_eq!(from_gguf.stop_tokens(), from_doc.stop_tokens());
    assert_eq!(from_gguf.ple_vocab(), from_doc.ple_vocab());
}
