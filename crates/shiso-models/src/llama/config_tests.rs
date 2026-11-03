//! The refusals of the family's config, on forged containers, and the geometry
//! of the real one when `SHISO_TEST_MODEL_CLEF_LLAMA` names it.

use shiso_core::CoreError;
use shiso_engine::shape::LayerKind;
use shiso_modelfile::GgufFile;

use super::LlamaConfig;
use crate::diffusion::gguf_forge::{ForgedTensor, write_gguf};

/// The geometry the config reads out of a forged file.
fn read(kvs: &[(&str, &str)], tensors: &[&str]) -> Result<LlamaConfig, CoreError> {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("forged.gguf");
    let one = 1.0f32.to_le_bytes();
    let forged: Vec<ForgedTensor<'_>> = tensors
        .iter()
        .map(|name| ForgedTensor {
            name,
            ne: &[1],
            type_id: 0,
            bytes: &one,
        })
        .collect();
    write_gguf(&path, kvs, &forged);
    let file = GgufFile::open(&path).expect("the forged file opens");
    LlamaConfig::from_gguf(&file)
}

#[test]
fn a_stock_llama_export_is_refused_for_its_permuted_rows() {
    let refused = read(
        &[("general.architecture", "llama")],
        &["blk.0.attn_q.weight"],
    );
    assert!(
        matches!(refused, Err(CoreError::Unsupported { what, .. }) if what.contains("permuted")),
        "{refused:?}"
    );
}

#[test]
fn a_clef_file_over_the_hybrid_is_not_this_family() {
    let other = read(
        &[
            ("general.architecture", "clef"),
            ("clef.decision.type", "clef"),
        ],
        &["blk.0.ssm_a", "blk.3.attn_q.weight"],
    );
    assert!(
        matches!(other, Err(CoreError::UnknownArch { .. })),
        "{other:?}"
    );
}

#[test]
fn a_clef_file_must_name_the_head_it_carries() {
    let missing = read(
        &[("general.architecture", "clef")],
        &["blk.0.attn_q.weight"],
    );
    assert!(
        matches!(&missing, Err(CoreError::ConfigMissing { key }) if key == "clef.decision.type"),
        "{missing:?}"
    );
    let unknown = read(
        &[
            ("general.architecture", "clef"),
            ("clef.decision.type", "other"),
        ],
        &["blk.0.attn_q.weight"],
    );
    assert!(
        matches!(unknown, Err(CoreError::InvalidValue { .. })),
        "{unknown:?}"
    );
}

#[test]
fn a_scaled_rotation_is_refused() {
    let scaled = read(
        &[
            ("general.architecture", "clef"),
            ("clef.decision.type", "clef"),
            ("clef.rope.scaling.type", "linear"),
        ],
        &["blk.0.attn_q.weight"],
    );
    assert!(
        matches!(scaled, Err(CoreError::Unsupported { .. })),
        "{scaled:?}"
    );
}

/// The released checkpoint's geometry, read from the file itself.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "a test that needs a checkpoint says on stderr why it skipped"
)]
fn the_released_container_reads_as_a_rope_only_dense_stack() {
    let Some(path) = std::env::var_os("SHISO_TEST_MODEL_CLEF_LLAMA") else {
        eprintln!("SKIP: SHISO_TEST_MODEL_CLEF_LLAMA is not set");
        return;
    };
    let file = GgufFile::open(&path).expect("the container opens");
    let config = LlamaConfig::from_gguf(&file).expect("the container's geometry");
    let shape = config.shape();
    assert_eq!(shape.hidden, 576);
    assert_eq!(shape.vocab, 49_153);
    assert_eq!(shape.intermediate, 1536);
    assert_eq!(shape.layers, vec![LayerKind::Attn; 30]);
    let slot = shape.attn[0];
    assert_eq!((slot.q_heads, slot.kv_heads, slot.head_dim), (9, 3, 64));
    assert_eq!((slot.n_rot, slot.freq_dims), (64, 64));
    assert!(!slot.qk_norm, "a llama slot normalizes neither head");
    assert!((slot.rope_theta - 1e5).abs() < 1.0);
    assert_eq!(
        config.stop_tokens(),
        &[shiso_core::ids::TokenId::new(2)],
        "the turn ends at the container's eos"
    );
}
