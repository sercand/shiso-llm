//! Tests for Gemma 4 MLX role binding and name mapping.
#![expect(clippy::print_stdout, reason = "test skip")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::config::HfConfig;
use shiso_modelfile::NameMap;
use shiso_modelfile::safetensors::SafetensorsFile;
use shiso_models::gemma4::{Gemma4Config, Gemma4Names, bind_gemma4_mlx_desc};

fn vectors() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors")
}

fn config() -> Gemma4Config {
    let path = vectors().join("loader/gemma4-config.json");
    let document = HfConfig::from_file(&path).expect("loader/gemma4-config.json");
    Gemma4Config::from_hf(&document, Path::new("/nonexistent")).expect("config")
}

fn mlx_dir() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4_MLX").ok()?);
    path.is_dir().then_some(path)
}

#[test]
fn gemma4_mlx_name_mapping_translates_standard_tensors() {
    let names = Gemma4Names;

    assert!(
        names
            .map_role("language_model.model.embed_tokens")
            .is_some()
    );
    assert!(names.map_role("language_model.model.norm.weight").is_some());
    assert!(
        names
            .map_role("language_model.model.per_layer_model_projection")
            .is_some()
    );
    assert!(
        names
            .map_role("language_model.model.per_layer_projection_norm.weight")
            .is_some()
    );
    assert!(
        names
            .map_role("language_model.model.embed_tokens_per_layer")
            .is_some()
    );

    // Layer 0 has k and v
    assert!(
        names
            .map_role("language_model.model.layers.0.self_attn.k_proj")
            .is_some()
    );
    assert!(
        names
            .map_role("language_model.model.layers.0.self_attn.v_proj")
            .is_some()
    );

    // Layer 25 shares its key and value, but the name map does not know that
    // and must not decide it: it translates the names whatever slot they sit
    // on, and the binder is what refuses an export that ships them on a
    // sharing slot. A map that dropped them here would make that refusal
    // unreachable and the export's schedule unchecked.
    assert!(
        names
            .map_role("language_model.model.layers.25.self_attn.k_proj")
            .is_some()
    );
    assert!(
        names
            .map_role("language_model.model.layers.25.self_attn.v_proj")
            .is_some()
    );

    // Vision tensors mapped to None
    assert!(
        names
            .map_role("vision_tower.patch_embedder.input_proj.weight")
            .is_none()
    );
}

/// A sharing slot's key translates to the name the binder looks it up by.
///
/// The name, not merely something: the binder's refusal probes
/// `blk.N.attn_k.weight` by hand, so a map that translated this to any other
/// spelling would leave that probe looking for a tensor nothing produces and
/// the refusal would pass on an export it should have stopped.
#[test]
fn a_sharing_slots_key_translates_to_the_name_the_refusal_probes() {
    let names = Gemma4Names;
    for (mlx, gguf) in [
        ("self_attn.k_proj", "blk.25.attn_k.weight"),
        ("self_attn.v_proj", "blk.25.attn_v.weight"),
        ("self_attn.k_norm.weight", "blk.25.attn_k_norm.weight"),
    ] {
        let role = names
            .map_role(&format!("language_model.model.layers.25.{mlx}"))
            .unwrap_or_else(|| panic!("{mlx} translates"));
        assert_eq!(role.gguf, gguf);
    }
}

#[test]
fn gemma4_mlx_binds_against_real_directory() {
    let Some(dir) = mlx_dir() else {
        println!("SKIP: no local Gemma 4 MLX directory");
        return;
    };

    let cfg = config();
    let st = SafetensorsFile::open_with(&dir, Arc::new(Gemma4Names)).expect("open safetensors");
    let binding = bind_gemma4_mlx_desc(&st, &cfg).expect("binding succeeds");

    assert_eq!(binding.layers.len(), 42);
    assert_eq!(binding.token_embd.name.as_ref(), "token_embd.weight");
    assert_eq!(binding.output_norm.name.as_ref(), "output_norm.weight");

    // Check quantization bits
    assert_eq!(st.bits_for("token_embd.weight"), 4);
    assert_eq!(st.bits_for("blk.0.attn_q.weight"), 4);
    assert_eq!(st.bits_for("blk.0.ffn_gate.weight"), 8);
    assert_eq!(st.bits_for("blk.0.ffn_up.weight"), 8);
    assert_eq!(st.bits_for("blk.0.ffn_down.weight"), 8);

    // Slots 0..24 own KV, slots 24..42 share KV
    for (i, layer) in binding.layers.iter().enumerate() {
        if i < 24 {
            assert!(layer.attn.k.is_some(), "layer {i} should own k");
            assert!(layer.attn.v.is_some(), "layer {i} should own v");
        } else {
            assert!(layer.attn.k.is_none(), "layer {i} should share k");
            assert!(layer.attn.v.is_none(), "layer {i} should share v");
        }
    }
}
