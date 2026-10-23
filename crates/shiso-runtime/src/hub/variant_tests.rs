//! A recipe variant's components become the serve options' paths.

use std::path::Path;

use shiso_hub::{Recipe, Role, ValidationMode, validate_recipe};

use super::fill_options_from_variant;
use crate::options::{ExplicitlySet, ServeOptions};
use crate::recipe_defaults::apply_recipe_defaults_unless_set;

/// The recipe the registry fixture carries for the sparse family: the
/// published trellis export as one directory, its tower file bound to
/// `--vision`, its draft head inside the directory.
const QWEN38: &str = include_str!("../../../shiso-registry/tests/fixtures/qwen3.8-flash.json");

/// The CUDA variant resolves to the directory as `--model`, the tower
/// and n-gram table files inside it as `--vision` and `--ngram`, and the
/// tuned knobs as recipe defaults.
#[test]
fn exl3_variant_resolves_to_directory_and_tower() {
    let recipe: Recipe = serde_json::from_str(QWEN38).expect("fixture parses");
    validate_recipe(&recipe, ValidationMode::Strict).expect("fixture validates");
    let variant = recipe
        .variants
        .iter()
        .find(|v| v.key.as_str() == "cuda-exl3-405")
        .expect("cuda variant");

    let mut args = ServeOptions {
        ctx: 8192,
        port: 1,
        ..ServeOptions::default()
    };
    let explicit: ExplicitlySet = ["ctx", "port"].into_iter().collect();
    let vdir = Path::new("/hub/models/qwen3.8-flash/cuda-exl3-405");
    fill_options_from_variant(&mut args, vdir, variant, &recipe.name).expect("populates");
    apply_recipe_defaults_unless_set(
        &mut args,
        &explicit,
        variant.server_defaults.as_ref().expect("defaults"),
    );

    let base = vdir.join(Role::Base.as_str());
    assert_eq!(args.model, Some(base.clone()));
    assert_eq!(args.vision, Some(base.join("vision_k6.safetensors")));
    assert_eq!(args.ngram, Some(base.join("ngram_embedding.safetensors")));
    // The head is inside the directory: no flag names it.
    assert_eq!(args.draft_model, None);
    assert_eq!(args.config, None);
    assert_eq!(args.model_name.as_deref(), Some("qwen3.8-flash"));

    assert_eq!(args.ctx, 8192, "an explicit flag beats the recipe");
    assert_eq!(args.spec_k, 4);
    assert_eq!(args.spec_lookup, 8);
    assert!(args.capture_verify && args.capture_batch);
    assert_eq!(args.gpu_memory_utilization, Some(0.88));
    assert_eq!(args.solo_prefill_chunk, 4096);
}

/// The re-quantized export of the same family: its shards carry no tower
/// file, so the base checkpoint's tower comes from another repository as a
/// component of its own.
const QWEN38_UNCENSORED: &str =
    include_str!("../../../shiso-registry/tests/fixtures/qwen3.8-flash-uncensored.json");

/// A tower supplied by a component of its own resolves to that
/// component's directory and entry, while the n-gram table stays the file
/// bound inside the base directory.
#[test]
fn a_tower_from_another_repository_resolves_to_its_component() {
    let recipe: Recipe = serde_json::from_str(QWEN38_UNCENSORED).expect("fixture parses");
    validate_recipe(&recipe, ValidationMode::Strict).expect("fixture validates");
    let variant = recipe
        .variants
        .iter()
        .find(|v| v.key.as_str() == "exl3-405")
        .expect("cuda variant");
    let mut args = ServeOptions::default();
    let vdir = Path::new("/hub/models/qwen3.8-flash-uncensored/exl3-405");
    fill_options_from_variant(&mut args, vdir, variant, &recipe.name).expect("populates");

    let base = vdir.join(Role::Base.as_str());
    assert_eq!(args.model, Some(base.clone()));
    assert_eq!(
        args.vision,
        Some(
            vdir.join(Role::Vision.as_str())
                .join("vision_k6.safetensors")
        )
    );
    assert_eq!(args.ngram, Some(base.join("ngram_embedding.safetensors")));
    assert_eq!(args.draft_model, None);
    assert_eq!(args.model_name.as_deref(), Some("qwen3.8-flash-uncensored"));
}

/// The MLX variant is a directory checkpoint with no entry file, and the
/// residency choice is the server's own `auto`.
#[test]
fn mlx_variant_points_at_its_directory() {
    let recipe: Recipe = serde_json::from_str(QWEN38).expect("fixture parses");
    let variant = recipe
        .variants
        .iter()
        .find(|v| v.key.as_str() == "metal-mlx4")
        .expect("metal variant");
    let mut args = ServeOptions::default();
    // A real directory: the config is looked for inside a checkpoint
    // directory and beside a checkpoint file, decided on the filesystem.
    let home = tempfile::tempdir().expect("tempdir");
    let vdir = home.path();
    std::fs::create_dir_all(vdir.join("base")).expect("base dir");
    fill_options_from_variant(&mut args, vdir, variant, &recipe.name).expect("populates");
    assert_eq!(args.model, Some(vdir.join("base")));
    assert_eq!(args.config, None);
    assert_eq!(args.config_path(), vdir.join("base").join("config.json"));
    assert_eq!(args.draft_model, None);
}

/// A hub-hosted draft vocabulary reaches `--draft-vocab` from its own
/// component directory, beside a base that speculates by itself.
#[test]
fn a_hub_draft_vocabulary_resolves_to_its_file() {
    let recipe: Recipe = serde_json::from_str(
        r#"{
          "schema_version": 1, "name": "q-test", "family": "qwen4exp",
          "min_shiso_version": "0.2.0", "description": "x",
          "tasks": ["chat"], "features": {},
          "modalities": {"inputs": ["text"], "outputs": ["text"]},
          "variants": [{
            "key": "exl3", "backends": ["cuda"], "quant": "exl3",
            "requirements": {"min_ram_bytes": 1, "disk_bytes": 20},
            "server_defaults": {"spec_k": 4},
            "components": {
              "base": {"provider": "huggingface", "repo": "o/q",
                "revision": "3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c",
                "files": [{"name": "model.safetensors", "size": 10,
                  "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"}]},
              "draft_vocab": {"provider": "shiso", "entry": "draft_vocab_counts.txt",
                "files": [{"name": "draft_vocab_counts.txt", "size": 10,
                  "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"}]}
            }
          }]
        }"#,
    )
    .expect("recipe parses");
    validate_recipe(&recipe, ValidationMode::Strict).expect("recipe validates");
    let mut args = ServeOptions::default();
    let vdir = Path::new("/hub/models/q-test/exl3");
    fill_options_from_variant(&mut args, vdir, &recipe.variants[0], &recipe.name)
        .expect("populates");
    assert_eq!(
        args.draft_vocab,
        Some(vdir.join("draft_vocab").join("draft_vocab_counts.txt"))
    );
}

/// Files bound to `image_text_encoder`, `image_vae` and
/// `image_tokenizer` reach their own flags as paths inside the component
/// that downloads them.
#[test]
fn image_variant_routes_bound_text_encoder_vae_and_tokenizer() {
    let recipe: Recipe = serde_json::from_str(
        r#"{
          "schema_version": 1, "name": "qwen-test", "family": "flux2",
          "min_shiso_version": "0.1.0", "description": "x",
          "tasks": ["image_generation"], "features": {},
          "modalities": {"inputs": ["text"], "outputs": ["image"]},
          "variants": [{
            "key": "fp8", "backends": ["metal"], "quant": "fp8",
            "requirements": {"min_ram_bytes": 1, "disk_bytes": 40},
            "components": {
              "image": {"provider": "huggingface", "repo": "o/split",
                "revision": "3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c",
                "entry": "dit.safetensors",
                "files": [
                  {"name": "dit.safetensors", "size": 10,
                   "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"},
                  {"name": "te.safetensors", "size": 10, "binds": "image_text_encoder",
                   "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"},
                  {"name": "vae.safetensors", "size": 10, "binds": "image_vae",
                   "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"},
                  {"name": "tokenizer.json", "size": 10, "binds": "image_tokenizer",
                   "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"}]}
            }
          }]
        }"#,
    )
    .expect("recipe parses");
    validate_recipe(&recipe, ValidationMode::Strict).expect("recipe validates");
    let mut args = ServeOptions::default();
    let vdir = Path::new("/hub/models/qwen-test/fp8");
    fill_options_from_variant(&mut args, vdir, &recipe.variants[0], &recipe.name)
        .expect("populates");
    assert_eq!(
        args.image_model,
        Some(vdir.join("image").join("dit.safetensors"))
    );
    assert_eq!(
        args.image_text_encoder,
        Some(vdir.join("image").join("te.safetensors"))
    );
    assert_eq!(
        args.image_vae,
        Some(vdir.join("image").join("vae.safetensors"))
    );
    assert_eq!(
        args.image_tokenizer,
        Some(vdir.join("image").join("tokenizer.json"))
    );
    assert_eq!(args.image_assets_dir, None);
}

/// An image variant hands its checkpoint file to `--image-model`, its
/// asset tree to `--image-assets-dir`, and answers to the recipe's name.
#[test]
fn image_variant_resolves_to_model_and_assets() {
    let recipe: Recipe = serde_json::from_str(
        r#"{
          "schema_version": 1, "name": "flux-test", "family": "flux2",
          "min_shiso_version": "0.1.0", "description": "x",
          "tasks": ["image_generation"], "features": {},
          "modalities": {"inputs": ["text"], "outputs": ["image"]},
          "variants": [{
            "key": "fp8", "backends": ["metal"], "quant": "fp8",
            "requirements": {"min_ram_bytes": 1, "disk_bytes": 20},
            "components": {
              "image": {"provider": "huggingface", "repo": "o/fp8",
                "revision": "3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c",
                "entry": "model.safetensors",
                "files": [{"name": "model.safetensors", "size": 10,
                  "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"}]},
              "image_assets": {"provider": "huggingface", "repo": "o/bf16",
                "revision": "3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c",
                "files": [{"name": "vae/config.json", "size": 10,
                  "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"}]}
            }
          }]
        }"#,
    )
    .expect("recipe parses");
    validate_recipe(&recipe, ValidationMode::Strict).expect("recipe validates");
    let mut args = ServeOptions::default();
    let vdir = Path::new("/hub/models/flux-test/fp8");
    fill_options_from_variant(&mut args, vdir, &recipe.variants[0], &recipe.name)
        .expect("populates");
    assert_eq!(
        args.image_model,
        Some(vdir.join("image").join("model.safetensors"))
    );
    assert_eq!(args.image_assets_dir, Some(vdir.join("image_assets")));
    assert_eq!(args.image_model_id.as_deref(), Some("flux-test"));
    assert_eq!(args.model, None);
}
