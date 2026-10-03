#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]
#![expect(clippy::indexing_slicing, reason = "test fixture indexes")]

//! Unit tests for `shiso-hub` (§9.1, §9.3).

use std::collections::BTreeMap;
use std::fs;
use tempfile::tempdir;

use crate::config::{HfToken, HubConfig, RegistryEndpoint};
use crate::error::HubError;
use crate::host::{Backend, CudaInfo, HostProfile};
use crate::recipe::{
    CURRENT_SCHEMA_VERSION, CommitSha, Component, FileEntry, HostConstraints, ModalCapabilities,
    Modality, ModelName, ModelRef, Recipe, RecipeFeatures, Requirements, Role, ServerDefaults,
    Sha256, Task, Variant, VariantKey,
};
use crate::select::select_variant;
use crate::store::BlobStore;
use crate::validate::{ValidationMode, validate_recipe};

fn make_test_recipe() -> Recipe {
    let mut components = BTreeMap::new();
    components.insert(
        Role::Base,
        Component {
            provider: "huggingface".to_string(),
            repo: "shiso/test-model".to_string(),
            revision: CommitSha::parse("3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c").unwrap(),
            entry: None,
            files: vec![
                FileEntry {
                    name: "model.safetensors".to_string(),
                    size: 5000,
                    sha256: Sha256::parse(
                        "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a",
                    )
                    .unwrap(),
                },
                FileEntry {
                    name: "config.json".to_string(),
                    size: 100,
                    sha256: Sha256::parse(
                        "0d1f7c5b2e9a8b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4",
                    )
                    .unwrap(),
                },
            ],
        },
    );

    let variant_metal = Variant {
        key: VariantKey::parse("metal-mlx4").unwrap(),
        backend: "metal".to_string(),
        quant: "mlx-4bit".to_string(),
        requires: None,
        requirements: Requirements {
            min_ram_bytes: 8 * 1024 * 1024 * 1024,
            recommended_ram_bytes: Some(16 * 1024 * 1024 * 1024),
            min_vram_bytes: None,
            disk_bytes: 5100,
        },
        components: components.clone(),
        server_defaults: Some(ServerDefaults {
            ctx: Some(8192),
            ..Default::default()
        }),
    };

    let variant_cuda = Variant {
        key: VariantKey::parse("cuda-fp8").unwrap(),
        backend: "cuda".to_string(),
        quant: "fp8".to_string(),
        requires: Some(HostConstraints {
            min_cuda_sm: Some(89),
        }),
        requirements: Requirements {
            min_ram_bytes: 16 * 1024 * 1024 * 1024,
            recommended_ram_bytes: Some(32 * 1024 * 1024 * 1024),
            min_vram_bytes: Some(12 * 1024 * 1024 * 1024),
            disk_bytes: 5100,
        },
        components,
        server_defaults: None,
    };

    Recipe {
        schema_version: CURRENT_SCHEMA_VERSION,
        name: ModelName::parse("test-model").unwrap(),
        aliases: vec![ModelName::parse("test-alias").unwrap()],
        family: "qwen35".to_string(),
        min_shiso_version: semver::Version::parse("0.1.0").unwrap(),
        description: "A test model recipe".to_string(),
        tags: vec!["test".to_string()],
        tasks: vec![Task::Chat],
        modalities: ModalCapabilities {
            inputs: vec![Modality::Text],
            outputs: vec![Modality::Text],
        },
        features: RecipeFeatures {
            tools: true,
            reasoning: false,
            reasoning_efforts: vec![],
            default_reasoning_effort: None,
            structured_output: true,
        },
        variants: vec![variant_metal, variant_cuda],
        published_at: None,
        updated_at: None,
    }
}

#[test]
fn hftoken_debug_redacts_secret() {
    let token = HfToken::new("hf_secret_token_12345");
    let debug_str = format!("{token:?}");
    assert_eq!(debug_str, "HfToken(***)");
    assert_eq!(token.as_str(), "hf_secret_token_12345");
}

#[test]
fn model_ref_parsing() {
    let r1 = ModelRef::parse("qwen3.5-9b").unwrap();
    assert_eq!(r1.model.as_str(), "qwen3.5-9b");
    assert_eq!(r1.variant, None);
    assert_eq!(r1.to_string(), "qwen3.5-9b");

    let r2 = ModelRef::parse("qwen3.5-9b:metal-mlx4").unwrap();
    assert_eq!(r2.model.as_str(), "qwen3.5-9b");
    assert_eq!(r2.variant.as_ref().unwrap().as_str(), "metal-mlx4");
    assert_eq!(r2.to_string(), "qwen3.5-9b:metal-mlx4");

    assert!(ModelRef::parse("UPPERCASE").is_err());
    assert!(ModelRef::parse("invalid/path").is_err());
}

#[test]
fn variant_selection_metal_host() {
    let recipe = make_test_recipe();
    let host = HostProfile {
        backends: vec![Backend::Metal],
        ram_bytes: 32 * 1024 * 1024 * 1024,
        cuda: None,
    };

    let selected = select_variant(&recipe, &host, None).unwrap();
    assert_eq!(selected.variant.key.as_str(), "metal-mlx4");
    assert!(selected.warning.is_none());
}

#[test]
fn variant_selection_cuda_host() {
    let recipe = make_test_recipe();
    let host = HostProfile {
        backends: vec![Backend::Cuda],
        ram_bytes: 64 * 1024 * 1024 * 1024,
        cuda: Some(CudaInfo {
            sm: 89,
            vram_bytes: 24 * 1024 * 1024 * 1024,
        }),
    };

    let selected = select_variant(&recipe, &host, None).unwrap();
    assert_eq!(selected.variant.key.as_str(), "cuda-fp8");
}

#[test]
fn variant_selection_cuda_sm_insufficient() {
    let recipe = make_test_recipe();
    let host = HostProfile {
        backends: vec![Backend::Cuda],
        ram_bytes: 64 * 1024 * 1024 * 1024,
        cuda: Some(CudaInfo {
            sm: 80, // Needs 89
            vram_bytes: 24 * 1024 * 1024 * 1024,
        }),
    };

    let err = select_variant(&recipe, &host, None).unwrap_err();
    match err {
        HubError::NoCompatibleVariant { reasons, .. } => {
            assert!(reasons.contains("requires CUDA sm89"));
        }
        other => panic!("expected NoCompatibleVariant, got {other:?}"),
    }
}

#[test]
fn variant_selection_explicit_key_overcommit_warning() {
    let recipe = make_test_recipe();
    let host = HostProfile {
        backends: vec![Backend::Metal],
        ram_bytes: 4 * 1024 * 1024 * 1024, // 4GB RAM, variant needs 8GB
        cuda: None,
    };

    let explicit_key = VariantKey::parse("metal-mlx4").unwrap();
    let selected = select_variant(&recipe, &host, Some(&explicit_key)).unwrap();
    assert_eq!(selected.variant.key.as_str(), "metal-mlx4");
    assert!(selected.warning.is_some());
    assert!(selected.warning.unwrap().contains("overcommitting"));
}

#[test]
fn registry_endpoint_parse_normalizes_and_accepts_custom_urls() {
    // Any third-party or preview registry URL must work with no key material.
    let endpoint =
        RegistryEndpoint::parse("https://preview-abc.example.workers.dev/registry/v1").unwrap();
    assert_eq!(
        endpoint.url.as_str(),
        "https://preview-abc.example.workers.dev/registry/v1/"
    );

    let with_slash =
        RegistryEndpoint::parse("https://preview-abc.example.workers.dev/registry/v1/").unwrap();
    assert_eq!(endpoint, with_slash);

    assert!(RegistryEndpoint::parse("not a url").is_err());
}
#[test]
fn blob_store_hardlinks_and_gc() {
    let temp = tempdir().unwrap();
    let config = HubConfig::with_defaults(temp.path().to_path_buf());
    let store = BlobStore::new(config);

    let recipe = make_test_recipe();
    let variant = &recipe.variants[0];
    let manifest_sha =
        Sha256::parse("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855").unwrap();

    // Create blobs
    let blobs_dir = temp.path().join("blobs").join("sha256");
    fs::create_dir_all(&blobs_dir).unwrap();

    let comp = &variant.components[&Role::Base];
    for f in &comp.files {
        let bpath = blobs_dir.join(f.sha256.as_str());
        fs::write(&bpath, vec![0u8; usize::try_from(f.size).unwrap()]).unwrap();
    }

    // Install variant
    store
        .install_variant(&recipe, variant, &manifest_sha)
        .unwrap();

    // Verify installation
    assert!(store.is_up_to_date(&recipe.name, &variant.key, &manifest_sha, variant));

    // Verify files on disk are hardlinks
    let vdir = store.variant_dir(&recipe.name, &variant.key);
    assert!(vdir.join("recipe.json").is_file());
    assert!(vdir.join("installed.json").is_file());
    let model_file = vdir.join("base").join("model.safetensors");
    assert!(model_file.is_file());

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let blob_meta = fs::metadata(blobs_dir.join(comp.files[0].sha256.as_str())).unwrap();
        let target_meta = fs::metadata(&model_file).unwrap();
        assert_eq!(blob_meta.ino(), target_meta.ino());
        assert_eq!(blob_meta.nlink(), 2);
    }

    // Remove variant (triggers GC internally)
    store
        .remove_variant(&recipe.name, &variant.key, false)
        .unwrap();
    assert!(!vdir.is_dir());
    assert!(!blobs_dir.join(comp.files[0].sha256.as_str()).is_file());

    // Now create an unreferenced loose blob and test explicit run_gc
    let loose_blob =
        blobs_dir.join("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    fs::write(&loose_blob, b"hello loose blob").unwrap();
    assert!(loose_blob.is_file());

    let gc_stats = store.run_gc().unwrap();
    assert_eq!(gc_stats.deleted_blobs, 1);
    assert_eq!(gc_stats.reclaimed_bytes, 16);
    assert!(!loose_blob.is_file());
}
