#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]
#![expect(clippy::indexing_slicing, reason = "test fixture indexes")]

//! Unit tests for `shiso-hub` (§9.1, §9.3).

use std::collections::BTreeMap;
use std::fs;
use tempfile::tempdir;

use crate::config::{HfToken, HubConfig, RegistryEndpoint};
use crate::error::HubError;
use crate::host::{Backend, CudaInfo, HostProfile, MetalInfo, ServingDevice};
use crate::recipe::{
    CURRENT_SCHEMA_VERSION, CommitSha, Component, FileEntry, HUB_FILE_MAX_BYTES, HostConstraints,
    KvCacheDtype, ModalCapabilities, Modality, ModelName, ModelRef, Provider, Recipe,
    RecipeFeatures, Requirements, Role, ServerDefaults, Sha256, Task, Variant, VariantKey,
    hub_file_name_ok,
};
use crate::select::{fits_host, select_variant};
use crate::store::BlobStore;
use crate::validate::{ValidationMode, validate_recipe};

/// The development machine: family 7, 64 GB with a 55.6 GB working set.
fn m1_max() -> MetalInfo {
    MetalInfo {
        name: "Apple M1 Max".to_string(),
        family: 7,
        working_set_bytes: 55_662_788_608,
    }
}

fn make_test_recipe() -> Recipe {
    let mut components = BTreeMap::new();
    components.insert(
        Role::Base,
        Component {
            provider: Provider::HuggingFace,
            repo: Some("shiso/test-model".to_string()),
            revision: Some(CommitSha::parse("3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c").unwrap()),
            entry: None,
            files: vec![
                FileEntry {
                    name: "model.safetensors".to_string(),
                    size: 5000,
                    sha256: Sha256::parse(
                        "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a",
                    )
                    .unwrap(),
                    binds: None,
                },
                FileEntry {
                    name: "config.json".to_string(),
                    size: 100,
                    sha256: Sha256::parse(
                        "0d1f7c5b2e9a8b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4",
                    )
                    .unwrap(),
                    binds: None,
                },
            ],
        },
    );

    let variant_metal = Variant {
        key: VariantKey::parse("metal-mlx4").unwrap(),
        backends: vec![Backend::Metal],
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
        backends: vec![Backend::Cuda],
        quant: "fp8".to_string(),
        requires: Some(HostConstraints {
            min_cuda_sm: Some(89),
            min_metal_family: None,
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
        info: None,
        quality: Vec::new(),
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
        device: ServingDevice::Gpu,
        backends: vec![Backend::Metal],
        ram_bytes: 32 * 1024 * 1024 * 1024,
        cuda: None,
        metal: Some(m1_max()),
    };

    let selected = select_variant(&recipe, &host, None).unwrap();
    assert_eq!(selected.variant.key.as_str(), "metal-mlx4");
    assert!(selected.warning.is_none());
}

#[test]
fn variant_selection_cuda_host() {
    let recipe = make_test_recipe();
    let host = HostProfile {
        device: ServingDevice::Gpu,
        backends: vec![Backend::Cuda],
        ram_bytes: 64 * 1024 * 1024 * 1024,
        cuda: Some(CudaInfo {
            sm: 89,
            vram_bytes: 24 * 1024 * 1024 * 1024,
        }),
        metal: None,
    };

    let selected = select_variant(&recipe, &host, None).unwrap();
    assert_eq!(selected.variant.key.as_str(), "cuda-fp8");
}

#[test]
fn variant_selection_cuda_sm_insufficient() {
    let recipe = make_test_recipe();
    let host = HostProfile {
        device: ServingDevice::Gpu,
        backends: vec![Backend::Cuda],
        ram_bytes: 64 * 1024 * 1024 * 1024,
        cuda: Some(CudaInfo {
            sm: 80, // Needs 89
            vram_bytes: 24 * 1024 * 1024 * 1024,
        }),
        metal: None,
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
        device: ServingDevice::Gpu,
        backends: vec![Backend::Metal],
        ram_bytes: 4 * 1024 * 1024 * 1024, // 4GB RAM, variant needs 8GB
        cuda: None,
        metal: Some(m1_max()),
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

#[test]
fn backends_read_from_legacy_string_and_from_list() {
    let mut json = serde_json::to_value(make_test_recipe()).unwrap();
    let variant = &mut json["variants"][0];
    variant.as_object_mut().unwrap().remove("backends");
    variant["backend"] = serde_json::json!("metal");
    let legacy: Recipe = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(legacy.variants[0].backends, vec![Backend::Metal]);

    let variant = &mut json["variants"][0];
    variant.as_object_mut().unwrap().remove("backend");
    variant["backends"] = serde_json::json!(["metal", "cuda"]);
    let listed: Recipe = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(
        listed.variants[0].backends,
        vec![Backend::Metal, Backend::Cuda]
    );

    json["variants"][0]["backends"] = serde_json::json!(["vulkan"]);
    assert!(serde_json::from_value::<Recipe>(json).is_err());
}

#[test]
fn validation_refuses_empty_or_repeated_backends() {
    let mut recipe = make_test_recipe();
    recipe.variants[0].backends = vec![];
    assert!(validate_recipe(&recipe, ValidationMode::Strict).is_err());
    recipe.variants[0].backends = vec![Backend::Metal, Backend::Metal];
    assert!(validate_recipe(&recipe, ValidationMode::Strict).is_err());
}

#[test]
fn multi_backend_variant_selects_on_either_host_and_skips_cuda_checks_on_metal() {
    let mut recipe = make_test_recipe();
    recipe.variants.truncate(1);
    recipe.variants[0].backends = vec![Backend::Cuda, Backend::Metal];
    // A CUDA floor the Metal host cannot meet must not matter when the
    // variant would run on Metal there.
    recipe.variants[0].requires = Some(HostConstraints {
        min_cuda_sm: Some(120),
        min_metal_family: None,
    });

    let mac = HostProfile {
        device: ServingDevice::Gpu,
        backends: vec![Backend::Metal],
        cuda: None,
        ram_bytes: 64 * 1024 * 1024 * 1024,
        metal: Some(m1_max()),
    };
    assert_eq!(
        crate::select::host_backend(&recipe.variants[0], &mac).unwrap(),
        Backend::Metal
    );
    select_variant(&recipe, &mac, None).unwrap();

    let old_gpu = HostProfile {
        device: ServingDevice::Gpu,
        backends: vec![Backend::Cuda],
        cuda: Some(CudaInfo {
            sm: 89,
            vram_bytes: 48 * 1024 * 1024 * 1024,
        }),
        ram_bytes: 64 * 1024 * 1024 * 1024,
        metal: None,
    };
    assert!(select_variant(&recipe, &old_gpu, None).is_err());
}

/// One repository shipping the checkpoint, its draft head and its tower, and
/// a second one carrying the config documents.
fn single_repo_variant() -> Variant {
    let file = |name: &str, binds: Option<Role>| FileEntry {
        name: name.to_string(),
        size: 10,
        sha256: Sha256::parse("4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a")
            .unwrap(),
        binds,
    };
    let mut components = BTreeMap::new();
    components.insert(
        Role::Base,
        Component {
            provider: Provider::HuggingFace,
            repo: Some("shiso/test-gguf".to_string()),
            revision: Some(CommitSha::parse("3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c").unwrap()),
            entry: Some("model.gguf".to_string()),
            files: vec![
                file("model.gguf", None),
                file("mtp.gguf", Some(Role::Draft)),
                file("vision.gguf", Some(Role::Vision)),
            ],
        },
    );
    components.insert(
        Role::Config,
        Component {
            provider: Provider::HuggingFace,
            repo: Some("shiso/test-upstream".to_string()),
            revision: Some(CommitSha::parse("3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c").unwrap()),
            entry: Some("config.json".to_string()),
            files: vec![file("config.json", None), file("tokenizer.json", None)],
        },
    );
    Variant {
        key: VariantKey::parse("cuda-gguf").unwrap(),
        backends: vec![Backend::Cuda],
        quant: "q4".to_string(),
        requires: None,
        requirements: Requirements {
            min_ram_bytes: 1,
            recommended_ram_bytes: None,
            min_vram_bytes: None,
            disk_bytes: 50,
        },
        components,
        server_defaults: Some(ServerDefaults {
            spec_k: Some(4),
            ..Default::default()
        }),
    }
}

#[test]
fn bound_files_supply_their_roles() {
    let variant = single_repo_variant();
    let roles = variant.role_paths().unwrap();
    let place = |role| roles[&role];
    assert_eq!(place(Role::Base).dir, Role::Base);
    assert_eq!(place(Role::Base).file, Some("model.gguf"));
    assert_eq!(place(Role::Draft).dir, Role::Base);
    assert_eq!(place(Role::Draft).file, Some("mtp.gguf"));
    assert_eq!(place(Role::Vision).file, Some("vision.gguf"));
    assert_eq!(place(Role::Config).dir, Role::Config);
    assert_eq!(place(Role::Config).file, Some("config.json"));

    let mut recipe = make_test_recipe();
    recipe.variants = vec![variant];
    validate_recipe(&recipe, ValidationMode::Strict).unwrap();
}

#[test]
fn bound_draft_still_needs_a_speculation_depth() {
    let mut variant = single_repo_variant();
    variant.server_defaults = None;
    let mut recipe = make_test_recipe();
    recipe.variants = vec![variant];
    let err = validate_recipe(&recipe, ValidationMode::Strict).unwrap_err();
    assert!(err.to_string().contains("spec_k"), "{err}");
}

#[test]
fn a_role_is_supplied_once() {
    let mut variant = single_repo_variant();
    // The config component already supplies `config`.
    variant.components.get_mut(&Role::Base).unwrap().files[1].binds = Some(Role::Config);
    assert!(variant.role_paths().is_err());

    let mut variant = single_repo_variant();
    variant.components.get_mut(&Role::Base).unwrap().files[1].binds = Some(Role::Base);
    assert!(variant.role_paths().is_err());
}

#[test]
fn config_role_must_name_its_file() {
    let mut variant = single_repo_variant();
    variant.components.get_mut(&Role::Config).unwrap().entry = None;
    let mut recipe = make_test_recipe();
    recipe.variants = vec![variant];
    assert!(validate_recipe(&recipe, ValidationMode::Strict).is_err());
}

#[test]
fn kv_cache_dtype_is_typed() {
    let parsed: ServerDefaults = serde_json::from_str(r#"{"kv_cache_dtype": "fp8"}"#).unwrap();
    assert_eq!(parsed.kv_cache_dtype, Some(KvCacheDtype::Fp8));
    assert!(serde_json::from_str::<ServerDefaults>(r#"{"kv_cache_dtype": "int4"}"#).is_err());
}

#[test]
fn expert_residency_is_typed() {
    let parsed: ServerDefaults =
        serde_json::from_str(r#"{"expert_residency": "streamed"}"#).unwrap();
    assert_eq!(
        parsed.expert_residency,
        Some(crate::recipe::ExpertResidency::Stream)
    );
    assert!(serde_json::from_str::<ServerDefaults>(r#"{"expert_residency": "ssd"}"#).is_err());
}

#[test]
fn metal_family_and_working_set_gate_selection() {
    let mut recipe = make_test_recipe();
    recipe.variants.truncate(1);
    let mac = HostProfile {
        device: ServingDevice::Gpu,
        backends: vec![Backend::Metal],
        cuda: None,
        ram_bytes: 64 * 1024 * 1024 * 1024,
        metal: Some(m1_max()),
    };

    // An M2-or-newer variant is refused on the M1 Max, auto or explicit.
    recipe.variants[0].requires = Some(HostConstraints {
        min_cuda_sm: None,
        min_metal_family: Some(8),
    });
    let err = select_variant(&recipe, &mac, None).unwrap_err().to_string();
    assert!(err.contains("family 8"), "{err}");
    let key = recipe.variants[0].key.clone();
    assert!(select_variant(&recipe, &mac, Some(&key)).is_err());

    // Family 7 is enough.
    recipe.variants[0].requires = Some(HostConstraints {
        min_cuda_sm: None,
        min_metal_family: Some(7),
    });
    select_variant(&recipe, &mac, None).unwrap();

    // The working set is the device-memory floor on Metal.
    recipe.variants[0].requirements.min_vram_bytes = Some(60 * 1024 * 1024 * 1024);
    let err = select_variant(&recipe, &mac, None).unwrap_err().to_string();
    assert!(err.contains("working set"), "{err}");

    // No Metal device at all is refused, not assumed.
    let headless = HostProfile { metal: None, ..mac };
    recipe.variants[0].requirements.min_vram_bytes = None;
    assert!(select_variant(&recipe, &headless, None).is_err());
}

#[test]
fn ngram_role_names_its_flag_and_must_name_a_file() {
    assert_eq!("ngram".parse::<Role>().unwrap(), Role::Ngram);
    assert_eq!(Role::Ngram.flag_name(), "--ngram");

    let mut variant = single_repo_variant();
    let base = variant.components.get_mut(&Role::Base).unwrap();
    base.files.push(FileEntry {
        name: "ngram.safetensors".to_string(),
        size: 10,
        sha256: base.files[0].sha256.clone(),
        binds: Some(Role::Ngram),
    });
    variant.requirements.disk_bytes += 10;
    assert_eq!(
        variant.role_paths().unwrap()[&Role::Ngram].file,
        Some("ngram.safetensors")
    );
    let mut recipe = make_test_recipe();
    recipe.variants = vec![variant];
    validate_recipe(&recipe, ValidationMode::Strict).unwrap();
}

/// An FP8 transformer file from one repository and the BF16 tree's text
/// encoder, VAE and tokenizer from another, as a directory.
fn image_variant() -> Variant {
    let file = |name: &str| FileEntry {
        name: name.to_string(),
        size: 10,
        sha256: Sha256::parse("4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a")
            .unwrap(),
        binds: None,
    };
    let component = |repo: &str, entry: Option<&str>, files: Vec<FileEntry>| Component {
        provider: Provider::HuggingFace,
        repo: Some(repo.to_string()),
        revision: Some(CommitSha::parse("3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c").unwrap()),
        entry: entry.map(str::to_string),
        files,
    };
    let mut components = BTreeMap::new();
    components.insert(
        Role::Image,
        component(
            "shiso/test-fp8",
            Some("model-fp8.safetensors"),
            vec![file("model-fp8.safetensors")],
        ),
    );
    components.insert(
        Role::ImageAssets,
        component(
            "shiso/test-bf16",
            None,
            vec![
                file("transformer/config.json"),
                file("vae/config.json"),
                file("tokenizer/tokenizer.json"),
            ],
        ),
    );
    Variant {
        key: VariantKey::parse("fp8").unwrap(),
        backends: vec![Backend::Metal, Backend::Cuda],
        quant: "fp8".to_string(),
        requires: None,
        requirements: Requirements {
            min_ram_bytes: 1,
            recommended_ram_bytes: None,
            min_vram_bytes: None,
            disk_bytes: 40,
        },
        components,
        server_defaults: None,
    }
}

#[test]
fn image_assets_role_is_a_directory_beside_an_image() {
    assert_eq!("image_assets".parse::<Role>().unwrap(), Role::ImageAssets);
    assert_eq!(Role::ImageAssets.flag_name(), "--image-assets-dir");
    let mut recipe = make_test_recipe();
    recipe.variants = vec![image_variant()];
    validate_recipe(&recipe, ValidationMode::Strict).unwrap();

    // A file is not a directory the locator can search.
    let mut variant = image_variant();
    variant
        .components
        .get_mut(&Role::ImageAssets)
        .unwrap()
        .entry = Some("vae/config.json".to_string());
    recipe.variants = vec![variant];
    let err = validate_recipe(&recipe, ValidationMode::Strict).unwrap_err();
    assert!(err.to_string().contains("directory"), "{err}");

    // Assets with no image model have nothing to feed.
    let mut variant = image_variant();
    let assets = variant.components.remove(&Role::ImageAssets).unwrap();
    variant.components.remove(&Role::Image);
    variant.components.insert(Role::ImageAssets, assets);
    variant.components.insert(
        Role::Base,
        make_test_recipe().variants[0].components[&Role::Base].clone(),
    );
    recipe.variants = vec![variant];
    assert!(validate_recipe(&recipe, ValidationMode::Strict).is_err());
}

/// An image variant whose text encoder and VAE are files of one more repository
/// bound to their roles, as a single-repo image recipe ships them.
fn split_image_variant() -> Variant {
    let mut variant = image_variant();
    variant.components.remove(&Role::ImageAssets);
    let image = variant.components.get_mut(&Role::Image).unwrap();
    let sha = image.files[0].sha256.clone();
    for (name, role) in [
        ("te.safetensors", Role::ImageTextEncoder),
        ("vae.safetensors", Role::ImageVae),
    ] {
        image.files.push(FileEntry {
            name: name.to_string(),
            size: 10,
            sha256: sha.clone(),
            binds: Some(role),
        });
    }
    variant.requirements.disk_bytes = 30;
    variant
}

#[test]
fn image_text_encoder_and_vae_roles_bind_files_beside_an_image() {
    assert_eq!(
        "image_text_encoder".parse::<Role>().unwrap(),
        Role::ImageTextEncoder
    );
    assert_eq!("image_vae".parse::<Role>().unwrap(), Role::ImageVae);
    assert_eq!(Role::ImageTextEncoder.flag_name(), "--image-text-encoder");
    assert_eq!(Role::ImageVae.flag_name(), "--image-vae");

    let mut recipe = make_test_recipe();
    let variant = split_image_variant();
    let paths = variant.role_paths().unwrap();
    assert_eq!(paths[&Role::ImageTextEncoder].file, Some("te.safetensors"));
    assert_eq!(paths[&Role::ImageVae].file, Some("vae.safetensors"));
    recipe.variants = vec![variant];
    validate_recipe(&recipe, ValidationMode::Strict).unwrap();

    // Neither has anything to feed without an image model.
    for role in [Role::ImageTextEncoder, Role::ImageVae] {
        let mut variant = split_image_variant();
        let image = variant.components.remove(&Role::Image).unwrap();
        let mut stray = image.clone();
        stray.entry = None;
        stray.files.retain(|f| f.binds == Some(role));
        variant.components.insert(Role::Vision, stray);
        variant.components.insert(
            Role::Base,
            make_test_recipe().variants[0].components[&Role::Base].clone(),
        );
        recipe.variants = vec![variant];
        let err = validate_recipe(&recipe, ValidationMode::Strict).unwrap_err();
        assert!(err.to_string().contains(role.as_str()), "{err}");
    }

    // A directory is not a file the loader can open.
    let mut variant = image_variant();
    variant.components.insert(
        Role::ImageTextEncoder,
        variant.components[&Role::ImageAssets].clone(),
    );
    recipe.variants = vec![variant];
    let err = validate_recipe(&recipe, ValidationMode::Strict).unwrap_err();
    assert!(err.to_string().contains("image_text_encoder"), "{err}");

    // One role, one source.
    let mut variant = split_image_variant();
    let mut vae = variant.components[&Role::Image].clone();
    vae.entry = None;
    vae.files.retain(|f| f.binds == Some(Role::ImageVae));
    variant.components.insert(Role::ImageVae, vae);
    recipe.variants = vec![variant];
    let err = validate_recipe(&recipe, ValidationMode::Strict).unwrap_err();
    assert!(err.to_string().contains("image_vae"), "{err}");
}

/// An image variant whose prompt tokenizer is a `tokenizer.json` of the
/// image repository bound to `image_tokenizer`, beside the assets.
fn tokenizer_image_variant() -> Variant {
    let mut variant = image_variant();
    let image = variant.components.get_mut(&Role::Image).unwrap();
    let sha = image.files[0].sha256.clone();
    image.files.push(FileEntry {
        name: "tokenizer.json".to_string(),
        size: 10,
        sha256: sha,
        binds: Some(Role::ImageTokenizer),
    });
    variant.requirements.disk_bytes = 50;
    variant
}

#[test]
fn image_tokenizer_role_binds_a_file_beside_an_image() {
    assert_eq!(
        "image_tokenizer".parse::<Role>().unwrap(),
        Role::ImageTokenizer
    );
    assert_eq!(Role::ImageTokenizer.as_str(), "image_tokenizer");
    assert_eq!(Role::ImageTokenizer.flag_name(), "--image-tokenizer");

    let mut recipe = make_test_recipe();
    let variant = tokenizer_image_variant();
    let paths = variant.role_paths().unwrap();
    assert_eq!(paths[&Role::ImageTokenizer].dir, Role::Image);
    assert_eq!(paths[&Role::ImageTokenizer].file, Some("tokenizer.json"));
    recipe.variants = vec![variant];
    validate_recipe(&recipe, ValidationMode::Strict).unwrap();

    // Nothing to tokenize for without an image model.
    let mut variant = tokenizer_image_variant();
    variant.components.remove(&Role::ImageAssets);
    let image = variant.components.remove(&Role::Image).unwrap();
    let mut stray = image.clone();
    stray.entry = None;
    stray
        .files
        .retain(|f| f.binds == Some(Role::ImageTokenizer));
    variant.components.insert(Role::Vision, stray);
    variant.components.insert(
        Role::Base,
        make_test_recipe().variants[0].components[&Role::Base].clone(),
    );
    recipe.variants = vec![variant];
    let err = validate_recipe(&recipe, ValidationMode::Strict).unwrap_err();
    assert!(err.to_string().contains("image_tokenizer"), "{err}");

    // A directory is not the file the flag names.
    let mut variant = image_variant();
    variant.components.insert(
        Role::ImageTokenizer,
        variant.components[&Role::ImageAssets].clone(),
    );
    recipe.variants = vec![variant];
    let err = validate_recipe(&recipe, ValidationMode::Strict).unwrap_err();
    assert!(err.to_string().contains("image_tokenizer"), "{err}");

    // A component of its own naming the file by entry is the other source.
    let mut variant = image_variant();
    let mut own = variant.components[&Role::ImageAssets].clone();
    own.entry = Some("tokenizer/tokenizer.json".to_string());
    variant.components.insert(Role::ImageTokenizer, own);
    variant.requirements.disk_bytes = 70;
    recipe.variants = vec![variant];
    validate_recipe(&recipe, ValidationMode::Strict).unwrap();

    // One role, one source.
    let mut variant = tokenizer_image_variant();
    let mut own = variant.components[&Role::Image].clone();
    own.entry = None;
    own.files.retain(|f| f.binds == Some(Role::ImageTokenizer));
    variant.components.insert(Role::ImageTokenizer, own);
    recipe.variants = vec![variant];
    let err = validate_recipe(&recipe, ValidationMode::Strict).unwrap_err();
    assert!(err.to_string().contains("image_tokenizer"), "{err}");
}

#[test]
fn info_and_quality_are_checked() {
    use crate::recipe::{CardSource, CommitSha, ModelInfo, QualityScope, QualityScore};
    let score = |scope, variant: Option<&str>| QualityScore {
        source: "Some Lab".to_string(),
        metric: "Index".to_string(),
        value: 41.0,
        max: Some(100.0),
        url: "https://example.com/model".to_string(),
        as_of: "2026-10-05".to_string(),
        scope,
        variant: variant.map(|k| VariantKey::parse(k).unwrap()),
        subject: None,
    };
    let mut recipe = make_test_recipe();
    recipe.info = Some(ModelInfo {
        publisher: Some("Someone".to_string()),
        license: Some("apache-2.0".to_string()),
        homepage: Some("https://example.com".to_string()),
        card: Some(CardSource {
            provider: "huggingface".to_string(),
            repo: "owner/name".to_string(),
            revision: CommitSha::parse("0123456789abcdef0123456789abcdef01234567").unwrap(),
        }),
        parameters: Some(27_000_000_000),
        active_parameters: Some(3_000_000_000),
        context_length: Some(262_144),
    });
    let key = recipe.variants[0].key.to_string();
    recipe.quality = vec![
        score(QualityScope::BaseModel, None),
        score(QualityScope::Variant, Some(&key)),
    ];
    validate_recipe(&recipe, ValidationMode::Strict).unwrap();

    let refused = |mutate: &dyn Fn(&mut Recipe)| {
        let mut r = recipe.clone();
        mutate(&mut r);
        validate_recipe(&r, ValidationMode::Strict).is_err()
    };
    assert!(refused(
        &|r| r.info.as_mut().unwrap().homepage = Some("javascript:alert(1)".into())
    ));
    assert!(refused(
        &|r| r.info.as_mut().unwrap().active_parameters = Some(u64::MAX)
    ));
    assert!(refused(&|r| r
        .info
        .as_mut()
        .unwrap()
        .card
        .as_mut()
        .unwrap()
        .repo = "name".into()));
    assert!(refused(&|r| r.quality[0].as_of = "Oct 2026".into()));
    assert!(refused(&|r| r.quality[0].value = 101.0));
    assert!(refused(
        &|r| r.quality[0].variant = r.quality[1].variant.clone()
    ));
    assert!(refused(
        &|r| r.quality[1].variant = Some(VariantKey::parse("nope").unwrap())
    ));
    assert!(refused(&|r| r.quality[1].variant = None));
}

#[test]
fn a_huggingface_component_round_trips_byte_for_byte() {
    let text = include_str!("../../shiso-registry/tests/fixtures/qwen3.8-flash.json");
    let recipe: Recipe = serde_json::from_str(text).unwrap();
    let again = serde_json::to_string_pretty(&recipe).unwrap();
    let reparsed: Recipe = serde_json::from_str(&again).unwrap();
    assert_eq!(recipe, reparsed);
    let comp = &recipe.variants[0].components[&Role::Base];
    assert_eq!(comp.provider, Provider::HuggingFace);
    let (repo, rev) = comp.hf_source().unwrap();
    assert!(repo.contains('/'));
    assert_eq!(rev.as_str().len(), 40);
    // `repo` and `revision` are written exactly where they were.
    assert!(again.contains("\"repo\": \"") && again.contains("\"revision\": \""));
}

#[test]
fn a_hub_component_parses_without_repo_or_revision() {
    let comp: Component = serde_json::from_str(
        r#"{"provider": "shiso", "entry": "v.txt",
            "files": [{"name": "v.txt", "size": 3,
              "sha256": "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a"}]}"#,
    )
    .unwrap();
    assert_eq!(comp.provider, Provider::Shiso);
    assert!(comp.hf_source().is_none());
    let out = serde_json::to_string(&comp).unwrap();
    assert!(!out.contains("repo") && !out.contains("revision"), "{out}");
}

#[test]
fn an_unknown_provider_does_not_parse() {
    let res: Result<Component, _> = serde_json::from_str(r#"{"provider": "s3", "files": []}"#);
    assert!(res.is_err());
}

#[test]
fn hub_file_names_may_nest_up_to_four_safe_segments() {
    for ok in [
        "draft_vocab_counts.txt",
        "scheduler/scheduler_config.json",
        "a/b/c/d.json",
        "model_index.json",
    ] {
        assert!(hub_file_name_ok(ok), "{ok}");
    }
    let long_segment = "a".repeat(256);
    for bad in [
        "",
        "..",
        ".",
        "index.json",
        "benchmarks.json",
        "index.json/x",
        "benchmarks.json/x",
        "/x",
        "/a",
        "a/",
        "a//b",
        "../a",
        "a/../b",
        "a/./b",
        "a\\b",
        ".hidden/x",
        "a/.hidden",
        "a/b/c/d/e.json",
        long_segment.as_str(),
    ] {
        assert!(!hub_file_name_ok(bad), "{bad}");
    }
    // Only the first segment is reserved: a nested `index.json` is a file.
    assert!(hub_file_name_ok("vae/index.json"));
}

#[test]
fn a_file_and_a_directory_of_the_same_name_conflict() {
    use crate::recipe::hub_file_set_conflict;
    assert_eq!(hub_file_set_conflict(&["a", "a/b"]), Some(("a", "a/b")));
    assert_eq!(hub_file_set_conflict(&["a/b", "a"]), Some(("a", "a/b")));
    assert_eq!(hub_file_set_conflict(&["a", "ab/c", "a.b"]), None);
    assert_eq!(hub_file_set_conflict(&["s/a.json", "s/b.json"]), None);
}

/// A variant whose base speculates through its own built-in head (no `draft`
/// role), with a hub-hosted draft vocabulary.
fn hub_vocab_variant() -> Variant {
    let mut variant = single_repo_variant();
    // Drop the bound draft so only `spec_k` says the variant speculates.
    variant.components.get_mut(&Role::Base).unwrap().files[1].binds = None;
    variant.components.insert(
        Role::DraftVocab,
        Component {
            provider: Provider::Shiso,
            repo: None,
            revision: None,
            entry: Some("draft_vocab_counts.txt".to_string()),
            files: vec![FileEntry {
                name: "draft_vocab_counts.txt".to_string(),
                size: 10,
                sha256: Sha256::parse(
                    "4b6f1e3a479a044b7d1be20042a969f6e01a88ccb21f37e19392e2d9b69b567a",
                )
                .unwrap(),
                binds: None,
            }],
        },
    );
    variant.requirements.disk_bytes += 10;
    variant
}

fn hub_recipe(variant: Variant) -> Recipe {
    let mut recipe = make_test_recipe();
    recipe.min_shiso_version = semver::Version::new(0, 2, 0);
    recipe.variants = vec![variant];
    recipe
}

#[test]
fn a_nested_hub_file_validates_but_a_file_and_its_directory_do_not() {
    let mut variant = hub_vocab_variant();
    let comp = variant.components.get_mut(&Role::DraftVocab).unwrap();
    comp.files[0].name = "scheduler/scheduler_config.json".to_string();
    comp.entry = Some(comp.files[0].name.clone());
    validate_recipe(&hub_recipe(variant.clone()), ValidationMode::Strict).unwrap();

    let comp = variant.components.get_mut(&Role::DraftVocab).unwrap();
    let mut clash = comp.files[0].clone();
    clash.name = "scheduler".to_string();
    comp.files.push(clash);
    let err = validate_recipe(&hub_recipe(variant), ValidationMode::Strict).unwrap_err();
    assert!(err.to_string().contains("beneath"), "{err}");
}

#[test]
fn a_hub_draft_vocabulary_beside_a_speculating_base_validates() {
    validate_recipe(&hub_recipe(hub_vocab_variant()), ValidationMode::Strict).unwrap();
}

#[test]
fn a_draft_vocabulary_needs_a_draft_or_a_speculation_depth() {
    let mut variant = hub_vocab_variant();
    variant.server_defaults = None;
    let err = validate_recipe(&hub_recipe(variant), ValidationMode::Strict).unwrap_err();
    assert!(err.to_string().contains("draft_vocab"), "{err}");
}

#[test]
fn a_draft_vocabulary_is_a_file() {
    let mut variant = hub_vocab_variant();
    variant.components.get_mut(&Role::DraftVocab).unwrap().entry = None;
    assert!(validate_recipe(&hub_recipe(variant), ValidationMode::Strict).is_err());
}

#[test]
fn a_hub_component_refuses_repo_revision_deep_names_and_big_files() {
    let cases: [fn(&mut Component); 5] = [
        |c| c.repo = Some("o/r".to_string()),
        |c| {
            c.revision =
                Some(CommitSha::parse("3f1c2a9d8e7b6a5c4d3e2f1a0b9c8d7e6f5a4b3c").unwrap());
        },
        |c| {
            c.files[0].name = "a/b/c/d/v.txt".to_string();
            c.entry = Some("a/b/c/d/v.txt".to_string());
        },
        |c| {
            c.files[0].name = "index.json".to_string();
            c.entry = Some("index.json".to_string());
        },
        |c| c.files[0].size = HUB_FILE_MAX_BYTES + 1,
    ];
    for (i, edit) in cases.iter().enumerate() {
        let mut variant = hub_vocab_variant();
        edit(variant.components.get_mut(&Role::DraftVocab).unwrap());
        let size = variant.components[&Role::DraftVocab].files[0].size;
        variant.requirements.disk_bytes = variant.requirements.disk_bytes - 10 + size;
        assert!(
            validate_recipe(&hub_recipe(variant), ValidationMode::Strict).is_err(),
            "case {i} should be refused"
        );
    }
}

#[test]
fn a_huggingface_component_still_needs_repo_and_revision() {
    let mut variant = hub_vocab_variant();
    variant.components.get_mut(&Role::Base).unwrap().revision = None;
    assert!(validate_recipe(&hub_recipe(variant), ValidationMode::Strict).is_err());
}

#[test]
fn hub_files_need_a_client_that_knows_them() {
    let mut recipe = hub_recipe(hub_vocab_variant());
    recipe.min_shiso_version = semver::Version::new(0, 1, 0);
    let err = validate_recipe(&recipe, ValidationMode::Client).unwrap_err();
    assert!(err.to_string().contains("min_shiso_version"), "{err}");
}

#[test]
fn the_hub_component_supplies_its_role_through_its_entry() {
    let variant = hub_vocab_variant();
    let roles = variant.role_paths().unwrap();
    assert_eq!(roles[&Role::DraftVocab].dir, Role::DraftVocab);
    assert_eq!(
        roles[&Role::DraftVocab].file,
        Some("draft_vocab_counts.txt")
    );
}

#[test]
fn hub_file_names_are_plain_url_and_git_safe() {
    for ok in ["draft_vocab_counts.txt", "v1.2-x_y.txt", "A9"] {
        assert!(hub_file_name_ok(ok), "{ok}");
    }
    for bad in [
        "my vocab.txt",
        "a?b.txt",
        "a#b.txt",
        "%41.txt",
        ".hidden",
        "-x.txt",
        "çok.txt",
        "a\tb",
    ] {
        assert!(!hub_file_name_ok(bad), "{bad:?}");
    }
}

/// The published `smollm2-135m-decision` shape: one Metal-only variant.
fn metal_only_recipe(mutate: impl FnOnce(&mut Variant)) -> Recipe {
    let mut recipe = make_test_recipe();
    recipe.variants.truncate(1);
    let variant = &mut recipe.variants[0];
    variant.key = VariantKey::parse("q8_0").unwrap();
    variant.backends = vec![Backend::Metal];
    variant.quant = "Q8_0".to_string();
    variant.requires = None;
    variant.requirements = Requirements {
        min_ram_bytes: 2 * 1024 * 1024 * 1024,
        recommended_ram_bytes: Some(4 * 1024 * 1024 * 1024),
        min_vram_bytes: None,
        disk_bytes: 176_100_588,
    };
    mutate(variant);
    recipe
}

fn cpu_host(ram_gib: u64) -> HostProfile {
    let mut host = HostProfile::probe(vec![], None, None).on(ServingDevice::Cpu);
    host.ram_bytes = ram_gib * 1024 * 1024 * 1024;
    host
}

#[test]
fn a_cpu_host_takes_a_metal_only_variant() {
    let recipe = metal_only_recipe(|_| {});
    let selected = select_variant(&recipe, &cpu_host(8), None).unwrap();
    assert_eq!(selected.variant.key.as_str(), "q8_0");
    assert!(selected.explanation.contains("on the CPU device"));
    // The same recipe on a GPU host with no backend is still refused.
    let gpu = HostProfile::probe(vec![], None, None);
    assert!(select_variant(&recipe, &gpu, None).is_err());
}

#[test]
fn a_cpu_host_still_applies_ram_fit() {
    let recipe = metal_only_recipe(|_| {});
    match select_variant(&recipe, &cpu_host(1), None).unwrap_err() {
        HubError::NoCompatibleVariant { reasons, .. } => {
            assert!(reasons.contains("RAM"), "{reasons}");
        }
        other => panic!("expected NoCompatibleVariant, got {other:?}"),
    }
}

#[test]
fn a_cpu_host_skips_metal_family_and_cuda_sm() {
    let recipe = metal_only_recipe(|v| {
        v.requires = Some(HostConstraints {
            min_cuda_sm: Some(89),
            min_metal_family: Some(9),
        });
        v.requirements.min_vram_bytes = Some(48 * 1024 * 1024 * 1024);
    });
    let selected = select_variant(&recipe, &cpu_host(8), None).unwrap();
    assert_eq!(selected.variant.key.as_str(), "q8_0");
}

#[test]
fn a_cpu_host_takes_an_explicit_key_without_gpu_checks() {
    let recipe = metal_only_recipe(|v| {
        v.requires = Some(HostConstraints {
            min_cuda_sm: Some(89),
            min_metal_family: Some(9),
        });
    });
    let key = VariantKey::parse("q8_0").unwrap();
    let selected = select_variant(&recipe, &cpu_host(8), Some(&key)).unwrap();
    assert_eq!(selected.variant.key.as_str(), "q8_0");
    assert!(selected.warning.is_none());
}

#[test]
fn fits_host_is_true_for_every_variant_on_cpu() {
    let recipe = make_test_recipe();
    let cpu = cpu_host(1);
    let gpu = HostProfile::probe(vec![], None, None);
    for variant in &recipe.variants {
        assert!(fits_host(variant, &cpu));
        assert!(!fits_host(variant, &gpu));
    }
}
