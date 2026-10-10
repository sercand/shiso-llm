#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]

//! Mock HTTP integration tests (§9.3).

use std::fs;
use std::num::NonZeroUsize;

use sha2::{Digest, Sha256 as Sha256Hasher};
use tempfile::tempdir;
use wiremock::matchers::{header, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use shiso_hub::config::{HfToken, HubConfig, RegistryEndpoint};
use shiso_hub::download::{DownloadFileDesc, Downloader, FileOrigin};
use shiso_hub::error::HubError;
use shiso_hub::recipe::{FileEntry, Role, Sha256};
use shiso_hub::registry::RegistryClient;

#[tokio::test]
async fn cross_host_redirect_strips_authorization() {
    let host_a = MockServer::start().await;
    let host_b = MockServer::start().await;

    // Host B should NOT receive the Authorization header
    Mock::given(method("GET"))
        .and(path("/dest"))
        .and(header_exists("authorization"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&host_b)
        .await;

    Mock::given(method("GET"))
        .and(path("/dest"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"payload from host b"))
        .expect(1)
        .mount(&host_b)
        .await;

    let temp = tempdir().unwrap();
    let endpoint = RegistryEndpoint {
        url: url::Url::parse(&format!("{}/v1", host_a.uri())).unwrap(),
    };

    let config = HubConfig {
        home: temp.path().to_path_buf(),
        registry: endpoint,
        hf_endpoint: url::Url::parse(&host_a.uri()).unwrap(),
        hf_token: Some(HfToken::new("test-token-123")),
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    };

    let downloader = Downloader::new(config);
    let payload = b"payload from host b";
    let sha = hex::encode(Sha256Hasher::digest(payload));

    let file_entry = FileEntry {
        name: "test.bin".to_string(),
        size: payload.len() as u64,
        sha256: Sha256::parse(&sha).unwrap(),
        binds: None,
    };

    // Use our downloader to download a file where resolve redirects cross-host
    // Target URL path in Downloader::download_file is:
    // {hf_endpoint}/{repo}/resolve/{commit}/{name}
    // We map repo="redirect", commit="cross", name="test.bin"
    Mock::given(method("GET"))
        .and(path("/redirect/resolve/cross/test.bin"))
        .and(header("authorization", "Bearer test-token-123"))
        .respond_with(
            ResponseTemplate::new(302).insert_header("location", format!("{}/dest", host_b.uri())),
        )
        .expect(1)
        .mount(&host_a)
        .await;

    let res = downloader
        .download_file(DownloadFileDesc {
            model_name: "test-model",
            variant_key: "var",
            role: Role::Base,
            origin: FileOrigin::HuggingFace {
                repo: "redirect",
                revision: "cross",
            },
            file: &file_entry,
            progress_tx: None,
        })
        .await;
    assert!(
        res.is_ok(),
        "download with cross-host redirect succeeded: {res:?}"
    );
}

#[tokio::test]
async fn same_host_redirect_preserves_authorization() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/shiso/resolve/commit/file.txt"))
        .and(header("authorization", "Bearer secret-token"))
        .respond_with(ResponseTemplate::new(302).insert_header(
            "location",
            format!("{}/shiso/target/file.txt", server.uri()),
        ))
        .expect(1)
        .mount(&server)
        .await;

    let payload = b"same host payload";
    let sha = hex::encode(Sha256Hasher::digest(payload));

    Mock::given(method("GET"))
        .and(path("/shiso/target/file.txt"))
        .and(header("authorization", "Bearer secret-token"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(payload.as_slice()))
        .expect(1)
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let endpoint = RegistryEndpoint {
        url: url::Url::parse(&format!("{}/v1", server.uri())).unwrap(),
    };

    let config = HubConfig {
        home: temp.path().to_path_buf(),
        registry: endpoint,
        hf_endpoint: url::Url::parse(&server.uri()).unwrap(),
        hf_token: Some(HfToken::new("secret-token")),
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    };

    let downloader = Downloader::new(config);
    let file_entry = FileEntry {
        name: "file.txt".to_string(),
        size: payload.len() as u64,
        sha256: Sha256::parse(&sha).unwrap(),
        binds: None,
    };

    let res = downloader
        .download_file(DownloadFileDesc {
            model_name: "m",
            variant_key: "v",
            role: Role::Base,
            origin: FileOrigin::HuggingFace {
                repo: "shiso",
                revision: "commit",
            },
            file: &file_entry,
            progress_tx: None,
        })
        .await;
    assert!(res.is_ok());
}

#[tokio::test]
async fn etag_304_reuse() {
    let server = MockServer::start().await;

    let index_bytes =
        b"{\"schema_version\":1,\"generated_at\":\"2026-10-03T12:00:00Z\",\"models\":[]}";

    Mock::given(method("GET"))
        .and(path("/v1/index.json"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("etag", "\"etag-12345\"")
                .set_body_bytes(index_bytes.as_slice()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let endpoint = RegistryEndpoint {
        url: url::Url::parse(&format!("{}/v1", server.uri())).unwrap(),
    };

    let config = HubConfig {
        home: temp.path().to_path_buf(),
        registry: endpoint,
        hf_endpoint: url::Url::parse("https://huggingface.co").unwrap(),
        hf_token: None,
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    };

    let client = RegistryClient::new(config.clone());

    // First fetch: 200 OK
    let index1 = client.fetch_index().await.unwrap();
    assert_eq!(index1.models.len(), 0);

    // Second fetch with ETag -> 304 Not Modified
    server.reset().await;
    Mock::given(method("GET"))
        .and(path("/v1/index.json"))
        .and(header("if-none-match", "\"etag-12345\""))
        .respond_with(ResponseTemplate::new(304))
        .expect(1)
        .mount(&server)
        .await;

    let index2 = client.fetch_index().await.unwrap();
    assert_eq!(index2.models.len(), 0);
}

#[tokio::test]
async fn bit_flip_leads_to_digest_mismatch_and_cleanup() {
    let server = MockServer::start().await;

    let payload = b"correct payload bytes";
    let sha = hex::encode(Sha256Hasher::digest(payload));

    let corrupted_payload = b"corrupt payload bytes";

    Mock::given(method("GET"))
        .and(path("/repo/resolve/sha/corrupt.bin"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(corrupted_payload.as_slice()))
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let endpoint = RegistryEndpoint {
        url: url::Url::parse(&format!("{}/v1", server.uri())).unwrap(),
    };

    let config = HubConfig {
        home: temp.path().to_path_buf(),
        registry: endpoint,
        hf_endpoint: url::Url::parse(&server.uri()).unwrap(),
        hf_token: None,
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    };

    let downloader = Downloader::new(config);
    let file_entry = FileEntry {
        name: "corrupt.bin".to_string(),
        size: payload.len() as u64,
        sha256: Sha256::parse(&sha).unwrap(),
        binds: None,
    };

    let err = downloader
        .download_file(DownloadFileDesc {
            model_name: "m",
            variant_key: "v",
            role: Role::Base,
            origin: FileOrigin::HuggingFace {
                repo: "repo",
                revision: "sha",
            },
            file: &file_entry,
            progress_tx: None,
        })
        .await
        .unwrap_err();

    match err {
        HubError::DigestMismatch { file, .. } => {
            assert_eq!(file, "corrupt.bin");
        }
        other => panic!("expected DigestMismatch, got {other:?}"),
    }

    // Ensure the .part file was deleted
    let part_path = downloader.tmp_dir().join(format!("{sha}.part"));
    assert!(!part_path.exists());
}

#[tokio::test]
async fn range_resume_appends() {
    let server = MockServer::start().await;

    let full_payload = b"part1_data_part2_data";
    let part1 = b"part1_data_";
    let part2 = b"part2_data";
    let sha = hex::encode(Sha256Hasher::digest(full_payload));

    Mock::given(method("GET"))
        .and(path("/repo/resolve/sha/resume.bin"))
        .and(header("range", format!("bytes={}-", part1.len())))
        .respond_with(ResponseTemplate::new(206).set_body_bytes(part2.as_slice()))
        .expect(1)
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let endpoint = RegistryEndpoint {
        url: url::Url::parse(&format!("{}/v1", server.uri())).unwrap(),
    };

    let config = HubConfig {
        home: temp.path().to_path_buf(),
        registry: endpoint,
        hf_endpoint: url::Url::parse(&server.uri()).unwrap(),
        hf_token: None,
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    };

    let downloader = Downloader::new(config);
    let file_entry = FileEntry {
        name: "resume.bin".to_string(),
        size: full_payload.len() as u64,
        sha256: Sha256::parse(&sha).unwrap(),
        binds: None,
    };

    // Pre-create .part file with part1
    fs::create_dir_all(downloader.tmp_dir()).unwrap();
    let part_path = downloader.tmp_dir().join(format!("{sha}.part"));
    fs::write(&part_path, part1).unwrap();

    let res = downloader
        .download_file(DownloadFileDesc {
            model_name: "m",
            variant_key: "v",
            role: Role::Base,
            origin: FileOrigin::HuggingFace {
                repo: "repo",
                revision: "sha",
            },
            file: &file_entry,
            progress_tx: None,
        })
        .await;
    assert!(res.is_ok());

    // Verify committed blob has the full payload
    let blob_path = downloader.blob_path(&file_entry.sha256);
    let content = fs::read(blob_path).unwrap();
    assert_eq!(content, full_payload);
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_pulls_single_download() {
    let server = MockServer::start().await;

    let payload = b"shared blob bytes for concurrent test";
    let sha = hex::encode(Sha256Hasher::digest(payload));

    Mock::given(method("GET"))
        .and(path("/repo/resolve/sha/shared.bin"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(payload.as_slice()))
        .expect(1) // EXACTLY ONE download despite two concurrent tasks!
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let endpoint = RegistryEndpoint {
        url: url::Url::parse(&format!("{}/v1", server.uri())).unwrap(),
    };

    let config = HubConfig {
        home: temp.path().to_path_buf(),
        registry: endpoint,
        hf_endpoint: url::Url::parse(&server.uri()).unwrap(),
        hf_token: None,
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    };

    let downloader = Downloader::new(config);
    let file_entry = FileEntry {
        name: "shared.bin".to_string(),
        size: payload.len() as u64,
        sha256: Sha256::parse(&sha).unwrap(),
        binds: None,
    };

    let d1 = downloader.clone();
    let d2 = downloader.clone();
    let f1 = file_entry.clone();
    let f2 = file_entry.clone();

    let (res1, res2) = tokio::join!(
        tokio::spawn(async move {
            d1.download_file(DownloadFileDesc {
                model_name: "m",
                variant_key: "v1",
                role: Role::Base,
                origin: FileOrigin::HuggingFace {
                    repo: "repo",
                    revision: "sha",
                },
                file: &f1,
                progress_tx: None,
            })
            .await
        }),
        tokio::spawn(async move {
            d2.download_file(DownloadFileDesc {
                model_name: "m",
                variant_key: "v2",
                role: Role::Base,
                origin: FileOrigin::HuggingFace {
                    repo: "repo",
                    revision: "sha",
                },
                file: &f2,
                progress_tx: None,
            })
            .await
        })
    );

    res1.unwrap().unwrap();
    res2.unwrap().unwrap();

    let blob_path = downloader.blob_path(&file_entry.sha256);
    assert!(blob_path.is_file());
}

/// The manifest is fetched from the path the index names, not from a layout
/// the client assumes, and a path that would leave the registry is refused
/// before any request is made.
#[tokio::test]
async fn manifest_is_fetched_from_the_index_path() {
    use shiso_hub::recipe::IndexModelEntry;

    let server = MockServer::start().await;
    let manifest = br#"{"schema_version":1,"name":"m","family":"qwen35","min_shiso_version":"0.1.0","description":"d","modalities":{"inputs":["text"],"outputs":["text"]},"features":{},"variants":[{"key":"k","backends":["metal"],"quant":"q","requirements":{"min_ram_bytes":1,"disk_bytes":1},"components":{"base":{"provider":"huggingface","repo":"o/n","revision":"0123456789abcdef0123456789abcdef01234567","entry":"m.gguf","files":[{"name":"m.gguf","size":1,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]}}}]}"#;
    Mock::given(method("GET"))
        .and(path("/v1/models/m/index.json"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(manifest.as_slice()))
        .expect(1)
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let client = RegistryClient::new(HubConfig {
        home: temp.path().to_path_buf(),
        registry: RegistryEndpoint {
            url: url::Url::parse(&format!("{}/v1", server.uri())).unwrap(),
        },
        hf_endpoint: url::Url::parse("https://huggingface.co").unwrap(),
        hf_token: None,
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    });
    let entry = |manifest_path: &str| -> IndexModelEntry {
        serde_json::from_value(serde_json::json!({
            "name": "m",
            "family": "qwen35",
            "min_shiso_version": "0.1.0",
            "description": "d",
            "updated_at": "2026-10-05T00:00:00Z",
            "manifest": manifest_path,
            "manifest_sha256": hex::encode(Sha256Hasher::digest(manifest)),
            "variants": [],
        }))
        .unwrap()
    };
    let version = semver::Version::parse("0.1.0").unwrap();

    let recipe = client
        .fetch_manifest(&entry("models/m/index.json"), &version)
        .await
        .unwrap();
    assert_eq!(recipe.name.as_str(), "m");

    for bad in [
        "/models/m/index.json",
        "../m.json",
        "models//m.json",
        "https://evil/m.json",
    ] {
        let err = client
            .fetch_manifest(&entry(bad), &version)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("manifest path"), "{bad}: {err}");
    }
}

/// A manifest written for a newer client that this client's rules would also
/// refuse (a hub component carrying a `repo`): the answer is "upgrade", not
/// the rule.
#[tokio::test]
async fn a_recipe_for_a_newer_client_asks_for_an_upgrade() {
    use shiso_hub::recipe::IndexModelEntry;

    let server = MockServer::start().await;
    let manifest = br#"{"schema_version":1,"name":"m","family":"qwen35","min_shiso_version":"99.0.0","description":"d","modalities":{"inputs":["text"],"outputs":["text"]},"features":{},"variants":[{"key":"k","backends":["metal"],"quant":"q","requirements":{"min_ram_bytes":1,"disk_bytes":2},"server_defaults":{"spec_k":4},"components":{"base":{"provider":"huggingface","repo":"o/n","revision":"0123456789abcdef0123456789abcdef01234567","entry":"m.gguf","files":[{"name":"m.gguf","size":1,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]},"draft_vocab":{"provider":"shiso","repo":"o/r","entry":"v.txt","files":[{"name":"v.txt","size":1,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]}}}]}"#;
    Mock::given(method("GET"))
        .and(path("/v1/models/m/index.json"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(manifest.as_slice()))
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let client = RegistryClient::new(HubConfig {
        home: temp.path().to_path_buf(),
        registry: RegistryEndpoint {
            url: url::Url::parse(&format!("{}/v1", server.uri())).unwrap(),
        },
        hf_endpoint: url::Url::parse("https://huggingface.co").unwrap(),
        hf_token: None,
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    });
    let entry: IndexModelEntry = serde_json::from_value(serde_json::json!({
        "name": "m",
        "family": "qwen35",
        "min_shiso_version": "99.0.0",
        "description": "d",
        "updated_at": "2026-10-05T00:00:00Z",
        "manifest": "models/m/index.json",
        "manifest_sha256": hex::encode(Sha256Hasher::digest(manifest)),
        "variants": [],
    }))
    .unwrap();

    let err = client
        .fetch_manifest(&entry, &semver::Version::new(0, 2, 0))
        .await
        .unwrap_err();
    assert!(matches!(err, HubError::VersionMismatch { .. }), "{err:?}");
}

#[tokio::test]
async fn a_hub_file_is_fetched_beside_the_manifest_without_a_token() {
    let server = MockServer::start().await;
    let payload = b"198 3037739\n11 1468292\n";
    let sha = hex::encode(Sha256Hasher::digest(payload));
    // Any request carrying a token fails the test.
    Mock::given(method("GET"))
        .and(path("/v1/models/qwen-test/draft_vocab_counts.txt"))
        .and(header_exists("authorization"))
        .respond_with(ResponseTemplate::new(401))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/models/qwen-test/draft_vocab_counts.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(payload.as_slice()))
        .expect(1)
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let config = HubConfig {
        home: temp.path().to_path_buf(),
        // No trailing slash on purpose: the join must still land under /v1/.
        registry: RegistryEndpoint {
            url: url::Url::parse(&format!("{}/v1", server.uri())).unwrap(),
        },
        // Same host as the registry, and a token: the token must still stay home.
        hf_endpoint: url::Url::parse(&server.uri()).unwrap(),
        hf_token: Some(HfToken::new("secret-token")),
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    };
    let downloader = Downloader::new(config);
    let file = FileEntry {
        name: "draft_vocab_counts.txt".to_string(),
        size: payload.len() as u64,
        sha256: Sha256::parse(&sha).unwrap(),
        binds: None,
    };
    downloader
        .download_file(DownloadFileDesc {
            model_name: "qwen-test",
            variant_key: "v",
            role: Role::DraftVocab,
            origin: FileOrigin::Hub,
            file: &file,
            progress_tx: None,
        })
        .await
        .unwrap();
    assert_eq!(
        fs::read(downloader.blob_path(&file.sha256)).unwrap(),
        payload
    );
}

#[tokio::test]
async fn a_corrupt_hub_file_is_a_digest_mismatch_and_leaves_nothing() {
    let server = MockServer::start().await;
    let payload = b"good";
    let sha = hex::encode(Sha256Hasher::digest(payload));
    Mock::given(method("GET"))
        .and(path("/v1/models/m/f.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"bad!".as_slice()))
        .mount(&server)
        .await;
    let temp = tempdir().unwrap();
    let downloader = Downloader::new(HubConfig {
        home: temp.path().to_path_buf(),
        registry: RegistryEndpoint {
            url: url::Url::parse(&format!("{}/v1/", server.uri())).unwrap(),
        },
        hf_endpoint: url::Url::parse("https://hf.invalid").unwrap(),
        hf_token: None,
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    });
    let file = FileEntry {
        name: "f.txt".to_string(),
        size: 4,
        sha256: Sha256::parse(&sha).unwrap(),
        binds: None,
    };
    let err = downloader
        .download_file(DownloadFileDesc {
            model_name: "m",
            variant_key: "v",
            role: Role::DraftVocab,
            origin: FileOrigin::Hub,
            file: &file,
            progress_tx: None,
        })
        .await
        .unwrap_err();
    assert!(matches!(err, HubError::DigestMismatch { .. }), "{err:?}");
    assert!(!downloader.tmp_dir().join(format!("{sha}.part")).exists());
    assert!(!downloader.blob_path(&file.sha256).exists());
}

/// A recipe for a newer client that this client cannot even parse (a provider
/// it has never heard of) still answers "upgrade": the index entry already
/// says which version it needs.
#[tokio::test]
async fn a_recipe_this_client_cannot_parse_still_asks_for_an_upgrade() {
    use shiso_hub::recipe::IndexModelEntry;

    let server = MockServer::start().await;
    let manifest = br#"{"schema_version":1,"name":"m","family":"qwen35","min_shiso_version":"99.0.0","description":"d","modalities":{"inputs":["text"],"outputs":["text"]},"features":{},"variants":[{"key":"k","backends":["metal"],"quant":"q","requirements":{"min_ram_bytes":1,"disk_bytes":1},"components":{"base":{"provider":"s3","bucket":"b","files":[{"name":"m.gguf","size":1,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]}}}]}"#;
    Mock::given(method("GET"))
        .and(path("/v1/models/m/index.json"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(manifest.as_slice()))
        .mount(&server)
        .await;
    let temp = tempdir().unwrap();
    let client = RegistryClient::new(HubConfig {
        home: temp.path().to_path_buf(),
        registry: RegistryEndpoint {
            url: url::Url::parse(&format!("{}/v1", server.uri())).unwrap(),
        },
        hf_endpoint: url::Url::parse("https://huggingface.co").unwrap(),
        hf_token: None,
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    });
    let entry: IndexModelEntry = serde_json::from_value(serde_json::json!({
        "name": "m",
        "family": "qwen35",
        "min_shiso_version": "99.0.0",
        "description": "d",
        "updated_at": "2026-10-05T00:00:00Z",
        "manifest": "models/m/index.json",
        "manifest_sha256": hex::encode(Sha256Hasher::digest(manifest)),
        "variants": [],
    }))
    .unwrap();
    let err = client
        .fetch_manifest(&entry, &semver::Version::new(0, 2, 0))
        .await
        .unwrap_err();
    assert!(matches!(err, HubError::VersionMismatch { .. }), "{err:?}");
}

/// A nested hub file is fetched from `models/<name>/<nested name>` and lands at
/// `<variant dir>/<role dir>/<nested name>`, parent directories created.
#[tokio::test]
async fn a_nested_hub_file_is_fetched_and_installed_under_its_directory() {
    use shiso_hub::recipe::Recipe;
    use shiso_hub::store::BlobStore;

    let server = MockServer::start().await;
    let payload = br#"{"shift": 3.0}"#;
    let sha = hex::encode(Sha256Hasher::digest(payload));
    Mock::given(method("GET"))
        .and(path(
            "/v1/models/qwen-image-test/scheduler/scheduler_config.json",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(payload.as_slice()))
        .expect(1)
        .mount(&server)
        .await;

    let temp = tempdir().unwrap();
    let config = HubConfig {
        home: temp.path().to_path_buf(),
        registry: RegistryEndpoint {
            url: url::Url::parse(&format!("{}/v1", server.uri())).unwrap(),
        },
        hf_endpoint: url::Url::parse("https://hf.invalid").unwrap(),
        hf_token: None,
        max_parallel_files: NonZeroUsize::new(4).unwrap(),
    };
    let recipe: Recipe = serde_json::from_value(serde_json::json!({
        "schema_version": 1,
        "name": "qwen-image-test",
        "family": "qwen-image",
        "min_shiso_version": "0.2.0",
        "description": "d",
        "modalities": {"inputs": ["text"], "outputs": ["image"]},
        "features": {},
        "variants": [{
            "key": "k",
            "backends": ["metal"],
            "quant": "q",
            "requirements": {"min_ram_bytes": 1, "disk_bytes": payload.len()},
            "components": {"image_assets": {
                "provider": "shiso",
                "files": [{"name": "scheduler/scheduler_config.json",
                           "size": payload.len(), "sha256": sha}]
            }}
        }]
    }))
    .unwrap();
    let variant = &recipe.variants[0];

    Downloader::new(config.clone())
        .download_variant("qwen-image-test", variant, None)
        .await
        .unwrap();
    let store = BlobStore::new(config);
    let manifest_sha = Sha256::parse(&"0".repeat(64)).unwrap();
    store
        .install_variant(&recipe, variant, &manifest_sha)
        .unwrap();
    let installed = store
        .variant_dir(&recipe.name, &variant.key)
        .join("image_assets")
        .join("scheduler")
        .join("scheduler_config.json");
    assert_eq!(fs::read(installed).unwrap(), payload);
}
