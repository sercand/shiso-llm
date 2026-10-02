//! Tests for registry editor (§9.5).

#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]

use std::fs;
use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use ed25519_dalek::Verifier;
use tempfile::tempdir;
use tower::ServiceExt;

use crate::editor;
use crate::server::{ServerState, build_router};

#[test]
fn checked_in_recipes_pass_strict_validation() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let reports = editor::validate_all(root).unwrap();

    assert!(
        !reports.is_empty(),
        "must find at least one recipe in registry/recipes/"
    );
    for r in reports {
        assert!(
            r.valid,
            "recipe '{}' in '{}' failed strict validation: {}",
            r.name,
            r.file,
            r.errors.join("; ")
        );
    }
}

#[test]
fn checked_in_public_key_matches_compiled_constant() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let pubkey_path = root.join("registry").join("keys").join("registry.pub");

    let content = fs::read_to_string(&pubkey_path).unwrap();
    let trimmed = content.trim();
    let expected = hex::encode(shiso_hub::config::COMPILED_REGISTRY_PUBKEY);
    assert_eq!(
        trimmed, expected,
        "registry.pub must match shiso-hub constant"
    );
}

#[test]
fn atomic_put_recipe_with_stable_formatting() {
    let temp = tempdir().unwrap();
    let recipes_dir = temp.path().join("registry").join("recipes");
    fs::create_dir_all(&recipes_dir).unwrap();

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let sample = root
        .join("registry")
        .join("recipes")
        .join("qwen3.5-9b.json");
    let content = fs::read_to_string(sample).unwrap();

    // Save recipe via editor function
    editor::save_recipe(temp.path(), "qwen3.5-9b", &content).unwrap();

    let target = recipes_dir.join("qwen3.5-9b.json");
    assert!(target.is_file());

    let saved_content = fs::read_to_string(&target).unwrap();
    // Saving again should produce identical bytes (stable formatting)
    editor::save_recipe(temp.path(), "qwen3.5-9b", &saved_content).unwrap();
    let resaved_content = fs::read_to_string(&target).unwrap();
    assert_eq!(
        saved_content, resaved_content,
        "formatting must be stable across saves"
    );
}

#[test]
fn keygen_and_build_registry_produces_verifiable_signature() {
    let temp = tempdir().unwrap();
    let recipes_dir = temp.path().join("registry").join("recipes");
    fs::create_dir_all(&recipes_dir).unwrap();

    // Copy sample recipe into temp repo
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let sample = root
        .join("registry")
        .join("recipes")
        .join("qwen3.5-9b.json");
    fs::copy(sample, recipes_dir.join("qwen3.5-9b.json")).unwrap();

    // Generate keypair
    let key_path = temp.path().join("registry.key");
    let keygen = editor::generate_keypair(&key_path).unwrap();

    // Build registry
    let out_dir = temp.path().join("target").join("registry").join("v1");
    let build = editor::build_registry(temp.path(), &out_dir, Some(&key_path), true).unwrap();
    assert_eq!(build.models_count, 1);

    // Verify index.json exists
    let index_file = out_dir.join("index.json");
    let sig_file = out_dir.join("index.json.sig");
    assert!(index_file.is_file());
    assert!(sig_file.is_file());

    let index_bytes = fs::read(&index_file).unwrap();
    let sig_bytes = fs::read(&sig_file).unwrap();

    // Verify signature with generated public key
    let pub_bytes = hex::decode(keygen.public_hex).unwrap();
    let verifying_key =
        ed25519_dalek::VerifyingKey::from_bytes(&pub_bytes.try_into().unwrap()).unwrap();
    let signature = ed25519_dalek::Signature::from_bytes(&sig_bytes.try_into().unwrap());
    assert!(verifying_key.verify(&index_bytes, &signature).is_ok());
}

#[tokio::test]
async fn http_hardening_rejects_unauthorized_and_invalid_hosts() {
    let temp = tempdir().unwrap();
    let state = Arc::new(ServerState {
        repo: temp.path().to_path_buf(),
        session_token: "test_token_secret_12345".to_string(),
        port: 8095,
    });

    let app = build_router(state);

    // 1. Missing token on /api/recipes -> 401 Unauthorized
    let req = Request::builder()
        .uri("/api/recipes")
        .header(header::HOST, "127.0.0.1:8095")
        .body(Body::empty())
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    // 2. Invalid Host header (DNS rebinding attack) -> 400 Bad Request
    let req = Request::builder()
        .uri("/api/recipes")
        .header(header::HOST, "attacker.com")
        .header(header::AUTHORIZATION, "Bearer test_token_secret_12345")
        .body(Body::empty())
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    // 3. Invalid Origin on mutating request (CSRF attack) -> 403 Forbidden
    let req = Request::builder()
        .method("POST")
        .uri("/api/validate")
        .header(header::HOST, "127.0.0.1:8095")
        .header(header::ORIGIN, "http://malicious-website.com")
        .header(header::AUTHORIZATION, "Bearer test_token_secret_12345")
        .body(Body::empty())
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    // 4. Valid request with matching token and host -> 200 OK
    let req = Request::builder()
        .uri("/api/families")
        .header(header::HOST, "127.0.0.1:8095")
        .header(header::AUTHORIZATION, "Bearer test_token_secret_12345")
        .body(Body::empty())
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
