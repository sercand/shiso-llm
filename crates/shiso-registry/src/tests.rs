//! Tests for registry editor (§9.5).

#![expect(clippy::unwrap_used, reason = "test assertions panic on failure")]

use std::fs;
use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use tempfile::tempdir;
use tower::ServiceExt;

use crate::editor;
use crate::server::{ServerState, build_router};
/// Sample recipe vendored under `tests/fixtures/`. The registry itself lives
/// in its own repository, so the editor's tests carry their own fixture to
/// pin the shape `validate_all`, `save_recipe` and `build_registry` consume.
const SAMPLE_RECIPE: &str = include_str!("../tests/fixtures/qwen3.5-9b.json");

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

#[test]
fn sample_recipe_passes_strict_validation() {
    let temp = tempdir().unwrap();
    let recipes_dir = temp.path().join("recipes");
    fs::create_dir_all(&recipes_dir).unwrap();
    fs::write(recipes_dir.join("qwen3.5-9b.json"), SAMPLE_RECIPE).unwrap();

    let reports = editor::validate_all(temp.path()).unwrap();

    assert!(
        !reports.is_empty(),
        "must find at least one recipe in recipes/"
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

/// Validates the maintainer's real registry repository when one is pointed
/// at via `SHISO_REGISTRY_REPO`; the registry lives outside this workspace,
/// so without the variable there is nothing here to check.
#[test]
fn registry_repo_recipes_pass_strict_validation() {
    let Ok(repo) = std::env::var("SHISO_REGISTRY_REPO") else {
        skip("SHISO_REGISTRY_REPO is not set; the registry lives in its own repository");
        return;
    };
    let reports = editor::validate_all(Path::new(&repo)).unwrap();
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
fn atomic_put_recipe_with_stable_formatting() {
    let temp = tempdir().unwrap();
    let recipes_dir = temp.path().join("recipes");
    fs::create_dir_all(&recipes_dir).unwrap();

    // Save recipe via editor function
    editor::save_recipe(temp.path(), "qwen3.5-9b", SAMPLE_RECIPE).unwrap();

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
fn build_registry_compiles_unsigned_index_and_manifests() {
    let temp = tempdir().unwrap();
    let recipes_dir = temp.path().join("recipes");
    fs::create_dir_all(&recipes_dir).unwrap();

    // Write the vendored sample recipe into the temp repo
    fs::write(recipes_dir.join("qwen3.5-9b.json"), SAMPLE_RECIPE).unwrap();
    // The build reads publication dates from the repository's history.
    git(temp.path(), &["init", "-q"]);

    // Build registry
    let out_dir = temp.path().join("registry").join("v1");
    let build = editor::build_registry(temp.path(), &out_dir, true).unwrap();
    assert_eq!(build.models_count, 1);

    let index_file = out_dir.join("index.json");
    assert!(index_file.is_file());
    assert!(out_dir.join("models").join("qwen3.5-9b.json").is_file());
    // Unsigned model: no detached signature artifact is produced.
    assert!(!out_dir.join("index.json.sig").exists());

    // The index digest chain still holds: index.json's manifest_sha256 must
    // equal the digest of the compiled manifest bytes.
    let index: shiso_hub::recipe::Index =
        serde_json::from_slice(&fs::read(&index_file).unwrap()).unwrap();
    assert_eq!(index.models.len(), 1);
    let manifest_bytes = fs::read(out_dir.join("models").join("qwen3.5-9b.json")).unwrap();
    let digest = <sha2::Sha256 as sha2::Digest>::digest(&manifest_bytes);
    assert_eq!(
        index.models[0].manifest_sha256.as_str(),
        hex::encode(digest)
    );
}

#[tokio::test]
async fn http_hardening_rejects_unauthorized_and_invalid_hosts() {
    let temp = tempdir().unwrap();
    let state = Arc::new(ServerState {
        repo: temp.path().to_path_buf(),
        session_token: Some("test_token_secret_12345".to_string()),
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

/// Without `--require-token` the API takes requests with no token, and the
/// Host and Origin checks still refuse what they refused before.
#[tokio::test]
async fn without_a_required_token_only_the_host_and_origin_checks_apply() {
    let temp = tempdir().unwrap();
    let app = build_router(Arc::new(ServerState {
        repo: temp.path().to_path_buf(),
        session_token: None,
        port: 8095,
    }));
    let get = |host: &str| {
        Request::builder()
            .uri("/api/families")
            .header(header::HOST, host)
            .body(Body::empty())
            .unwrap()
    };
    let resp = app.clone().oneshot(get("127.0.0.1:8095")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = app.clone().oneshot(get("attacker.com")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let req = Request::builder()
        .method("POST")
        .uri("/api/validate")
        .header(header::HOST, "127.0.0.1:8095")
        .header(header::ORIGIN, "http://malicious-website.com")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
}

/// Runs git in `dir` with a fixed identity and commit date.
fn git(dir: &Path, args: &[&str]) {
    let status = std::process::Command::new("git")
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .current_dir(dir)
        .env("GIT_COMMITTER_DATE", "2026-01-02T03:04:05+00:00")
        .env("GIT_AUTHOR_DATE", "2026-01-02T03:04:05+00:00")
        .status()
        .unwrap();
    assert!(status.success(), "git {args:?} failed");
}

/// The output and the git history both belong to the repository, not to the
/// directory the tool happens to run in: a relative output lands inside the
/// repository and the manifests carry the repository's commit dates even
/// though this test process runs inside another git checkout.
#[test]
fn build_writes_into_the_repository_and_dates_from_its_history() {
    let temp = tempdir().unwrap();
    let repo = temp.path();
    fs::create_dir_all(repo.join("recipes")).unwrap();
    fs::write(repo.join("recipes").join("qwen3.5-9b.json"), SAMPLE_RECIPE).unwrap();
    git(repo, &["init", "-q"]);
    git(repo, &["add", "-A"]);
    git(repo, &["commit", "-q", "-m", "recipe"]);

    for asked in [None, Some(""), Some("registry/v1")] {
        let out = editor::resolve_out(repo, asked.map(Path::new));
        assert_eq!(out, repo.join("registry/v1"), "{asked:?}");
    }
    assert_eq!(
        editor::resolve_out(repo, Some(Path::new("/elsewhere/v1"))),
        Path::new("/elsewhere/v1")
    );

    let out = editor::resolve_out(repo, Some(Path::new("registry/v1")));
    editor::build_registry(repo, &out, false).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(out.join("models").join("qwen3.5-9b.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["updated_at"], "2026-01-02T03:04:05Z");
    assert_eq!(manifest["published_at"], "2026-01-02T03:04:05Z");
}

/// A directory git cannot read is refused rather than stamped with the build
/// time.
#[test]
fn build_refuses_a_repository_git_cannot_read() {
    let temp = tempdir().unwrap();
    fs::create_dir_all(temp.path().join("recipes")).unwrap();
    fs::write(
        temp.path().join("recipes").join("qwen3.5-9b.json"),
        SAMPLE_RECIPE,
    )
    .unwrap();
    let out = editor::resolve_out(temp.path(), None);
    let err = editor::build_registry(temp.path(), &out, true).unwrap_err();
    assert!(err.to_string().contains("git"), "{err}");
}
