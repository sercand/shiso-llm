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
    let build = editor::build_registry(temp.path(), temp.path(), true).unwrap();
    assert_eq!(build.models_count, 1);
    let out_dir = temp.path().join("registry").join("v1");

    let index_file = out_dir.join("index.json");
    assert!(index_file.is_file());
    let manifest_file = out_dir.join("models").join("qwen3.5-9b").join("index.json");
    assert!(manifest_file.is_file());
    // Unsigned model: no detached signature artifact is produced.
    assert!(!out_dir.join("index.json.sig").exists());

    // The index names the manifest's path, and its digest chain still holds:
    // manifest_sha256 must equal the digest of the compiled manifest bytes.
    let index: shiso_hub::recipe::Index =
        serde_json::from_slice(&fs::read(&index_file).unwrap()).unwrap();
    assert_eq!(index.models.len(), 1);
    assert_eq!(index.models[0].manifest, "models/qwen3.5-9b/index.json");
    let manifest_bytes = fs::read(&manifest_file).unwrap();
    let digest = <sha2::Sha256 as sha2::Digest>::digest(&manifest_bytes);
    assert_eq!(
        index.models[0].manifest_sha256.as_str(),
        hex::encode(digest)
    );

    // The site sits beside the registry: a catalog that embeds its search
    // documents, and one page per model linking back with relative paths.
    let home = fs::read_to_string(temp.path().join("index.html")).unwrap();
    assert!(home.contains(r#"href="models/qwen3.5-9b/""#), "{home}");
    assert!(home.contains(r#"id="search-docs""#));
    let page = fs::read_to_string(
        temp.path()
            .join("models")
            .join("qwen3.5-9b")
            .join("index.html"),
    )
    .unwrap();
    assert!(page.contains("../../registry/v1/models/qwen3.5-9b/index.json"));
    assert!(page.contains("shiso pull qwen3.5-9b"));
    for asset in [
        "404.html",
        "search-index.json",
        "assets/site.css",
        "assets/site.js",
        "assets/minisearch.js",
    ] {
        assert!(temp.path().join(asset).is_file(), "{asset}");
    }
    // No card is cached in a fresh repository.
    assert_eq!(build.missing_cards, vec!["qwen3.5-9b".to_string()]);
}

/// A model whose recipe is gone loses its page and manifest on the next
/// build, and so does a manifest left by the old flat layout; a directory
/// holding files the build never writes is refused instead.
#[test]
fn build_prunes_what_it_wrote_for_removed_models() {
    let temp = tempdir().unwrap();
    let repo = temp.path();
    fs::create_dir_all(repo.join("recipes")).unwrap();
    fs::write(repo.join("recipes").join("qwen3.5-9b.json"), SAMPLE_RECIPE).unwrap();
    git(repo, &["init", "-q"]);

    let gone_page = repo.join("models").join("old-model");
    let gone_manifest = repo.join("registry/v1/models/old-model");
    fs::create_dir_all(&gone_page).unwrap();
    fs::create_dir_all(&gone_manifest).unwrap();
    fs::write(gone_page.join("index.html"), "old").unwrap();
    fs::write(gone_manifest.join("index.json"), "{}").unwrap();
    fs::write(repo.join("registry/v1/models/old-flat.json"), "{}").unwrap();

    let build = editor::build_registry(repo, repo, true).unwrap();
    assert_eq!(build.pruned.len(), 3, "{:?}", build.pruned);
    assert!(!gone_page.exists() && !gone_manifest.exists());
    assert!(!repo.join("registry/v1/models/old-flat.json").exists());

    let foreign = repo.join("models").join("kept");
    fs::create_dir_all(&foreign).unwrap();
    fs::write(foreign.join("notes.txt"), "mine").unwrap();
    let err = editor::build_registry(repo, repo, true).unwrap_err();
    assert!(err.to_string().contains("notes.txt"), "{err}");
}

/// The editor serves the built site under `/hub/` and nothing under a dot
/// path, so `.git` stays out of the preview.
#[tokio::test]
async fn the_hub_preview_serves_the_site_and_hides_dot_paths() {
    let temp = tempdir().unwrap();
    fs::write(temp.path().join("index.html"), "<p>hub</p>").unwrap();
    fs::create_dir_all(temp.path().join(".git")).unwrap();
    fs::write(temp.path().join(".git").join("config"), "secret").unwrap();
    let app = build_router(Arc::new(ServerState::new(
        temp.path().to_path_buf(),
        None,
        8095,
        temp.path().join("home"),
    )));
    let get = |uri: &str| {
        Request::builder()
            .uri(uri)
            .header(header::HOST, "127.0.0.1:8095")
            .body(Body::empty())
            .unwrap()
    };
    let res = app.clone().oneshot(get("/hub/")).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let res = app.clone().oneshot(get("/hub")).await.unwrap();
    assert!(res.status().is_redirection());
    let res = app.clone().oneshot(get("/hub/.git/config")).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn http_hardening_rejects_unauthorized_and_invalid_hosts() {
    let temp = tempdir().unwrap();
    let state = Arc::new(ServerState::new(
        temp.path().to_path_buf(),
        Some("test_token_secret_12345".to_string()),
        8095,
        temp.path().join("home"),
    ));

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
    let app = build_router(Arc::new(ServerState::new(
        temp.path().to_path_buf(),
        None,
        8095,
        temp.path().join("home"),
    )));
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

    for asked in [None, Some("")] {
        let out = editor::resolve_out(repo, asked.map(Path::new));
        assert_eq!(out, repo, "{asked:?}");
    }
    assert_eq!(
        editor::resolve_out(repo, Some(Path::new("site"))),
        repo.join("site")
    );
    assert_eq!(
        editor::resolve_out(repo, Some(Path::new("/elsewhere"))),
        Path::new("/elsewhere")
    );

    let out = editor::resolve_out(repo, None);
    editor::build_registry(repo, &out, false).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(out.join("registry/v1/models/qwen3.5-9b/index.json")).unwrap(),
    )
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

/// The fixture recipe with a hub-hosted draft vocabulary added to its first
/// variant, and the file it names written where the hub serves it.
fn repo_with_hosted_file(contents: &[u8]) -> tempfile::TempDir {
    let temp = tempdir().unwrap();
    let recipes_dir = temp.path().join("recipes");
    fs::create_dir_all(&recipes_dir).unwrap();
    let mut recipe: shiso_hub::recipe::Recipe =
        serde_json::from_str(include_str!("../tests/fixtures/qwen3.8-flash.json")).unwrap();
    let sha = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(contents));
    recipe.min_shiso_version = serde_json::from_value(serde_json::json!("0.2.0")).unwrap();
    let variant = &mut recipe.variants[0];
    variant.components.insert(
        shiso_hub::recipe::Role::DraftVocab,
        serde_json::from_value(serde_json::json!({
            "provider": "shiso", "entry": "draft_vocab_counts.txt",
            "files": [{"name": "draft_vocab_counts.txt", "size": contents.len(), "sha256": sha}]
        }))
        .unwrap(),
    );
    variant.requirements.disk_bytes += contents.len() as u64;
    fs::write(
        recipes_dir.join("qwen3.8-flash.json"),
        serde_json::to_string_pretty(&recipe).unwrap(),
    )
    .unwrap();
    let hosted = editor::hosted_file_path(temp.path(), "qwen3.8-flash", "draft_vocab_counts.txt");
    fs::create_dir_all(hosted.parent().unwrap()).unwrap();
    fs::write(&hosted, contents).unwrap();
    temp
}

#[test]
fn a_hosted_file_must_match_its_recipe() {
    let temp = repo_with_hosted_file(b"198 3037739\n");
    let reports = editor::validate_all(temp.path()).unwrap();
    assert!(reports.iter().all(|r| r.valid), "{:?}", reports[0].errors);

    let hosted = editor::hosted_file_path(temp.path(), "qwen3.8-flash", "draft_vocab_counts.txt");
    fs::write(&hosted, b"198 3037740\n").unwrap(); // same size, different bytes
    let reports = editor::validate_all(temp.path()).unwrap();
    assert!(
        reports[0].errors.iter().any(|e| e.contains("sha256")),
        "{:?}",
        reports[0].errors
    );

    fs::write(&hosted, b"198\n").unwrap();
    let reports = editor::validate_all(temp.path()).unwrap();
    assert!(
        reports[0].errors.iter().any(|e| e.contains("size")),
        "{:?}",
        reports[0].errors
    );

    fs::remove_file(&hosted).unwrap();
    let reports = editor::validate_all(temp.path()).unwrap();
    assert!(
        reports[0].errors.iter().any(|e| e.contains("missing")),
        "{:?}",
        reports[0].errors
    );
}

#[test]
fn a_build_into_another_directory_carries_the_hosted_file() {
    let temp = repo_with_hosted_file(b"198 3037739\n");
    git(temp.path(), &["init", "-q"]);
    let out = tempdir().unwrap();
    editor::build_registry(temp.path(), out.path(), true).unwrap();
    let copied = editor::hosted_file_path(out.path(), "qwen3.8-flash", "draft_vocab_counts.txt");
    assert_eq!(fs::read(copied).unwrap(), b"198 3037739\n");
}

#[test]
fn an_uncommitted_hosted_file_makes_the_build_dirty() {
    let temp = repo_with_hosted_file(b"198 3037739\n");
    git(temp.path(), &["init", "-q"]);
    git(temp.path(), &["add", "recipes"]);
    git(temp.path(), &["commit", "-qm", "r"]);
    let status = editor::git_status(temp.path());
    assert!(status.is_dirty, "the hosted file is untracked");
    assert!(
        status
            .changed_recipes
            .iter()
            .any(|f| f.ends_with("draft_vocab_counts.txt")),
        "{:?}",
        status.changed_recipes
    );
    assert!(editor::build_registry(temp.path(), temp.path(), false).is_err());
}

#[test]
fn the_model_page_links_a_hosted_file_to_the_hub() {
    let temp = repo_with_hosted_file(b"198 3037739\n");
    git(temp.path(), &["init", "-q"]);
    editor::build_registry(temp.path(), temp.path(), true).unwrap();
    let page = fs::read_to_string(
        temp.path()
            .join("models")
            .join("qwen3.8-flash")
            .join("index.html"),
    )
    .unwrap();
    assert!(page.contains("hosted on the hub"), "{page}");
    assert!(
        page.contains("registry/v1/models/qwen3.8-flash/draft_vocab_counts.txt"),
        "{page}"
    );
    assert!(
        page.contains("turboderp/Qwen3.8-Flash-Next-exl3@"),
        "the HF component keeps its label"
    );
}
